from dataclasses import asdict, dataclass
import math
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal

@dataclass
class Config:
    features: int = 6
    history: int = 12
    hidden: int = 128
    heads: int = 4
    code_dim: int = 128
    measures: int = 3
    waves: int = 3
    action_dim: int = 8
    max_candidates: int = 64
    max_nodes: int = 32
    max_edits: int = 8
    threshold: float = 0.63
    probe_pairs: int = 4
    calibration_steps: int = 5
    calibration_lr: float = 0.05
    value_samples: int = 4
    consistency_weight: float = 0.10
    effect_weight: float = 0.20
    distill_weight: float = 0.10
    encoder_weight: float = 0.05
    calibration_action_weight: float = 0.50
    calibration_value_weight: float = 0.20
    enabled_measures: tuple = (True, True, True)
    calibration_enabled: bool = True

def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, outputs))

class GraphAttention(nn.Module):

    def __init__(self, hidden, heads):
        super().__init__()
        (self.heads, self.width) = (heads, hidden // heads)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.gate = nn.Linear(hidden * 2, heads)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes, adjacency):
        count = len(nodes)
        edges = adjacency.bool() | torch.eye(count, dtype=torch.bool, device=nodes.device)
        (dst, src) = edges.nonzero(as_tuple=True)
        (q, k, v) = self.qkv(nodes).reshape(count, 3, self.heads, self.width).unbind(1)
        logits = (q[dst] * k[src]).sum(-1) / math.sqrt(self.width)
        logits = logits + torch.nn.functional.logsigmoid(self.gate(torch.cat([nodes[dst], nodes[src]], -1)))
        index = dst[:, None].expand(-1, self.heads)
        maxima = logits.new_full((count, self.heads), -torch.inf)
        maxima.scatter_reduce_(0, index, logits.detach(), reduce='amax', include_self=True)
        weights = (logits - maxima[dst]).exp()
        denominator = logits.new_zeros((count, self.heads)).scatter_add_(0, index, weights)
        weighted = (weights / denominator[dst].clamp_min(1e-12))[..., None] * v[src]
        output = nodes.new_zeros((count, self.heads, self.width)).index_add_(0, dst, weighted)
        return self.norm(nodes + self.out(output.flatten(1)))

class CAPE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config.hidden
        self.input = nn.Linear(config.features, h)
        self.position = nn.Parameter(torch.zeros(config.history, h))
        layer = nn.TransformerEncoderLayer(d_model=h, nhead=config.heads, dim_feedforward=256, dropout=0.0, activation='gelu', batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.graph = nn.ModuleList([GraphAttention(h, config.heads) for _ in range(3)])
        self.measure = nn.Embedding(config.measures, h)
        self.pair_encoder = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, h))
        self.reconstruct = nn.Linear(h, config.features)
        self.predict = nn.ModuleList([nn.Linear(h, config.features) for _ in range(3)])
        self.candidate = mlp(h * 3 + 2, 1)
        self.set_input = nn.Linear(h * 2 + 2, 256)
        set_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.0, activation='gelu', batch_first=True)
        self.set_encoder = nn.TransformerEncoder(set_layer, 2)
        self.edit_dim = 4 + h * 2 + 256
        self.planner_actor = mlp(self.edit_dim, 1)
        self.planner_q1 = mlp(self.edit_dim, 1)
        self.planner_q2 = mlp(self.edit_dim, 1)
        self.generator = mlp(h * 2, config.code_dim)
        self.executor_body = nn.Sequential(nn.Linear(h * 3 + config.code_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.executor = nn.ModuleList([nn.Linear(256, 2 * config.waves * config.action_dim) for _ in range(config.measures)])
        self.action_encoder = nn.Linear(config.waves * config.action_dim + h, h)
        self.executor_q1 = mlp(h * 3 + config.code_dim, 1)
        self.executor_q2 = mlp(h * 3 + config.code_dim, 1)
        self.executor_v = mlp(h * 2 + config.code_dim, 1)

    def temporal_features(self, history):
        values = self.input(history.transpose(0, 1)) + self.position[None]
        return self.temporal(values)

    def encode(self, history, adjacency):
        nodes = self.temporal_features(history)[:, -1]
        for layer in self.graph:
            nodes = layer(nodes, adjacency)
        return (nodes, nodes.mean(0))

    def pair_features(self, nodes):
        n, h = nodes.shape
        m = self.config.measures
        return self.pair_encoder(torch.cat([nodes[:, None].expand(n, m, h), self.measure.weight[None].expand(n, m, h)], -1)).flatten(0, 1)

    def context(self, nodes, global_state, support):
        pair = self.pair_features(nodes)
        selected = support.flatten()
        pooled = pair[selected].mean(0) if selected.any() else pair.sum(0) * 0
        return torch.cat([pooled, global_state])

    def code(self, nodes, global_state, support):
        return self.generator(self.context(nodes, global_state, support))

    def pair_scores(self, nodes, global_state, support, previous):
        (n, m) = support.shape
        inp = torch.cat([nodes[:, None].expand(-1, m, -1), global_state.expand(n, m, -1), self.measure.weight[None].expand(n, -1, -1), support[..., None].float(), previous[..., None].float()], -1)
        return self.candidate(inp).squeeze(-1).flatten()

    def edit_features(self, nodes, global_state, support, previous, candidates, edits):
        pair = self.pair_features(nodes)
        flags = torch.stack([support.flatten()[candidates], previous.flatten()[candidates]], -1)
        tokens_input = torch.cat([pair[candidates], global_state.expand(len(candidates), -1), flags.float()], -1)
        if not candidates:
            tokens_input = torch.cat([nodes.new_zeros((1, nodes.shape[-1])), global_state[None], nodes.new_zeros(1, 2)], -1)
        tokens = self.set_input(tokens_input)
        if candidates:
            scores = self.pair_scores(nodes, global_state, support, previous)[candidates]
            tokens = tokens * (1 + scores.sigmoid()[:, None])
        pooled = self.set_encoder(tokens[None]).mean(1)[0]
        operations = torch.tensor([e[0] for e in edits], device=nodes.device)
        remove = torch.stack([pair[e[1]] if e[1] >= 0 else pair.new_zeros(pair.shape[-1]) for e in edits])
        add = torch.stack([pair[e[2]] if e[2] >= 0 else pair.new_zeros(pair.shape[-1]) for e in edits])
        return torch.cat([torch.nn.functional.one_hot(operations, 4).float(), remove, add, pooled.expand(len(edits), -1)], -1)

    def distribution(self, nodes, global_state, support, code):
        (n, m) = support.shape
        local = self.pair_features(nodes).reshape(n, m, -1)
        context = self.context(nodes, global_state, support)
        outputs = [head(self.executor_body(torch.cat([local[:, i], context.expand(n, -1), code.expand(n, -1)], -1))) for (i, head) in enumerate(self.executor)]
        params = torch.stack(outputs, 1).reshape(n, m, self.config.waves, self.config.action_dim, 2)
        (mean, log_std) = params.unbind(-1)
        return Normal(mean, log_std.clamp(-5, 2).exp())

    def sample(self, nodes, global_state, support, code, dim_mask, deterministic=False):
        dist = self.distribution(nodes, global_state, support, code)
        raw = dist.loc if deterministic else dist.rsample()
        action = raw.tanh()
        mask = support[..., None, None] & dim_mask
        logp = ((dist.log_prob(raw) - torch.log(1 - action.square() + 1e-06)) * mask).sum()
        return (action * mask, logp)

    def log_prob(self, nodes, global_state, support, code, actions, dim_mask):
        dist = self.distribution(nodes, global_state, support, code)
        actions = actions.clamp(-1 + 1e-06, 1 - 1e-06)
        mask = support[..., None, None] & dim_mask
        return ((dist.log_prob(torch.atanh(actions)) - torch.log(1 - actions.square() + 1e-06)) * mask).sum()

    def q_input(self, nodes, global_state, support, code, actions, dim_mask):
        pairs = self.pair_features(nodes)
        actions = (actions * dim_mask).flatten(2).flatten(0, 1)
        selected = support.flatten()
        embedded = self.action_encoder(torch.cat([actions, pairs], -1))
        pooled = embedded[selected].mean(0) if selected.any() else embedded.sum(0) * 0
        return torch.cat([self.context(nodes, global_state, support), code, pooled])

    def execution_value(self, nodes, global_state, support, code, dim_mask=None):
        return self.executor_v(torch.cat([self.context(nodes, global_state, support), code])).squeeze(-1)

    def policy_q_expectation(self, nodes, global_state, support, code, dim_mask=None):
        if dim_mask is None:
            dim_mask = torch.ones((*support.shape, self.config.waves, self.config.action_dim), dtype=torch.bool, device=nodes.device)
        values = []
        for _ in range(self.config.value_samples):
            action, _ = self.sample(nodes, global_state, support, code, dim_mask)
            inputs = self.q_input(nodes, global_state, support, code, action, dim_mask)
            values.append(torch.minimum(self.executor_q1(inputs), self.executor_q2(inputs)).squeeze())
        return torch.stack(values).mean()

    def auxiliary_loss(self, history, adjacency, future, mask=None):
        if mask is None:
            mask = torch.rand_like(history) < .15
        masked = history.masked_fill(mask, 0)
        reconstruction = self.reconstruct(self.temporal_features(masked)).transpose(0, 1)
        loss = torch.nn.functional.huber_loss(reconstruction[mask], history[mask]) if mask.any() else reconstruction.sum() * 0
        hidden, _ = self.encode(masked, adjacency)
        for weight, head, target in zip([.5, .3, .2], self.predict, future):
            loss = loss + weight * torch.nn.functional.huber_loss(head(hidden), target)
        return loss

def save_checkpoint(path, model, metadata, optimizer=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, parameter_count=sum(p.numel() for p in model.parameters()))
    torch.save({'format': 'cape-v3', 'config': asdict(model.config), 'model': model.state_dict(), 'metadata': metadata, 'optimizer': None if optimizer is None else optimizer.state_dict()}, path)

def load_checkpoint(path, device='cpu'):
    record = torch.load(path, map_location=device)
    if record.get('format') != 'cape-v3':
        raise ValueError('Expected cape-v3 checkpoint with a trained independent V head.')
    model = CAPE(Config(**record['config'])).to(device)
    model.load_state_dict(record['model'], strict=True)
    model.eval()
    return (model, record['metadata'])
