import torch
from torch import nn
from cape.models.encoder import TrafficEncoder

class CAPEAgent(nn.Module):
    def __init__(self, feature_dim, hidden_dim, code_dim, measures, heads, device):
        super().__init__()
        self.encoder = TrafficEncoder(feature_dim, hidden_dim, heads)
        self.measure = nn.Embedding(measures, hidden_dim)
        self.candidate = nn.Sequential(nn.Linear(hidden_dim * 2 + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.planner_actor = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 4))
        self.planner_q1 = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 4))
        self.planner_q2 = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 4))
        self.generator = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, code_dim))
        self.executor = nn.Sequential(nn.Linear(hidden_dim * 2 + code_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, measures * 6))
        self.executor_q1 = nn.Sequential(nn.Linear(hidden_dim * 2 + code_dim + measures * 3, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 1))
        self.executor_q2 = nn.Sequential(nn.Linear(hidden_dim * 2 + code_dim + measures * 3, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 1))
        self.value = nn.Sequential(nn.Linear(hidden_dim * 2 + code_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 1))
        self.to(device)

    def representations(self, history, adjacency): return self.encoder(history, adjacency)
    def score_pairs(self, nodes, global_state, active, previous):
        batch, count, hidden = nodes.shape; measures = self.measure.weight.shape[0]
        node = nodes.unsqueeze(2).expand(-1, -1, measures, -1); global_part = global_state[:, None, None, :].expand(-1, count, measures, -1)
        flags = torch.stack([active, previous], -1).unsqueeze(2).expand(-1, -1, measures, -1)
        return self.candidate(torch.cat([node, global_part, flags], -1)).squeeze(-1)

    def membership(self, pairs, nodes):
        value = torch.zeros(1, nodes, device=self.measure.weight.device)
        for node, _ in pairs: value[0, node] = 1
        return value

    def context(self, nodes, global_state, membership):
        pooled = (nodes * membership.unsqueeze(-1)).sum(1) / membership.sum(1, keepdim=True).clamp_min(1)
        return torch.cat([pooled, global_state], -1)

    def code(self, nodes, global_state, membership): return self.generator(self.context(nodes, global_state, membership))

    def planner_state(self, nodes, global_state, membership, candidate):
        return torch.cat([self.context(nodes, global_state, membership), nodes[:, candidate]], -1)

    def executor_distribution(self, context, code):
        output = self.executor(torch.cat([context, code], -1)).reshape(context.shape[0], -1, 6)
        mean, log_std = output[..., :3].tanh(), output[..., 3:].clamp(-5, 2)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = distribution.rsample(); action = raw.tanh()
        logp = (distribution.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum((-1, -2))
        return action, logp, mean

    def executor_value(self, context, code): return self.value(torch.cat([context, code], -1)).squeeze(-1)

    def actions(self, nodes, global_state, pairs, code):
        membership = self.membership(pairs, nodes.shape[1]); code = self.code(nodes, global_state, membership) if code is None else code
        action, _, _ = self.executor_distribution(self.context(nodes, global_state, membership), code)
        return {pair: action[0, pair[1]].detach().cpu().numpy() for pair in pairs}

    def distill(self, nodes, global_state, membership, target):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=3e-4)
        loss = (self.code(nodes, global_state, membership) - target).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
