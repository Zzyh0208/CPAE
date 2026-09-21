import numpy as np
import torch

class Calibrator:
    def __init__(self, agent, pairs, steps, learning_rate): self.agent, self.pairs, self.steps, self.learning_rate = agent, pairs, steps, learning_rate
    def run(self, runtime, reference, edited, state):
        snapshot = runtime.snapshot(); records = []
        nodes, global_state, membership = state
        code = self.agent.code(nodes, global_state, membership).detach().clone().requires_grad_(True)
        for _ in range(self.pairs):
            runtime.restore(snapshot); _, baseline, _, _ = runtime.step(reference, self.agent.actions(nodes, global_state, reference, None))
            runtime.restore(snapshot); action_map = self.agent.actions(nodes, global_state, edited, code); _, outcome, _, _ = runtime.step(edited, action_map)
            records.append((outcome - baseline, action_map))
        runtime.restore(snapshot)
        optimizer = torch.optim.Adam([code], lr=self.learning_rate)
        returns = torch.tensor([value[0] for value in records], dtype=torch.float32, device=code.device)
        best = records[int(returns.argmax())][1]
        target = torch.as_tensor(np.stack([best[pair] for pair in edited]), device=code.device).mean(0)
        context = self.agent.context(nodes, global_state, membership)
        for _ in range(self.steps):
            action, logp, mean = self.agent.executor_distribution(context, code)
            weights = ((returns - returns.mean()) / returns.std().clamp_min(1e-3)).clamp(-3, 3)
            value = self.agent.executor_value(context, code)
            loss = -(weights * logp).mean() + .5 * (mean.mean(0) - target).square().mean() + .2 * torch.nn.functional.huber_loss(value, returns.mean().expand_as(value))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([code], 1); optimizer.step()
            with torch.no_grad(): code.copy_(torch.nan_to_num(code).clamp(-5, 5))
        self.agent.distill(nodes, global_state, membership, code.detach())
        return code.detach(), float(returns.mean())
