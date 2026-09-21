import copy
import torch
from torch import nn

def soft_update(target, source, rate):
    with torch.no_grad():
        for target_value, source_value in zip(target.parameters(), source.parameters()): target_value.lerp_(source_value, rate)

class PlannerSAC:
    def __init__(self, agent, learning_rate, gamma, alpha):
        self.actor, self.q1, self.q2 = agent.planner_actor, agent.planner_q1, agent.planner_q2
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.gamma, self.alpha = gamma, alpha
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)

    def update(self, state, action, reward, next_state, terminal):
        logits = self.actor(state); distribution = torch.distributions.Categorical(logits=logits)
        current = torch.minimum(self.q1(state), self.q2(state)).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            next_logits = self.actor(next_state); probabilities = next_logits.softmax(-1); log_probabilities = next_logits.log_softmax(-1)
            value = (probabilities * (torch.minimum(self.target_q1(next_state), self.target_q2(next_state)) - self.alpha * log_probabilities)).sum(-1)
            target = reward + self.gamma * (1 - terminal) * value
        critic_loss = nn.functional.mse_loss(current, target)
        policy_loss = (distribution.probs * (self.alpha * distribution.logits - torch.minimum(self.q1(state), self.q2(state)))).sum(-1).mean()
        self.optimizer.zero_grad(); (critic_loss + policy_loss).backward(); torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()), 5); self.optimizer.step()
        soft_update(self.target_q1, self.q1, .005); soft_update(self.target_q2, self.q2, .005)
        return float((critic_loss + policy_loss).detach())

class ExecutorSAC:
    def __init__(self, agent, learning_rate, gamma, alpha):
        self.agent, self.q1, self.q2 = agent, agent.executor_q1, agent.executor_q2
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.gamma, self.alpha = gamma, alpha
        self.optimizer = torch.optim.Adam(list(agent.executor.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)

    def update(self, context, code, action, reward, next_context, next_code, terminal):
        state = torch.cat([context, code], -1); next_state = torch.cat([next_context, next_code], -1)
        current1, current2 = self.q1(torch.cat([state, action], -1)).squeeze(-1), self.q2(torch.cat([state, action], -1)).squeeze(-1)
        with torch.no_grad():
            next_action, next_logp, _ = self.agent.executor_distribution(next_context, next_code); next_action = next_action.flatten(1)
            target = reward + self.gamma * (1 - terminal) * (torch.minimum(self.target_q1(torch.cat([next_state, next_action], -1)).squeeze(-1), self.target_q2(torch.cat([next_state, next_action], -1)).squeeze(-1)) - self.alpha * next_logp)
        critic_loss = nn.functional.mse_loss(current1, target) + nn.functional.mse_loss(current2, target)
        sampled, logp, _ = self.agent.executor_distribution(context, code); sampled = sampled.flatten(1)
        policy_loss = (self.alpha * logp - torch.minimum(self.q1(torch.cat([state, sampled], -1)).squeeze(-1), self.q2(torch.cat([state, sampled], -1)).squeeze(-1))).mean()
        self.optimizer.zero_grad(); (critic_loss + policy_loss).backward(); torch.nn.utils.clip_grad_norm_(list(self.agent.executor.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()), 5); self.optimizer.step()
        soft_update(self.target_q1, self.q1, .005); soft_update(self.target_q2, self.q2, .005)
        return float((critic_loss + policy_loss).detach())
