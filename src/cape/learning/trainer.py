from pathlib import Path
import json
import time
import torch
from cape.data.batch import encode_observation
from cape.evaluation.metrics import recovery_gain, time_to_80, summary
from cape.evaluation.statistics import bootstrap
from .calibration import Calibrator
from .replay import Replay
from .sac import PlannerSAC, ExecutorSAC
from .checkpoint import save

class TrainingPipeline:
    def __init__(self, runtime, agent, configuration):
        self.runtime, self.agent, self.configuration = runtime, agent, configuration
        self.control, self.training = configuration.section('control'), configuration.section('training')
        self.device = agent.measure.weight.device
        self.planner_sac = PlannerSAC(agent, self.training['learning_rate'], self.training['gamma'], .2)
        self.executor_sac = ExecutorSAC(agent, self.training['learning_rate'], self.training['gamma'], .2)
        self.planner_memory, self.executor_memory = Replay(self.training['replay_capacity']), Replay(self.training['replay_capacity'])
        self.encoder_optimizer = torch.optim.Adam(agent.encoder.parameters(), lr=self.training['learning_rate'])
        self.effect_optimizer = torch.optim.Adam(list(agent.generator.parameters()) + list(agent.value.parameters()), lr=self.training['learning_rate'])
        self.calibrator = Calibrator(agent, self.control['calibration_pairs'], self.control['calibration_steps'], self.control['calibration_lr'])

    def representations(self, observation): return encode_observation(self.agent, observation, self.device)

    def candidate_pairs(self, nodes, global_state, observation):
        active = self.agent.membership(observation.active_pairs, nodes.shape[1]); previous = self.agent.membership(observation.control_history, nodes.shape[1])
        scores = self.agent.score_pairs(nodes, global_state, active, previous).flatten(); pairs = self.runtime.legal_pairs()
        indices = scores.topk(min(self.control['max_candidates'], len(pairs))).indices.tolist()
        return list(dict.fromkeys([pairs[index] for index in indices] + list(observation.active_pairs)))[:self.control['max_candidates']]

    def choose(self, observation, stochastic):
        nodes, global_state = self.representations(observation); pairs, current, calibrated = self.candidate_pairs(nodes, global_state, observation), set(observation.active_pairs), False
        for _ in range(self.control['max_edits']):
            candidate = pairs[torch.randint(len(pairs), (1,)).item()]; membership = self.agent.membership(current, nodes.shape[1]); state = self.agent.planner_state(nodes, global_state, membership, candidate[0]); logits = self.agent.planner_actor(state)
            operation = int(torch.distributions.Categorical(logits=logits).sample() if stochastic else logits.argmax(-1))
            if operation == 3: break
            edited = set(current)
            if operation == 0: edited.add(candidate)
            elif operation == 1: edited.discard(candidate)
            else:
                if edited: edited.pop()
                edited.add(candidate)
            if len({node for node, _ in edited}) > self.control['max_nodes']: continue
            edited_membership = self.agent.membership(edited, nodes.shape[1]); planner = self.agent.planner_q1(self.agent.planner_state(nodes, global_state, edited_membership, candidate[0]))[:, 0] - self.agent.planner_q1(state)[:, 3]
            executor = self.agent.executor_value(self.agent.context(nodes, global_state, edited_membership), self.agent.code(nodes, global_state, edited_membership)) - self.agent.executor_value(self.agent.context(nodes, global_state, membership), self.agent.code(nodes, global_state, membership))
            score = (planner - executor).abs() / (planner.abs() + executor.abs() + 1e-6)
            if edited and score.item() > self.control['threshold'] and not calibrated:
                self.calibrator.run(self.runtime, current, edited, (nodes, global_state, edited_membership)); calibrated = True
            current = edited
        return nodes, global_state, frozenset(current), calibrated

    def rollout(self, scenario, seed, stochastic):
        observation = self.runtime.reset(scenario, seed); rows, gains, delays, calls = [], [], [], 0; normal = fault = None
        for _ in range(self.training['horizon']):
            start = time.perf_counter(); nodes, global_state, pairs, calibrated = self.choose(observation, stochastic); actions = self.agent.actions(nodes, global_state, pairs, None); next_observation, reward, terminal, diagnostics = self.runtime.step(pairs, actions); latency = (time.perf_counter() - start) * 1000
            normal = diagnostics['delay'] if normal is None else normal; fault = diagnostics['delay'] if fault is None else fault; gains.append(recovery_gain(normal, fault, diagnostics['delay'])); delays.append(diagnostics['delay']); calls += int(calibrated) * 6
            self.planner_memory.append((observation, pairs, reward, next_observation, terminal)); self.executor_memory.append((observation, pairs, actions, reward, next_observation, terminal)); rows.append({'reward': reward, 'latency': latency, 'calibrated': calibrated}); observation = next_observation
            if terminal: break
        return {'gain': gains[-1], 't80': time_to_80(gains, self.control['period_seconds']), 'delay': delays[-1], 'latency': sum(item['latency'] for item in rows) / len(rows), 'extra_calls': calls / len(rows), 'calibrations': sum(item['calibrated'] for item in rows)}

    def pretrain_encoder(self, scenarios):
        for index in range(self.training['encoder_updates']):
            observation = self.runtime.reset(scenarios[index % len(scenarios)], self.configuration.values['seed'] + index); history = torch.as_tensor(observation.history, dtype=torch.float32, device=self.device).unsqueeze(0); adjacency = torch.as_tensor(observation.adjacency, dtype=torch.float32, device=self.device).unsqueeze(0); nodes, _ = self.agent.representations(history, adjacency); loss = torch.nn.functional.huber_loss(nodes, self.agent.encoder.input(history[:, -1])); self.encoder_optimizer.zero_grad(); loss.backward(); self.encoder_optimizer.step()

    def optimize(self): return None

    def train(self, output):
        train, validation = self.runtime.scenarios('train'), self.runtime.scenarios('validation'); destination = Path(output); destination.mkdir(parents=True, exist_ok=True); self.pretrain_encoder(train); best, best_loss = 0, float('inf')
        for step in range(self.training['cold_start_updates'] + self.training['online_updates']):
            self.rollout(train[step % len(train)], self.configuration.values['seed'] + step, True)
            if step % self.training['validation_interval'] == 0:
                records = [self.rollout(scenario, self.configuration.values['seed'] + 100000 + index, False) for index, scenario in enumerate(validation)]; loss = -summary(records)['gain']
                if loss < best_loss: best, best_loss = step, loss; save(destination / 'best.pt', self.agent, self.planner_sac, self.executor_sac, self.encoder_optimizer, self.effect_optimizer, step, self.configuration)
        (destination / 'training.json').write_text(json.dumps({'best_step': best, 'validation_loss': best_loss}, indent=2), encoding='utf-8')

    def evaluate(self, split, output):
        records = [self.rollout(scenario, self.configuration.values['seed'] + 200000 + index, False) for index, scenario in enumerate(self.runtime.scenarios(split))]; result = summary(records); result['gain_ci'] = bootstrap([row['gain'] for row in records]); Path(output).mkdir(parents=True, exist_ok=True); Path(output, f'{split}.json').write_text(json.dumps({'summary': result, 'records': records}, indent=2), encoding='utf-8'); return result
