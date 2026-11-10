import copy
import hashlib
import json
import random
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from cape.models.agent import CAPE, Config, save_checkpoint, load_checkpoint
from cape.planning.controller import decide, candidates, legal_edits, apply_edit
from cape.runtime.sumo import SUMOEnvironment
from cape.runtime.probes import MatchedProbes
from .losses import planner_loss, executor_loss, effect_loss
from cape.data.trajectories import load_encoder_data


def cpu_observation(obs):
    return {key: value.detach().cpu().clone() for key, value in obs.items()}


class Replay:
    def __init__(self, capacity, seed):
        self.rows = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, *rows):
        self.rows.extend(rows)

    def sample(self, count):
        if not self.rows:
            return []
        return self.rng.sample(list(self.rows), count) if len(self.rows) >= count else self.rng.choices(list(self.rows), k=count)

    def __len__(self):
        return len(self.rows)


def collect_episode(env, model, scenario, seed, calibration=True, stochastic=False,
                    budget=None, coverage=False, gamma=.99, balanced=False,
                    operation_counts=None, visit_counts=None, run_id=0):
    obs = env.reset(scenario, seed)
    rows, diagnostics, auxiliary, observations = [], [], [], [cpu_observation(obs)]
    visits = np.zeros(obs['legal'].shape) if visit_counts is None else visit_counts
    if operation_counts is None and (balanced or coverage):
        operation_counts = np.zeros(4)
    while env.time < env.end:
        deadline = None if budget is None else time.perf_counter() + budget
        probes = MatchedProbes(env, workers=4, gamma=gamma, deadline=deadline)
        result = decide(model, obs, probes if calibration and model.config.calibration_enabled else None, stochastic=stochastic,
                        deadline=deadline, capture=True, coverage_counts=visits if coverage or balanced else None,
                        operation_counts=operation_counts if coverage or balanced else None,
                        collect_pairs=balanced, action_preparer=env.prepare_action)
        support = torch.as_tensor(result['support'], dtype=torch.bool, device=obs['history'].device)
        if rows:
            for preceding in reversed(rows):
                if preceding[2]['kind'] == 'executor':
                    preceding[1]['active'] = support.cpu().clone()
                    break
        visits += support.cpu().numpy()
        action = result['normalized_actions']
        nxt, reward, terminal, terms = env.step(result['support'], action, prepared=result.pop('prepared_action'))
        for before, edit, after, pool in result.pop('transitions'):
            first, second = cpu_observation(obs), cpu_observation(obs)
            first['active'], second['active'] = before.cpu(), after.cpu()
            first['candidate_pool'] = torch.tensor(pool, dtype=torch.long)
            rows.append((first, second, {'kind': 'planner', 'edit': edit, 'commit': False,
                        'reward': 0., 'discount': 1., 'terminal': False}))
        final = cpu_observation(obs)
        final['active'] = support.cpu()
        with torch.no_grad():
            encoded, global_state = model.encode(obs['history'], obs['adjacency'])
            final['candidate_pool'] = torch.tensor(candidates(model, encoded, global_state, obs, support), dtype=torch.long)
        rows.append((final, cpu_observation(nxt), {'kind': 'planner', 'edit': (3, -1, -1),
                    'commit': True, 'reward': reward, 'discount': gamma, 'terminal': terminal}))
        rows.append((final, cpu_observation(nxt), {'kind': 'executor', 'action': action,
                    'reward': reward, 'discount': gamma, 'terminal': terminal}))
        for effect in result.pop('effect_records'):
            partial = cpu_observation(obs)
            partial['active'] = effect['support']
            partial['candidate_pool'] = torch.tensor(effect['pool'], dtype=torch.long)
            rows.append((partial, partial, {'kind': 'effect', 'edit': effect['edit'],
                         'paired_effect': effect['paired_effect'], 'adapted_code': effect['adapted_code']}))
        result.update(period_end=env.time, reward=reward, reward_terms=terms,
                      additional_calls=probes.additional_calls, simulated_seconds=probes.simulated_seconds)
        diagnostics.append(result)
        obs = nxt
        observations.append(cpu_observation(obs))
    for index in range(max(0, len(observations) - 5)):
        auxiliary.append((observations[index], [observations[index + horizon]['history'][-1] for horizon in [1, 3, 5]]))
    record = {'scenario': scenario['id'], 'environment_seed': seed, 'run_id': run_id,
              'network': env.manifest['name'], 'fault_type': scenario.get('fault_type', 'unspecified'),
              'normal_delay': env.normal_delay, 'fault_delay': env.fault_delay,
              'recovery_seconds': env.recovery_seconds, 'recovery_delays': env.recovery_delays,
              'periods': diagnostics}
    return record, rows, auxiliary


class Trainer:
    def __init__(self, model, env, settings, seed=17):
        self.model, self.env, self.settings = model, env, settings
        self.target = copy.deepcopy(model).eval()
        self.seed = seed
        self.rng = random.Random(seed)
        self.memories = {kind: Replay(settings.get('replay_capacity', 1000000), seed)
                         for kind in ['planner', 'executor', 'effect']}
        self.auxiliary = deque(maxlen=100000)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        self.update_count = 0
        self.operation_counts = np.zeros(4)
        self.visit_counts = np.zeros((len(env.nodes), model.config.measures))

    def _move(self, obs):
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in obs.items()}

    def optimize(self, batch_size, encoder_only=False):
        self.model.train()
        self.optimizer.zero_grad()
        totals = {}
        if not encoder_only:
            for kind, function in [('planner', planner_loss), ('executor', executor_loss), ('effect', effect_loss)]:
                batch = self.memories[kind].sample(batch_size)
                for first, second, row in batch:
                    a, b = self._move(first), self._move(second)
                    loss = function(self.model, a, row) if kind == 'effect' else function(self.model, self.target, a, b, row, self.settings.get('alpha', .2))
                    if not torch.isfinite(loss):
                        raise ValueError(f'Nonfinite {kind} loss')
                    (loss / len(batch)).backward()
                    totals[kind] = totals.get(kind, 0.) + float(loss.detach()) / len(batch)
        samples = (self.rng.sample(list(self.auxiliary), batch_size) if len(self.auxiliary) >= batch_size
                   else self.rng.choices(list(self.auxiliary), k=batch_size)) if self.auxiliary else []
        for obs, future in samples:
            obs = self._move(obs)
            future = [value.to(obs['history'].device) for value in future]
            loss = self.model.auxiliary_loss(obs['history'], obs['adjacency'], future)
            ((1. if encoder_only else self.model.config.encoder_weight) * loss / len(samples)).backward()
            totals['encoder'] = totals.get('encoder', 0.) + float(loss.detach()) / len(samples)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        with torch.no_grad():
            for dst, src in zip(self.target.parameters(), self.model.parameters()):
                dst.lerp_(src, .005)
        self.update_count += 1
        return totals

    def collect(self, scenario, seed, calibration, coverage, balanced=False):
        record, rows, auxiliary = collect_episode(self.env, self.model, scenario, seed,
            calibration=calibration, stochastic=True, coverage=coverage, gamma=self.settings.get('gamma', .99),
            balanced=balanced, operation_counts=self.operation_counts, visit_counts=self.visit_counts)
        for row in rows:
            self.memories[row[2]['kind']].add(row)
        self.auxiliary.extend(auxiliary)
        return record

    def run(self, scenarios, validation, output):
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        logs = []
        stages = [('encoder', self.settings.get('encoder_updates', 100000), 256, 3e-4),
                  ('cold', self.settings.get('cold_start_updates', 50000), 256, 1e-4),
                  ('online', self.settings.get('online_updates', 200000), 512, 3e-4)]
        episode = 0
        if not scenarios or not validation:
            raise ValueError('Nonempty training and validation splits are required.')
        encoder_validation = []
        if self.settings.get('encoder_updates', 100000):
            path = self.settings.get('encoder_data')
            if not path:
                raise ValueError('Provide training.encoder_data with the five pretraining trajectory sources.')
            self.auxiliary.extend(load_encoder_data(path, self.model.config, 'train'))
            encoder_validation = load_encoder_data(path, self.model.config, 'validation')
        for stage, updates, default_batch, lr in stages:
            if stage == 'cold':
                for parameter in self.model.input.parameters(): parameter.requires_grad_(False)
                for parameter in self.model.temporal.parameters(): parameter.requires_grad_(False)
                for parameter in self.model.graph.parameters(): parameter.requires_grad_(False)
                self.model.position.requires_grad_(False)
            elif stage == 'online':
                for parameter in self.model.parameters(): parameter.requires_grad_(True)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            best = float('inf')
            for index in range(updates):
                if stage != 'encoder' and (index % self.settings.get('collect_every', 10) == 0 or not self.auxiliary):
                    scenario = scenarios[episode % len(scenarios)]
                    self.collect(scenario, self.seed + episode, stage != 'encoder',
                                 stage == 'online' and self.rng.random() < .15, balanced=stage == 'cold')
                    episode += 1
                totals = self.optimize(self.settings.get('batch_size_override', default_batch), stage == 'encoder')
                if (index + 1) % self.settings.get('validation_interval', 1000) == 0 or index + 1 == updates:
                    parameters = copy.deepcopy(self.model.state_dict())
                    values = []
                    if stage == 'encoder':
                        self.model.eval()
                        with torch.no_grad():
                            for o, f in encoder_validation:
                                moved = self._move(o)
                                values.append(float(self.model.auxiliary_loss(moved['history'], moved['adjacency'],
                                    [v.to(next(self.model.parameters()).device) for v in f])))
                    for j, scenario in enumerate(validation if stage != 'encoder' else []):
                        self.model.load_state_dict(parameters)
                        result, _, aux = collect_episode(self.env, self.model, scenario, self.seed + 100000 + j,
                                                         calibration=stage != 'encoder')
                        values.append(float(np.sum(result['recovery_delays'])))
                    self.model.load_state_dict(parameters)
                    score = float(np.mean(values))
                    if score < best:
                        best = score
                        save_checkpoint(output / f'{stage}_best.pt', self.model,
                            {'stage': stage, 'training_seed': self.seed, 'updates': self.update_count,
                             'validation_score': score, 'manifest_sha256': hashlib.sha256(self.env.path.read_bytes()).hexdigest()}, self.optimizer)
                    logs.append({'stage': stage, 'update': index + 1, 'loss': totals, 'validation': score})
            if updates:
                selected, _ = load_checkpoint(output / f'{stage}_best.pt', next(self.model.parameters()).device)
                self.model.load_state_dict(selected.state_dict())
                self.target.load_state_dict(self.model.state_dict())
        save_checkpoint(output / 'final.pt', self.model, {'training_seed': self.seed, 'updates': self.update_count}, self.optimizer)
        (output / 'training.json').write_text(json.dumps(logs, indent=2), encoding='utf-8')
