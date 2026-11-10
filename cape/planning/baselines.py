import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from cape.runtime.sumo import SUMOEnvironment


class FixedTime:
    def choose(self, environment):
        shape = (len(environment.nodes), 3, environment.config.waves, environment.config.action_dim)
        return {'support': np.zeros(shape[:2], dtype=bool).tolist(), 'normalized_actions': np.zeros(shape).tolist(),
                'timing': {'end_to_end_seconds': 0.}, 'additional_calls': 0, 'timeout': False}


class CEMMPC:
    def __init__(self, samples=32, iterations=3, horizon=3, elite_fraction=.2,
                 workers=4, gamma=.99, seed=17):
        if samples < 2 or iterations < 1 or horizon < 1:
            raise ValueError('Invalid CEM budget.')
        self.samples, self.iterations, self.horizon = samples, iterations, horizon
        self.elites = max(1, int(samples * elite_fraction))
        self.workers, self.gamma = workers, gamma
        self.rng = np.random.default_rng(seed)

    def choose(self, environment, budget):
        start = time.perf_counter()
        deadline = start + budget
        obs = environment.observation()
        support = obs['legal'].cpu().numpy()
        shape = (self.horizon,) + tuple(obs['dim_mask'].shape)
        mask = obs['dim_mask'].cpu().numpy()[None]
        mean, std = np.zeros(shape), np.ones(shape)
        best_action = np.zeros(shape[1:]); best_return = -np.inf
        calls, simulated_seconds, timed_out = 0, 0, False
        snapshot = environment.snapshot()

        def evaluate(sequence):
            if time.perf_counter() >= deadline:
                return None, 0, 0
            branch = SUMOEnvironment.fork(environment.path, environment.config, snapshot)
            branch.deadline = deadline
            initial_time = branch.time
            value, discount = 0., 1.
            try:
                for action in sequence:
                    if branch.time >= branch.end:
                        break
                    _, reward, _, _ = branch.step(support, action)
                    value += discount * reward; discount *= self.gamma
                return value, branch.time - initial_time, 1
            except TimeoutError:
                return None, branch.time - initial_time, 1
            finally:
                branch.dispose()

        for _ in range(self.iterations):
            sequences = np.clip(mean + std * self.rng.standard_normal((self.samples,) + shape), -1, 1) * mask
            results = []
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(evaluate, sequence): i for i, sequence in enumerate(sequences)}
                for future in as_completed(futures):
                    value, seconds, count = future.result()
                    calls += count; simulated_seconds += seconds
                    if value is not None:
                        index = futures[future]; results.append((value, index))
                        if value > best_return:
                            best_return, best_action = value, sequences[index, 0].copy()
            if len(results) < self.samples:
                timed_out = True
            if results:
                indices = [i for _, i in sorted(results, reverse=True)[:self.elites]]
                mean = sequences[indices].mean(0); std = sequences[indices].std(0).clip(.05, 1)
            if time.perf_counter() >= deadline:
                timed_out = True; break
        return {'support': support.tolist(), 'normalized_actions': best_action.tolist(),
                'timing': {'end_to_end_seconds': time.perf_counter() - start},
                'additional_calls': calls, 'simulated_seconds': simulated_seconds, 'timeout': timed_out,
                'fallback': not np.isfinite(best_return),
                'cem': {'samples': self.samples, 'iterations': self.iterations, 'horizon': self.horizon}}
