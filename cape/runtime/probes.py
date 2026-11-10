import copy
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from .sumo import SUMOEnvironment


class MatchedProbes:
    def __init__(self, environment, workers=4, gamma=.99, deadline=None):
        self.environment, self.workers, self.gamma = environment, workers, gamma
        self.deadline = deadline
        self.cache = {}
        self.additional_calls = 0
        self.simulated_seconds = 0
        self.last = None
        self.snapshot_cache = None
        self.lock = threading.Lock()

    def __call__(self, model, observation, reference, edited, code, pairs):
        if self.snapshot_cache is None or self.snapshot_cache['state']['time'] != self.environment.time:
            self.snapshot_cache = self.environment.snapshot()
        snapshot = self.snapshot_cache
        policy = copy.deepcopy(model).eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        digest = hashlib.sha256()
        for value in policy.state_dict().values():
            digest.update(value.detach().cpu().numpy().tobytes())
        policy_id = digest.hexdigest()
        references, edited_returns = {}, {}
        jobs = []
        for pair in range(pairs):
            seed = int(self.environment.seed * 100003 + self.environment.time * 101 + pair)
            key = (snapshot['file'], policy_id, reference.cpu().numpy().tobytes(), seed)
            if key in self.cache:
                references[pair] = self.cache[key]
            else:
                jobs.append((pair, False, seed, key))
            jobs.append((pair, True, seed, None))
        before = self.additional_calls
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._branch, policy, snapshot, edited if is_edit else reference,
                                       code if is_edit else None, seed): (pair, is_edit, key)
                           for pair, is_edit, seed, key in jobs}
                for future in as_completed(futures):
                    pair, is_edit, key = futures[future]
                    value, action, seconds = future.result()
                    if is_edit:
                        edited_returns[pair] = (value, action)
                    else:
                        references[pair] = value
                        self.cache[key] = value
        except TimeoutError:
            raise
        records = [{'reference_return': references[i], 'edited_return': edited_returns[i][0],
                    'edited_action': edited_returns[i][1]} for i in range(pairs)]
        self.last = {'records': records, 'additional_calls': self.additional_calls - before}
        return self.last

    def _branch(self, policy, snapshot, support, first_code, seed):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError('Probe deadline exceeded before launch.')
        device = next(policy.parameters()).device
        env = SUMOEnvironment.fork(self.environment.path, policy.config, snapshot, device=device)
        with self.lock:
            self.additional_calls += 1
        env.deadline = self.deadline
        support = support.detach().to(device)
        support_array = support.cpu().numpy()
        rng = np.random.default_rng(seed)
        total, discount, first_action, start = 0., 1., None, env.time
        try:
            while env.time < env.end:
                obs = env.observation()
                with torch.no_grad():
                    h, g = policy.encode(obs['history'], obs['adjacency'])
                    z = first_code.detach().to(device) if first_action is None and first_code is not None else policy.code(h, g, support)
                    distribution = policy.distribution(h, g, support, z)
                    noise = torch.as_tensor(rng.standard_normal(distribution.loc.shape), dtype=distribution.loc.dtype, device=device)
                    action = (distribution.loc + distribution.scale * noise).tanh()
                    action *= support[..., None, None] & obs['dim_mask']
                if first_action is None:
                    first_action = action.cpu().numpy().copy()
                _, reward, _, _ = env.step(support_array, action.cpu().numpy())
                total += discount * reward
                discount *= self.gamma
            return total, first_action, env.time - start
        finally:
            with self.lock:
                self.simulated_seconds += env.time - start
            env.dispose()
