import copy
import hashlib
import json
import socket
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
import traci
from .safety import Phase, project_program, validate_states, metering_program


class SUMOEnvironment:
    def __init__(self, manifest, config, device='cpu'):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
        self.config, self.device = config, device
        self.nodes = self.manifest['nodes']
        self.protocol = self.manifest.get('protocol', {})
        self.normal_end = int(self.protocol.get('normal_end', 900))
        self.recovery_start = int(self.protocol.get('recovery_start', 1800))
        self.end = int(self.protocol.get('end', 3600))
        self.period = int(self.protocol.get('period', 180))
        self.sample_interval = int(self.protocol.get('sample_interval', 15))
        self.metric_interval = int(self.protocol.get('metric_interval', 30))
        self.delay_scale = float(self.manifest['delay_scale'])
        if (self.delay_scale <= 0 or self.period % config.waves or
            not 0 < self.normal_end < self.recovery_start < self.end or
            (self.end - self.recovery_start) % self.period or self.period % self.metric_interval or
            config.measures != 3 or config.features != 6):
            raise ValueError('Invalid normalization or wave duration.')
        self.adjacency = np.asarray(self.manifest['adjacency'], dtype=np.float32)
        if self.adjacency.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError('Node adjacency size mismatch.')
        self.directory = tempfile.TemporaryDirectory(prefix='cape_sumo_')
        self.connection = None
        self.process = None
        self._log = None
        self.origin = None
        self.deadline = None

    def _launch(self, seed, snapshot=None, begin=0):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        configuration = str((self.path.parent / self.manifest['sumocfg']).resolve())
        command = [self.manifest.get('binary', 'sumo'), '-c', configuration,
                   '--remote-port', str(port), '--seed', str(seed), '--step-length', '1',
                   '--no-step-log', 'true', '--duration-log.disable', 'true',
                   '--save-state.rng', 'true', '--save-state.precision', '16',
                   '--time-to-teleport', '-1', '--begin', str(begin),
                   '--end', str(self.end + 1), '--route-steps', '0',
                   '--log', str(Path(self.directory.name) / 'sumo.log')]
        if snapshot:
            command += ['--load-state', str(snapshot)]
        self._log = open(Path(self.directory.name) / 'process.log', 'w', encoding='utf-8')
        self.process = subprocess.Popen(command, stdout=self._log, stderr=self._log)
        try:
            self.connection = traci.connect(port=port, numRetries=30, host='127.0.0.1',
                                             proc=self.process, waitBetweenRetries=.05)
        except Exception:
            self.close()
            raise

    def reset(self, scenario, seed):
        self.close()
        self.seed, self.scenario = int(seed), copy.deepcopy(scenario)
        self._launch(seed)
        c = self.connection
        self.lanes = tuple(lane for lane in c.lane.getIDList() if not lane.startswith(':'))
        self.storage = {lane: max(c.lane.getLength(lane) / 7.5, 1) for lane in self.lanes}
        self.free_speed = {lane: c.lane.getMaxSpeed(lane) for lane in self.lanes}
        self.initial_speeds = dict(self.free_speed)
        self.tls = {}
        owned = set()
        for node in self.nodes:
            for name, identity in [('signal', node.get('signal')), ('flow', node.get('meter'))]:
                if identity:
                    if identity in owned:
                        raise ValueError('Each signal or meter must have one control owner.')
                    owned.add(identity)
                    logic = next(logic for logic in c.trafficlight.getAllProgramLogics(identity)
                                 if logic.programID == c.trafficlight.getProgram(identity))
                    states = [phase.state for phase in logic.phases]
                    validate_states(states)
                    if name == 'signal' and sum(any(ch in 'gG' for ch in s) for s in states) + 2 > self.config.action_dim:
                        raise ValueError('Model action_dim does not cover this traffic-signal program.')
                    self.tls[identity] = {'program': logic.programID, 'states': states,
                        'durations': [phase.duration for phase in logic.phases]}
        self.history = deque(maxlen=self.config.history)
        self.trajectory_windows = []
        self.delays = []
        self.window = deque(maxlen=self.metric_interval)
        self.losses = {}
        self.active = np.zeros((len(self.nodes), self.config.measures), dtype=bool)
        self.previous = self.active.copy()
        self.manual = {}
        self.flow_release = {}
        self.guided = set()
        self.events_done = set()
        self.pending_vehicles = []
        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.last_diagnostics = {}
        self._record()
        normal = []
        while self.time < self.recovery_start:
            delay = self._tick()
            if self.time <= self.normal_end:
                normal.append(delay)
        self.normal_delay = float(np.mean(normal))
        self.normal_series = list(normal)
        self.fault_delay = float(np.mean(self.window))
        self.recovery_delays, self.recovery_seconds = [], []
        while len(self.history) < self.config.history:
            self.history.appendleft(self.history[0].copy())
        return self.observation()

    def _events(self):
        c = self.connection
        for index, event in enumerate(self.scenario.get('events', [])):
            if index in self.events_done or self.time < event['time']:
                continue
            kind = event['type']
            if kind == 'lane_speed':
                c.lane.setMaxSpeed(event['lane'], float(event['speed']))
            elif kind == 'lane_permission':
                c.lane.setDisallowed(event['lane'], event['disallowed'])
            elif kind == 'signal_program':
                c.trafficlight.setProgram(event['signal'], event['program'])
            elif kind == 'signal_state':
                c.trafficlight.setRedYellowGreenState(event['signal'], event['state'])
            elif kind == 'demand':
                for j in range(int(event['count'])):
                    self.pending_vehicles.append({'id': f'fault_{index}_{j}', 'route': event['route'],
                        'depart': self.time + j * event.get('interval', 1)})
            else:
                raise ValueError(f'Unsupported fault event {kind}')
            self.events_done.add(index)
        pending = []
        for vehicle in self.pending_vehicles:
            if vehicle['depart'] <= self.time:
                c.vehicle.add(vehicle['id'], vehicle['route'], depart='now')
            else:
                pending.append(vehicle)
        self.pending_vehicles = pending

    def _tick(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError('Simulation branch deadline exceeded.')
        self._events()
        for tls, timer in self.manual.items():
            if self.time >= timer['next']:
                timer['index'] = (timer['index'] + 1) % len(timer['phases'])
                phase = timer['phases'][timer['index']]
                self.connection.trafficlight.setRedYellowGreenState(tls, phase.state)
                timer['next'] = self.time + phase.duration
                timer['started'] = self.time
        self.connection.simulationStep()
        self.time = int(round(self.connection.simulation.getTime()))
        vehicle_ids = self.connection.vehicle.getIDList()
        losses = {v: self.connection.vehicle.getTimeLoss(v) for v in vehicle_ids}
        delay = float(np.mean([max(0., value - self.losses.get(v, 0.)) for v, value in losses.items()])) if losses else 0.
        self.losses = losses
        self.window.append(delay)
        if self.time % self.sample_interval == 0:
            self._record()
        if self.time > self.recovery_start and self.time % self.metric_interval == 0:
            self.recovery_seconds.append(self.time - self.recovery_start)
            self.recovery_delays.append(float(np.mean(self.window)))
        return delay

    def _record(self):
        c = self.connection
        rows = []
        for node in self.nodes:
            lanes = node['lanes']
            queue = [c.lane.getLastStepHaltingNumber(lane) / self.storage[lane] for lane in lanes]
            counts = [c.lane.getLastStepVehicleNumber(lane) / self.storage[lane] for lane in lanes]
            speed = [max(0, c.lane.getLastStepMeanSpeed(lane)) / max(self.free_speed[lane], 1e-6) for lane in lanes]
            occupancy = [c.lane.getLastStepOccupancy(lane) / 100 for lane in lanes]
            waiting = [c.lane.getWaitingTime(lane) / max(1, c.lane.getLastStepVehicleNumber(lane)) for lane in lanes]
            rows.append([np.mean(queue), np.mean(counts), np.mean(speed), np.mean(occupancy),
                         np.mean(waiting) / float(self.manifest.get('waiting_scale', 180)), np.mean(np.asarray(queue) >= .9)])
        self.history.append(np.asarray(rows, dtype=np.float32))
        if hasattr(self, 'trajectory_windows') and len(self.history) == self.config.history and self.time % self.period == 0:
            self.trajectory_windows.append((self.time, np.stack(self.history).copy()))

    def observation(self):
        n, m, w, d = len(self.nodes), self.config.measures, self.config.waves, self.config.action_dim
        legal = np.array([[bool(node.get('signal')), bool(node.get('meter')), bool(node.get('diversion'))] for node in self.nodes])
        legal &= np.asarray(self.config.enabled_measures, dtype=bool)[None]
        for node_index, measure in self.scenario.get('unavailable_pairs', []):
            legal[node_index, measure] = False
        mask = np.zeros((n, m, w, d), dtype=bool)
        for index, node in enumerate(self.nodes):
            if legal[index, 0]:
                count = 2 + sum(any(ch in 'gG' for ch in state) for state in self.tls[node['signal']]['states'])
                mask[index, 0, :, :count] = True
        mask[:, 1:, :, 0] = legal[:, 1:, None]
        values = {'history': np.stack(self.history), 'adjacency': self.adjacency,
                  'legal': legal, 'active': self.active, 'previous': self.previous,
                  'dim_mask': mask, 'lower': -np.ones(mask.shape), 'upper': np.ones(mask.shape)}
        return {key: torch.as_tensor(value, dtype=torch.bool if key in ['legal', 'active', 'previous', 'dim_mask'] else torch.float32,
                                     device=self.device) for key, value in values.items()}

    def _set_program(self, tls, phases, offset=None):
        c = self.connection
        current = c.trafficlight.getRedYellowGreenState(tls)
        matches = [i for i, phase in enumerate(phases) if phase.state == current]
        if not matches:
            raise ValueError(f'Current state of {tls} cannot transition into the configured legal phase sequence.')
        if tls in self.manual:
            elapsed = max(0., self.time - self.manual[tls]['started'])
        else:
            elapsed = float(c.trafficlight.getSpentDuration(tls))
        index = matches[0]
        remaining = max(1., phases[index].minimum - elapsed,
                        phases[index].duration - elapsed)
        if offset is not None:
            cycle = sum(phase.duration for phase in phases)
            desired_end = (offset + sum(phase.duration for phase in phases[:index + 1]) - self.time) % cycle
            remaining = float(np.clip(desired_end, max(1., phases[index].minimum - elapsed),
                                      max(1., phases[index].duration - elapsed)))
        self.manual[tls] = {'index': index, 'phases': phases, 'next': self.time + remaining,
                            'started': self.time - elapsed}
        c.trafficlight.setRedYellowGreenState(tls, current)

    def prepare_action(self, support, action):
        """Project all three waves without advancing or modifying the environment."""
        support = np.asarray(support, dtype=bool)
        action = np.asarray(action, dtype=float)
        expected = (len(self.nodes), self.config.measures, self.config.waves, self.config.action_dim)
        if support.shape != expected[:2] or action.shape != expected or not np.isfinite(action).all():
            raise ValueError('Invalid control support or action.')
        if np.any(support & ~self.observation()['legal'].cpu().numpy()):
            raise ValueError('Illegal node-measure pair.')
        action = np.clip(action, -1, 1)
        waves = []
        for wave in range(self.config.waves):
            programs = {}
            releases = {}
            for i, node in enumerate(self.nodes):
                for measure, key in [(0, 'signal'), (1, 'meter')]:
                    tls = node.get(key)
                    if not tls:
                        continue
                    plan = self.tls[tls]
                    rules = node.get('signal_rules' if measure == 0 else 'meter_rules', {})
                    offset = None
                    if support[i, measure]:
                        if measure == 0:
                            phases = project_program(plan['states'], plan['durations'], action[i, measure, wave], rules)
                            offset = (float(action[i, measure, wave, 1]) + 1) * .5 * sum(p.duration for p in phases)
                        else:
                            release = .4 + .6 * (action[i, measure, wave, 0] + 1) / 2
                            phases = metering_program(plan['states'], release, rules)
                            releases[str(i)] = float(release)
                    elif tls in self.manual:
                        phases = []
                        for index, (state, duration) in enumerate(zip(plan['states'], plan['durations'])):
                            minimum = rules.get('yellow', 3.) if 'y' in state.lower() else rules.get('all_red', 1.) if not any(ch in 'gG' for ch in state) else rules.get('min_green', 5.)
                            phases.append(Phase(state, max(duration, minimum), minimum))
                            nxt = plan['states'][(index + 1) % len(plan['states'])]
                            if 'y' in state.lower() and any(ch in 'gG' for ch in nxt):
                                if any(ch in 'gG' for ch in state):
                                    yellow = rules.get('yellow', 3.)
                                    phases.append(Phase(''.join('y' if ch in 'gG' else ch for ch in state), yellow, yellow))
                                phases.append(Phase('r' * len(state), rules.get('all_red', 1.), rules.get('all_red', 1.)))
                    else:
                        continue
                    validate_states([phase.state for phase in phases])
                    current = self.connection.trafficlight.getRedYellowGreenState(tls)
                    if current not in {phase.state for phase in phases}:
                        raise ValueError(f'Current state of {tls} is outside the legal phase sequence.')
                    programs[tls] = {'phases': [vars(phase) for phase in phases], 'offset': offset}
            waves.append({'programs': programs, 'releases': releases})
        return {'time': self.time, 'support': support.tolist(), 'normalized_actions': action.tolist(), 'waves': waves}

    def _apply_wave(self, support, action, wave, prepared):
        plan = prepared['waves'][wave]
        for tls, program in plan['programs'].items():
            self._set_program(tls, [Phase(**phase) for phase in program['phases']], program['offset'])
        self.flow_release.update({int(i): release for i, release in plan['releases'].items()})
        self._apply_diversion(support, action, wave)

    def _apply_diversion(self, support, action, wave):
        for i, node in enumerate(self.nodes):
            if support[i, 2]:
                ratio = .3 * (action[i, 2, wave, 0] + 1) / 2
                for mapping in node['diversion']:
                    for vehicle in self.connection.edge.getLastStepVehicleIDs(mapping['edge']):
                        if vehicle in self.guided:
                            continue
                        self.guided.add(vehicle)
                        current_route = self.connection.vehicle.getRoute(vehicle)
                        alternatives = [route for route in mapping['routes'] if route[0] == mapping['edge'] and route[-1] == current_route[-1]]
                        if alternatives and self.rng.random() < ratio:
                            self.connection.vehicle.setRoute(vehicle, alternatives[int(self.rng.integers(len(alternatives)))])

    def step(self, support, action, prepared=None):
        if self.time >= self.end:
            raise RuntimeError('Episode already ended.')
        support = np.asarray(support, dtype=bool)
        action = np.asarray(action, dtype=float)
        if support.shape != (len(self.nodes), self.config.measures):
            raise ValueError('Invalid control-set shape.')
        if action.shape != (len(self.nodes), 3, self.config.waves, self.config.action_dim) or not np.isfinite(action).all():
            raise ValueError('Invalid action shape or values.')
        legal = self.observation()['legal'].cpu().numpy()
        if np.any(support & ~legal):
            raise ValueError('Illegal node-measure pair.')
        if prepared is None:
            prepared = self.prepare_action(support, action)
        if (prepared['time'] != self.time or prepared['support'] != support.tolist()
                or prepared['normalized_actions'] != np.clip(action, -1, 1).tolist()):
            raise ValueError('Prepared action does not match this state and action.')
        old = self.active.copy()
        self.previous, self.active = old, support.copy()
        period_end = min(self.end, self.time + self.period)
        delays, queues, spillbacks = [], [], []
        for wave in range(self.config.waves):
            self._apply_wave(support, np.clip(action, -1, 1), wave, prepared)
            wave_end = min(period_end, self.time + self.period // self.config.waves)
            while self.time < wave_end:
                delays.append(self._tick())
                ratios = [self.connection.lane.getLastStepHaltingNumber(lane) / self.storage[lane] for lane in self.lanes]
                queues.append(np.mean(ratios)); spillbacks.append(np.mean(np.asarray(ratios) >= .9))
        mask = self.observation()['dim_mask'].cpu().numpy()
        default_nominal = np.zeros_like(action)
        default_nominal[:, 1, :, 0] = 1.
        default_nominal[:, 2, :, 0] = -1.
        nominal = np.asarray(self.manifest.get('nominal_actions', default_nominal), dtype=float)
        pair_cost = []
        for node, measure in zip(*np.nonzero(support)):
            valid = mask[node, measure]
            if valid.any():
                pair_cost.append(float(np.abs(action[node, measure] - nominal[node, measure])[valid].mean() / 2))
        cost = float(np.mean(pair_cost)) if pair_cost else 0.
        switching = np.count_nonzero(old ^ support) / max(1, np.count_nonzero(old | support))
        terms = np.array([np.mean(delays) / self.delay_scale, np.mean(queues), np.mean(spillbacks), cost, switching])
        weights = np.array(self.manifest.get('reward_weights', [1, 1, 2, .10, .05]))
        reward = -float(terms @ weights)
        self.last_diagnostics = dict(zip(['delay', 'queue', 'spillback', 'control', 'switching'], terms.tolist()))
        return self.observation(), reward, self.time >= self.end, dict(self.last_diagnostics)

    def snapshot(self):
        target = Path(self.directory.name) / f'state_{self.time}_{time.time_ns()}.xml.gz'
        self.connection.simulation.saveState(str(target))
        keys = ['seed', 'scenario', 'lanes', 'storage', 'free_speed', 'initial_speeds', 'tls', 'history',
                'delays', 'window', 'losses', 'active', 'previous', 'manual', 'flow_release', 'guided',
                'events_done', 'pending_vehicles', 'time', 'normal_delay', 'normal_series', 'fault_delay', 'recovery_delays', 'recovery_seconds', 'last_diagnostics']
        state = {key: copy.deepcopy(getattr(self, key)) for key in keys}
        state['rng_state'] = copy.deepcopy(self.rng.bit_generator.state)
        state['lane_speeds'] = {lane: self.connection.lane.getMaxSpeed(lane) for lane in self.lanes}
        state['lane_disallowed'] = {lane: self.connection.lane.getDisallowed(lane) for lane in self.lanes}
        return {'file': str(target), 'state': state}

    @classmethod
    def fork(cls, manifest, config, snapshot, device='cpu'):
        env = cls(manifest, config, device)
        state = copy.deepcopy(snapshot['state'])
        env._launch(state['seed'], snapshot['file'], state['time'])
        for key, value in state.items():
            if key not in ['rng_state', 'lane_speeds', 'lane_disallowed']:
                setattr(env, key, value)
        env.rng = np.random.default_rng()
        env.rng.bit_generator.state = state['rng_state']
        for lane, speed in state['lane_speeds'].items():
            env.connection.lane.setMaxSpeed(lane, speed)
            env.connection.lane.setDisallowed(lane, state['lane_disallowed'][lane])
        for tls, timer in env.manual.items():
            env.connection.trafficlight.setRedYellowGreenState(tls, timer['phases'][timer['index']].state)
        return env

    def close(self):
        if self.connection:
            try:
                self.connection.close(False)
            except Exception:
                pass
            self.connection = None
        if self.process:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
            self.process = None
        if self._log:
            self._log.close(); self._log = None

    def dispose(self):
        self.close()
        self.directory.cleanup()
