from pathlib import Path
import json
import uuid
import numpy as np
import traci
import yaml
from cape.data.schema import ActionSpec, NetworkSpec, Observation, Scenario

class SUMORuntime:
    def __init__(self, manifest_path, history, period_seconds):
        manifest = yaml.safe_load(Path(manifest_path).read_text(encoding='utf-8'))
        self.sumo, self.period_seconds, self.history_length = manifest['sumo'], period_seconds, history
        network = manifest['network']
        self.specification = NetworkSpec(tuple(network['nodes']), tuple(network['lanes']), np.asarray(network['adjacency'], dtype=np.float32), tuple(ActionSpec(item['name'], np.asarray(item['lower'], dtype=np.float32), np.asarray(item['upper'], dtype=np.float32)) for item in manifest['actions']))
        self.scenario_files = manifest['scenarios']
        self.history, self.active_pairs, self.control_history = [], frozenset(), frozenset()

    def scenarios(self, split):
        records = json.loads(Path(self.scenario_files[split]).read_text(encoding='utf-8'))
        return [Scenario(item['identifier'], tuple(item.get('arguments', [])), item['fault']) for item in records]

    def reset(self, scenario, seed):
        self.close()
        command = [self.sumo['binary'], '-c', self.sumo['configuration'], '--seed', str(seed), '--no-step-log', 'true'] + list(scenario.arguments)
        traci.start(command)
        self.history, self.active_pairs, self.control_history = [], frozenset(), frozenset()
        self._inject_fault(scenario.fault)
        for _ in range(self.history_length): self._advance(); self._record()
        return self.observation()

    def close(self):
        try: traci.close(False)
        except Exception: pass

    def _inject_fault(self, fault):
        for command in fault.get('commands', []):
            if command['type'] == 'lane_speed': traci.lane.setMaxSpeed(command['lane'], float(command['value']))
            elif command['type'] == 'traffic_program': traci.trafficlight.setProgram(command['node'], command['value'])

    def _advance(self):
        for _ in range(self.period_seconds): traci.simulationStep()

    def _record(self):
        rows = [[traci.lane.getLastStepHaltingNumber(lane), traci.lane.getLastStepVehicleNumber(lane), traci.lane.getLastStepMeanSpeed(lane), traci.lane.getWaitingTime(lane), traci.lane.getLastStepOccupancy(lane), traci.lane.getLength(lane)] for lane in self.specification.lanes]
        self.history = (self.history + [np.asarray(rows, dtype=np.float32)])[-self.history_length:]

    def observation(self): return Observation(np.stack(self.history), self.specification.adjacency, self.active_pairs, self.control_history)

    def snapshot(self):
        directory = Path(self.sumo['snapshot_directory']); directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{uuid.uuid4().hex}.xml'; traci.simulation.saveState(str(path))
        return str(path), list(self.history), self.active_pairs, self.control_history

    def restore(self, snapshot):
        path, self.history, self.active_pairs, self.control_history = snapshot; traci.simulation.loadState(path)

    def legal_pairs(self): return [(node, action) for node in range(len(self.specification.nodes)) for action in range(len(self.specification.actions))]

    def step(self, pairs, actions):
        self.control_history, self.active_pairs = self.active_pairs, frozenset(pairs)
        self._apply(pairs, actions); self._advance(); self._record()
        queue = np.asarray([traci.lane.getLastStepHaltingNumber(lane) for lane in self.specification.lanes], dtype=np.float32)
        storage = np.asarray([traci.lane.getLength(lane) for lane in self.specification.lanes], dtype=np.float32)
        ratio = np.divide(queue, storage, out=np.zeros_like(queue), where=storage > 0)
        delay = float(np.mean([traci.lane.getWaitingTime(lane) for lane in self.specification.lanes]))
        return self.observation(), -(delay + ratio.mean() + 2 * (ratio >= .9).mean()), traci.simulation.getMinExpectedNumber() <= 0, {'delay': delay, 'queue': float(ratio.mean()), 'spillback': float((ratio >= .9).mean())}

    def _apply(self, pairs, actions):
        for pair, values in zip(pairs, actions):
            node, action = pair; spec = self.specification.actions[action]; values = np.clip(values, spec.lower, spec.upper); target = self.specification.nodes[node]
            if spec.name == 'signal':
                phase = traci.trafficlight.getPhase(target); duration = max(float(values[0]), 1.0); traci.trafficlight.setPhaseDuration(target, duration)
                if len(values) > 1 and values[1] > .5: traci.trafficlight.setPhase(target, (phase + 1) % len(traci.trafficlight.getAllProgramLogics(target)[0].phases))
            elif spec.name == 'flow':
                for lane in self.manifest_lane_targets(target): traci.lane.setMaxSpeed(lane, float(values[0]))
            elif spec.name == 'diversion':
                for vehicle in traci.edge.getLastStepVehicleIDs(target): traci.vehicle.rerouteTraveltime(vehicle)

    def manifest_lane_targets(self, node):
        mapping = self.sumo.get('flow_lanes', {})
        return mapping.get(node, [node])
