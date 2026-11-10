import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cape.models.agent import Config
from cape.runtime.sumo import SUMOEnvironment


def pressure_action(env):
    obs = env.observation()
    action = np.zeros(obs['dim_mask'].shape)
    support = np.zeros(obs['legal'].shape, dtype=bool)
    for index, node in enumerate(env.nodes):
        tls = node.get('signal')
        if not tls or not bool(obs['legal'][index, 0]):
            continue
        support[index, 0] = True
        links = env.connection.trafficlight.getControlledLinks(tls)
        pressures = []
        for state in env.tls[tls]['states']:
            if not any(c in 'gG' for c in state):
                continue
            pressure = 0.
            for link_index, signal in enumerate(state):
                if signal in 'gG':
                    for incoming, outgoing, _ in links[link_index]:
                        pressure += env.connection.lane.getLastStepHaltingNumber(incoming) - env.connection.lane.getLastStepHaltingNumber(outgoing)
            pressures.append(pressure)
        if pressures:
            scale = max(1., max(abs(v) for v in pressures))
            for phase, value in enumerate(pressures):
                coordinate = 2 + phase
                if coordinate >= env.config.action_dim:
                    raise ValueError('action_dim is smaller than signal phase representation.')
                action[index, 0, :, coordinate] = np.clip(value / scale, -1, 1)
    return support, action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True); parser.add_argument('--scenarios', required=True)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    config = Config(**json.loads(Path(args.config).read_text(encoding='utf-8'))['model'])
    splits = json.loads(Path(args.scenarios).read_text(encoding='utf-8'))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    index_rows = []
    env = SUMOEnvironment(args.manifest, config)
    try:
        for split in ['train', 'validation']:
            for i, scenario in enumerate(splits[split]):
                for style in ['nominal-control', 'bounded-random-control', 'max-pressure']:
                    rng = np.random.default_rng(args.seed + i)
                    obs = env.reset(scenario, args.seed + i)
                    while env.time < env.end:
                        if style == 'max-pressure':
                            support, action = pressure_action(env)
                        elif style == 'bounded-random-control':
                            support = obs['legal'].cpu().numpy()
                            action = rng.uniform(-.5, .5, size=tuple(obs['dim_mask'].shape))
                        else:
                            support = np.zeros(obs['legal'].shape, dtype=bool)
                            action = np.zeros(obs['dim_mask'].shape)
                        obs, _, _, _ = env.step(support, action)
                    windows = {t: x for t, x in env.trajectory_windows}
                    groups = {}
                    for t, history in windows.items():
                        targets = [t + env.period * horizon for horizon in [1, 3, 5]]
                        if not all(value in windows for value in targets):
                            continue
                        source = 'normal' if t < env.normal_end else 'fault-event' if t < env.recovery_start else style
                        groups.setdefault(source, []).append((history, np.stack([windows[value][-1] for value in targets])))
                    for source, samples in groups.items():
                        identity = f'{split}_{i}_{style}_{source}.npz'
                        np.savez_compressed(output / identity, history=np.stack([s[0] for s in samples]),
                                            future=np.stack([s[1] for s in samples]), adjacency=env.adjacency)
                        index_rows.append({'file': identity, 'source': source, 'split': split, 'scenario': scenario['id']})
    finally:
        env.dispose()
    (output / 'index.json').write_text(json.dumps(index_rows, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
