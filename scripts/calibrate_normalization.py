import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cape.models.agent import Config
from cape.runtime.sumo import SUMOEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True); parser.add_argument('--scenarios', required=True)
    parser.add_argument('--seed', type=int, default=17); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    env = SUMOEnvironment(args.manifest, Config()); values = []
    try:
        for i, scenario in enumerate(json.loads(Path(args.scenarios).read_text(encoding='utf-8'))['train']):
            env.reset(scenario, args.seed + i)
            values.extend(env.normal_series)
    finally:
        env.dispose()
    scale = float(np.quantile(values, .95))
    if scale <= 0: raise ValueError('Normal training delays cannot define a positive scale.')
    source = Path(args.manifest).resolve()
    target = Path(args.output).resolve()
    manifest = json.loads(source.read_text(encoding='utf-8'))
    manifest['sumocfg'] = str((source.parent / manifest['sumocfg']).resolve())
    manifest['delay_scale'] = scale
    manifest['normalization'] = {'split': 'train', 'percentile': 95, 'samples': len(values), 'seed': args.seed}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
