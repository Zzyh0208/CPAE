import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cape.models.agent import Config
from cape.runtime.sumo import SUMOEnvironment
from cape.planning.baselines import CEMMPC, FixedTime
from cape.evaluation.metrics import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True); parser.add_argument('--scenarios', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--method', choices=['fixed', 'cem'], required=True)
    parser.add_argument('--budget', type=float, default=45); parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=3); parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--seed', type=int, default=10017)
    parser.add_argument('--run-id', default='0')
    parser.add_argument('--sampling-seed', type=int, default=17)
    args = parser.parse_args(); torch.set_num_threads(2)
    env = SUMOEnvironment(args.manifest, Config())
    results = []
    try:
        for index, scenario in enumerate(json.loads(Path(args.scenarios).read_text(encoding='utf-8'))['test']):
            env.reset(scenario, args.seed + index)
            policy = CEMMPC(args.samples, args.iterations, args.horizon, seed=args.sampling_seed + index) if args.method == 'cem' else FixedTime()
            periods = []
            while env.time < env.end:
                start = time.perf_counter()
                action = policy.choose(env, args.budget) if args.method == 'cem' else policy.choose(env)
                prepared = env.prepare_action(action['support'], action['normalized_actions'])
                elapsed = time.perf_counter() - start
                action['timing'].update(end_to_end_seconds=elapsed, policy_inference_seconds=elapsed,
                                       includes_safety_projection=True)
                _, reward, _, terms = env.step(action['support'], action['normalized_actions'], prepared=prepared)
                periods.append(dict(action, reward=reward, terms=terms))
            row = {'network': env.manifest['name'], 'scenario': scenario['id'], 'fault_type': scenario['fault_type'],
                   'environment_seed': args.seed + index, 'run_id': args.run_id,
                   'method': args.method, 'normal_delay': env.normal_delay,
                   'fault_delay': env.fault_delay, 'recovery_seconds': env.recovery_seconds,
                   'recovery_delays': env.recovery_delays, 'periods': periods}
            if args.method == 'cem':
                row['sampling_seed'] = args.sampling_seed + index
            row.update(metrics(row, env.end - env.recovery_start, env.metric_interval)); results.append(row)
    finally:
        env.dispose()
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__': main()
