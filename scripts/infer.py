import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cape.models.agent import load_checkpoint
from cape.runtime.sumo import SUMOEnvironment
from cape.runtime.probes import MatchedProbes
from cape.planning.controller import decide
from cape.runtime.memory import reset_peak_memory, peak_memory


def main():
    parser = argparse.ArgumentParser(description='Run closed-loop CAPE inference using trained weights.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--scenario', required=True, help='JSON object containing one scenario')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--budget', type=float)
    parser.add_argument('--no-calibration', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    environment = SUMOEnvironment(args.manifest, model.config, args.device)
    records = []
    reset_peak_memory(args.device)
    try:
        observation = environment.reset(json.loads(Path(args.scenario).read_text(encoding='utf-8')), args.seed)
        while environment.time < environment.end:
            deadline = None if args.budget is None else time.perf_counter() + args.budget
            probes = MatchedProbes(environment, workers=4, deadline=deadline)
            decision = decide(model, observation,
                probes if model.config.calibration_enabled and not args.no_calibration else None, deadline=deadline,
                action_preparer=environment.prepare_action)
            observation, reward, done, components = environment.step(decision['support'], decision['normalized_actions'],
                prepared=decision.pop('prepared_action'))
            records.append({'time': environment.time, 'decision': decision, 'reward': reward,
                            'reward_components': components, 'additional_calls': probes.additional_calls,
                            'simulated_seconds': probes.simulated_seconds})
            if done:
                break
    finally:
        environment.dispose()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'checkpoint_metadata': metadata, 'gpu_memory': peak_memory(args.device), 'periods': records}, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
