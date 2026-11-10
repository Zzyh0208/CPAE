import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cape.models.agent import load_checkpoint
from cape.runtime.sumo import SUMOEnvironment
from cape.learning.pipeline import collect_episode
from cape.evaluation.metrics import metrics
from cape.runtime.memory import reset_peak_memory, peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--scenarios', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=10017)
    parser.add_argument('--run-id', default='0')
    parser.add_argument('--budget', type=float)
    parser.add_argument('--no-calibration', action='store_true')
    parser.add_argument('--profile', help='JSON object of inference resource overrides')
    parser.add_argument('--method-name', default='CAPE')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=['train','validation','test'], default='test')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    if args.profile:
        profile = json.loads(Path(args.profile).read_text(encoding='utf-8'))
        allowed = {'max_candidates', 'max_nodes', 'max_edits', 'probe_pairs', 'calibration_steps',
                   'threshold', 'calibration_enabled', 'enabled_measures'}
        if set(profile) - allowed:
            raise ValueError('Profile may override inference resources only, not architecture.')
        for key, value in profile.items():
            setattr(model.config, key, value)
    original = copy.deepcopy(model.state_dict())
    env = SUMOEnvironment(args.manifest, model.config, args.device)
    rows = []
    try:
        for i, scenario in enumerate(json.loads(Path(args.scenarios).read_text(encoding='utf-8'))[args.split]):
            model.load_state_dict(original)
            reset_peak_memory(args.device)
            row, _, _ = collect_episode(env, model, scenario, args.seed + i,
                calibration=not args.no_calibration, budget=args.budget, run_id=args.run_id)
            row.update(training_seed=metadata.get('training_seed'), method=args.method_name,
                       gpu_memory=peak_memory(args.device),
                       checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
            row.update(metrics(row, horizon=env.end - env.recovery_start, interval=env.metric_interval))
            rows.append(row)
    finally:
        env.dispose()
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
