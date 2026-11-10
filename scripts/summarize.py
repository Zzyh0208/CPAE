import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cape.evaluation.statistics import paired_hierarchical_bootstrap, benjamini_hochberg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference'); parser.add_argument('--candidate')
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()
    rows = [row for file in args.inputs for row in json.loads(Path(file).read_text(encoding='utf-8'))]
    summary = {}
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        periods = [p for r in selected for p in r['periods']]
        groups = {}
        for metric in ['policy_inference_seconds', 'end_to_end_seconds']:
            groups[metric] = {}
            for name, group in [('all', periods), ('triggered', [p for p in periods if p.get('calibrated')]),
                                ('untriggered', [p for p in periods if not p.get('calibrated')])]:
                times = [p['timing'][metric] for p in group]
                groups[metric][name] = {'count': len(times), 'mean_s': float(np.mean(times)), 'p95_s': float(np.quantile(times, .95)),
                                'max_s': float(np.max(times)), 'over_180s': float(np.mean(np.asarray(times) > 180))} if times else {'count': 0}
        strata = {}
        for row in selected:
            strata.setdefault((row['network'], row['fault_type']), []).append(row['gain'])
        summary[method] = {'episodes': len(selected), 'rg': float(np.mean([np.mean(v) for v in strata.values()])),
                           't80': float(np.mean([r['t80'] for r in selected])), 'latency': groups,
                           'extra_calls_per_episode': float(np.mean([sum(p.get('additional_calls', 0) for p in r['periods']) for r in selected]))}
    if args.reference and args.candidate:
        paired = paired_hierarchical_bootstrap(rows, args.reference, args.candidate, samples=args.samples)
        paired['bh_adjusted_p'] = benjamini_hochberg([paired['p']])[0]
        summary['paired'] = paired
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__': main()
