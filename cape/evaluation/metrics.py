import argparse
import json
from pathlib import Path
import numpy as np

def metrics(episode, horizon=1800, interval=30):
    (normal, fault) = (float(episode['normal_delay']), float(episode['fault_delay']))
    if not np.isfinite([normal, fault]).all() or fault <= normal:
        raise ValueError('Need separate finite normal/fault references with fault > normal.')
    times = np.asarray(episode['recovery_seconds'], dtype=float)
    delays = np.asarray(episode['recovery_delays'], dtype=float)
    expected = np.arange(interval, horizon + 1, interval)
    if times.shape != expected.shape or not np.array_equal(times, expected) or delays.shape != times.shape or (not np.isfinite(delays).all()):
        raise ValueError('Need 60 finite 30-second recovery-window delays at t=30,...,1800.')
    gains = 100 * (fault - delays) / (fault - normal)
    indices = np.flatnonzero(gains >= 0.8 * gains.max())
    return {'gain': float(gains[-1]), 't80': float(times[indices[0]]) if len(indices) else float(horizon)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', required=True, help='JSON array of independent reference records')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    episodes = json.loads(Path(args.episodes).read_text(encoding='utf-8'))
    rows = [metrics(episode) for episode in episodes]
    if not rows:
        raise ValueError('No episode records.')
    result = {'records': rows, 'summary': {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
if __name__ == '__main__':
    main()
