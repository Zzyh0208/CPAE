import json
from pathlib import Path
import numpy as np
import torch


SOURCES = {'normal', 'fault-event', 'nominal-control', 'bounded-random-control', 'max-pressure'}


def load_encoder_data(index_path, config, split):
    index_path = Path(index_path).resolve()
    records = json.loads(index_path.read_text(encoding='utf-8'))
    selected = [r for r in records if r['split'] == split]
    if not selected or {r['source'] for r in selected} != SOURCES:
        raise ValueError(f'Encoder {split} data must include all five trajectory sources: {sorted(SOURCES)}')
    result = []
    for record in selected:
        with np.load(index_path.parent / record['file'], allow_pickle=False) as data:
            history, future, adjacency = data['history'], data['future'], data['adjacency']
            if history.ndim != 4 or history.shape[1] != config.history or history.shape[-1] != config.features:
                raise ValueError('Encoder history must be [samples,time,nodes,features].')
            if future.shape != (len(history), 3, history.shape[2], config.features):
                raise ValueError('Future targets must be horizons 1,3,5: [samples,3,nodes,features].')
            if adjacency.shape != (history.shape[2], history.shape[2]):
                raise ValueError('Encoder adjacency dimension mismatch.')
            if not all(np.isfinite(v).all() for v in [history, future, adjacency]):
                raise ValueError('Nonfinite encoder data.')
            for x, y in zip(history, future):
                obs = {'history': torch.tensor(x, dtype=torch.float32),
                       'adjacency': torch.tensor(adjacency, dtype=torch.float32)}
                result.append((obs, [torch.tensor(target, dtype=torch.float32) for target in y]))
    return result
