import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cape.models.agent import CAPE, Config
from cape.runtime.sumo import SUMOEnvironment
from cape.learning.pipeline import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--scenarios', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if cfg['training'].get('encoder_data'):
        cfg['training']['encoder_data'] = str((Path(args.config).resolve().parent / cfg['training']['encoder_data']).resolve())
    splits = json.loads(Path(args.scenarios).read_text(encoding='utf-8'))
    model = CAPE(Config(**cfg['model'])).to(args.device)
    env = SUMOEnvironment(args.manifest, model.config, args.device)
    try:
        Trainer(model, env, cfg['training'], args.seed).run(splits['train'], splits['validation'], args.output)
    finally:
        env.dispose()


if __name__ == '__main__':
    main()
