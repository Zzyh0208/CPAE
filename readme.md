# CAPE

## Installation

```powershell
python -m pip install -e .
```

## Commands

```powershell
python scripts/train.py --config configs/cape.yaml --manifest manifests/your_network.yaml --output artifacts/train
python scripts/evaluate.py --config configs/cape.yaml --manifest manifests/your_network.yaml --output artifacts/test
```

## Architecture

The project contains temporal and graph encoding, node--measure candidate scoring, bounded Add/Delete/Swap/Stop planning, discrete Planner SAC, continuous Executor SAC, twin Q networks, target networks, strategy-code generation, deterministic action projection, matched branch evaluation, paired-effect alignment, triggered code adaptation, and generator distillation.

## Experiment Parameters

The values are 12 observation snapshots, 180-second control periods, three 60-second action waves, 64 candidates, at most 32 active nodes, eight internal edits, 128-dimensional strategy codes, four matched probe pairs, five code updates at learning rate 0.05, and disagreement threshold 0.63. Dataset-specific batch size, update count, discount, training horizon, seed protocol, and fault scenarios belong in `configs/experiments`.