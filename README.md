# CAPE

Closed-Loop Adaptive Planning and Execution for Multi-Node Traffic Fault Recovery.

## Installation

- Python 3.10+
- PyTorch with CUDA support
- SUMO 1.20.0 and TraCI
- NVIDIA RTX 4060 8GB

The experiment server uses dual Xeon Gold 6338 processors and 256GB RAM. Add `sumo` to `PATH`, then install from the repository root:

```bash
python -m pip install -e .
```

Training, inference, and evaluation use `cuda:0` by default. Select a device with `--device cuda:0`.

## Repository Structure

```text
cape/
  data/          Encoder trajectory loading and validation
  models/        Temporal and graph encoders, Planner, Executor, twin Q and V heads
  planning/      Control-set editing, calibration, FixedTime, and CEM-MPC
  learning/      SAC objectives, replay buffers, and three-stage training
  runtime/       SUMO execution, safety projection, and matched probe branches
  evaluation/    Recovery metrics, paired bootstrap, and BH correction
configs/         Model, training, ablation, and inference budget settings
scripts/         Data collection, training, inference, evaluation, and summaries
```

## Input Data

Prepare a network manifest, SUMO network and route files, and a scenario file with `train`, `validation`, and `test` lists.

The manifest follows [configs/manifest.schema.json](configs/manifest.schema.json):

| Field | Description |
|---|---|
| `name` | Network identifier |
| `sumocfg` | SUMO configuration path, relative to the manifest |
| `delay_scale` | Positive delay normalization scale |
| `nodes` | Node IDs, incoming `lanes`, and available `signal`, `meter`, and `diversion` controls |
| `adjacency` | Directed adjacency matrix; `adjacency[target][source] > 0` denotes an edge |
| `protocol` | Episode duration, control period, and observation intervals |

Assign a separate signal controller to each signal or meter control. Each scenario contains `id`, `fault_type`, and `events`. Supported event types are `lane_speed`, `lane_permission`, `signal_program`, `signal_state`, and `demand`. Use multiple events for compound faults. `unavailable_pairs` lists unavailable node and measure indices.

The default episode has normal traffic at 0-900s, a fault interval at 900-1800s, and recovery at 1800-3600s. Planning runs every 180s with three 60s action waves. Observations are sampled every 15s, and recovery metrics every 30s. Each input contains 12 frames of six node features: queue-to-storage ratio, vehicle-to-storage ratio, normalized speed, occupancy, normalized waiting time, and spillback ratio.

## Training

Compute the delay normalization scale, collect encoder trajectories, and run training:

```bash
python scripts/calibrate_normalization.py --manifest data/network.json --scenarios data/scenarios.json --output data/network_normalized.json
python scripts/collect_encoder.py --manifest data/network_normalized.json --scenarios data/scenarios.json --config configs/default.json --output data/encoder
python scripts/train.py --manifest data/network_normalized.json --scenarios data/scenarios.json --config configs/default.json --output outputs/seed17 --seed 17 --device cuda:0
```

Set `training.encoder_data` in the training configuration to the collected `index.json`; this path is resolved relative to the configuration file. Encoder data covers normal, fault-event, nominal-control, bounded-random-control, and MaxPressure trajectories.

| Stage | Updates | Batch size | Learning rate | Checkpoint selection |
|---|---:|---:|---:|---|
| Encoder initialization | 100,000 | 256 | 0.0003 | Validation auxiliary loss |
| Planner-Executor cold start | 50,000 | 256 | 0.0001 | Validation recovery delay |
| Joint online training | 200,000 | 512 | 0.0003 | Validation recovery delay |

Training writes `encoder_best.pt`, `cold_best.pt`, `online_best.pt`, `final.pt`, and `training.json`. Checkpoints store model configuration, weights, parameter count, training seed, and stage metadata. The default model has 3,380,667 parameters and uses the `cape-v3` checkpoint format.

## Inference

Run one scenario supplied as a JSON object:

```bash
python scripts/infer.py --manifest data/network_normalized.json --scenario data/scenario.json --checkpoint outputs/seed17/final.pt --output outputs/trajectory.json --device cuda:0
```

The trajectory output contains control sets, edit traces, calibration events, rewards, timing, and probe counts for each planning period.

## Evaluation

Evaluate a checkpoint on the test split:

```bash
python scripts/evaluate.py --manifest data/network_normalized.json --scenarios data/scenarios.json --checkpoint outputs/seed17/final.pt --output results/seed17.json --seed 10017 --run-id 0 --device cuda:0
```

Use `--split validation` to select the validation split. Evaluation restores the checkpoint weights before each episode and records Recovery Gain, T80, period diagnostics, and GPU memory statistics.

## Configuration

[configs/default.json](configs/default.json) defines the model and training settings. Default planning settings are 64 candidate pairs, 32 active nodes, eight edits, a 128-dimensional strategy code, and a disagreement threshold of 0.63.

Signal actions specify cycle length, offset, and green-phase allocation. Set `model.action_dim` to at least two plus the number of green phases; the default is eight. Safety projection produces legal phase plans for all three waves. Metering uses release rates from 0.4 to 1.0, and diversion uses alternative-route ratios from 0 to 0.3.

Calibration uses four matched probe pairs with four parallel SUMO branches and reference caching. The four records support five code updates at learning rate 0.05, followed by generator distillation. The independent V head estimates the policy expectation of the minimum twin-Q value.

| Configuration | Purpose |
|---|---|
| `signal_only.json` | Train with signal controls |
| `signal_only_no_calibration.json` | Train with signal controls and calibration disabled |
| `full_no_calibration.json` | Train with all control families and calibration disabled |
| `consistency_half.json`, `consistency_double.json` | Train with scaled consistency weights |
| `effect_half.json`, `effect_double.json` | Train with scaled effect weights |
| `distill_half.json`, `distill_double.json` | Train with scaled distillation weights |
| `budget_45s.json`, `budget_90s.json` | Set inference resources through `--profile` |

Run evaluation with an inference budget:

```bash
python scripts/evaluate.py --manifest data/network_normalized.json --scenarios data/scenarios.json --checkpoint outputs/seed17/final.pt --profile configs/budget_45s.json --budget 45 --output results/b45.json
python scripts/evaluate.py --manifest data/network_normalized.json --scenarios data/scenarios.json --checkpoint outputs/seed17/final.pt --profile configs/budget_90s.json --budget 90 --output results/b90.json
```

## Baselines

```bash
python scripts/benchmark.py --method fixed --manifest data/network_normalized.json --scenarios data/scenarios.json --seed 10017 --run-id 0 --output results/fixed.json
python scripts/benchmark.py --method cem --manifest data/network_normalized.json --scenarios data/scenarios.json --seed 10017 --run-id 0 --sampling-seed 17 --budget 45 --samples 32 --iterations 3 --horizon 3 --output results/cem.json
```

`--seed` sets the starting environment seed, and `--run-id` identifies a repeated run. CEM-MPC uses `--sampling-seed` for action-sequence sampling.

## Result Summaries

```bash
python scripts/summarize.py --inputs results/seed17.json results/seed18.json --output results/summary.json
python scripts/summarize.py --inputs results/fixed.json results/seed17.json --reference fixed --candidate CAPE --output results/paired_summary.json
```

Paired summaries match `network`, `fault_type`, `scenario`, `environment_seed`, and `run_id`. Set the same `--seed`, `--run-id`, and scenario ordering for the methods in a paired run. Learning methods also record `training_seed`; CEM-MPC records `sampling_seed`.

Outputs include the following timing and memory fields:

| Field | Description |
|---|---|
| `policy_inference_seconds` | Encoding, planning, action generation, and safety projection |
| `end_to_end_seconds` | Complete decision time, including matched probes, code updates, distillation, and safety projection, ending before the main traffic rollout |
| `peak_allocated_bytes` | Peak GPU memory allocated by the PyTorch allocator |
| `peak_reserved_bytes` | Peak GPU memory reserved by the PyTorch allocator |

Summaries report mean, P95, and maximum latency, the fraction of decisions exceeding 180s, and additional SUMO calls per episode.
