# CAPE

CAPE: Closed-Loop Adaptive Planning and Execution for Multi-Node Traffic Fault Recovery.

## 安装

运行环境：双路 Xeon Gold 6338、256 GB RAM、RTX 4060 8GB；Python 3.10+、支持 CUDA 的 PyTorch、SUMO 1.20.0 及对应版本的 TraCI。确保sumo在PATH中。

训练、推理和评估默认使用 `cuda:0`。模型及配对分支的策略计算在 GPU 上执行，SUMO 在 CPU 上运行；配对分支最多四路并行，并缓存匹配的对照结果。

```powershell
python -m pip install -e .
```

## 目录

```text
cape/
  data/          预训练轨迹格式与加载
  models/        时序/图编码、集合编码、Planner、Executor、双Q及独立V head
  planning/      集合编辑、校准、门控、FixedTime和CEM-MPC
  learning/      SAC、联合目标、回放与三阶段训练
  runtime/       SUMO闭环、相位约束、入口计量、分流和配对分支
  evaluation/    RG、T80、配对bootstrap和BH校正
configs/         训练、消融、损失权重及预算配置
scripts/         数据采集、训练、推理、评估与汇总入口
```

## 数据准备

提供网络manifest、SUMO网络/路线文件，以及含train、validation、test划分的场景JSON。网络字段见`configs/manifest.schema.json`。文件路径相对于manifest解析。

manifest含`name`、`sumocfg`、`delay_scale`、`nodes`和`adjacency`。每个节点有`id`、入车道`lanes`，按实际设施设置`signal`、`meter`和`diversion`。不同节点的signal/meter不能共用同一个信号控制器。`adjacency[目标,来源]>0`表示有向边。

场景项包含`id`、`fault_type`和`events`。事件支持lane_speed、lane_permission、signal_program、signal_state、demand；多个事件组成复合故障。故障后不可接管的节点—措施对写入`unavailable_pairs`。所有方法使用同一场景清单与随机种子。

默认时间协议：正常0–900秒，故障900–1800秒，恢复1800–3600秒。每180秒规划一次，每周期三个60秒动作波次；15秒采样观测，30秒计算恢复指标。输入为12帧、每节点6维特征。六维特征分别为排队存储比、车辆存储比、归一化速度、占有率、归一化等待和溢出比例。

## 训练

先用正常训练区间计算延迟归一化尺度，再采集五类编码器轨迹：

```powershell
python scripts/calibrate_normalization.py --manifest data/network.json --scenarios data/scenarios.json --output data/network_normalized.json
python scripts/collect_encoder.py --manifest data/network_normalized.json --scenarios data/scenarios.json --config configs/default.json --output data/encoder
python scripts/train.py --manifest data/network_normalized.json --scenarios data/scenarios.json --config configs/default.json --output outputs/seed17 --seed 17 --device cuda:0
```

`configs/default.json`中的training.encoder_data相对配置文件定位。使用其他 GPU 时通过 `--device cuda:1` 等参数指定设备。

三个阶段为：

| 阶段 | 更新次数 | 批量 | 学习率 | 选模指标 |
|---|---:|---:|---:|---|
| 编码器初始化 | 100000 | 256 | 0.0003 | 验证辅助损失 |
| Planner–Executor冷启动 | 50000 | 256 | 0.0001 | 验证恢复延迟 |
| 联合在线训练 | 200000 | 512 | 0.0003 | 验证恢复延迟 |

冷启动按合法编辑类型及低访问节点—措施进行平衡采样；在线阶段15%的episodes使用覆盖采样。内部编辑奖励为0，Stop提交周期奖励；达到编辑上限增加提交转移。Stop不是恢复episode终止。编码辅助目标包括掩码重建及1/3/5周期预测。

输出包括encoder_best.pt、cold_best.pt、online_best.pt、final.pt和training.json。checkpoint保存结构配置、模型参数、训练种子及训练阶段元信息，推理严格加载对应参数。

## 推理与评估

对单个场景对象运行闭环推理：

```powershell
python scripts/infer.py --manifest data/network_normalized.json --scenario data/scenario.json --checkpoint outputs/seed17/final.pt --output outputs/trajectory.json --device cuda:0
```

对测试清单运行评估：

```powershell
python scripts/evaluate.py --manifest data/network_normalized.json --scenarios data/scenarios.json --checkpoint outputs/seed17/final.pt --output results/seed17.json --seed 10017
```

每个episode从同一checkpoint重新加载。推理和评估需要提供训练生成的checkpoint。输出包含逐周期控制集合、候选评分、校准事件、耗时、调用数、奖励，以及正常与故障条件下的延迟和RG/T80。

## 动作与校准

信号动作包含周期、偏移目标和各绿灯阶段的时间分配。action_dim默认8，根据绿灯阶段数量在训练前设置足够维度，其余坐标使用mask。安全执行保留合法相序、最短绿灯、黄灯、全红和周期范围。入口计量用独立三相信号调节0.4–1.0放行强度；分流用0–0.3概率选择同目的地的合法替代路线。

默认64候选、32个不同控制节点、8次编辑、128维策略码；分歧阈值0.63。四对匹配分支运行完整剩余时域，冻结模型并共享随机条件，形成四条匹配记录；在这四条记录上进行5次局部策略码更新，学习率0.05，随后蒸馏。校准后使用min(A_P,A_E_cal)修正候选评分，修正后的优势非正时，否决该编辑并重新选择。每周期最多触发一次。

## 消融与预算

signal_only.json、signal_only_no_calibration.json、full_no_calibration.json为独立训练配置。六个consistency/effect/distill配置对应损失权重减半/加倍，改变训练权重时应使用相应配置训练。

```powershell
python scripts/evaluate.py --manifest data/network_normalized.json --scenarios data/scenarios.json --checkpoint outputs/seed17/final.pt --profile configs/budget_45s.json --budget 45 --output results/b45.json
python scripts/evaluate.py --manifest data/network_normalized.json --scenarios data/scenarios.json --checkpoint outputs/seed17/final.pt --profile configs/budget_90s.json --budget 90 --output results/b90.json
python scripts/benchmark.py --method cem --manifest data/network_normalized.json --scenarios data/scenarios.json --budget 45 --samples 32 --iterations 3 --horizon 3 --output results/cem.json
```

CEM样本、迭代和时域应在验证集选择。预算超时后保留未校准候选评价或已完成的CEM最优序列；记录实际最终耗时，进程回收和动作生成的时间也计入。90秒档为128候选、最多6对探测、10步更新。

## 汇总

```powershell
python scripts/summarize.py --inputs results/seed17.json results/seed18.json --output results/summary.json
python scripts/summarize.py --inputs results/fixed.json results/seed17.json --reference fixed --candidate CAPE --output results/paired_summary.json
```

配对统计按网络、故障类型、场景、`environment_seed` 和 `run_id` 匹配。评估和基线命令通过 `--seed` 设置环境种子起点，通过 `--run-id` 标识运行重复；每次重复使用不同的 `run_id`，各方法使用相同场景顺序和环境种子。学习方法另记录 `training_seed`，CEM-MPC 通过 `--sampling-seed` 指定并记录采样种子，FixedTime 不设置训练种子。RG使用独立正常与故障条件下的延迟，T80按各方法自身最佳恢复的80%定义。仅将相同场景协议的结果汇总比较。

## 模型与运行统计

默认结构共 3,380,667 个参数，checkpoint 元信息同时保存实际参数量。独立 V head 学习策略下双 Q 最小值的期望；校准期间冻结 V head 参数，保留对策略码的梯度。包含 V head 的 checkpoint 格式为 `cape-v3`，训练与推理需使用同一结构的权重。

每个周期分别输出 `policy_inference_seconds` 和 `end_to_end_seconds`。前者包含编码、规划、动作生成和安全投影，排除校准时间；后者额外包含配对分支、局部更新和蒸馏。两者均不包含随后主环境的交通运行。安全投影预先生成三个动作波次的合法相位计划，由环境在相应波次执行。汇总分别报告两种延迟的均值、P95、最大值及超过180秒的比例。计时包含首次调用，比较预热后的延迟时应按相同规则选择运行记录。

推理和评估输出所选单卡的 PyTorch 分配器峰值，包括 `peak_allocated_bytes` 和 `peak_reserved_bytes`。该统计覆盖本进程，不含其他进程及驱动开销。当前使用单 GPU，不进行多卡调度，也不输出多卡汇总峰值。多卡实验若报告 Peak aggregate GPU memory，需记录卡数，并取同时刻各卡占用之和的峰值。
