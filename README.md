# SuperMimic TITATIT：速度跟踪、复杂地形与深度感知

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Built on mjlab](https://img.shields.io/badge/built%20on-mjlab-orange.svg)](https://github.com/mujocolab/mjlab)

本项目将 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic) 适配到 **TITATIT 四轮足机器人**，在 mjlab / MuJoCo Warp 中训练速度跟踪、复杂地形行走，以及追球、跳跃越杆、下蹲和躲避动作。新增参考 WMP 的深度编码器与时序记忆，继续使用 SuperMimic 训练策略。

机器人采用 **12 个腿关节位置控制 + 4 个轮关节速度控制**。训练沿用 SuperMimic 的世界模型与可微想象轨迹优化方法：采集仿真数据、学习动力学，再通过世界模型中的想象轨迹更新策略。当前为仿真研究项目，尚未验证实机部署。

原项目由 Ziyue Zheng（Johns Hopkins University）开发；本仓库由 harryma03 维护 TITATIT 模型及四轮足适配。Python 包名保留为 `go2_supermimic`。

## 六种 TITATIT 任务

| 任务 | 场景 | 额外观测 | 策略输入维度 | 动作维度 |
| --- | --- | --- | ---: | ---: |
| `Titatit-Chase-Flat` | 平地追逐与躲避 | 无 | 410 | 16 |
| `Titatit-Chase-Obstacles` | 箱体、柱体、墙体等静态障碍 | 16 射线二维 LiDAR | 490 | 16 |
| `Titatit-Chase-Terrain` | 台阶、方块、粗糙地面和斜坡 | 77 点地形高度扫描 | 795 | 16 |

| `Titatit-Velocity-Flat` | 平地速度跟踪 | 本体与速度指令 | 285 | 16 |
| `Titatit-Velocity-Terrain` | 九类混合地形，纵向指令 ±1.5 m/s | 77 点真值高度扫描 | 670 | 16 |
| `Titatit-Velocity-Depth` | 九类混合地形，纵向指令 ±1.5 m/s | 32×32 深度 + RSSM 时序记忆 | 445（默认） | 16 |

Depth 的环境观测为 285 维，训练器拼接 160 维时序特征后输入 Actor。**Depth 使用 `scripts/train_depth.py`；其余任务使用 `scripts/train.py`。**

详细说明：[追逐与躲避](docs/titatit.md) · [速度跟踪与地形迁移](docs/titatit_velocity.md) · [WMP 感知 + SuperMimic](docs/titatit_depth.md)。

- **追逐**：追踪绿色目标球，机器人头部接近目标后触发 tag，目标随后逃离。
- **躲避**：应对不同高度的球和横杆；低杆需要跳跃，高杆需要下蹲，也可以通过转向或侧移避让。
- 平地和静态障碍任务按 30% / 30% / 40% 采样追逐、躲避、两者同时进行；地形任务按 60% / 40% 采样追逐、追逐加躲避。

## 安装

需要 Python 3.10+ 和支持 CUDA 的 NVIDIA GPU。使用已经配置好的 `mjlab` Conda 环境时：

```bash
conda activate mjlab
```

```bash
git clone https://github.com/harryma03/supermimic_titatit.git
```

```bash
cd supermimic_titatit
```

安装项目和测试依赖：

```bash
python -m pip install -e '.[dev]' onnx
```

```bash
python -m pytest tests -q
```

`onnx` 用于 Actor 导出及相关测试。项目依赖见 [pyproject.toml](pyproject.toml)。已有本地目录时，直接进入项目目录安装即可。

## 训练

以下命令指定物理 **GPU1**。`CUDA_VISIBLE_DEVICES=1` 会将它映射为程序内的 `cuda:0`，因此日志显示 `cuda:0` 是正常的。

### 速度跟踪与 1.5 m/s 混合地形

平地速度跟踪从头训练（最终纵向范围 ±1.0 m/s）：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Velocity-Flat --num-envs 1024 --compile false --iterations 40000 --run-name titatit_velocity_flat
```

从本项目平地速度模型迁移到复杂地形；将 `FLAT_RUN` 替换为实际目录：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Velocity-Terrain --num-envs 1024 --compile false --resume logs/supermimic/FLAT_RUN/model_40000.pt --velocity-transfer true --reset-critic true --continue-run false --wm-rewarmup-iterations 1500 --critic-warmup-iterations 500 --actor-lr-scale 0.25 --actor-ramp-iterations 2000 --terrain-curriculum-iterations 20000 --iterations 40000 --run-name titatit_velocity_terrain_1p5
```

迁移复用 Actor 与动力学模型，新增扫描输入权重置零，重新训练 Critic；课程从第 0 轮开始。只接受本项目 Flat 速度模型，不能加载 Chase 模型。从头训练时删除 `--resume` 及其路径、`--velocity-transfer true`。

地形入口为 [terrains.py](go2_supermimic/env/terrains.py)，速度配置为 [velocity_env_cfg.py](go2_supermimic/env/velocity_env_cfg.py)：

| 地形 | 当前范围 |
| --- | --- |
| 平地 | 基础速度跟踪 |
| 上台阶、下台阶 | 阶高 2–10 cm，踏面 40 cm |
| 随机方块 | 高度参数 2–5 cm |
| 随机起伏 | 高度参数 1–4 cm |
| 正向、反向坡面 | 坡度参数 0–0.3 |
| 沟壑 gap | 沟宽 10–40 cm，深 60 cm |
| 高台 climb | 高 5–25 cm，台面宽 80 cm |

地图为 5 行难度 × 9 类地形，每块 12×12 m。上述命令从第 0 行开始，第 16000 轮开放全部难度，重置时仍保留简单地形。纵向指令上限从 0.375、0.75 逐步提高至 1.5 m/s，第 10000 轮进入最终范围；横向最终 ±0.4 m/s，转向最终 ±1 rad/s。“全地形”指这九类训练地形；1.5 m/s 是指令范围，实际通过能力需要评估。

### WMP 感知 + SuperMimic 训练

Depth 使用深度编码器与 RSSM 时序记忆，策略通过 SuperMimic 可微想象轨迹更新。这是参考 [WMP](https://github.com/harryma03/wmp) 的独立紧凑实现，不兼容 WMP/PPO checkpoint。

从本项目平地速度模型迁移：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_depth.py --resume logs/supermimic/FLAT_RUN/model_40000.pt --depth-transfer true --reset-critic true --continue-run false --iterations 40000 --run-name titatit_depth_supermimic_1p5
```

默认 256 个环境，前 1500 轮预热动力学与感知模型，随后预热 Critic 500 轮，再逐步更新 Actor；使用上述九类地形及 ±1.5 m/s 指令课程。从头训练时删除 `--resume` 及其路径、`--depth-transfer true`。

相机为 32×32 光轴深度、70° 视场角、最远 3 m，随完整机身姿态旋转。控制频率 50 Hz，每 5 步融合新图像（10 Hz），重置时立即融合。Actor 运行时使用本体、指令和时序特征；真值地形用于训练辅助监督、动力学模型和 Critic。结构与迁移限制见 [深度训练说明](docs/titatit_depth.md)。

### 追逐与躲避

平地任务：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Chase-Flat --run-name titatit_flat_gpu1
```

静态障碍任务：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Chase-Obstacles --run-name titatit_obstacles_gpu1
```

地形任务：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Chase-Terrain --run-name titatit_terrain_gpu1
```

默认训练 40000 轮。完整参数见 [TrainCfg](go2_supermimic/supermimic/train.py) 和 [DepthTrainCfg](go2_supermimic/supermimic/depth_train.py)，每个配置字段都可以通过命令行设置。

输出保存在 `logs/supermimic/<时间戳>_<run-name>/`，包含 TensorBoard 日志和 `model_<轮数>.pt`。训练日志、视频和新训练的 checkpoint 默认不提交到 Git。

### 从 checkpoint 继续训练

将 `YOUR_RUN` 替换为实际训练目录。下面示例从 7000 轮继续到总计 40000 轮：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Chase-Flat --resume logs/supermimic/YOUR_RUN/model_7000.pt --continue-run true --reset-critic false --iterations 40000 --run-name titatit_flat_resume7000_gpu1
```

`--continue-run true` 保留迭代计数，并接续课程、引导奖励衰减和学习率计划；同结构 checkpoint 会恢复 actor、critic、世界模型及优化器。仿真现场、经验缓存和 archive 会重新建立，因此不是逐步完全一致的运行恢复。续训日志写入新目录，原 checkpoint 保留。

Terrain 同任务续训需使用 `--task Titatit-Velocity-Terrain`，移除迁移标记，并保持原来的地形课程与网络配置；完整命令见 [速度训练说明](docs/titatit_velocity.md)。Depth 同任务续训使用独立入口：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_depth.py --resume logs/supermimic/DEPTH_RUN/model_10000.pt --continue-run true --reset-critic false --iterations 40000 --run-name titatit_depth_resume
```

Depth 恢复两个模型、Actor、Critic 及优化器；Replay 和在线隐状态重新建立。若修改过网络或感知配置，续训需传入相同参数。

### 查看 TensorBoard

```bash
tensorboard --logdir logs/supermimic
```

Chase 的 `hit_rate` 按 **球 / 低杆 / 高杆** 排列，表示被击中率，越低越好。`imagined_return` 和 `imagined_survival` 是世界模型内的预测指标，实际动作效果应结合评估指标和视频判断。

## 评估与录制

使用自己训练的 TITATIT checkpoint；将 `YOUR_RUN` 和轮数替换为实际路径。

### 速度跟踪与深度感知

速度评估器按 checkpoint 自动识别 Flat / Terrain。下面检查最高难度的 1.5 m/s 指令：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_velocity.py --checkpoint logs/supermimic/TERRAIN_RUN/model_40000.pt --terrain-level 4 --vx 1.5 --num-envs 9 --seconds 30 --output terrain_level4_vx1p5.json
```

Depth 使用专用评估器，先从低难度、低速度开始：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_depth.py --checkpoint logs/supermimic/DEPTH_RUN/model_40000.pt --device cuda:0 --num-envs 9 --seconds 30 --terrain-level 0 --vx 0.5 --output depth_real.json
```

追加 `--mask-depth` 并更换输出文件可做遮挡深度对照。Depth 输出每类地形的速度 RMSE、摔倒数和有效样本数。逐一测试 `--terrain-level 0` 至 `4`，再将 `--vx` 提高至 `1.5`，并检查倒车、转向和停止。Depth 暂无视频录制及 ONNX 导出入口；`eval_velocity.py` 和 `export_actor.py` 不接受 Depth checkpoint。

### 追逐与躲避

平地追逐与躲避视频：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval.py Titatit-Chase-Flat --checkpoint logs/supermimic/YOUR_RUN/model_7000.pt --mode both --num-envs 1 --seconds 60 --video titatit_flat.mp4
```

多环境统计评估：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval.py Titatit-Chase-Flat --checkpoint logs/supermimic/YOUR_RUN/model_7000.pt --num-envs 256 --seconds 60
```

障碍和地形任务需使用对应任务的 checkpoint，并分别将任务名改为 `Titatit-Chase-Obstacles`、`Titatit-Chase-Terrain`。地形录制可加 `--terrain stairs`。

- `--mode chase|dodge|both|mixed`：覆盖任务默认模式。
- `--difficulty`：评估课程难度，默认 `1.0`（最高难度）。比较不同 checkpoint 时保持相同难度、模式和时长。
- 评估输出包括追逐次数、各类抛掷物命中率、摔倒次数和移动速度。

仓库中的 `weights/flat.pt`、`weights/obstacles.pt`、`weights/terrain.pt` 是原项目的 **Go2 策略**，不能用于 TITATIT。当前仓库未发布 TITATIT 预训练权重。

## 四轮足控制与观测

模型入口为 [titatit.py](go2_supermimic/robot/titatit.py)，模型和网格位于 [robot/titatit](go2_supermimic/robot/titatit)。

| 项目 | 当前配置 |
| --- | --- |
| 腿控制 | 位置 PD，Kp = 60，Kd = 1.5 |
| 腿动作比例 | hip = 0.125，thigh / calf = 0.25 |
| 轮控制 | 目标轮速 = action × 5 rad/s |
| 轮力矩 | 0.5 ×（目标轮速 − 实际轮速），限幅 ±10 Nm |
| 腿力矩限制 | ±55 Nm |
| 策略周期 | 0.02 s（50 Hz） |
| 物理步长 | 0.005 s，decimation = 4 |

动作排列为 12 个腿关节，再 4 个轮关节；腿和轮均按 FR、FL、RR、RL 顺序排列，每条腿依次为 hip、thigh、calf。

Chase 策略使用 5 帧历史，包括机身角速度、投影重力、关节位置偏差、关节速度、上一动作和目标观测；障碍/地形任务额外加入对应扫描。轮子的累计转角在位置观测和站姿惩罚中屏蔽，轮速仍保留。

Chase 沿用原追逐/躲避奖励，按 TITATIT 尺寸调整碰撞近似和轮半径，以 `stand_height=0.37 m` 作为跳跃/下蹲引导参考。完整控制参数和奖励诊断说明见 [TITATIT 说明](docs/titatit.md)。

### 感知范围

目标球和抛掷物观测目前来自仿真真值。`Titatit-Chase-Obstacles` 的 LiDAR 是以 base 的 XY 为原点、随 yaw 旋转的二维几何扫描，16 射线覆盖 360°、最大距离 4 m；未建模实体雷达的安装高度和俯仰角，也不扫描机器人本体或抛掷物。实机感知和雷达外参需要另行适配。

Velocity 使用速度指令替代目标观测；Flat 为 285 维本体/指令历史，Terrain 每帧增加 77 点真值扫描，Depth 通过相机与时序记忆提供感知。历史排列和部署输入约定见 [速度跟踪说明](docs/titatit_velocity.md)。


### 新增 gap / climb（2026-10-30）

Velocity-Terrain 和 Velocity-Depth 现在默认使用九类地形；Chase 保留原七类地图。新增两条横贯地块的沟壑或高台，中央保留 1.5 m 平地出生区，正反向均可接近障碍。

- `gap`：沟宽随难度由 10 cm 增至 40 cm，沟底深 60 cm；使用真实碰撞几何和有限沟底，深度射线与高度扫描均能读取。它不是 WMP 的无底窄桥地图。
- `climb`：高台高度由 5 cm 增至 25 cm，台面沿前进方向宽 80 cm，区别于连续台阶。
- 参数见 `go2_supermimic/env/terrains.py` 的 `GapClimbTerrainCfg`。最终几何随地形生成器的难度采样变化。
- 正在运行的旧训练继续使用内存中的旧地图。旧七类 Terrain / Depth checkpoint 与新地图合同不同，不能直接续训或用新配置评估；旧模型评估需保留原配置/代码。新地图可使用上面的平地模型迁移命令，建议换新 run-name（例如 `titatit_depth_gap_climb_1p5`）。不要绕过合同检查。
- 九类地形评估使用 `--num-envs 9`。沟底落地不能仅凭“未倾倒”算作成功；还需检查是否跨过沟壑、登上并离开高台。当前奖励和倾倒判定沿用原任务，尚未加入专门的越沟成功率统计。

## 验证状态（2026-10-30）

已完成 CPU MuJoCo Warp 地形/深度采集、平地模型迁移、动力学与感知模型更新、Actor 可微想象更新、Critic 更新和 checkpoint 保存/加载验证。默认网络与完整 5×7 地图也完成短程训练链路检查，损失与梯度有限。

上述验证证明训练链路可运行。新增 Terrain / Depth 尚未完成 GPU 长训与 1.5 m/s 复杂地形通过率验证；深度感知收益需结合仿真评估和遮挡对照判断。详细记录见 [地形说明](docs/titatit_velocity.md) 和 [深度说明](docs/titatit_depth.md)。

## 代码目录

```text
go2_supermimic/
  robot/titatit.py          TITATIT 控制、执行器和模型加载
  robot/titatit/            URDF、MJCF 和模型网格
  env/titatit_env_cfg.py    三个 TITATIT Chase 任务配置
  env/velocity_env_cfg.py   Flat / Terrain 速度跟踪配置
  env/depth_env_cfg.py      深度相机与 Depth 任务配置
  env/terrains.py           混合地形及难度范围
  env/command.py           目标生成、命中判定与课程
  env/mdp.py               观测、奖励和诊断指标
  target/                  目标动力学、碰撞近似、LiDAR 和高度扫描
  supermimic/              世界模型、经验缓存、想象轨迹和训练器
scripts/                   train.py / train_depth.py；eval.py / eval_velocity.py / eval_depth.py
tests/                    任务、模型、奖励和学习链路测试
docs/titatit.md            追逐与躲避说明
docs/titatit_velocity.md   速度跟踪、地形迁移与部署输入
docs/titatit_depth.md      深度感知与独立训练入口
weights/                  原 Go2 策略，仅供 Go2 任务使用
```

## 提交文档修改

仅提交 README 时，先检查变更，再推送当前分支：

```bash
git status --short
git diff -- README.md
git add README.md
git commit -m "Update README"
git push -u origin HEAD
```

其他代码与文件删除需另行检查后按需加入提交。

## 来源与许可证

本项目基于 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic)，保留原代码署名和 [MIT 许可证](LICENSE)。训练方法来自 [SuperMimic 论文](https://arxiv.org/abs/2610.28480)，仿真基于 [mjlab](https://github.com/mujocolab/mjlab) 和 MuJoCo Warp。

TITATIT 控制配置参考 [titatit_dreamwaq](https://github.com/harryma03/titatit_dreamwaq)。仓库中保留的 Go2 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，遵循其 [模型许可证](go2_supermimic/robot/LICENSE_menagerie_go2)。
