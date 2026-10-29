# SuperMimic TITATIT：四轮足追逐与躲避

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Built on mjlab](https://img.shields.io/badge/built%20on-mjlab-orange.svg)](https://github.com/mujocolab/mjlab)

本项目将 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic) 适配到 **TITATIT 四轮足机器人**，在 mjlab / MuJoCo Warp 中训练追球、跳跃越杆、下蹲和躲避动作。

机器人采用 **12 个腿关节位置控制 + 4 个轮关节速度控制**。训练沿用 SuperMimic 的世界模型与可微想象轨迹优化方法：采集仿真数据、学习动力学，再通过世界模型中的想象轨迹更新策略。当前为仿真研究项目，尚未验证实机部署。

原项目由 Ziyue Zheng（Johns Hopkins University）开发；本仓库由 harryma03 维护 TITATIT 模型及四轮足适配。Python 包名保留为 `go2_supermimic`。

## 三种任务

| 任务 | 场景 | 额外观测 | 策略输入维度 | 动作维度 |
| --- | --- | --- | ---: | ---: |
| `Titatit-Chase-Flat` | 平地追逐与躲避 | 无 | 410 | 16 |
| `Titatit-Chase-Obstacles` | 箱体、柱体、墙体等静态障碍 | 16 射线二维 LiDAR | 490 | 16 |
| `Titatit-Chase-Terrain` | 台阶、方块、粗糙地面和斜坡 | 77 点地形高度扫描 | 795 | 16 |

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
python -m pip install -e '.[dev]'
```

```bash
python -m pytest tests -q
```

项目依赖见 [pyproject.toml](pyproject.toml)。已有本地目录时，直接进入项目目录安装即可。

## 训练

以下命令指定物理 **GPU1**。`CUDA_VISIBLE_DEVICES=1` 会将它映射为程序内的 `cuda:0`，因此日志显示 `cuda:0` 是正常的。

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

默认训练 40000 轮。完整参数见 [TrainCfg](go2_supermimic/supermimic/train.py)，每个配置字段都可以通过命令行设置。

输出保存在 `logs/supermimic/<时间戳>_<run-name>/`，包含 TensorBoard 日志和 `model_<轮数>.pt`。训练日志、视频和新训练的 checkpoint 默认不提交到 Git。

### 从 checkpoint 继续训练

将 `YOUR_RUN` 替换为实际训练目录。下面示例从 7000 轮继续到总计 40000 轮：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Chase-Flat --resume logs/supermimic/YOUR_RUN/model_7000.pt --continue-run true --reset-critic false --iterations 40000 --run-name titatit_flat_resume7000_gpu1
```

`--continue-run true` 保留迭代计数，并接续课程、引导奖励衰减和学习率计划；同结构 checkpoint 会恢复 actor、critic、世界模型及优化器。仿真现场、经验缓存和 archive 会重新建立，因此不是逐步完全一致的运行恢复。续训日志写入新目录，原 checkpoint 保留。

### 查看 TensorBoard

```bash
tensorboard --logdir logs/supermimic
```

`hit_rate` 按 **球 / 低杆 / 高杆** 排列，表示被击中率，越低越好。`imagined_return` 和 `imagined_survival` 是世界模型内的预测指标，实际动作效果应结合评估指标和视频判断。

## 评估与录制

使用自己训练的 TITATIT checkpoint；将 `YOUR_RUN` 和轮数替换为实际路径。

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

策略使用 5 帧历史，包括机身角速度、投影重力、关节位置偏差、关节速度、上一动作和目标观测；障碍/地形任务额外加入对应扫描。轮子的累计转角在位置观测和站姿惩罚中屏蔽，轮速仍保留。

当前沿用原追逐/躲避奖励，按 TITATIT 尺寸调整碰撞近似和轮半径，以 `stand_height=0.37 m` 作为跳跃/下蹲引导参考。完整控制参数和奖励诊断说明见 [TITATIT 说明](docs/titatit.md)。

### 感知范围

目标球和抛掷物观测目前来自仿真真值。任务 2 的 LiDAR 是以 base 的 XY 为原点、随 yaw 旋转的二维几何扫描，16 射线覆盖 360°、最大距离 4 m；未建模实体雷达的安装高度和俯仰角，也不扫描机器人本体或抛掷物。实机感知和雷达外参需要另行适配。

## 代码目录

```text
go2_supermimic/
  robot/titatit.py          TITATIT 控制、执行器和模型加载
  robot/titatit/            URDF、MJCF 和模型网格
  env/titatit_env_cfg.py    三个 TITATIT 任务配置
  env/command.py           目标生成、命中判定与课程
  env/mdp.py               观测、奖励和诊断指标
  target/                  目标动力学、碰撞近似、LiDAR 和高度扫描
  supermimic/              世界模型、经验缓存、想象轨迹和训练器
scripts/                   train.py、eval.py
tests/                    任务、模型、奖励和学习链路测试
docs/titatit.md            TITATIT 详细说明
weights/                  原 Go2 策略，仅供 Go2 任务使用
```

## 更新代码
git status
git add .
git commit -m "Update titatit_vel"
git push origin main

## 来源与许可证

本项目基于 [Go2-SuperMimic](https://github.com/Zhengtianjin/go2-supermimic)，保留原代码署名和 [MIT 许可证](LICENSE)。训练方法来自 [SuperMimic 论文](https://arxiv.org/abs/2610.28480)，仿真基于 [mjlab](https://github.com/mujocolab/mjlab) 和 MuJoCo Warp。

TITATIT 控制配置参考 [titatit_dreamwaq](https://github.com/harryma03/titatit_dreamwaq)。仓库中保留的 Go2 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，遵循其 [模型许可证](go2_supermimic/robot/LICENSE_menagerie_go2)。
