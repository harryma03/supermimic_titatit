# TITATIT：WMP 感知 + SuperMimic 训练

任务为 `Titatit-Velocity-Depth`，训练入口为 `scripts/train_depth.py`。机器人、MuJoCo/mjlab、16 维轮腿控制和九类混合地形沿用当前项目，最终纵向指令范围 ±1.5 m/s。

这是参考 [WMP 世界模型](https://github.com/harryma03/wmp/blob/main/dreamer/models.py) 的独立紧凑实现：深度编码器 + RSSM 时序感知，策略用 SuperMimic 动力学想象/BPTT 更新。不是原论文的完整复现，也不兼容 WMP/PPO checkpoint。未引入 PPO 或 AMP 数据集。

## 数据与模型

| 部分 | 当前实现 |
| --- | --- |
| 深度 | 32×32 单通道；视场角 70°；距离上限 3 m；安装在 base 的 `(0.45,0,0.08)`，向前下俯 20° |
| 射线 | 与完整机身姿态一起旋转；查询真实 MuJoCo 几何 group 0；当前地图只含静态地形，排除机器人自身 |
| 图像语义 | 相机光轴深度，不是射线长度；无命中填远平面，范围 `[0.05,3] m` 后除以 3 |
| 感知更新 | 控制 50 Hz；每 5 步融合新深度，即 10 Hz；reset 立即融合新帧；引擎仍逐步计算射线，并非减少引擎 sense 次数 |
| RSSM | GRU 确定状态 128 + 高斯随机状态 32；prior 每控制步更新，posterior 只在新图像/reset 时更新；训练采样，采集/评估/想象使用均值 |
| Actor | 285 维 term-major 本体/指令历史 + 160 维时序特征 → 16 维动作；默认输入 445 |
| Critic | 物理状态、指令、上一动作、物理参数、真值地形扫描和时序特征；默认输入 383 |
| 动力学 WM | 原 SuperMimic 236→126，接收真值地形扫描，预测物理状态与平面位姿增量 |
| 感知损失 | 深度重建、本体观测重建、局部地形扫描辅助监督、posterior/prior KL；奖励沿用可微物理奖励，无另一个奖励预测头 |

Actor 和 RSSM 的运行输入不包含地形真值、世界位置、真值速度或接触状态。地形真值仅用于训练感知的辅助监督、动力学 WM、奖励和 Critic。想象阶段 RSSM 根据预测动作推进 prior，不查询未来相机图像，也不将真值扫描输入 RSSM。每一步通过两个冻结模型向 Actor 反传梯度；模型参数只在各自监督训练阶段更新。

感知 Replay 保存 uint8 深度、fresh/first 标记，动作、本体历史与物理状态复用现有 Replay。默认先用 16 个控制步（0.32 s）burn-in，再训练 8 步序列；策略种子也重新运行 burn-in，不把旧模型产生的缓存隐状态直接当作当前隐状态。burn-in 是有限历史近似，不保证与从 episode 起点回放完全相同。uint8 深度分辨率约 1.18 cm；相机噪声、失效、标定误差及真实硬件部署尚未验证。

配置与代码入口：

- `go2_supermimic/env/depth_env_cfg.py`：相机、深度换算、任务环境；env 只提供 285 维本体历史。
- `go2_supermimic/supermimic/perception.py`：CNN、Gaussian RSSM、感知损失、含感知状态的想象轨迹。
- `go2_supermimic/supermimic/depth_train.py`：采集时序、Replay、burn-in、训练、迁移与保存。
- `scripts/eval_depth.py`：按地形统计实际速度误差和摔倒，支持遮挡深度对照。

## 训练

在仓库根目录、所选 CUDA GPU 可用且空闲时，从已有 40000 轮平地策略迁移：

```bash
CUDA_VISIBLE_DEVICES=1 /home/yons/anaconda3/envs/mjlab/bin/python scripts/train_depth.py --resume logs/supermimic/2026-10-29_16-53-35_titatit_velocity_resume/model_40000.pt --depth-transfer true --reset-critic true --continue-run false --iterations 40000 --run-name titatit_depth_supermimic_1p5
```

默认 256 环境、Replay 512 帧、WM batch 1024、想象 batch 2048、感知 batch 32，`compile=false`。前 1500 轮训练动力学/感知模型，接下来 500 轮预热 Critic，再渐进更新 Actor。指令与地形课程重新从 0 开始；第 10000 轮起全纵向范围 ±1.5 m/s，第 16000 轮开放全部地形行。`--terrain-curriculum-iterations` 默认 20000；15000 轮启用摩擦与 COM 随机化时会同步重置感知记忆。

平地迁移复用 WM 和 Actor 原本体输入权重，新时序特征输入权重置零。也接受本项目 Terrain checkpoint，但会舍弃原 Actor 的真值高度扫描输入权重，**不保证复现原地形策略输出**。RSSM、Critic 和优化器均重新初始化。不能加载外部 WMP/PPO 权重。

从头训练：

```bash
CUDA_VISIBLE_DEVICES=1 /home/yons/anaconda3/envs/mjlab/bin/python scripts/train_depth.py --iterations 40000 --run-name titatit_depth_scratch
```

同任务续训必须关闭迁移，并恢复计数及 Critic：

```bash
CUDA_VISIBLE_DEVICES=1 /home/yons/anaconda3/envs/mjlab/bin/python scripts/train_depth.py --resume /absolute/path/depth_run/model_10000.pt --continue-run true --reset-critic false --iterations 40000 --run-name titatit_depth_resume
```

若之前改变过网络维度或感知配置，续训需传入相同参数。checkpoint 合同 v4 记录相机、RSSM、机器人观测与地形参数；保存两个模型、Actor、Critic 及四个优化器。Replay 和在线隐状态重新建立，不承诺逐位复现。`scripts/train.py` 拒绝直接启动 Depth，防止漏掉感知训练；原 `eval_velocity.py` 和 `export_actor.py` 也不接受 Depth 模型。

## 评估

用相同 checkpoint、难度、速度和初始 seed 对照真实深度与遮挡深度。九个环境覆盖九类地形；报告每类的线速度/yaw RMSE、摔倒数和有效样本数。边界/超时 reset 的追踪样本不计入 RMSE，倾倒终止单独计数。

```bash
CUDA_VISIBLE_DEVICES=1 /home/yons/anaconda3/envs/mjlab/bin/python scripts/eval_depth.py --checkpoint /absolute/path/depth_run/model_40000.pt --device cuda:0 --num-envs 9 --seconds 30 --terrain-level 0 --vx 0.5 --output depth_real.json
```

```bash
CUDA_VISIBLE_DEVICES=1 /home/yons/anaconda3/envs/mjlab/bin/python scripts/eval_depth.py --checkpoint /absolute/path/depth_run/model_40000.pt --device cuda:0 --num-envs 9 --seconds 30 --terrain-level 0 --vx 0.5 --mask-depth --output depth_masked.json
```

之后逐行检查 `--terrain-level 0..4`，把 `--vx` 提高至 1.5，并检查倒车、转向和停止。相同种子的初始状态一致，但两种策略行为可能导致不同 reset 时刻，因此不是逐步完全配对的轨迹。仅有感知损失下降或高 imagined survival 不能证明机器人会看地形越障。


### 新增 gap / climb（2026-10-30）

Velocity-Terrain 和 Velocity-Depth 现在默认使用九类地形；Chase 保留原七类地图。新增两条横贯地块的沟壑或高台，中央保留 1.5 m 平地出生区，正反向均可接近障碍。

- `gap`：沟宽随难度由 10 cm 增至 40 cm，沟底深 60 cm；使用真实碰撞几何和有限沟底，深度射线与高度扫描均能读取。它不是 WMP 的无底窄桥地图。
- `climb`：高台高度由 5 cm 增至 25 cm，台面沿前进方向宽 80 cm，区别于连续台阶。
- 参数见 `go2_supermimic/env/terrains.py` 的 `GapClimbTerrainCfg`。最终几何随地形生成器的难度采样变化。
- 正在运行的旧训练继续使用内存中的旧地图。旧七类 Terrain / Depth checkpoint 与新地图合同不同，不能直接续训或用新配置评估；旧模型评估需保留原配置/代码。新地图可使用上面的平地模型迁移命令，建议换新 run-name（例如 `titatit_depth_gap_climb_1p5`）。不要绕过合同检查。
- 九类地形评估使用 `--num-envs 9`。沟底落地不能仅凭“未倾倒”算作成功；还需检查是否跨过沟壑、登上并离开高台。当前奖励和倾倒判定沿用原任务，尚未加入专门的越沟成功率统计。

## 验证（2026-10-30）

使用 `/home/yons/anaconda3/envs/mjlab/bin/python`：

- 实际 CPU MuJoCo Warp 深度与独立 MuJoCo 射线查询一致（测试像素误差容限 0.2 mm）；覆盖全部七类地形的缩小地图。
- 验证 5 步图像更新、reset 不受旧隐状态影响、非新帧不读取图像变化、RSSM prior 对动作有非零有限梯度。
- 验证感知参数发生更新；Actor 通过 32 步想象更新，感知/动力学参数在策略更新期间冻结；checkpoint 保存/加载、DR reset 和固定难度遮挡评估通过。
- 使用真实 `2026-10-29_16-53-35_titatit_velocity_resume/model_40000.pt`、默认网络及完整 5×7 地图，完成 7 环境 CPU 采集、WM/感知/Actor/Critic 更新和保存。Actor 输入 `(7,445)`，图像 `(7,1,32,32)`；损失与梯度有限。这是链路证明，不是收敛证明。

GPU 长训、视觉越障通过率、交互 viewer 和 Depth 的 ONNX/实机导出尚未验证或实现。
