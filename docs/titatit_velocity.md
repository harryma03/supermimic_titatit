# TITATIT 三维遥控速度跟踪（SuperMimic）

视觉感知版本：[`Titatit-Velocity-Depth`（WMP 感知 + SuperMimic）](titatit_depth.md)，使用独立的 `scripts/train_depth.py`。

新增任务 `Titatit-Velocity-Flat`，与原有 Go2/TITATIT Chase Flat、Obstacles、Terrain 独立注册。使用在线仿真数据，不需要专家示范；算法仍是 World Model + 可微想象 + BPTT + TD(lambda) Critic，未引入 PPO。注册表中的 RSL-RL 配置只是 mjlab 注册接口所需，不用于 `scripts/train.py`。

## 1.5 m/s 混合地形训练（2026-10-30）

新增独立任务 `Titatit-Velocity-Terrain`，保留原 Flat 任务、机器人模型、PD、动作比例和力矩限制。这里“全地形”指配置中的九类混合地形，不表示任意障碍均可通过，也不表示训练后已能以 1.5 m/s 通过所有台阶。

- 速度入口：`go2_supermimic/env/velocity_env_cfg.py` 的 `terrain` 分支。最终范围为纵向 **±1.5 m/s**、横向 **±0.4 m/s**、转向 **±1 rad/s**。每类地形使用相同最终指令范围。
- 地形入口：`go2_supermimic/env/terrains.py` 的 `ROUGH_TERRAINS`，包含平地、上台阶、下台阶、随机方块、随机起伏、正/反坡面。台阶高度 2–10 cm、踏面 40 cm；方块高度参数 2–5 cm；起伏 1–4 cm；坡度参数 0–0.3。mjlab curriculum 模式为每种地形生成一列，不按 `proportion` 分配列数；这里有 5 行 × 9 类、每块 12×12 m。
- 环境从第 0 行开始，`--terrain-curriculum-iterations 20000` 表示逐步开放更难地形，到 16000 次迭代开放第 4 行；reset 时从已开放行随机选择，持续保留简单地形。机器人离开分配地块中心 ±5.5 m 时按 timeout 重置，防止长期跑离训练区域。
- 指令课程沿用阶段 0/3000/6000/10000/15000：纵向上限依次为 0.375/0.75/0.75/1.5/1.5 m/s，逐步加入倒车、横移、旋转和组合。15000 次迭代后加入原有摩擦/COM 随机化。
- 实际奖励和 Replay 的站高均相对脚下地面；World Model 每步接收地形 relief，Actor 和 Critic 加入相同地形扫描。Actor 为 **670→16**（134 维单帧 × 5 帧，term-major），Critic 输入 223，WM 仍为 236→126。Actor 导出后也需要外部提供 77 点高度扫描，不能继续用旧 285 维部署接口。
- 为允许越阶动作，新任务中 height/vertical_speed/leg_deviation/contact_change 权重分别为 -3/-0.1/-0.03/-0.005；其他奖励保持原值。这是初始训练配置，尚无长训通过率证明。

推荐从现有 40000 轮平地模型迁移，在仓库根目录执行（CUDA 可用且所选 GPU 空闲时）：

```bash
CUDA_VISIBLE_DEVICES=1 /home/yons/anaconda3/envs/mjlab/bin/python scripts/train.py --task Titatit-Velocity-Terrain --num-envs 1024 --compile false --resume logs/supermimic/2026-10-29_16-53-35_titatit_velocity_resume/model_40000.pt --velocity-transfer true --reset-critic true --continue-run false --wm-rewarmup-iterations 1500 --critic-warmup-iterations 500 --actor-lr-scale 0.25 --actor-ramp-iterations 2000 --terrain-curriculum-iterations 20000 --iterations 40000 --run-name titatit_velocity_terrain_1p5
```

迁移会复用 Actor/WM 权重，扫描新增输入权重置零，保留原 Actor 初始输出；重新建立 Replay、优化器和 Critic。从新任务第 0 轮开始：前 1500 次迭代只学习 WM，随后预热 Critic 500 次迭代，再逐步启动 Actor。`--velocity-transfer true` 仅允许 Flat v2 → Terrain v3，且必须 `--reset-critic true --continue-run false`；不接受 Chase 模型。Terrain v3 合同保存有效地形参数，后续直接改地图参数会拒绝同任务续训。

若从头训练，使用相同任务，删除 `--resume` 和 `--velocity-transfer`；无需平地模型。若是同一个 Terrain run 中断后续训，使用新地形 checkpoint，去掉迁移标记并恢复计数：

```bash
CUDA_VISIBLE_DEVICES=1 /home/yons/anaconda3/envs/mjlab/bin/python scripts/train.py --task Titatit-Velocity-Terrain --num-envs 1024 --compile false --resume /absolute/path/terrain_run/model_10000.pt --continue-run true --reset-critic false --actor-lr-scale 0.25 --actor-ramp-iterations 2000 --terrain-curriculum-iterations 20000 --iterations 40000 --run-name titatit_velocity_terrain_resume
```

评估器自动按 checkpoint 选择平地/地形任务；`--terrain-level` 固定难度行 0–4。九个环境可以覆盖九类地形；下面检验最高难度的 1.5 m/s 前进指令，报告整体跟踪、滑移和摔倒指标。需分别评估各行、正负速度及停止，不能仅看整体 reward 判断成功：

```bash
CUDA_VISIBLE_DEVICES=1 /home/yons/anaconda3/envs/mjlab/bin/python scripts/eval_velocity.py --checkpoint /absolute/path/terrain_run/model_40000.pt --terrain-level 4 --vx 1.5 --num-envs 9 --seconds 30 --output terrain_level4_vx1p5.json
```


### 新增 gap / climb（2026-10-30）

Velocity-Terrain 和 Velocity-Depth 现在默认使用九类地形；Chase 保留原七类地图。新增两条横贯地块的沟壑或高台，中央保留 1.5 m 平地出生区，正反向均可接近障碍。

- `gap`：沟宽随难度由 10 cm 增至 40 cm，沟底深 60 cm；使用真实碰撞几何和有限沟底，深度射线与高度扫描均能读取。它不是 WMP 的无底窄桥地图。
- `climb`：高台高度由 5 cm 增至 25 cm，台面沿前进方向宽 80 cm，区别于连续台阶。
- 参数见 `go2_supermimic/env/terrains.py` 的 `GapClimbTerrainCfg`。最终几何随地形生成器的难度采样变化。
- 正在运行的旧训练继续使用内存中的旧地图。旧七类 Terrain / Depth checkpoint 与新地图合同不同，不能直接续训或用新配置评估；旧模型评估需保留原配置/代码。新地图可使用上面的平地模型迁移命令，建议换新 run-name（例如 `titatit_depth_gap_climb_1p5`）。不要绕过合同检查。
- 九类地形评估使用 `--num-envs 9`。沟底落地不能仅凭“未倾倒”算作成功；还需检查是否跨过沟壑、登上并离开高台。当前奖励和倾倒判定沿用原任务，尚未加入专门的越沟成功率统计。

当前验证覆盖 CPU 实际生成地形、观测/奖励/Replay 对齐、平地迁移、8 步 WM、32 步 BPTT、Critic、保存/加载、地形评估和 ONNX 导出；GPU 长训及 1.5 m/s 台阶通过能力另需验证。

本次全仓库测试为 `51 passed`。此外使用真实 `2026-10-29_16-53-35_titatit_velocity_resume/model_40000.pt`、默认完整 5×7 地图和默认网络，在 CPU 上完成 7 环境采集、WM 更新、32 步 BPTT/Critic 更新及 checkpoint 保存，输入为 `(7,670)`，高度场为 `(2175,1575)`，loss 与梯度有限。集成回归测试使用缩小地图但保留全部七类地形。这些短程验证不代表策略已学会复杂地形。

## 文件与调用链

- `env/velocity_command.py`：三维速度采样、阶段课程、固定评估指令。
- `env/velocity_env_cfg.py`：任务配置、接触传感器、共享奖励接入、第五阶段域随机化。
- `supermimic/velocity.py`：物理附加状态、坐标约定、观测重建、统一奖励公式。
- `supermimic/velocity_imagine.py`：无目标球的 32 步可微 rollout，复用现有 TD(lambda)。
- `env/__init__.py`：任务注册；`supermimic/state.py`：按任务推导观测项并读取扩展动力学状态。
- `supermimic/buffer.py`：按需扩展状态，保存 command、xi、encoder bias；Velocity 的目标生成器维度为 0。
- `supermimic/world_model.py`：支持任务附加状态，旧任务默认维度不变。
- `supermimic/train.py`：按任务分离命令、想象、课程、日志与 checkpoint；复用采集、优化器、稀有窗口归档、BPTT 和 Critic 更新。
- `scripts/eval_velocity.py`、`scripts/export_actor.py`、`scripts/calibrate_velocity_height.py`：独立评估、Actor 导出、站高诊断。
- `tests/test_velocity.py`：配置、采样、真实 CPU Warp 观测/奖励/接触与训练链路回归。

上述相对代码路径均位于 `go2_supermimic/` 下，`scripts/`、`tests/` 除外。原 `scripts/eval.py` 继续评估 Chase。

## 机器人与网络

以下网络维度与课程描述针对原 `Titatit-Velocity-Flat`；新 Terrain 任务的差异和命令见上节。

控制顺序不变：FR、FL、RR、RL 各 hip/thigh/calf 共 12 项，最后 FR/FL/RR/RL foot 共 4 项。腿位置 PD 为 Kp=60、Kd=1.5，hip scale=0.125，thigh/calf scale=0.25；轮速度目标为 action×5 rad/s，轮 Kp=0、Kd=0.5。腿/轮力矩分别限幅 ±55/±10 Nm。物理步长 0.005 s，decimation=4，策略周期 0.02 s。

没有修改 MJCF、网格、惯量、关节或执行器配置。**新任务额外启用了 MJCF 已有的 8 个机身碰撞盒**：继承的 wheel-only `CollisionCfg` 原本会关闭它们。盒子尺寸及位置不变；旧 Chase 的碰撞设置不变。腿部其他原有碰撞设置没有变动。

| 网络 | 输入 | 默认隐藏层 | 输出 |
| --- | --- | --- | --- |
| Actor（含 RunningNorm） | 285 | 512/256/128，ELU | 16 |
| World Model | 236 | 1024/1024/1024，SiLU | 126 |
| Critic（含 RunningNorm） | 146 | 512/256/128，ELU | 1 |

Actor 单帧 57 维，5 帧历史：gyro 3、projected gravity 3、joint pos 16、joint vel 16、previous executed action 16、velocity command 3。

**历史是 term-major，不是把五个 57 维帧直接拼接**：gyro 的五帧 15 项，然后 gravity 的五帧 15 项，依此类推；每项内部 oldest→newest。轮子累计转角和其 encoder bias 槽始终为零；轮速保留。训练仍对 gyro、gravity、joint velocity 加噪；Velocity 的位置观测不加随机噪声，但保留腿编码器偏差。真实和想象观测采用相同约定。首次 reset 时 mjlab 用首帧填满历史。

World Model 原机器人核心状态仍为 78 维；Velocity 追加 45 维：四轮接触点 yaw-frame 平面切向速度 8、轮接触状态 4、机身接触状态 1、关节驱动力矩 16、关节加速度 16。总状态 123 维。

输入 236 = 123 + 当前动作16 + 上一动作16 + context4 + scan77；输出 126 = 状态增量123 + 平面位姿增量3。平地 relief=0，原扫描语义仍是 base height + relief。context 为轮地滑动摩擦系数和 base COM 偏移3；轮滚动/扭转摩擦固定，避免产生 context 未表达的随机动力学。

Critic 输入 146 = 状态123 + command3 + previous action16 + context4。旧 TITATIT Chase Actor 410/490/795 和 World Model 191→81 均不变。

**禁止将旧 Chase checkpoint 强行加载到 Velocity**：扩展动力学语义与形状不同，Actor/Critic 也不兼容。常规续训只接受相同任务、相同配置合同的 checkpoint；Flat → Terrain 必须使用上节的显式迁移入口。平地合同版本 2 包含有效机身碰撞、状态维度、观测项、历史长度、dt 和指令/奖励配置；地形版本 3 另保存有效地形参数。

## 坐标与 ROS 2 接口

`[vx, vy, yaw_rate]` 表示机身水平坐标系下前、左、逆时针，单位 m/s、m/s、rad/s。线速度使用已有 World Model 的 **yaw frame**，只消除世界 yaw，不引入机身 roll/pitch。角速度跟踪使用 `-dot(normalize(projected_gravity_b), gyro_b)`，即世界竖直轴分量；不要把倾斜时的 body-z gyro 直接当成它。

部署端需按这个水平坐标约定解释 `cmd_vel`；若上游消息使用随 roll/pitch 倾斜的完整 body frame，先转换到水平坐标系。Actor 不需要估计线速度作为输入。gyro、gravity 必须使用与训练相同的机身坐标轴及符号，关节位置减去 `robot/titatit.py` 的默认角度。previous action 是上一周期实际执行的、未乘 scale 的动作；遇到部署端动作限幅时也应回填限幅后的值。

构建 285 维 term-major 历史，以 float32 `[1,285]` 送入 ONNX，得到 `[1,16]`；归一化已包含在模型内，不要再次归一化。前12维还原为默认角+scale×action，后4维为 5×action rad/s。每20 ms更新，保留当前执行器力矩限幅。没有加入真实机器人驱动或 ROS 节点。

## 指令课程

配置入口：`VelocityCommandCfg`。每个环境独立在 2–5 s 重采样。最终范围 vx±1、vy±0.4、yaw±1；最终模式概率为静止25%、纯纵向20%、纯横移15%、纯旋转20%、组合20%。运动→静止、静止→运动由这些离散模式转换产生。

| 训练迭代 | 内容 |
| --- | --- |
| 0–2999 | 静止/低速向前，最大0.25 m/s |
| 3000–5999 | 静止/前后，范围缩放0.5 |
| 6000–9999 | 加入纯横移、纯旋转，范围缩放0.5 |
| 10000–14999 | 完整范围，加入组合指令 |
| ≥15000 | 保持完整指令，加入每环境摩擦0.5–1.3和COM各轴±0.015 m随机化 |

早期阶段将未启用模式概率置零，再对剩余概率重新归一化。进度依据明确的 Trainer iteration，`collect_steps` 不会隐式改变阶段阈值。第五阶段只应用一次物理随机化，然后 reset；清断 Replay 跨物理变更的窗口，保留每条旧记录的 xi 和 encoder bias，避免把旧数据当成新物理参数。

每次想象从 Replay 读取当时的 command，32 步（0.64 s）内保持不变，不调用 BallTarget。真实仿真依然按2–5 s重采样；因此这是短期 held-command 近似，未预测接近重采样边界的命令跳变。评估指令切换通过 command manager 在奖励之后、下一帧观测之前更新，不手工重排历史。

## 统一奖励

入口：`VelocityRewards`；真实仿真与 imagination 共用 `reward_terms`，每项最后乘控制 dt=0.02。Trainer 从环境 RewardTermCfg 同步权重。sigma_lin=sigma_yaw=0.5。

| 项 | 默认权重 | 定义 |
| --- | ---: | --- |
| track_linear | 1.5 | exp(-平面速度误差平方和/0.5²) |
| track_yaw | 1.0 | exp(-竖直角速度误差平方/0.5²) |
| upright | 0.5 | exp(-gravity_xy平方和/0.4²) |
| tilt_rate | -0.05 | body gyro x/y平方和，不惩罚正常yaw |
| height | -10 | (base_z−0.3622)² |
| vertical_speed | -0.5 | 竖直速度平方 |
| action_rate | -0.05 | 当前与上一执行动作差平方和 |
| joint_acc | -2.5e-7 | 实测/预测关节加速度平方和 |
| joint_torque | -2e-5 | 实测/预测关节驱动力矩平方和 |
| joint_limits | -1 | 腿关节超过soft limit的距离和 |
| leg_deviation | -0.1 | 腿角偏离默认值平方和 |
| wheel_slip | -0.2 | 有接触时，四轮接触点平面切向速度平方和 |
| wheel_speed | -0.01 | 超过25 rad/s的轮速部分平方和；不是硬限速 |
| body_contact | -2 | 机身实际接触指示/模型预测值 |
| contact_change | -0.02 | 相邻两帧轮接触状态差平方和 |
| stand_motion | -1 | 零指令时平面速度²+竖直角速度²+0.01×关节速度² |
| fall | -200 | 倾角70°附近、宽度5°的sigmoid概率 |

滑移采用实际最强轮地接触点，计算 `v_link + omega_link × (contact_pos − link_pos)`，再去掉接触法向分量。轮子完整刚体角速度包含轮速，纯滚动时接触点速度趋近零；这不是用轮轴平移速度冒充打滑。当前每轮保留一个最强接触，不代表所有接触点的面积积分。

真实终止仍使用倾角70°，奖励中的fall项与 imagination 同为软概率；想象回报另用累计软存活率加权，不通过硬done截断梯度。接触/关节限位项使用分段可微计算。

当前安装的 mjlab 在 reward/termination 阶段读取的派生量比最终 observation 早一个物理子步（0.005 s）。两侧奖励公式、量纲、权重和dt一致，但不能宣称预测状态奖励与真实引擎逐子步数值完全一致。

站高标定命令见下。原零动作落地会逐渐倾倒，不能以初始高度0.50 m作为稳定站高。诊断支持力控制器在不改变控制模式的情况下生成50 Hz腿位置目标，原PD在200 Hz执行；20 s实验最后5 s平均高度0.362225127 m、标准差0.000009008 m、最大姿态角约0.020075 rad，四轮接触。它仅校准可达站姿，不参与训练、不提供专家数据，也不证明神经网络策略已学会站立。

## 训练与命令

先激活用户的现有环境：

```bash
conda activate mjlab
```

训练（如只有CPU，将设备设为cpu并大幅减小批量；默认规模面向GPU）：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Velocity-Flat --num-envs 1024 --run-name titatit_velocity
```

一次迭代：在线actor+探索噪声采样 → Replay → 8步自回归World Model loss → 冻结WM参数但保留计算图 → 32步actor闭环想象 → 累计折扣/软存活回报+末端Critic → BPTT更新Actor → TD(lambda)更新Critic。Actor、WM、Critic的RunningNorm、优化器和网络参数写入checkpoint；TensorBoard复用原系统。Velocity运行目录额外保存`cmd.txt`与`config.json`。

同结构续训（`iterations`为总迭代目标，`continue-run`恢复课程进度；buffer重新采集，不承诺逐位复现）：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Velocity-Flat --resume /absolute/path/model_1000.pt --continue-run true --iterations 40000 --run-name titatit_velocity_resume
```

单指令、多环境评估：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_velocity.py --checkpoint /absolute/path/model_1000.pt --vx 0.5 --vy 0 --yaw 0 --num-envs 64 --seconds 30 --output velocity_forward.json
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_velocity.py --checkpoint /absolute/path/model_1000.pt --vx 0 --vy 0 --yaw 0.8 --num-envs 64 --seconds 30
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_velocity.py --checkpoint /absolute/path/model_1000.pt --vx 0.5 --vy 0 --yaw 0.5 --num-envs 64 --seconds 30
```

完整连续指令序列：站立、前进、后退、左右横移、正负旋转、前进转弯、横移旋转、停止；不在阶段之间重置机器人。`seconds`是每阶段时长。

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_velocity.py --checkpoint /absolute/path/model_1000.pt --suite --num-envs 64 --seconds 10 --output velocity_suite.json
```

报告线/角速度RMSE、站高与roll/pitch统计、每环境每分钟摔倒次数、接触滑移RMS、动作变化、真实return、32步想象return及WM的1/8/16/32步开环误差。线速度RMSE使用二维误差向量的平方范数。WM误差同时报告归一化状态RMSE和物理单位下速度/高度/平面位移RMSE；跨reset窗口不计。每阶段少于32步时可能没有32步有效误差，用null表达。真实累计return和带Critic bootstrap的想象return口径不同，不应直接比较大小或据此判定收敛。

单环境交互查看/录像：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_velocity.py --checkpoint /absolute/path/model_1000.pt --vx 0.5 --yaw 0.5 --num-envs 1 --viewer
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_velocity.py --checkpoint /absolute/path/model_1000.pt --suite --num-envs 1 --seconds 5 --video velocity.mp4
```

Actor导出（需要`onnx`；本机mjlab环境已有）：

```bash
python scripts/export_actor.py --checkpoint /absolute/path/model_1000.pt --output exported/titatit_velocity.onnx
```

输出固定`[1,285]→[1,16]`的Actor+RunningNorm及同名JSON接口合同。导出后使用相同随机/零/归一化均值输入对比PyTorch与ONNX ReferenceEvaluator，rtol=1e-4、atol=1e-5；失败直接报错。无需World Model或Critic参与部署。ONNX Runtime未安装，尚未验证具体部署端ORT后端。

站高重现、测试与TensorBoard：

```bash
python scripts/calibrate_velocity_height.py
```

```bash
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/velocity-cache python -m pytest -q
```

```bash
tensorboard --logdir logs/supermimic
```

## 验证边界（2026-10-29）

使用`/home/yons/anaconda3/envs/mjlab/bin/python`。已完成真实CPU MuJoCo Warp采样、默认网络短程训练/续训、8步WM和32步BPTT、TD(lambda)、观测/奖励对齐、接触传感器、域随机化阶段切换、ONNX ReferenceEvaluator对比，以及10段速度指令评估。详细结果见[验证记录](titatit_velocity_validation.md)。

这些是实现与训练链路验证。当前CUDA驱动不可用，未运行GPU长程训练；短程checkpoint没有学会运动，不可部署。原地旋转压低机身、轮滑与停止漂移仍需在训练收敛后以真实仿真指标检验。默认`torch.compile=true`的GPU路径及交互viewer窗口尚未验证；CPU验证使用`compile=false`。
