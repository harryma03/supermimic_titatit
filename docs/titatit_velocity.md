# TITATIT 三维遥控速度跟踪（SuperMimic）

新增任务 `Titatit-Velocity-Flat`，与原有 Go2/TITATIT Chase Flat、Obstacles、Terrain 独立注册。使用在线仿真数据，不需要专家示范；算法仍是 World Model + 可微想象 + BPTT + TD(lambda) Critic，未引入 PPO。注册表中的 RSL-RL 配置只是 mjlab 注册接口所需，不用于 `scripts/train.py`。

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

**禁止将旧 Chase checkpoint 强行加载到 Velocity**：扩展动力学语义与形状不同，Actor/Critic 也不兼容。新任务从头训练；只接受相同任务、相同配置合同的续训 checkpoint。合同版本 2 包含有效机身碰撞、状态维度、观测项、历史长度、dt 和指令/奖励配置。

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
