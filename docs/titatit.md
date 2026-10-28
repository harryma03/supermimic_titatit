# TITATIT 三种追逐/躲避任务

| 任务 | 场景 | actor 输入 | 动作 |
| --- | --- | ---: | ---: |
| `Titatit-Chase-Flat` | 平地 | 410 | 16 |
| `Titatit-Chase-Obstacles` | 静态障碍与 16 射线 LiDAR | 490 | 16 |
| `Titatit-Chase-Terrain` | 台阶、方块、粗糙地面、斜坡，77 点高度扫描 | 795 | 16 |

复用 Go2 三种任务的目标生成、奖励权重、地形、课程、随机化和 50 Hz 策略周期。
物理步长仍是本项目的 0.005 秒，decimation=4；没有移植 DreamWaQ 的算法、奖励或随机化。
机器人使用 `go2_supermimic/robot/titatit/mujoco/wheeled_titatit_mjlab.xml` 及其
`meshes_mujoco` 网格。运行时补充 IMU、头部/轮心 site、接触 geom 名字和执行器，源 XML 不变。
保留模型的关节限位、惯量、碰撞体与 55 Nm 腿部 / 10 Nm 轮部力矩限制。

## 控制参数

配置入口：`go2_supermimic/robot/titatit.py`。

默认姿态和腿部增益使用用户提供值；轮控制公式核对自
[TITATIT 原控制器](https://github.com/harryma03/titatit_dreamwaq/blob/main/legged_gym/envs/titatit/titatit.py)
及 [配置](https://github.com/harryma03/titatit_dreamwaq/blob/main/legged_gym/envs/titatit/titatit_config.py)。

- 左髋 0.1、右髋 -0.1；前大腿 1.0、后大腿 -0.8；前小腿 -1.5、后小腿 1.5 rad；轮初始角 0。
- 初始 base 高度 0.50 m，与原训练配置一致。
- 腿：Kp=60、Kd=1.5；髋 action scale=0.125，大腿/小腿=0.25。
- 轮：目标速度 `action * 5.0` rad/s；力矩 `0.5 * (目标速度 - 实际速度)`，限幅 ±10 Nm。
- 原配置虽然写 `foot stiffness=10`，原控制代码把轮位置误差和位置动作清零，所以位置项实际为 0；此处等价采用 Kp=0、Kd=0.5。原 action_scale 中轮的 0.25 不参与控制。
- URDF 标注轮速 25 rad/s；原控制器并未将它作为目标或实际轮速硬限幅，这里保持相同语义。

动作顺序为 **12 个腿关节，再 4 个轮关节**；腿按 FR、FL、RR、RL 排列，每腿 hip/thigh/calf，轮也按 FR、FL、RR、RL 排列。
这是本项目新任务的策略接口，不直接兼容 DreamWaQ 的交错动作顺序。
位置观测和站姿惩罚屏蔽轮累计角度，速度观测包含轮速；世界模型状态 78 维。
模拟和想象使用同一动作顺序、控制公式、机器人尺寸和轮半径。
躲避判定沿用可微胶囊/机身盒近似，轮用半径 0.0948 m 的保守球体近似；物理碰撞仍使用原 MJCF 形状。

## 从头训练

在项目根目录、已安装项目的 mjlab 环境中执行；三条命令分别训练三个任务。

```bash
python scripts/train.py --task Titatit-Chase-Flat --run-name titatit_flat
```

```bash
python scripts/train.py --task Titatit-Chase-Obstacles --run-name titatit_obstacles
```

```bash
python scripts/train.py --task Titatit-Chase-Terrain --run-name titatit_terrain
```

新增任务会跳过与当前 mujoco-warp 不兼容的 `quiet_solver_warnings` 事件，可直接启动。
Go2 原任务配置保持原状。现有 `weights/flat.pt`、`obstacles.pt`、`terrain.pt` 是 Go2 的 12 动作策略，不能用于 TITATIT。
TITATIT 需要重新训练；Go2 的 MoE-CTS driver 也不支持四轮足。

训练后，将下面的 `YOUR_TITATIT_RUN` 和 checkpoint 文件名替换成实际输出路径：

```bash
python scripts/eval.py Titatit-Chase-Flat --checkpoint logs/supermimic/YOUR_TITATIT_RUN/model_1000.pt --mode both --num-envs 1 --seconds 30 --video titatit_flat.mp4
```

障碍/地形评估时，对应更换 task 和该任务的 TITATIT checkpoint；地形可加 `--terrain stairs`。

## 验证范围

```bash
python -m pytest tests
```

测试覆盖三任务配置、模型/力矩限位/默认姿态、Go2 和 TITATIT 的世界模型/缓存/想象反向传播及轮部碰撞。
CPU 短仿真与短训练用于验证数据链路；不代表策略已收敛或验证了实际运动能力。

本次检查：32 项测试通过；平地、障碍、完整 5×7 地形均完成 CPU reset、5 步动作、真实/想象观测对齐与策略反向传播。平地另完成 2 次训练迭代，世界模型、actor、critic 更新与 checkpoint 保存通过；AOT eager 完整图反向传播通过。当前执行会话未检测到 CUDA 设备，GPU/Inductor 运行和长程收敛未验证。

## 指定物理 GPU1

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Chase-Flat --run-name titatit_flat_gpu1
```

`CUDA_VISIBLE_DEVICES=1` 将物理 GPU1 映射为进程内的 `cuda:0`，与现有脚本匹配。另两任务对应换成 `Titatit-Chase-Obstacles`、`Titatit-Chase-Terrain`。

## 任务 2 的虚拟雷达原点

当前是由静态障碍几何直接计算的二维 LiDAR：原点为机器人 base 的 XY，方向随 base yaw 旋转，16 条射线均匀覆盖 360°，最大距离 4 m。不使用实体雷达 link、安装高度或俯仰角，也不扫描机器人本体和抛掷物。仿真任务可直接训练；若要匹配实物雷达，需明确相对 base 的安装位置、朝向和扫描扇区，并同步修改仿真与想象中的扫描实现。轮足模型替换本身不会自动提供雷达外参。

## 轮足高度奖励适配

TITATIT 三个任务统一使用 `stand_height=0.37 m` 作为跳跃/下蹲引导基准（默认姿态运动学估计，仍需以稳定站姿校准）；Go2 保持 0.27 m。
`BallTargetCfg.crouch_min_height=0.23 m`、`base_min_height=0.18 m` 已参数化，当前数值保持不变。
环境与想象奖励读取同一配置，原奖励权重、PD 和动作尺度保持不变。

TensorBoard 新增六项 `Episode_Metrics/penalty/{action_rate_l2,joint_acc_l2,joint_torques_l2}/{legs,wheels}`。
这些数值是每个策略步的 `原始平方和 × 原奖励权重 × step_dt`，在回合内求均值，再对结束的环境求均值；腿轮之和对应原项的每步加权贡献。
它们仅用于诊断，不增加任何奖励。与 `Episode_Reward` 的归一化方式不同，不能直接比较其绝对值。
终端不会逐项显示它们，应在 TensorBoard 查看。

已验证：38 项测试通过；CPU 两次训练迭代、checkpoint 保存及六个 TensorBoard 标量写入通过。
尚未验证修改后的 GPU 长程训练或低横杆躲避效果。
修改不会热加载到已经启动的进程。

从之前的 TITATIT 模型建立独立实验（保留 actor/世界模型，重置 critic；不加 `--continue-run`，迭代和课程从头计数）：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task Titatit-Chase-Flat --resume logs/supermimic/2026-10-28_14-47-23_titatit_flat_gpu1/model_5000.pt --reset-critic true --run-name titatit_flat_height037_gpu1
```
