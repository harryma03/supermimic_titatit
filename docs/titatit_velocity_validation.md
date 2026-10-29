# Velocity 实际验证记录（2026-10-29）

环境：`/home/yons/anaconda3/envs/mjlab/bin/python`，mjlab 1.6.0、MuJoCo 3.11.0、mujoco-warp 3.11.0、PyTorch 2.11.0+cu128。`nvidia-smi`不能连接驱动，`torch.cuda.is_available()`为False。以下仿真、训练、评估均为CPU执行。

## 结果

| 检查 | 实际结果 |
| --- | --- |
| pytest全部已有/新增测试 | 46 passed；包含38项原测试及8项Velocity测试 |
| 新任务真实观测 | 2环境，每环境285维；与无噪声想象重建及历史移位逐项对齐 |
| 动作 | 12腿位置+4轮速度，原关节顺序和控制常数不变 |
| 指令 | 最终范围/零指令比例/纯轴模式通过统计检查；实际manager定时重采样及固定指令进入历史通过 |
| Replay | command、xi、encoder bias、执行动作和历史保存通过；第五阶段物理变更切断跨阶段窗口 |
| 8步WM | 实际CPU采集数据，loss有限、参数梯度有限 |
| 32步imagination | reward (2,32)、survival (2,33)、Critic输入 (2,33,146)，finite |
| Actor BPTT | 所有参数梯度存在且有限，参数实际变化；WM参数和normalizer完全不变，无WM梯度 |
| Critic | TD(lambda) shape (2,32)，有限；优化后参数变化 |
| 机身接触 | 将机器人降低到机身触地，真实传感器产生接触，状态第90槽非零 |
| 默认网络训练 | Actor 512/256/128、WM 1024/1024/1024、Critic 512/256/128，CPU 2环境3迭代跑通 |
| 同结构续训 | 从model_3.pt接到iteration 4并保存model_4.pt |
| ONNX | 默认Actor+RunningNorm导出[1,285]→[1,16]；三种同输入下ReferenceEvaluator最大绝对误差2.6077032e-7 |
| 旧TITATIT Chase实际环境 | Flat/Obstacles/Terrain均在CPU构建、reset并完成一步；输出分别410/490/795维，观测/奖励有限 |
| 视频 | CPU EGL离屏录制，ffprobe确认320×240、50 FPS、10帧MP4 |
| 站高标定 | 20 s native MuJoCo，50 Hz诊断位置目标；末5 s高度0.362225127±0.000009008 m |

完整pytest日志：`/tmp/velocity-final-tests.log`。旧任务实际环境日志：`/tmp/chase-regression.log`。短程训练日志：`/tmp/velocity-full-train.log`，续训日志：`/tmp/velocity-resume.log`。这些`/tmp`文件是本次本机验证产物，不保证重启后仍存在。

默认网络短程训练checkpoint：`/tmp/velocity-validation/2026-10-29_14-42-08_velocity_full_network_cpu/model_3.pt`。第二、三迭代actor梯度范数22.51、22.53，Critic loss 45.04、35.30；续训第四迭代actor梯度范数30.28。它们不是收敛策略。

## 真实评估，不作为收敛声明

使用上述3迭代checkpoint，2环境，10个连续指令阶段，各0.7 s。包括站立、前后、左右横移、正负旋转、前进转弯、横移旋转、停止。报告路径`/tmp/velocity-suite.json`。

| 指标 | 结果 |
| --- | ---: |
| 二维线速度跟踪RMSE | 0.282121 m/s |
| 竖直角速度跟踪RMSE | 0.431336 rad/s |
| 机身高度mean/std | 0.243003 / 0.047325 m |
| pitch mean/std | 0.257872 / 0.096917 rad |
| 接触点滑移RMS | 0.017705 m/s |
| 每步动作差平方和均值 | 0.013350 |
| 真实全序列return均值 | 0.077954 |
| 真实32步折扣奖励均值 | 0.006192 |
| 想象32步带末端Critic的return均值 | -8.672708 |
| WM 1/8/16/32步归一化状态RMSE | 0.473685 / 2.853623 / 5.365153 / 13.216415 |
| WM 1/8/16/32步高度RMSE | 0.020288 / 0.156772 / 0.313386 / 0.670557 m |

观察到高度不足及长时预测漂移；短序列没有摔倒不等于已经稳定站立。不同return定义和时长不可直接比较。零动作默认姿态也不是稳定站立控制器；高度标定使用独立诊断支持力控制，不参与RL数据采集。

## 未完成的运行验证

- GPU训练、默认torch.compile在GPU上的编译/性能、多环境长程收敛。
- 已训练策略的稳定原地旋转、横移、松杆停止、随机化鲁棒性和实机动作。
- 交互式viewer窗口操作；已验证的是EGL离屏视频。
- ONNX Runtime部署后端；已验证的是标准ONNX ReferenceEvaluator。

这些限制不影响CPU训练链路、任务注册和导出接口的验证结论，但不能据此宣称获得可部署策略。
