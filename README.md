# SO-101 触觉仿真环境

在 Isaac Sim 6.0 / Isaac Lab 中还原实验室的 SO-101 机械臂和 DP-S2015 Elite
触觉传感器，目标是**在仿真里训练基于触觉的策略并零样本部署到真机**。

## 当前状态

**已完成**

- 触觉几何：两块指垫各 52 个 taxel，坐标取自 `array.xlsx` 实测值，最大 CAD
  表面误差 0.00412 mm。fixed / moving 各一个 Isaac `ContactSensor`，融合为
  `(2, 52, 3)`，同时保留合力 `(2, 3)` 和幅值 `(2, 52)`。
- Isaac Lab 接口：policy observation 输出归一化 `(N, 312)`，调试 observation
  保留原始维度。
- **脚本化取放任务全流程跑通且可复现**：抬升 +98.7 mm，球入碗距碗心 18.3 mm
  （阈值 28.5 mm），峰值 4.9 N、0.0% 削顶，双垫同时接触 87.3%，静置漂移
  2.32 mm。
- 并行场景：`BallPickPlaceSceneCfg` 在 Isaac Lab 里克隆桌、球、碗和触觉传感
  器，球落位误差 0.0 mm。
- 真机数据对齐：采样率按 `Jingyi-Z/sotac` 原始 sidecar 修正为 90.9 Hz（数据
  手册写 83.3 Hz），21 条红球 episode 的观测分布已刻画。

**未解决 —— 接触模型**

仿真每垫 2.6 个激活 taxel / 1.8–2.4 N；真机 14.3–20.1 个 / 6.2–8.2 N。
单 taxel 力量级是可比的（仿真 0.7–0.85 N，真机约 0.35 N），差的是**参与接触
的 taxel 数量**。已排除且有数据支撑：

| 尝试 | 结果 |
| --- | --- |
| contact offset 标定 | 无效，且 2 mm 会破坏整段搬运（双垫接触 87%→39%） |
| compliant contact 刚度扫描（0 至 1e7，四个数量级） | 激活 taxel 恒为 6/4，无变化 |
| domain randomization（10 episode） | 分布重叠度最多 +0.04 |

刚性球夹在两块平垫之间，接触点数量由碰撞算法决定，改材料参数动不了它。
后续方向是软体 / FEM 球，或 TacSL 式带侧向耦合的 SDF 力场。

**已测出但尚未回灌仿真的真机标定**

- `wrist_flex` 真机上限 +79.3°，URDF 写的是 +95°
- 指令延迟 100–133 ms，仿真未建模
- 保持误差真机 0.2–0.7°，仿真因位置驱动限力有约 9 mm 重力下垂

## 目录结构

```text
.
├── build_scene.py              # 生成场景 USD
├── scene_config.py             # 尺寸、安装位姿、物体布局、物理材质
├── so101_scene_spec.yaml       # 场景与传感器规格（真机测量值的来源）
├── so101_descriptor.yaml       # Lula 运动学描述
├── tactile_*.py                # taxel 几何、融合、SDF 接触模型
├── assets/                     # SO-101 USD、传感器 CAD、taxel 坐标
├── calibration/                # 相机内外参标定数据与结果
├── scripts/
│   ├── tasks/                  # 脚本化取放任务、抓取几何、domain randomization
│   ├── tactile/                # 触觉分析、真机对比、可视化
│   ├── isaac_lab/              # 并行场景、录制、ManagerBasedRLEnv 回归
│   ├── calibration/            # 相机标定流程
│   ├── diagnostics/            # 关节、夹爪、碰撞、可达性诊断
│   ├── experiments/            # 历史实验和参数扫描，非稳定入口
│   ├── rendering/              # 单帧和视频渲染
│   ├── streaming/              # WebRTC 浏览器串流
│   └── teleop/                 # 键盘遥操作
├── so101_isaac_lab/            # 触觉传感器、MDP terms、场景与环境配置
└── official_so101_workshop/    # NVIDIA 官方 Workshop（独立仓库，未纳入版本控制）
```

## 环境

需要 Isaac Sim 6.0 与 Isaac Lab，装在名为 `isaacsim` 的 conda 环境里。
所有命令从项目根目录执行。

SO-101 的 URDF 默认在相邻目录 `../SO-ARM100/Simulation/SO101/so101_new_calib.urdf`，
放在别处时用 `SO101_URDF_PATH` 覆盖。

## 构建场景

场景 USD 不纳入版本控制，用下面两条命令生成：

```bash
# 参考布局：与 real_scene.jpg 对齐，球在碗中
conda run -n isaacsim python build_scene.py

# 任务初态：球在碗外，供取放任务使用
SO101_SCENE_OUT=lab_scene_task.usda SO101_SCENE_LAYOUT=task_ready \
  conda run -n isaacsim python build_scene.py
```

触觉几何默认 `cad + high_fidelity`，可回退：

```bash
SO101_TACTILE_GEOMETRY=mvp conda run -n isaacsim python build_scene.py
SO101_TACTILE_COLLISION=fast conda run -n isaacsim python build_scene.py
```

## 主要入口

```bash
# 脚本化取放任务（确定性专家，用于物理验证）
conda run --no-capture-output -n isaacsim \
  python scripts/tasks/scripted_ball_pick_place.py

# 同上并录像到 renders/pick_place.mp4
conda run --no-capture-output -n isaacsim \
  python scripts/tasks/scripted_ball_pick_place.py --video

# 触觉几何回归
conda run -n isaacsim python scripts/tactile/validate_taxel_geometry.py

# Isaac Lab 端到端回归
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/smoke_manager_env.py --viz none --num_envs 2

# 并行场景录像到 renders/parallel_pick_place.mp4
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/record_parallel_scene.py --num_envs 9
```

`grasp_v2.py` 和 `pick_place_policy.py` 是方块时代的遗留入口，已被
`scripts/tasks/scripted_ball_pick_place.py` 取代。`scene_config.py` 末尾保留了
`BASKET_* = BOWL_*`、`TARGET_CUBE_* = TARGET_BALL_*` 的兼容别名让它们仍能导入，
但不作为稳定入口维护。

触觉实现、回归方法和未标定项见 `scripts/tactile/README.md`；
Isaac Lab 张量接口和训练环境入口见 `so101_isaac_lab/README.md`。

## 键盘遥操作

在 Isaac Sim GUI 里用键盘直接驱动机械臂，写的是 `UsdPhysics.DriveAPI` 角度目标：

```bash
conda run --no-capture-output -n isaacsim \
  python scripts/teleop/keyboard_teleop.py
```

`TAB` 在关节点动和笛卡尔平移之间切换，`1`–`6` 选关节，`↑`/`↓` 点动，
`W/S A/D Q/E` 平移夹爪，`O`/`C` 开合夹爪，`ESC` 退出并打印最终关节目标。
完整按键表见 `scripts/teleop/README.md`。

键盘事件来自 Kit app window，因此必须有真实窗口，不能 headless。

## 浏览器串流

项目包含由 NVIDIA `@nvidia/create-ov-web-rtc-app` 生成的本地 WebRTC 客户端
（`web_client/`，依赖用 `npm install` 还原）：

```bash
./scripts/streaming/start_browser_viewer.sh
```

然后浏览器打开 `http://localhost:5173`。主机防火墙需放行 TCP 49100 和
UDP 47998。

**注意**：媒体通道走 UDP，所以单纯的 SSH 端口转发不够——信令能连上，画面会
一直停在 "waiting for stream"。远程看画面请改用
`scripts/isaac_lab/record_parallel_scene.py` 之类的离屏录制。

该串流没有身份验证和传输加密，不要把 5173、49100 或 47998 暴露到公网。

## 场景坐标系

- X 轴沿桌子长边
- Y 轴从机械臂安装边指向桌面内部
- Z 轴竖直向上，**桌面就是 z = 0**

注意最后一条：在 Isaac Lab 并行场景里地面因此放在 z = −10 mm（桌板底面），
放在 z = 0 会和每张桌面共面 z-fighting。

## 下一步

1. 接触模型：软体球或带侧向耦合的 SDF 力场，把激活 taxel 数量做上去。
2. 给并行环境加任务奖励和终止条件——现在只有触觉相关的
   `bilateral_contact` / `force_balance` / `overload`，没有"把球放进碗"这个目标。
3. 把已测出的真机标定回灌仿真：收紧 `wrist_flex` 上限、建模指令延迟。
4. 球和碗的位姿随机化。
5. 把录制相机从训练场景里拆出来——现在 `CameraCfg` 挂在
   `BallPickPlaceSceneCfg` 上，导致所有使用者都被迫加 `--enable_cameras`。
