# 键盘遥操作

`keyboard_teleop.py` 在 Isaac Sim GUI 里用键盘直接驱动 SO-101。它写的是
`/World/Robot/Physics/<joint>` 上的 `UsdPhysics.DriveAPI` 角度目标，
与 `grasp_v2.py` 完全相同的接口，因此两者可以互相替换。

```bash
conda run --no-capture-output -n isaacsim \
  python scripts/teleop/keyboard_teleop.py
```

常用参数：

```text
--usd PATH            打开的场景，默认 lab_scene_task.usda
--mode joint|cartesian  初始模式，默认 joint
--joint-speed DEG_S   关节点动速度，默认 45 deg/s
--linear-speed M_S    笛卡尔平移速度，默认 0.08 m/s
--gripper-speed DEG_S 夹爪开合速度，默认 90 deg/s
--self-test STEPS     不等键盘，用合成按键把两种模式各跑 STEPS 帧后退出
```

`--self-test` 用来在没人坐在窗口前的情况下回归控制链路：它依次点动六个关节、
合一次夹爪，再在笛卡尔模式下沿 ±X/±Y/±Z 各平移一段，打印每步实际变化的关节
角和 TCP 位置。仍然需要窗口，因为键盘订阅挂在 app window 上。

```bash
conda run --no-capture-output -n isaacsim \
  python -u scripts/teleop/keyboard_teleop.py --self-test 30
```

## 按键

| 键 | 作用 |
| --- | --- |
| `TAB` | 在 joint / cartesian 模式间切换 |
| `1`–`6` | 选择关节（joint 模式） |
| `↑` / `↓` | 点动所选关节（joint 模式） |
| `W`/`S` `A`/`D` `Q`/`E` | 沿世界 X / Y / Z 平移夹爪（cartesian 模式） |
| `Z` / `X` | 点动 `wrist_roll`（cartesian 模式） |
| `O` / `C` | 开 / 合夹爪（两种模式都可用） |
| `R` | 回到载入场景时的位姿 |
| `H` | 打印帮助 |
| `ESC` | 退出并打印最终关节目标 |

## 两种模式

**joint** 只写单个关节目标，不需要 IK，也不依赖 URDF 之外的任何标定，
适合做限位、驱动增益和接触的排查。

**cartesian** 用 `so101_descriptor.yaml` 和 `grasp_v2.py` 相同的
`LulaKinematicsSolver`，以当前关节目标做 warm start 求解 `gripper_frame_link`。
进入该模式时会锁定当前工具姿态，之后只做平移，避免在桌面附近跳到另一个 IK 分支。
IK 失败的那一步不会提交，笛卡尔目标不会累积漂移，终端会打印不可达位置。

## 限制

- 必须有真实窗口：键盘事件来自 Kit app window 的 carb input 接口，
  所以不能 headless，也不能走 `start_browser_viewer.sh` 的 `--no-window` 串流。
- 关节限位在运行时从 USD 的 `physics:lowerLimit` / `physics:upperLimit` 读取，
  没有额外的自碰撞检查——`scripts/diagnostics/check_collision.py` 仍然需要单独跑。
- 遥操作不写任何轨迹记录。要采数据需要另外接
  `isaacsim.replicator.episode_recorder`。
