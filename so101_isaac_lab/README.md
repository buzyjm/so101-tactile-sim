# SO-101 触觉 Isaac Lab 接口

这一层把现有 Isaac Sim 场景和 52-taxel 融合链路接入 Isaac Lab 3.0 的
`ManagerBasedRLEnv`。Isaac Sim 继续负责 USD、PhysX 接触和渲染；Isaac Lab
负责多环境复制、action/observation/reward/termination manager 以及训练张量。

本接口已在 Python 3.12、Isaac Sim `6.0.1.0`、Isaac Lab
`3.0.0b2.post1` 和 Torch `2.11.0`（CUDA）环境验证。当前 `isaacsim` conda
环境已安装这些版本，且 `pip check` 通过。

## 张量契约

默认环境 `SO101TactileEnvCfg` 使用 240 Hz 物理、`decimation=4`，因此 policy
步频为 60 Hz。两个触觉传感器仍独立按 83.3 Hz 积分并 sample-and-hold。

| 接口 | 形状 | 含义 |
| --- | --- | --- |
| `actions` | `(N, 6)` | 六关节归一化位置命令 |
| `obs["policy"]` | `(N, 312)` | 归一化后按 `[finger,taxel,xyz]` 展平的触觉输入 |
| `obs["tactile_debug"]["taxel_forces"]` | `(N, 2, 52, 3)` | 未归一化三轴分力，单位 N |
| `obs["tactile_debug"]["force_magnitudes"]` | `(N, 2, 52)` | 每个 taxel 的力幅值，单位 N |
| `obs["tactile_debug"]["total_forces"]` | `(N, 2, 3)` | 独立合力通道，单位 N |
| `rewards` | `(N,)` | 当前通用触觉 reward 之和 |

`total_forces` 是独立通道，不用量化后的 52 点简单求和替代。未量化的空间分配
严格保持矢量合力守恒；每个 taxel 独立执行 0.1 N 阈值和量化后，两种合计允许有
量化差异。

当前 reward 仅作为接口基线：双指接触 `1.0`、法向力平衡 `0.1`、超过 20 N
的过载项 `-0.1`。它们不是最终抓取任务的标定权重。

## 关键文件

- `tactile_tensor.py`：纯 Torch、批量、CPU/CUDA 均可运行的 52 点映射与输出时钟。
- `sensors/tactile_contact_sensor.py`：每指一个 PhysX ContactSensor，读取接触点、
  法向力和摩擦力，并转换到真实 CAD 传感器坐标系。
- `scene_cfg.py`：可被 `{ENV_REGEX_NS}` 克隆的机器人、探针和双传感器场景。
- `env_cfg.py`：正式 action、observation、reward、termination 配置。
- `assets/isaac_lab/so101_tactile.usda`：对当前 `lab_scene.usda/World/Robot`
  的轻量引用，不复制机器人资产。

`InteractiveSceneCfg.lazy_sensor_update` 必须为 `False`，否则没有 observation
读取的物理子步不会进入 240 Hz 触觉积分。目前配置已经固定该选项。

## 回归命令

从项目根目录执行：

```bash
# 纯 Torch：CPU + CUDA、形状、拒绝、量化、输出时钟和守恒
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/test_tactile_tensor.py

# 两个克隆环境中的真实 PhysX 动态接触
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/smoke_tactile_observation.py \
  --viz none --num_envs 2 --steps 120 --contact

# 正式 ManagerBasedRLEnv：默认 decimation=4 的零接触接口
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/smoke_manager_env.py \
  --viz none --num_envs 2 --steps 3

# 正式 ManagerBasedRLEnv：人工探针的非零接触路径
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/smoke_manager_env.py \
  --viz none --num_envs 2 --steps 30 --contact
```

接触模式中的小探针会在每个物理步重置，因此该测试临时使用
`decimation=1`；生产配置仍为 `decimation=4`。这只是确定性回归激励，不是训练
场景动力学。

当前 CAD 子引用在 PhysX 启动时会打印若干 `Failed to find articulation at
.../MillimetersToMeters/Geometry` 警告。它们来自仅用于外观的嵌套 CAD 引用；
机器人 articulation、双传感器接触和所有张量回归均正常。后续可通过扁平化视觉
USD 消除日志噪声，不影响本接口使用。

## 展示场景渲染

`TactilePresentationSceneCfg` 是单独的 Isaac Lab 展示场景：它保留完整实验室
背景和真实夹爪，为左右指尖分别配置独立动态探针，并通过 Lab `CameraCfg` 渲染
104 个真实空间位置的 taxel 热力点。探针只是可重复的接触激励，不代表已完成的
橡胶动力学标定。

```bash
conda run --no-capture-output -n isaacsim python \
  scripts/isaac_lab/render_tactile_presentation.py \
  --viz none --device cuda:0 --steps 180 --quality preview
```

输出包括：

- `renders/isaac_lab_tactile_scene_raw.png`：原始 RTX 相机图。
- `renders/isaac_lab_tactile_presentation.png`：带双侧触觉与 policy HUD 的展示图。
- `tactile_logs/isaac_lab_tactile_presentation.json`：本次接触对应的完整
  `(2,52,3)` 数值、合力、taxel 世界坐标、峰值和守恒误差。

空闲 GPU 上可把 `--quality` 改为 `final`，增加相机累计帧数。

### 独立 104-taxel 力矢量场

矢量场模式不生成 taxel 热力球，并在最终渲染中隐藏两个接触激励探针。超过
`0.1 N` 的 taxel 显示为箭头：箭头起点对应 taxel 世界位置，方向对应转换到
世界坐标系的三轴力，显示长度按力幅值映射到 6–32 mm。未受力 taxel 不画箭头，
但 104 个位置和零力值仍完整写入 JSON。

```bash
conda run --no-capture-output -n isaacsim python \
  scripts/isaac_lab/render_tactile_vector_field.py \
  --viz none --device cuda:0 --steps 180 --quality preview
```

独立输出：

- `renders/isaac_lab_taxel_vector_field.png`：带矢量说明 HUD 的结果图。
- `renders/isaac_lab_taxel_vector_field_raw.png`：无 HUD 的 RTX 原图。
- `tactile_logs/isaac_lab_taxel_vector_field.json`：传感器坐标系和世界坐标系下
  的 `(2,52,3)` 力、真实 taxel 位置、显示起点/终点、箭头长度和有效 taxel
  编号。
