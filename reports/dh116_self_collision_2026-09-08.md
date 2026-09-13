# DH116 自碰撞修复与验证

> 这是旧关节模型的历史结果。2026-09-10 已按厂商当前手册和
> 2026-04-09 资产包重新校准，当前结果见
> [dh116_manual_calibration_2026-09-10.md](dh116_manual_calibration_2026-09-10.md)。

实验日期：2026-09-08；文档整理：2026-09-10。

手指穿过另一根手指或手掌会改变策略可利用的动作空间，必须先处理再继续训练。
两套模型此前都设置了 `enabled_self_collisions=False`。
现在 hand-only 和 xArm5 + DH116 默认启用自碰撞，保留原来的视觉网格和配色。
本次完成了接触诊断、七阶段手势测试、短训练和旧策略回放；尚未训练出持续掌面转笔。

## 修复内容

仅打开自碰撞开关并不稳定：初次测试出现明显穿透、关节翻转及越限。
另外，运行时递归设置碰撞参数会跳过 instance proxy，未改到实际碰撞网格。
因此从原资产生成独立的 `usd_self_collision` 版本，在共享 collider payload
中直接写入参数，训练与展示默认引用这个版本。

| 参数 | 最终配置 |
| --- | --- |
| 手部自碰撞 | 开启 |
| 位置 / 速度求解迭代 | 32 / 8 |
| `solve_articulation_contact_last` | `True` |
| 手部刚体最大去穿透速度 | 1 m/s |
| Collider contact / rest offset | 1 mm / 0 mm |
| 手势测试物理步长 | 1/120 s |

关闭 contact-last 的对照测试即使保留其他修改，仍出现约 19 mm 穿透及大幅越限。
16 / 4 次迭代虽然接触穿透很小，瞬态关节越限仍超过本次 1° 阈值；最终采用 32 / 8。
这些设置是本模型的实测选择，不代表通用最优参数。

没有增加额外碰撞对过滤；引擎原有的相邻关节连接体过滤仍然存在。
因此这里不是对每一对视觉网格做零相交保证。接触会挡住部分旧手势的目标角度，
这时较大的目标跟踪误差不能再按关闭自碰撞时的标准判断。

资产生成脚本是 `scripts/dh116_hand/build_collision_assets.py`，支持
`--robot hand` 和 `--robot arm`，目标目录已存在时拒绝覆盖。
每份新资产包含 `collision_manifest.json`，记录来源文件哈希及参数。
回归检查确认原 USD 未改动，新旧视觉 geometry 和 material payload 字节一致，
并通过组合后的 USD 检查了 17 / 24 个实际 collider 的 offset。

## 测量方法与结果

每套模型运行 `open → rock → open → scissors → one → thumbs_up → open`，
每阶段 3 秒，目标用 1 秒平滑过渡。初始化后只使用物理驱动关节目标，不逐帧写关节姿态。
每个物理步读取 PhysX tensor contact view 的接触分离距离，并记录关节状态。
导入模型包含嵌套刚体，诊断为每个刚体启用 contact report API，使用完整刚体路径
构建传感器和过滤列表；早期回调返回零接触的记录不作为无穿透证据。

验收条件：开启自碰撞、有接触样本、采样最大穿透 < 0.5 mm、
最大关节越限 < 1°、各阶段最后 0.5 秒的关节角标准差 < 0.5°。
接触缓冲区达到容量或状态非有限时诊断会报错。

| 指标 | hand-only | xArm5 + DH116 |
| --- | ---: | ---: |
| 采样最大接触穿透 | 0.1254 mm | 0.1587 mm |
| 最大瞬态关节越限 | 0.8423° | 0.6602° |
| 最大稳定段关节角标准差 | 0.0372° | 0.0479° |
| 本次验收 | 通过 | 通过 |

原始结果：

- [hand JSON](../logs/collision_audit/hand_final_2026-09-08.json)
- [arm + hand JSON](../logs/collision_audit/arm_final_2026-09-08.json)
- [手势录像](../renders/dh116_self_collision_hand.mp4)：21 秒，1280×720，630 帧。

上述结果属于固定手势、接触形状和离散采样的有限验证，不覆盖所有策略动作，
也没有真机接触标定。检查过的录像帧未见之前明显的穿指现象；数值诊断是主要验收依据。
后续脚本增加了资产路径、根层哈希及失败退出码，历史 JSON 保持实验原样。

## 训练与回放兼容性

64 环境恢复 v1，使用 heading 奖励和最终碰撞配置，完成 2 次 PPO 迭代：
`logs/collision_audit/current_physics_final_smoke/result.json`。
这只验证训练接口和 checkpoint 保存，不能证明已学会技能。

[旧策略在最终物理配置下的回放](../renders/dh116_v1_self_collision_replay.mp4)
固定 seed 11、env 0，8 秒，1920×1080，240 帧。
该短样本没有掉笔，实际投影转速约 -0.06 rad/s，仍未形成持续掌面转笔。
该次启动日志出现 CUDA OOM 报错，随后继续运行并完成录像；已检查输出帧，
但该回放不作为运行全程无错误或总体性能的证据。

训练和回放默认 `--collision_profile current`，恢复旧 checkpoint 时也会覆盖碰撞配置。
评估默认 `--collision_profile saved`，用于与 checkpoint 保存的物理条件匹配；
显式选择 `current` 才是在新物理下评估历史策略。
新训练保存的是修复后的配置，其 `saved` 评估也会保留自碰撞。
脚本启动会打印所选配置和实际资产路径；关闭自碰撞时会提示属于历史物理。

```bash
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/audit_self_collision.py \
  --robot hand --device cuda:0 --output logs/collision_audit/hand_repeat.json
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/audit_self_collision.py \
  --robot arm --device cuda:0 --output logs/collision_audit/arm_repeat.json
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/play_pen_spin.py \
  --checkpoint logs/rsl_rl/dh116_pen_spin/2026-09-06_02-13-16_v1/model_2999.pt \
  --collision_profile current --device cuda:0 --num_envs 1 --seconds 8 --record \
  --output renders/dh116_v1_self_collision_repeat.mp4
```

## 下一阶段

在修复后的物理配置下建立可达目标姿态的重定向课程，再逐步推进连续掌面旋转。
继续使用独立的实际投影转角、首掉笔和固定朝向评估，避免奖励增加被误认成技能完成。
目前采用显式编写的奖励候选，无需本地 LLM 或外部模型服务；Eureka 的自动 LLM
奖励搜索部分尚未复现。
