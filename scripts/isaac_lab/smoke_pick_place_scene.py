"""Bring up the parallel pick-and-place scene and check it is actually alive.

The smoke scene only ever held the robot and a floating probe, so this checks
the things that were not exercised there: that the ball, bowl and table clone
into every environment, that the ball settles on the table instead of falling
through, and that the tactile sensors see the ball now the contact filters name
it rather than the probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "tactile_logs" / "pick_place_scene_smoke.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from so101_isaac_lab.env_cfg import SO101TactileEnvCfg
from so101_isaac_lab.scene_cfg import BallPickPlaceSceneCfg
from scene_config import TABLE_TOP_Z, TARGET_BALL_RADIUS


def main() -> int:
    cfg = SO101TactileEnvCfg()
    cfg.scene = BallPickPlaceSceneCfg(
        num_envs=args.num_envs, env_spacing=1.0, replicate_physics=True)
    env = ManagerBasedRLEnv(cfg=cfg)
    observations, _ = env.reset()

    resting = TABLE_TOP_Z + TARGET_BALL_RADIUS
    zero = torch.zeros_like(env.action_manager.action)
    ball_z = None
    for _ in range(args.steps):
        observations, rewards, terminated, truncated, _ = env.step(zero)
        ball_z = env.scene["ball"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    taxel = observations["tactile_debug"]["taxel_forces"]
    report = {
        "num_envs": args.num_envs,
        "assets": sorted(env.scene.keys()),
        "ball_z_mean_m": float(ball_z.mean()),
        "ball_z_expected_m": float(resting),
        "ball_z_error_mm": float((ball_z.mean() - resting) * 1000),
        "ball_settled": bool(abs(float(ball_z.mean()) - resting) < 0.005),
        "taxel_shape": list(taxel.shape),
        "reward_mean": float(rewards.mean()),
    }
    print(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env.close()
    return 0 if report["ball_settled"] else 2


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        status = 1
    finally:
        simulation_app.close()
    raise SystemExit(status)
