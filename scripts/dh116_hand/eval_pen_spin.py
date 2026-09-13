"""Evaluate a checkpoint on fixed yaw bins, with pre-reset, first-episode metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=32.0)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--yaw_bins", type=int, default=16)
parser.add_argument("--yaw_center_deg", type=float, default=0.0)
parser.add_argument("--yaw_range_deg", type=float, default=180.0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--save_trajectory", action="store_true")
parser.add_argument("--force_drop_env", type=int, default=None, help="Integration check: drop this env at step 0")
parser.add_argument("--collision_profile", choices=("current", "saved"), default="saved")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seconds < 2 or args.num_envs < 1 or not 0 <= args.yaw_bins <= args.num_envs:
    parser.error("need seconds >= 2, num_envs >= 1, and 0 <= yaw_bins <= num_envs")
if args.force_drop_env is not None and not 0 <= args.force_drop_env < args.num_envs:
    parser.error("force_drop_env must be an environment index")
if not 0 <= args.yaw_range_deg <= 180:
    parser.error("yaw_range_deg must be in [0, 180]")
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dh116_hand_lab.collision_profile import apply_collision_profile
from dh116_hand_lab.checkpoint import load_checkpoint_configs
from dh116_hand_lab.evaluation import summarize_rollout
from dh116_hand_lab.pen_spin_env import DH116PenSpinEnv


def main():
    env_cfg, agent_cfg = load_checkpoint_configs(args.checkpoint)
    apply_collision_profile(env_cfg, args.collision_profile)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.seconds + 1.0
    env_cfg.eval_yaw_bins = args.yaw_bins
    env_cfg.eval_yaw_center = math.radians(args.yaw_center_deg)
    env_cfg.eval_yaw_range = math.radians(args.yaw_range_deg)
    env_cfg.capture_step_state = True
    agent_cfg["device"] = args.device
    agent_cfg["seed"] = args.seed
    env = DH116PenSpinEnv(env_cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
        runner = OnPolicyRunner(wrapped, agent_cfg, log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        # Explicit reset after model construction gives all candidates the same starts.
        wrapped.seed(args.seed)
        obs, _ = wrapped.reset()
        initial_axis = quat_apply(env.pen_rot, env.z_unit).cpu().numpy().copy()
        initial_yaw = env.initial_yaw.cpu().numpy().copy()
        home = env.pen_rest[0].cpu().numpy().copy()
        history = {name: [] for name in ("spin_rate", "pen_axis", "pen_pos", "on_palm", "pen_angvel")}
        with torch.inference_mode():
            for step in range(round(args.seconds / env.step_dt)):
                if step == 0 and args.force_drop_env is not None:
                    ids = torch.tensor([args.force_drop_env], device=env.device, dtype=torch.int32)
                    pose = env.pen.data.root_pose_w.torch[ids].clone()
                    pose[:, 2] = 0.1
                    env._write_pen_pose(root_pose=pose, env_ids=ids)
                obs, _, _, _ = wrapped.step(policy(obs))
                for name in history:
                    history[name].append(env.step_state[name].cpu().numpy().copy())
        report = summarize_rollout(history["spin_rate"], history["pen_axis"], history["pen_pos"],
                                   history["on_palm"], initial_axis, initial_yaw, home,
                                   env.step_dt, env_cfg.spin_sign, args.yaw_bins)
        report["checkpoint"] = str(args.checkpoint.resolve())
        report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        report["seed"] = args.seed
        report["collision_profile"] = args.collision_profile
        report["effective_usd_path"] = env_cfg.robot_cfg.spawn.usd_path
        report["force_drop_env"] = args.force_drop_env
        report["evaluation_overrides"] = {
            "num_envs": args.num_envs, "seed": args.seed,
            "episode_length_s": env_cfg.episode_length_s,
            "eval_yaw_bins": args.yaw_bins,
            "eval_yaw_center_deg": args.yaw_center_deg,
            "eval_yaw_range_deg": args.yaw_range_deg,
            "device": args.device,
        }
        report["source_sha256"] = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [Path(__file__).resolve(),
                                              PROJECT_ROOT / "dh116_hand_lab/evaluation.py",
                                              PROJECT_ROOT / "dh116_hand_lab/pen_spin_env.py",
                                              PROJECT_ROOT / "dh116_hand_lab/rewards.py"]}
        report["config_sha256"] = {name: hashlib.sha256((args.checkpoint.parent / name).read_bytes()).hexdigest()
                                    for name in ("env_cfg.yaml", "agent_cfg.yaml")}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.save_trajectory:
            np.savez_compressed(args.output.with_suffix(".npz"), **{k: np.asarray(v) for k, v in history.items()},
                                initial_axis=initial_axis, initial_yaw=initial_yaw, home=home, dt=env.step_dt)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["summary"], indent=2), flush=True)
        print(f"Evaluation saved: {args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        status = 1
    finally:
        app.close()
    raise SystemExit(status)
