"""Train the DH116 pen-spinning policy with rsl_rl PPO.

    conda run --no-capture-output -n isaacsim python scripts/dh116_hand/train_pen_spin.py --num_envs 4096 --max_iterations 2000

Logs and checkpoints go to logs/rsl_rl/dh116_pen_spin/<timestamp>/ (tensorboard).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--resume", type=str, default=None, help="checkpoint .pt to resume from")
parser.add_argument("--spin_reward_scale", type=float, default=None)
parser.add_argument("--forward_spin_reward_scale", type=float, default=None)
parser.add_argument("--target_spin_reward_scale", type=float, default=None)
parser.add_argument("--target_spin_rate", type=float, default=None)
parser.add_argument("--target_spin_sigma", type=float, default=None)
parser.add_argument("--best_progress_reward_scale", type=float, default=None)
parser.add_argument("--progress_goal_bonus_scale", type=float, default=None)
parser.add_argument("--progress_target_deg", type=float, default=None)
parser.add_argument("--alive_bonus", type=float, default=None)
parser.add_argument("--fall_penalty", type=float, default=None)
parser.add_argument("--action_penalty_scale", type=float, default=None)
parser.add_argument("--action_rate_scale", type=float, default=None)
parser.add_argument("--palm_center_penalty_scale", type=float, default=None)
parser.add_argument("--velocity_limit", type=float, default=None)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--reset_position_noise", type=float, default=None)
parser.add_argument("--reset_dof_pos_noise", type=float, default=None)
parser.add_argument("--yaw_center_deg", type=float, default=None)
parser.add_argument("--yaw_range_deg", type=float, default=None)
parser.add_argument("--spin_reward_mode", choices=("omega_z", "heading"), default=None)
parser.add_argument("--tilt_penalty_scale", type=float, default=None)
parser.add_argument("--entropy_coef", type=float, default=None)
parser.add_argument("--init_std", type=float, default=None)
parser.add_argument("--std_type", choices=("scalar", "log"), default=None)
parser.add_argument("--log_dir", type=Path, default=None)
parser.add_argument("--collision_profile", choices=("current", "saved"), default="current")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 4 or (args.max_iterations is not None and args.max_iterations < 1):
    parser.error("need num_envs >= 4 and max_iterations >= 1")
if args.episode_length_s is not None and args.episode_length_s <= 0:
    parser.error("episode_length_s must be positive")
if args.target_spin_rate is not None and args.target_spin_rate <= 0:
    parser.error("target_spin_rate must be positive")
if args.target_spin_sigma is not None and args.target_spin_sigma <= 0:
    parser.error("target_spin_sigma must be positive")
if args.progress_target_deg is not None and args.progress_target_deg <= 0:
    parser.error("progress_target_deg must be positive")
if args.yaw_range_deg is not None and not 0 <= args.yaw_range_deg <= 180:
    parser.error("yaw_range_deg must be in [0, 180]")
if args.init_std is not None and args.init_std <= 0:
    parser.error("init_std must be positive")
if args.log_dir is not None and args.log_dir.exists():
    parser.error("log_dir must be a new directory")
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from dh116_hand_lab.agents import DH116PenSpinPPORunnerCfg  # noqa: E402
from dh116_hand_lab.pen_spin_env import DH116PenSpinEnv  # noqa: E402
from dh116_hand_lab.pen_spin_env_cfg import DH116PenSpinEnvCfg  # noqa: E402
from dh116_hand_lab.collision_profile import apply_collision_profile
from dh116_hand_lab.checkpoint import load_checkpoint_configs  # noqa: E402


DEPRECATED_MODEL_KEYS = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")


def clean_agent_cfg(cfg: dict) -> dict:
    """Drop the pre-rsl-rl-5 model fields that isaaclab_rl still serialises
    (as MISSING -> {}) and that rsl_rl 5.x's models reject as kwargs."""
    for key in ("actor", "critic"):
        model = cfg.get(key)
        if isinstance(model, dict):
            for name in DEPRECATED_MODEL_KEYS:
                model.pop(name, None)
    return cfg


def main() -> int:
    if args.resume:
        env_cfg, agent_dict = load_checkpoint_configs(args.resume)
    else:
        env_cfg = DH116PenSpinEnvCfg()
        agent_dict = clean_agent_cfg(DH116PenSpinPPORunnerCfg().to_dict())
    apply_collision_profile(env_cfg, args.collision_profile)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.eval_yaw_bins = 0
    env_cfg.capture_step_state = False
    for name in ("spin_reward_scale", "forward_spin_reward_scale", "target_spin_reward_scale",
                 "target_spin_rate", "target_spin_sigma", "alive_bonus", "fall_penalty",
                 "best_progress_reward_scale", "progress_goal_bonus_scale",
                 "action_penalty_scale", "action_rate_scale", "palm_center_penalty_scale",
                 "episode_length_s", "reset_position_noise", "reset_dof_pos_noise",
                 "tilt_penalty_scale"):
        value = getattr(args, name)
        if value is not None:
            setattr(env_cfg, name, value)
    if args.progress_target_deg is not None:
        env_cfg.progress_target_angle = math.radians(args.progress_target_deg)
    if (env_cfg.best_progress_reward_scale != 0.0
            or env_cfg.progress_goal_bonus_scale != 0.0):
        env_cfg.progress_observation = True
    if args.yaw_center_deg is not None:
        env_cfg.train_yaw_center = math.radians(args.yaw_center_deg)
    if args.yaw_range_deg is not None:
        env_cfg.train_yaw_range = math.radians(args.yaw_range_deg)
        env_cfg.randomize_pen_yaw = args.yaw_range_deg > 0
    if args.spin_reward_mode is not None:
        env_cfg.spin_reward_mode = args.spin_reward_mode
        if args.spin_reward_mode == "heading":
            # Symmetric bounds: back-and-forth motion must not get net spin credit.
            env_cfg.spin_clip = (-10.0, 10.0)
    if args.velocity_limit is not None:
        for actuator in env_cfg.robot_cfg.actuators.values():
            actuator.velocity_limit_sim = args.velocity_limit

    agent_dict["seed"] = args.seed
    agent_dict["device"] = args.device
    if args.max_iterations is not None:
        agent_dict["max_iterations"] = args.max_iterations
    if args.run_name:
        agent_dict["run_name"] = args.run_name
    if args.entropy_coef is not None:
        agent_dict["algorithm"]["entropy_coef"] = args.entropy_coef
    distribution = agent_dict["actor"].get("distribution_cfg")
    if distribution is not None:
        if args.init_std is not None:
            distribution["init_std"] = args.init_std
        if args.std_type is not None:
            distribution["std_type"] = args.std_type

    log_root = PROJECT_ROOT / "logs" / "rsl_rl" / agent_dict["experiment_name"]
    log_dir = args.log_dir or log_root / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                          + (f"_{agent_dict['run_name']}" if agent_dict["run_name"] else ""))
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    dump_yaml(str(log_dir / "env_cfg.yaml"), env_cfg)
    dump_yaml(str(log_dir / "agent_cfg.yaml"), agent_dict)
    manifest = {"resume": str(Path(args.resume).resolve()) if args.resume else None,
                "resume_sha256": hashlib.sha256(Path(args.resume).read_bytes()).hexdigest() if args.resume else None,
                "argv": sys.argv, "source_sha256": {}}
    for src in sorted((PROJECT_ROOT / "dh116_hand_lab").glob("*.py")) + [Path(__file__).resolve()]:
        relative = src.relative_to(PROJECT_ROOT)
        target = log_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest["source_sha256"][str(relative)] = hashlib.sha256(src.read_bytes()).hexdigest()
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"logging to {log_dir}", flush=True)

    env = DH116PenSpinEnv(env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_dict.get("clip_actions"))
    runner = OnPolicyRunner(env, agent_dict, log_dir=str(log_dir), device=args.device)
    if args.resume:
        runner.load(args.resume, map_location=args.device)
        print(f"resumed from {args.resume}", flush=True)
    runner.learn(num_learning_iterations=agent_dict["max_iterations"], init_at_random_ep_len=True)
    checkpoint = log_dir / f"model_{runner.current_learning_iteration}.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Runner did not write final checkpoint: {checkpoint}")
    (log_dir / "result.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "iterations_added": agent_dict["max_iterations"],
        "final_iteration": runner.current_learning_iteration,
    }, indent=2) + "\n")
    env.close()
    return 0


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
