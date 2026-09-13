"""Run a trained pen-spinning policy and report / record what it does.

    conda run --no-capture-output -n isaacsim python scripts/dh116_hand/play_pen_spin.py --checkpoint logs/rsl_rl/dh116_pen_spin/<run>/model_1000.pt --num_envs 16
    ... --record --num_envs 4     # video of env 0 (closeup) to renders/dh116_pen_spin.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=16.0)
parser.add_argument("--record", action="store_true")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--record_env", type=int, default=0, help="Record this fixed env; no automatic best-case selection")
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "renders" / "dh116_pen_spin.mp4")
parser.add_argument("--collision_profile", choices=("current", "saved"), default="current")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.seconds <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0
        or not 0 <= args.record_env < args.num_envs):
    parser.error("need positive time/fps/resolution and 0 <= record_env < num_envs")
args.headless = True
if args.record:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from dh116_hand_lab.pen_spin_env import DH116PenSpinEnv  # noqa: E402
from dh116_hand_lab.pen_spin_env_cfg import PEN_REST_WORLD  # noqa: E402
from dh116_hand_lab.collision_profile import apply_collision_profile
from dh116_hand_lab.checkpoint import load_checkpoint_configs  # noqa: E402
from dh116_hand_lab.rewards import projected_spin_rate  # noqa: E402


def main() -> int:
    env_cfg, agent_dict = load_checkpoint_configs(args.checkpoint)
    apply_collision_profile(env_cfg, args.collision_profile)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = args.seconds + 1.0
    env_cfg.seed = args.seed
    env_cfg.capture_step_state = True
    if args.record:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        env_cfg.camera_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/Camera", update_period=0.0,
            height=args.height, width=args.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=40.0, horizontal_aperture=20.955,
                                             clipping_range=(0.02, 10.0)))
    env = DH116PenSpinEnv(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_dict.get("clip_actions"))
    agent_dict["device"] = args.device
    runner = OnPolicyRunner(wrapped, agent_dict, log_dir=None, device=args.device)
    runner.load(str(args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    wrapped.seed(args.seed)
    wrapped.reset()

    writer = None
    if args.record:
        import cv2
        origins = env.scene.env_origins.detach().cpu().numpy()
        rest = np.array(PEN_REST_WORLD)
        eyes = origins + rest + np.array([0.28, 0.32, 0.24])
        targets = origins + rest
        env.camera.set_world_poses_from_view(
            eyes=torch.tensor(eyes, dtype=torch.float32, device=env.device),
            targets=torch.tensor(targets, dtype=torch.float32, device=env.device))
        for _ in range(60):
            env.sim.render()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record_envs = [args.record_env]
        writers = {
            k: cv2.VideoWriter(str(args.output),
                               cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                               (args.width, args.height))
            for k in record_envs
        }
        writer = writers[args.record_env]
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {args.output}")

    obs = wrapped.get_observations()
    steps = int(args.seconds / env.step_dt)
    spin = torch.zeros(env.num_envs, device=env.device)
    turns = torch.zeros(env.num_envs, device=env.device)
    drops = torch.zeros(env.num_envs, device=env.device)
    height = torch.zeros(env.num_envs, device=env.device)   # pen centre above home
    tilt = torch.zeros(env.num_envs, device=env.device)     # pen axis angle from the palm plane
    from isaaclab.utils.math import quat_apply
    axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    previous_axis = quat_apply(env.pen_rot, axis_local).clone()
    first_episode = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    observed_steps = torch.zeros(env.num_envs, device=env.device)
    render_every = max(1, round(1.0 / (args.fps * env.step_dt)))
    frames = 0
    with torch.inference_mode():
        for step in range(steps):
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            state = env.step_state
            axis_w = state["pen_axis"]
            drops += (first_episode & ~state["on_palm"]).float()
            first_episode &= state["on_palm"]
            rate = projected_spin_rate(previous_axis, axis_w, env.step_dt, env_cfg.spin_sign)
            previous_axis = axis_w.clone()
            rate = torch.where(first_episode, rate, 0.0)
            spin += rate
            turns += rate * env.step_dt / (2.0 * math.pi)
            observed_steps += first_episode.float()
            height += (state["pen_pos"][:, 2] - env.pen_rest[:, 2]) * first_episode
            tilt += torch.rad2deg(torch.asin(axis_w[:, 2].abs().clamp(max=1.0))) * first_episode
            if writer is not None and step % render_every == 0:
                pixels = env.camera.data.output["rgb"]
                if pixels is not None and pixels.numel():
                    for k, wr in writers.items():
                        image = pixels[k, ..., :3].detach().cpu().numpy().astype(np.uint8)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        text = f"env {k} axis spin {rate[k].item():5.1f} rad/s   turns {turns[k].item():5.1f}"
                        if not first_episode[k]:
                            text += "  DROPPED (trial ended)"
                        text_y = args.height - 28
                        text_scale = max(0.7, args.width / 1600.0)
                        cv2.putText(image, text, (24, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    text_scale, (20, 20, 20), 5, cv2.LINE_AA)
                        cv2.putText(image, text, (24, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    text_scale, (240, 240, 240), 2, cv2.LINE_AA)
                        wr.write(image)
                    frames += 1
    if writer is not None:
        for wr in writers.values():
            wr.release()
        print(f"{frames} frames -> {args.output} (fixed env {args.record_env}, seed {args.seed})")
    spin = (spin / steps).cpu().numpy()
    turns = turns.cpu().numpy()
    drops = drops.cpu().numpy()
    height = (height / observed_steps.clamp(min=1)).cpu().numpy()
    tilt = (tilt / observed_steps.clamp(min=1)).cpu().numpy()
    print(f"\n{args.seconds:.0f} s, {env.num_envs} envs: mean projected spin {spin.mean():.2f} rad/s "
          f"(median {np.median(spin):.2f}, best {spin.max():.2f}); turns mean {turns.mean():.1f} "
          f"max {turns.max():.1f}; envs that dropped the pen: {int((drops > 0).sum())}/{env.num_envs}")
    print(f"pen centre height above home: mean {height.mean() * 1000:.1f} mm; "
          f"pen axis tilt from the palm plane: mean {tilt.mean():.1f} deg (max env {tilt.max():.1f})")
    print("per-env spin rad/s:", np.round(spin, 2))
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
