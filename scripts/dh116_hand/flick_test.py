"""Scripted feasibility test: can the DH116 spin a pen on its palm at all?

Runs a few open-loop finger patterns in the pen-spin environment and reports
the mean signed spin rate about the palm normal, the best rate, how many
full turns the pen makes and how often it falls off.  No learning.

    conda run --no-capture-output -n isaacsim python scripts/dh116_hand/flick_test.py
    conda run --no-capture-output -n isaacsim python scripts/dh116_hand/flick_test.py --record --envs_per_pattern 4
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
parser.add_argument("--envs_per_pattern", type=int, default=32)
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--period", type=float, default=0.8, help="flick period (s)")
parser.add_argument("--yaw_center_deg", type=float, default=0.0)
parser.add_argument("--yaw_range_deg", type=float, default=15.0)
parser.add_argument("--metrics_output", type=Path, default=None, help="optional JSON metrics path")
parser.add_argument("--velocity_limit", type=float, default=None,
                    help="override the finger joint velocity limit (rad/s)")
parser.add_argument("--record", action="store_true", help="write a video of env 0 of each pattern")
parser.add_argument("--fall_height", type=float, default=None, help="override cfg.fall_height (m)")
parser.add_argument("--fall_dist", type=float, default=None, help="override cfg.fall_dist (m)")
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "renders" / "dh116_flick_test.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.record:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from dh116_hand_lab.evaluation import summarize_rollout  # noqa: E402
from dh116_hand_lab.hand_cfg import HAND_BASE_POS  # noqa: E402
from dh116_hand_lab.pen_spin_env import DH116PenSpinEnv  # noqa: E402
from dh116_hand_lab.pen_spin_env_cfg import DH116PenSpinEnvCfg, PEN_LENGTH, PEN_REST_WORLD  # noqa: E402

# action order: thumb abduction, thumb flexion, index, middle, ring, pinky.
# A pattern maps (t, period, state) -> 6 actions; ``state`` holds the pen's
# end points in the hand frame (see pen_ends_hand) for closed-loop tricks.
PATTERNS = {
    "rest_thumb_up": lambda t, p, s: (0.0, -1.0, -1.0, -1.0, -1.0, -1.0),
    "rest_spread": lambda t, p, s: (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
    "index_flick": lambda t, p, s: (0.0, -1.0, flick(t, p), -1.0, -1.0, -1.0),
    "index_middle_flick": lambda t, p, s: (0.0, -1.0, flick(t, p), flick(t, p), -1.0, -1.0),
    "wave": lambda t, p, s: (0.0, -1.0, flick(t, p), flick(t - 0.25 * p, p),
                             flick(t - 0.5 * p, p), flick(t - 0.75 * p, p)),
    "index_closed_loop": lambda t, p, s: (0.0, -1.0, s["index_hit"], -1.0, -1.0, -1.0),
    "index_middle_closed": lambda t, p, s: (0.0, -1.0, s["index_hit"], s["index_hit"], -1.0, -1.0),
    "pinky_closed_loop": lambda t, p, s: (0.0, -1.0, -1.0, -1.0, -1.0, s["pinky_hit"]),
}


def flick(t: float, period: float) -> float:
    """Open -> closed -> open pulse, closed for the first third of the period."""
    phase = (t % period) / period
    return 1.0 if phase < 0.35 else -1.0


def sweep(t: float, period: float) -> float:
    """Thumb abduction sweeping across the palm and back."""
    phase = (t % period) / period
    return math.sin(2.0 * math.pi * phase)


def pen_ends_hand(env, half_length: float) -> torch.Tensor:
    """Both pen ends in the hand frame (N, 2, 3): x palm normal, y thumb
    side, z along the fingers."""
    from isaaclab.utils.math import quat_apply
    axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    axis_w = quat_apply(env.pen_rot, axis_local)
    centre_w = env.pen_pos  # env-relative world
    ends_w = torch.stack((centre_w + half_length * axis_w, centre_w - half_length * axis_w), dim=1)
    # world -> hand (palm-up base at HAND_BASE_POS): x_h = z_w - 0.5, y_h = -y_w, z_h = x_w
    base = torch.tensor(HAND_BASE_POS, device=env.device)
    rel = ends_w - base
    return torch.stack((rel[..., 2], -rel[..., 1], rel[..., 0]), dim=-1)


def closed_loop_state(env) -> dict:
    """Per-env triggers: curl the index when a pen end sits under it,
    likewise for the pinky (the opposite turning sense)."""
    ends = pen_ends_hand(env, 0.5 * PEN_LENGTH)
    y, z = ends[..., 1], ends[..., 2]
    under_index = ((y > 0.015) & (y < 0.055) & (z > 0.06) & (z < 0.13)).any(dim=1)
    under_pinky = ((y < -0.015) & (y > -0.055) & (z > 0.06) & (z < 0.13)).any(dim=1)
    return {
        "index_hit": torch.where(under_index, 1.0, -1.0),
        "pinky_hit": torch.where(under_pinky, 1.0, -1.0),
    }


def main() -> int:
    names = list(PATTERNS)
    per = args.envs_per_pattern
    cfg = DH116PenSpinEnvCfg()
    cfg.scene.num_envs = per * len(names)
    cfg.scene.env_spacing = 0.6
    cfg.sim.device = args.device
    cfg.act_moving_average = 1.0
    cfg.episode_length_s = args.seconds + 1.0
    cfg.capture_step_state = True
    # Repeating ``id % per`` gives every pattern the same evenly-spaced yaws.
    cfg.eval_yaw_bins = per
    cfg.eval_yaw_center = math.radians(args.yaw_center_deg)
    cfg.eval_yaw_range = math.radians(args.yaw_range_deg)
    if args.velocity_limit is not None:
        for actuator in cfg.robot_cfg.actuators.values():
            actuator.velocity_limit_sim = args.velocity_limit
    if args.fall_height is not None:
        cfg.fall_height = args.fall_height
    if args.fall_dist is not None:
        cfg.fall_dist = args.fall_dist
    if args.record:
        from isaaclab.sensors import CameraCfg
        import isaaclab.sim as sim_utils
        cfg.camera_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/Camera", update_period=0.0, height=720, width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=35.0, horizontal_aperture=20.955,
                                             clipping_range=(0.02, 10.0)))
    env = DH116PenSpinEnv(cfg)
    device = env.device
    count = env.num_envs
    obs, _ = env.reset()
    initial_axis = quat_apply(env.pen_rot, env.z_unit).cpu().numpy().copy()
    initial_yaw = env.initial_yaw.cpu().numpy().copy()

    writer = None
    camera = None
    if args.record:
        import cv2
        camera = env.camera
        origins = env.scene.env_origins.detach().cpu().numpy()
        rest = np.array(PEN_REST_WORLD)
        eyes = origins + rest + np.array([0.25, 0.30, 0.22])
        targets = origins + rest
        camera.set_world_poses_from_view(
            eyes=torch.tensor(eyes, dtype=torch.float32, device=device),
            targets=torch.tensor(targets, dtype=torch.float32, device=device))
        for _ in range(60):
            env.sim.render()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cols = len(names)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                 (1280 * min(cols, 4), 720 * math.ceil(cols / 4)))

    steps = int(args.seconds / env.step_dt)
    spin_sum = torch.zeros(count, device=device)
    spin_max = torch.full((count,), -1e9, device=device)
    turns = torch.zeros(count, device=device)
    drops = torch.zeros(count, device=device)
    drop_step = torch.full((count,), -1, dtype=torch.long, device=device)
    history = {name: [] for name in ("spin_rate", "pen_axis", "pen_pos", "on_palm", "pen_angvel")}
    actions = torch.zeros((count, cfg.action_space), device=device)
    frames = 0
    for step in range(steps):
        t = step * env.step_dt
        state = closed_loop_state(env)
        for k, name in enumerate(names):
            sl = slice(k * per, (k + 1) * per)
            local = {key: val[sl] for key, val in state.items()}
            a = PATTERNS[name](t, args.period, local)
            for j, val in enumerate(a):
                actions[sl, j] = val if isinstance(val, torch.Tensor) else float(val)
        obs, reward, terminated, truncated, info = env.step(actions)
        for key in history:
            history[key].append(env.step_state[key].cpu().numpy().copy())
        spin = env.spin_rate
        spin_sum += spin
        spin_max = torch.maximum(spin_max, spin)
        turns += spin * env.step_dt / (2.0 * math.pi)
        if step in tuple(int(x / env.step_dt) for x in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0)):
            for k in range(min(3, len(names))):
                off = (env.pen_pos[k * per] - env.pen_rest[k * per]).cpu().numpy()
                print(f"t={t:.2f}s {names[k]:16s} env0 pen offset from assumed rest (m): {np.round(off, 4)} "
                      f"|v| {torch.linalg.norm(env.pen_linvel[k * per]).item():.3f} m/s", flush=True)
        newly = terminated & (drop_step < 0)
        drop_step[newly] = step
        drops += terminated.float()
        if writer is not None and step % 2 == 0:
            pixels = camera.data.output["rgb"]
            if pixels is not None and pixels.numel():
                tiles = []
                for k in range(len(names)):
                    img = pixels[k * per, ..., :3].detach().cpu().numpy().astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(img, names[k], (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                    tiles.append(img)
                while len(tiles) % 4:
                    tiles.append(np.zeros_like(tiles[0]))
                rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
                writer.write(np.concatenate(rows, axis=0))
                frames += 1
    if writer is not None:
        writer.release()
        print(f"{frames} frames -> {args.output}")

    spin_mean = (spin_sum / steps).cpu().numpy()
    spin_max = spin_max.cpu().numpy()
    turns = turns.cpu().numpy()
    drops = drops.cpu().numpy()
    drop_step = drop_step.cpu().numpy()
    arrays = {key: np.asarray(value) for key, value in history.items()}
    reports = {}
    print(f"\n{args.seconds:.0f} s per pattern, period {args.period} s, {per} envs each, "
          f"velocity limit {next(iter(cfg.robot_cfg.actuators.values())).velocity_limit_sim} rad/s")
    print(f"{'pattern':20s} {'heading rad/s':>13s} {'turns mean/max':>17s} "
          f"{'omega-z rad/s':>14s} {'dropped':>8s}")
    for k, name in enumerate(names):
        sl = slice(k * per, (k + 1) * per)
        report = summarize_rollout(
            arrays["spin_rate"][:, sl], arrays["pen_axis"][:, sl], arrays["pen_pos"][:, sl],
            arrays["on_palm"][:, sl], initial_axis[sl], initial_yaw[sl],
            env.pen_rest[0].cpu().numpy(), env.step_dt, cfg.spin_sign, per)
        reports[name] = report
        summary = report["summary"]
        projected_turns = np.asarray([trial["projected_turns"] for trial in report["trials"]])
        print(f"{name:20s} {summary['projected_rate_mean_rad_s']:13.4f} "
              f"{projected_turns.mean():7.3f}/{projected_turns.max():<7.3f} "
              f"{summary['omega_z_rate_mean_rad_s']:14.3f} "
              f"{round(summary['drop_rate'] * per):3d}/{per:<3d}")
    if args.metrics_output is not None:
        import json
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "settings": {"seconds": args.seconds, "period": args.period,
                         "envs_per_pattern": per, "yaw_center_deg": args.yaw_center_deg,
                         "yaw_range_deg": args.yaw_range_deg},
            "patterns": reports,
        }
        args.metrics_output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"Metrics -> {args.metrics_output}")
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
