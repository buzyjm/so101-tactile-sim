"""Record the parallel pick-and-place scene to a video file.

Rendering happens off-screen and is written straight to disk, which sidesteps
the WebRTC route entirely: the media channel wants UDP and an SSH tunnel only
carries TCP, so the browser client sits on "waiting for stream" even though the
signalling connects.  A recording is also the more useful artefact for showing
someone -- no stutter, and it can be replayed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--env_spacing", type=float, default=1.8,
                    help="Tables are 1.0 x 0.6 m, so the default 1.0 m spacing "
                         "runs them into one another.")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fps", type=int, default=30)
# Solved by frustum projection over the 3x3 grid of environments so all nine
# fit with margin.  The first attempt sat too low and too close: three
# environments fell outside the frame and the robots were cut off at the top.
parser.add_argument("--eye", type=float, nargs=3, default=(-1.15, -3.95, 4.22))
parser.add_argument("--target", type=float, nargs=3, default=(1.275, 1.25, 0.20))
parser.add_argument("--focal", type=float, default=24.0)
parser.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "renders" / "parallel_pick_place.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Rendering must be on for a recording, whatever the launcher defaults to.
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from so101_isaac_lab.env_cfg import SO101TactileEnvCfg
from so101_isaac_lab.scene_cfg import BallPickPlaceSceneCfg

def main() -> int:
    cfg = SO101TactileEnvCfg()
    cfg.scene = BallPickPlaceSceneCfg(
        num_envs=args.num_envs, env_spacing=args.env_spacing,
        replicate_physics=True)
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    scene = env.scene
    scene["overview_camera"].cfg.spawn.focal_length = args.focal
    count = scene.num_envs
    # Solve the framing from the real env_origins rather than assuming a grid:
    # guessing a 3x3 layout put a third of the environments outside the frame.
    origins = scene.env_origins.detach().cpu().numpy()
    corners = []
    for origin in origins:
        for dx in (-0.30, 0.85):
            for dy in (-0.15, 0.65):
                for dz in (0.0, 0.45):
                    corners.append(origin + np.array([dx, dy, dz]))
    corners = np.array(corners)
    centre = (corners.max(axis=0) + corners.min(axis=0)) / 2.0
    aspect = args.width / args.height
    h_fov = 2.0 * np.arctan(20.955 / (2.0 * args.focal))
    v_fov = 2.0 * np.arctan(np.tan(h_fov / 2.0) / aspect)

    def fits(eye):
        forward = centre - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        for point in corners:
            offset = point - eye
            depth = float(offset @ forward)
            if depth <= 0:
                return False
            if abs((offset @ right) / depth) > np.tan(h_fov / 2) * 0.90:
                return False
            if abs((offset @ up) / depth) > np.tan(v_fov / 2) * 0.90:
                return False
        return True

    # Low and across the array, the way Isaac Lab demos are usually framed;
    # 40 degrees looked down on the scene and flattened it.
    # Look down one edge of the grid rather than across its diagonal: the
    # diagonal leaves the array as a thin band with the frame mostly empty.
    azimuth, elevation = np.radians(200.0), np.radians(14.0)
    direction = np.array([np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth),
                          np.sin(elevation)])
    low, high = 1.0, 40.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if fits(centre + direction * middle):
            high = middle
        else:
            low = middle
    eye_position = centre + direction * high
    print(f"envs {count}, span {np.round(corners.min(axis=0), 2)} -> "
          f"{np.round(corners.max(axis=0), 2)}, camera at "
          f"{np.round(eye_position, 2)}, distance {high:.2f} m", flush=True)
    args.eye = tuple(float(v) for v in eye_position)
    args.target = tuple(float(v) for v in centre)
    scene["overview_camera"].set_world_poses_from_view(
        eyes=torch.tensor(args.eye, dtype=torch.float32,
                          device=env.device).repeat(count, 1),
        targets=torch.tensor(args.target, dtype=torch.float32,
                             device=env.device).repeat(count, 1),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (args.width, args.height))
    # A slow sweep of the joints: enough motion to show every environment is
    # live and independent, without pretending to be a trained policy.
    frames = 0
    for step in range(args.steps):
        phase = 2.0 * np.pi * step / max(args.steps - 1, 1)
        action = torch.zeros_like(env.action_manager.action)
        action[:, 1] = 0.45 * np.sin(phase)
        action[:, 2] = 0.35 * np.sin(phase + 0.6)
        action[:, 3] = 0.30 * np.sin(phase + 1.2)
        action[:, 5] = 0.60 * np.sin(2.0 * phase)
        env.step(action)
        # ManagerBasedRLEnv only advances physics when headless; without an
        # explicit render the annotator hands back nothing and the file ends
        # up empty.
        pixels = scene["overview_camera"].data.output["rgb"]
        if pixels is not None and pixels.numel():
            image = pixels[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            frames += 1
        if step % 50 == 0:
            print(f"  step {step}/{args.steps}", flush=True)
    writer.release()
    print(f"{frames} frames -> {args.output}")
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
