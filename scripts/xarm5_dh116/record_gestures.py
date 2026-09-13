"""Gesture show for the xArm5 + DH116 hand, cloned across environments.

Every environment holds the hand up to the camera and runs its own random
take of the choreography in xarm5_dh116_lab.gestures (count to five, three
rounds of rock-paper-scissors with a fist shake, a wave, a thumbs up).
Pure joint-target kinematics under physics: no contacts, nothing to grasp.

    conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/record_gestures.py --probe
    conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/record_gestures.py --num_envs 25
    conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/record_gestures.py --num_envs 4 --closeup

--probe checks the simulated hand pose against the numpy FK in
xarm5_dh116_lab.kinematics (base orientation, palm direction, finger
tracking) without rendering.
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
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--env_spacing", type=float, default=1.5)
parser.add_argument("--probe", action="store_true",
                    help="No rendering: verify base/hand pose vs FK and "
                         "finger tracking per gesture, then exit.")
parser.add_argument("--closeup", action="store_true",
                    help="Frame env 0's hand instead of the whole grid.")
parser.add_argument("--stream", action="store_true",
                    help="Live show instead of a recording: loop random takes "
                         "in real time on the Kit viewport, for the WebRTC "
                         "browser stream (scripts/streaming/"
                         "start_browser_viewer.sh with STREAM_APP_CMD).  "
                         "Implies --livestream 1 unless given.")
parser.add_argument("--stagger_s", type=float, default=2.0,
                    help="Random start delay per environment (env 0 starts "
                         "at once).")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--focal", type=float, default=None,
                    help="Camera focal length (mm); default 24 overview, "
                         "30 closeup.")
parser.add_argument("--azimuth_deg", type=float, default=25.0,
                    help="Overview camera azimuth from +X, degrees.")
parser.add_argument("--elevation_deg", type=float, default=16.0)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.headless = True
if args.stream:
    import os
    if args.livestream < 0:
        args.livestream = 1
    # start_browser_viewer.sh exports the GPU pinning and media address the
    # stock streaming app gets; AppLauncher takes them as kit args.
    extra = os.environ.get("STREAM_KIT_ARGS", "")
    args.kit_args = " ".join(part for part in (args.kit_args or "", extra) if part)
    if args.livestream == 1 and not os.environ.get("PUBLIC_IP"):
        import subprocess
        os.environ["PUBLIC_IP"] = subprocess.check_output(
            ["hostname", "-I"], text=True).split()[0]
elif not args.probe:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_physx.physics import PhysxCfg
from isaaclab.scene import InteractiveScene  # noqa: E402

from xarm5_dh116_lab.gestures import (  # noqa: E402
    GESTURES, arm_targets, build_timeline, choreography, hand_targets,
    label_at)
from xarm5_dh116_lab.kinematics import UrdfChain  # noqa: E402
from xarm5_dh116_lab.robot_cfg import ALL_JOINTS, ARM_PRESENT, HAND_JOINTS  # noqa: E402
from xarm5_dh116_lab.scene_cfg import PEDESTAL_HEIGHT, GestureSceneCfg  # noqa: E402

PHYSICS_HZ = 120
CONTROL_HZ = 60


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    # Isaac Lab 3.0's data API hands quaternions back as (x, y, z, w) -- the
    # same convention its root-pose write path expects, whatever the
    # InitialStateCfg docstring says.
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def solve_overview(origins: np.ndarray, width: int, height: int, focal: float,
                   azimuth_deg: float, elevation_deg: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Closest camera on the given bearing that keeps every robot in frame."""
    corners = []
    for origin in origins:
        for dx in (-0.25, 0.85):
            for dy in (-0.30, 0.30):
                for dz in (0.0, PEDESTAL_HEIGHT + 0.95):
                    corners.append(origin + np.array([dx, dy, dz]))
    corners = np.array(corners)
    centre = (corners.max(axis=0) + corners.min(axis=0)) / 2.0
    aspect = width / height
    h_fov = 2.0 * np.arctan(20.955 / (2.0 * focal))
    v_fov = 2.0 * np.arctan(np.tan(h_fov / 2.0) / aspect)

    def fits(eye: np.ndarray) -> bool:
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
            if abs((offset @ right) / depth) > np.tan(h_fov / 2) * 0.93:
                return False
            if abs((offset @ up) / depth) > np.tan(v_fov / 2) * 0.93:
                return False
        return True

    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    direction = np.array([np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth),
                          np.sin(elevation)])
    low, high = 0.5, 80.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if fits(centre + direction * middle):
            high = middle
        else:
            low = middle
    return centre + direction * high, centre


def stream_show(sim, scene, command, origins: np.ndarray, count: int,
                dt: float) -> int:
    """Loop random takes in real time on the Kit viewport until the app
    closes.  The viewport is what the WebRTC stream carries, and the viewer
    can orbit it with the mouse."""
    import time

    import carb.settings
    import omni.kit.viewport.utility as vp_utils
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    from pxr import Gf

    settings = carb.settings.get_settings()

    def pump_app() -> None:
        # Isaac Lab 3.0 no longer pumps the Kit app loop from
        # sim.step(render=True) unless a Kit visualizer is configured, and
        # the headless livestream launch leaves that visualizer inert: the
        # stream connected but its encoder timed out waiting for a frame.
        # Pump Kit ourselves; physics stays owned by sim.step().
        settings.set_bool("/app/player/playSimulations", False)
        simulation_app.update()
        settings.set_bool("/app/player/playSimulations", True)

    # Kit's default viewport camera: 18.15 mm on a 20.955 mm aperture.
    eye, target = solve_overview(origins, 1920, 1080, 18.15,
                                 args.azimuth_deg, args.elevation_deg)
    viewport = vp_utils.get_active_viewport()
    camera_path = viewport.get_active_camera() or "/OmniverseKit_Persp"
    state = ViewportCameraState(str(camera_path), viewport)
    state.set_position_world(Gf.Vec3d(*[float(v) for v in eye]), False)
    state.set_target_world(Gf.Vec3d(*[float(v) for v in target]), True)
    print(f"stream viewport {camera_path} eye {np.round(eye, 2)} "
          f"target {np.round(target, 2)}", flush=True)
    for _ in range(30):
        pump_app()

    rng = np.random.default_rng(args.seed)
    substeps = PHYSICS_HZ // CONTROL_HZ
    env_index = np.arange(count)
    take = 0
    while simulation_app.is_running():
        timelines, labels = [], []
        for _ in range(count):
            timeline, label = build_timeline(choreography(rng), CONTROL_HZ)
            timelines.append(timeline)
            labels.append(label)
        steps = min(t.shape[0] for t in timelines)
        timelines = np.stack([t[:steps] for t in timelines])
        offsets = np.round(rng.uniform(0.0, args.stagger_s, size=count) * CONTROL_HZ).astype(int)
        offsets[0] = 0
        total = steps + int(offsets.max()) + CONTROL_HZ
        take += 1
        print(f"take {take}: {total / CONTROL_HZ:.1f} s", flush=True)
        start = time.perf_counter()
        for control_step in range(total):
            if not simulation_app.is_running():
                break
            index = np.clip(control_step - offsets, 0, steps - 1)
            command(timelines[env_index, index])
            for sub in range(substeps):
                sim.step(render=False)
                scene.update(dt)
            pump_app()
            # the livestream rate limiter paces app updates at 60 Hz; this
            # keeps the show at 1x even when it does not
            lag = (control_step + 1) / CONTROL_HZ - (time.perf_counter() - start)
            if lag > 0:
                time.sleep(lag)
    return 0


def main() -> int:
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / PHYSICS_HZ, render_interval=1, device=args.device,
        physics=PhysxCfg(solve_articulation_contact_last=True))
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = GestureSceneCfg(
        num_envs=args.num_envs, env_spacing=args.env_spacing,
        replicate_physics=True)
    if args.probe or args.stream:
        scene_cfg.camera = None
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    device = sim.device
    count = scene.num_envs
    origins = scene.env_origins.detach().cpu().numpy()
    joint_ids, joint_names = robot.find_joints(ALL_JOINTS, preserve_order=True)
    assert list(joint_names) == ALL_JOINTS, joint_names
    hand_body, _ = robot.find_bodies("hand_base_link")
    hand_body = hand_body[0]
    dt = sim.get_physics_dt()

    def command(targets: np.ndarray) -> None:
        """targets: (count, 16) joint positions in ALL_JOINTS order."""
        robot.set_joint_position_target_index(
            target=torch.as_tensor(targets, dtype=torch.float32, device=device),
            joint_ids=joint_ids)
        scene.write_data_to_sim()

    def hand_pose_env0() -> tuple[np.ndarray, np.ndarray]:
        pos = robot.data.body_link_pos_w[0, hand_body].detach().cpu().numpy()
        quat = robot.data.body_link_quat_w[0, hand_body].detach().cpu().numpy()
        return pos - origins[0], quat_to_matrix(quat)

    if args.probe:
        chain = UrdfChain()
        print("bodies:", list(robot.body_names))
        print("joints:", list(robot.joint_names))
        arm_links = ["link1", "link2", "link3", "link4", "link5", "hand_base_link"]
        arm_ids, arm_link_names = robot.find_bodies(arm_links, preserve_order=True)
        print("arm link bodies:", arm_link_names)

        def compare(tag: str, q_arm: dict[str, float]) -> None:
            q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
            print(f"--- {tag}: sim joints (deg) {np.round(np.degrees(q[:5]), 1)} "
                  f"targets {np.round(np.degrees([q_arm[n] for n in ALL_JOINTS[:5]]), 1)}")
            sim_q = {name: float(q[i]) for i, name in enumerate(ALL_JOINTS[:5])}
            for body_id, name in zip(arm_ids, arm_link_names):
                sim_pos = (robot.data.body_link_pos_w[0, body_id].detach().cpu().numpy()
                           - origins[0] - np.array([0.0, 0.0, PEDESTAL_HEIGHT]))
                fk_pos, _ = chain.link_pose(name, sim_q)
                print(f"    {name:15s} sim {np.round(sim_pos, 3)}  fk(sim q) {np.round(fk_pos, 3)}")

        zero = np.concatenate([arm_targets({n: 0.0 for n in ALL_JOINTS[:5]}), hand_targets("open")])
        command(np.tile(zero, (count, 1)))
        for _ in range(int(2.0 * PHYSICS_HZ)):
            sim.step(render=False)
            scene.update(dt)
        compare("zero pose", {n: 0.0 for n in ALL_JOINTS[:5]})
        rest = np.concatenate([arm_targets(), hand_targets("open")])
        command(np.tile(rest, (count, 1)))
        for _ in range(int(2.0 * PHYSICS_HZ)):
            sim.step(render=False)
            scene.update(dt)
        compare("present pose", ARM_PRESENT)
        command(np.tile(rest, (count, 1)))
        for _ in range(int(2.0 * PHYSICS_HZ)):
            sim.step(render=False)
            scene.update(dt)
        root = robot.data.root_pos_w[0].detach().cpu().numpy() - origins[0]
        root_q = robot.data.root_quat_w[0].detach().cpu().numpy()
        pos, rot = hand_pose_env0()
        fk_pos, fk_rot = chain.link_pose("hand_base_link", ARM_PRESENT)
        fk_pos = fk_pos + np.array([0.0, 0.0, PEDESTAL_HEIGHT])
        print(f"root pos {np.round(root, 4)} quat(wxyz) {np.round(root_q, 4)}")
        print(f"hand base sim {np.round(pos, 4)}  fk {np.round(fk_pos, 4)}  "
              f"err {np.linalg.norm(pos - fk_pos) * 1000:.1f} mm")
        print(f"fingers (+z) sim {np.round(rot[:, 2], 3)} fk {np.round(fk_rot[:, 2], 3)}")
        print(f"palm    (+x) sim {np.round(rot[:, 0], 3)} fk {np.round(fk_rot[:, 0], 3)}")
        q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
        print("arm joint error (deg):",
              np.round(np.degrees(q[:5] - rest[:5]), 2))
        # joint5 sweep: which way does the palm face, where is the thumb
        thumb_id, _ = robot.find_bodies("hand_finger11_link")
        index_id, _ = robot.find_bodies("hand_finger21_link")
        # thumbs-up search: fingers towards the camera (joint4 = 0 makes the
        # pitch sum -pi/2), palm sideways, thumb spread -> which roll puts
        # the thumb on top?
        for j5 in (0.0, math.pi / 2.0, math.pi, 1.5 * math.pi):
            pose = dict(ARM_PRESENT)
            pose["joint4"] = 0.0
            pose["joint5"] = j5
            command(np.tile(np.concatenate([arm_targets(pose), hand_targets("thumbs_up")]), (count, 1)))
            for _ in range(int(1.5 * PHYSICS_HZ)):
                sim.step(render=False)
                scene.update(dt)
            pos, rot = hand_pose_env0()
            thumb = robot.data.body_link_pos_w[0, thumb_id[0]].detach().cpu().numpy() - origins[0] - pos
            index = robot.data.body_link_pos_w[0, index_id[0]].detach().cpu().numpy() - origins[0] - pos
            tip_id, _ = robot.find_bodies("hand_finger14_link")
            tip = robot.data.body_link_pos_w[0, tip_id[0]].detach().cpu().numpy() - origins[0] - pos
            print(f"joint4=0 joint5={math.degrees(j5):5.1f} deg: fingers(+z) {np.round(rot[:, 2], 2)} "
                  f"palm(+x) {np.round(rot[:, 0], 2)} thumb base {np.round(thumb, 3)} "
                  f"thumb tip {np.round(tip, 3)} index base {np.round(index, 3)}")
        for name in ("rock", "scissors", "thumbs_up", "open"):
            goal = np.concatenate([arm_targets(), hand_targets(name)])
            command(np.tile(goal, (count, 1)))
            for _ in range(int(1.0 * PHYSICS_HZ)):
                sim.step(render=False)
                scene.update(dt)
            q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
            err = np.degrees(q[5:] - goal[5:])
            print(f"{name:10s} hand joint error (deg): max {np.abs(err).max():.2f} "
                  f"per joint {np.round(err, 1)}")
            fingertip_names = ["hand_finger14_link", "hand_finger23_link",
                               "hand_finger33_link", "hand_finger43_link",
                               "hand_finger53_link"]
            ids, _ = robot.find_bodies(fingertip_names, preserve_order=True)
            tips = robot.data.body_link_pos_w[0, ids].detach().cpu().numpy() - origins[0]
            print("           fingertips z (m):", np.round(tips[:, 2], 3),
                  " x:", np.round(tips[:, 0], 3))
        return 0

    if args.stream:
        return stream_show(sim, scene, command, origins, count, dt)

    # choreography, one random take per environment, env 0 undelayed
    rng = np.random.default_rng(args.seed)
    timelines = []
    labels = []
    for _ in range(count):
        timeline, label = build_timeline(choreography(rng), CONTROL_HZ)
        timelines.append(timeline)
        labels.append(label)
    steps = min(t.shape[0] for t in timelines)
    timelines = np.stack([t[:steps] for t in timelines])  # (count, steps, 16)
    offsets = np.round(rng.uniform(0.0, args.stagger_s, size=count) * CONTROL_HZ).astype(int)
    offsets[0] = 0
    total_control = steps + int(offsets.max()) + CONTROL_HZ  # 1 s tail
    print(f"{count} envs, {steps / CONTROL_HZ:.1f} s per take, "
          f"{total_control / CONTROL_HZ:.1f} s total", flush=True)

    # hold the opening pose while the renderer streams the meshes in
    command(timelines[:, 0])
    for _ in range(int(1.0 * PHYSICS_HZ)):
        sim.step(render=False)
        scene.update(dt)

    import cv2

    focal = args.focal or (30.0 if args.closeup else 24.0)
    if args.closeup:
        # Aim between the upright hand and where the thumbs-up puts it
        # (fingers towards the camera, ~0.2 m closer and lower), so neither
        # pose leaves the frame.
        hand_pos, _ = hand_pose_env0()
        target = origins[0] + hand_pos + np.array([0.10, 0.0, 0.0])
        eye = target + np.array([0.95, 0.30, 0.12])
    else:
        eye, target = solve_overview(origins, args.width, args.height, focal,
                                     args.azimuth_deg, args.elevation_deg)
    print(f"camera eye {np.round(eye, 2)} target {np.round(target, 2)} "
          f"focal {focal}", flush=True)
    camera = scene["camera"]
    camera.cfg.spawn.focal_length = focal
    camera.set_world_poses_from_view(
        eyes=torch.tensor(eye, dtype=torch.float32, device=device).repeat(count, 1),
        targets=torch.tensor(target, dtype=torch.float32, device=device).repeat(count, 1))
    for _ in range(3 * args.fps):
        sim.render()
        scene.update(dt)

    output = args.output or (PROJECT_ROOT / "renders" /
                             ("dh116_gestures_closeup.mp4" if args.closeup
                              else "dh116_gestures.mp4"))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    render_every = PHYSICS_HZ // args.fps
    substeps = PHYSICS_HZ // CONTROL_HZ
    frames = 0
    env_index = np.arange(count)
    for control_step in range(total_control):
        index = np.clip(control_step - offsets, 0, steps - 1)
        command(timelines[env_index, index])
        for sub in range(substeps):
            physics_step = control_step * substeps + sub
            render = physics_step % render_every == 0
            sim.step(render=render)
            scene.update(dt)
            if not render:
                continue
            pixels = camera.data.output["rgb"]
            if pixels is None or not pixels.numel():
                continue
            image = pixels[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if args.closeup:
                text = label_at(labels[0], int(index[0]))
                cv2.putText(image, text, (48, args.height - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 6,
                            cv2.LINE_AA)
                cv2.putText(image, text, (48, args.height - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 240), 2,
                            cv2.LINE_AA)
            writer.write(image)
            frames += 1
        if control_step % (5 * CONTROL_HZ) == 0:
            print(f"  {control_step / CONTROL_HZ:5.1f} s  env0: "
                  f"{label_at(labels[0], int(index[0]))}", flush=True)
    writer.release()
    print(f"{frames} frames -> {output}", flush=True)
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
