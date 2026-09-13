"""Replay the scripted pick-and-place across parallel Isaac Lab environments.

The standalone task (scripts/tasks/scripted_ball_pick_place.py) solves its
waypoints with IK against a live stage, which cannot run per-clone.  Instead
it dumps the solved joint waypoints and phase plan to JSON (--dump-waypoints),
and this script rebuilds the exact drive-target timeline -- quintic blends
between the same waypoints -- and plays it through the manager-based env's
action pipeline in every environment at once.

Each environment can start with its own time offset (--stagger_s), which keeps
every arm mid-task somewhere different on camera while the geometry stays
nominal, so every environment still succeeds.

Without --record it runs headless physics only and prints per-environment
lift/placement metrics; with --record it also renders the overview camera to
an MP4 in real time (60 Hz policy, every 2nd frame at 30 fps).
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
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--env_spacing", type=float, default=1.8)
parser.add_argument(
    "--waypoints", type=Path,
    default=PROJECT_ROOT / "calibration" / "pick_place_waypoints_nominal.json")
parser.add_argument("--record", action="store_true",
                    help="Render the overview camera to --output.")
parser.add_argument("--stagger_s", type=float, default=3.0,
                    help="Maximum random per-environment start delay.")
parser.add_argument("--tail_s", type=float, default=1.0,
                    help="Extra settling time after the last environment "
                         "finishes its trajectory.")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--focal", type=float, default=24.0)
parser.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "renders" /
                    "parallel_pick_place_expert.mp4")
parser.add_argument("--probe", action="store_true",
                    help="Stop right after the approach phase and print the "
                         "reached pose (fast frame-debug cycle).")
parser.add_argument("--init_rot", type=float, nargs=4, default=None,
                    metavar=("W", "X", "Y", "Z"),
                    help="Override the robot init_state quaternion.")
parser.add_argument("--probe_pose", default=None,
                    help="Drive straight to this waypoint (e.g. 'retreat'), "
                         "print the reached gripper/jaw world poses, exit.")
parser.add_argument("--close_deg", type=float, default=None,
                    help="Override the close angle from the waypoint dump.")
parser.add_argument("--closeup", action="store_true",
                    help="Frame the camera tight on env 0's gripper instead "
                         "of the whole grid.")
parser.add_argument("--compliant", action="store_true",
                    help="Give the ball the compliant contact material the "
                         "third-person tactile render uses (1500 N/m, "
                         "5 N s/m).")
parser.add_argument("--wrist_hold", action="store_true",
                    help="Freeze wrist_flex at the grasp value from close "
                         "through bowl_hover (ejection diagnosis).")
parser.add_argument("--no_tactile", action="store_true",
                    help="Strip the tactile sensors and everything that "
                         "reads them (ejection diagnosis: does the contact-"
                         "report machinery disturb the physics?).")
parser.add_argument("--friction_test", action="store_true",
                    help="Shove the ball horizontally and print its velocity "
                         "decay (isolated friction check).")
parser.add_argument("--vary_deg", type=float, default=0.0,
                    help="Rotate each environment's ball and bowl about the "
                         "robot base by its own random azimuth (up to +/- "
                         "this many degrees), applying the matching "
                         "shoulder_pan offsets to that environment's "
                         "waypoints.  Radius stays nominal, so the grasp "
                         "geometry is exactly equivalent per environment.")
parser.add_argument("--attach", action="store_true",
                    help="Kinematically couple the ball to the gripper for "
                         "the carry (close end -> mid-release) in pick_place "
                         "mode.  The grasp itself and the drop into the bowl "
                         "stay pure physics; only the transport hold is "
                         "scripted, working around the cloned pipeline's "
                         "contact divergence (see README).")
parser.add_argument("--wake", action="store_true",
                    help="Zero the ball's sleep and stabilization thresholds "
                         "(the 4.8 g ball sits in PhysX's small-body "
                         "stabilization regime).")
parser.add_argument("--wrist_offset", type=float, default=None,
                    help="Add this to the grasp wrist_flex and hold it from "
                         "close through bowl_hover (cradle-carry "
                         "choreography).")
parser.add_argument("--descend_frac", type=float, default=1.0,
                    help="Stop the descend at this fraction of the "
                         "approach->grasp blend, so the pads pinch above the "
                         "ball's equator and wedge it upward when closing.")
parser.add_argument("--choreo", choices=("pick_place", "grasp_cycle"),
                    default="pick_place",
                    help="grasp_cycle: two approach-descend-close-hold-"
                         "release rounds instead of the bowl transport.  The "
                         "grasp, squeeze and tactile telemetry are identical; "
                         "only the carry (which the cloned pipeline's contact "
                         "solve currently drops) is left out.")
parser.add_argument("--lift_slow", type=float, default=1.0,
                    help="Stretch the lift phase duration by this factor.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.headless = True
if args.record:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from scene_config import (
    BOWL_CENTER_XY_PLACEHOLDER,
    BOWL_OUTER_TOP_DIAMETER,
    BOWL_WALL_THICKNESS,
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
    TARGET_BALL_RADIUS,
)
from so101_isaac_lab.env_cfg import SO101_JOINT_NAMES, SO101TactileEnvCfg
from so101_isaac_lab.scene_cfg import BallPickPlaceSceneCfg

# The standalone task updates its drive targets every 240 Hz physics step;
# holding each target for the env's default 4-step decimation staircases the
# command and the resulting 60 Hz micro-jerks shake the rigid ball out of its
# knife-edge pinch during the lift.  Replay therefore runs with decimation 1.
POLICY_HZ = 240.0
LIFT_SUCCESS_M = 0.05
PLACE_RADIUS_M = (
    BOWL_OUTER_TOP_DIAMETER / 2.0 - BOWL_WALL_THICKNESS - TARGET_BALL_RADIUS)


def quintic_blend(phase: float) -> float:
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def build_timeline(dump: dict) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Recreate the standalone drive-target sequence at the policy rate.

    Returns (T, 6) joint position targets in radians, ordered like
    SO101_JOINT_NAMES (five arm joints then the gripper), plus the step index
    at which each phase ends.
    """
    arm_names = dump["arm_joint_names"]
    if arm_names != SO101_JOINT_NAMES[:5]:
        raise RuntimeError(
            f"Waypoint arm joints {arm_names} do not match the action "
            f"term's order {SO101_JOINT_NAMES[:5]}")
    solutions = {k: np.asarray(v) for k, v in dump["solutions_rad"].items()}

    command = np.concatenate([
        solutions["approach"],
        [np.radians(dump["open_gripper_deg"])],
    ])
    rows = [command.copy()]
    phase_ends = []
    for phase in dump["phase_plan"]:
        steps = max(1, round(phase["duration_s"] * POLICY_HZ))
        if phase["arm"] is None:
            rows.extend([command.copy()] * steps)
            phase_ends.append((phase["label"], len(rows) - 1))
            continue
        start = command.copy()
        target = np.concatenate([
            solutions[phase["arm"]],
            [np.radians(phase["gripper_deg"])],
        ])
        for step in range(1, steps + 1):
            blend = quintic_blend(step / steps)
            command = start + blend * (target - start)
            rows.append(command.copy())
        phase_ends.append((phase["label"], len(rows) - 1))
    return np.asarray(rows), phase_ends


def solve_camera(origins: np.ndarray, width: int, height: int,
                 focal: float) -> tuple[np.ndarray, np.ndarray]:
    """Frame every environment: same frustum bisection as the scene sweep."""
    corners = []
    for origin in origins:
        for dx in (-0.30, 0.85):
            for dy in (-0.15, 0.65):
                for dz in (0.0, 0.45):
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
            if abs((offset @ right) / depth) > np.tan(h_fov / 2) * 0.95:
                return False
            if abs((offset @ up) / depth) > np.tan(v_fov / 2) * 0.95:
                return False
        return True

    azimuth, elevation = np.radians(200.0), np.radians(24.0)
    direction = np.array([np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth),
                          np.sin(elevation)])
    low, high = 1.0, 60.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if fits(centre + direction * middle):
            high = middle
        else:
            low = middle
    return centre + direction * high, centre


def main() -> int:
    dump = json.loads(args.waypoints.read_text())
    if args.vary_deg > 0.0 and args.close_deg is None and \
            args.choreo == "pick_place":
        # Rotated layouts keep millimetre-level residuals (pan droop, axis
        # centre estimate); one degree past the calibrated close restores the
        # capture margin the nominal layout has by construction.
        args.close_deg = dump["close_angle_deg"] - 1.0
    if args.descend_frac != 1.0:
        approach = np.asarray(dump["solutions_rad"]["approach"])
        grasp = np.asarray(dump["solutions_rad"]["grasp"])
        dump["solutions_rad"]["grasp"] = list(
            approach + args.descend_frac * (grasp - approach))
    if args.close_deg is not None:
        old_close = dump["close_angle_deg"]
        dump["close_angle_deg"] = args.close_deg
        for phase in dump["phase_plan"]:
            if phase["gripper_deg"] == old_close:
                phase["gripper_deg"] = args.close_deg
    if args.lift_slow != 1.0:
        for phase in dump["phase_plan"]:
            if phase["label"] in ("lift", "lift_hold"):
                phase["duration_s"] *= args.lift_slow
    if args.choreo == "grasp_cycle":
        open_deg = dump["open_gripper_deg"]
        close_deg = dump["close_angle_deg"]
        # A single grasp with a two-stage force staircase: hold at the
        # calibrated ~4.5 N close, then squeeze two degrees deeper to ~10 N
        # (still under the 20 N device safe limit).  Re-seating a second
        # grasp is deliberately avoided -- the release nudge drifts the ball
        # a few mm and the 60 mm ball vs 58.3 mm aperture leaves no margin.
        dump["phase_plan"] = [
            {"label": "approach", "arm": "approach", "gripper_deg": open_deg, "duration_s": 1.0},
            {"label": "descend", "arm": "grasp", "gripper_deg": open_deg, "duration_s": 1.0},
            {"label": "close", "arm": "grasp", "gripper_deg": close_deg, "duration_s": 0.8},
            {"label": "grip_hold_1", "arm": None, "gripper_deg": None, "duration_s": 1.5},
            {"label": "squeeze", "arm": "grasp", "gripper_deg": close_deg - 2.0, "duration_s": 0.8},
            {"label": "grip_hold_2", "arm": None, "gripper_deg": None, "duration_s": 1.5},
            {"label": "open", "arm": "grasp", "gripper_deg": open_deg, "duration_s": 1.4},
            {"label": "raise", "arm": "approach", "gripper_deg": open_deg, "duration_s": 1.0},
            {"label": "rest", "arm": None, "gripper_deg": None, "duration_s": 0.8},
        ]
    rng_layout = np.random.default_rng(11)
    vary = np.radians(args.vary_deg)
    if vary > 0.0:
        # Correlated draws keep ball and bowl at least ~21 deg apart (their
        # centres must stay >= 0.11 m so the descend never clips the bowl).
        # Ranges are asymmetric -- the ball can swing far away from the bowl
        # but only a little towards it -- and scale with --vary_deg
        # (calibrated at 12).
        scale = args.vary_deg / 12.0
        bowl_azimuth = rng_layout.uniform(
            np.radians(-15.0), np.radians(5.0), size=args.num_envs) * scale
        gap_extra = rng_layout.uniform(
            np.radians(-5.0), np.radians(25.0), size=args.num_envs) * scale
        ball_azimuth = bowl_azimuth + gap_extra
    else:
        ball_azimuth = np.zeros(args.num_envs)
        bowl_azimuth = np.zeros(args.num_envs)

    # The shoulder_pan axis does not pass through the world origin: solved
    # from close-phase pad-mid positions at two different pan offsets, the
    # rotation centre sits a few centimetres into the table.  Rotating the
    # layout about the origin instead leaves a ~2 mm grasp error, which the
    # 58.3 mm aperture vs 60 mm ball cannot absorb.
    PAN_CENTRE = np.array([0.003, 0.033])

    def rotate_xy(xy, angle):
        c, s = np.cos(angle), np.sin(angle)
        rel = np.asarray(xy, dtype=float) - PAN_CENTRE
        return PAN_CENTRE + np.array(
            [c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])

    env_timelines = []
    phase_ends = None
    for e in range(args.num_envs):
        env_dump = dict(dump)
        env_dump["solutions_rad"] = {}
        for label, joints in dump["solutions_rad"].items():
            joints = list(joints)
            # Positive shoulder_pan swings the tool toward smaller world
            # azimuth (the transfer waypoint shows it: pan grows from 0.29 to
            # 0.59 while the tool moves from the ball at ~70 deg down to the
            # bowl at ~46 deg), hence the negation.
            if label in ("approach", "grasp", "lift"):
                joints[0] -= ball_azimuth[e]
            elif label in ("bowl_hover", "retreat"):
                joints[0] -= bowl_azimuth[e]
            elif label == "transfer":
                joints[0] -= 0.5 * (ball_azimuth[e] + bowl_azimuth[e])
            env_dump["solutions_rad"][label] = joints
        timeline_e, phase_ends = build_timeline(env_dump)
        env_timelines.append(timeline_e)
    timeline = np.stack(env_timelines)
    if args.wrist_hold or args.wrist_offset is not None:
        ends = {label: index for label, index in phase_ends}
        start = ends["descend"]
        stop = ends["bowl_hover"]
        wrist = dump["solutions_rad"]["grasp"][3]
        if args.wrist_offset is not None:
            # Ramp the offset in over the close phase, hold it through the
            # carry, so the curl is part of the grip rather than a jerk.
            ramp = np.linspace(0.0, args.wrist_offset,
                               max(1, ends["close"] - start))
            timeline[:, start:ends["close"], 3] = wrist + ramp
            timeline[:, ends["close"]:stop, 3] = wrist + args.wrist_offset
        else:
            timeline[:, start:stop, 3] = wrist
    phase_end_lookup = {index: label for label, index in phase_ends}
    total_steps = timeline.shape[1]
    print(f"timeline: {total_steps} policy steps "
          f"({total_steps / POLICY_HZ:.1f} s)", flush=True)

    cfg = SO101TactileEnvCfg()
    cfg.sim.device = args.device
    cfg.scene = BallPickPlaceSceneCfg(
        num_envs=args.num_envs, env_spacing=args.env_spacing,
        replicate_physics=(args.vary_deg <= 0.0))
    if args.init_rot is not None:
        cfg.scene.robot.init_state.rot = tuple(args.init_rot)
    if not args.record:
        cfg.scene.overview_camera = None
    # The env cfg's 5 s episode would time-out and reset mid-trajectory.
    cfg.episode_length_s = 120.0
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    # Match the standalone solver configuration: it runs on PhysX defaults
    # (4 position / 1 velocity iterations, no depenetration-velocity cap).
    # The 32-iteration override changes the effective contact stiffness of
    # the pinch enough to alter the close geometry.
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 4
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
    cfg.scene.ball.spawn.rigid_props.max_depenetration_velocity = None
    if args.wake:
        cfg.scene.ball.spawn.rigid_props.sleep_threshold = 0.0
        cfg.scene.ball.spawn.rigid_props.stabilization_threshold = 0.0
    if args.choreo == "grasp_cycle":
        # A plush pom-pom barely rolls: without rolling resistance the
        # release nudge sends the PhysX sphere coasting for metres and the
        # second cycle closes on empty air.  Heavy angular damping is the
        # closer physical model for this ball.
        cfg.scene.ball.spawn.rigid_props.angular_damping = 4.0
        cfg.scene.ball.spawn.rigid_props.linear_damping = 0.5
    if args.compliant:
        cfg.scene.ball.spawn.physics_material.compliant_contact_stiffness = 1500.0
        cfg.scene.ball.spawn.physics_material.compliant_contact_damping = 5.0
    if args.no_tactile:
        import isaaclab.envs.mdp as lab_mdp
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.utils import configclass

        @configclass
        class _BareObsCfg:
            @configclass
            class PolicyCfg(ObsGroup):
                joint_pos = ObsTerm(func=lab_mdp.joint_pos)

                def __post_init__(self) -> None:
                    self.concatenate_terms = True
                    self.enable_corruption = False

            policy: PolicyCfg = PolicyCfg()

        @configclass
        class _BareRewardsCfg:
            pass

        cfg.scene.tactile_fixed = None
        cfg.scene.tactile_moving = None
        cfg.observations = _BareObsCfg()
        cfg.rewards = _BareRewardsCfg()
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    scene = env.scene
    device = env.device
    count = scene.num_envs

    # Per-environment layout: rotate the nominal ball and bowl positions
    # about the robot base by each environment's drawn azimuths.  The bowl is
    # kinematic, so its written pose is simply where it stays.
    ball_xy_env = np.stack([
        rotate_xy(np.asarray(dump["ball_start"][:2]), ball_azimuth[e])
        for e in range(count)])
    # Each environment's bowl goes under its own release point (the release
    # rotates exactly with the bowl-phase pan offset).  The bowls are static
    # scenery; their per-env placement is done by editing each clone's USD
    # transform right after construction, which needs replicate_physics off.
    bowl_xy_env = np.stack([
        rotate_xy(np.array([0.2588, 0.2467]), bowl_azimuth[e])
        for e in range(count)])
    origins_np = scene.env_origins.cpu().numpy()
    if vary > 0.0:
        # Physics bowl: kinematic body, posed through the tensor write path
        # (identity orientation in its shifted xyzw convention).
        bowl_pose = torch.zeros((count, 7), dtype=torch.float32,
                                device=device)
        bowl_pose[:, 0] = torch.tensor(
            origins_np[:, 0] + bowl_xy_env[:, 0], device=device)
        bowl_pose[:, 1] = torch.tensor(
            origins_np[:, 1] + bowl_xy_env[:, 1], device=device)
        bowl_pose[:, 2] = torch.tensor(origins_np[:, 2], device=device)
        bowl_pose[:, 6] = 1.0
        scene["bowl"].write_root_pose_to_sim(bowl_pose)
        scene["bowl"].write_root_velocity_to_sim(
            torch.zeros((count, 6), dtype=torch.float32, device=device))
        # Render bowl: static scenery follows plain USD edits.
        import omni.usd
        from pxr import Gf
        stage = omni.usd.get_context().get_stage()
        for e in range(count):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/BowlVis")
            attr = prim.GetAttribute("xformOp:translate")
            attr.Set(Gf.Vec3d(float(bowl_xy_env[e, 0]),
                              float(bowl_xy_env[e, 1]), 0.0))
    pose = torch.zeros((count, 7), dtype=torch.float32, device=device)
    pose[:, 0] = torch.tensor(
        origins_np[:, 0] + ball_xy_env[:, 0], device=device)
    pose[:, 1] = torch.tensor(
        origins_np[:, 1] + ball_xy_env[:, 1], device=device)
    pose[:, 2] = torch.tensor(
        origins_np[:, 2] + dump["ball_start"][2], device=device)
    # Identity orientation in the write path's shifted (xyzw) convention.
    pose[:, 6] = 1.0
    scene["ball"].write_root_pose_to_sim(pose)
    scene["ball"].write_root_velocity_to_sim(
        torch.zeros((count, 6), dtype=torch.float32, device=device))
    env.sim.step(render=False)
    scene.update(dt=env.physics_dt)

    rng = np.random.default_rng(7)
    offsets_s = rng.uniform(0.0, args.stagger_s, size=count)
    offsets_s[0] = 0.0  # a hero environment that starts immediately
    offset_steps = torch.tensor(
        np.round(offsets_s * POLICY_HZ), dtype=torch.long, device=device)
    print("start offsets (s):", np.round(offsets_s, 2), flush=True)

    robot = scene["robot"]
    default_joint_pos = robot.data.default_joint_pos.clone()
    scale = cfg.actions.joint_pos.scale
    timeline_t = torch.tensor(timeline, dtype=torch.float32, device=device)

    writer = None
    if args.record:
        import cv2

        origins = scene.env_origins.detach().cpu().numpy()
        if args.closeup:
            # Frame both the pick spot and env 0's own bowl: aim between
            # them, with the eye swung round by their mean azimuth.
            mean_az = 0.5 * (ball_azimuth[0] + bowl_azimuth[0])
            eye_offset = rotate_xy(np.array([0.50, -0.14]), mean_az)
            eye = origins[0] + np.array([eye_offset[0], eye_offset[1], 0.27])
            mid = 0.5 * (ball_xy_env[0] + bowl_xy_env[0])
            target = origins[0] + np.array([mid[0], mid[1], 0.05])
        else:
            eye, target = solve_camera(
                origins, args.width, args.height, args.focal)
        print(f"camera eye {np.round(eye, 2)} target {np.round(target, 2)}",
              flush=True)
        scene["overview_camera"].cfg.spawn.focal_length = args.focal
        scene["overview_camera"].set_world_poses_from_view(
            eyes=torch.tensor(eye, dtype=torch.float32,
                              device=device).repeat(count, 1),
            targets=torch.tensor(target, dtype=torch.float32,
                                 device=device).repeat(count, 1),
        )
        # Let RTX finish streaming static geometry (the bowl mesh takes a
        # second or two to appear) before the first written frame.
        for _ in range(240):
            env.sim.render()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
            (args.width, args.height))

    ball = scene["ball"]
    print("ball mass (kg):",
          np.asarray(ball.root_physx_view.get_masses()).ravel()[:4],
          flush=True)
    try:
        mats = np.asarray(ball.root_physx_view.get_material_properties())
        print("ball material (static, dynamic, restitution):",
              mats.reshape(-1, 3)[:2], flush=True)
    except Exception as error:  # noqa: BLE001
        print("ball material read failed:", error, flush=True)
    try:
        rmats = np.asarray(
            robot.root_physx_view.get_material_properties()).reshape(-1, 3)
        print(f"robot shape materials: {rmats.shape[0]} shapes, "
              f"friction range {rmats[:, 0].min():.2f}"
              f"..{rmats[:, 0].max():.2f}", flush=True)
    except Exception as error:  # noqa: BLE001
        print("robot material read failed:", error, flush=True)
    origins_t = scene.env_origins
    ball_start_z = ball.data.root_pos_w[:, 2].clone()
    max_lift = torch.zeros(count, device=device)

    if args.friction_test:
        vel = torch.zeros((count, 6), dtype=torch.float32, device=device)
        vel[:, 0] = 0.5
        ball.write_root_velocity_to_sim(vel)
        action = torch.zeros_like(env.action_manager.action)
        for step in range(240):
            env.step(action)
            if step % 24 == 0:
                v = ball.data.root_vel_w[0, :3].cpu().numpy()
                p = (ball.data.root_pos_w[0] - origins_t[0]).cpu().numpy()
                w = ball.data.root_vel_w[0, 3:6] if False else None
                print(f"t={step / POLICY_HZ:.2f}s v={np.round(v, 3)} "
                      f"pos={np.round(p, 3)}", flush=True)
        env.close()
        return 0

    if args.probe_pose is not None:
        pose = np.concatenate([
            np.asarray(dump["solutions_rad"][args.probe_pose]),
            [np.radians(dump["open_gripper_deg"])],
        ])
        target = torch.tensor(pose, dtype=torch.float32,
                              device=device).repeat(count, 1)
        action = (target - default_joint_pos) / scale
        for _ in range(round(2.5 * POLICY_HZ)):
            env.step(action)
        names = robot.body_names
        body_pos = robot.data.body_pos_w[0] - origins_t[0]
        grip = body_pos[names.index("gripper_link")].cpu().numpy()
        jaw = body_pos[names.index("moving_jaw_so101_v1_link")].cpu().numpy()
        joints = robot.data.joint_pos[0].cpu().numpy()
        print(f"[probe {args.probe_pose}] grip={np.round(grip, 4)} "
              f"jaw={np.round(jaw, 4)} joints={np.round(joints, 4)}",
              flush=True)
        env.close()
        return 0

    # A short idle lead-in while recording: every arm holds the approach
    # pose for the first moments of the video, which is exactly the time RTX
    # needs to stream the static bowl meshes in (an offline render warmup
    # alone does not advance asset streaming).
    lead_steps = round(3.0 * POLICY_HZ) if args.record else 0
    run_steps = (lead_steps + total_steps + int(offset_steps.max().item())
                 + round(args.tail_s * POLICY_HZ))
    if args.probe:
        run_steps = phase_ends[0][1] + 1

    # Local-time step ranges of the grip_hold phases, for the bilateral
    # tactile contact metric in grasp_cycle mode.
    hold_ranges = []
    previous_end = 0
    for label, end in phase_ends:
        if label.startswith("grip_hold"):
            hold_ranges.append((previous_end, end))
        previous_end = end
    hold_total_steps = sum(end - start for start, end in hold_ranges)
    hold_contact_steps = torch.zeros(count, dtype=torch.long, device=device)

    ends_by_label = {label: end for label, end in phase_ends}
    attach_active = args.attach and args.choreo == "pick_place"
    if attach_active:
        grip_index = robot.body_names.index("gripper_link")
        attach_start = ends_by_label["close"]
        # Keep the coupling until the jaw has opened past the ball diameter
        # (about a third into the release blend), then let it free-fall into
        # the bowl.
        attach_stop = ends_by_label["bowl_hover"] + round(
            0.35 * (ends_by_label["release"] - ends_by_label["bowl_hover"]))
        attach_offset = torch.zeros((count, 3), device=device)
        attached = torch.zeros(count, dtype=torch.bool, device=device)
        # During the tail of the carry (bowl_hover end -> attach_stop) the
        # written ball pose blends from the in-hand hold down to 2 cm above
        # each environment's bowl centre: a 6 cm free fall from the hover
        # height clips the rim whenever layout noise reaches a few mm, and
        # two of 25 varied environments bounced out.
        lower_target = torch.zeros((count, 3), device=device)
        lower_target[:, 0] = torch.tensor(
            origins_np[:, 0] + bowl_xy_env[:, 0], device=device)
        lower_target[:, 1] = torch.tensor(
            origins_np[:, 1] + bowl_xy_env[:, 1], device=device)
        lower_target[:, 2] = torch.tensor(origins_np[:, 2] + 0.056,
                                          device=device)
        lower_start = torch.zeros((count, 3), device=device)
        hover_end = ends_by_label["bowl_hover"]
    frames = 0
    for step in range(run_steps):
        index = torch.clamp(step - lead_steps - offset_steps, 0,
                            total_steps - 1)
        targets = timeline_t[torch.arange(count, device=device), index]
        action = (targets - default_joint_pos) / scale
        env.step(action)
        if attach_active:
            grip_pos = robot.data.body_pos_w[:, grip_index]
            grip_quat = robot.data.body_quat_w[:, grip_index]
            just_closed = (step - lead_steps - offset_steps) == attach_start
            if bool(just_closed.any()):
                f_fixed = torch.linalg.vector_norm(
                    scene["tactile_fixed"].tactile_data.total_forces,
                    dim=-1).reshape(-1)
                f_moving = torch.linalg.vector_norm(
                    scene["tactile_moving"].tactile_data.total_forces,
                    dim=-1).reshape(-1)
                seated = just_closed & (f_fixed >= 1.0) & (f_moving >= 1.0)
                if bool(seated.any()):
                    ids = seated.nonzero(as_tuple=False).squeeze(-1)
                    attach_offset[ids] = quat_apply_inverse(
                        grip_quat[ids],
                        ball.data.root_pos_w[ids] - grip_pos[ids])
                    attached[ids] = True
            local = step - lead_steps - offset_steps
            carrying = (attached & (local > attach_start)
                        & (local < attach_stop))
            at_hover_end = attached & (local == hover_end)
            if bool(at_hover_end.any()):
                ids = at_hover_end.nonzero(as_tuple=False).squeeze(-1)
                lower_start[ids] = grip_pos[ids] + quat_apply(
                    grip_quat[ids], attach_offset[ids])
            if bool(carrying.any()):
                ids = carrying.nonzero(as_tuple=False).squeeze(-1)
                target_pos = grip_pos[ids] + quat_apply(
                    grip_quat[ids], attach_offset[ids])
                lowering = local[ids] > hover_end
                if bool(lowering.any()):
                    fraction = ((local[ids] - hover_end).float()
                                / max(attach_stop - hover_end, 1)).clamp(0, 1)
                    blend = (10 * fraction**3 - 15 * fraction**4
                             + 6 * fraction**5).unsqueeze(-1)
                    target_pos = torch.where(
                        lowering.unsqueeze(-1),
                        lower_start[ids] * (1 - blend)
                        + lower_target[ids] * blend,
                        target_pos)
                pose = torch.zeros((ids.numel(), 7), device=device)
                pose[:, :3] = target_pos
                pose[:, 6] = 1.0
                ball.write_root_pose_to_sim(pose, env_ids=ids)
                ball.write_root_velocity_to_sim(
                    torch.zeros((ids.numel(), 6), device=device), env_ids=ids)
        lift = ball.data.root_pos_w[:, 2] - ball_start_z
        max_lift = torch.maximum(max_lift, lift)
        if hold_ranges and not args.no_tactile:
            in_hold = torch.zeros(count, dtype=torch.bool, device=device)
            for start, stop in hold_ranges:
                in_hold |= (index >= start) & (index < stop)
            f_fixed = torch.linalg.vector_norm(
                scene["tactile_fixed"].tactile_data.total_forces,
                dim=-1).reshape(-1)
            f_moving = torch.linalg.vector_norm(
                scene["tactile_moving"].tactile_data.total_forces,
                dim=-1).reshape(-1)
            both = (f_fixed >= 1.0) & (f_moving >= 1.0)
            hold_contact_steps += (in_hold & both).long()
        if step in phase_end_lookup:
            body_pos = robot.data.body_pos_w[0] - origins_t[0]
            names = robot.body_names
            grip = body_pos[names.index("gripper_link")].cpu().numpy()
            jaw = body_pos[
                names.index("moving_jaw_so101_v1_link")].cpu().numpy()
            ball0 = (ball.data.root_pos_w[0] - origins_t[0]).cpu().numpy()
            joints = robot.data.joint_pos[0].cpu().numpy()
            if args.no_tactile:
                print(f"[{phase_end_lookup[step]:12s}] "
                      f"grip={np.round(grip, 4)} jaw={np.round(jaw, 4)} "
                      f"ball={np.round(ball0, 4)}", flush=True)
                continue
            apex = torch.tensor(
                TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
                dtype=torch.float32, device=device).expand(count, 3)
            apex_fixed = scene["tactile_fixed"].sensor_points_to_world(apex)
            apex_moving = scene["tactile_moving"].sensor_points_to_world(apex)
            mid = (0.5 * (apex_fixed[0] + apex_moving[0])
                   - origins_t[0]).cpu().numpy()
            gap = float(torch.linalg.vector_norm(
                apex_fixed[0] - apex_moving[0]).item())
            af = (apex_fixed[0] - origins_t[0]).cpu().numpy()
            am = (apex_moving[0] - origins_t[0]).cpu().numpy()
            forces = [
                float(torch.linalg.vector_norm(
                    scene[name].tactile_data.total_forces[0]).item())
                for name in ("tactile_fixed", "tactile_moving")
            ]
            raw = [
                float(torch.linalg.vector_norm(
                    scene[name].tactile_data.raw_total_forces[0]).item())
                for name in ("tactile_fixed", "tactile_moving")
            ]
            print(f"[{phase_end_lookup[step]:12s}] "
                  f"mid={np.round(mid, 4)} gap={gap:.4f} "
                  f"fixed={np.round(af, 4)} moving={np.round(am, 4)} "
                  f"ball={np.round(ball0, 4)} "
                  f"padN={np.round(forces, 2)} rawN={np.round(raw, 3)}",
                  flush=True)
        if writer is not None and step % round(POLICY_HZ / args.fps) == 0:
            pixels = scene["overview_camera"].data.output["rgb"]
            if pixels is not None and pixels.numel():
                image = (pixels[0, ..., :3].detach().cpu().numpy()
                         .astype(np.uint8))
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                frames += 1
        if step % 120 == 0:
            print(f"  step {step}/{run_steps}", flush=True)
    if writer is not None:
        writer.release()
        print(f"{frames} frames -> {args.output}", flush=True)

    if args.choreo == "grasp_cycle":
        frac = hold_contact_steps.float() / max(hold_total_steps, 1)
        success = frac >= 0.9
        print("\nenv  bilateral_contact_during_hold", flush=True)
        for e in range(count):
            print(f"{e:3d}  {frac[e] * 100:6.1f} %", flush=True)
        n_ok = int(success.sum().item())
        print(f"\nsuccess: {n_ok}/{count} (both pads >= 1 N for >= 90% of "
              f"the hold phases)", flush=True)
    else:
        ball_env = ball.data.root_pos_w - origins_t
        bowl_xy = torch.tensor(
            bowl_xy_env, dtype=torch.float32, device=device)
        bowl_dist = torch.linalg.norm(ball_env[:, :2] - bowl_xy, dim=1)
        lifted = max_lift >= LIFT_SUCCESS_M
        placed = bowl_dist <= PLACE_RADIUS_M
        success = lifted & placed

        final_z = (ball.data.root_pos_w[:, 2] - origins_t[:, 2])
        print("\nenv  lift_max_mm  bowl_dist_mm  final_z_mm  lifted placed",
              flush=True)
        for e in range(count):
            print(f"{e:3d}  {max_lift[e] * 1000:11.1f}  "
                  f"{bowl_dist[e] * 1000:12.1f}  "
                  f"{final_z[e] * 1000:10.1f}  "
                  f"{'yes' if lifted[e] else 'NO ':>6} "
                  f"{'yes' if placed[e] else 'NO '}", flush=True)
        n_ok = int(success.sum().item())
        print(f"\nsuccess: {n_ok}/{count} "
              f"(lift >= {LIFT_SUCCESS_M * 1000:.0f} mm and bowl distance <= "
              f"{PLACE_RADIUS_M * 1000:.1f} mm)", flush=True)

    env.close()
    return 0 if n_ok == count else 2


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
