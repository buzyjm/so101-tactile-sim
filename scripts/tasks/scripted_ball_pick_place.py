"""Scripted SO-101 expert for the task-ready red-ball / round-bowl scene.

This is a deterministic physics-validation task, not the final learned policy.
It deliberately uses scene ground-truth poses, so no real-camera calibration is
required.  The same scene coordinates and success predicates can later be used
by the Isaac Lab manager-based task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("grasp", "pick-place"),
        default="pick-place",
        help="Stop after validating the lift, or complete the bowl placement.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "lab_scene_task.usda",
    )
    parser.add_argument(
        "--tactile-close", action="store_true",
        help="Close against the tactile reading instead of driving to a fixed "
             "angle.  The loop itself works, but the rigid-sphere grasp only "
             "holds across about two degrees, so the baseline stays open-loop "
             "until the contact model is fixed.")
    parser.add_argument(
        "--grip-force", type=float, default=None,
        help="Target grip force for the tactile closing loop, in newtons.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Draw this episode's physics parameters from the randomisation "
             "ranges. Omit to run the nominal, hand-tuned configuration.")
    parser.add_argument(
        "--no-tactile",
        action="store_true",
        help="Skip the 52-taxel tactile log (physics validation only).",
    )
    parser.add_argument(
        "--camera",
        default="",
        help=(
            "Camera prim to record from.  Empty builds a front third-person "
            "look-at camera from --eye / --target."
        ),
    )
    # Front-right three-quarter view, chosen from rendered candidates in
    # scripts/rendering/preview_camera_framings.py.  The distance is solved so
    # the robot, ball, bowl and table all sit inside the frame with margin;
    # the previous framing cropped the robot out of shot.
    parser.add_argument(
        "--eye", type=float, nargs=3, default=(0.682, 1.182, 0.719),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--target", type=float, nargs=3, default=(0.11, 0.19, 0.11),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--focal-length", type=float, default=24.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "renders",
    )
    parser.add_argument(
        "--dump-waypoints",
        type=Path,
        default=None,
        help="Write the solved joint waypoints, phase plan and reference "
             "world poses to this JSON file, for replaying the trajectory "
             "in the Isaac Lab parallel environment.",
    )
    return parser.parse_args()


ARGS = parse_args()

# Kit re-parses whatever is left in sys.argv and stalls at startup on the
# flags meant for this script, so hand it a clean argv before booting.
sys.argv = sys.argv[:1]

from isaacsim import SimulationApp


APP = SimulationApp(
    {
        "headless": True,
        "active_gpu": ARGS.gpu,
        "physics_gpu": ARGS.gpu,
        "multi_gpu": False,
        "anti_aliasing": 1,
    }
)

import carb
import cv2
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import Gf, UsdGeom, UsdPhysics

from tactile_sensor import DPS2015EliteMvp, SENSOR_NAMES
from scripts.tasks.domain_randomization import apply as apply_randomization
from scripts.tasks.domain_randomization import sample_episode
from scripts.tasks.seated_grasp import (
    calibrate_grasp_frame,
    seated_grasp_frame,
)
from tactile_taxels import load_taxel_positions_m
from scripts.tasks.grasp_geometry import (
    DEFAULT_TOOL_TILT_DEG,
    RadialToolGeometry,
)
from scene_config import (
    GRIPPER_CONTACT_PRIM_PATHS,
    GRIPPER_LINK_PRIM_PATH,
    TACTILE_NORMAL_FORCE_RANGE_N,
    TACTILE_SHEAR_FORCE_RANGE_N,
    TACTILE_OUTPUT_FREQUENCY_HZ,
    TACTILE_TAXEL_COUNT,
    BOWL_CENTER_XY_PLACEHOLDER,
    BOWL_HEIGHT,
    BOWL_OUTER_TOP_DIAMETER,
    BOWL_WALL_THICKNESS,
    PROJECT_ROOT,
    ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG,
    TABLE_TOP_Z,
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
    TARGET_BALL_RADIUS,
    TARGET_BALL_XY_TASK_READY_PLACEHOLDER,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS,
)


DESCRIPTOR_PATH = PROJECT_ROOT / "so101_descriptor.yaml"
URDF_PATH = Path(
    os.environ.get(
        "SO101_URDF_PATH",
        str(
            PROJECT_ROOT.parent
            / "SO-ARM100"
            / "Simulation"
            / "SO101"
            / "so101_new_calib.urdf"
        ),
    )
)

JOINT_ROOT = "/World/Robot/Physics"
BALL_PATH = "/World/TargetBall"
DEMO_CAMERA_PATH = "/World/Cameras/Demo"
IK_FRAME = "gripper_frame_link"

PHYSICS_HZ = TACTILE_MVP_PHYSICS_FREQUENCY_HZ
RENDER_HZ = 30.0
CAPTURE_EVERY = max(1, round(PHYSICS_HZ / RENDER_HZ))

# The shallow side-entry tilt that clears the table is validated for the
# grasp, but scripts/diagnostics/probe_bowl_release_ik.py shows it has no IK
# solution above the bowl.  Carry the ball there with a steeper, more top-down
# wrist instead, and let the task pick the first tilt that solves the whole
# transport chain rather than hard-coding one that may not.
GRASP_TILT_DEG = DEFAULT_TOOL_TILT_DEG
# Try the grasp tilt first.  Now that the ball is seated between the pads
# rather than jammed on the fingertips, rotating the wrist mid-carry throws it
# out, so prefer the tilt that needs no rotation at all.
TRANSPORT_TILT_CANDIDATES_DEG = (GRASP_TILT_DEG, 25.0, 35.0, 45.0, 55.0, 65.0)
OPEN_GRIPPER_DEG = 70.0
# Tactile-closed-loop grip.  The recorded hardware carries 7.96 +/- 2.22 N on
# the fixed pad and 11.93 +/- 3.41 N on the moving one, so aim inside that
# band; the floor is a safety stop, not the operating point.
TARGET_GRIP_FORCE_N = 5.0
CLOSE_FLOOR_DEG = 40.0
# First contact with a seated ball is at 46.4 deg.  The old 5 deg command was
# 41 deg of position error, which pinned the servo at its 3.35 N m torque limit
# and drove 109 N through a 25 N sensor.  Stopping just past contact gives a
# measured 8.5 N.
# A sphere between two pads is ejected if squeezed and slips if held too
# lightly: 42 deg (16 N) shoots the ball sideways out of the jaw, while 44 deg
# (8.5 N) seats and lifts it.  Contact begins at 46.4 deg.
CLOSED_GRIPPER_DEG = 44.0
LIFT_HEIGHT_M = 0.12
TRANSFER_HEIGHT_M = 0.15
# Clearance of the ball's underside above the bowl rim at release.  Keep this
# small so the ball drops into the bowl instead of bouncing off the rim.
RELEASE_CLEARANCE_M = 0.012
RETREAT_RISE_M = 0.06

BALL_START = np.array(
    [
        TARGET_BALL_XY_TASK_READY_PLACEHOLDER[0],
        TARGET_BALL_XY_TASK_READY_PLACEHOLDER[1],
        TABLE_TOP_Z + TARGET_BALL_RADIUS,
    ],
    dtype=float,
)
GRIPPER_FRAME_PRIM = GRIPPER_LINK_PRIM_PATH + "/gripper_frame_link"
TAXEL_POSITIONS_M = np.asarray(load_taxel_positions_m())
# Height the tool descends from.  The descent is collision-free at this tilt;
# a lateral entry shoves the ball onto the fingertips before the jaws close.
APPROACH_RISE_M = 0.06
CALIBRATION_PARK_M = np.array([2.0, 2.0, 0.03])
BOWL_CENTER = np.array(
    [BOWL_CENTER_XY_PLACEHOLDER[0], BOWL_CENTER_XY_PLACEHOLDER[1]],
    dtype=float,
)


def quintic_blend(phase: float) -> float:
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def world_position(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return np.asarray(transform.ExtractTranslation(), dtype=float)


def world_point(stage, prim_path: str, point: tuple[float, ...]) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    transformed = transform.Transform(Gf.Vec3d(*point))
    return np.asarray(transformed, dtype=float)


def world_point_to_local(
    stage, prim_path: str, point_world: np.ndarray
) -> np.ndarray:
    """Express a world point in a prim's local frame."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    local = transform.GetInverse().Transform(Gf.Vec3d(*point_world))
    return np.asarray(local, dtype=float)


def main() -> int:
    if not ARGS.usd.is_file():
        raise FileNotFoundError(
            f"Task scene not found: {ARGS.usd}. Build it with "
            "SO101_SCENE_LAYOUT=task_ready first."
        )
    if not URDF_PATH.is_file():
        raise FileNotFoundError(f"SO-101 URDF not found: {URDF_PATH}")

    base_yaw = math.radians(ROBOT_BASE_YAW_DEG)
    base_quaternion = np.array(
        [math.cos(base_yaw / 2.0), 0.0, 0.0, math.sin(base_yaw / 2.0)]
    )
    solver = LulaKinematicsSolver(
        robot_description_path=str(DESCRIPTOR_PATH),
        urdf_path=str(URDF_PATH),
    )
    solver.set_robot_base_pose(
        np.asarray(ROBOT_BASE_POSITION), base_quaternion
    )
    arm_joint_names = solver.get_joint_names()


    omni.usd.get_context().open_stage(str(ARGS.usd.resolve()))
    stage = omni.usd.get_context().get_stage()

    camera_path = ARGS.camera
    if ARGS.video:
        # Opening a USD can restore persisted renderer settings, so re-apply
        # TAA after the stage is open rather than trusting the app config.
        carb.settings.get_settings().set("/rtx/post/aa/op", 1)
        if not camera_path:
            view = Gf.Matrix4d()
            view.SetLookAt(
                Gf.Vec3d(*ARGS.eye),
                Gf.Vec3d(*ARGS.target),
                Gf.Vec3d(0.0, 0.0, 1.0),
            )
            camera = UsdGeom.Camera.Define(stage, DEMO_CAMERA_PATH)
            camera.CreateFocalLengthAttr(ARGS.focal_length)
            camera.CreateHorizontalApertureAttr(20.955)
            # Match the render aspect, otherwise the vertical field of view
            # authored in the USD does not correspond to the output image.
            camera.CreateVerticalApertureAttr(
                20.955 * ARGS.height / ARGS.width
            )
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
            UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
            camera_path = DEMO_CAMERA_PATH
        elif not stage.GetPrimAtPath(camera_path).IsValid():
            raise RuntimeError(f"Camera prim not found: {camera_path}")
    drives = {
        name: UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}"), "angular"
        )
        for name in arm_joint_names + ["gripper"]
    }
    missing_drives = [name for name, drive in drives.items() if not drive]
    if missing_drives:
        raise RuntimeError(f"Missing joint drives: {missing_drives}")

    # rendering_dt must equal physics_dt.  With the default 1/30 against a
    # 1/240 physics step, world.step(render=True) advances eight times as far
    # as render=False, so recording compressed the trajectory to roughly double
    # speed -- survivable for the old 109 N pinch, fatal for an 8.5 N grasp.
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / PHYSICS_HZ,
        rendering_dt=1.0 / PHYSICS_HZ,
    )
    world.reset()

    geometry = RadialToolGeometry(ROBOT_BASE_POSITION[:2])

    solutions: dict[str, np.ndarray] = {}

    def try_ik(
        position: np.ndarray,
        orientation: np.ndarray,
        warm_start: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        joint_positions, success = solver.compute_inverse_kinematics(
            frame_name=IK_FRAME,
            target_position=position,
            target_orientation=orientation,
            warm_start=warm_start,
            position_tolerance=0.010,
            orientation_tolerance=0.3,
        )
        return joint_positions, bool(success)

    def solve_waypoint(
        label: str,
        position: np.ndarray,
        orientation: np.ndarray,
        warm_start: np.ndarray | None,
    ) -> np.ndarray:
        joint_positions, success = try_ik(position, orientation, warm_start)
        print(
            f"[IK {label:10s}] ok={success} "
            f"frame={np.round(position, 4)}"
        )
        if not success:
            raise RuntimeError(f"IK failed for waypoint: {label}")
        solutions[label] = joint_positions
        return joint_positions


    # Solve the grasp so the ball sits on the fixed pad's centroid, then
    # calibrate that frame against the pose the simulation actually reaches:
    # the position drives leave about 9 mm of gravity droop, so forward
    # kinematics is not where the jaw ends up.
    ball_target = world_position(stage, BALL_PATH)
    episode_randomization = None
    target_grip_force_n = (
        ARGS.grip_force if ARGS.grip_force is not None else TARGET_GRIP_FORCE_N)
    if ARGS.seed is not None:
        episode_randomization = sample_episode(ARGS.seed)
        apply_randomization(
            stage, episode_randomization, BALL_PATH,
            GRIPPER_CONTACT_PRIM_PATHS, JOINT_ROOT, arm_joint_names,
        )
        target_grip_force_n = episode_randomization.target_grip_force_n
        print(f"[randomize seed={ARGS.seed}] "
              f"friction {episode_randomization.static_friction:.2f} "
              f"mass {episode_randomization.ball_mass_kg * 1000:.2f} g "
              f"maxForce {episode_randomization.joint_max_force:.2f} "
              f"offset {episode_randomization.contact_offset_m * 1000:.2f} mm "
              f"grip target {target_grip_force_n:.2f} N")

    grasp_frame, grasp_orientation = seated_grasp_frame(
        stage, geometry, ball_target, GRASP_TILT_DEG, TAXEL_POSITIONS_M,
        TARGET_BALL_RADIUS, TACTILE_ROOT_PRIM_PATHS[0],
        GRIPPER_LINK_PRIM_PATH, GRIPPER_FRAME_PRIM,
    )

    ball_prim = RigidPrim(BALL_PATH)

    def orientation_for_frame(frame: np.ndarray) -> np.ndarray:
        """Tool orientation whose yaw plane matches this frame position.

        The SO-101 has five joints, so orientation and position share one base
        yaw.  Calibration shifts the frame laterally, which moves its azimuth;
        keeping the original orientation left a 1.6 deg mismatch and the IK
        stopped converging.
        """
        yaw = math.atan2(frame[1], frame[0])
        radius = float(np.linalg.norm(frame[:2]))
        virtual = np.array([radius * math.cos(yaw), radius * math.sin(yaw),
                            frame[2]])
        return geometry.tool_pose(virtual, GRASP_TILT_DEG)["orientation"]

    def settle_at(frame: np.ndarray) -> None:
        """Drive to a candidate grasp frame, ball parked clear of the table."""
        joints, ok = try_ik(frame, orientation_for_frame(frame), None)
        if not ok:
            raise RuntimeError(
                f"IK failed while calibrating the grasp at {np.round(frame, 4)}"
            )
        world.reset()
        ball_prim.set_world_poses(
            positions=np.array([CALIBRATION_PARK_M]),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
        )
        ball_prim.set_velocities(np.zeros((1, 6)))
        for name, value in zip(arm_joint_names, np.degrees(joints)):
            drives[name].CreateTargetPositionAttr().Set(float(value))
        drives["gripper"].CreateTargetPositionAttr().Set(OPEN_GRIPPER_DEG)
        for _ in range(round(1.5 * PHYSICS_HZ)):
            world.step(render=False)

    def report_calibration(attempt, seat_world, residual):
        print(
            f"[calibrate {attempt}] measured seat {np.round(seat_world, 4)} "
            f"residual {np.round(residual * 1000, 2)} mm"
        )

    grasp_frame, seating_residual_m = calibrate_grasp_frame(
        stage, grasp_frame, ball_target, TAXEL_POSITIONS_M, TARGET_BALL_RADIUS,
        TACTILE_ROOT_PRIM_PATHS[0], settle_at, report=report_calibration,
    )
    grasp_orientation = orientation_for_frame(grasp_frame)
    approach_frame = grasp_frame + np.array([0.0, 0.0, APPROACH_RISE_M])
    lift_frame = grasp_frame + np.array([0.0, 0.0, LIFT_HEIGHT_M])
    lift_ik_object = ball_target + np.array([0.0, 0.0, LIFT_HEIGHT_M])
    warm_start = None
    for label, position in (("approach", approach_frame),
                            ("grasp", grasp_frame),
                            ("lift", lift_frame)):
        warm_start = solve_waypoint(
            label, position, grasp_orientation, warm_start
        )
    lift_warm_start = warm_start

    # Calibration leaves the arm sitting at the grasp pose.  Raise it to the
    # approach pose before the ball goes back, otherwise the ball is restored
    # inside the jaw and is flicked away during the settling steps.
    world.reset()
    for name, value in zip(arm_joint_names, np.degrees(solutions["approach"])):
        drives[name].CreateTargetPositionAttr().Set(float(value))
    drives["gripper"].CreateTargetPositionAttr().Set(OPEN_GRIPPER_DEG)
    for _ in range(round(1.0 * PHYSICS_HZ)):
        world.step(render=False)
    ball_prim.set_world_poses(
        positions=np.array([ball_target]),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    ball_prim.set_velocities(np.zeros((1, 6)))
    for _ in range(round(0.3 * PHYSICS_HZ)):
        world.step(render=False)

    rgb = None
    frames: list[np.ndarray] = []
    if ARGS.video:
        render_product = rep.create.render_product(
            camera_path, (ARGS.width, ARGS.height)
        )
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([render_product])

    for _ in range(round(0.5 * PHYSICS_HZ)):
        world.step(render=ARGS.video)

    # The device model integrates PhysX contacts every physics step and emits
    # on its own clock, so the log carries the real sensor cadence rather than
    # the control rate.  Contacts are not filtered to the ball: the pads report
    # whatever they actually touch, table and bowl included.
    tactile = None if ARGS.no_tactile else DPS2015EliteMvp(stage)
    if tactile is not None:
        tactile.reset(world.current_time)
    tactile_timestamps: list[float] = []
    tactile_steps: list[int] = []
    tactile_phase_ids: list[int] = []
    tactile_total_forces: list[np.ndarray] = []
    tactile_taxel_forces: list[np.ndarray] = []
    tactile_contacts: list[np.ndarray] = []
    # The device model clips to the DP-S2015 range, so keep the pre-clip PhysX
    # forces too; without them a saturated channel is indistinguishable from a
    # correct one sitting at full scale.
    tactile_raw_total_forces: list[np.ndarray] = []
    tactile_raw_taxel_forces: list[np.ndarray] = []
    tactile_backend_magnitudes: list[np.ndarray] = []
    tactile_contact_counts: list[np.ndarray] = []
    tactile_rejected_counts: list[np.ndarray] = []
    tactile_nearest_taxel_m: list[np.ndarray] = []
    phase_names: list[str] = ["start"]
    active_phase_id = 0

    arm_command = np.array(
        [
            float(drives[name].GetTargetPositionAttr().Get() or 0.0)
            for name in arm_joint_names
        ]
    )
    gripper_command = float(
        drives["gripper"].GetTargetPositionAttr().Get() or 0.0
    )
    phase_log: list[dict[str, object]] = []
    global_step = 0

    def step_once() -> None:
        nonlocal global_step
        global_step += 1
        capture = ARGS.video and global_step % CAPTURE_EVERY == 0
        world.step(render=capture)
        if tactile is not None:
            sample = tactile.maybe_sample(
                world.current_time, world.current_time_step_index
            )
            if sample is not None:
                tactile_timestamps.append(float(sample.timestamp))
                tactile_steps.append(int(sample.physics_step))
                tactile_phase_ids.append(active_phase_id)
                tactile_total_forces.append(sample.total_forces)
                tactile_taxel_forces.append(sample.taxel_forces)
                tactile_contacts.append(sample.contacts)
                tactile_raw_total_forces.append(sample.raw_total_forces)
                tactile_raw_taxel_forces.append(sample.raw_taxel_forces)
                tactile_backend_magnitudes.append(
                    sample.backend_force_magnitudes
                )
                tactile_contact_counts.append(sample.number_of_contacts)
                tactile_rejected_counts.append(sample.rejected_contact_counts)
                tactile_nearest_taxel_m.append(
                    sample.nearest_taxel_distances_m
                )
        if capture and rgb is not None:
            image = rgb.get_data()
            if image is not None and image.size:
                frames.append(image[:, :, :3].copy())

    def log_phase(label: str) -> None:
        position = world_position(stage, BALL_PATH)
        fixed_apex = world_point(
            stage,
            TACTILE_ROOT_PRIM_PATHS[0],
            TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
        )
        moving_apex = world_point(
            stage,
            TACTILE_ROOT_PRIM_PATHS[1],
            TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
        )
        aperture_midpoint = 0.5 * (fixed_apex + moving_apex)
        # Where the ball sits in each pad's own frame.  Compared against the
        # taxel array extent this says whether a rejected contact is a frame
        # error or the ball genuinely touching off-pad.
        ball_in_pad_frames = [
            world_point_to_local(stage, root, position).tolist()
            for root in TACTILE_ROOT_PRIM_PATHS
        ]
        tactile_summary = None
        if tactile is not None and tactile.latest is not None:
            latest = tactile.latest
            tactile_summary = {
                "total_force_n": [
                    float(np.linalg.norm(force)) for force in latest.total_forces
                ],
                "active_taxels": [
                    int(np.count_nonzero(magnitudes > 0.0))
                    for magnitudes in latest.taxel_force_magnitudes
                ],
                "in_contact": [bool(value) for value in latest.contacts],
            }
        phase_log.append(
            {
                "phase": label,
                "ball_in_pad_frames_m": ball_in_pad_frames,
                "tactile": tactile_summary,
                "physics_step": global_step,
                "ball_position_m": position.tolist(),
                "fixed_tactile_apex_m": fixed_apex.tolist(),
                "moving_tactile_apex_m": moving_apex.tolist(),
                "aperture_midpoint_m": aperture_midpoint.tolist(),
                "aperture_m": float(
                    np.linalg.norm(fixed_apex - moving_apex)
                ),
            }
        )
        print(
            f"[{label:12s}] ball={np.round(position, 4)} "
            f"mid={np.round(aperture_midpoint, 4)} "
            f"gap={np.linalg.norm(fixed_apex - moving_apex):.4f}"
        )

    def begin_phase(label: str) -> None:
        nonlocal active_phase_id
        phase_names.append(label)
        active_phase_id = len(phase_names) - 1

    def move(
        label: str,
        target: np.ndarray,
        gripper_target_deg: float,
        duration_s: float,
    ) -> None:
        nonlocal arm_command, gripper_command
        begin_phase(label)
        start_arm = arm_command.copy()
        target_arm = np.degrees(np.asarray(target))
        start_gripper = gripper_command
        steps = max(1, round(duration_s * PHYSICS_HZ))
        for step_index in range(1, steps + 1):
            blend = quintic_blend(step_index / steps)
            arm_command = start_arm + blend * (target_arm - start_arm)
            gripper_command = start_gripper + blend * (
                gripper_target_deg - start_gripper
            )
            for name, value in zip(arm_joint_names, arm_command):
                drives[name].CreateTargetPositionAttr().Set(float(value))
            drives["gripper"].CreateTargetPositionAttr().Set(
                float(gripper_command)
            )
            step_once()
        log_phase(label)

    def close_until_force(label: str, target_n: float, floor_deg: float,
                          rate_deg_per_s: float = 4.0,
                          settle_s: float = 0.6) -> tuple[float, float]:
        """Squeeze until a pad reports the target force, reading the device.

        Commanding a fixed close angle makes the grasp brittle: the angle that
        holds depends on friction and mass, so a randomised episode either
        misses contact entirely or crushes.  Closing against the sensor is what
        the real controller would do, and it is also the first point at which
        tactile actually drives an action rather than just being recorded.
        """
        nonlocal gripper_command
        begin_phase(label)
        reached = 0.0
        while gripper_command > floor_deg:
            gripper_command = max(
                floor_deg, gripper_command - rate_deg_per_s / PHYSICS_HZ)
            drives["gripper"].CreateTargetPositionAttr().Set(
                float(gripper_command))
            step_once()
            if tactile is not None and tactile.latest is not None:
                pads = [float(np.linalg.norm(force))
                        for force in tactile.latest.total_forces]
                reached = max(pads)
                if reached >= target_n:
                    break
        # Stopping the command dead leaves the jaw still moving, and that kick
        # is enough to throw a sphere out of a two-point pinch during the lift.
        # Hold the angle and let the contact settle before anything else moves.
        for _ in range(round(settle_s * PHYSICS_HZ)):
            step_once()
        log_phase(label)
        print(f"[close] stopped at {gripper_command:.2f} deg with "
              f"{reached:.2f} N (target {target_n:.2f} N)")
        return gripper_command, reached

    def hold(label: str, duration_s: float) -> None:
        begin_phase(label)
        for _ in range(max(1, round(duration_s * PHYSICS_HZ))):
            step_once()
        log_phase(label)

    initial_ball_position = world_position(stage, BALL_PATH)
    start_error = float(np.linalg.norm(initial_ball_position - BALL_START))
    if start_error > 0.005:
        raise RuntimeError(
            f"Ball starts at {np.round(initial_ball_position, 4)} but the grasp "
            f"pose is tuned for {np.round(BALL_START, 4)} "
            f"(error {start_error:.4f} m).  Rebuild the scene with "
            "SO101_SCENE_LAYOUT=task_ready."
        )

    move("approach", solutions["approach"], OPEN_GRIPPER_DEG, 1.0)
    move("descend", solutions["grasp"], OPEN_GRIPPER_DEG, 1.0)
    if ARGS.tactile_close:
        close_angle, close_force = close_until_force(
            "close", target_grip_force_n, CLOSE_FLOOR_DEG)
    else:
        move("close", solutions["grasp"], CLOSED_GRIPPER_DEG, 0.8)
        close_angle, close_force = CLOSED_GRIPPER_DEG, None
    hold("grip_hold", 0.25)
    move("lift", solutions["lift"], close_angle, 1.4)
    hold("lift_hold", 0.4)

    lifted_ball_position = world_position(stage, BALL_PATH)
    lift_delta = float(lifted_ball_position[2] - initial_ball_position[2])
    lift_success = lift_delta >= 0.05
    placed_success = False

    held_offset_tool = None
    release_ball_target = None
    transport_tilt_deg = None
    final_drift_m = None
    if ARGS.mode == "pick-place" and lift_success:
        # Where the ball actually sits inside the closed gripper.  Expressed in
        # tool axes it is the rigid-body invariant, so it stays valid when the
        # transport waypoints change both approach yaw and wrist tilt.
        held_offset_tool = geometry.held_offset_tool(
            lift_ik_object, lifted_ball_position, GRASP_TILT_DEG
        )
        print(
            "[handoff] in-hand ball offset (tool frame) = "
            f"{np.round(held_offset_tool, 4)}"
        )

        transfer_ball_target = np.array(
            [
                (ball_target[0] + BOWL_CENTER[0]) / 2.0,
                (ball_target[1] + BOWL_CENTER[1]) / 2.0,
                TABLE_TOP_Z + TRANSFER_HEIGHT_M,
            ]
        )
        release_ball_target = np.array(
            [
                BOWL_CENTER[0],
                BOWL_CENTER[1],
                TABLE_TOP_Z
                + BOWL_HEIGHT
                + TARGET_BALL_RADIUS
                + RELEASE_CLEARANCE_M,
            ]
        )

        def plan_transport(tilt_deg: float):
            """Solve the whole transport chain at one wrist tilt, or fail."""
            planned: dict[str, np.ndarray] = {}
            warm = lift_warm_start
            bowl_frame = None
            bowl_orientation = None
            for label, ball_target in (
                ("transfer", transfer_ball_target),
                ("bowl_hover", release_ball_target),
            ):
                ik_object = geometry.object_for_held_target(
                    ball_target, held_offset_tool, tilt_deg
                )
                pose = geometry.tool_pose(ik_object, tilt_deg)
                warm, ok = try_ik(
                    pose["frame_position"], pose["orientation"], warm
                )
                if not ok:
                    return None, label
                planned[label] = warm
                if label == "bowl_hover":
                    bowl_frame = pose["frame_position"]
                    bowl_orientation = pose["orientation"]
            # Retreat straight up out of the bowl so the fingers clear the rim.
            retreat_joints, ok = try_ik(
                bowl_frame + np.array([0.0, 0.0, RETREAT_RISE_M]),
                bowl_orientation,
                warm,
            )
            if not ok:
                return None, "retreat"
            planned["retreat"] = retreat_joints
            return planned, None

        for candidate_tilt in TRANSPORT_TILT_CANDIDATES_DEG:
            planned, failed_at = plan_transport(candidate_tilt)
            print(
                f"[IK transport] tilt={candidate_tilt:5.1f} deg "
                + ("ok" if planned else f"failed at {failed_at}")
            )
            if planned:
                transport_tilt_deg = candidate_tilt
                solutions.update(planned)
                break
        if transport_tilt_deg is None:
            raise RuntimeError(
                "No wrist tilt in TRANSPORT_TILT_CANDIDATES_DEG reaches the "
                "bowl.  Re-run scripts/diagnostics/probe_bowl_release_ik.py "
                "and move the bowl into the mapped reachable region."
            )

        move("transfer", solutions["transfer"], close_angle, 1.6)
        move("bowl_hover", solutions["bowl_hover"], close_angle, 1.6)
        move("release", solutions["bowl_hover"], OPEN_GRIPPER_DEG, 0.7)
        hold("settle", 1.5)
        move("retreat", solutions["retreat"], OPEN_GRIPPER_DEG, 0.8)
        # The bowl has a flat inner base, so a ball that lands off-centre has
        # no restoring force and coasts.  Hold long enough for it to stop, then
        # sample again to prove it actually did.
        hold("post_retreat", 2.0)
        settled_reference = world_position(stage, BALL_PATH)
        hold("rest_check", 0.5)
        final_drift_m = float(
            np.linalg.norm(world_position(stage, BALL_PATH) - settled_reference)
        )
        print(f"[rest] drift over final 0.5 s = {final_drift_m * 1000:.2f} mm")

    def dump_waypoints_json() -> None:
        import json

        def world_matrix(prim_path: str) -> list[list[float]]:
            prim = stage.GetPrimAtPath(prim_path)
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            return [list(map(float, row)) for row in transform]

        base_link_path = "/World/Robot/Geometry/base_link"
        # The executed sequence from the move()/hold() calls above, expressed
        # so a replayer can rebuild the exact drive-target timeline: quintic
        # blend from the previous command to (arm_key, gripper_deg) over
        # duration_s, or hold the last command.
        phase_plan = [
            {"label": "approach", "arm": "approach", "gripper_deg": OPEN_GRIPPER_DEG, "duration_s": 1.0},
            {"label": "descend", "arm": "grasp", "gripper_deg": OPEN_GRIPPER_DEG, "duration_s": 1.0},
            {"label": "close", "arm": "grasp", "gripper_deg": close_angle, "duration_s": 0.8},
            {"label": "grip_hold", "arm": None, "gripper_deg": None, "duration_s": 0.25},
            {"label": "lift", "arm": "lift", "gripper_deg": close_angle, "duration_s": 1.4},
            {"label": "lift_hold", "arm": None, "gripper_deg": None, "duration_s": 0.4},
        ]
        if ARGS.mode == "pick-place" and "transfer" in solutions:
            phase_plan += [
                {"label": "transfer", "arm": "transfer", "gripper_deg": close_angle, "duration_s": 1.6},
                {"label": "bowl_hover", "arm": "bowl_hover", "gripper_deg": close_angle, "duration_s": 1.6},
                {"label": "release", "arm": "bowl_hover", "gripper_deg": OPEN_GRIPPER_DEG, "duration_s": 0.7},
                {"label": "settle", "arm": None, "gripper_deg": None, "duration_s": 1.5},
                {"label": "retreat", "arm": "retreat", "gripper_deg": OPEN_GRIPPER_DEG, "duration_s": 0.8},
                {"label": "post_retreat", "arm": None, "gripper_deg": None, "duration_s": 2.0},
            ]
        dump = {
            "arm_joint_names": list(arm_joint_names),
            "solutions_rad": {
                label: [float(v) for v in joints]
                for label, joints in solutions.items()
            },
            "open_gripper_deg": OPEN_GRIPPER_DEG,
            "close_angle_deg": float(close_angle),
            "phase_plan": phase_plan,
            "physics_hz": PHYSICS_HZ,
            "ball_start": [float(v) for v in BALL_START],
            "bowl_center": [float(v) for v in BOWL_CENTER],
            "final_ball_position": [float(v) for v in final_ball_position],
            "lift_delta_m": lift_delta,
            "base_link_world": world_matrix(base_link_path),
            "gripper_link_world": world_matrix(GRIPPER_LINK_PRIM_PATH),
            "moving_jaw_world": world_matrix(
                GRIPPER_LINK_PRIM_PATH + "/moving_jaw_so101_v1_link"),
        }
        ARGS.dump_waypoints.parent.mkdir(parents=True, exist_ok=True)
        ARGS.dump_waypoints.write_text(json.dumps(dump, indent=2))
        print(f"[dump] waypoints -> {ARGS.dump_waypoints}")
        sys.stdout.flush()

    stem = (
        "scripted_ball_grasp"
        if ARGS.mode == "grasp"
        else "scripted_ball_pick_place"
    )
    final_ball_position = world_position(stage, BALL_PATH)
    bowl_distance_xy = float(
        np.linalg.norm(final_ball_position[:2] - BOWL_CENTER)
    )
    radial_margin = (
        BOWL_OUTER_TOP_DIAMETER / 2.0
        - BOWL_WALL_THICKNESS
        - TARGET_BALL_RADIUS
    )
    placed_success = (
        bowl_distance_xy <= radial_margin
        and TABLE_TOP_Z + TARGET_BALL_RADIUS * 0.75
        <= final_ball_position[2]
        <= TABLE_TOP_Z + BOWL_HEIGHT + 2.0 * TARGET_BALL_RADIUS
    )
    task_success = lift_success and (
        ARGS.mode == "grasp" or placed_success
    )
    if ARGS.dump_waypoints is not None:
        dump_waypoints_json()

    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = ARGS.output_dir / f"{stem}.mp4"
    if ARGS.video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(RENDER_HZ),
            (ARGS.width, ARGS.height),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    tactile_path = None
    grasp_contact_fraction = None
    saturated_fraction = None
    peak_raw_normal = None
    tactile_sample_count = len(tactile_timestamps)
    if tactile is not None and tactile_sample_count:
        taxel_array = np.stack(tactile_taxel_forces).astype(np.float32)
        total_array = np.stack(tactile_total_forces).astype(np.float32)
        contact_array = np.stack(tactile_contacts)
        phase_id_array = np.asarray(tactile_phase_ids, dtype=np.int32)

        # Was the ball held by both pads for the whole carry?  Measured from
        # the close onwards, up to but not including the release.
        carry_ids = [
            index
            for index, name in enumerate(phase_names)
            if name in {"close", "grip_hold", "lift", "lift_hold",
                        "transfer", "bowl_hover"}
        ]
        carry_mask = np.isin(phase_id_array, carry_ids)
        if carry_mask.any():
            both = np.logical_and(
                contact_array[carry_mask, 0], contact_array[carry_mask, 1]
            )
            grasp_contact_fraction = float(np.count_nonzero(both) / both.size)

        tactile_path = (
            PROJECT_ROOT / "tactile_logs" / f"{stem}_tactile.npz"
        )
        tactile_path.parent.mkdir(parents=True, exist_ok=True)
        raw_total_array = np.stack(tactile_raw_total_forces).astype(np.float32)
        raw_taxel_array = np.stack(tactile_raw_taxel_forces).astype(np.float32)
        np.savez_compressed(
            tactile_path,
            timestamps=np.asarray(tactile_timestamps, dtype=np.float64),
            physics_steps=np.asarray(tactile_steps, dtype=np.int64),
            phase_ids=phase_id_array,
            taxel_forces=taxel_array,
            total_forces=total_array,
            contacts=contact_array,
            raw_total_forces=raw_total_array,
            raw_taxel_forces=raw_taxel_array,
            backend_force_magnitudes=np.stack(
                tactile_backend_magnitudes
            ).astype(np.float32),
            contact_counts=np.stack(tactile_contact_counts).astype(np.int32),
            rejected_counts=np.stack(tactile_rejected_counts).astype(np.int32),
            nearest_taxel_m=np.stack(tactile_nearest_taxel_m).astype(np.float32),
        )

        # Saturation is a modelling failure, not a reading: say so in the report.
        clipped = np.abs(raw_total_array) > np.array(
            [
                TACTILE_SHEAR_FORCE_RANGE_N[1],
                TACTILE_SHEAR_FORCE_RANGE_N[1],
                TACTILE_NORMAL_FORCE_RANGE_N[1],
            ]
        )
        saturated_fraction = float(np.count_nonzero(clipped.any(axis=-1)) / clipped[..., 0].size)
        peak_raw_normal = float(np.abs(raw_total_array[..., 2]).max())
        print(
            f"raw peak normal force {peak_raw_normal:.1f} N vs "
            f"{TACTILE_NORMAL_FORCE_RANGE_N[1]:.1f} N full scale; "
            f"{saturated_fraction:.1%} of pad readings clipped"
        )
        print(
            f"tactile: {taxel_array.shape} samples at "
            f"{TACTILE_OUTPUT_FREQUENCY_HZ:.1f} Hz -> {tactile_path}"
        )

    report = {
        "mode": ARGS.mode,
        "scene": str(ARGS.usd.resolve()),
        "uses_camera_calibration": False,
        "physics_hz": PHYSICS_HZ,
        "render_hz": RENDER_HZ if ARGS.video else None,
        "initial_ball_position_m": initial_ball_position.tolist(),
        "lifted_ball_position_m": lifted_ball_position.tolist(),
        "final_ball_position_m": final_ball_position.tolist(),
        "lift_delta_m": lift_delta,
        "bowl_center_xy_m": BOWL_CENTER.tolist(),
        "seed": ARGS.seed,
        "randomization": (
            episode_randomization.as_dict() if episode_randomization else None
        ),
        "grasp_tilt_deg": GRASP_TILT_DEG,
        "grasp_frame_m": grasp_frame.tolist(),
        "seating_residual_m": seating_residual_m,
        "target_grip_force_n": target_grip_force_n,
        "close_angle_deg": close_angle,
        "close_force_n": close_force,
        "transport_tilt_deg": transport_tilt_deg,
        "held_ball_offset_tool_frame_m": (
            held_offset_tool.tolist() if held_offset_tool is not None else None
        ),
        "release_ball_target_m": (
            release_ball_target.tolist()
            if release_ball_target is not None
            else None
        ),
        "final_bowl_distance_xy_m": bowl_distance_xy,
        "bowl_radial_success_margin_m": radial_margin,
        "tactile_sensor_names": list(SENSOR_NAMES),
        "tactile_output_hz": TACTILE_OUTPUT_FREQUENCY_HZ,
        "tactile_taxel_count": TACTILE_TAXEL_COUNT,
        "tactile_sample_count": tactile_sample_count,
        "tactile_tensor_shape": (
            [tactile_sample_count, 2, TACTILE_TAXEL_COUNT, 3]
            if tactile_sample_count
            else None
        ),
        "tactile_log": str(tactile_path) if tactile_path else None,
        "tactile_phase_names": phase_names,
        "grasp_contact_fraction": grasp_contact_fraction,
        "tactile_saturated_fraction": saturated_fraction,
        "tactile_peak_raw_normal_force_n": peak_raw_normal,
        "tactile_normal_full_scale_n": TACTILE_NORMAL_FORCE_RANGE_N[1],
        "final_drift_m": final_drift_m,
        "ball_at_rest": (
            None if final_drift_m is None else bool(final_drift_m < 0.002)
        ),
        "camera": camera_path if ARGS.video else None,
        "lift_success": bool(lift_success),
        "placed_success": bool(placed_success),
        "task_success": bool(task_success),
        "phase_log": phase_log,
        "video": str(video_path.resolve()) if ARGS.video else None,
    }
    report_path = PROJECT_ROOT / "tactile_logs" / f"{stem}_result.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print(f"lift delta: {lift_delta:+.4f} m")
    print(f"bowl XY distance: {bowl_distance_xy:.4f} m")
    if grasp_contact_fraction is not None:
        print(f"both pads in contact during carry: {grasp_contact_fraction:.1%}")
    print("TASK SUCCESS" if task_success else "TASK FAILED")
    print("report:", report_path)
    if ARGS.video:
        print("video:", video_path, f"({len(frames)} frames)")
    print("=" * 78)
    return 0 if task_success else 2


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        # Print before closing the app: Isaac's shutdown can truncate stdout,
        # which hid the real error behind a bare "Simulation App Shutting Down".
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        status = 1
    finally:
        APP.close()
    raise SystemExit(status)
