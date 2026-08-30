"""Iteratively solve the grasp pose that seats the ball on the tactile pads.

Grid-sweeping the grasp reference in the motion frame never brought the ball
closer than ~12 mm to the seated position, because the pad's own axes are not
aligned with the motion frame.  Pure geometry (the moving pad reaches the ball
at 46.4 deg, its finger only at 37.1 deg) says the seated grasp does exist, so
this closes the loop instead: measure where the ball lands in the fixed pad's
frame, convert that error through the pad's world rotation, and correct the IK
reference by it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--usd", type=Path, default=ROOT / "lab_scene_task.usda")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--gain", type=float, default=0.7)
    p.add_argument("--tilt", type=float, default=15.0)
    p.add_argument("--dx", type=float, default=0.0022)
    p.add_argument("--dy", type=float, default=0.0315)
    p.add_argument("--dz", type=float, default=0.0)
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "tactile_logs" / "grasp_seating_solution.json",
    )
    return p.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "active_gpu": ARGS.gpu,
                     "physics_gpu": ARGS.gpu, "multi_gpu": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import Gf, UsdGeom, UsdPhysics

from tactile_sensor import DPS2015EliteMvp
from tactile_taxels import load_taxel_positions_m
from scripts.tasks.grasp_geometry import RadialToolGeometry
from scene_config import (
    PROJECT_ROOT, ROBOT_BASE_POSITION, ROBOT_BASE_YAW_DEG, TABLE_TOP_Z,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ, TACTILE_ROOT_PRIM_PATHS,
    TARGET_BALL_RADIUS, TARGET_BALL_XY_TASK_READY_PLACEHOLDER,
)

JOINT_ROOT = "/World/Robot/Physics"
BALL_PATH = "/World/TargetBall"
IK_FRAME = "gripper_frame_link"
HZ = TACTILE_MVP_PHYSICS_FREQUENCY_HZ
OPEN_DEG, CLOSED_DEG = 70.0, 5.0
BALL_START = np.array([TARGET_BALL_XY_TASK_READY_PLACEHOLDER[0],
                       TARGET_BALL_XY_TASK_READY_PLACEHOLDER[1],
                       TABLE_TOP_Z + TARGET_BALL_RADIUS])
TAXELS = np.asarray(load_taxel_positions_m())


def seated_ball_centre_in_pad_frame() -> np.ndarray:
    """Ball centre when a ball of this radius rests on the pad's centroid."""
    centroid = TAXELS.mean(axis=0)
    for offset in np.linspace(0.020, 0.050, 3001):
        candidate = centroid + np.array([0.0, 0.0, offset])
        gap = np.linalg.norm(TAXELS - candidate, axis=1).min()
        if abs(gap - TARGET_BALL_RADIUS) < 2.0e-5:
            return candidate
    raise RuntimeError("Could not solve the seated ball centre")


IDEAL_PAD = seated_ball_centre_in_pad_frame()


def quintic(t: float) -> float:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def main() -> int:
    solver = LulaKinematicsSolver(
        robot_description_path=str(PROJECT_ROOT / "so101_descriptor.yaml"),
        urdf_path=str(PROJECT_ROOT.parent / "SO-ARM100" / "Simulation"
                      / "SO101" / "so101_new_calib.urdf"),
    )
    yaw = math.radians(ROBOT_BASE_YAW_DEG)
    solver.set_robot_base_pose(
        np.asarray(ROBOT_BASE_POSITION),
        np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
    arm = solver.get_joint_names()
    geometry = RadialToolGeometry(ROBOT_BASE_POSITION[:2])

    omni.usd.get_context().open_stage(str(ARGS.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    drives = {n: UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath(f"{JOINT_ROOT}/{n}"), "angular")
        for n in arm + ["gripper"]}
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / HZ)
    world.reset()
    ball = RigidPrim(BALL_PATH)
    tactile = DPS2015EliteMvp(stage)

    motion_yaw = geometry.motion_yaw(BALL_START)
    offset = np.array([ARGS.dx, ARGS.dy, ARGS.dz])
    ik_object = BALL_START + geometry.motion_rotation(motion_yaw) @ offset

    print(f"seated ball centre in pad frame: {np.round(IDEAL_PAD*1000,2)} mm")
    history = []
    for iteration in range(ARGS.iterations):
        result = trial(stage, world, ball, tactile, solver, geometry,
                       drives, arm, ik_object)
        if result is None:
            print(f"[{iteration}] IK failed at {np.round(ik_object,4)}")
            break
        err_pad, pad_rotation, measured, movnear, lift, peak = result
        history.append({
            "iteration": iteration,
            "ik_object_m": ik_object.tolist(),
            "ball_in_fixed_pad_mm": (measured * 1000).tolist(),
            "seating_error_mm": float(np.linalg.norm(err_pad) * 1000),
            "moving_pad_nearest_mm": movnear,
            "lift_delta_mm": lift * 1000,
            "peak_raw_normal_n": peak,
        })
        print(f"[{iteration}] ball_in_pad={np.round(measured*1000,2)} mm  "
              f"err={np.linalg.norm(err_pad)*1000:6.2f} mm  "
              f"movnear={('%.2f' % movnear) if movnear is not None else 'n/a'}  "
              f"lift={lift*1000:7.1f} mm  peak={peak:.1f} N", flush=True)
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        ARGS.output.write_text(json.dumps(
            {"ideal_pad_frame_mm": (IDEAL_PAD * 1000).tolist(),
             "history": history}, indent=2) + "\n", encoding="utf-8")
        if np.linalg.norm(err_pad) < 0.002:
            print("converged")
            break
        # The pad rides on the gripper, so moving the tool by -delta moves the
        # ball by +delta in the pad frame.
        ik_object = ik_object - ARGS.gain * (pad_rotation @ err_pad)
    return 0


def trial(stage, world, ball, tactile, solver, geometry, drives, arm, ik_object):
    pose = geometry.tool_pose(ik_object, ARGS.tilt)
    frame, orient, axis = (pose["frame_position"], pose["orientation"],
                           pose["insertion_axis"])
    solutions, warm = {}, None
    for label, position in (("grasp", frame),
                            ("pregrasp", frame - 0.055 * axis),
                            ("approach", frame - 0.075 * axis),
                            ("lift", frame + np.array([0, 0, 0.12]))):
        joints, ok = solver.compute_inverse_kinematics(
            frame_name=IK_FRAME, target_position=position,
            target_orientation=orient, warm_start=warm,
            position_tolerance=0.005, orientation_tolerance=0.3)
        if not ok:
            return None
        solutions[label] = joints
        warm = joints

    world.reset()
    ball.set_world_poses(positions=np.array([BALL_START]),
                         orientations=np.array([[1.0, 0, 0, 0]]))
    ball.set_velocities(np.zeros((1, 6)))
    command = np.degrees(solutions["approach"])
    for n, v in zip(arm, command):
        drives[n].CreateTargetPositionAttr().Set(float(v))
    drives["gripper"].CreateTargetPositionAttr().Set(OPEN_DEG)
    for _ in range(round(0.6 * HZ)):
        world.step(render=False)
    tactile.reset(world.current_time)
    grip, peak = OPEN_DEG, 0.0

    def segment(target, gripper_target, seconds):
        nonlocal command, grip, peak
        start, start_grip = command.copy(), grip
        goal = np.degrees(target)
        steps = max(1, round(seconds * HZ))
        for i in range(1, steps + 1):
            b = quintic(i / steps)
            command = start + b * (goal - start)
            grip = start_grip + b * (gripper_target - start_grip)
            for n, v in zip(arm, command):
                drives[n].CreateTargetPositionAttr().Set(float(v))
            drives["gripper"].CreateTargetPositionAttr().Set(float(grip))
            world.step(render=False)
            s = tactile.maybe_sample(world.current_time,
                                     world.current_time_step_index)
            if s is not None:
                peak = max(peak, float(np.abs(s.raw_total_forces[..., 2]).max()))

    segment(solutions["pregrasp"], OPEN_DEG, 0.4)
    segment(solutions["grasp"], OPEN_DEG, 0.4)
    segment(solutions["grasp"], CLOSED_DEG, 0.7)
    segment(solutions["grasp"], CLOSED_DEG, 0.3)

    pad = stage.GetPrimAtPath(TACTILE_ROOT_PRIM_PATHS[0])
    pad_to_world = UsdGeom.Xformable(pad).ComputeLocalToWorldTransform(0)
    rotation = np.array([[pad_to_world[i][j] for j in range(3)]
                         for i in range(3)]).T
    ball_world = np.asarray(UsdGeom.Xformable(
        stage.GetPrimAtPath(BALL_PATH)
    ).ComputeLocalToWorldTransform(0).ExtractTranslation())
    measured = np.asarray(pad_to_world.GetInverse().Transform(
        Gf.Vec3d(*ball_world)), dtype=float)
    sample = tactile.latest
    movnear = None
    if sample is not None and np.isfinite(sample.nearest_taxel_distances_m[1]):
        movnear = float(sample.nearest_taxel_distances_m[1]) * 1000

    before = ball_world[2]
    segment(solutions["lift"], CLOSED_DEG, 0.8)
    segment(solutions["lift"], CLOSED_DEG, 0.3)
    after = float(UsdGeom.Xformable(stage.GetPrimAtPath(BALL_PATH))
                  .ComputeLocalToWorldTransform(0).ExtractTranslation()[2])
    return IDEAL_PAD - measured, rotation, measured, movnear, after - before, peak


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
