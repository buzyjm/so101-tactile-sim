"""Record ball and pad world poses through the validated grasp.

The SDF tactile model only needs geometry, so calibrating it does not need to
re-run physics for every candidate parameter set: trace the poses once, then
sweep offline.
"""

from __future__ import annotations

import argparse
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
    p.add_argument(
        "--out", type=Path,
        default=ROOT / "tactile_logs" / "grasp_pose_trace.npz")
    p.add_argument("--every", type=int, default=3,
                   help="Record one frame in N physics steps.")
    return p.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "active_gpu": ARGS.gpu,
                     "physics_gpu": ARGS.gpu, "multi_gpu": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import UsdGeom, UsdPhysics

from scripts.tasks.grasp_geometry import RadialToolGeometry
from scripts.tasks.seated_grasp import calibrate_grasp_frame, seated_grasp_frame
from tactile_taxels import load_taxel_positions_m
from scene_config import (
    GRIPPER_LINK_PRIM_PATH, PROJECT_ROOT, ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG, TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS, TARGET_BALL_RADIUS,
)

JOINT_ROOT = "/World/Robot/Physics"
BALL_PATH = "/World/TargetBall"
IK_FRAME = "gripper_frame_link"
GRIPPER_FRAME_PRIM = GRIPPER_LINK_PRIM_PATH + "/gripper_frame_link"
HZ = TACTILE_MVP_PHYSICS_FREQUENCY_HZ
OPEN_DEG, CLOSE_DEG, TILT = 70.0, 44.0, 15.0


def quintic(t: float) -> float:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def matrix_of(stage, path) -> np.ndarray:
    m = UsdGeom.Xformable(
        stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
    return np.array([[m[i][j] for j in range(4)] for i in range(4)])


def main() -> int:
    solver = LulaKinematicsSolver(
        robot_description_path=str(PROJECT_ROOT / "so101_descriptor.yaml"),
        urdf_path=str(PROJECT_ROOT.parent / "SO-ARM100" / "Simulation"
                      / "SO101" / "so101_new_calib.urdf"))
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
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / HZ,
                  rendering_dt=1.0 / HZ)
    world.reset()
    ball = RigidPrim(BALL_PATH)
    taxels = np.asarray(load_taxel_positions_m())

    ball_target = np.asarray(UsdGeom.Xformable(
        stage.GetPrimAtPath(BALL_PATH)
    ).ComputeLocalToWorldTransform(0).ExtractTranslation())
    grasp_frame, orientation = seated_grasp_frame(
        stage, geometry, ball_target, TILT, taxels, TARGET_BALL_RADIUS,
        TACTILE_ROOT_PRIM_PATHS[0], GRIPPER_LINK_PRIM_PATH, GRIPPER_FRAME_PRIM)

    def orientation_for(frame):
        angle = math.atan2(frame[1], frame[0])
        radius = float(np.linalg.norm(frame[:2]))
        virtual = np.array([radius * math.cos(angle),
                            radius * math.sin(angle), frame[2]])
        return geometry.tool_pose(virtual, TILT)["orientation"]

    def solve(frame):
        joints, ok = solver.compute_inverse_kinematics(
            frame_name=IK_FRAME, target_position=frame,
            target_orientation=orientation_for(frame), warm_start=None,
            position_tolerance=0.010, orientation_tolerance=0.3)
        if not ok:
            raise RuntimeError(f"IK failed at {np.round(frame, 4)}")
        return joints

    def settle_at(frame):
        joints = solve(frame)
        world.reset()
        ball.set_world_poses(positions=np.array([[2.0, 2.0, 0.03]]),
                             orientations=np.array([[1.0, 0, 0, 0]]))
        ball.set_velocities(np.zeros((1, 6)))
        for name, value in zip(arm, np.degrees(joints)):
            drives[name].CreateTargetPositionAttr().Set(float(value))
        drives["gripper"].CreateTargetPositionAttr().Set(OPEN_DEG)
        for _ in range(round(1.5 * HZ)):
            world.step(render=False)

    grasp_frame, residual = calibrate_grasp_frame(
        stage, grasp_frame, ball_target, taxels, TARGET_BALL_RADIUS,
        TACTILE_ROOT_PRIM_PATHS[0], settle_at)
    print(f"calibrated, residual {residual * 1000:.2f} mm")

    approach = grasp_frame + np.array([0.0, 0.0, 0.06])
    lift = grasp_frame + np.array([0.0, 0.0, 0.12])
    solutions = {"approach": solve(approach), "grasp": solve(grasp_frame),
                 "lift": solve(lift)}

    world.reset()
    for name, value in zip(arm, np.degrees(solutions["approach"])):
        drives[name].CreateTargetPositionAttr().Set(float(value))
    drives["gripper"].CreateTargetPositionAttr().Set(OPEN_DEG)
    for _ in range(round(1.0 * HZ)):
        world.step(render=False)
    ball.set_world_poses(positions=np.array([ball_target]),
                         orientations=np.array([[1.0, 0, 0, 0]]))
    ball.set_velocities(np.zeros((1, 6)))
    for _ in range(round(0.3 * HZ)):
        world.step(render=False)

    command = np.degrees(solutions["approach"])
    grip = OPEN_DEG
    balls, mats, steps = [], [], 0

    def segment(target, gripper_target, seconds):
        nonlocal command, grip, steps
        start, start_grip, goal = command.copy(), grip, np.degrees(target)
        for i in range(1, max(1, round(seconds * HZ)) + 1):
            blend = quintic(i / max(1, round(seconds * HZ)))
            command = start + blend * (goal - start)
            grip = start_grip + blend * (gripper_target - start_grip)
            for name, value in zip(arm, command):
                drives[name].CreateTargetPositionAttr().Set(float(value))
            drives["gripper"].CreateTargetPositionAttr().Set(float(grip))
            world.step(render=False)
            steps += 1
            if steps % ARGS.every == 0:
                balls.append(np.asarray(UsdGeom.Xformable(
                    stage.GetPrimAtPath(BALL_PATH)
                ).ComputeLocalToWorldTransform(0).ExtractTranslation()))
                mats.append(np.stack([matrix_of(stage, p)
                                      for p in TACTILE_ROOT_PRIM_PATHS]))

    segment(solutions["grasp"], OPEN_DEG, 1.0)     # descend
    segment(solutions["grasp"], CLOSE_DEG, 0.8)    # close
    segment(solutions["grasp"], CLOSE_DEG, 0.25)   # settle
    segment(solutions["lift"], CLOSE_DEG, 1.4)     # lift
    segment(solutions["lift"], CLOSE_DEG, 0.4)     # hold

    ARGS.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARGS.out, ball_position=np.array(balls),
                        pad_matrix=np.array(mats))
    print(f"{len(balls)} frames -> {ARGS.out}")
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
        APP.close()
    raise SystemExit(status)
