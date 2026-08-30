"""Check the calibrated grasp against the joint range the hardware reaches.

The URDF lets wrist_flex run to +95 deg, but across 21 recorded episodes the
real arm never passes +79.3: commands beyond that are silently truncated. If
the calibrated grasp asks for anything in that band, it is a pose the robot
cannot reproduce, and nothing trained on it transfers.
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

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--usd", type=Path, default=ROOT / "lab_scene_task.usda")
ap.add_argument("--tracking", type=Path,
                default=ROOT / "tactile_logs" / "real_arm_tracking.json")
ARGS = ap.parse_args()

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
OPEN_DEG, TILT = 70.0, 15.0


def main() -> int:
    reached = json.loads(ARGS.tracking.read_text(encoding="utf-8"))
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
    grasp_frame, _ = seated_grasp_frame(
        stage, geometry, ball_target, TILT, taxels, TARGET_BALL_RADIUS,
        TACTILE_ROOT_PRIM_PATHS[0], GRIPPER_LINK_PRIM_PATH, GRIPPER_FRAME_PRIM)

    def orient(frame):
        angle = math.atan2(frame[1], frame[0])
        radius = float(np.linalg.norm(frame[:2]))
        virtual = np.array([radius * math.cos(angle),
                            radius * math.sin(angle), frame[2]])
        return geometry.tool_pose(virtual, TILT)["orientation"]

    def solve(frame):
        joints, ok = solver.compute_inverse_kinematics(
            frame_name=IK_FRAME, target_position=frame,
            target_orientation=orient(frame), warm_start=None,
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
    print(f"calibrated, residual {residual * 1000:.2f} mm\n")

    waypoints = {
        "approach": grasp_frame + np.array([0.0, 0.0, 0.06]),
        "grasp": grasp_frame,
        "lift": grasp_frame + np.array([0.0, 0.0, 0.12]),
    }
    print(f"{'waypoint':>10s} " + " ".join(f"{n:>13s}" for n in arm))
    solutions = {}
    for label, frame in waypoints.items():
        joints = np.degrees(solve(frame))
        solutions[label] = joints
        print(f"{label:>10s} " + " ".join(f"{v:13.2f}" for v in joints))

    print(f"\n{'joint':>14s} {'sim max':>8s} {'real max':>9s} {'verdict':>26s}")
    problems = []
    for index, name in enumerate(arm):
        simulated = max(abs(solutions[k][index]) for k in solutions)
        signed = max(solutions[k][index] for k in solutions)
        entry = reached.get(name)
        if entry is None:
            continue
        real_high = entry["state_max_deg"]
        real_low = entry["state_min_deg"]
        inside = real_low - 2.0 <= signed <= real_high + 2.0
        verdict = "within recorded range" if inside else "OUTSIDE recorded range"
        if not inside:
            problems.append((name, signed, real_low, real_high))
        print(f"{name:>14s} {signed:8.2f} {real_high:9.2f} {verdict:>26s}")

    if problems:
        print("\nposes the hardware never reached:")
        for name, value, low, high in problems:
            print(f"  {name}: grasp needs {value:+.2f} deg, "
                  f"recorded range {low:+.2f}..{high:+.2f}")
    else:
        print("\nevery joint the grasp uses stays inside the recorded range")
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
