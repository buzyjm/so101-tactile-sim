"""Reconcile two measurements that cannot both describe the ball surface.

The contact-report path puts the fixed pad's contacts 0.46 mm from the nearest
taxel, while the traced poses put every taxel 3.81 mm clear of the ball. Both
cannot hold if the reported points lie on the ball, so this measures where they
actually lie: distance from each reported contact to the ball centre, against
the 30 mm radius.
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

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--usd", type=Path, default=ROOT / "lab_scene_task.usda")
ARGS = ap.parse_args()

from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "active_gpu": ARGS.gpu,
                     "physics_gpu": ARGS.gpu, "multi_gpu": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.sensors.experimental.physics import ContactSensor
from pxr import PhysicsSchemaTools, UsdGeom, UsdPhysics

from scripts.tasks.grasp_geometry import RadialToolGeometry
from scripts.tasks.seated_grasp import calibrate_grasp_frame, seated_grasp_frame
from tactile_taxels import load_taxel_positions_m
from scene_config import (
    GRIPPER_LINK_PRIM_PATH, PROJECT_ROOT, ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG, TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS, TACTILE_SENSOR_PRIM_PATHS, TARGET_BALL_RADIUS,
)

JOINT_ROOT = "/World/Robot/Physics"
BALL_PATH = "/World/TargetBall"
IK_FRAME = "gripper_frame_link"
GRIPPER_FRAME_PRIM = GRIPPER_LINK_PRIM_PATH + "/gripper_frame_link"
HZ = TACTILE_MVP_PHYSICS_FREQUENCY_HZ
OPEN_DEG, CLOSE_DEG, TILT = 70.0, 44.0, 15.0


def quintic(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5


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
            raise RuntimeError("IK failed")
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
    solutions = {"approach": solve(approach), "grasp": solve(grasp_frame)}
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

    def segment(target, gripper_target, seconds):
        nonlocal command, grip
        start, start_grip, goal = command.copy(), grip, np.degrees(target)
        steps = max(1, round(seconds * HZ))
        for i in range(1, steps + 1):
            blend = quintic(i / steps)
            command = start + blend * (goal - start)
            grip = start_grip + blend * (gripper_target - start_grip)
            for name, value in zip(arm, command):
                drives[name].CreateTargetPositionAttr().Set(float(value))
            drives["gripper"].CreateTargetPositionAttr().Set(float(grip))
            world.step(render=False)

    segment(solutions["grasp"], OPEN_DEG, 1.0)
    segment(solutions["grasp"], CLOSE_DEG, 0.8)
    segment(solutions["grasp"], CLOSE_DEG, 0.4)

    ball_centre = np.asarray(UsdGeom.Xformable(
        stage.GetPrimAtPath(BALL_PATH)
    ).ComputeLocalToWorldTransform(0).ExtractTranslation())
    print(f"\nball centre {np.round(ball_centre, 4)}  radius "
          f"{TARGET_BALL_RADIUS * 1000:.1f} mm\n")

    sensors = [ContactSensor(p) for p in TACTILE_SENSOR_PRIM_PATHS]
    for index, sensor in enumerate(sensors):
        name = "fixed" if index == 0 else "moving"
        reading = sensor.get_sensor_reading()
        raw = sensor.get_raw_data() if (
            reading.is_valid and reading.in_contact) else []
        pad = UsdGeom.Xformable(stage.GetPrimAtPath(
            TACTILE_ROOT_PRIM_PATHS[index])).ComputeLocalToWorldTransform(0)
        pad_rotation = np.array([[pad[i][j] for j in range(3)]
                                 for i in range(3)]).T
        pad_origin = np.asarray(pad.ExtractTranslation())
        taxels_world = taxels @ pad_rotation.T + pad_origin
        print(f"{name}: {len(raw)} raw contacts, sensor value "
              f"{reading.value if reading.is_valid else float('nan'):.2f}")
        for contact in raw[:6]:
            position = np.array([contact["position"]["x"],
                                 contact["position"]["y"],
                                 contact["position"]["z"]])
            to_centre = np.linalg.norm(position - ball_centre) * 1000
            to_taxel = np.linalg.norm(taxels_world - position, axis=1).min() * 1000
            body0 = str(PhysicsSchemaTools.intToSdfPath(int(contact["body0"])))
            body1 = str(PhysicsSchemaTools.intToSdfPath(int(contact["body1"])))
            partner = body1 if "TargetBall" not in body1 else body0
            print(f"   |contact - ball centre| = {to_centre:6.2f} mm   "
                  f"nearest taxel {to_taxel:6.2f} mm   partner "
                  f"{partner.split('/')[-1]}")
        print(f"   taxel array closest to ball centre: "
              f"{np.linalg.norm(taxels_world - ball_centre, axis=1).min() * 1000:.2f} mm")
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
