import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import itertools
import numpy as np

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from isaacsim.core.api import World
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import UsdGeom, UsdPhysics


USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
DESCRIPTOR = str(PROJECT_ROOT / "so101_descriptor.yaml")
URDF = os.environ.get(
    "SO101_URDF_PATH",
    str(PROJECT_ROOT.parent / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
)
JOINT_ROOT = "/World/Robot/Physics"
CUBE_PATH = "/World/TargetCube"

BASE_POS = np.array([-0.20, 0.0, 0.75])
BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
TCP_OFFSET = 0.025

TILTS_DEG = [0.0, 15.0, 30.0, 45.0]
LATERAL_OFFSETS = [0.015, 0.025, 0.035, 0.045]
HEIGHT_OFFSETS = [0.005, 0.015, 0.025]


def rot_to_quat(rotation):
    trace = np.trace(rotation)
    if trace > 0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 / scale,
                (rotation[2, 1] - rotation[1, 2]) * scale,
                (rotation[0, 2] - rotation[2, 0]) * scale,
                (rotation[1, 0] - rotation[0, 1]) * scale,
            ]
        )
    index = int(np.argmax(np.diag(rotation)))
    if index == 0:
        scale = 2.0 * np.sqrt(
            1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
        )
        return np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ]
        )
    if index == 1:
        scale = 2.0 * np.sqrt(
            1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
        )
        return np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ]
        )
    scale = 2.0 * np.sqrt(
        1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
    )
    return np.array(
        [
            (rotation[1, 0] - rotation[0, 1]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
        ]
    )


def orientation_for_tilt(tilt_deg):
    tilt = np.radians(tilt_deg)
    tool_z = np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
    tool_x = np.cross(np.array([0.0, 1.0, 0.0]), tool_z)
    tool_x /= np.linalg.norm(tool_x)
    tool_y = np.cross(tool_z, tool_x)
    base_rotation = np.column_stack([tool_x, tool_y, tool_z])
    roll_90 = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return tool_z, rot_to_quat(base_rotation @ roll_90)


def world_position(stage, path):
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
    return np.asarray(transform.ExtractTranslation())


solver = LulaKinematicsSolver(
    robot_description_path=DESCRIPTOR,
    urdf_path=URDF,
)
solver.set_robot_base_pose(BASE_POS, BASE_QUAT)
arm_joints = solver.get_joint_names()

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)
world.reset()

drives = {
    name: UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}"), "angular"
    )
    for name in arm_joints + ["gripper"]
}


def set_arm(joint_positions):
    for name, value in zip(arm_joints, np.degrees(joint_positions)):
        drives[name].CreateTargetPositionAttr().Set(float(value))


def step(count):
    for _ in range(count):
        world.step(render=False)


def reset_trial():
    for name in arm_joints:
        drives[name].CreateTargetPositionAttr().Set(0.0)
    drives["gripper"].CreateTargetPositionAttr().Set(70.0)
    world.reset()
    step(90)


results = []
print("=" * 96)
print(
    f"{'tilt':>5} {'y_off':>7} {'z_off':>7} {'IK':>4} "
    f"{'start xyz':>25} {'closed xyz':>25} {'lift xyz':>25} {'rise':>8}"
)
print("=" * 96)

for tilt_deg, lateral_offset, height_offset in itertools.product(
    TILTS_DEG, LATERAL_OFFSETS, HEIGHT_OFFSETS
):
    reset_trial()
    cube_start = world_position(stage, CUBE_PATH)
    tool_z, orientation = orientation_for_tilt(tilt_deg)

    grasp_point = cube_start + np.array([0.0, lateral_offset, height_offset])
    grasp_target = grasp_point + TCP_OFFSET * tool_z
    lift_target = grasp_target + np.array([0.0, 0.0, 0.12])

    grasp_q, grasp_ok = solver.compute_inverse_kinematics(
        frame_name="gripper_frame_link",
        target_position=grasp_target,
        target_orientation=orientation,
        position_tolerance=0.005,
        orientation_tolerance=0.3,
    )
    lift_q, lift_ok = solver.compute_inverse_kinematics(
        frame_name="gripper_frame_link",
        target_position=lift_target,
        target_orientation=orientation,
        warm_start=grasp_q if grasp_ok else None,
        position_tolerance=0.005,
        orientation_tolerance=0.3,
    )
    if not (grasp_ok and lift_ok):
        print(
            f"{tilt_deg:5.0f} {lateral_offset:7.3f} {height_offset:7.3f} "
            f"{str(False):>4}"
        )
        continue

    set_arm(grasp_q)
    step(150)
    drives["gripper"].CreateTargetPositionAttr().Set(5.0)
    step(120)
    cube_closed = world_position(stage, CUBE_PATH)
    set_arm(lift_q)
    step(180)
    cube_lift = world_position(stage, CUBE_PATH)
    rise = cube_lift[2] - cube_start[2]
    results.append((rise, tilt_deg, lateral_offset, height_offset, cube_lift.copy()))
    print(
        f"{tilt_deg:5.0f} {lateral_offset:7.3f} {height_offset:7.3f} "
        f"{str(True):>4} {str(np.round(cube_start, 3)):>25} "
        f"{str(np.round(cube_closed, 3)):>25} "
        f"{str(np.round(cube_lift, 3)):>25} {rise:8.4f}"
    )

print("=" * 96)
print("best candidates:")
for rise, tilt_deg, lateral_offset, height_offset, cube_lift in sorted(
    results, reverse=True
)[:10]:
    print(
        f"  rise={rise:+.4f} tilt={tilt_deg:.0f} "
        f"y_off={lateral_offset:.3f} z_off={height_offset:.3f} "
        f"cube={np.round(cube_lift, 4)}"
    )
print("=" * 96)

app.close()
