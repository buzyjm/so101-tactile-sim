import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import UsdGeom, UsdPhysics


USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
DESCRIPTOR = str(PROJECT_ROOT / "so101_descriptor.yaml")
URDF = os.environ.get(
    "SO101_URDF_PATH",
    str(PROJECT_ROOT.parent / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
)

JOINT_ROOT = "/World/Robot/Physics"
ROBOT_PATH = "/World/Robot/Geometry"
CUBE_PATH = "/World/TargetCube"
PREFIX = (
    "/World/Robot/Geometry/base_link/shoulder_link/upper_arm_link"
    "/lower_arm_link/wrist_link/gripper_link"
)
TCP_PATH = PREFIX + "/gripper_frame_link"
STATIC_COLLIDER = (
    PREFIX
    + "/wrist_roll_follower_so101_v1_1/wrist_roll_follower_so101_v1"
)
MOVING_COLLIDER = (
    PREFIX
    + "/moving_jaw_so101_v1_link/moving_jaw_so101_v1_1/moving_jaw_so101_v1"
)

BASE_POS = np.array([-0.20, 0.0, 0.75])
BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
CUBE_CENTER = np.array([0.15, 0.0, 0.765])
TCP_OFFSET = 0.025
TILT_DEG = 45.0


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


def world_position(stage, path):
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
    return np.asarray(transform.ExtractTranslation())


def world_bounds(cache, stage, path):
    cache.Clear()
    bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    minimum = np.asarray(bounds.GetMin())
    maximum = np.asarray(bounds.GetMax())
    return minimum, maximum


tilt = np.radians(TILT_DEG)
tool_z = np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
tool_x = np.cross(np.array([0.0, 1.0, 0.0]), tool_z)
tool_x /= np.linalg.norm(tool_x)
tool_y = np.cross(tool_z, tool_x)
base_rotation = np.column_stack([tool_x, tool_y, tool_z])
roll_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
orientation = rot_to_quat(base_rotation @ roll_90)
frame_target = CUBE_CENTER + TCP_OFFSET * tool_z

solver = LulaKinematicsSolver(
    robot_description_path=DESCRIPTOR,
    urdf_path=URDF,
)
solver.set_robot_base_pose(BASE_POS, BASE_QUAT)
arm_joints = solver.get_joint_names()
joint_positions, success = solver.compute_inverse_kinematics(
    frame_name="gripper_frame_link",
    target_position=frame_target,
    target_orientation=orientation,
    position_tolerance=0.005,
    orientation_tolerance=0.3,
)
if not success:
    raise RuntimeError("IK failed for the current grasp pose")

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
for name, value in zip(arm_joints, np.degrees(joint_positions)):
    drives[name].CreateTargetPositionAttr().Set(float(value))
drives["gripper"].CreateTargetPositionAttr().Set(70.0)
for _ in range(180):
    world.step(render=False)

cache = UsdGeom.BBoxCache(
    0,
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
    useExtentsHint=False,
)
articulation = Articulation(ROBOT_PATH)
articulation.initialize()


def report(label):
    cube_position = world_position(stage, CUBE_PATH)
    tcp_position = world_position(stage, TCP_PATH)
    print(f"\n[{label}]")
    print(f"  cube:       {np.round(cube_position, 5)}")
    print(f"  TCP:        {np.round(tcp_position, 5)}")
    print(f"  TCP error:  {np.round(tcp_position - frame_target, 5)}")
    for name, path in [
        ("static jaw", STATIC_COLLIDER),
        ("moving jaw", MOVING_COLLIDER),
    ]:
        minimum, maximum = world_bounds(cache, stage, path)
        center = (minimum + maximum) * 0.5
        print(
            f"  {name:<10} center={np.round(center, 5)} "
            f"min={np.round(minimum, 5)} max={np.round(maximum, 5)}"
        )
    actual = articulation.get_joint_positions()[0]
    print(f"  joints(deg): {np.round(np.degrees(actual), 2)}")


print("=" * 88)
print(f"frame target: {np.round(frame_target, 5)}")
print(f"IK joints(deg): {np.round(np.degrees(joint_positions), 2)}")
report("open")

drives["gripper"].CreateTargetPositionAttr().Set(5.0)
for _ in range(180):
    world.step(render=False)
report("closed")
print("=" * 88)

app.close()
