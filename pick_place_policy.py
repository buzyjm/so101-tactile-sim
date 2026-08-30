"""Smooth scripted expert: pick the red cube and drop it into the basket."""

import os

import numpy as np

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import cv2
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import UsdGeom, UsdPhysics

from scene_config import (
    BASKET_CENTER,
    BASKET_INNER_LENGTH,
    BASKET_INNER_WIDTH,
    BASKET_WALL_HEIGHT,
    CAMERA_RESOLUTION,
    PHYSICS_FREQUENCY_HZ,
    PROJECT_ROOT,
    RENDER_FREQUENCY_HZ,
    ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG,
    TABLE_TOP_Z,
    TARGET_CUBE_CENTER,
    TARGET_CUBE_SIZE,
)


USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
DESCRIPTOR = str(PROJECT_ROOT / "so101_descriptor.yaml")
URDF = os.environ.get(
    "SO101_URDF_PATH",
    str(
        PROJECT_ROOT.parent
        / "SO-ARM100"
        / "Simulation"
        / "SO101"
        / "so101_new_calib.urdf"
    ),
)
OUT_PATH = str(PROJECT_ROOT / "renders" / "pick_place.mp4")

JOINT_ROOT = "/World/Robot/Physics"
CUBE_PATH = "/World/TargetCube"
CAMERA_PATH = "/World/Cameras/External"
FRAME = "gripper_frame_link"
RESOLUTION = CAMERA_RESOLUTION
CAPTURE_EVERY = round(PHYSICS_FREQUENCY_HZ / RENDER_FREQUENCY_HZ)

TILT_DEG = 15.0
TCP_OFFSET = 0.025
LATERAL_OFFSET = 0.015
GRASP_HEIGHT_OFFSET = 0.015
OPEN_GRIPPER_DEG = 70.0
CLOSED_GRIPPER_DEG = 5.0


def rotation_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
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

    diagonal_index = int(np.argmax(np.diag(rotation)))
    if diagonal_index == 0:
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
    if diagonal_index == 1:
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


base_yaw = np.radians(ROBOT_BASE_YAW_DEG)
base_quaternion = np.array(
    [np.cos(base_yaw / 2.0), 0.0, 0.0, np.sin(base_yaw / 2.0)]
)

tilt = np.radians(TILT_DEG)
tool_axis_base = np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
tool_x_base = np.cross(np.array([0.0, 1.0, 0.0]), tool_axis_base)
tool_x_base /= np.linalg.norm(tool_x_base)
tool_y_base = np.cross(tool_axis_base, tool_x_base)
tool_rotation_base = np.column_stack(
    [tool_x_base, tool_y_base, tool_axis_base]
)
gripper_roll = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)


def tool_pose_for_object(object_center):
    """Return a reachable radial tool pose that holds the given cube center."""
    object_center = np.asarray(object_center, dtype=float)
    base_position = np.asarray(ROBOT_BASE_POSITION)
    direction = object_center[:2] - base_position[:2]
    motion_yaw = np.arctan2(direction[1], direction[0])
    world_from_motion = np.array(
        [
            [np.cos(motion_yaw), -np.sin(motion_yaw), 0.0],
            [np.sin(motion_yaw), np.cos(motion_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tool_axis_world = world_from_motion @ tool_axis_base
    grasp_offset_world = world_from_motion @ np.array(
        [0.0, LATERAL_OFFSET, GRASP_HEIGHT_OFFSET]
    )
    frame_target = (
        object_center
        + grasp_offset_world
        + TCP_OFFSET * tool_axis_world
    )
    tool_quaternion = rotation_to_quaternion(
        world_from_motion @ tool_rotation_base @ gripper_roll
    )
    return frame_target, tool_quaternion, np.degrees(motion_yaw)


def quintic_blend(t):
    """Zero velocity and acceleration at both ends of a segment."""
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


solver = LulaKinematicsSolver(
    robot_description_path=DESCRIPTOR,
    urdf_path=URDF,
)
solver.set_robot_base_pose(
    np.array(ROBOT_BASE_POSITION),
    base_quaternion,
)
arm_joint_names = solver.get_joint_names()

source_center = np.array(TARGET_CUBE_CENTER)
basket_x, basket_y = BASKET_CENTER

# Carry the cube above the basket walls and let it settle under gravity.
object_waypoints = {
    "grasp": source_center,
    "lift": source_center + np.array([0.0, 0.0, 0.12]),
    "transfer": np.array(
        [
            (source_center[0] + basket_x) / 2.0,
            (source_center[1] + basket_y) / 2.0,
            TABLE_TOP_Z + 0.135,
        ]
    ),
    "basket_hover": np.array(
        [basket_x, basket_y, TABLE_TOP_Z + BASKET_WALL_HEIGHT + 0.055]
    ),
    "retreat": np.array(
        [
            (source_center[0] + basket_x) / 2.0,
            (source_center[1] + basket_y) / 2.0,
            TABLE_TOP_Z + 0.135,
        ]
    ),
}

solutions = {}
warm_start = None
print("=" * 78)
for label, object_center in object_waypoints.items():
    target, orientation, motion_yaw_deg = tool_pose_for_object(object_center)
    joints, success = solver.compute_inverse_kinematics(
        frame_name=FRAME,
        target_position=target,
        target_orientation=orientation,
        warm_start=warm_start,
        position_tolerance=0.005,
        orientation_tolerance=0.3,
    )
    solutions[label] = joints
    print(
        f"[{label:12s}] object={np.round(object_center, 4)} "
        f"yaw={motion_yaw_deg:5.1f} frame={np.round(target, 4)} "
        f"ok={success}"
    )
    if not success:
        app.close()
        raise RuntimeError(f"IK failed for waypoint: {label}")
    warm_start = joints
print("=" * 78)

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
drives = {
    name: UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}"),
        "angular",
    )
    for name in arm_joint_names + ["gripper"]
}

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / PHYSICS_FREQUENCY_HZ,
    rendering_dt=1.0 / RENDER_FREQUENCY_HZ,
)
world.reset()

render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])

for _ in range(90):
    world.step(render=True)

frames = []
arm_command = np.array(
    [
        float(drives[name].GetTargetPositionAttr().Get() or 0.0)
        for name in arm_joint_names
    ]
)
gripper_command = float(
    drives["gripper"].GetTargetPositionAttr().Get() or 0.0
)


def record_step(step_index):
    should_render = step_index % CAPTURE_EVERY == 0
    world.step(render=should_render)
    if should_render:
        data = rgb.get_data()
        if data is not None and data.size:
            frames.append(data[:, :, :3].copy())


def move_smooth(label, arm_target, gripper_target, steps):
    """Interpolate command targets with a C2-continuous quintic curve."""
    global arm_command, gripper_command

    start_arm = arm_command.copy()
    target_arm = np.degrees(np.asarray(arm_target))
    start_gripper = gripper_command

    for step_index in range(1, steps + 1):
        phase = step_index / steps
        blend = quintic_blend(phase)
        arm_command = start_arm + blend * (target_arm - start_arm)
        gripper_command = (
            start_gripper + blend * (gripper_target - start_gripper)
        )

        for name, value in zip(arm_joint_names, arm_command):
            drives[name].CreateTargetPositionAttr().Set(float(value))
        drives["gripper"].CreateTargetPositionAttr().Set(
            float(gripper_command)
        )
        record_step(step_index)

    print(
        f"[{label:12s}] gripper={gripper_command:5.1f} "
        f"cube={np.round(cube_position(), 4)}"
    )


def hold(label, steps):
    for step_index in range(1, steps + 1):
        record_step(step_index)
    print(f"[{label:12s}] cube={np.round(cube_position(), 4)}")


def cube_position():
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(CUBE_PATH)
    ).ComputeLocalToWorldTransform(0)
    return np.array(transform.ExtractTranslation())


# The first segment uses the calibrated direct entry that already has a stable
# contact result; every command along it is nevertheless smoothly interpolated.
move_smooth("move_to_grasp", solutions["grasp"], OPEN_GRIPPER_DEG, 210)
move_smooth("close", solutions["grasp"], CLOSED_GRIPPER_DEG, 90)
move_smooth("lift", solutions["lift"], CLOSED_GRIPPER_DEG, 150)
move_smooth("transfer", solutions["transfer"], CLOSED_GRIPPER_DEG, 150)
move_smooth(
    "basket_hover",
    solutions["basket_hover"],
    CLOSED_GRIPPER_DEG,
    150,
)
move_smooth("release", solutions["basket_hover"], OPEN_GRIPPER_DEG, 90)
hold("settle", 180)
move_smooth("retreat", solutions["retreat"], OPEN_GRIPPER_DEG, 120)

final_position = cube_position()
half_cube = TARGET_CUBE_SIZE / 2.0
inside_x = abs(final_position[0] - basket_x) < (
    BASKET_INNER_LENGTH / 2.0 - half_cube
)
inside_y = abs(final_position[1] - basket_y) < (
    BASKET_INNER_WIDTH / 2.0 - half_cube
)
inside_z = TABLE_TOP_Z < final_position[2] < (
    TABLE_TOP_Z + BASKET_WALL_HEIGHT + TARGET_CUBE_SIZE
)
success = inside_x and inside_y and inside_z

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
writer = cv2.VideoWriter(
    OUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    RESOLUTION,
)
for frame in frames:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()

print("=" * 78)
print("final cube:", np.round(final_position, 4))
print("PICK PLACE SUCCESS" if success else "PICK PLACE FAILED")
print("saved:", OUT_PATH)
print("=" * 78)

app.close()
