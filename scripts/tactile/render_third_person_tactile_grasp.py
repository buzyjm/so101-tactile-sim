"""Render a third-person scripted grasp with a compact live tactile HUD."""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VIDEO_GPU = int(os.environ.get("SO101_TACTILE_VIDEO_GPU", "0"))
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": VIDEO_GPU,
        "physics_gpu": VIDEO_GPU,
        "multi_gpu": False,
        "anti_aliasing": 1,
    }
)

import carb
import omni.replicator.core as rep
import omni.usd
from omni.physx.scripts import physicsUtils
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from scene_config import (
    PROJECT_ROOT,
    ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_BODY_PRIM_PATHS,
    TACTILE_ROOT_PRIM_PATHS,
    TACTILE_SENSOR_SIZE_SENSOR_FRAME,
)
from tactile_sensor import (
    DPS2015EliteMvp,
    SENSOR_NAMES,
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
BALL_PATH = "/World/TargetBall"
JOINT_ROOT = "/World/Robot/Physics"
FRAME_NAME = "gripper_frame_link"
RAW_VIDEO_PATH = PROJECT_ROOT / "renders" / "tactile_third_person_grasp_raw.mp4"
POSTER_PATH = PROJECT_ROOT / "renders" / "tactile_third_person_grasp_poster.png"
LOG_PATH = PROJECT_ROOT / "tactile_logs" / "third_person_grasp.json"
RESOLUTION = (1280, 720)
FPS = 30
STEPS_PER_FRAME = int(round(TACTILE_MVP_PHYSICS_FREQUENCY_HZ / FPS))
DURATION_S = 8.2
FRAME_COUNT = int(round(DURATION_S * FPS))
OPEN_GRIPPER_DEG = 70.0
CLOSE_GRIPPER_DEG = 35.0
CONTACT_PROBE_RADIUS = 0.004
CONTACT_CLEARANCE = 0.0010
CONTACT_PENETRATION = 0.0020
CONTACT_STIFFNESS_N_PER_M = 1500.0
CONTACT_DAMPING_N_S_PER_M = 5.0


def quintic(value):
    value = min(max(float(value), 0.0), 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def rotation_to_quaternion(rotation):
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


def solve_waypoints():
    solver = LulaKinematicsSolver(
        robot_description_path=DESCRIPTOR,
        urdf_path=URDF,
    )
    base_yaw = np.radians(ROBOT_BASE_YAW_DEG)
    solver.set_robot_base_pose(
        np.array(ROBOT_BASE_POSITION),
        np.array(
            [np.cos(base_yaw / 2.0), 0.0, 0.0, np.sin(base_yaw / 2.0)]
        ),
    )
    arm_names = solver.get_joint_names()

    tilt = np.radians(15.0)
    world_from_base = np.array(
        [
            [np.cos(base_yaw), -np.sin(base_yaw), 0.0],
            [np.sin(base_yaw), np.cos(base_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
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
    orientation = rotation_to_quaternion(
        world_from_base @ tool_rotation_base @ gripper_roll
    )
    tool_axis_world = world_from_base @ tool_axis_base

    # First find one unquestionably reachable grasp pose. The exact object
    # location is then measured from the simulated fingertips, so this search
    # only chooses a clean, visible part of the current robot workspace.
    candidate_positions = (
        np.array([0.20, 0.23, 0.105]),
        np.array([0.16, 0.27, 0.115]),
        np.array([0.10, 0.30, 0.125]),
    )
    orientation_modes = (
        ("side", orientation, 0.35),
        ("down", np.array([0.0, 1.0, 0.0, 0.0]), 0.55),
        ("position-only", None, None),
    )
    grasp_joints = None
    grasp_target = None
    selected_mode = None
    for candidate in candidate_positions:
        for mode, candidate_orientation, tolerance in orientation_modes:
            joints, success = solver.compute_inverse_kinematics(
                frame_name=FRAME_NAME,
                target_position=candidate,
                target_orientation=candidate_orientation,
                warm_start=None,
                position_tolerance=0.006,
                orientation_tolerance=tolerance,
            )
            print(
                f"IK grasp ({mode}): success={success} "
                f"target={np.round(candidate, 4)}"
            )
            if success:
                grasp_joints = np.asarray(joints)
                grasp_target = candidate
                selected_mode = mode
                break
        if grasp_joints is not None:
            break
    if grasp_joints is None:
        raise RuntimeError("IK failed for every grasp candidate")

    # Reuse the orientation actually returned for the reachable grasp. This
    # keeps approach/lift continuous even when the position-only fallback was
    # needed by the current Lula/URDF combination.
    fk_position, fk_rotation = solver.compute_forward_kinematics(
        FRAME_NAME, grasp_joints
    )
    achieved_orientation = rotation_to_quaternion(np.asarray(fk_rotation))
    achieved_tool_axis = np.asarray(fk_rotation)[:, 2]
    print(
        "selected grasp:",
        selected_mode,
        "fk_position=",
        np.round(fk_position, 4),
        "tool_z=",
        np.round(achieved_tool_axis, 4),
    )

    points = {
        "grasp": grasp_target,
        "pregrasp": grasp_target - 0.045 * achieved_tool_axis,
        "approach": grasp_target - 0.105 * achieved_tool_axis,
        "lift": grasp_target + np.array([0.0, 0.0, 0.115]),
    }
    solutions_radians = {"grasp": grasp_joints}
    for label in ("pregrasp", "approach", "lift"):
        warm_start = (
            solutions_radians["pregrasp"]
            if label == "approach"
            else solutions_radians["grasp"]
        )
        joints, success = solver.compute_inverse_kinematics(
            frame_name=FRAME_NAME,
            target_position=points[label],
            target_orientation=achieved_orientation,
            warm_start=warm_start,
            position_tolerance=0.006,
            orientation_tolerance=0.35,
        )
        print(
            f"IK {label}: success={success} "
            f"target={np.round(points[label], 4)}"
        )
        if not success:
            # A small orientation relaxation is acceptable for this visual
            # demonstration, while the warm start preserves continuity.
            joints, success = solver.compute_inverse_kinematics(
                frame_name=FRAME_NAME,
                target_position=points[label],
                target_orientation=None,
                warm_start=warm_start,
                position_tolerance=0.008,
                orientation_tolerance=None,
            )
            print(f"IK {label} position-only fallback: success={success}")
        if not success:
            raise RuntimeError(f"IK failed for {label}")
        solutions_radians[label] = np.asarray(joints)

    return arm_names, {
        label: np.degrees(joints)
        for label, joints in solutions_radians.items()
    }


def add_camera(stage):
    view = Gf.Matrix4d()
    view.SetLookAt(
        Gf.Vec3d(0.68, -0.40, 0.44),
        Gf.Vec3d(0.06, 0.27, 0.14),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    camera_path = "/World/Cameras/ThirdPersonTactileGrasp"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(25.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    return camera_path


def sensor_points_in_body_frames(stage, sensor_local_z):
    """Return sensor-local points expressed in monitored rigid-body frames."""
    points = []
    for sensor_path in TACTILE_ROOT_PRIM_PATHS:
        # Each ContactSensor prim is a direct child of its monitored body.
        sensor_local = UsdGeom.Xformable(
            stage.GetPrimAtPath(sensor_path)
        ).GetLocalTransformation()
        point_body = sensor_local.Transform(
            Gf.Vec3d(0.0, 0.0, sensor_local_z)
        )
        points.append(np.array(point_body, dtype=np.float64))
    return np.stack(points)


def quaternion_to_rotation(quaternion):
    """Convert Isaac's scalar-first quaternion into a rotation matrix."""
    w, x, y, z = [float(value) for value in quaternion]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def dynamic_body_points(sensor_bodies, points_body):
    """Read current PhysX body poses and transform two body-local points."""
    positions, orientations = sensor_bodies.get_world_poses()
    positions = np.asarray(positions)
    orientations = np.asarray(orientations)
    return np.stack(
        [
            positions[index]
            + quaternion_to_rotation(orientations[index])
            @ points_body[index]
            for index in range(2)
        ]
    )


def create_contact_probe(stage, path):
    """Create an invisible kinematic sphere that loads one tactile dome."""
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(-10.0, -10.0, -10.0))
    collision = UsdGeom.Sphere.Define(stage, path + "/Collision")
    collision.CreateRadiusAttr(CONTACT_PROBE_RADIUS)
    collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    material_path = "/World/Looks/TactileContactProbeMaterial"
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        UsdShade.Material.Define(stage, material_path)
        material_prim = stage.GetPrimAtPath(material_path)
        UsdPhysics.MaterialAPI.Apply(material_prim)
        material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        material.CreateCompliantContactStiffnessAttr().Set(
            CONTACT_STIFFNESS_N_PER_M
        )
        material.CreateCompliantContactDampingAttr().Set(
            CONTACT_DAMPING_N_S_PER_M
        )
    physicsUtils.add_physics_material_to_prim(
        stage,
        collision.GetPrim(),
        material_path,
    )
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.001)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    return path


def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


BODY_FONT = load_font(15)
SMALL_FONT = load_font(12)


def draw_screen_arrow(draw, start, end, color, width=3, head_size=8):
    """Draw a clean arrow in image coordinates."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return
    direction = delta / length
    perpendicular = np.array([-direction[1], direction[0]])
    head_base = end - direction * head_size
    head_left = head_base + perpendicular * head_size * 0.55
    head_right = head_base - perpendicular * head_size * 0.55
    draw.line([tuple(start), tuple(end)], fill=color, width=width)
    draw.polygon(
        [tuple(end), tuple(head_left), tuple(head_right)],
        fill=color,
    )


def draw_sensor_frame(draw, origin, force, force_color):
    """Draw an axonometric sensor-local XYZ frame and its force vector."""
    origin = np.asarray(origin, dtype=np.float64)
    projection = np.array(
        [[0.86, -0.76, 0.0], [0.40, 0.46, -1.0]],
        dtype=np.float64,
    )
    axes = (
        ("X", np.array([1.0, 0.0, 0.0]), (239, 74, 68, 255)),
        ("Y", np.array([0.0, 1.0, 0.0]), (61, 205, 105, 255)),
        ("Z", np.array([0.0, 0.0, 1.0]), (72, 132, 255, 255)),
    )
    for label, axis, color in axes:
        endpoint = origin + projection @ axis * 43.0
        draw_screen_arrow(draw, origin, endpoint, color, width=3, head_size=7)
        draw.text(
            tuple(endpoint + np.array([3.0, -7.0])),
            label,
            font=SMALL_FONT,
            fill=color,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 210),
        )

    draw.ellipse(
        (origin[0] - 3, origin[1] - 3, origin[0] + 3, origin[1] + 3),
        fill=(245, 245, 245, 255),
        outline=(0, 0, 0, 220),
        width=1,
    )
    force = np.asarray(force, dtype=np.float64)
    magnitude = float(np.linalg.norm(force))
    if magnitude >= 0.1:
        direction = force / magnitude
        display_length = 32.0 + 34.0 * min(magnitude / 5.0, 1.0)
        endpoint = origin + projection @ direction * display_length
        draw_screen_arrow(
            draw,
            origin,
            endpoint,
            (0, 0, 0, 210),
            width=8,
            head_size=13,
        )
        draw_screen_arrow(
            draw,
            origin,
            endpoint,
            force_color,
            width=5,
            head_size=10,
        )


def compose_frame(pixels, sample, phase, time_seconds):
    image = Image.fromarray(pixels[:, :, :3]).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    forces = sample.total_forces if sample is not None else np.zeros((2, 3))
    text_colors = ((255, 112, 42, 255), (0, 205, 255, 255))
    origins = ((1120, 125), (1120, 300))
    text_y = (25, 200)
    for index, name in enumerate(SENSOR_NAMES):
        force = np.asarray(forces[index], dtype=np.float64)
        draw.text(
            (925, text_y[index]),
            "{}  F=({:+4.1f}, {:+4.1f}, {:+4.1f}) N".format(
                name.upper(), *force
            ),
            font=BODY_FONT,
            fill=text_colors[index],
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )
        draw_sensor_frame(draw, origins[index], force, text_colors[index])
    return np.asarray(image)


def command_pose(drives, arm_names, arm_degrees, gripper_degrees):
    for name, value in zip(arm_names, arm_degrees):
        drives[name].CreateTargetPositionAttr().Set(float(value))
    drives["gripper"].CreateTargetPositionAttr().Set(float(gripper_degrees))


def phase_commands(time_seconds, solutions):
    if time_seconds < 1.5:
        blend = quintic(time_seconds / 1.5)
        arm = solutions["approach"] + blend * (
            solutions["pregrasp"] - solutions["approach"]
        )
        return "APPROACH", arm, OPEN_GRIPPER_DEG, False
    if time_seconds < 2.7:
        blend = quintic((time_seconds - 1.5) / 1.2)
        arm = solutions["pregrasp"] + blend * (
            solutions["grasp"] - solutions["pregrasp"]
        )
        return "PRE-GRASP", arm, OPEN_GRIPPER_DEG, False
    if time_seconds < 4.3:
        blend = quintic((time_seconds - 2.7) / 1.6)
        gripper = OPEN_GRIPPER_DEG + blend * (
            CLOSE_GRIPPER_DEG - OPEN_GRIPPER_DEG
        )
        return "CLOSING", solutions["grasp"], gripper, False
    if time_seconds < 4.45:
        return "GRASP", solutions["grasp"], CLOSE_GRIPPER_DEG, False
    if time_seconds < DURATION_S:
        blend = quintic((time_seconds - 4.45) / (DURATION_S - 4.45))
        arm = solutions["grasp"] + blend * (
            solutions["lift"] - solutions["grasp"]
        )
        return "LIFT", arm, CLOSE_GRIPPER_DEG, True
    return "HOLD", solutions["lift"], CLOSE_GRIPPER_DEG, True


arm_names, solutions = solve_waypoints()
omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
carb.settings.get_settings().set("/rtx/post/aa/op", 1)
# The reachable demonstration point overlaps the photo-reference bowl pose.
# Disable the bowl only in this in-memory video stage; lab_scene.usda is not saved.
bowl_prim = stage.GetPrimAtPath("/World/Bowl")
if bowl_prim.IsValid():
    bowl_prim.SetActive(False)
drives = {
    name: UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}"), "angular"
    )
    for name in arm_names + ["gripper"]
}
sensor_bodies = RigidPrim(list(TACTILE_BODY_PRIM_PATHS))
sensor_depth = TACTILE_SENSOR_SIZE_SENSOR_FRAME[2]
apex_points_body = sensor_points_in_body_frames(stage, -sensor_depth)
probe_clear_points_body = sensor_points_in_body_frames(
    stage,
    -sensor_depth - CONTACT_PROBE_RADIUS - CONTACT_CLEARANCE,
)
probe_contact_points_body = sensor_points_in_body_frames(
    stage,
    -sensor_depth - CONTACT_PROBE_RADIUS + CONTACT_PENETRATION,
)
probe_paths = (
    create_contact_probe(stage, "/World/TactileContactProbeFixed"),
    create_contact_probe(stage, "/World/TactileContactProbeMoving"),
)
contact_probes = RigidPrim(list(probe_paths))

# Decouple the visible red ball from physics-tensor rendering. Contact is
# generated by the hidden probes above; this sphere is placed explicitly in
# USD so the RTX frame and the measured fingertip pose cannot diverge.
ball_prim = stage.GetPrimAtPath(BALL_PATH)
ball_body = UsdPhysics.RigidBodyAPI.Get(stage, BALL_PATH)
ball_body.CreateRigidBodyEnabledAttr().Set(False)
ball_collision = UsdPhysics.CollisionAPI.Get(stage, BALL_PATH)
if ball_collision:
    ball_collision.CreateCollisionEnabledAttr().Set(False)
ball_xform = UsdGeom.Xformable(ball_prim)
ball_translate_ops = [
    op for op in ball_xform.GetOrderedXformOps()
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
]
if not ball_translate_ops:
    raise RuntimeError(f"{BALL_PATH} has no translate xform op")
ball_translate_op = ball_translate_ops[0]


def set_visual_ball_position(position):
    ball_translate_op.Set(Gf.Vec3d(*[float(value) for value in position]))


world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    rendering_dt=1.0 / FPS,
)
model = DPS2015EliteMvp(stage, contact_partner_paths=probe_paths)
world.reset()
contact_probes.set_world_poses(
    positions=np.full((2, 3), -10.0, dtype=np.float32)
)
set_visual_ball_position((-10.0, -10.0, -10.0))

# Pre-position at the grasp pose to measure the real aperture center, then
# return to the approach pose before recording.
command_pose(drives, arm_names, solutions["grasp"], OPEN_GRIPPER_DEG)
for _ in range(360):
    world.step(render=False)
apices = dynamic_body_points(sensor_bodies, apex_points_body)
object_center = (apices[0] + apices[1]) * 0.5
print("apex points in body frames:", np.round(apex_points_body, 4))
print("prepass dynamic apices:", np.round(apices, 4))

world.reset()
contact_probes.set_world_poses(
    positions=np.full((2, 3), -10.0, dtype=np.float32)
)
command_pose(drives, arm_names, solutions["approach"], OPEN_GRIPPER_DEG)
set_visual_ball_position(object_center)
model.reset(world.current_time)
for _ in range(300):
    world.step(render=False)
    model.maybe_sample(world.current_time, world.current_time_step_index)
model.reset(world.current_time)

camera_path = add_camera(stage)
render_product = rep.create.render_product(camera_path, RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])
for _ in range(8):
    world.step(render=True)

RAW_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(RAW_VIDEO_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    RESOLUTION,
)
if not writer.isOpened():
    app.close()
    raise RuntimeError(f"Could not open video writer: {RAW_VIDEO_PATH}")

frame_log = []
poster = None
poster_score = -1.0
for frame_index in range(FRAME_COUNT):
    time_seconds = frame_index / FPS
    phase, arm_target, gripper_target, follow_gripper = phase_commands(
        time_seconds, solutions
    )
    command_pose(drives, arm_names, arm_target, gripper_target)
    for substep in range(STEPS_PER_FRAME):
        current_apices = dynamic_body_points(sensor_bodies, apex_points_body)
        if follow_gripper:
            midpoint = (current_apices[0] + current_apices[1]) * 0.5
            set_visual_ball_position(midpoint)

        # Two small kinematic spheres stand in for the two contact patches of
        # the visible ball. They start 1 mm clear of each dome, then gently
        # load the actual tactile collision surfaces during close/grasp/lift.
        # The resulting HUD force comes from the ContactSensor raw impulses.
        if time_seconds < 3.35:
            probe_positions = np.full((2, 3), -10.0, dtype=np.float32)
        else:
            load_blend = quintic((time_seconds - 3.35) / 0.95)
            probe_points_body = (
                (1.0 - load_blend) * probe_clear_points_body
                + load_blend * probe_contact_points_body
            )
            probe_positions = dynamic_body_points(
                sensor_bodies,
                probe_points_body,
            ).astype(np.float32)
        contact_probes.set_world_poses(positions=probe_positions)
        world.step(render=False)
        model.maybe_sample(world.current_time, world.current_time_step_index)

    sample = model.latest
    forces = sample.total_forces if sample is not None else np.zeros((2, 3))
    world.render()
    if frame_index in (0, 75, 120, 180, 240):
        debug_apices = dynamic_body_points(sensor_bodies, apex_points_body)
        print(
            f"frame {frame_index} dynamic apices:",
            np.round(debug_apices, 4),
            "midpoint=",
            np.round(np.mean(debug_apices, axis=0), 4),
        )
    pixels = rgb.get_data()
    if pixels is None or not pixels.size:
        writer.release()
        app.close()
        raise RuntimeError(f"No RGB data at frame {frame_index}")
    frame = compose_frame(pixels, sample, phase, time_seconds)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    score = float(np.sum(np.maximum(forces[:, 2], 0.0)))
    if score > poster_score:
        poster_score = score
        poster = frame.copy()
    frame_log.append(
        {
            "frame": frame_index,
            "video_time_s": time_seconds,
            "phase": phase,
            "gripper_target_deg": float(gripper_target),
            "sample": sample.to_dict() if sample is not None else None,
        }
    )

writer.release()
if poster is not None:
    Image.fromarray(poster).save(POSTER_PATH)
with LOG_PATH.open("w", encoding="utf-8") as output_file:
    json.dump(
        {
            "fps": FPS,
            "duration_s": DURATION_S,
            "frame_count": FRAME_COUNT,
            "object_center_m": [float(value) for value in object_center],
            "contact_source": (
                "Two invisible kinematic 4 mm contact-patch spheres acting "
                "on the visible fingertip dome collision meshes."
            ),
            "note": (
                "Scripted third-person visual grasp. HUD values are live "
                "Isaac ContactSensor aggregate forces generated by two "
                "invisible kinematic contact patches loaded against the "
                "visible fingertip domes. The visible ball follows the "
                "measured fingertip midpoint during lift; this is not a "
                "stable free-body grasp."
            ),
            "frames": frame_log,
        },
        output_file,
        indent=2,
    )

print("object center:", tuple(object_center))
print("maximum displayed normal-force sum N:", poster_score)
print("saved:", RAW_VIDEO_PATH)
print("saved:", POSTER_PATH)
print("saved:", LOG_PATH)
rgb.detach([render_product])
render_product.destroy()
model.close()
app.close()
