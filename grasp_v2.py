import os

import numpy as np

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import cv2
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import UsdGeom, UsdPhysics, Gf

from scene_config import (
    EXTERNAL_CAMERA_EYE,
    EXTERNAL_CAMERA_TARGET,
    PROJECT_ROOT,
    ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG,
    TARGET_CUBE_CENTER,
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
OUT_DIR = str(PROJECT_ROOT / "renders")
RESOLUTION = (1280, 720)

JOINT_ROOT = "/World/Robot/Physics"
CUBE_PATH = "/World/TargetCube"
FRAME = "gripper_frame_link"

BASE_POS = np.array(ROBOT_BASE_POSITION)
base_yaw = np.radians(ROBOT_BASE_YAW_DEG)
BASE_QUAT = np.array(
    [np.cos(base_yaw / 2.0), 0.0, 0.0, np.sin(base_yaw / 2.0)]
)
CUBE_CENTER = np.array(TARGET_CUBE_CENTER)

TILT_DEG = 15.0
# The grasp point sits behind the frame origin, roughly mid jaw
TCP_OFFSET = 0.025
# Calibrated from the physical sweep: center the aperture and clear the table.
LATERAL_OFFSET = 0.015
GRASP_HEIGHT_OFFSET = 0.015

os.makedirs(OUT_DIR, exist_ok=True)

solver = LulaKinematicsSolver(robot_description_path=DESCRIPTOR, urdf_path=URDF)
solver.set_robot_base_pose(BASE_POS, BASE_QUAT)
ARM = solver.get_joint_names()


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


t = np.radians(TILT_DEG)
R_WORLD_FROM_BASE = np.array(
    [
        [np.cos(base_yaw), -np.sin(base_yaw), 0.0],
        [np.sin(base_yaw), np.cos(base_yaw), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
z_dir_base = np.array([np.cos(t), 0.0, -np.sin(t)])
Z_DIR = R_WORLD_FROM_BASE @ z_dir_base
x = np.cross(np.array([0.0, 1.0, 0.0]), z_dir_base)
x /= np.linalg.norm(x)
y = np.cross(z_dir_base, x)
R_tool_base = np.column_stack([x, y, z_dir_base])
# roll 90 turns the jaws from opening vertically to opening sideways
R_roll = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
QUAT = rot_to_quat(R_WORLD_FROM_BASE @ R_tool_base @ R_roll)

GRASP_POINT = CUBE_CENTER + R_WORLD_FROM_BASE @ np.array(
    [0.0, LATERAL_OFFSET, GRASP_HEIGHT_OFFSET]
)
POINTS = {
    # Retreat along the tool axis so the open fingers do not sweep the cube.
    "approach": GRASP_POINT - 0.10 * Z_DIR,
    "pregrasp": GRASP_POINT - 0.04 * Z_DIR,
    "grasp": GRASP_POINT.copy(),
    "lift": GRASP_POINT + np.array([0.0, 0.0, 0.12]),
}

print("=" * 78)
solutions = {}
# Anchor the branch at contact; top-down warm starts converge to a miss.
for label in ["grasp", "pregrasp", "approach", "lift"]:
    point = POINTS[label]
    target = point + TCP_OFFSET * Z_DIR
    if label == "grasp":
        warm = None
    elif label == "pregrasp":
        warm = solutions["grasp"][0]
    elif label == "approach":
        warm = solutions["pregrasp"][0]
    else:
        warm = solutions["grasp"][0]
    q, ok = solver.compute_inverse_kinematics(
        frame_name=FRAME, target_position=target, target_orientation=QUAT,
        warm_start=warm, position_tolerance=0.005, orientation_tolerance=0.3,
    )
    solutions[label] = (q, ok)

for label in POINTS:
    point = POINTS[label]
    target = point + TCP_OFFSET * Z_DIR
    q, ok = solutions[label]
    print(f"[{label:9s}] frame target {np.round(target, 4)}  ok={ok}  "
          f"joints={np.round(np.degrees(q), 1) if ok else None}")
print("=" * 78)

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

drives = {n: UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{JOINT_ROOT}/{n}"), "angular")
          for n in ARM + ["gripper"]}

view = Gf.Matrix4d()
view.SetLookAt(
    Gf.Vec3d(*EXTERNAL_CAMERA_EYE),
    Gf.Vec3d(*EXTERNAL_CAMERA_TARGET),
    Gf.Vec3d(0.0, 0.0, 1.0),
)
cam = UsdGeom.Camera.Define(stage, "/World/Cameras/Grasp")
cam.CreateFocalLengthAttr(28.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())

world = World(stage_units_in_meters=1.0)
world.reset()

rp = rep.create.render_product("/World/Cameras/Grasp", RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
for _ in range(90):
    world.step(render=True)

frames = []


def cube_pos():
    m = UsdGeom.Xformable(stage.GetPrimAtPath(CUBE_PATH)).ComputeLocalToWorldTransform(0)
    return np.array(m.ExtractTranslation())


def go(label, gripper_deg, steps):
    q, ok = solutions[label]
    if ok:
        for n, v in zip(ARM, np.degrees(q)):
            drives[n].CreateTargetPositionAttr().Set(float(v))
    drives["gripper"].CreateTargetPositionAttr().Set(float(gripper_deg))
    for _ in range(steps):
        world.step(render=True)
        d = rgb.get_data()
        if d is not None and d.size:
            frames.append(d[:, :, :3].copy())
    print(f"[{label:9s}] gripper={gripper_deg:5.1f}  cube={np.round(cube_pos(), 4)}")


z0 = cube_pos()[2]
# First validate contact and lifting with the known-good joint-space entry.
go("grasp", 70, 150)
go("grasp", 5, 120)
go("lift", 5, 180)
go("lift", 5, 60)
z1 = cube_pos()[2]

print("=" * 78)
print(f"cube z: {z0:.4f} -> {z1:.4f}   rise = {z1 - z0:+.4f} m")
print("GRASP SUCCESS" if z1 - z0 > 0.02 else "GRASP FAILED")
print("=" * 78)

out = os.path.join(OUT_DIR, "grasp_v2.mp4")
w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, RESOLUTION)
for f in frames:
    w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
w.release()
print("saved:", out)

app.close()
