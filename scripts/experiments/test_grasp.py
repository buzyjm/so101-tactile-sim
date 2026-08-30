from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import math

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import numpy as np
import cv2

import omni.usd
import omni.replicator.core as rep
from isaacsim.core.api import World
from pxr import Usd, UsdGeom, UsdPhysics, Gf

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
OUT_DIR = str(PROJECT_ROOT / "renders")
RESOLUTION = (1280, 720)
FPS = 30

JOINT_ROOT = "/World/Robot/Physics"
CUBE_PATH = "/World/TargetCube"
TCP_PATH = ("/World/Robot/Geometry/base_link/shoulder_link/upper_arm_link"
            "/lower_arm_link/wrist_link/gripper_link/gripper_frame_link")

os.makedirs(OUT_DIR, exist_ok=True)

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

eye = Gf.Vec3d(0.85, -0.75, 1.15)
target = Gf.Vec3d(0.0, 0.0, 0.82)
view = Gf.Matrix4d()
view.SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))

cam = UsdGeom.Camera.Define(stage, "/World/Cameras/Grasp")
cam.CreateFocalLengthAttr(24.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())

world = World(stage_units_in_meters=1.0)
world.reset()

drives = {}
for name in ["shoulder_pan", "shoulder_lift", "elbow_flex",
             "wrist_flex", "wrist_roll", "gripper"]:
    prim = stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}")
    if prim.IsValid():
        drives[name] = UsdPhysics.DriveAPI.Get(prim, "angular")
    else:
        print("MISSING JOINT:", name)
print("drives:", list(drives.keys()))


def set_targets(targets):
    for name, value in targets.items():
        if name in drives:
            drives[name].CreateTargetPositionAttr().Set(float(value))


def world_pos(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return m.ExtractTranslation()


render_product = rep.create.render_product("/World/Cameras/Grasp", RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])

for _ in range(20):
    world.step(render=True)

# Joint angles in degrees. Values are a first guess for a vertical approach;
# expect to tune them from the logged TCP position.
POSE_HOME = dict(shoulder_pan=0, shoulder_lift=0, elbow_flex=0,
                 wrist_flex=0, wrist_roll=0, gripper=60)
POSE_ABOVE = dict(shoulder_pan=0, shoulder_lift=-35, elbow_flex=55,
                  wrist_flex=-25, wrist_roll=0, gripper=60)
POSE_DOWN = dict(shoulder_pan=0, shoulder_lift=-50, elbow_flex=70,
                 wrist_flex=-25, wrist_roll=0, gripper=60)
POSE_CLOSED = dict(POSE_DOWN, gripper=2)
POSE_LIFT = dict(POSE_ABOVE, gripper=2)

PHASES = [
    ("home", POSE_HOME, 40),
    ("above", POSE_ABOVE, 70),
    ("descend", POSE_DOWN, 70),
    ("close", POSE_CLOSED, 60),
    ("lift", POSE_LIFT, 80),
    ("hold", POSE_LIFT, 40),
]

frames = []
log = []
for label, pose, n_steps in PHASES:
    set_targets(pose)
    for i in range(n_steps):
        world.step(render=True)
        data = rgb.get_data()
        if data is not None and data.size:
            frames.append(data[:, :, :3].copy())
    tcp = world_pos(TCP_PATH)
    cube = world_pos(CUBE_PATH)
    gap = (Gf.Vec3d(cube) - Gf.Vec3d(tcp)).GetLength()
    log.append((label, tcp, cube, gap))
    print(f"[{label:8s}] tcp_z={tcp[2]:.4f}  cube=({cube[0]:.4f}, "
          f"{cube[1]:.4f}, {cube[2]:.4f})  tcp-cube={gap:.4f}")

print()
print("=" * 70)
cube_start = log[0][2][2]
cube_end = log[-1][2][2]
rise = cube_end - cube_start
print(f"cube z: {cube_start:.4f} -> {cube_end:.4f}   rise = {rise:+.4f} m")
print("GRASP SUCCESS" if rise > 0.02 else "GRASP FAILED")
print("=" * 70)

out_path = os.path.join(OUT_DIR, "grasp.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, RESOLUTION)
for f in frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
print("saved:", out_path, f"({len(frames)} frames)")

app.close()