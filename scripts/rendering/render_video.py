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
N_FRAMES = 180

os.makedirs(OUT_DIR, exist_ok=True)

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

# Inspection camera: three quarter view, computed with SetLookAt so we do not
# have to guess euler angles
eye = Gf.Vec3d(1.1, -1.0, 1.35)
target = Gf.Vec3d(0.0, 0.0, 0.80)
up = Gf.Vec3d(0.0, 0.0, 1.0)

view = Gf.Matrix4d()
view.SetLookAt(eye, target, up)

cam = UsdGeom.Camera.Define(stage, "/World/Cameras/Inspect")
cam.CreateFocalLengthAttr(20.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())

world = World(stage_units_in_meters=1.0)
world.reset()

# Drive targets are written straight to USD, no articulation wrapper needed
JOINT_ROOT = "/World/Robot/Physics"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

drives = {}
for name in JOINTS:
    prim = stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}")
    if not prim.IsValid():
        print("MISSING JOINT:", name)
        continue
    drives[name] = UsdPhysics.DriveAPI.Get(prim, "angular")
print("drives found:", list(drives.keys()))

render_product = rep.create.render_product("/World/Cameras/Inspect", RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])

for _ in range(20):
    world.step(render=True)

frames = []
for i in range(N_FRAMES):
    t = i / N_FRAMES
    # Degrees. Sweep a few joints so the motion is visible.
    targets = {
        "shoulder_pan": 30.0 * math.sin(2 * math.pi * t),
        "shoulder_lift": -40.0 * math.sin(math.pi * t),
        "elbow_flex": 50.0 * math.sin(math.pi * t),
        "wrist_flex": 20.0 * math.sin(2 * math.pi * t),
        "gripper": 45.0 * (1 - math.cos(4 * math.pi * t)) / 2,
    }
    for name, value in targets.items():
        if name in drives:
            drives[name].CreateTargetPositionAttr().Set(value)

    world.step(render=True)
    data = rgb.get_data()
    if data is not None and data.size:
        frames.append(data[:, :, :3].copy())

print("frames captured:", len(frames))

out_path = os.path.join(OUT_DIR, "inspect.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, FPS, RESOLUTION)
for f in frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
print("saved:", out_path)

# Keep a still as a fallback in case the codec misbehaves
if frames:
    cv2.imwrite(os.path.join(OUT_DIR, "inspect_last.png"),
                cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    print("saved still: inspect_last.png")

app.close()