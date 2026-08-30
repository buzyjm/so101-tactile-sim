from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from isaacsim.core.api import World
from pxr import UsdGeom, Gf

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
PREFIX = ("/World/Robot/Geometry/base_link/shoulder_link/upper_arm_link"
          "/lower_arm_link/wrist_link/gripper_link")

PATHS = {
    "gripper_link": PREFIX,
    "gripper_frame_link": PREFIX + "/gripper_frame_link",
    "moving_jaw_link": PREFIX + "/moving_jaw_so101_v1_link",
    "moving_jaw_mesh": PREFIX + "/moving_jaw_so101_v1_link/moving_jaw_so101_v1_1/moving_jaw_so101_v1",
    "static_jaw_mesh": PREFIX + "/wrist_roll_follower_so101_v1",
}

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)
world.reset()
for _ in range(60):
    world.step(render=False)

print("=" * 78)
print("WORLD POSITIONS AT ZERO POSE")
print("=" * 78)
poses = {}
for label, path in PATHS.items():
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"  {label:<22} MISSING  {path}")
        continue
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = m.ExtractTranslation()
    poses[label] = np.array([t[0], t[1], t[2]])
    print(f"  {label:<22} ({t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f})")

print()
print("=" * 78)
print("BOUNDING BOXES (world aligned)")
print("=" * 78)
cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
for label, path in PATHS.items():
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        continue
    bound = cache.ComputeWorldBound(prim)
    r = bound.ComputeAlignedRange()
    if r.IsEmpty():
        continue
    mn, mx = r.GetMin(), r.GetMax()
    print(f"  {label:<22} min=({mn[0]:7.4f},{mn[1]:7.4f},{mn[2]:7.4f})  "
          f"max=({mx[0]:7.4f},{mx[1]:7.4f},{mx[2]:7.4f})")

if "gripper_frame_link" in poses and "moving_jaw_link" in poses:
    d = poses["moving_jaw_link"] - poses["gripper_frame_link"]
    print(f"\n  frame -> jaw offset: {np.round(d, 4)}  |d| = {np.linalg.norm(d):.4f}")

print("=" * 78)
app.close()