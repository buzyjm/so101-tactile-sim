from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from isaacsim.core.api import World
from pxr import UsdGeom, UsdPhysics

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
PREFIX = ("/World/Robot/Geometry/base_link/shoulder_link/upper_arm_link"
          "/lower_arm_link/wrist_link/gripper_link")
MOVING = PREFIX + "/moving_jaw_so101_v1_link"
STATIC = PREFIX + "/wrist_roll_follower_so101_v1"
GRIPPER_DRIVE = "/World/Robot/Physics/gripper"

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(GRIPPER_DRIVE), "angular")

world = World(stage_units_in_meters=1.0)
world.reset()

cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])


def bbox(path):
    cache.Clear()
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
    return mn, mx, (mn + mx) / 2.0


s_min, s_max, s_c = bbox(STATIC)
print("=" * 84)
print(f"static jaw  min={np.round(s_min, 4)}  max={np.round(s_max, 4)}  center={np.round(s_c, 4)}")
print("=" * 84)
print(f"{'angle':>6} | {'moving jaw center':>26} | {'moving jaw max':>26} | {'center dist':>11}")
print("=" * 84)

for angle in [0, 20, 40, 60, 80, 100]:
    drive.CreateTargetPositionAttr().Set(float(angle))
    for _ in range(150):
        world.step(render=False)
    m_min, m_max, m_c = bbox(MOVING)
    d = np.linalg.norm(m_c - s_c)
    print(f"{angle:6.0f} | {str(np.round(m_c, 4)):>26} | "
          f"{str(np.round(m_max, 4)):>26} | {d:11.4f}")

print("=" * 84)
app.close()