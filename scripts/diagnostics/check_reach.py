from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from isaacsim.core.api import World
from pxr import UsdGeom, Gf

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
TCP_PATH = "/World/Robot/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/gripper_frame_link"

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

world = World(stage_units_in_meters=1.0)
world.reset()
for _ in range(60):
    world.step(render=False)


def world_pos(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return m.ExtractTranslation()


base = world_pos("/World/Robot")
tcp = world_pos(TCP_PATH)
cube = world_pos("/World/TargetCube")

print("=" * 70)
print("base:", base)
print("tcp: ", tcp)
print("cube:", cube)

if base and cube:
    d = Gf.Vec3d(cube) - Gf.Vec3d(base)
    print(f"base to cube distance: {d.GetLength():.4f} m")
if tcp and cube:
    d = Gf.Vec3d(cube) - Gf.Vec3d(tcp)
    print(f"tcp  to cube distance: {d.GetLength():.4f} m")
print("=" * 70)

app.close()