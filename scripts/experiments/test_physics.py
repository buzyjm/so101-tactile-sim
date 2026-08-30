from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from isaacsim.core.api import World
from pxr import UsdGeom

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

world = World(stage_units_in_meters=1.0)
world.reset()

cube = stage.GetPrimAtPath("/World/TargetCube")


def world_z(prim):
    xform = UsdGeom.Xformable(prim)
    return xform.ComputeLocalToWorldTransform(0).ExtractTranslation()[2]


print("start z:", round(world_z(cube), 4))
for i in range(120):
    world.step(render=False)
    if i % 20 == 19:
        print(f"step {i + 1:3d}  z = {world_z(cube):.4f}")

app.close()

# Expected resting z = table_top(0.75) + cube_half_height(0.015) = 0.765