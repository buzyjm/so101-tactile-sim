from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.storage.native import get_assets_root_path

root = get_assets_root_path()
scene_path = root + "/Isaac/Environments/Simple_Room/simple_room.usd"
print("loading:", scene_path)

omni.usd.get_context().open_stage(scene_path)
stage = omni.usd.get_context().get_stage()

print("\n" + "=" * 70)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    depth = path.count("/") - 1
    indent = "  " * depth

    apis = [s for s in prim.GetAppliedSchemas() if "Physics" in s or "Collision" in s]
    tag = f"  <-- {apis}" if apis else ""

    print(f"{indent}{prim.GetName():<30} [{prim.GetTypeName()}]{tag}")
print("=" * 70)

app.close()