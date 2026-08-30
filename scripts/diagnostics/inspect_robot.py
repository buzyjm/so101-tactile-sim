from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

USD_PATH = str(PROJECT_ROOT / "assets/so101_new_calib/so101_new_calib.usda")

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

print("=" * 78)
print("default prim:", stage.GetDefaultPrim().GetPath())
print("up axis:", UsdGeom.GetStageUpAxis(stage))
print("meters per unit:", UsdGeom.GetStageMetersPerUnit(stage))
print("=" * 78)

for prim in stage.Traverse():
    path = str(prim.GetPath())
    depth = path.count("/") - 1
    indent = "  " * depth
    apis = [s for s in prim.GetAppliedSchemas()
            if "Physics" in s or "Collision" in s or "Articulation" in s or "Drive" in s]
    tag = f"  <-- {apis}" if apis else ""
    print(f"{indent}{prim.GetName():<34} [{prim.GetTypeName()}]{tag}")

print("=" * 78)
print("JOINT DRIVES")
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue
    joint = UsdPhysics.RevoluteJoint(prim)
    lower = joint.GetLowerLimitAttr().Get()
    upper = joint.GetUpperLimitAttr().Get()
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        stiff = drive.GetStiffnessAttr().Get()
        damp = drive.GetDampingAttr().Get()
        force = drive.GetMaxForceAttr().Get()
        dtype = drive.GetTypeAttr().Get()
    else:
        stiff = damp = force = dtype = None
    print(f"{prim.GetName():<20} limits=({lower}, {upper})  "
          f"stiffness={stiff}  damping={damp}  maxForce={force}  type={dtype}")

print("=" * 78)

app.close()