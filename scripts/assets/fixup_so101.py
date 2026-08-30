from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

USD_PATH = str(PROJECT_ROOT / "assets/so101_new_calib/so101_new_calib.usda")

# Measured STS3215 torque limit from joints_properties.xml
MAX_TORQUE = 3.35

# LoadAll is required: layer_structure=True puts geometry in payloads,
# which are not composed by default.
stage = Usd.Stage.Open(USD_PATH, load=Usd.Stage.LoadAll)

print("=" * 78)
print("COLLISION GEOMETRY")
print("=" * 78)

n_collision = 0
approximations = {}
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    n_collision += 1
    approx = "(none)"
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        if attr and attr.Get():
            approx = attr.Get()
    approximations[approx] = approximations.get(approx, 0) + 1
    name = prim.GetName()
    if "jaw" in name.lower() or "gripper" in name.lower():
        print(f"  GRIPPER PART  {name:<42} {approx}")

print(f"\ntotal colliders: {n_collision}")
for k, v in sorted(approximations.items()):
    print(f"  {k:<28} {v}")

print()
print("=" * 78)
print("JOINT DRIVES BEFORE")
print("=" * 78)

joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
for prim in joints:
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        print(f"  {prim.GetName():<18} maxForce={drive.GetMaxForceAttr().Get()}")

for prim in joints:
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        drive.CreateMaxForceAttr().Set(MAX_TORQUE)

print()
print("=" * 78)
print("JOINT DRIVES AFTER")
print("=" * 78)
for prim in joints:
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        d = drive
        print(f"  {prim.GetName():<18} maxForce={d.GetMaxForceAttr().Get()}  "
              f"stiffness={d.GetStiffnessAttr().Get():.5f}  "
              f"damping={d.GetDampingAttr().Get():.6f}")

stage.GetRootLayer().Save()
print("\nsaved overrides to:", stage.GetRootLayer().identifier)
print("=" * 78)

app.close()