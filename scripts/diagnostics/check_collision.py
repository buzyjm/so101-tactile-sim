from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

from pxr import Usd, UsdGeom, UsdPhysics

USD_PATH = str(PROJECT_ROOT / "assets/so101_new_calib/so101_new_calib.usda")

stage = Usd.Stage.Open(USD_PATH, load=Usd.Stage.LoadAll)

# Traverse into instance proxies, otherwise instanced geometry stays hidden
predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

print("=" * 78)
counts = {}
gripper_parts = []
for prim in Usd.PrimRange.Stage(stage, predicate):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    approx = "(unset)"
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        if attr and attr.Get():
            approx = attr.Get()
    counts[approx] = counts.get(approx, 0) + 1
    name = prim.GetName().lower()
    if "jaw" in name or "gripper" in name:
        gripper_parts.append((str(prim.GetPath()), approx))

print("collider count by approximation:")
for k, v in sorted(counts.items()):
    print(f"  {k:<26} {v}")

print("\ngripper-related colliders:")
for path, approx in gripper_parts:
    print(f"  {approx:<22} {path}")
print("=" * 78)

app.close()