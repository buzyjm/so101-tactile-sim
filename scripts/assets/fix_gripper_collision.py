from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from pxr import Usd, UsdPhysics

INSTANCES = str(PROJECT_ROOT / "assets/so101_new_calib/payloads/instances.usda")

JAW_PATTERNS = ["moving_jaw", "wrist_roll_follower"]

# Editing the source layer directly, no Kit runtime needed
stage = Usd.Stage.Open(INSTANCES)

print("=" * 78)
changed = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    name = prim.GetName().lower()
    if not any(pat in name for pat in JAW_PATTERNS):
        continue
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    before = mesh_api.GetApproximationAttr().Get()
    mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
    after = mesh_api.GetApproximationAttr().Get()
    print(f"  {prim.GetName():<34} {before} -> {after}")
    changed += 1

print(f"\nchanged: {changed}")
if changed:
    stage.GetRootLayer().Save()
    print("saved:", INSTANCES)
print("=" * 78)