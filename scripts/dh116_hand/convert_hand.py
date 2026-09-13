"""Convert assets/dh116_hand/dh116_hand.urdf to USD with the Isaac Sim 6.0 importer.

    conda run --no-capture-output -n isaacsim python scripts/dh116_hand/convert_hand.py

Writes assets/dh116_hand/usd_manual_2026_04/dh116_hand/dh116_hand.usda (layered payloads
next to it) and prints a joint summary so the import can be sanity-checked.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False}
)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.urdf")
app.update()
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "assets" / "dh116_hand"
URDF = ASSET_DIR / "dh116_hand.urdf"
USD_DIR = ASSET_DIR / "usd_manual_2026_04"

ARM = r"^joint[1-5]$"
HAND_ACTIVE = r"^hand_finger(11|12|21|31|41|51)$"
HAND_PASSIVE = r"^hand_finger(13|22|32|42|52)$"


def main() -> int:
    if USD_DIR.exists():
        shutil.rmtree(USD_DIR)
    USD_DIR.mkdir(parents=True)
    cfg = URDFImporterConfig(
        urdf_path=str(URDF),
        usd_path=str(USD_DIR),
        merge_fixed_joints=False,
        merge_mesh=False,
        collision_from_visuals=True,
        collision_type="Convex Decomposition",
        allow_self_collision=False,
        fix_base=True,
        joint_drive_type="force",
        joint_target_type="position",
        override_joint_stiffness={HAND_ACTIVE: 1.0, HAND_PASSIVE: 0.0},
        override_joint_damping={HAND_ACTIVE: 0.1, HAND_PASSIVE: 0.0},
    )
    t0 = time.time()
    usd = URDFImporter(cfg).import_urdf()
    print(f"IMPORT RESULT: {usd} ({time.time() - t0:.1f}s)")
    if not usd:
        return 1
    stage = Usd.Stage.Open(usd)
    rev, fixed, roots, bodies = [], [], [], []
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(p)
            d = UsdPhysics.DriveAPI(p, "angular")
            rev.append((p.GetName(), j.GetAxisAttr().Get(), round(j.GetLowerLimitAttr().Get(), 1),
                        round(j.GetUpperLimitAttr().Get(), 1), d.GetStiffnessAttr().Get(), d.GetDampingAttr().Get()))
        elif p.IsA(UsdPhysics.FixedJoint):
            fixed.append(p.GetName())
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(p.GetPath()))
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(p.GetName())
    print("articulation roots:", roots)
    print(f"rigid bodies ({len(bodies)}):", bodies)
    print(f"revolute joints ({len(rev)}):")
    for r in rev:
        print("   ", r)
    print(f"fixed joints ({len(fixed)}):", fixed)
    # mesh check: every link should own geometry
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    empty = []
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            r = bbox.ComputeWorldBound(p).ComputeAlignedRange()
            if r.IsEmpty():
                empty.append(p.GetName())
    print("links without geometry:", empty or "none")
    size = sum(f.stat().st_size for f in USD_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"usd dir size: {size:.1f} MB")
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)
