from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import numpy as np
import omni.usd
from pxr import UsdGeom, Gf

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

print("=" * 60)
for path in ["/World/Ground", "/World/Table", "/World/TargetCube", "/World/Cameras/TopDown"]:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(path, "-> MISSING")
        continue
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = m.ExtractTranslation()
    print(f"{path:32s} pos = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")

cam_prim = stage.GetPrimAtPath("/World/Cameras/TopDown")
m = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(0)
forward = m.TransformDir(Gf.Vec3d(0, 0, -1))
up = m.TransformDir(Gf.Vec3d(0, 1, 0))
print(f"camera forward = ({forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f})")
print(f"camera up      = ({up[0]:.3f}, {up[1]:.3f}, {up[2]:.3f})")

cam = UsdGeom.Camera(cam_prim)
print("focalLength:", cam.GetFocalLengthAttr().Get())
print("horizontalAperture:", cam.GetHorizontalApertureAttr().Get())
print("clippingRange:", cam.GetClippingRangeAttr().Get())
print("=" * 60)

app.close()