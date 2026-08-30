"""Render close-up inspection views of both fingertip tactile housings."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isaacsim import SimulationApp

RENDER_GPU = int(os.environ.get("SO101_RENDER_GPU", "0"))
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": RENDER_GPU,
        "physics_gpu": RENDER_GPU,
        "multi_gpu": False,
        "anti_aliasing": 1,
    }
)

import carb
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from PIL import Image
from pxr import Gf, Usd, UsdGeom

from scene_config import PROJECT_ROOT, TACTILE_ROOT_PRIM_PATHS


USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
OUT_DIR = PROJECT_ROOT / "renders"
RESOLUTION = (1600, 1200)


def add_look_at_camera(stage, path, eye, target, focal_length=52.0):
    view = Gf.Matrix4d()
    view.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())


omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
carb.settings.get_settings().set("/rtx/post/aa/op", 1)

world = World(stage_units_in_meters=1.0)
world.reset()
for _ in range(60):
    world.step(render=False)

sensor_centers = []
for path in TACTILE_ROOT_PRIM_PATHS:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        app.close()
        raise RuntimeError(f"Missing tactile sensor prim: {path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    sensor_centers.append(matrix.ExtractTranslation())

target = (sensor_centers[0] + sensor_centers[1]) * 0.5
views = (
    (
        "/World/Cameras/TactileCloseupOblique",
        "tactile_closeup_oblique.png",
        target + Gf.Vec3d(0.115, -0.075, 0.040),
    ),
    (
        "/World/Cameras/TactileCloseupSide",
        "tactile_closeup_side.png",
        target + Gf.Vec3d(-0.115, -0.040, 0.025),
    ),
    (
        "/World/Cameras/TactileGripperContext",
        "tactile_gripper_context.png",
        target + Gf.Vec3d(0.300, -0.220, 0.135),
    ),
)
only_filename = os.environ.get("SO101_TACTILE_RENDER_ONLY")
if only_filename:
    views = tuple(view for view in views if view[1] == only_filename)
    if not views:
        app.close()
        raise ValueError(f"Unknown tactile render filename: {only_filename}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
for camera_path, filename, eye in views:
    add_look_at_camera(stage, camera_path, eye, target)
    render_product = rep.create.render_product(camera_path, RESOLUTION)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])
    for _ in range(8):
        world.step(render=True)
    for _ in range(16):
        rep.orchestrator.step(rt_subframes=8)
    data = rgb.get_data()
    if data is None or not data.size:
        app.close()
        raise RuntimeError(f"Renderer returned no data for {filename}")
    output = OUT_DIR / filename
    Image.fromarray(data[:, :, :3]).save(output)
    print("saved:", output)
    rgb.detach([render_product])
    render_product.destroy()

print("sensor centers:", [tuple(center) for center in sensor_centers])
app.close()
