"""Render the current lab USD from a high-quality front third-person camera.

This renderer deliberately opens ``lab_scene.usda`` as the root stage. That
keeps the authored robot and tactile-sensor material bindings identical to the
validated close-up renders instead of reconstructing only part of the scene in
an Isaac Lab environment namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scene",
    type=Path,
    default=PROJECT_ROOT / "lab_scene.usda",
)
parser.add_argument("--width", type=int, default=3840)
parser.add_argument("--height", type=int, default=2160)
parser.add_argument(
    "--eye",
    type=float,
    nargs=3,
    default=(0.16, 1.02, 0.50),
    metavar=("X", "Y", "Z"),
)
parser.add_argument(
    "--target",
    type=float,
    nargs=3,
    default=(0.08, 0.20, 0.16),
    metavar=("X", "Y", "Z"),
)
parser.add_argument("--focal-length", type=float, default=27.0)
parser.add_argument(
    "--quality", choices=("preview", "final"), default="final"
)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "renders" / "isaac_setup_front_4k.png",
)
args = parser.parse_args()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        # TAA renders at the requested resolution. DLSS performance mode made
        # the small robot and fingertip geometry look soft in the overview.
        "anti_aliasing": 1,
    }
)

import carb
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from PIL import Image
from pxr import Gf, UsdGeom


CAMERA_PATH = "/World/Cameras/SetupFront"


def _add_look_at_camera(stage) -> None:
    view = Gf.Matrix4d()
    view.SetLookAt(
        Gf.Vec3d(*args.eye),
        Gf.Vec3d(*args.target),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    camera.CreateFocalLengthAttr(args.focal_length)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())


def main() -> None:
    if args.width < 320 or args.height < 240:
        raise ValueError("Resolution must be at least 320x240")

    scene_path = args.scene.resolve()
    if not omni.usd.get_context().open_stage(str(scene_path)):
        raise RuntimeError(f"Could not open scene: {scene_path}")
    stage = omni.usd.get_context().get_stage()

    # Opening a USD can restore persisted renderer settings. Re-apply TAA
    # after opening it so the output is not silently DLSS-upscaled.
    carb.settings.get_settings().set("/rtx/post/aa/op", 1)
    _add_look_at_camera(stage)

    world = World(stage_units_in_meters=1.0)
    world.reset()
    for _ in range(60):
        world.step(render=False)
    for _ in range(10):
        world.step(render=True)

    render_product = rep.create.render_product(
        CAMERA_PATH, (args.width, args.height)
    )
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])

    step_count = 8 if args.quality == "preview" else 24
    subframes = 4 if args.quality == "preview" else 8
    for _ in range(step_count):
        rep.orchestrator.step(rt_subframes=subframes)

    pixels = rgb.get_data()
    if pixels is None or not pixels.size:
        raise RuntimeError("Renderer returned no RGB data")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels[:, :, :3]).save(args.output)
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(args.output.resolve()),
                "resolution": [args.width, args.height],
                "eye_m": list(args.eye),
                "target_m": list(args.target),
                "focal_length_mm": args.focal_length,
                "quality": args.quality,
                "gpu": args.gpu,
                "robot_pose": "authored_lab_scene",
                "renderer": "RTX/TAA + Replicator subframes",
                "root_stage": str(scene_path),
            },
            indent=2,
        ),
        flush=True,
    )

    rgb.detach([render_product])
    render_product.destroy()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
