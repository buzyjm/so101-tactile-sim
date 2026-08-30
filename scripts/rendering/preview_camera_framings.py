"""Render several candidate demo camera framings in one Isaac session.

Choosing a demo camera by re-rendering the whole video is slow and the last
attempt cropped the robot out of frame.  This renders one still per candidate
so the framing can be judged before committing to a full video pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--scene", type=Path, default=PROJECT_ROOT_HINT / "lab_scene_task.usda"
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=120,
        help="Physics steps before capture so the scene is at rest.",
    )
    return parser.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp

APP = SimulationApp(
    {
        "headless": True,
        "active_gpu": ARGS.gpu,
        "physics_gpu": ARGS.gpu,
        "multi_gpu": False,
        "anti_aliasing": 1,
    }
)

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from PIL import Image
from pxr import Gf, UsdGeom


def main() -> int:
    candidates = json.loads(ARGS.candidates.read_text(encoding="utf-8"))
    if not omni.usd.get_context().open_stage(str(ARGS.scene.resolve())):
        raise RuntimeError(f"Could not open scene: {ARGS.scene}")
    stage = omni.usd.get_context().get_stage()
    carb.settings.get_settings().set("/rtx/post/aa/op", 1)

    world = World(stage_units_in_meters=1.0)
    world.reset()

    ARGS.out_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        path = f"/World/Cameras/Preview_{candidate['name']}"
        view = Gf.Matrix4d()
        view.SetLookAt(
            Gf.Vec3d(*candidate["eye"]),
            Gf.Vec3d(*candidate["target"]),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(float(candidate["focal"]))
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(
            20.955 * ARGS.height / ARGS.width
        )
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
        UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())

        product = rep.create.render_product(path, (ARGS.width, ARGS.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        for _ in range(ARGS.settle_steps):
            world.step(render=True)
        image = annotator.get_data()
        annotator.detach()
        product.destroy()
        if image is None or not image.size:
            print(f"[{candidate['name']}] no image")
            continue
        out = ARGS.out_dir / f"cam_{candidate['name']}.png"
        Image.fromarray(np.asarray(image)[:, :, :3]).save(out)
        print(f"[{candidate['name']}] az={candidate['az_deg']} "
              f"elev={candidate['elev_deg']} focal={candidate['focal']} -> {out}",
              flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
