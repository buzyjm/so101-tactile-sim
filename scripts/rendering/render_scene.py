import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isaacsim import SimulationApp
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": 0,
        "physics_gpu": 0,
        # TAA renders at the requested 640x480 instead of DLSS upscaling from
        # 320x240, which visibly aliases small round objects.
        "anti_aliasing": 1,
    }
)

import numpy as np
from PIL import Image

import carb
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.api import World

from scene_config import CAMERA_RESOLUTION, PROJECT_ROOT

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
OUT_DIR = str(PROJECT_ROOT / "renders")
RESOLUTION = (
    int(os.environ.get("SO101_RENDER_WIDTH", CAMERA_RESOLUTION[0])),
    int(os.environ.get("SO101_RENDER_HEIGHT", CAMERA_RESOLUTION[1])),
)
OUTPUT_SUFFIX = os.environ.get("SO101_RENDER_SUFFIX", "")

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

# Opening a stage can restore its persisted RTX settings. Re-apply TAA here so
# small round objects are rendered at the requested resolution instead of
# switching back to DLSS performance upscaling.
carb.settings.get_settings().set("/rtx/post/aa/op", 1)

world = World(stage_units_in_meters=1.0)
world.reset()

# Let the lightweight ball settle onto the table.
for _ in range(60):
    world.step(render=False)

# Warm up the RTX renderer before attaching the annotator
for _ in range(10):
    world.step(render=True)

views = []
for camera_path, filename in (
    ("/World/Cameras/TopDown", f"topdown{OUTPUT_SUFFIX}.png"),
    ("/World/Cameras/BirdEye", f"birdseye{OUTPUT_SUFFIX}.png"),
):
    render_product = rep.create.render_product(camera_path, RESOLUTION)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])
    views.append((camera_path, filename, rgb))

# RTX converges progressively, so accumulate several frames
for _ in range(30):
    rep.orchestrator.step(rt_subframes=8)

os.makedirs(OUT_DIR, exist_ok=True)
for camera_path, filename, rgb in views:
    data = rgb.get_data()
    print(camera_path, "shape:", data.shape, "dtype:", data.dtype)
    print(
        "min:", data.min(), "max:", data.max(),
        "mean:", round(float(data.mean()), 2),
    )
    out_path = os.path.join(OUT_DIR, filename)
    Image.fromarray(data[:, :, :3]).save(out_path)
    print("saved:", out_path)

app.close()
