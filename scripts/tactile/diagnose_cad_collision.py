"""Probe the authored tactile collision surfaces with PhysX scene queries."""

import os
import sys
from pathlib import Path

from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GPU = int(os.environ.get("SO101_TACTILE_TEST_GPU", "0"))
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": GPU,
        "physics_gpu": GPU,
        "multi_gpu": False,
    }
)

import carb
import omni.usd
from isaacsim.core.api import World
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, Usd, UsdGeom

from scene_config import (
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
    TACTILE_ROOT_PRIM_PATHS,
)


USD_PATH = os.environ.get(
    "SO101_SCENE_USD", str(PROJECT_ROOT / "lab_scene.usda")
)
QUERY_RADIUS_M = float(
    os.environ.get("SO101_TACTILE_QUERY_RADIUS_M", "0.00025")
)
OFFSETS_M = (0.004, 0.002, 0.001, 0.0, -0.001, -0.002, -0.004)


omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0)
world.reset()
for _ in range(30):
    world.step(render=False)

query = get_physx_scene_query_interface()
apex_local = Gf.Vec3d(*TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME)
outward_sign = 1.0 if apex_local[2] >= 0.0 else -1.0

for sensor_path in TACTILE_ROOT_PRIM_PATHS:
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(sensor_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    print("sensor:", sensor_path)
    print("  origin_world:", tuple(transform.ExtractTranslation()))
    print("  apex_world:", tuple(transform.Transform(apex_local)))
    for offset in OFFSETS_M:
        local = apex_local + Gf.Vec3d(
            0.0, 0.0, outward_sign * offset
        )
        position = transform.Transform(local)
        hits = []

        def report_hit(hit):
            hits.append((str(hit.rigid_body), str(hit.collision)))
            return True

        query.overlap_sphere(
            QUERY_RADIUS_M,
            carb.Float3(*position),
            report_hit,
            False,
        )
        print(
            f"  offset_mm={offset * 1000:+.2f}",
            "position=",
            tuple(round(float(value), 6) for value in position),
            "hits=",
            hits,
        )

app.close()
