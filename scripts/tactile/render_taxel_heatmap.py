"""Render the complete gripper with live 52-taxel force markers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RENDER_GPU = int(os.environ.get("SO101_TACTILE_HEATMAP_GPU", "0"))
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
from omni.physx.scripts import physicsUtils
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from scene_config import (
    PROJECT_ROOT,
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS,
)
from tactile_sensor import DPS2015EliteMvp, SENSOR_NAMES
from tactile_taxels import load_taxel_positions_m


USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
OUTPUT_PATH = PROJECT_ROOT / "renders" / "tactile_taxel_heatmap_gripper.png"
DATA_PATH = PROJECT_ROOT / "tactile_logs" / "taxel_heatmap_gripper.json"
RESOLUTION = (1600, 1100)
PROBE_RADIUS_M = 0.004
PROBE_MASS_KG = 0.010
PROBE_SPEED_M_S = 0.75
APPROACH_CLEARANCE_M = 0.0015
PENETRATION_M = 0.002
MARKER_RADIUS_M = 0.00065
MARKER_OUTWARD_OFFSET_M = 0.00045
PALETTE_SIZE = 16


def create_contact_probe(stage, path):
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -10.0))

    collision = UsdGeom.Sphere.Define(stage, path + "/Collision")
    collision.CreateRadiusAttr(PROBE_RADIUS_M)
    collision.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())

    material_path = "/World/Looks/TaxelHeatmapProbeMaterial"
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        UsdShade.Material.Define(stage, material_path)
        material_prim = stage.GetPrimAtPath(material_path)
        UsdPhysics.MaterialAPI.Apply(material_prim)
        material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        material.CreateCompliantContactStiffnessAttr().Set(1500.0)
        material.CreateCompliantContactDampingAttr().Set(5.0)
    physicsUtils.add_physics_material_to_prim(
        stage, collision.GetPrim(), material_path
    )

    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(PROBE_MASS_KG)
    PhysxSchema.PhysxRigidBodyAPI.Apply(
        root.GetPrim()
    ).CreateDisableGravityAttr().Set(True)
    PhysxSchema.PhysxContactReportAPI.Apply(
        root.GetPrim()
    ).CreateThresholdAttr().Set(0.0)
    return path


def sensor_point_world(stage, sensor_path, local_point):
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(sensor_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.asarray(
        transform.Transform(Gf.Vec3d(*local_point)), dtype=np.float32
    )


def drive_probes(stage, probes, outward_distance):
    positions = np.zeros((2, 3), dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    for index, sensor_path in enumerate(TACTILE_ROOT_PRIM_PATHS):
        apex = sensor_point_world(
            stage, sensor_path, TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME
        )
        offset_point = np.asarray(
            TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME, dtype=np.float64
        ).copy()
        offset_point[2] += outward_distance
        position = sensor_point_world(stage, sensor_path, offset_point)
        outward = position - apex
        outward /= np.linalg.norm(outward)
        positions[index] = position
        velocities[index] = -PROBE_SPEED_M_S * outward
    probes.set_world_poses(positions=positions)
    probes.set_velocities(linear_velocities=velocities)


def add_look_at_camera(stage, path, eye, target):
    view = Gf.Matrix4d()
    view.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(58.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    return path


def heat_color(value):
    value = float(np.clip(value, 0.0, 1.0))
    anchors = (
        (0.02, 0.08, 0.30),
        (0.00, 0.72, 1.00),
        (1.00, 0.88, 0.05),
        (1.00, 0.08, 0.02),
    )
    scaled = value * (len(anchors) - 1)
    lower = min(int(np.floor(scaled)), len(anchors) - 2)
    blend = scaled - lower
    return tuple(
        (1.0 - blend) * anchors[lower][axis]
        + blend * anchors[lower + 1][axis]
        for axis in range(3)
    )


def create_marker_material(stage, index, color):
    path = f"/World/Looks/TaxelHeat{index:02d}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(0.35 * channel for channel in color))
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def add_taxel_markers(stage, sample):
    magnitudes = sample.taxel_force_magnitudes
    scale_n = max(float(np.max(magnitudes)), 0.1)
    palette = [
        create_marker_material(
            stage,
            index,
            heat_color(index / (PALETTE_SIZE - 1)),
        )
        for index in range(PALETTE_SIZE)
    ]
    taxels = np.asarray(load_taxel_positions_m(), dtype=np.float64)
    world_positions = np.zeros((2, 52, 3), dtype=np.float64)

    UsdGeom.Scope.Define(stage, "/World/TaxelHeatmapMarkers")
    for sensor_index, sensor_path in enumerate(TACTILE_ROOT_PRIM_PATHS):
        transform = UsdGeom.Xformable(
            stage.GetPrimAtPath(sensor_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for taxel_index, local_position in enumerate(taxels):
            display_position = local_position.copy()
            display_position[2] += MARKER_OUTWARD_OFFSET_M
            world_position = transform.Transform(Gf.Vec3d(*display_position))
            world_positions[sensor_index, taxel_index] = world_position

            strength = float(magnitudes[sensor_index, taxel_index]) / scale_n
            palette_index = min(
                int(round(np.clip(strength, 0.0, 1.0) * (PALETTE_SIZE - 1))),
                PALETTE_SIZE - 1,
            )
            path = (
                f"/World/TaxelHeatmapMarkers/{SENSOR_NAMES[sensor_index]}"
                f"/Taxel_{taxel_index + 1:02d}"
            )
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr(MARKER_RADIUS_M)
            UsdGeom.Xformable(sphere).AddTranslateOp().Set(world_position)
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
                palette[palette_index],
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
    return world_positions, scale_n


def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def add_hud(image, sample, scale_n):
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (38, 35, 665, 260), radius=24, fill=(10, 14, 20, 218)
    )
    draw.text(
        (70, 60),
        "DP-S2015-ELITE  52-TAXEL FUSION",
        font=load_font(30),
        fill=(250, 252, 255),
    )
    draw.text(
        (70, 106),
        "One Isaac ContactSensor per fingertip -> 52 x 3",
        font=load_font(19),
        fill=(180, 191, 205),
    )
    y = 148
    colors = ((65, 190, 255), (255, 145, 40))
    for sensor_index, name in enumerate(SENSOR_NAMES):
        magnitudes = sample.taxel_force_magnitudes[sensor_index]
        active = (np.flatnonzero(magnitudes > 0.0) + 1).tolist()
        peak = int(np.argmax(magnitudes)) + 1
        draw.text(
            (70, y),
            (
                f"{name.upper()}: peak #{peak} "
                f"{float(np.max(magnitudes)):.1f} N  active {active}"
            ),
            font=load_font(18),
            fill=colors[sensor_index],
        )
        y += 37

    x0, y0, x1, y1 = 70, 226, 560, 244
    for x in range(x0, x1):
        value = (x - x0) / max(x1 - x0 - 1, 1)
        color = tuple(int(255 * channel) for channel in heat_color(value))
        draw.line((x, y0, x, y1), fill=color + (255,))
    draw.text(
        (570, 222),
        f"{scale_n:.1f} N",
        font=load_font(16),
        fill=(230, 234, 240),
    )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
carb.settings.get_settings().set("/rtx/post/aa/op", 1)

probe_paths = (
    create_contact_probe(stage, "/World/TaxelHeatmapProbeFixed"),
    create_contact_probe(stage, "/World/TaxelHeatmapProbeMoving"),
)
probes = RigidPrim(list(probe_paths))
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    rendering_dt=1.0 / 60.0,
)
model = DPS2015EliteMvp(stage, contact_partner_paths=probe_paths)
world.reset()
model.reset(world.current_time)

probes.set_world_poses(positions=np.full((2, 3), -10.0, dtype=np.float32))
for _ in range(120):
    world.step(render=False)
    model.maybe_sample(world.current_time, world.current_time_step_index)

best_sample = None
best_score = -1.0
for step in range(180):
    phase = (step + 1) / 180.0
    distance = (
        PROBE_RADIUS_M
        + APPROACH_CLEARANCE_M
        - phase * (APPROACH_CLEARANCE_M + PENETRATION_M)
    )
    drive_probes(stage, probes, distance)
    world.step(render=False)
    sample = model.maybe_sample(
        world.current_time, world.current_time_step_index
    )
    if sample is not None:
        score = float(np.sum(sample.taxel_force_magnitudes))
        if bool(np.all(sample.contacts)) and score > best_score:
            best_sample = sample
            best_score = score

if best_sample is None:
    model.close()
    app.close()
    raise RuntimeError("Both 52-taxel arrays must be active for rendering")

world_positions, scale_n = add_taxel_markers(stage, best_sample)
sensor_centers = []
for sensor_path in TACTILE_ROOT_PRIM_PATHS:
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(sensor_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    sensor_centers.append(transform.ExtractTranslation())
target = (sensor_centers[0] + sensor_centers[1]) * 0.5
camera_path = add_look_at_camera(
    stage,
    "/World/Cameras/TaxelHeatmapGripper",
    target + Gf.Vec3d(0.235, -0.180, 0.105),
    target,
)

render_product = rep.create.render_product(camera_path, RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])
for _ in range(4):
    world.step(render=True)
for _ in range(12):
    rep.orchestrator.step(rt_subframes=8)
pixels = rgb.get_data()
if pixels is None or not pixels.size:
    model.close()
    app.close()
    raise RuntimeError("Renderer returned no taxel heatmap data")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
annotated = add_hud(Image.fromarray(pixels[:, :, :3]), best_sample, scale_n)
annotated.save(OUTPUT_PATH)
with DATA_PATH.open("w", encoding="utf-8") as output:
    json.dump(
        {
            "sample": best_sample.to_dict(include_taxels=True),
            "taxel_world_positions_m": world_positions.tolist(),
            "marker_radius_m": MARKER_RADIUS_M,
            "display_scale_max_n": scale_n,
            "probe_mass_kg": PROBE_MASS_KG,
            "probe_speed_m_s": PROBE_SPEED_M_S,
            "note": (
                "Live PhysX contact visualization. Probe parameters are "
                "deterministic excitation values, not hardware calibration."
            ),
        },
        output,
        indent=2,
    )

print(
    json.dumps(
        {
            "output": str(OUTPUT_PATH),
            "data": str(DATA_PATH),
            "taxel_shape": list(best_sample.taxel_forces.shape),
            "peak_taxels": (
                np.argmax(best_sample.taxel_force_magnitudes, axis=1) + 1
            ).tolist(),
            "active_taxel_counts": np.count_nonzero(
                best_sample.taxel_force_magnitudes, axis=1
            ).tolist(),
            "display_scale_max_n": scale_n,
        },
        indent=2,
    )
)
rgb.detach([render_product])
render_product.destroy()
model.close()
app.close()
