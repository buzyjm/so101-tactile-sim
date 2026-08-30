"""Render an annotated two-fingertip aggregate-contact visualization."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VIS_GPU = int(os.environ.get("SO101_TACTILE_VIS_GPU", "0"))
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": VIS_GPU,
        "physics_gpu": VIS_GPU,
        "multi_gpu": False,
        "anti_aliasing": 1,
    }
)

import carb
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from scene_config import (
    PROJECT_ROOT,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS,
    TACTILE_SENSOR_SIZE_SENSOR_FRAME,
)
from tactile_sensor import DPS2015EliteMvp, SENSOR_NAMES


USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
OUTPUT_PATH = PROJECT_ROOT / "renders" / "tactile_contact_active.png"
DATA_PATH = PROJECT_ROOT / "tactile_logs" / "contact_visualization.json"
RESOLUTION = (1600, 1100)
PROBE_RADIUS = 0.004
APPROACH_CLEARANCE = 0.0015
FINAL_PENETRATION = 0.00005


def create_kinematic_probe(stage, path, color):
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -10.0))
    sphere = UsdGeom.Sphere.Define(stage, path + "/Collision")
    sphere.CreateRadiusAttr(PROBE_RADIUS)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    sphere.CreateDisplayOpacityAttr([1.0])
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.001)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    return path


def sensor_point_world(stage, sensor_path, local_z):
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(sensor_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return transform.Transform(Gf.Vec3d(0.0, 0.0, local_z))


def set_probe_distance(stage, probes, outward_distance):
    depth = TACTILE_SENSOR_SIZE_SENSOR_FRAME[2]
    positions = np.stack(
        [
            sensor_point_world(stage, path, -depth - outward_distance)
            for path in TACTILE_ROOT_PRIM_PATHS
        ]
    )
    probes.set_world_poses(positions=positions)


def add_look_at_camera(stage, path, eye, target):
    view = Gf.Matrix4d()
    view.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(48.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())


def create_heat_material(stage, path, color):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(0.12 * channel for channel in color))
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def bind_force_heat(stage, forces):
    materials = []
    for index, (path, force) in enumerate(zip(TACTILE_ROOT_PRIM_PATHS, forces)):
        strength = min(float(np.linalg.norm(force)) / 25.0, 1.0)
        color = (
            0.025 + 0.90 * strength,
            0.025 + 0.16 * (1.0 - strength),
            0.030,
        )
        material = create_heat_material(
            stage, f"/World/Looks/TactileContactHeat{index}", color
        )
        housing = stage.GetPrimAtPath(path + "/Housing")
        UsdShade.MaterialBindingAPI.Apply(housing).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        materials.append(material)
    return materials


def copy_display_dome(stage, source_path, path, transform, material):
    source = UsdGeom.Mesh(stage.GetPrimAtPath(source_path))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(source.GetPointsAttr().Get())
    mesh.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    mesh.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    mesh.CreateNormalsAttr(source.GetNormalsAttr().Get())
    mesh.SetNormalsInterpolation(source.GetNormalsInterpolation())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdGeom.Xformable(mesh).AddTransformOp().Set(transform)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def add_display_sphere(stage, path, position, color):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(PROBE_RADIUS)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(position)
    material = create_heat_material(stage, path + "Material", color)
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def add_contact_display(stage, heat_materials):
    center = Gf.Vec3d(0.62, -0.42, 0.62)
    lower_mount = center + Gf.Vec3d(0.0, 0.0, -0.020)
    upper_mount = center + Gf.Vec3d(0.0, 0.0, 0.020)

    lower_transform = Gf.Matrix4d(1.0)
    lower_transform.SetRotate(
        Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 180.0)
    )
    lower_transform.SetTranslateOnly(lower_mount)
    upper_transform = Gf.Matrix4d(1.0)
    upper_transform.SetTranslateOnly(upper_mount)

    copy_display_dome(
        stage,
        TACTILE_ROOT_PRIM_PATHS[0] + "/Housing",
        "/World/TactileContactDisplay/FixedDome",
        lower_transform,
        heat_materials[0],
    )
    copy_display_dome(
        stage,
        TACTILE_ROOT_PRIM_PATHS[1] + "/Housing",
        "/World/TactileContactDisplay/MovingDome",
        upper_transform,
        heat_materials[1],
    )

    depth = TACTILE_SENSOR_SIZE_SENSOR_FRAME[2]
    contact_offset = PROBE_RADIUS - FINAL_PENETRATION
    add_display_sphere(
        stage,
        "/World/TactileContactDisplay/FixedProbe",
        lower_mount + Gf.Vec3d(0.0, 0.0, depth + contact_offset),
        (0.06, 0.55, 1.0),
    )
    add_display_sphere(
        stage,
        "/World/TactileContactDisplay/MovingProbe",
        upper_mount - Gf.Vec3d(0.0, 0.0, depth + contact_offset),
        (1.0, 0.62, 0.04),
    )
    return center


def hide_scene_context(stage):
    for path in (
        "/World/Robot",
        "/World/Ground",
        "/World/Table",
        "/World/TableSeam",
        "/World/TableFrame",
        "/World/Bowl",
        "/World/TargetBall",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def load_font(size):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default()


def draw_force_bar(draw, x, y, width, value, limit, color, signed):
    draw.rounded_rectangle(
        (x, y, x + width, y + 24), radius=8, fill=(48, 52, 60, 235)
    )
    if signed:
        center = x + width // 2
        draw.line((center, y - 3, center, y + 27), fill=(205, 210, 218), width=2)
        endpoint = center + int((value / limit) * (width / 2))
        x0, x1 = sorted((center, endpoint))
    else:
        x0 = x
        x1 = x + int((max(0.0, value) / limit) * width)
    if x1 > x0:
        draw.rounded_rectangle(
            (x0, y, x1, y + 24), radius=7, fill=color
        )


def add_hud(image, sample):
    source = image.convert("RGB")
    canvas = Image.new("RGBA", source.size, (235, 238, 242, 255))
    scene_width = 920
    scene_height = int(round(source.height * scene_width / source.width))
    scene = source.resize((scene_width, scene_height), Image.Resampling.LANCZOS)
    canvas.paste(scene, (0, (source.height - scene_height) // 2))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    panel = (945, 45, 1560, 1045)
    draw.rounded_rectangle(panel, radius=28, fill=(13, 17, 23, 222))
    title_font = load_font(34)
    heading_font = load_font(28)
    body_font = load_font(22)
    small_font = load_font(18)
    draw.text((990, 78), "TACTILE CONTACT", font=title_font, fill=(255, 255, 255))
    draw.text(
        (990, 124),
        "DP-S2015-Elite aggregate-force MVP",
        font=small_font,
        fill=(170, 181, 194),
    )
    y = 185
    component_colors = (
        (65, 180, 255, 245),
        (180, 95, 255, 245),
        (255, 88, 42, 245),
    )
    component_limits = (10.0, 10.0, 25.0)
    for sensor_index, sensor_name in enumerate(SENSOR_NAMES):
        state = "CONTACT" if bool(sample.contacts[sensor_index]) else "CLEAR"
        state_color = (255, 103, 48) if state == "CONTACT" else (90, 220, 140)
        draw.text(
            (990, y),
            f"{sensor_name.upper()} FINGERTIP",
            font=heading_font,
            fill=(240, 243, 247),
        )
        draw.rounded_rectangle(
            (1350, y + 2, 1515, y + 34), radius=12, fill=state_color
        )
        draw.text((1370, y + 5), state, font=small_font, fill=(15, 17, 20))
        y += 55
        for axis, value, limit, color in zip(
            "xyz",
            sample.total_forces[sensor_index],
            component_limits,
            component_colors,
        ):
            draw.text(
                (990, y - 2),
                f"F{axis}  {float(value):+5.1f} N",
                font=body_font,
                fill=(224, 229, 235),
            )
            draw_force_bar(
                draw,
                1185,
                y + 1,
                325,
                float(value),
                limit,
                color,
                signed=axis != "z",
            )
            y += 46
        draw.text(
            (990, y),
            f"PhysX contact records: {int(sample.number_of_contacts[sensor_index])}",
            font=small_font,
            fill=(147, 158, 171),
        )
        y += 75
    draw.line((990, 890, 1515, 890), fill=(83, 91, 102), width=2)
    draw.text(
        (990, 914),
        "Orange/red shell = larger force",
        font=small_font,
        fill=(202, 207, 214),
    )
    draw.text(
        (990, 945),
        "Output: 2 fingertips x Fx/Fy/Fz",
        font=small_font,
        fill=(202, 207, 214),
    )
    draw.text(
        (990, 976),
        "52-taxel tensor: available; full-gripper heatmap rendered separately",
        font=small_font,
        fill=(255, 190, 88),
    )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
carb.settings.get_settings().set("/rtx/post/aa/op", 1)

probe_paths = (
    create_kinematic_probe(
        stage, "/World/TactileVizProbeFixed", (0.1, 0.75, 1.0)
    ),
    create_kinematic_probe(
        stage, "/World/TactileVizProbeMoving", (1.0, 0.65, 0.08)
    ),
)
probes = RigidPrim(list(probe_paths))
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    rendering_dt=1.0 / 60.0,
)
model = DPS2015EliteMvp(stage)
world.reset()
model.reset(world.current_time)

for _ in range(120):
    world.step(render=False)
    model.maybe_sample(world.current_time, world.current_time_step_index)

sample = None
for step in range(150):
    phase = (step + 1) / 150.0
    distance = (
        PROBE_RADIUS
        + APPROACH_CLEARANCE
        - phase * (APPROACH_CLEARANCE + FINAL_PENETRATION)
    )
    set_probe_distance(stage, probes, distance)
    world.step(render=False)
    current = model.maybe_sample(
        world.current_time, world.current_time_step_index
    )
    if current is not None and bool(np.all(current.contacts)):
        sample = current
        break

if sample is None or not bool(np.all(sample.contacts)):
    app.close()
    raise RuntimeError("Both tactile sensors must be active for visualization")

heat_materials = bind_force_heat(stage, sample.total_forces)
target = add_contact_display(stage, heat_materials)
hide_scene_context(stage)
camera_path = "/World/Cameras/TactileContactVisualization"
add_look_at_camera(
    stage,
    camera_path,
    target + Gf.Vec3d(0.090, -0.120, 0.008),
    target,
)

render_product = rep.create.render_product(camera_path, RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])
for _ in range(2):
    world.step(render=True)
for _ in range(16):
    rep.orchestrator.step(rt_subframes=8)
pixels = rgb.get_data()
if pixels is None or not pixels.size:
    app.close()
    raise RuntimeError("Renderer returned no tactile visualization data")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
annotated = add_hud(Image.fromarray(pixels[:, :, :3]), sample)
annotated.save(OUTPUT_PATH)
with DATA_PATH.open("w", encoding="utf-8") as output_file:
    json.dump(sample.to_dict(), output_file, indent=2)

print(json.dumps(sample.to_dict(), indent=2))
print("saved:", OUTPUT_PATH)
print("saved:", DATA_PATH)
rgb.detach([render_product])
render_product.destroy()
model.close()
app.close()
