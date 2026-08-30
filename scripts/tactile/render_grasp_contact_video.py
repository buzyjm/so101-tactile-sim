"""Render a grasp/release video with live two-fingertip force telemetry."""

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VIDEO_GPU = int(os.environ.get("SO101_TACTILE_VIDEO_GPU", "0"))
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": VIDEO_GPU,
        "physics_gpu": VIDEO_GPU,
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
    GRIPPER_DRIVE_PRIM_PATH,
    PROJECT_ROOT,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS,
    TACTILE_SENSOR_SIZE_SENSOR_FRAME,
)
from tactile_sensor import DPS2015EliteMvp, SENSOR_NAMES


USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
RAW_VIDEO_PATH = PROJECT_ROOT / "renders" / "tactile_grasp_live_raw.mp4"
POSTER_PATH = PROJECT_ROOT / "renders" / "tactile_grasp_live_poster.png"
LOG_PATH = PROJECT_ROOT / "tactile_logs" / "grasp_contact_video.json"
RESOLUTION = (1280, 720)
FPS = 30
PHYSICS_STEPS_PER_FRAME = int(round(TACTILE_MVP_PHYSICS_FREQUENCY_HZ / FPS))
VIDEO_SECONDS = 5.5
FRAME_COUNT = int(round(VIDEO_SECONDS * FPS))
BALL_PATH = "/World/TactileGraspDemoBall"
BALL_RADIUS = 0.0078
PROBE_RADIUS = 0.004
OPEN_CONTACT_GAP = 0.008
CONTACT_PENETRATION = 0.00005
OPEN_ANGLE_DEG = 20.0
CLOSED_ANGLE_DEG = 14.5


def smoothstep(value):
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def phase_and_target(time_seconds):
    if time_seconds < 0.8:
        return "OPEN", OPEN_ANGLE_DEG, time_seconds / 0.8
    if time_seconds < 2.5:
        blend = smoothstep((time_seconds - 0.8) / 1.7)
        target = OPEN_ANGLE_DEG + blend * (CLOSED_ANGLE_DEG - OPEN_ANGLE_DEG)
        return "CLOSING", target, blend
    if time_seconds < 3.8:
        return "HOLD", CLOSED_ANGLE_DEG, (time_seconds - 2.5) / 1.3
    blend = smoothstep((time_seconds - 3.8) / (VIDEO_SECONDS - 3.8))
    target = CLOSED_ANGLE_DEG + blend * (OPEN_ANGLE_DEG - CLOSED_ANGLE_DEG)
    return "RELEASE", target, blend


def create_material(stage, path, color, roughness=0.7):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material, shader


def create_demo_ball(stage):
    sphere = UsdGeom.Sphere.Define(stage, BALL_PATH)
    sphere.CreateRadiusAttr(BALL_RADIUS)
    material, _ = create_material(
        stage, "/World/Looks/TactileGraspDemoBall", (0.82, 0.04, 0.025), 0.86
    )
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(0.004)
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(sphere.GetPrim())
    physx_body.CreateDisableGravityAttr(True)
    physx_body.CreateLinearDampingAttr(8.0)
    physx_body.CreateAngularDampingAttr(8.0)
    return RigidPrim([BALL_PATH])


def create_kinematic_probe(stage, path):
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -10.0))
    sphere = UsdGeom.Sphere.Define(stage, path + "/Collision")
    sphere.CreateRadiusAttr(PROBE_RADIUS)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.001)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    return path


def sensor_point_world(stage, sensor_path, local_z):
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(sensor_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return transform.Transform(Gf.Vec3d(0.0, 0.0, local_z))


def set_probe_gap(stage, probes, gap):
    depth = TACTILE_SENSOR_SIZE_SENSOR_FRAME[2]
    outward_distance = PROBE_RADIUS + gap
    positions = np.stack(
        [
            sensor_point_world(stage, path, -depth - outward_distance)
            for path in TACTILE_ROOT_PRIM_PATHS
        ]
    )
    probes.set_world_poses(positions=positions)


def add_camera(stage, path, eye, target):
    view = Gf.Matrix4d()
    view.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(50.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())


def create_tactile_heat_materials(stage):
    materials = []
    shaders = []
    for index, path in enumerate(TACTILE_ROOT_PRIM_PATHS):
        material, shader = create_material(
            stage,
            f"/World/Looks/TactileVideoHeat{index}",
            (0.025, 0.18, 0.03),
            0.75,
        )
        housing = stage.GetPrimAtPath(path + "/Housing")
        UsdShade.MaterialBindingAPI.Apply(housing).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        materials.append(material)
        shaders.append(shader)
    return materials, shaders


def update_tactile_heat(shaders, forces):
    for shader, force in zip(shaders, forces):
        strength = min(float(np.linalg.norm(force)) / 25.0, 1.0)
        color = Gf.Vec3f(
            0.025 + 0.90 * strength,
            0.025 + 0.16 * (1.0 - strength),
            0.030,
        )
        shader.GetInput("diffuseColor").Set(color)


def copy_display_dome(stage, source_path, path, material):
    source = UsdGeom.Mesh(stage.GetPrimAtPath(source_path))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(source.GetPointsAttr().Get())
    mesh.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    mesh.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    mesh.CreateNormalsAttr(source.GetNormalsAttr().Get())
    mesh.SetNormalsInterpolation(source.GetNormalsInterpolation())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    transform_op = UsdGeom.Xformable(mesh).AddTransformOp()
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    return transform_op


def create_contact_display(stage, heat_materials):
    center = Gf.Vec3d(0.62, -0.42, 0.62)
    fixed_op = copy_display_dome(
        stage,
        TACTILE_ROOT_PRIM_PATHS[0] + "/Housing",
        "/World/TactileGraspDisplay/FixedDome",
        heat_materials[0],
    )
    moving_op = copy_display_dome(
        stage,
        TACTILE_ROOT_PRIM_PATHS[1] + "/Housing",
        "/World/TactileGraspDisplay/MovingDome",
        heat_materials[1],
    )
    ball = UsdGeom.Sphere.Define(stage, "/World/TactileGraspDisplay/Object")
    ball.CreateRadiusAttr(BALL_RADIUS)
    UsdGeom.Xformable(ball).AddTranslateOp().Set(center)
    ball_material, _ = create_material(
        stage,
        "/World/Looks/TactileGraspDisplayObject",
        (0.82, 0.035, 0.02),
        0.88,
    )
    UsdShade.MaterialBindingAPI.Apply(ball.GetPrim()).Bind(
        ball_material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    return center, (fixed_op, moving_op)


def update_contact_display(center, transform_ops, gap):
    depth = TACTILE_SENSOR_SIZE_SENSOR_FRAME[2]
    offset = BALL_RADIUS + gap + depth
    fixed_transform = Gf.Matrix4d(1.0)
    fixed_transform.SetRotate(
        Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 180.0)
    )
    fixed_transform.SetTranslateOnly(
        center - Gf.Vec3d(0.0, 0.0, offset)
    )
    moving_transform = Gf.Matrix4d(1.0)
    moving_transform.SetTranslateOnly(
        center + Gf.Vec3d(0.0, 0.0, offset)
    )
    transform_ops[0].Set(fixed_transform)
    transform_ops[1].Set(moving_transform)


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
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


TITLE_FONT = load_font(24)
HEADING_FONT = load_font(19)
BODY_FONT = load_font(15)
SMALL_FONT = load_font(12)


def draw_bar(draw, x, y, width, value, limit, color, signed):
    draw.rounded_rectangle(
        (x, y, x + width, y + 15), radius=5, fill=(49, 54, 62, 245)
    )
    if signed:
        center = x + width // 2
        draw.line((center, y - 2, center, y + 17), fill=(205, 211, 218), width=1)
        endpoint = center + int((value / limit) * width / 2)
        x0, x1 = sorted((center, endpoint))
    else:
        x0 = x
        x1 = x + int(max(0.0, value) / limit * width)
    if x1 > x0:
        draw.rounded_rectangle((x0, y, x1, y + 15), radius=5, fill=color)


def compose_frame(rgb, sample, phase, target_deg, frame_index):
    source = Image.fromarray(rgb[:, :, :3]).convert("RGB")
    canvas = Image.new("RGB", RESOLUTION, (235, 238, 242))
    scene_width = 785
    scene_height = int(round(source.height * scene_width / source.width))
    scene = source.resize((scene_width, scene_height), Image.Resampling.LANCZOS)
    canvas.paste(scene, (0, (RESOLUTION[1] - scene_height) // 2))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((805, 22, 1258, 698), radius=20, fill=(14, 18, 24, 235))
    draw.text((835, 45), "LIVE TACTILE GRASP", font=TITLE_FONT, fill=(255, 255, 255))
    draw.text(
        (835, 79),
        f"{phase}   gripper target {target_deg:4.1f} deg",
        font=BODY_FONT,
        fill=(255, 184, 82),
    )

    forces = (
        sample.total_forces
        if sample is not None
        else np.zeros((2, 3), dtype=np.float32)
    )
    contacts = (
        sample.contacts
        if sample is not None
        else np.zeros(2, dtype=bool)
    )
    limits = (10.0, 10.0, 25.0)
    colors = ((65, 180, 255, 245), (180, 95, 255, 245), (255, 88, 42, 245))
    y = 122
    for sensor_index, name in enumerate(SENSOR_NAMES):
        contact = bool(contacts[sensor_index])
        state = "CONTACT" if contact else "CLEAR"
        state_color = (255, 103, 48, 255) if contact else (72, 205, 125, 255)
        draw.text((835, y), name.upper(), font=HEADING_FONT, fill=(238, 242, 247))
        draw.rounded_rectangle((1110, y, 1225, y + 25), radius=9, fill=state_color)
        draw.text((1134, y + 4), state, font=SMALL_FONT, fill=(15, 18, 21))
        y += 38
        for axis, value, limit, color in zip("xyz", forces[sensor_index], limits, colors):
            draw.text(
                (835, y - 2),
                f"F{axis} {float(value):+5.1f} N",
                font=BODY_FONT,
                fill=(220, 226, 233),
            )
            draw_bar(draw, 985, y, 235, float(value), limit, color, axis != "z")
            y += 29
        y += 26

    draw.line((835, 520, 1225, 520), fill=(82, 91, 103, 255), width=1)
    draw.text(
        (835, 543),
        "Aggregate force: 2 fingertips x Fx/Fy/Fz",
        font=SMALL_FONT,
        fill=(198, 205, 213),
    )
    draw.text(
        (835, 565),
        "Display proxy driven by live robot sensors",
        font=SMALL_FONT,
        fill=(198, 205, 213),
    )
    draw.text(
        (835, 587),
        "52-taxel heatmap is not implemented yet",
        font=SMALL_FONT,
        fill=(255, 188, 84),
    )

    timeline_x0, timeline_x1 = 835, 1225
    timeline_y = 646
    draw.rounded_rectangle(
        (timeline_x0, timeline_y, timeline_x1, timeline_y + 10),
        radius=5,
        fill=(51, 57, 66, 255),
    )
    progress = frame_index / max(FRAME_COUNT - 1, 1)
    draw.rounded_rectangle(
        (
            timeline_x0,
            timeline_y,
            timeline_x0 + int(progress * (timeline_x1 - timeline_x0)),
            timeline_y + 10,
        ),
        radius=5,
        fill=(255, 108, 48, 255),
    )
    draw.text(
        (timeline_x0, timeline_y + 18),
        f"{frame_index / FPS:4.1f} s / {VIDEO_SECONDS:.1f} s",
        font=SMALL_FONT,
        fill=(170, 180, 191),
    )
    return np.asarray(canvas)


omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()
carb.settings.get_settings().set("/rtx/post/aa/op", 1)
gripper_drive = UsdPhysics.DriveAPI.Get(
    stage.GetPrimAtPath(GRIPPER_DRIVE_PRIM_PATH), "angular"
)
if not gripper_drive:
    app.close()
    raise RuntimeError(f"Missing gripper drive: {GRIPPER_DRIVE_PRIM_PATH}")
gripper_drive.CreateTargetPositionAttr().Set(OPEN_ANGLE_DEG)

probe_paths = (
    create_kinematic_probe(stage, "/World/TactileVideoProbeFixed"),
    create_kinematic_probe(stage, "/World/TactileVideoProbeMoving"),
)
probes = RigidPrim(list(probe_paths))
heat_materials, heat_shaders = create_tactile_heat_materials(stage)
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    rendering_dt=1.0 / FPS,
)
model = DPS2015EliteMvp(stage)
world.reset()
model.reset(world.current_time)
set_probe_gap(stage, probes, OPEN_CONTACT_GAP)
for _ in range(180):
    set_probe_gap(stage, probes, OPEN_CONTACT_GAP)
    world.step(render=False)
    model.maybe_sample(world.current_time, world.current_time_step_index)
model.reset(world.current_time)

camera_target, display_ops = create_contact_display(stage, heat_materials)
update_contact_display(camera_target, display_ops, OPEN_CONTACT_GAP)
hide_scene_context(stage)
camera_path = "/World/Cameras/TactileGraspVideo"
add_camera(
    stage,
    camera_path,
    camera_target + Gf.Vec3d(0.090, -0.120, 0.008),
    camera_target,
)
render_product = rep.create.render_product(camera_path, RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])
for _ in range(8):
    world.step(render=True)

RAW_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(RAW_VIDEO_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    RESOLUTION,
)
if not writer.isOpened():
    app.close()
    raise RuntimeError(f"Could not create video writer: {RAW_VIDEO_PATH}")

frame_log = []
poster_frame = None
poster_score = -1.0
for frame_index in range(FRAME_COUNT):
    time_seconds = frame_index / FPS
    phase, target_deg, phase_progress = phase_and_target(time_seconds)
    gripper_drive.CreateTargetPositionAttr().Set(float(target_deg))
    closure = (OPEN_ANGLE_DEG - target_deg) / (
        OPEN_ANGLE_DEG - CLOSED_ANGLE_DEG
    )
    contact_gap = OPEN_CONTACT_GAP - closure * (
        OPEN_CONTACT_GAP + CONTACT_PENETRATION
    )
    update_contact_display(camera_target, display_ops, contact_gap)

    for substep in range(PHYSICS_STEPS_PER_FRAME):
        set_probe_gap(stage, probes, contact_gap)
        world.step(render=substep == PHYSICS_STEPS_PER_FRAME - 1)
        model.maybe_sample(world.current_time, world.current_time_step_index)

    sample = model.latest
    forces = (
        sample.total_forces
        if sample is not None
        else np.zeros((2, 3), dtype=np.float32)
    )
    update_tactile_heat(heat_shaders, forces)
    pixels = rgb.get_data()
    if pixels is None or not pixels.size:
        writer.release()
        app.close()
        raise RuntimeError(f"Renderer returned no data at frame {frame_index}")
    frame_rgb = compose_frame(
        pixels,
        sample,
        phase,
        target_deg,
        frame_index,
    )
    writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    score = float(np.sum(np.maximum(forces[:, 2], 0.0)))
    if score > poster_score:
        poster_score = score
        poster_frame = frame_rgb.copy()
    frame_log.append(
        {
            "frame": frame_index,
            "video_time_s": time_seconds,
            "phase": phase,
            "phase_progress": phase_progress,
            "gripper_target_deg": target_deg,
            "display_contact_gap_m": contact_gap,
            "sample": sample.to_dict() if sample is not None else None,
        }
    )

writer.release()
if poster_frame is not None:
    Image.fromarray(poster_frame).save(POSTER_PATH)
with LOG_PATH.open("w", encoding="utf-8") as output_file:
    json.dump(
        {
            "fps": FPS,
            "frame_count": FRAME_COUNT,
            "duration_s": VIDEO_SECONDS,
            "ball_radius_m": BALL_RADIUS,
            "visualization": "isolated display proxy driven by live robot ContactSensors",
            "open_angle_deg": OPEN_ANGLE_DEG,
            "closed_angle_deg": CLOSED_ANGLE_DEG,
            "frames": frame_log,
        },
        output_file,
        indent=2,
    )

print("saved:", RAW_VIDEO_PATH)
print("saved:", POSTER_PATH)
print("saved:", LOG_PATH)
print("maximum displayed normal-force sum N:", poster_score)
rgb.detach([render_product])
render_product.destroy()
model.close()
app.close()
