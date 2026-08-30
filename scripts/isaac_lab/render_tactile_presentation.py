"""Render an Isaac Lab 104-taxel heatmap or standalone force vector field."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument(
    "--visualization",
    choices=("heatmap", "vector"),
    default="heatmap",
    help="Render taxel magnitude spheres or a separate 3-axis force field.",
)
parser.add_argument(
    "--quality",
    choices=("preview", "final"),
    default="preview",
    help="RTX accumulation preset; final is intended for an otherwise idle GPU.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

from scene_config import (
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
)
from so101_isaac_lab.scene_cfg import TactilePresentationSceneCfg
from tactile_taxels import load_taxel_positions_m


LAB_SCENE_USD = PROJECT_ROOT / "lab_scene.usda"
if args.visualization == "vector":
    RAW_OUTPUT = (
        PROJECT_ROOT / "renders" / "isaac_lab_taxel_vector_field_raw.png"
    )
    OUTPUT = PROJECT_ROOT / "renders" / "isaac_lab_taxel_vector_field.png"
    DATA_OUTPUT = (
        PROJECT_ROOT / "tactile_logs" / "isaac_lab_taxel_vector_field.json"
    )
else:
    RAW_OUTPUT = (
        PROJECT_ROOT / "renders" / "isaac_lab_tactile_scene_raw.png"
    )
    OUTPUT = PROJECT_ROOT / "renders" / "isaac_lab_tactile_presentation.png"
    DATA_OUTPUT = (
        PROJECT_ROOT
        / "tactile_logs"
        / "isaac_lab_tactile_presentation.json"
    )
MARKER_RADIUS_M = 0.00068
MARKER_OUTWARD_OFFSET_M = 0.00045
VECTOR_OUTWARD_OFFSET_M = 0.00075
VECTOR_DISPLAY_THRESHOLD_N = 0.1
VECTOR_MIN_LENGTH_M = 0.006
VECTOR_MAX_LENGTH_M = 0.032
VECTOR_SHAFT_RADIUS_M = 0.00042
VECTOR_HEAD_RADIUS_M = 0.00125
PROBE_PENETRATION_M = 0.002
PROBE_SPEED_M_S = 0.75
PALETTE_SIZE = 20


def _add_lab_background(stage) -> None:
    """Reference the visual laboratory around the clone-safe Lab robot."""
    prim_types = {
        "Looks": "Scope",
        "Ground": "Xform",
        "Table": "Xform",
        "TableFrame": "Scope",
        "TableSeam": "Xform",
        "TargetBall": "Xform",
        "Bowl": "Xform",
        "Lights": "Scope",
    }
    for name, prim_type in prim_types.items():
        target = stage.DefinePrim(f"/World/{name}", prim_type)
        target.GetReferences().AddReference(
            str(LAB_SCENE_USD), f"/World/{name}"
        )

    key = UsdLux.RectLight.Define(stage, "/World/Lights/TactileKey")
    key.CreateIntensityAttr(1800.0)
    key.CreateExposureAttr(1.0)
    key.CreateWidthAttr(0.45)
    key.CreateHeightAttr(0.30)
    key.CreateColorAttr(Gf.Vec3f(0.78, 0.88, 1.0))
    xform = UsdGeom.Xformable(key)
    xform.AddTranslateOp().Set(Gf.Vec3d(0.15, -0.25, 0.65))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 18.0))


def _drive_probe(sensor, probe) -> None:
    apex = torch.tensor(
        TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
        dtype=torch.float32,
        device=args.device,
    ).reshape(1, 3)
    contact_point = apex.clone()
    contact_point[:, 2] += PROBE_PENETRATION_M
    position_w = sensor.sensor_points_to_world(contact_point)
    quaternion_w = torch.zeros((1, 4), device=args.device)
    quaternion_w[:, 3] = 1.0
    probe.write_root_pose_to_sim_index(
        root_pose=torch.cat((position_w, quaternion_w), dim=1)
    )

    velocity_sensor = torch.zeros((1, 3), device=args.device)
    velocity_sensor[:, 2] = -PROBE_SPEED_M_S
    velocity_w = sensor.sensor_vectors_to_world(velocity_sensor)
    probe.write_root_velocity_to_sim_index(
        root_velocity=torch.cat(
            (velocity_w, torch.zeros_like(velocity_w)), dim=1
        )
    )


def _combined_snapshot(scene) -> dict[str, torch.Tensor]:
    fixed = scene["tactile_fixed"].tactile_data
    moving = scene["tactile_moving"].tactile_data
    fields = (
        "raw_total_forces",
        "total_forces",
        "raw_taxel_forces",
        "taxel_forces",
        "taxel_total_forces",
        "force_magnitudes",
        "contacts",
        "accepted_contact_counts",
        "rejected_contact_counts",
        "nearest_taxel_distances_m",
    )
    result = {}
    for field in fields:
        value = torch.cat((getattr(fixed, field), getattr(moving, field)), dim=1)
        result[field] = value[0].detach().clone()
    return result


def _quat_rotate_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., :3]
    real = quaternion[..., 3:4]
    cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + real * cross + torch.linalg.cross(xyz, cross, dim=-1)


def _heat_color(value: float) -> tuple[float, float, float]:
    value = float(np.clip(value, 0.0, 1.0))
    anchors = (
        (0.018, 0.055, 0.20),
        (0.00, 0.68, 1.00),
        (0.98, 0.88, 0.06),
        (1.00, 0.08, 0.015),
    )
    scaled = value * (len(anchors) - 1)
    lower = min(int(np.floor(scaled)), len(anchors) - 2)
    blend = scaled - lower
    return tuple(
        (1.0 - blend) * anchors[lower][axis]
        + blend * anchors[lower + 1][axis]
        for axis in range(3)
    )


def _create_material(stage, index: int, color) -> UsdShade.Material:
    path = f"/World/Looks/IsaacLabTaxelHeat{index:02d}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.38)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(0.42 * component for component in color))
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def _add_taxel_markers(stage, scene, snapshot):
    magnitudes = snapshot["force_magnitudes"]
    scale_n = max(float(magnitudes.max()), 0.1)
    materials = [
        _create_material(stage, index, _heat_color(index / (PALETTE_SIZE - 1)))
        for index in range(PALETTE_SIZE)
    ]
    local = torch.as_tensor(
        load_taxel_positions_m(),
        dtype=torch.float32,
        device=args.device,
    )
    local = local.clone()
    local[:, 2] += MARKER_OUTWARD_OFFSET_M
    world_positions = []
    UsdGeom.Scope.Define(stage, "/World/IsaacLabTaxelHeatmap")

    for finger_index, sensor_name in enumerate(
        ("tactile_fixed", "tactile_moving")
    ):
        sensor = scene[sensor_name]
        position_w, quaternion_w = sensor.sensor_frame_pose_w()
        rotated = _quat_rotate_xyzw(
            quaternion_w[0].expand(local.shape[0], -1), local
        )
        positions = rotated + position_w[0]
        world_positions.append(positions)
        finger_label = "Fixed" if finger_index == 0 else "Moving"
        UsdGeom.Scope.Define(
            stage, f"/World/IsaacLabTaxelHeatmap/{finger_label}"
        )
        for taxel_index in range(52):
            strength = float(magnitudes[finger_index, taxel_index]) / scale_n
            palette_index = min(
                int(round(np.clip(strength, 0.0, 1.0) * (PALETTE_SIZE - 1))),
                PALETTE_SIZE - 1,
            )
            path = (
                f"/World/IsaacLabTaxelHeatmap/{finger_label}/"
                f"Taxel_{taxel_index + 1:02d}"
            )
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr(MARKER_RADIUS_M)
            point = positions[taxel_index].tolist()
            UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*point))
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
                materials[palette_index],
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
    return torch.stack(world_positions).detach().cpu(), scale_n


def _create_vector_material(
    stage, label: str, color: tuple[float, float, float]
) -> UsdShade.Material:
    path = f"/World/Looks/IsaacLabTaxelVector{label}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(0.55 * component for component in color))
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def _set_primitive_pose(primitive, center, direction) -> None:
    rotation = Gf.Rotation(
        Gf.Vec3d(0.0, 0.0, 1.0),
        Gf.Vec3d(*[float(value) for value in direction]),
    )
    xform = UsdGeom.Xformable(primitive)
    xform.AddTranslateOp().Set(
        Gf.Vec3d(*[float(value) for value in center])
    )
    quaternion = rotation.GetQuat()
    xform.AddOrientOp().Set(
        Gf.Quatf(
            float(quaternion.GetReal()),
            Gf.Vec3f(
                *[float(value) for value in quaternion.GetImaginary()]
            ),
        )
    )


def _add_force_arrow(
    stage,
    path: str,
    origin: np.ndarray,
    unit_direction: np.ndarray,
    length_m: float,
    material: UsdShade.Material,
) -> None:
    head_length_m = min(0.0045, max(0.0020, 0.35 * length_m))
    shaft_length_m = length_m - head_length_m
    shaft_center = origin + unit_direction * (0.5 * shaft_length_m)
    head_center = origin + unit_direction * (
        shaft_length_m + 0.5 * head_length_m
    )

    shaft = UsdGeom.Cylinder.Define(stage, path + "/Shaft")
    shaft.CreateAxisAttr(UsdGeom.Tokens.z)
    shaft.CreateRadiusAttr(VECTOR_SHAFT_RADIUS_M)
    shaft.CreateHeightAttr(shaft_length_m)
    _set_primitive_pose(shaft, shaft_center, unit_direction)
    UsdShade.MaterialBindingAPI.Apply(shaft.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )

    head = UsdGeom.Cone.Define(stage, path + "/Head")
    head.CreateAxisAttr(UsdGeom.Tokens.z)
    head.CreateRadiusAttr(VECTOR_HEAD_RADIUS_M)
    head.CreateHeightAttr(head_length_m)
    _set_primitive_pose(head, head_center, unit_direction)
    UsdShade.MaterialBindingAPI.Apply(head.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def _add_taxel_vectors(stage, scene, snapshot) -> dict[str, object]:
    magnitudes = snapshot["force_magnitudes"]
    scale_n = max(float(magnitudes.max()), VECTOR_DISPLAY_THRESHOLD_N)
    local_positions = torch.as_tensor(
        load_taxel_positions_m(),
        dtype=torch.float32,
        device=args.device,
    )
    display_local_positions = local_positions.clone()
    display_local_positions[:, 2] += VECTOR_OUTWARD_OFFSET_M

    materials = (
        _create_vector_material(stage, "Fixed", (0.02, 0.78, 1.0)),
        _create_vector_material(stage, "Moving", (1.0, 0.38, 0.025)),
    )
    world_positions = []
    vector_origins = []
    world_forces = []
    display_lengths = torch.zeros_like(magnitudes)
    vector_ends = []
    active_indices = []
    UsdGeom.Scope.Define(stage, "/World/IsaacLabTaxelVectorField")

    for finger_index, sensor_name in enumerate(
        ("tactile_fixed", "tactile_moving")
    ):
        sensor = scene[sensor_name]
        position_w, quaternion_w = sensor.sensor_frame_pose_w()
        expanded_quaternion = quaternion_w[0].expand(
            local_positions.shape[0], -1
        )
        positions = (
            _quat_rotate_xyzw(expanded_quaternion, local_positions)
            + position_w[0]
        )
        origins = (
            _quat_rotate_xyzw(expanded_quaternion, display_local_positions)
            + position_w[0]
        )
        forces = _quat_rotate_xyzw(
            expanded_quaternion, snapshot["taxel_forces"][finger_index]
        )
        force_magnitudes = torch.linalg.vector_norm(forces, dim=-1)
        active = force_magnitudes >= VECTOR_DISPLAY_THRESHOLD_N
        ratio = (force_magnitudes / scale_n).clamp(0.0, 1.0)
        lengths = torch.where(
            active,
            VECTOR_MIN_LENGTH_M
            + ratio * (VECTOR_MAX_LENGTH_M - VECTOR_MIN_LENGTH_M),
            torch.zeros_like(ratio),
        )
        directions = forces / force_magnitudes.unsqueeze(-1).clamp_min(1.0e-12)
        ends = origins + directions * lengths.unsqueeze(-1)

        world_positions.append(positions)
        vector_origins.append(origins)
        world_forces.append(forces)
        display_lengths[finger_index] = lengths
        vector_ends.append(ends)
        indices = torch.nonzero(active, as_tuple=False).flatten()
        active_indices.append((indices + 1).detach().cpu().tolist())

        finger_label = "Fixed" if finger_index == 0 else "Moving"
        UsdGeom.Scope.Define(
            stage, f"/World/IsaacLabTaxelVectorField/{finger_label}"
        )
        for taxel_index in indices.detach().cpu().tolist():
            path = (
                f"/World/IsaacLabTaxelVectorField/{finger_label}/"
                f"Taxel_{taxel_index + 1:02d}"
            )
            _add_force_arrow(
                stage,
                path,
                origins[taxel_index].detach().cpu().numpy(),
                directions[taxel_index].detach().cpu().numpy(),
                float(lengths[taxel_index]),
                materials[finger_index],
            )

    return {
        "world_positions": torch.stack(world_positions).detach().cpu(),
        "vector_origins": torch.stack(vector_origins).detach().cpu(),
        "world_forces": torch.stack(world_forces).detach().cpu(),
        "display_lengths": display_lengths.detach().cpu(),
        "vector_ends": torch.stack(vector_ends).detach().cpu(),
        "active_indices": active_indices,
        "scale_n": scale_n,
    }


def _hide_contact_probes(stage) -> None:
    for probe_name in ("ProbeFixed", "ProbeMoving"):
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{probe_name}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def _set_camera_pose(scene):
    centers = []
    for sensor_name in ("tactile_fixed", "tactile_moving"):
        position_w, _ = scene[sensor_name].sensor_frame_pose_w()
        centers.append(position_w[0].detach().cpu().numpy())
    target_np = 0.5 * (centers[0] + centers[1])
    eye_np = target_np + np.array((0.245, -0.185, 0.115))
    scene["presentation_camera"].set_world_poses_from_view(
        eyes=eye_np.reshape(1, 3).astype(np.float32),
        targets=target_np.reshape(1, 3).astype(np.float32),
    )
    return eye_np, target_np


def _camera_rgb_numpy(scene) -> np.ndarray:
    pixels = scene["presentation_camera"].data.output["rgb"]
    if hasattr(pixels, "torch"):
        pixels = pixels.torch
    if not isinstance(pixels, torch.Tensor):
        pixels = torch.as_tensor(pixels, device="cpu")
    pixels = pixels[0, :, :, :3].detach().cpu().numpy()
    return pixels


def _font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _compose_hud(image: Image.Image, snapshot, scale_n: float) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (38, 34, 760, 302), radius=26, fill=(9, 13, 19, 224)
    )
    draw.text(
        (70, 58),
        "ISAAC LAB  ·  104-TAXEL CONTACT",
        font=_font(32),
        fill=(248, 251, 255, 255),
    )
    draw.text(
        (70, 105),
        "2 ContactSensors  |  240 Hz PhysX  |  83.3 Hz tactile",
        font=_font(18),
        fill=(169, 183, 199, 255),
    )

    colors = ((35, 194, 255, 255), (255, 128, 35, 255))
    names = ("FIXED", "MOVING")
    magnitudes = snapshot["force_magnitudes"].cpu().numpy()
    total_forces = snapshot["total_forces"].cpu().numpy()
    y = 153
    for finger_index, name in enumerate(names):
        active = np.flatnonzero(magnitudes[finger_index] > 0.0) + 1
        peak = int(np.argmax(magnitudes[finger_index])) + 1
        force = total_forces[finger_index]
        draw.text(
            (70, y),
            (
                f"{name}  peak #{peak:02d}  "
                f"{float(magnitudes[finger_index].max()):.1f} N  "
                f"active {len(active)}"
            ),
            font=_font(20),
            fill=colors[finger_index],
        )
        draw.text(
            (455, y),
            f"F=({force[0]:+.1f}, {force[1]:+.1f}, {force[2]:+.1f}) N",
            font=_font(18),
            fill=(229, 234, 241, 255),
        )
        y += 43

    x0, y0, x1, y1 = 70, 254, 620, 276
    for x in range(x0, x1):
        value = (x - x0) / max(x1 - x0 - 1, 1)
        color = tuple(int(255 * component) for component in _heat_color(value))
        draw.line((x, y0, x, y1), fill=color + (255,))
    draw.text(
        (635, 251),
        f"{scale_n:.1f} N",
        font=_font(17),
        fill=(235, 239, 245, 255),
    )

    draw.rounded_rectangle(
        (1170, 852, 1560, 960), radius=20, fill=(9, 13, 19, 210)
    )
    draw.text(
        (1200, 872),
        "POLICY INPUT  312-D",
        font=_font(22),
        fill=(247, 249, 252, 255),
    )
    draw.text(
        (1200, 910),
        "debug tensor  2 × 52 × 3",
        font=_font(17),
        fill=(165, 181, 199, 255),
    )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def _compose_vector_hud(
    image: Image.Image, snapshot, vector_data: dict[str, object]
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (38, 34, 1030, 318), radius=26, fill=(9, 13, 19, 226)
    )
    draw.text(
        (70, 56),
        "ISAAC LAB  ·  104-TAXEL FORCE VECTOR FIELD",
        font=_font(31),
        fill=(248, 251, 255, 255),
    )
    draw.text(
        (70, 103),
        "origin = taxel position  |  direction = F_world  |  length = |F|",
        font=_font(18),
        fill=(169, 183, 199, 255),
    )
    draw.text(
        (70, 134),
        "2 ContactSensors  |  240 Hz PhysX  |  83.3 Hz tactile  |  threshold 0.1 N",
        font=_font(17),
        fill=(143, 160, 180, 255),
    )

    colors = ((35, 205, 255, 255), (255, 112, 30, 255))
    names = ("FIXED", "MOVING")
    magnitudes = snapshot["force_magnitudes"].cpu().numpy()
    total_forces = snapshot["total_forces"].cpu().numpy()
    active_indices = vector_data["active_indices"]
    y = 178
    for finger_index, name in enumerate(names):
        peak = int(np.argmax(magnitudes[finger_index])) + 1
        force = total_forces[finger_index]
        draw.text(
            (70, y),
            (
                f"{name}  peak #{peak:02d}  "
                f"{float(magnitudes[finger_index].max()):.1f} N  "
                f"arrows {len(active_indices[finger_index])}"
            ),
            font=_font(20),
            fill=colors[finger_index],
        )
        draw.text(
            (510, y),
            f"F=({force[0]:+.1f}, {force[1]:+.1f}, {force[2]:+.1f}) N",
            font=_font(18),
            fill=(229, 234, 241, 255),
        )
        y += 42

    draw.line((70, 280, 142, 280), fill=colors[0], width=6)
    draw.polygon(
        ((142, 280), (128, 270), (128, 290)), fill=colors[0]
    )
    draw.text(
        (158, 268), "fixed", font=_font(16), fill=(220, 229, 240, 255)
    )
    draw.line((250, 280, 322, 280), fill=colors[1], width=6)
    draw.polygon(
        ((322, 280), (308, 270), (308, 290)), fill=colors[1]
    )
    draw.text(
        (338, 268), "moving", font=_font(16), fill=(220, 229, 240, 255)
    )
    draw.text(
        (470, 268),
        "display length 6–32 mm; exact vectors saved in JSON",
        font=_font(16),
        fill=(151, 169, 188, 255),
    )

    draw.rounded_rectangle(
        (1120, 838, 1560, 960), radius=20, fill=(9, 13, 19, 214)
    )
    draw.text(
        (1150, 858),
        "104 POSITIONS RECORDED",
        font=_font(21),
        fill=(247, 249, 252, 255),
    )
    draw.text(
        (1150, 897),
        "force tensor  2 × 52 × 3",
        font=_font(17),
        fill=(165, 181, 199, 255),
    )
    draw.text(
        (1150, 927),
        "sensor + world coordinate frames",
        font=_font(15),
        fill=(140, 158, 178, 255),
    )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def _tensor_json(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()


def main() -> None:
    if args.steps < 12:
        raise ValueError("--steps must be at least 12")

    sim_cfg = SimulationCfg(
        dt=1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
        device=args.device,
        render_interval=1,
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(
        TactilePresentationSceneCfg(
            num_envs=1,
            env_spacing=1.0,
            replicate_physics=False,
        )
    )
    stage = sim.stage
    _add_lab_background(stage)
    carb.settings.get_settings().set("/rtx/post/aa/op", 1)
    sim.reset()
    scene.reset()

    best_snapshot = None
    best_score = -1.0
    for _ in range(args.steps):
        _drive_probe(scene["tactile_fixed"], scene["probe_fixed"])
        _drive_probe(scene["tactile_moving"], scene["probe_moving"])
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_cfg.dt)
        snapshot = _combined_snapshot(scene)
        score = float(snapshot["force_magnitudes"].sum())
        if bool(snapshot["contacts"].all()) and score > best_score:
            best_snapshot = snapshot
            best_score = score

    if best_snapshot is None:
        raise RuntimeError("Both tactile arrays must be active for rendering")

    vector_data = None
    if args.visualization == "vector":
        vector_data = _add_taxel_vectors(stage, scene, best_snapshot)
        world_positions = vector_data["world_positions"]
        scale_n = vector_data["scale_n"]
        _hide_contact_probes(stage)
    else:
        world_positions, scale_n = _add_taxel_markers(
            stage, scene, best_snapshot
        )
    camera_eye, camera_target = _set_camera_pose(scene)
    render_frames = 1 if args.quality == "preview" else 4
    for _ in range(render_frames):
        _drive_probe(scene["tactile_fixed"], scene["probe_fixed"])
        _drive_probe(scene["tactile_moving"], scene["probe_moving"])
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_cfg.dt)
    pixels = _camera_rgb_numpy(scene)
    if pixels is None or not pixels.size:
        raise RuntimeError("Isaac Lab camera returned no RGB data")

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    raw = Image.fromarray(pixels)
    raw.save(RAW_OUTPUT)
    if vector_data is not None:
        _compose_vector_hud(raw, best_snapshot, vector_data).save(OUTPUT)
    else:
        _compose_hud(raw, best_snapshot, scale_n).save(OUTPUT)

    raw_error = (
        best_snapshot["raw_taxel_forces"].sum(dim=1)
        - best_snapshot["raw_total_forces"]
    ).abs().max()
    payload = {
        "renderer": "Isaac Lab InteractiveScene + RTX",
        "render_quality": args.quality,
        "visualization": args.visualization,
        "physics_hz": TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
        "tactile_output_hz": 83.3,
        "taxel_force_shape": list(best_snapshot["taxel_forces"].shape),
        "policy_feature_shape": [312],
        "contacts": _tensor_json(best_snapshot["contacts"]),
        "peak_taxels": (
            best_snapshot["force_magnitudes"].argmax(dim=1) + 1
        ).tolist(),
        "active_taxel_counts": best_snapshot["force_magnitudes"].gt(0.0)
        .sum(dim=1)
        .tolist(),
        "total_forces_n": _tensor_json(best_snapshot["total_forces"]),
        "taxel_force_frame": "sensor",
        "taxel_forces_n": _tensor_json(best_snapshot["taxel_forces"]),
        "force_magnitudes_n": _tensor_json(
            best_snapshot["force_magnitudes"]
        ),
        "accepted_contact_counts": _tensor_json(
            best_snapshot["accepted_contact_counts"]
        ),
        "rejected_contact_counts": _tensor_json(
            best_snapshot["rejected_contact_counts"]
        ),
        "raw_conservation_max_abs_error_n": float(raw_error),
        "taxel_world_positions_m": world_positions.tolist(),
        "display_scale_max_n": scale_n,
        "camera_eye_m": camera_eye.tolist(),
        "camera_target_m": camera_target.tolist(),
        "raw_output": str(RAW_OUTPUT),
        "annotated_output": str(OUTPUT),
        "note": (
            "Live PhysX contact render. Dynamic probes are deterministic "
            "regression excitation, not calibrated hardware dynamics."
        ),
    }
    if vector_data is not None:
        payload.update(
            {
                "taxel_world_force_frame": "world",
                "taxel_forces_world_n": vector_data[
                    "world_forces"
                ].tolist(),
                "vector_display_threshold_n": VECTOR_DISPLAY_THRESHOLD_N,
                "vector_display_length_range_m": [
                    VECTOR_MIN_LENGTH_M,
                    VECTOR_MAX_LENGTH_M,
                ],
                "vector_display_lengths_m": vector_data[
                    "display_lengths"
                ].tolist(),
                "vector_display_origins_world_m": vector_data[
                    "vector_origins"
                ].tolist(),
                "vector_display_ends_world_m": vector_data[
                    "vector_ends"
                ].tolist(),
                "active_taxel_indices_1based": vector_data[
                    "active_indices"
                ],
                "probe_geometry_visible": False,
            }
        )
    else:
        payload["marker_radius_m"] = MARKER_RADIUS_M
    DATA_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "passed": True,
                "output": str(OUTPUT),
                "raw_output": str(RAW_OUTPUT),
                "data": str(DATA_OUTPUT),
                "visualization": args.visualization,
                "taxel_force_shape": payload["taxel_force_shape"],
                "contacts": payload["contacts"],
                "peak_taxels": payload["peak_taxels"],
                "active_taxel_counts": payload["active_taxel_counts"],
                "display_scale_max_n": scale_n,
                "raw_conservation_max_abs_error_n": float(raw_error),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
