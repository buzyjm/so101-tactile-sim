import math
import os

from isaacsim import SimulationApp

BUILD_GPU = int(os.environ.get("SO101_BUILD_GPU", "0"))
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": BUILD_GPU,
        "physics_gpu": BUILD_GPU,
        "multi_gpu": False,
    }
)

import omni.isaac.IsaacSensorSchema as IsaacSensorSchema

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from scipy.spatial import ConvexHull

from scene_config import (
    BIRDSEYE_CAMERA_EYE,
    CONTACT_OFFSET_M,
    GRIPPER_CONTACT_PRIM_PATHS,
    PHYSICS_MATERIALS,
    REST_OFFSET_M,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    BIRDSEYE_CAMERA_FOCAL_LENGTH_MM,
    BIRDSEYE_CAMERA_TARGET,
    BOWL_CENTER_XY_PLACEHOLDER,
    BOWL_COLOR,
    BOWL_HEIGHT,
    BOWL_INNER_BOTTOM_DIAMETER,
    BOWL_OPACITY,
    BOWL_OUTER_TOP_DIAMETER,
    BOWL_WALL_THICKNESS,
    CAMERA_HORIZONTAL_APERTURE_MM_PLACEHOLDER,
    CAMERA_VERTICAL_APERTURE_MM_PLACEHOLDER,
    DEBUG_CAMERA_EYE_PLACEHOLDER,
    DEBUG_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER,
    DEBUG_CAMERA_TARGET_PLACEHOLDER,
    FLOOR_Z,
    FIXED_ADAPTER_CAD_BASIS_ROWS,
    FIXED_ADAPTER_CAD_MESH_RELATIVE_PATH,
    FIXED_ADAPTER_CAD_PRIM_PATH,
    FIXED_ADAPTER_CAD_TRANSLATION_MM,
    FIXED_ADAPTER_CAD_USD_PATH,
    FIXED_SENSOR_CAD_BASIS_ROWS,
    FIXED_SENSOR_CAD_TRANSLATION_MM,
    FIXED_STOCK_COLLISION_PRIM_PATH,
    FIXED_STOCK_VISUAL_PRIM_PATH,
    FIXED_TACTILE_BASIS_ROWS,
    FIXED_TACTILE_ROOT_PRIM_PATH,
    FIXED_TACTILE_TRANSLATION_PLACEHOLDER,
    GRAVITY_M_S2,
    GRIPPER_DRIVE_PRIM_PATH,
    GRIPPER_LINK_PRIM_PATH,
    GROUND_COLOR,
    GROUND_THICKNESS,
    MOVING_ADAPTER_CAD_BASIS_ROWS,
    MOVING_ADAPTER_CAD_MESH_RELATIVE_PATH,
    MOVING_ADAPTER_CAD_PRIM_PATH,
    MOVING_ADAPTER_CAD_TRANSLATION_MM,
    MOVING_ADAPTER_CAD_USD_PATH,
    MOVING_SENSOR_CAD_BASIS_ROWS,
    MOVING_SENSOR_CAD_TRANSLATION_MM,
    MOVING_STOCK_COLLISION_PRIM_PATH,
    MOVING_STOCK_VISUAL_PRIM_PATH,
    PROJECT_ROOT,
    ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG,
    SCENE_LAYOUT_MODE,
    TABLE_CENTER,
    TABLE_COLOR,
    TABLE_FRAME_COLOR,
    TABLE_LENGTH,
    TABLE_PEDESTAL_COLUMN_CENTER_Z,
    TABLE_PEDESTAL_COLUMN_HEIGHT,
    TABLE_PEDESTAL_COLUMN_SIZE_XY_PLACEHOLDER,
    TABLE_PEDESTAL_FOOT_CENTER_Z,
    TABLE_PEDESTAL_FOOT_SIZE_PLACEHOLDER,
    TABLE_PEDESTAL_TOP_SUPPORT_CENTER_Z,
    TABLE_PEDESTAL_TOP_SUPPORT_SIZE_PLACEHOLDER,
    TABLE_PEDESTAL_X_POSITIONS,
    TABLE_SEAM_COLOR,
    TABLE_SEAM_WIDTH,
    TABLE_SEAM_Y_PLACEHOLDER,
    TABLE_TOP_THICKNESS,
    TABLE_WIDTH,
    TARGET_BALL_CENTER,
    TARGET_BALL_COLOR,
    TARGET_BALL_MASS_KG,
    TARGET_BALL_RADIUS,
    TACTILE_BODY_PRIM_PATHS,
    TACTILE_CAD_SCALE_TO_METERS,
    TACTILE_COLLISION_MODE,
    TACTILE_CONTACT_RADIUS_PLACEHOLDER,
    TACTILE_DEFAULT_GRIPPER_OPEN_DEG,
    TACTILE_DOME_ANGULAR_SEGMENTS,
    TACTILE_DOME_FOOTPRINT_EXPONENT_PLACEHOLDER,
    TACTILE_DOME_PROFILE_EXPONENT_PLACEHOLDER,
    TACTILE_DOME_RADIAL_SEGMENTS,
    TACTILE_GEOMETRY_MODE,
    TACTILE_SENSOR_COLOR,
    TACTILE_SENSOR_CAD_MAIN_MESH_RELATIVE_PATH,
    TACTILE_SENSOR_CAD_PRIM_PATH,
    TACTILE_SENSOR_CAD_SECONDARY_MESH_RELATIVE_PATH,
    TACTILE_SENSOR_CAD_USD_PATH,
    TACTILE_SENSOR_PRIM_PATHS,
    TACTILE_SENSOR_SIZE_SENSOR_FRAME,
    TOPDOWN_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER,
    TOPDOWN_CAMERA_POSITION_PLACEHOLDER,
    WRIST_CAMERA_ROTATION_PLACEHOLDER_DEG,
    WRIST_CAMERA_TRANSLATION_PLACEHOLDER,
    MOVING_TACTILE_ROOT_PRIM_PATH,
    MOVING_TACTILE_TRANSLATION_PLACEHOLDER,
    MOVING_TACTILE_BASIS_ROWS,
)
from tactile_taxels import load_taxel_positions_mm


OUT = os.environ.get(
    "SO101_SCENE_OUT", str(PROJECT_ROOT / "lab_scene.usda")
)
if not os.path.isabs(OUT):
    OUT = str(PROJECT_ROOT / OUT)

if os.path.exists(OUT):
    os.remove(OUT)

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(GRAVITY_M_S2)


def set_appearance(gprim, color, opacity=1.0):
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([opacity])


def create_preview_material(path, color, roughness, opacity=1.0, ior=1.5):
    """Create a simple RTX-compatible UsdPreviewSurface material."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def create_physics_material(name):
    """Define a PhysX material from the measured PHYSICS_MATERIALS table."""
    static_friction, dynamic_friction, restitution = PHYSICS_MATERIALS[name]
    path = f"/World/PhysicsMaterials/{name}"
    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr().Set(float(static_friction))
    physics.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    physics.CreateRestitutionAttr().Set(float(restitution))
    return material


def bind_physics_material(prim_path, material):
    """Bind a physics material to a collider, leaving its visual look alone."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"cannot bind physics material, missing: {prim_path}")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def make_box(path, size, position, color):
    """Create a unit cube scaled to the requested full dimensions in meters."""
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube = UsdGeom.Cube.Define(stage, path + "/Geom")
    cube.CreateSizeAttr(1.0)
    set_appearance(cube, color)
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*size))
    return xform.GetPrim(), cube.GetPrim()


def add_static_box(path, size, position, color):
    body, geom = make_box(path, size, position, color)
    UsdPhysics.CollisionAPI.Apply(geom)
    UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr("convexHull")
    return body


def add_dynamic_sphere(
    path,
    radius,
    position,
    color,
    mass_kg,
    material=None,
    latitude_segments=48,
    longitude_segments=96,
):
    """Use a dense visual mesh and a cheap analytic sphere collider."""
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    points = [Gf.Vec3f(0.0, 0.0, radius)]
    normals = [Gf.Vec3f(0.0, 0.0, 1.0)]
    for latitude in range(1, latitude_segments):
        phi = math.pi * latitude / latitude_segments
        ring_radius = math.sin(phi)
        z = math.cos(phi)
        for longitude in range(longitude_segments):
            theta = 2.0 * math.pi * longitude / longitude_segments
            normal = Gf.Vec3f(
                ring_radius * math.cos(theta),
                ring_radius * math.sin(theta),
                z,
            )
            normals.append(normal)
            points.append(normal * radius)
    south_pole = len(points)
    points.append(Gf.Vec3f(0.0, 0.0, -radius))
    normals.append(Gf.Vec3f(0.0, 0.0, -1.0))

    face_counts = []
    face_indices = []
    first_ring = 1
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        face_counts.append(3)
        face_indices.extend(
            (0, first_ring + longitude, first_ring + next_longitude)
        )

    ring_count = latitude_segments - 1
    for ring in range(ring_count - 1):
        upper = first_ring + ring * longitude_segments
        lower = upper + longitude_segments
        for longitude in range(longitude_segments):
            next_longitude = (longitude + 1) % longitude_segments
            face_counts.append(4)
            face_indices.extend(
                (
                    upper + longitude,
                    lower + longitude,
                    lower + next_longitude,
                    upper + next_longitude,
                )
            )

    last_ring = first_ring + (ring_count - 1) * longitude_segments
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        face_counts.append(3)
        face_indices.extend(
            (south_pole, last_ring + next_longitude, last_ring + longitude)
        )

    visual = UsdGeom.Mesh.Define(stage, path + "/Geom")
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr(face_counts)
    visual.CreateFaceVertexIndicesAttr(face_indices)
    visual.CreateNormalsAttr(normals)
    visual.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    visual.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    visual.CreateDoubleSidedAttr(True)
    visual.CreateExtentAttr(
        [Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)]
    )
    set_appearance(visual, color)
    if material is not None:
        bind_material(visual.GetPrim(), material)

    collision = UsdGeom.Sphere.Define(stage, path + "/Collision")
    collision.CreateRadiusAttr(radius)
    collision.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())

    body = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass_kg)
    PhysxSchema.PhysxRigidBodyAPI.Apply(body)
    return body


def add_static_bowl(path, center_xy, segments=96, profile_segments=12):
    """Create a measured hollow bowl with smooth circular, curved walls."""
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(center_xy[0], center_xy[1], 0.0))

    outer_top_radius = BOWL_OUTER_TOP_DIAMETER / 2.0
    inner_top_radius = outer_top_radius - BOWL_WALL_THICKNESS
    inner_bottom_radius = BOWL_INNER_BOTTOM_DIAMETER / 2.0
    outer_bottom_radius = inner_bottom_radius + BOWL_WALL_THICKNESS

    # The shallow bottom plate closes the bowl while leaving the interior open.
    base = UsdGeom.Cylinder.Define(stage, path + "/Base")
    base.CreateAxisAttr(UsdGeom.Tokens.z)
    base.CreateRadiusAttr(outer_bottom_radius)
    base.CreateHeightAttr(BOWL_WALL_THICKNESS)
    set_appearance(base, BOWL_COLOR, BOWL_OPACITY)
    bind_material(base.GetPrim(), BOWL_MATERIAL)
    UsdGeom.Xformable(base).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, BOWL_WALL_THICKNESS / 2.0)
    )
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())

    # Revolve a rounded radial profile instead of joining only two rings.  The
    # sine profile gives the lower wall a bowl-like sweep and approaches the
    # top rim smoothly.  Ninety-six angular segments and explicit vertex
    # normals prevent the transparent wall from showing polygonal facets.
    outer_profile = []
    inner_profile = []
    for profile_index in range(profile_segments + 1):
        blend = profile_index / profile_segments
        radial_blend = math.sin(0.5 * math.pi * blend)
        outer_profile.append(
            (
                outer_bottom_radius
                + (outer_top_radius - outer_bottom_radius) * radial_blend,
                BOWL_HEIGHT * blend,
            )
        )
        inner_profile.append(
            (
                inner_bottom_radius
                + (inner_top_radius - inner_bottom_radius) * radial_blend,
                BOWL_WALL_THICKNESS
                + (BOWL_HEIGHT - BOWL_WALL_THICKNESS) * blend,
            )
        )

    points = []
    normals = []
    for profile, normal_sign in ((outer_profile, 1.0), (inner_profile, -1.0)):
        for profile_index, (radius, height) in enumerate(profile):
            before = profile[max(0, profile_index - 1)]
            after = profile[min(profile_segments, profile_index + 1)]
            radius_delta = after[0] - before[0]
            height_delta = after[1] - before[1]
            normal_length = math.hypot(height_delta, radius_delta)
            radial_normal = normal_sign * height_delta / normal_length
            vertical_normal = -normal_sign * radius_delta / normal_length
            for angular_index in range(segments):
                theta = 2.0 * math.pi * angular_index / segments
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                points.append(
                    Gf.Vec3f(
                        radius * cos_theta,
                        radius * sin_theta,
                        height,
                    )
                )
                normals.append(
                    Gf.Vec3f(
                        radial_normal * cos_theta,
                        radial_normal * sin_theta,
                        vertical_normal,
                    )
                )

    face_counts = []
    face_indices = []
    profile_ring_count = profile_segments + 1
    outer_bottom = 0
    outer_top = profile_segments * segments
    inner_bottom = profile_ring_count * segments
    inner_top = inner_bottom + profile_segments * segments

    for profile_index in range(profile_segments):
        outer_ring = profile_index * segments
        outer_next_ring = outer_ring + segments
        inner_ring = inner_bottom + profile_index * segments
        inner_next_ring = inner_ring + segments
        for index in range(segments):
            next_index = (index + 1) % segments
            for face in (
                (
                    outer_ring + index,
                    outer_ring + next_index,
                    outer_next_ring + next_index,
                    outer_next_ring + index,
                ),
                (
                    inner_ring + index,
                    inner_next_ring + index,
                    inner_next_ring + next_index,
                    inner_ring + next_index,
                ),
            ):
                face_counts.append(4)
                face_indices.extend(face)

    for index in range(segments):
        next_index = (index + 1) % segments
        for face in (
            # Rounded top rim and the lower seam into the bottom plate.
            (
                outer_top + index,
                outer_top + next_index,
                inner_top + next_index,
                inner_top + index,
            ),
            (
                outer_bottom + index,
                inner_bottom + index,
                inner_bottom + next_index,
                outer_bottom + next_index,
            ),
        ):
            face_counts.append(4)
            face_indices.extend(face)

    wall = UsdGeom.Mesh.Define(stage, path + "/Wall")
    wall.CreatePointsAttr(points)
    wall.CreateNormalsAttr(normals)
    wall.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    wall.CreateFaceVertexCountsAttr(face_counts)
    wall.CreateFaceVertexIndicesAttr(face_indices)
    wall.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    wall.CreateDoubleSidedAttr(True)
    set_appearance(wall, BOWL_COLOR, BOWL_OPACITY)
    bind_material(wall.GetPrim(), BOWL_MATERIAL)
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(wall.GetPrim()).CreateApproximationAttr(
        "none"
    )
    return root.GetPrim()


# Materials remain deliberately simple and texture-free for this baseline.

def add_superellipsoidal_tactile_housing(path, width, length, depth):
    """Create a closed double-curvature dome over a flat mounting plane."""
    footprint_exponent = TACTILE_DOME_FOOTPRINT_EXPONENT_PLACEHOLDER
    profile_exponent = TACTILE_DOME_PROFILE_EXPONENT_PLACEHOLDER
    radial_segments = TACTILE_DOME_RADIAL_SEGMENTS
    angular_segments = TACTILE_DOME_ANGULAR_SEGMENTS
    half_width = width / 2.0
    half_length = length / 2.0

    if footprint_exponent <= 1.0 or profile_exponent <= 1.0:
        raise ValueError("Tactile dome exponents must be greater than one")
    if radial_segments < 2 or angular_segments < 8:
        raise ValueError("Tactile dome mesh resolution is too low")

    def signed_power(value, exponent):
        if value == 0.0:
            return 0.0
        return math.copysign(abs(value) ** exponent, value)

    # The root is the flat mounting-plane center. The convex active dome
    # protrudes along local -Z into the grasp gap, reaching -depth at its apex.
    points = [Gf.Vec3f(0.0, 0.0, -depth)]
    normals = [Gf.Vec3f(0.0, 0.0, -1.0)]

    for radial_index in range(1, radial_segments + 1):
        radial = radial_index / radial_segments
        vertical = max(
            0.0,
            1.0 - radial ** profile_exponent,
        ) ** (1.0 / profile_exponent)
        z_value = depth * (1.0 - vertical)

        for angular_index in range(angular_segments):
            angle = 2.0 * math.pi * angular_index / angular_segments
            cosine = math.cos(angle)
            sine = math.sin(angle)
            boundary_x = half_width * signed_power(
                cosine, 2.0 / footprint_exponent
            )
            boundary_y = half_length * signed_power(
                sine, 2.0 / footprint_exponent
            )
            x_value = radial * boundary_x
            y_value = radial * boundary_y
            points.append(Gf.Vec3f(x_value, y_value, z_value - depth))

            x_term = abs(x_value / half_width) ** footprint_exponent
            y_term = abs(y_value / half_length) ** footprint_exponent
            footprint_sum = x_term + y_term
            if footprint_sum <= 1.0e-14:
                normals.append(Gf.Vec3f(0.0, 0.0, -1.0))
                continue

            footprint_radius = footprint_sum ** (1.0 / footprint_exponent)
            radius_scale = (
                profile_exponent
                * footprint_radius ** (profile_exponent - 1.0)
                * footprint_sum ** (1.0 / footprint_exponent - 1.0)
            )
            gradient_x = (
                radius_scale
                * math.copysign(
                    abs(x_value) ** (footprint_exponent - 1.0),
                    x_value,
                )
                / half_width ** footprint_exponent
            )
            gradient_y = (
                radius_scale
                * math.copysign(
                    abs(y_value) ** (footprint_exponent - 1.0),
                    y_value,
                )
                / half_length ** footprint_exponent
            )
            gradient_z = (
                -profile_exponent
                * vertical ** (profile_exponent - 1.0)
                / depth
            )
            normals.append(
                Gf.Vec3f(gradient_x, gradient_y, gradient_z).GetNormalized()
            )

    face_counts = []
    face_indices = []
    first_ring = 1
    for angular_index in range(angular_segments):
        next_angular = (angular_index + 1) % angular_segments
        face_counts.append(3)
        face_indices.extend(
            (
                0,
                first_ring + next_angular,
                first_ring + angular_index,
            )
        )

    for radial_index in range(radial_segments - 1):
        inner = 1 + radial_index * angular_segments
        outer = inner + angular_segments
        for angular_index in range(angular_segments):
            next_angular = (angular_index + 1) % angular_segments
            face_counts.append(4)
            face_indices.extend(
                (
                    inner + angular_index,
                    inner + next_angular,
                    outer + next_angular,
                    outer + angular_index,
                )
            )

    outer_ring = 1 + (radial_segments - 1) * angular_segments
    mount_center = len(points)
    points.append(Gf.Vec3f(0.0, 0.0, 0.0))
    normals.append(Gf.Vec3f(0.0, 0.0, 1.0))
    mount_ring = len(points)
    for angular_index in range(angular_segments):
        points.append(points[outer_ring + angular_index])
        normals.append(Gf.Vec3f(0.0, 0.0, 1.0))
    for angular_index in range(angular_segments):
        next_angular = (angular_index + 1) % angular_segments
        face_counts.append(3)
        face_indices.extend(
            (
                mount_center,
                mount_ring + angular_index,
                mount_ring + next_angular,
            )
        )

    housing = UsdGeom.Mesh.Define(stage, path)
    housing.CreatePointsAttr(points)
    housing.CreateFaceVertexCountsAttr(face_counts)
    housing.CreateFaceVertexIndicesAttr(face_indices)
    housing.CreateNormalsAttr(normals)
    housing.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    housing.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    set_appearance(housing, TACTILE_SENSOR_COLOR)
    bind_material(housing.GetPrim(), TACTILE_MATERIAL)
    UsdPhysics.CollisionAPI.Apply(housing.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(
        housing.GetPrim()
    ).CreateApproximationAttr("convexHull")
    return housing.GetPrim()


def add_tactile_sensor_mvp(
    root_path,
    sensor_path,
    body_path,
    translation,
    basis_rows,
):
    """Author a double-curvature dome and Isaac contact sensor prim.

    The sensor origin is the flat mounting-plane center. Sensor-local -Z points
    from the finger through the dome apex and into the grasp gap.
    """
    root = UsdGeom.Xform.Define(stage, root_path)
    local_transform = Gf.Matrix4d(
        basis_rows[0][0], basis_rows[0][1], basis_rows[0][2], 0.0,
        basis_rows[1][0], basis_rows[1][1], basis_rows[1][2], 0.0,
        basis_rows[2][0], basis_rows[2][1], basis_rows[2][2], 0.0,
        translation[0], translation[1], translation[2], 1.0,
    )
    root.AddTransformOp().Set(local_transform)

    width, length, depth = TACTILE_SENSOR_SIZE_SENSOR_FRAME
    add_superellipsoidal_tactile_housing(
        root_path + "/Housing", width, length, depth
    )

    return configure_contact_sensor(sensor_path, body_path, translation)


def configure_contact_sensor(sensor_path, body_path, translation):
    """Create the Isaac backend sensor on a rigid body containing the pad."""
    sensor = IsaacSensorSchema.IsaacContactSensor.Define(stage, sensor_path)

    sensor.CreateThresholdAttr().Set((0.0, 1.0e6))
    # IsaacContactSensor has no local-pose attribute in Isaac Sim 6.0. Since
    # it must remain a direct child of the monitored rigid body, its backend
    # sphere is centered at the body origin. Expand that backend sphere far
    # enough to reach the distal mount; tactile_sensor.py still reports forces
    # in the separate mount frame.
    backend_radius = (
        math.sqrt(sum(float(value) ** 2 for value in translation))
        + TACTILE_CONTACT_RADIUS_PLACEHOLDER
    )
    sensor.CreateRadiusAttr().Set(backend_radius)
    sensor.CreateColorAttr().Set(Gf.Vec4f(0.1, 0.8, 1.0, 1.0))

    body = stage.GetPrimAtPath(body_path)
    contact_report = PhysxSchema.PhysxContactReportAPI.Apply(body)
    contact_report.CreateThresholdAttr().Set(0.0)
    return sensor.GetPrim()


def component_matrix(basis_rows, translation, translation_scale=1.0):
    """Build a row-vector Gf transform from a CAD component occurrence."""
    return Gf.Matrix4d(
        basis_rows[0][0], basis_rows[0][1], basis_rows[0][2], 0.0,
        basis_rows[1][0], basis_rows[1][1], basis_rows[1][2], 0.0,
        basis_rows[2][0], basis_rows[2][1], basis_rows[2][2], 0.0,
        translation[0] * translation_scale,
        translation[1] * translation_scale,
        translation[2] * translation_scale,
        1.0,
    )


def require_prim(path, description):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing {description} prim: {path}")
    return prim


def copy_cad_collision_mesh(
    source_prim,
    target_path,
    target_parent_path,
    approximation,
):
    """Author a local collision mesh with the exact composed CAD topology.

    PhysX scene queries accept collision APIs applied as reference overrides,
    but Isaac Sim 6.0 does not reliably emit contact reports for those shapes
    on articulation links. Keeping the visual as a reference and authoring the
    same points/faces locally gives stable contact reporting without changing
    the physical surface.
    """
    source = UsdGeom.Mesh(source_prim)
    target = UsdGeom.Mesh.Define(stage, target_path)
    source_to_world = UsdGeom.Xformable(source_prim).ComputeLocalToWorldTransform(0)
    parent_to_world = UsdGeom.Xformable(
        require_prim(target_parent_path, "CAD collision parent")
    ).ComputeLocalToWorldTransform(0)
    source_to_parent = source_to_world * parent_to_world.GetInverse()
    points = [
        Gf.Vec3f(source_to_parent.Transform(Gf.Vec3d(*point)))
        for point in source.GetPointsAttr().Get()
    ]
    target.CreatePointsAttr(points)
    target.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    target.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    target.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    target.CreateExtentAttr(
        [
            Gf.Vec3f(*(min(point[axis] for point in points) for axis in range(3))),
            Gf.Vec3f(*(max(point[axis] for point in points) for axis in range(3))),
        ]
    )
    target.MakeInvisible()

    UsdPhysics.CollisionAPI.Apply(target.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(
        target.GetPrim()
    ).CreateApproximationAttr(approximation)
    return target.GetPrim()


def add_taxel_aligned_sensor_collision(root_path, body_path, sensor_local):
    """Create a clean convex pad whose active face contains all 52 taxels.

    The converted main CAD mesh is an open tessellated shell whose automatic
    convex hull has thousands of vertices. That shape is valid for scene
    queries but does not produce simulation contacts reliably in PhysX. The
    measured taxels define the physical front surface; eight boundary samples
    projected to the CAD back plane close it into a compact convex solid.
    """
    taxels_mm = list(load_taxel_positions_mm())
    back_plane_z_mm = -4.4109976
    back_boundary_taxel_indices = (0, 4, 12, 21, 29, 40, 46, 51)
    vertices_mm = taxels_mm + [
        (
            taxels_mm[index][0],
            taxels_mm[index][1],
            back_plane_z_mm,
        )
        for index in back_boundary_taxel_indices
    ]
    hull = ConvexHull(vertices_mm)
    points = []
    for vertex in vertices_mm:
        point_sensor = Gf.Vec3d(
            *(coordinate / 1000.0 for coordinate in vertex)
        )
        points.append(Gf.Vec3f(sensor_local.Transform(point_sensor)))
    oriented_triangles = []
    for triangle, equation in zip(hull.simplices, hull.equations):
        indices = [int(index) for index in triangle]
        point0, point1, point2 = (
            Gf.Vec3d(*vertices_mm[index]) for index in indices
        )
        triangle_normal = Gf.Cross(point1 - point0, point2 - point0)
        outward_normal = Gf.Vec3d(*equation[:3])
        if Gf.Dot(triangle_normal, outward_normal) < 0.0:
            indices[1], indices[2] = indices[2], indices[1]
        oriented_triangles.append(indices)
    faces = [index for triangle in oriented_triangles for index in triangle]

    collision_name = root_path.rsplit("/", 1)[-1] + "TaxelAlignedCollision"
    collision = UsdGeom.Mesh.Define(
        stage, body_path + "/" + collision_name
    )
    collision.CreatePointsAttr(points)
    collision.CreateFaceVertexCountsAttr([3] * len(oriented_triangles))
    collision.CreateFaceVertexIndicesAttr(faces)
    collision.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    collision.CreateExtentAttr(
        [
            Gf.Vec3f(*(min(point[axis] for point in points) for axis in range(3))),
            Gf.Vec3f(*(max(point[axis] for point in points) for axis in range(3))),
        ]
    )
    collision.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(
        collision.GetPrim()
    ).CreateApproximationAttr("convexHull")
    return collision.GetPrim(), len(hull.vertices)


def add_tactile_sensor_cad(
    root_path,
    sensor_path,
    body_path,
    stock_visual_path,
    stock_collision_path,
    adapter_asset_path,
    adapter_prim_path,
    adapter_mesh_relative_path,
    adapter_basis_rows,
    adapter_translation_mm,
    sensor_basis_rows,
    sensor_translation_mm,
):
    """Replace one stock finger with its adapter and mounted sensor CAD.

    The adapter component is registered to the stock part transform. The
    sensor transform is then propagated from its occurrence in the same STEP
    assembly, preserving both the 12-degree mounting tilt and CAD origin used
    by the measured 52-taxel coordinates.
    """
    for asset_path in (adapter_asset_path, TACTILE_SENSOR_CAD_USD_PATH):
        if not asset_path.is_file():
            raise FileNotFoundError(
                f"Missing converted CAD asset {asset_path}. Run "
                "scripts/assets/convert_tactile_cad.py first."
            )

    stock_visual = require_prim(stock_visual_path, "stock visual")
    stock_collision = require_prim(stock_collision_path, "stock collision")
    stock_local = UsdGeom.Xformable(stock_visual).GetLocalTransformation()
    UsdGeom.Imageable(stock_visual).MakeInvisible()

    adapter_occurrence_m = component_matrix(
        adapter_basis_rows,
        adapter_translation_mm,
        TACTILE_CAD_SCALE_TO_METERS,
    )
    # G * placement == the original stock-part transform. Component points are
    # row vectors in Gf/USD, so local transforms compose left-to-right here.
    adapter_placement = adapter_occurrence_m.GetInverse() * stock_local

    adapter_root_path = body_path + "/TactileCadAdapter"
    adapter_root = UsdGeom.Xform.Define(stage, adapter_root_path)
    adapter_root.AddTransformOp().Set(adapter_placement)
    adapter_scale = UsdGeom.Xform.Define(
        stage, adapter_root_path + "/MillimetersToMeters"
    )
    adapter_scale.AddScaleOp().Set(
        Gf.Vec3f(
            TACTILE_CAD_SCALE_TO_METERS,
            TACTILE_CAD_SCALE_TO_METERS,
            TACTILE_CAD_SCALE_TO_METERS,
        )
    )
    adapter_geometry_path = (
        adapter_root_path + "/MillimetersToMeters/Geometry"
    )
    adapter_geometry = UsdGeom.Xform.Define(stage, adapter_geometry_path)
    adapter_geometry.GetPrim().GetReferences().AddReference(
        str(adapter_asset_path), Sdf.Path(adapter_prim_path)
    )
    adapter_mesh = require_prim(
        adapter_geometry_path + "/" + adapter_mesh_relative_path,
        "referenced adapter mesh",
    )
    bind_material(adapter_mesh, TACTILE_ADAPTER_MATERIAL)

    if TACTILE_COLLISION_MODE == "high_fidelity":
        # The old collision is an instance root, so deactivating the instance
        # cleanly removes its whole prototype without editing the robot asset.
        stock_collision.SetActive(False)
        copy_cad_collision_mesh(
            adapter_mesh,
            adapter_root_path + "/AdapterCollision",
            adapter_root_path,
            "convexDecomposition",
        )

    sensor_occurrence_m = component_matrix(
        sensor_basis_rows,
        sensor_translation_mm,
        TACTILE_CAD_SCALE_TO_METERS,
    )
    sensor_local = sensor_occurrence_m * adapter_placement
    root = UsdGeom.Xform.Define(stage, root_path)
    root.AddTransformOp().Set(sensor_local)

    sensor_scale = UsdGeom.Xform.Define(
        stage, root_path + "/MillimetersToMeters"
    )
    sensor_scale.AddScaleOp().Set(
        Gf.Vec3f(
            TACTILE_CAD_SCALE_TO_METERS,
            TACTILE_CAD_SCALE_TO_METERS,
            TACTILE_CAD_SCALE_TO_METERS,
        )
    )
    sensor_geometry_path = root_path + "/MillimetersToMeters/Geometry"
    sensor_geometry = UsdGeom.Xform.Define(stage, sensor_geometry_path)
    sensor_geometry.GetPrim().GetReferences().AddReference(
        str(TACTILE_SENSOR_CAD_USD_PATH),
        Sdf.Path(TACTILE_SENSOR_CAD_PRIM_PATH),
    )
    sensor_meshes = (
        require_prim(
            sensor_geometry_path
            + "/"
            + TACTILE_SENSOR_CAD_MAIN_MESH_RELATIVE_PATH,
            "referenced main sensor mesh",
        ),
        require_prim(
            sensor_geometry_path
            + "/"
            + TACTILE_SENSOR_CAD_SECONDARY_MESH_RELATIVE_PATH,
            "referenced secondary sensor mesh",
        ),
    )
    for sensor_mesh in sensor_meshes:
        bind_material(sensor_mesh, TACTILE_MATERIAL)
    add_taxel_aligned_sensor_collision(root_path, body_path, sensor_local)

    sensor_translation = sensor_local.ExtractTranslation()
    configure_contact_sensor(sensor_path, body_path, sensor_translation)
    return sensor_local


UsdGeom.Scope.Define(stage, "/World/Looks")
TABLE_MATERIAL = create_preview_material(
    "/World/Looks/TableMatte", TABLE_COLOR, roughness=0.88
)
TABLE_FRAME_MATERIAL = create_preview_material(
    "/World/Looks/TableFrameMetal", TABLE_FRAME_COLOR, roughness=0.62
)
BALL_MATERIAL = create_preview_material(
    "/World/Looks/RedPlush", TARGET_BALL_COLOR, roughness=0.96
)
BOWL_MATERIAL = create_preview_material(
    "/World/Looks/ClearPlastic",
    BOWL_COLOR,
    roughness=0.10,
    opacity=BOWL_OPACITY,
    ior=1.49,
)
TACTILE_MATERIAL = create_preview_material(
    "/World/Looks/TactileRubber",
    TACTILE_SENSOR_COLOR,
    roughness=0.82,
)
TACTILE_ADAPTER_MATERIAL = create_preview_material(
    "/World/Looks/TactileAdapter",
    (0.20, 0.22, 0.24),
    roughness=0.46,
)

# The table top is a measured 10 mm slab. The ground is shifted below the robot
# base so Z=0 remains the tabletop/base frame used by the real robot.
add_static_box(
    "/World/Ground",
    (4.0, 4.0, GROUND_THICKNESS),
    (0.0, 0.0, FLOOR_Z - GROUND_THICKNESS / 2.0),
    GROUND_COLOR,
)
add_static_box(
    "/World/Table",
    (TABLE_LENGTH, TABLE_WIDTH, TABLE_TOP_THICKNESS),
    TABLE_CENTER,
    TABLE_COLOR,
)
bind_material(stage.GetPrimAtPath("/World/Table/Geom"), TABLE_MATERIAL)

# The real table is height-adjustable. Its frame dimensions were not measured,
# so represent it with two dark T-shaped static pedestals while preserving the
# measured tabletop and floor heights.
UsdGeom.Scope.Define(stage, "/World/TableFrame")
for pedestal_index, pedestal_x in enumerate(TABLE_PEDESTAL_X_POSITIONS):
    pedestal_path = f"/World/TableFrame/Pedestal_{pedestal_index}"
    pedestal_parts = (
        (
            "Foot",
            TABLE_PEDESTAL_FOOT_SIZE_PLACEHOLDER,
            (pedestal_x, TABLE_CENTER[1], TABLE_PEDESTAL_FOOT_CENTER_Z),
        ),
        (
            "Column",
            (
                TABLE_PEDESTAL_COLUMN_SIZE_XY_PLACEHOLDER[0],
                TABLE_PEDESTAL_COLUMN_SIZE_XY_PLACEHOLDER[1],
                TABLE_PEDESTAL_COLUMN_HEIGHT,
            ),
            (pedestal_x, TABLE_CENTER[1], TABLE_PEDESTAL_COLUMN_CENTER_Z),
        ),
        (
            "TopSupport",
            TABLE_PEDESTAL_TOP_SUPPORT_SIZE_PLACEHOLDER,
            (
                pedestal_x,
                TABLE_CENTER[1],
                TABLE_PEDESTAL_TOP_SUPPORT_CENTER_Z,
            ),
        ),
    )
    for part_name, part_size, part_position in pedestal_parts:
        part_path = f"{pedestal_path}/{part_name}"
        add_static_box(
            part_path,
            part_size,
            part_position,
            TABLE_FRAME_COLOR,
        )
        bind_material(
            stage.GetPrimAtPath(part_path + "/Geom"), TABLE_FRAME_MATERIAL
        )

# A subtle visual-only seam is visible in the real dark composite tabletop.
make_box(
    "/World/TableSeam",
    (TABLE_LENGTH, TABLE_SEAM_WIDTH, 0.0002),
    (TABLE_CENTER[0], TABLE_SEAM_Y_PLACEHOLDER, 0.0001),
    TABLE_SEAM_COLOR,
)

add_dynamic_sphere(
    "/World/TargetBall",
    TARGET_BALL_RADIUS,
    TARGET_BALL_CENTER,
    TARGET_BALL_COLOR,
    TARGET_BALL_MASS_KG,
    BALL_MATERIAL,
)
add_static_bowl("/World/Bowl", BOWL_CENTER_XY_PLACEHOLDER)

UsdGeom.Scope.Define(stage, "/World/Lights")
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
dome.CreateIntensityAttr(1500.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
sun.CreateIntensityAttr(2500.0)
sun.CreateAngleAttr(0.53)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))

UsdGeom.Scope.Define(stage, "/World/Cameras")

topdown_camera = UsdGeom.Camera.Define(stage, "/World/Cameras/TopDown")
topdown_camera.CreateFocalLengthAttr(
    TOPDOWN_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER
)
topdown_camera.CreateHorizontalApertureAttr(
    CAMERA_HORIZONTAL_APERTURE_MM_PLACEHOLDER
)
topdown_camera.CreateVerticalApertureAttr(
    CAMERA_VERTICAL_APERTURE_MM_PLACEHOLDER
)
topdown_camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
topdown_xform = UsdGeom.Xformable(topdown_camera)
topdown_xform.AddTranslateOp().Set(
    Gf.Vec3d(*TOPDOWN_CAMERA_POSITION_PLACEHOLDER)
)
topdown_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

# A separate debug camera keeps scene inspection possible while the real wrist
# camera hand-eye transform is still unknown.
debug_view = Gf.Matrix4d()
debug_view.SetLookAt(
    Gf.Vec3d(*DEBUG_CAMERA_EYE_PLACEHOLDER),
    Gf.Vec3d(*DEBUG_CAMERA_TARGET_PLACEHOLDER),
    Gf.Vec3d(0.0, 0.0, 1.0),
)
debug_camera = UsdGeom.Camera.Define(stage, "/World/Cameras/Debug")
debug_camera.CreateFocalLengthAttr(DEBUG_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER)
debug_camera.CreateHorizontalApertureAttr(
    CAMERA_HORIZONTAL_APERTURE_MM_PLACEHOLDER
)
debug_camera.CreateVerticalApertureAttr(
    CAMERA_VERTICAL_APERTURE_MM_PLACEHOLDER
)
debug_camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
UsdGeom.Xformable(debug_camera).AddTransformOp().Set(debug_view.GetInverse())

# Angled bird's-eye view for inspecting the complete reconstructed workspace.
birdseye_view = Gf.Matrix4d()
birdseye_view.SetLookAt(
    Gf.Vec3d(*BIRDSEYE_CAMERA_EYE),
    Gf.Vec3d(*BIRDSEYE_CAMERA_TARGET),
    Gf.Vec3d(0.0, 0.0, 1.0),
)
birdseye_camera = UsdGeom.Camera.Define(stage, "/World/Cameras/BirdEye")
birdseye_camera.CreateFocalLengthAttr(BIRDSEYE_CAMERA_FOCAL_LENGTH_MM)
birdseye_camera.CreateHorizontalApertureAttr(
    CAMERA_HORIZONTAL_APERTURE_MM_PLACEHOLDER
)
birdseye_camera.CreateVerticalApertureAttr(
    CAMERA_VERTICAL_APERTURE_MM_PLACEHOLDER
)
birdseye_camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
UsdGeom.Xformable(birdseye_camera).AddTransformOp().Set(
    birdseye_view.GetInverse()
)

robot_usd = str(
    PROJECT_ROOT / "assets" / "so101_new_calib" / "so101_new_calib.usda"
)
robot = UsdGeom.Xform.Define(stage, "/World/Robot")
robot.GetPrim().GetReferences().AddReference(robot_usd)
robot_transform = Gf.Matrix4d(1.0)
robot_transform.SetRotate(
    Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), ROBOT_BASE_YAW_DEG)
)
robot_transform.SetTranslateOnly(Gf.Vec3d(*ROBOT_BASE_POSITION))
robot.AddTransformOp().Set(robot_transform)

# The two 10 mm domes occupy more of the grasp gap than the stock pads. Author
# an open inspection pose so the tactile shells do not start interpenetrating.
gripper_drive = UsdPhysics.DriveAPI.Get(
    stage.GetPrimAtPath(GRIPPER_DRIVE_PRIM_PATH), "angular"
)
if not gripper_drive:
    raise RuntimeError(
        f"Missing angular gripper drive at {GRIPPER_DRIVE_PRIM_PATH}"
    )
gripper_drive.CreateTargetPositionAttr().Set(
    TACTILE_DEFAULT_GRIPPER_OPEN_DEG
)

if TACTILE_GEOMETRY_MODE == "cad":
    fixed_sensor_local = add_tactile_sensor_cad(
        FIXED_TACTILE_ROOT_PRIM_PATH,
        TACTILE_SENSOR_PRIM_PATHS[0],
        TACTILE_BODY_PRIM_PATHS[0],
        FIXED_STOCK_VISUAL_PRIM_PATH,
        FIXED_STOCK_COLLISION_PRIM_PATH,
        FIXED_ADAPTER_CAD_USD_PATH,
        FIXED_ADAPTER_CAD_PRIM_PATH,
        FIXED_ADAPTER_CAD_MESH_RELATIVE_PATH,
        FIXED_ADAPTER_CAD_BASIS_ROWS,
        FIXED_ADAPTER_CAD_TRANSLATION_MM,
        FIXED_SENSOR_CAD_BASIS_ROWS,
        FIXED_SENSOR_CAD_TRANSLATION_MM,
    )
    moving_sensor_local = add_tactile_sensor_cad(
        MOVING_TACTILE_ROOT_PRIM_PATH,
        TACTILE_SENSOR_PRIM_PATHS[1],
        TACTILE_BODY_PRIM_PATHS[1],
        MOVING_STOCK_VISUAL_PRIM_PATH,
        MOVING_STOCK_COLLISION_PRIM_PATH,
        MOVING_ADAPTER_CAD_USD_PATH,
        MOVING_ADAPTER_CAD_PRIM_PATH,
        MOVING_ADAPTER_CAD_MESH_RELATIVE_PATH,
        MOVING_ADAPTER_CAD_BASIS_ROWS,
        MOVING_ADAPTER_CAD_TRANSLATION_MM,
        MOVING_SENSOR_CAD_BASIS_ROWS,
        MOVING_SENSOR_CAD_TRANSLATION_MM,
    )
    print(
        "tactile CAD sensor translations (link frames):",
        tuple(fixed_sensor_local.ExtractTranslation()),
        tuple(moving_sensor_local.ExtractTranslation()),
    )
else:
    # Retained only for visual/collision regression comparisons.
    add_tactile_sensor_mvp(
        FIXED_TACTILE_ROOT_PRIM_PATH,
        TACTILE_SENSOR_PRIM_PATHS[0],
        TACTILE_BODY_PRIM_PATHS[0],
        FIXED_TACTILE_TRANSLATION_PLACEHOLDER,
        FIXED_TACTILE_BASIS_ROWS,
    )
    add_tactile_sensor_mvp(
        MOVING_TACTILE_ROOT_PRIM_PATH,
        TACTILE_SENSOR_PRIM_PATHS[1],
        TACTILE_BODY_PRIM_PATHS[1],
        MOVING_TACTILE_TRANSLATION_PLACEHOLDER,
        MOVING_TACTILE_BASIS_ROWS,
    )


# This camera follows the gripper because it is authored below the gripper link.
# Its local transform remains an explicit identity placeholder until calibrated.
wrist_camera = UsdGeom.Camera.Define(stage, GRIPPER_LINK_PRIM_PATH + "/WristCamera")
wrist_camera.CreateFocalLengthAttr(TOPDOWN_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER)
wrist_camera.CreateHorizontalApertureAttr(
    CAMERA_HORIZONTAL_APERTURE_MM_PLACEHOLDER
)
wrist_camera.CreateVerticalApertureAttr(
    CAMERA_VERTICAL_APERTURE_MM_PLACEHOLDER
)
wrist_camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
wrist_xform = UsdGeom.Xformable(wrist_camera)
wrist_xform.AddTranslateOp().Set(Gf.Vec3d(*WRIST_CAMERA_TRANSLATION_PLACEHOLDER))
wrist_xform.AddRotateXYZOp().Set(
    Gf.Vec3f(*WRIST_CAMERA_ROTATION_PLACEHOLDER_DEG)
)

# Physics materials.  Without these the whole scene runs on PhysX defaults,
# which is why the seated grasp could not retain the ball.
UsdGeom.Scope.Define(stage, "/World/PhysicsMaterials")
_ball_material = create_physics_material("plush_ball")
_pad_material = create_physics_material("gripper_pad")
_table_material = create_physics_material("table")
_bowl_material = create_physics_material("plastic")
_floor_material = create_physics_material("floor")
bind_physics_material("/World/TargetBall", _ball_material)
for _contact_path in GRIPPER_CONTACT_PRIM_PATHS:
    bind_physics_material(_contact_path, _pad_material)
bind_physics_material("/World/Table", _table_material)
bind_physics_material("/World/Bowl", _bowl_material)
bind_physics_material("/World/Ground", _floor_material)

# Record the solver settings the scripts actually run at.  The scene carried
# only gravity, so opening it anywhere else silently used PhysX defaults --
# notably 60 Hz instead of the 240 Hz the tactile pipeline is sampled against.
# Contact offsets.  Author rest before contact: PhysxCollisionAPI creates each
# attribute at its schema default first, and PhysX validates the pair on every
# write, so the other order trips a spurious "rest >= contact" error.
def author_contact_offsets(prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"cannot author offsets, missing: {prim_path}")
    count = 0
    for descendant in Usd.PrimRange(prim):
        if not descendant.HasAPI(UsdPhysics.CollisionAPI):
            continue
        api = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
        api.CreateRestOffsetAttr().Set(float(REST_OFFSET_M))
        api.CreateContactOffsetAttr().Set(float(CONTACT_OFFSET_M))
        count += 1
    return count


if CONTACT_OFFSET_M is None:
    print("contact offsets: left to PhysX defaults (see scene_config)")
else:
    _offset_targets = ["/World/TargetBall", *GRIPPER_CONTACT_PRIM_PATHS]
    _offset_count = sum(author_contact_offsets(p) for p in _offset_targets)
    print(f"contact offsets authored on {_offset_count} colliders")

_physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
_physx_scene.CreateTimeStepsPerSecondAttr(
    int(round(TACTILE_MVP_PHYSICS_FREQUENCY_HZ)))
_physx_scene.CreateSolverTypeAttr("TGS")

stage.GetRootLayer().Save()
print("saved:", OUT)
print("layout:", SCENE_LAYOUT_MODE)
print("frame: robot base at", ROBOT_BASE_POSITION, "table center:", TABLE_CENTER)
print("target ball:", TARGET_BALL_CENTER, "bowl:", BOWL_CENTER_XY_PLACEHOLDER)

app.close()
