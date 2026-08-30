"""Measured SO-101 lab-scene configuration.

Measured values are loaded from ``so101_scene_spec.yaml``. Values that have not
yet been measured are deliberately named ``*_PLACEHOLDER`` so they cannot be
mistaken for calibrated sim-to-real parameters.

The simulation frame is the SO-101 base frame: X follows the table long edge,
Y points from the mounting edge into the table, and Z points upward. The robot
base and tabletop are both at Z=0; consequently the lab floor is at Z=-0.735 m.
"""

import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_SPEC_PATH = PROJECT_ROOT / "so101_scene_spec.yaml"

with SCENE_SPEC_PATH.open("r", encoding="utf-8") as spec_file:
    SCENE_SPEC = yaml.safe_load(spec_file)


PHYSICS_FREQUENCY_HZ = 60.0
RENDER_FREQUENCY_HZ = 30.0
GRAVITY_M_S2 = 9.81

GROUND_THICKNESS = 0.02
GROUND_COLOR = (0.22, 0.23, 0.25)

_table = SCENE_SPEC["table"]
_table_dimensions = _table["dimensions"]
TABLE_LENGTH = float(_table_dimensions["length"])
TABLE_WIDTH = float(_table_dimensions["width"])
TABLE_TOP_THICKNESS = float(_table_dimensions["tabletop_thickness"])
TABLE_HEIGHT_FROM_FLOOR = float(_table["tabletop_height_from_floor"])
TABLE_TOP_Z = 0.0
# Choose +X from the robot toward the larger side of the tabletop, matching the
# left-to-right relationship in real_scene.jpg. The YAML explicitly leaves the
# X sign free as long as it is used consistently.
TABLE_CENTER_X = (
    float(
        SCENE_SPEC["robot"]["derived"]["robot_base_center_from_table_left_edge"]
    )
    - TABLE_LENGTH / 2.0
)
TABLE_CENTER_Y = TABLE_WIDTH / 2.0
TABLE_CENTER_Z = TABLE_TOP_Z - TABLE_TOP_THICKNESS / 2.0
TABLE_CENTER = (TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_CENTER_Z)
FLOOR_Z = TABLE_TOP_Z - TABLE_HEIGHT_FROM_FLOOR
TABLE_COLOR = (0.007, 0.008, 0.009)
TABLE_SEAM_COLOR = (0.012, 0.013, 0.014)
TABLE_SEAM_WIDTH = 0.0007
TABLE_SEAM_Y_PLACEHOLDER = TABLE_WIDTH / 2.0
TABLE_FRAME_COLOR = (0.025, 0.028, 0.032)
# The standing-desk frame was identified from the real setup but not measured.
# Model it as two T-shaped pedestals while retaining the measured tabletop pose.
TABLE_PEDESTAL_END_INSET_PLACEHOLDER = 0.15
TABLE_PEDESTAL_X_POSITIONS = (
    TABLE_CENTER_X - TABLE_LENGTH / 2.0 + TABLE_PEDESTAL_END_INSET_PLACEHOLDER,
    TABLE_CENTER_X + TABLE_LENGTH / 2.0 - TABLE_PEDESTAL_END_INSET_PLACEHOLDER,
)
TABLE_PEDESTAL_COLUMN_SIZE_XY_PLACEHOLDER = (0.060, 0.090)
TABLE_PEDESTAL_FOOT_SIZE_PLACEHOLDER = (0.075, TABLE_WIDTH - 0.10, 0.035)
TABLE_PEDESTAL_TOP_SUPPORT_SIZE_PLACEHOLDER = (
    0.075,
    TABLE_WIDTH - 0.08,
    0.030,
)
TABLE_PEDESTAL_FOOT_CENTER_Z = (
    FLOOR_Z + TABLE_PEDESTAL_FOOT_SIZE_PLACEHOLDER[2] / 2.0
)
TABLE_PEDESTAL_TOP_SUPPORT_CENTER_Z = (
    TABLE_TOP_Z
    - TABLE_TOP_THICKNESS
    - TABLE_PEDESTAL_TOP_SUPPORT_SIZE_PLACEHOLDER[2] / 2.0
)
_table_pedestal_column_bottom_z = (
    FLOOR_Z + TABLE_PEDESTAL_FOOT_SIZE_PLACEHOLDER[2]
)
_table_pedestal_column_top_z = (
    TABLE_TOP_Z
    - TABLE_TOP_THICKNESS
    - TABLE_PEDESTAL_TOP_SUPPORT_SIZE_PLACEHOLDER[2]
)
TABLE_PEDESTAL_COLUMN_HEIGHT = (
    _table_pedestal_column_top_z - _table_pedestal_column_bottom_z
)
TABLE_PEDESTAL_COLUMN_CENTER_Z = (
    _table_pedestal_column_bottom_z + _table_pedestal_column_top_z
) / 2.0

# Compatibility name: tabletop height relative to the lab floor.
TABLE_HEIGHT = TABLE_HEIGHT_FROM_FLOOR

# The YAML yaw is semantic. The imported asset needs +90 degrees to face +Y.
ROBOT_BASE_POSITION = (0.0, 0.0, TABLE_TOP_Z)
ROBOT_MOUNT_YAW_DEG = float(SCENE_SPEC["robot"]["mounting"]["yaw_deg"])
ROBOT_ASSET_YAW_OFFSET_DEG = 90.0
ROBOT_BASE_YAW_DEG = ROBOT_MOUNT_YAW_DEG + ROBOT_ASSET_YAW_OFFSET_DEG

_bowl = SCENE_SPEC["receptacle"]
BOWL_OUTER_TOP_DIAMETER = float(_bowl["geometry"]["outer_top_diameter"])
BOWL_INNER_BOTTOM_DIAMETER = float(_bowl["geometry"]["inner_bottom_diameter"])
BOWL_HEIGHT = float(_bowl["geometry"]["height"])
BOWL_WALL_THICKNESS = float(_bowl["geometry"]["wall_thickness"])
BOWL_MASS_KG = float(_bowl["mass"])
_photo_layout = SCENE_SPEC["visual_reference"][
    "photo_layout_placeholders_in_base_frame"
]
BOWL_CENTER_XY_PHOTO_PLACEHOLDER = tuple(
    float(value) for value in _photo_layout["bowl_center_xy"]
)
BOWL_CENTER_XY_PLACEHOLDER = BOWL_CENTER_XY_PHOTO_PLACEHOLDER
BOWL_COLOR = (0.82, 0.88, 0.90)
BOWL_OPACITY = 0.16

_target = SCENE_SPEC["target_object"]
TARGET_BALL_DIAMETER = float(_target["geometry"]["diameter"])
TARGET_BALL_RADIUS = float(_target["geometry"]["radius"])
TARGET_BALL_MASS_KG = float(_target["mass"])
TARGET_BALL_COLOR = (0.82, 0.035, 0.025)

# The photograph shows the ball already in the bowl. Keep that reconstruction
# as the default inspection layout, while providing a separate task-ready pose
# for pick-and-place experiments. These X/Y values are visual placeholders, not
# calibrated measurements.
SCENE_LAYOUT_MODE = os.environ.get(
    "SO101_SCENE_LAYOUT", "photo_reference"
).strip().lower()
if SCENE_LAYOUT_MODE not in {"photo_reference", "task_ready"}:
    raise ValueError(
        "SO101_SCENE_LAYOUT must be 'photo_reference' or 'task_ready', got "
        f"{SCENE_LAYOUT_MODE!r}"
    )

TARGET_BALL_XY_TASK_READY_PLACEHOLDER = tuple(
    float(value) for value in _photo_layout["task_ready_ball_center_xy"]
)
if SCENE_LAYOUT_MODE == "photo_reference":
    TARGET_BALL_XY_PLACEHOLDER = BOWL_CENTER_XY_PLACEHOLDER
    target_ball_z = TABLE_TOP_Z + BOWL_WALL_THICKNESS + TARGET_BALL_RADIUS
else:
    TARGET_BALL_XY_PLACEHOLDER = TARGET_BALL_XY_TASK_READY_PLACEHOLDER
    target_ball_z = TABLE_TOP_Z + TARGET_BALL_RADIUS

TARGET_BALL_CENTER = (
    TARGET_BALL_XY_PLACEHOLDER[0],
    TARGET_BALL_XY_PLACEHOLDER[1],
    target_ball_z,
)

_cameras = SCENE_SPEC["cameras"]
CAMERA_RESOLUTION = (
    int(_cameras["common"]["capture_resolution"]["width"]),
    int(_cameras["common"]["capture_resolution"]["height"]),
)
CAMERA_FPS = int(_cameras["common"]["fps"])
CAMERA_HORIZONTAL_APERTURE_MM_PLACEHOLDER = 20.955
CAMERA_VERTICAL_APERTURE_MM_PLACEHOLDER = (
    CAMERA_HORIZONTAL_APERTURE_MM_PLACEHOLDER
    * CAMERA_RESOLUTION[1]
    / CAMERA_RESOLUTION[0]
)
TOPDOWN_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER = 18.0
TOPDOWN_CAMERA_HEIGHT_ABOVE_TABLE = float(
    _cameras["top_down_camera"]["approximate_measurements"][
        "vertical_distance_from_tabletop"
    ]
)
# This provisional X/Y centers the view over the task objects; it is not a
# measured camera extrinsic.
TOPDOWN_CAMERA_POSITION_PLACEHOLDER = (
    (TARGET_BALL_CENTER[0] + BOWL_CENTER_XY_PLACEHOLDER[0]) / 2.0,
    (TARGET_BALL_CENTER[1] + BOWL_CENTER_XY_PLACEHOLDER[1]) / 2.0,
    TABLE_TOP_Z + TOPDOWN_CAMERA_HEIGHT_ABOVE_TABLE,
)

# Debug camera: not one of the two real policy cameras.
DEBUG_CAMERA_EYE_PLACEHOLDER = (0.62, -0.32, 0.37)
DEBUG_CAMERA_TARGET_PLACEHOLDER = (0.02, 0.30, 0.04)
DEBUG_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER = 24.0

# Angled overview used only to inspect the complete reconstructed workspace.
# It is a simulation/debug view, not a measured physical camera.
BIRDSEYE_CAMERA_EYE = (-1.05, -0.80, 1.00)
BIRDSEYE_CAMERA_TARGET = (TABLE_CENTER_X, TABLE_CENTER_Y, -0.03)
BIRDSEYE_CAMERA_FOCAL_LENGTH_MM = 24.0

# Identity until hand-eye calibration provides T_gripper_to_wrist_camera.
WRIST_CAMERA_TRANSLATION_PLACEHOLDER = (0.0, 0.0, 0.0)
WRIST_CAMERA_ROTATION_PLACEHOLDER_DEG = (0.0, 0.0, 0.0)
GRIPPER_LINK_PRIM_PATH = (
    "/World/Robot/Geometry/base_link/shoulder_link/upper_arm_link/"
    "lower_arm_link/wrist_link/gripper_link"
)
MOVING_JAW_LINK_PRIM_PATH = (
    GRIPPER_LINK_PRIM_PATH + "/moving_jaw_so101_v1_link"
)
GRIPPER_DRIVE_PRIM_PATH = "/World/Robot/Physics/gripper"

# DP-S2015-Elite tactile configuration. Product limits are measured
# specifications. The legacy procedural MVP remains selectable for regression
# comparisons, while the default geometry uses the supplied production CAD.
_tactile = SCENE_SPEC["tactile"]
_tactile_housing = _tactile["housing"]
_tactile_signal = _tactile["signal"]
_tactile_mvp = _tactile["first_sim_mvp"]
TACTILE_SENSOR_MODEL = str(_tactile["model"])
TACTILE_SENSOR_PRODUCT_CODE = str(_tactile["product_code"])
TACTILE_SENSOR_COUNT = int(_tactile["count"])
TACTILE_TAXEL_COUNT = int(_tactile_signal["taxel_count"])
TACTILE_OUTPUT_FREQUENCY_HZ = float(
    _tactile_signal["output_frequency_hz"]
)
TACTILE_MVP_PHYSICS_FREQUENCY_HZ = float(
    _tactile_mvp["physics_frequency_hz"]
)
TACTILE_DEFAULT_GRIPPER_OPEN_DEG = float(
    _tactile_mvp["default_gripper_open_deg"]
)
TACTILE_FORCE_RESOLUTION_N = float(_tactile_signal["resolution"])
TACTILE_MIN_FORCE_N = float(_tactile_signal["minimum_detectable_force"])
TACTILE_NORMAL_FORCE_RANGE_N = tuple(
    float(value) for value in _tactile_signal["normal_force_range"]
)
TACTILE_SHEAR_FORCE_RANGE_N = tuple(
    float(value) for value in _tactile_signal["shear_force_range"]
)
TACTILE_CONTACT_RADIUS_PLACEHOLDER = float(
    _tactile_mvp["contact_region_radius_placeholder"]
)

# Initial spatial decoder for the 52-value firmware-like output. These values
# are deliberately marked as placeholders: the SoTac corpus can constrain the
# real spatial spread, but it does not provide paired ground-truth contact
# locations. The mapper therefore starts with a local Gaussian over the six
# nearest measured taxels and preserves every accepted PhysX contact force.
TACTILE_TAXEL_KERNEL_SIGMA_PLACEHOLDER_M = 0.0015
TACTILE_TAXEL_NEIGHBOR_COUNT = 6
# Reject contacts on the adapter/back of the convex sensor solid. Contacts on
# the measured active hull are at most roughly half a taxel pitch from a point.
TACTILE_TAXEL_MAX_CONTACT_DISTANCE_M = 0.004

TACTILE_GEOMETRY_MODE = os.environ.get(
    "SO101_TACTILE_GEOMETRY", "cad"
).strip().lower()
if TACTILE_GEOMETRY_MODE not in {"cad", "mvp"}:
    raise ValueError(
        "SO101_TACTILE_GEOMETRY must be 'cad' or 'mvp', got "
        f"{TACTILE_GEOMETRY_MODE!r}"
    )

TACTILE_COLLISION_MODE = os.environ.get(
    "SO101_TACTILE_COLLISION", "high_fidelity"
).strip().lower()
if TACTILE_COLLISION_MODE not in {"high_fidelity", "fast"}:
    raise ValueError(
        "SO101_TACTILE_COLLISION must be 'high_fidelity' or 'fast', got "
        f"{TACTILE_COLLISION_MODE!r}"
    )

# HOOPS preserves the STEP authoring unit (millimetres) in these derived USD
# layers. USD references do not automatically rescale between layer metrics, so
# the scene builder applies this explicit uniform scale.
TACTILE_CAD_SCALE_TO_METERS = 0.001
TACTILE_CAD_DIRECTORY = PROJECT_ROOT / "assets" / "tactile" / "usd"
FIXED_ADAPTER_CAD_USD_PATH = (
    TACTILE_CAD_DIRECTORY / "Wrist_Roll_Follower_SO101_v2.usd"
)
MOVING_ADAPTER_CAD_USD_PATH = (
    TACTILE_CAD_DIRECTORY / "Moving_Jaw_SO101_v4_split.usd"
)
TACTILE_SENSOR_CAD_USD_PATH = TACTILE_CAD_DIRECTORY / "sensor.usd"

# Reference only the adapter subassemblies. The embedded sensor copies in the
# two assembly files are intentionally not referenced; one shared standalone
# sensor asset is mounted below each link instead.
FIXED_ADAPTER_CAD_PRIM_PATH = (
    "/Wrist_Roll_Follower_SO101_v2/tn__20260602011043117_dP7ADGJ/"
    "tn__Gripper1_XG"
)
MOVING_ADAPTER_CAD_PRIM_PATH = (
    "/Moving_Jaw_SO101_v4_split/Moving_Jaw_SO101/tn__Gripper1_XG"
)
FIXED_ADAPTER_CAD_MESH_RELATIVE_PATH = "Body1/Mesh"
MOVING_ADAPTER_CAD_MESH_RELATIVE_PATH = "Body15/Mesh"
TACTILE_SENSOR_CAD_PRIM_PATH = "/sensor"
TACTILE_SENSOR_CAD_MAIN_MESH_RELATIVE_PATH = (
    "tn__20151000_27_1_1_1_bP5/Mesh"
)
TACTILE_SENSOR_CAD_SECONDARY_MESH_RELATIVE_PATH = (
    "tn__20151000_27_1_1_1_bP5/Mesh_1"
)

# Product envelope used by the legacy procedural MVP. The supplied CAD and the
# 52 measured taxel positions use their native sensor frame instead: the active
# surface faces local +Z and its crown is offset along +Y.
TACTILE_SENSOR_SIZE_SENSOR_FRAME = (
    float(_tactile_housing["width"]),
    float(_tactile_housing["length"]),
    float(_tactile_housing["depth"]),
)
TACTILE_SENSOR_COLOR = (0.018, 0.020, 0.023)
TACTILE_DOME_FOOTPRINT_EXPONENT_PLACEHOLDER = float(
    _tactile_housing["dome_footprint_exponent_placeholder"]
)
TACTILE_DOME_PROFILE_EXPONENT_PLACEHOLDER = float(
    _tactile_housing["dome_profile_exponent_placeholder"]
)
TACTILE_DOME_RADIAL_SEGMENTS = int(
    _tactile_housing["dome_radial_segments"]
)
TACTILE_DOME_ANGULAR_SEGMENTS = int(
    _tactile_housing["dome_angular_segments"]
)

if TACTILE_GEOMETRY_MODE == "cad":
    # Central measured taxel 26, used as the reproducible contact-test point.
    # The visual CAD crown lies about 0.46 mm away in Y between taxel samples.
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME = (
        -2.0e-11,
        0.00863424733,
        0.00669381112,
    )
    # The device reports compressive normal load as positive Fz, whereas the
    # physical contact force on a +Z-facing pad points along -Z.
    TACTILE_NORMAL_FORCE_FROM_PHYSICAL_SIGN = -1.0
else:
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME = (
        0.0,
        0.0,
        -TACTILE_SENSOR_SIZE_SENSOR_FRAME[2],
    )
    TACTILE_NORMAL_FORCE_FROM_PHYSICAL_SIGN = 1.0

FIXED_TACTILE_ROOT_PRIM_PATH = GRIPPER_LINK_PRIM_PATH + "/TactileFixedMount"
MOVING_TACTILE_ROOT_PRIM_PATH = (
    MOVING_JAW_LINK_PRIM_PATH + "/TactileMovingMount"
)
TACTILE_ROOT_PRIM_PATHS = (
    FIXED_TACTILE_ROOT_PRIM_PATH,
    MOVING_TACTILE_ROOT_PRIM_PATH,
)
# Isaac ContactSensor must be a direct child of the monitored rigid body.
# It is not a UsdGeom.Xform, so keep the visual/collision mount as a separate
# sibling Xform; otherwise its authored installation transform is ignored.
TACTILE_SENSOR_PRIM_PATHS = (
    GRIPPER_LINK_PRIM_PATH + "/TactileFixedContactSensor",
    MOVING_JAW_LINK_PRIM_PATH + "/TactileMovingContactSensor",
)
TACTILE_BODY_PRIM_PATHS = (
    GRIPPER_LINK_PRIM_PATH,
    MOVING_JAW_LINK_PRIM_PATH,
)

FIXED_STOCK_VISUAL_PRIM_PATH = (
    GRIPPER_LINK_PRIM_PATH + "/wrist_roll_follower_so101_v1"
)
FIXED_STOCK_COLLISION_PRIM_PATH = (
    GRIPPER_LINK_PRIM_PATH + "/wrist_roll_follower_so101_v1_1"
)
MOVING_STOCK_VISUAL_PRIM_PATH = (
    MOVING_JAW_LINK_PRIM_PATH + "/moving_jaw_so101_v1"
)
MOVING_STOCK_COLLISION_PRIM_PATH = (
    MOVING_JAW_LINK_PRIM_PATH + "/moving_jaw_so101_v1_1"
)

# Component occurrence transforms read from the two supplied STEP assemblies.
# Rows are component-local +X/+Y/+Z axes in the assembly frame; translations
# remain in the source CAD unit (millimetres). The builder aligns each Gripper
# component with the corresponding stock part, then derives the sensor pose from
# the same assembly relationship. No hand-tuned fingertip offset is involved.
FIXED_ADAPTER_CAD_BASIS_ROWS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
FIXED_ADAPTER_CAD_TRANSLATION_MM = (0.0, 0.0, 0.0)
MOVING_ADAPTER_CAD_BASIS_ROWS = FIXED_ADAPTER_CAD_BASIS_ROWS
MOVING_ADAPTER_CAD_TRANSLATION_MM = (
    0.0786001712928885,
    -0.414473414565224,
    -0.0573458015966444,
)

FIXED_SENSOR_CAD_BASIS_ROWS = (
    (-2.25291895361579e-7, 0.9999999999999584, 1.8243777949455975e-7),
    (-0.20791169081779481, -2.25291895361579e-7, 0.9781476007337726),
    (0.9781476007337726, 1.8243777938353745e-7, 0.20791169081783645),
)
FIXED_SENSOR_CAD_TRANSLATION_MM = (
    -5.21799364250168,
    2.45308925673941e-6,
    85.8659045331667,
)
MOVING_SENSOR_CAD_BASIS_ROWS = (
    (-2.220446049250313e-16, 2.7755575615628914e-16, -1.0),
    (0.20791169081779187, -0.9781476007337986, -3.191891195797325e-16),
    (-0.9781476007337987, -0.20791169081779187, 1.1102230246251565e-16),
)
MOVING_SENSOR_CAD_TRANSLATION_MM = (
    -14.6376779862402,
    -62.8627119341432,
    -2.22062886118284e-15,
)

# These values come from the stock SO-101 mesh tip bounds. They place each
# sensor mounting-plane center at the current inner fingertip surface near the
# distal 20 mm. They reference the configurable modified fingertip, not measured
# transforms of the real custom metal mount.
FIXED_TACTILE_TRANSLATION_PLACEHOLDER = (-0.0079, -0.000218, -0.0944)
MOVING_TACTILE_TRANSLATION_PLACEHOLDER = (-0.0123, -0.0710, 0.0190)

# Rows are sensor-local +X/+Y/+Z axes expressed in the parent-body frame.
# Fixed: +Z -> -gripper X. Moving: +Z -> +gripper X.
FIXED_TACTILE_BASIS_ROWS = (
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
)
MOVING_TACTILE_BASIS_ROWS = (
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
)


# (static friction, dynamic friction, restitution).
# The ball/pad pair is measured, not estimated: with the ball correctly seated
# on both tactile pads, scripts/experiments/test_seated_grasp.py shows it rolls
# straight back out at 1.2 and is only retained from 1.6 upwards.  The earlier
# 1.20 gripper_pad figure was an unused placeholder -- nothing in the project
# ever bound these materials, so the scene ran on PhysX defaults throughout.
# Contact offset is deliberately left to PhysX until it is measured against the
# WHOLE task.  scripts/experiments/test_seated_grasp.py showed 1 mm and 2 mm to
# be identical through the grasp and >=5 mm to kill the taxel signal outright,
# but that trial stops at the lift: authoring 2 mm from it made the full
# pick-and-place throw the ball during the carry (both-pad contact 87% -> 39%).
# Set a value here only with full-task evidence behind it.
CONTACT_OFFSET_M = None
REST_OFFSET_M = 0.0

PHYSICS_MATERIALS = {
    "floor": (0.85, 0.70, 0.05),
    "table": (0.35, 0.25, 0.03),
    "plastic": (0.55, 0.42, 0.05),
    "gripper_pad": (1.60, 1.36, 0.00),
    "plush_ball": (1.60, 1.36, 0.00),
}

GRIPPER_CONTACT_PRIM_PATHS = (
    GRIPPER_LINK_PRIM_PATH,
    MOVING_JAW_LINK_PRIM_PATH,
)

# Legacy aliases keep older scripts importable. They now describe the ball and
# bowl, and the old cube-grasp policy still requires retuning before use.
TARGET_CUBE_SIZE = TARGET_BALL_DIAMETER
TARGET_CUBE_CENTER = TARGET_BALL_CENTER
TARGET_CUBE_COLOR = TARGET_BALL_COLOR
TARGET_CUBE_MASS_KG = TARGET_BALL_MASS_KG
DISTRACTOR_CUBES = ()
BASKET_CENTER = BOWL_CENTER_XY_PLACEHOLDER
BASKET_INNER_LENGTH = BOWL_OUTER_TOP_DIAMETER - 2.0 * BOWL_WALL_THICKNESS
BASKET_INNER_WIDTH = BASKET_INNER_LENGTH
BASKET_WALL_HEIGHT = BOWL_HEIGHT
BASKET_WALL_THICKNESS = BOWL_WALL_THICKNESS
BASKET_BASE_THICKNESS = BOWL_WALL_THICKNESS
BASKET_COLOR = BOWL_COLOR
CAMERA_HORIZONTAL_APERTURE_MM = CAMERA_HORIZONTAL_APERTURE_MM_PLACEHOLDER
CAMERA_VERTICAL_APERTURE_MM = CAMERA_VERTICAL_APERTURE_MM_PLACEHOLDER
TOPDOWN_CAMERA_FOCAL_LENGTH_MM = TOPDOWN_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER
EXTERNAL_CAMERA_FOCAL_LENGTH_MM = DEBUG_CAMERA_FOCAL_LENGTH_MM_PLACEHOLDER
EXTERNAL_CAMERA_EYE = DEBUG_CAMERA_EYE_PLACEHOLDER
EXTERNAL_CAMERA_TARGET = DEBUG_CAMERA_TARGET_PLACEHOLDER
