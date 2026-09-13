"""Clone-safe Isaac Lab scene configuration for tactile integration tests."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from so101_isaac_lab.sensors import DPS2015ContactSensorCfg
from scene_config import (
    BOWL_CENTER_XY_PLACEHOLDER,
    FLOOR_Z,
    ROBOT_BASE_YAW_DEG,
    PHYSICS_MATERIALS,
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
    TABLE_TOP_THICKNESS,
    TABLE_TOP_Z,
    TABLE_WIDTH,
    TARGET_BALL_COLOR,
    TARGET_BALL_MASS_KG,
    TARGET_BALL_RADIUS,
    TARGET_BALL_XY_TASK_READY_PLACEHOLDER,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TACTILE_ROBOT_USD = PROJECT_ROOT / "assets" / "isaac_lab" / "so101_tactile.usda"

GRIPPER_LINK_RELATIVE_PATH = (
    "Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/"
    "wrist_link/gripper_link"
)
MOVING_JAW_RELATIVE_PATH = (
    GRIPPER_LINK_RELATIVE_PATH + "/moving_jaw_so101_v1_link"
)


SO101_TACTILE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TACTILE_ROBOT_USD),
        # The current USD already applies PhysxContactReportAPI to exactly the
        # two fingertip rigid bodies; avoid redundantly authoring the schema
        # over every rigid body in the referenced asset.
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=2,
            # The imported asset already owns /Physics/root_joint.  Asking the
            # spawner to create another fixed joint at /Geometry is invalid.
            fix_root_link=None,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The imported asset faces +X, but every object position in
        # scene_config is expressed for a base yawed to face +Y -- that is what
        # ROBOT_ASSET_YAW_OFFSET_DEG exists for, and lab_scene_task.usda bakes
        # it into the robot Xform.  Without it here the arm looks sideways and,
        # worse, cannot reach the ball at all.
        rot=(
            math.cos(math.radians(ROBOT_BASE_YAW_DEG) / 2.0),
            0.0,
            0.0,
            math.sin(math.radians(ROBOT_BASE_YAW_DEG) / 2.0),
        ),
        # All-zero joints leave the arm reaching backwards and below the table:
        # the scripted task never shows that because it drives straight to an
        # IK solution.  These are the median angles the real arm holds while
        # grasping (t = 5.0-6.5 s across the 21 red-ball episodes), so the arm
        # starts upright over the table.  Its own idle pose was the first
        # choice, but that has the arm folded back off the table edge.
        joint_pos={
            "shoulder_pan": -0.0093,
            "shoulder_lift": 0.0713,
            "elbow_flex": 0.2108,
            "wrist_flex": 1.3826,
            "wrist_roll": -0.1559,
            "gripper": 0.2333,
        }
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=10.0,
            # The scripted standalone task runs on the drive gains authored in
            # the USD: 0.31 N*m/deg and 0.0105 N*m*s/deg -- USD angular drives
            # are per-degree, so in Isaac Lab's per-radian units that is 17.8
            # and 0.6.  The previous 8.0 was less than half as stiff and let
            # the arm sag visibly under gravity.
            stiffness=17.8,
            damping=0.6,
        )
    },
)


# Initial state for the pick-and-place scene, replacing the shared smoke-scene
# defaults: the same semantic base yaw, but idling in the scripted task's
# "approach" pose.
#
# The rot is authored in (x, y, z, w) order even though the field is
# documented as (w, x, y, z): for this asset (floating-base articulation held
# by a world fixed joint, ArticulationRootAPI on the non-rigid /Geometry
# Xform) the root-pose write path consumes the quaternion shifted by one
# component -- phys = (w=cmd[3], x=cmd[0], y=cmd[1], z=cmd[2]).  Calibrated
# against the standalone task's measured gripper pose: two probe poses match
# the standalone to within droop noise only under this permutation (see
# scripts/isaac_lab/record_parallel_pick_place.py --probe_pose).  With the
# documented order the robot ends up pitched 90 degrees, lying on its side --
# the broken mount in the first parallel render.
_TASK_INIT_STATE = ArticulationCfg.InitialStateCfg(
    rot=(
        0.0,
        0.0,
        math.sin(math.radians(ROBOT_BASE_YAW_DEG) / 2.0),
        math.cos(math.radians(ROBOT_BASE_YAW_DEG) / 2.0),
    ),
    joint_pos={
        # The scripted task's "approach" IK solution (hover above the ball,
        # jaw open), from calibration/pick_place_waypoints_nominal.json.  It
        # is collision-free by construction, and idling here means the replay
        # timeline starts from exactly this command.
        "shoulder_pan": 0.2939,
        "shoulder_lift": -0.2482,
        "elbow_flex": 1.0409,
        "wrist_flex": -0.5305,
        "wrist_roll": -1.5140,
        "gripper": 1.2217,
    },
)


@configclass
class TactileSmokeSceneCfg(InteractiveSceneCfg):
    """Small scene used to validate batched sensor and observation plumbing."""

    lazy_sensor_update = False

    robot: ArticulationCfg = SO101_TACTILE_CFG

    probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        spawn=sim_utils.SphereCfg(
            radius=0.004,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.8, 1.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    tactile_fixed = DPS2015ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_LINK_RELATIVE_PATH}",
        sensor_frame_prim_name="TactileFixedMount",
        update_period=0.0,
        history_length=0,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Probe"],
        debug_vis=False,
    )

    tactile_moving = DPS2015ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{MOVING_JAW_RELATIVE_PATH}",
        sensor_frame_prim_name="TactileMovingMount",
        update_period=0.0,
        history_length=0,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Probe"],
        debug_vis=False,
    )


def _pedestal_part_cfg(
    name: str, pedestal_index: int, size: tuple[float, float, float],
    center_z: float,
) -> AssetBaseCfg:
    """One cuboid of a table pedestal, sized/placed from scene_config."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TABLE_FRAME_COLOR, roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_PEDESTAL_X_POSITIONS[pedestal_index], TABLE_CENTER[1],
                 center_z)),
    )


@configclass
class BallPickPlaceSceneCfg(InteractiveSceneCfg):
    """The pick-and-place task, cloned across parallel environments.

    Same geometry and physics as lab_scene_task.usda, but declared as spawn
    configs so each environment gets its own copy.  The values are the ones
    already calibrated for the scripted task -- friction 1.6 is what holds the
    ball, and the contact filters must name the ball, since the smoke scene
    filters on a probe and would otherwise report nothing at all.
    """

    lazy_sensor_update = False

    # Isaac Lab's stock grid plane -- the same asset every published example
    # renders on, so the demo reads as an Isaac Lab scene rather than a robot
    # floating in white space.  It sits at the measured lab-floor height, the
    # tabletop being z = 0; the tables stand on their pedestals instead of
    # having the plane glued to their underside.  Declared before the robot
    # because InteractiveScene spawns terrain first.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, FLOOR_Z)),
    )

    robot: ArticulationCfg = SO101_TACTILE_CFG.replace(
        init_state=_TASK_INIT_STATE)

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_LENGTH, TABLE_WIDTH, TABLE_TOP_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=PHYSICS_MATERIALS["table"][0],
                dynamic_friction=PHYSICS_MATERIALS["table"][1],
                restitution=PHYSICS_MATERIALS["table"][2],
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TABLE_COLOR, roughness=0.88),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_CENTER[2])),
    )

    # The two T-shaped pedestals from the standalone scene, so the tables
    # stand on the floor instead of reading as slabs lying on the ground
    # plane.  Visual-only: nothing in the task can touch them, so collision
    # shapes would just add parse cost across the clone grid.
    pedestal_a_foot = _pedestal_part_cfg(
        "PedestalAFoot", 0, TABLE_PEDESTAL_FOOT_SIZE_PLACEHOLDER,
        TABLE_PEDESTAL_FOOT_CENTER_Z)
    pedestal_a_column = _pedestal_part_cfg(
        "PedestalAColumn", 0,
        (*TABLE_PEDESTAL_COLUMN_SIZE_XY_PLACEHOLDER,
         TABLE_PEDESTAL_COLUMN_HEIGHT),
        TABLE_PEDESTAL_COLUMN_CENTER_Z)
    pedestal_a_top = _pedestal_part_cfg(
        "PedestalATop", 0, TABLE_PEDESTAL_TOP_SUPPORT_SIZE_PLACEHOLDER,
        TABLE_PEDESTAL_TOP_SUPPORT_CENTER_Z)
    pedestal_b_foot = _pedestal_part_cfg(
        "PedestalBFoot", 1, TABLE_PEDESTAL_FOOT_SIZE_PLACEHOLDER,
        TABLE_PEDESTAL_FOOT_CENTER_Z)
    pedestal_b_column = _pedestal_part_cfg(
        "PedestalBColumn", 1,
        (*TABLE_PEDESTAL_COLUMN_SIZE_XY_PLACEHOLDER,
         TABLE_PEDESTAL_COLUMN_HEIGHT),
        TABLE_PEDESTAL_COLUMN_CENTER_Z)
    pedestal_b_top = _pedestal_part_cfg(
        "PedestalBTop", 1, TABLE_PEDESTAL_TOP_SUPPORT_SIZE_PLACEHOLDER,
        TABLE_PEDESTAL_TOP_SUPPORT_CENTER_Z)

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetBall",
        spawn=sim_utils.SphereCfg(
            radius=TARGET_BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=TARGET_BALL_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=PHYSICS_MATERIALS["plush_ball"][0],
                dynamic_friction=PHYSICS_MATERIALS["plush_ball"][1],
                restitution=PHYSICS_MATERIALS["plush_ball"][2],
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TARGET_BALL_COLOR, roughness=0.96),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TARGET_BALL_XY_TASK_READY_PLACEHOLDER[0],
                 TARGET_BALL_XY_TASK_READY_PLACEHOLDER[1],
                 TABLE_TOP_Z + TARGET_BALL_RADIUS)),
    )

    # The real concave bowl, split in two so each environment can place its
    # own (see the header comments in the two assets): a kinematic rigid
    # body carries the colliders and accepts per-env pose writes but never
    # renders, and a collision-free static twin does the drawing and follows
    # per-env USD transform edits.  Both default to the scripted trajectory's
    # measured release point -- the Lab grasp seats the ball at the pad
    # apexes, ~40 mm shallower in the hand than the standalone's deep seat,
    # so the drop lands there rather than at the calibrated scene position.
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PROJECT_ROOT / "assets" / "isaac_lab"
                         / "bowl_phys.usda"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2588, 0.2467, TABLE_TOP_Z)),
    )

    bowl_vis = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BowlVis",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PROJECT_ROOT / "assets" / "isaac_lab"
                         / "bowl_scenery.usda"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.2588, 0.2467, TABLE_TOP_Z)),
    )

    tactile_fixed = DPS2015ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_LINK_RELATIVE_PATH}",
        sensor_frame_prim_name="TactileFixedMount",
        update_period=0.0,
        history_length=0,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TargetBall"],
        debug_vis=False,
    )

    tactile_moving = DPS2015ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{MOVING_JAW_RELATIVE_PATH}",
        sensor_frame_prim_name="TactileMovingMount",
        update_period=0.0,
        history_length=0,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TargetBall"],
        debug_vis=False,
    )

    # Declared in the scene rather than built by hand: Isaac Lab drives the
    # render product itself, and a raw replicator product created outside the
    # scene never renders in a headless env -- the annotator just returns
    # nothing and the recording comes out empty.
    # One per environment, like every other sensor here: Isaac Lab sizes the
    # camera buffers by env count, so a single global prim fails to reset.
    # They are all driven to the same overview pose and only env 0 is read.
    overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCamera",
        update_period=0.0,
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0),
    )


@configclass
class TactilePresentationSceneCfg(InteractiveSceneCfg):
    """Single-lab presentation scene with one contact probe per fingertip."""

    lazy_sensor_update = False

    robot: ArticulationCfg = SO101_TACTILE_CFG

    presentation_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/PresentationCamera",
        update_period=0.0,
        height=1000,
        width=1600,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=55.0,
            focus_distance=0.35,
            horizontal_aperture=20.955,
            clipping_range=(0.005, 10.0),
        ),
    )

    probe_fixed = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ProbeFixed",
        spawn=sim_utils.SphereCfg(
            radius=0.004,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.02, 0.68, 1.0),
                emissive_color=(0.0, 0.06, 0.12),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    probe_moving = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ProbeMoving",
        spawn=sim_utils.SphereCfg(
            radius=0.004,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.42, 0.03),
                emissive_color=(0.12, 0.025, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    tactile_fixed = DPS2015ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_LINK_RELATIVE_PATH}",
        sensor_frame_prim_name="TactileFixedMount",
        update_period=0.0,
        history_length=0,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/ProbeFixed"],
        debug_vis=False,
    )

    tactile_moving = DPS2015ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{MOVING_JAW_RELATIVE_PATH}",
        sensor_frame_prim_name="TactileMovingMount",
        update_period=0.0,
        history_length=0,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/ProbeMoving"],
        debug_vis=False,
    )
