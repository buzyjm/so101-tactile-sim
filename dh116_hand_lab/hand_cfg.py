"""Articulation config for the stand-alone DH116 hand, palm up."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from dh116_joint_model import FINGER_PASSIVE_MIN, THUMB_PASSIVE_MIN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAND_USD = (PROJECT_ROOT / "assets" / "dh116_hand" / "usd_manual_2026_04_self_collision"
            / "dh116_hand" / "dh116_hand.usda")

# Six commanded joints; five dependent drives approximate the nonlinear
# mechanical coupling published by Leadshine in April 2026.
ACTUATED_JOINTS = ["hand_finger11", "hand_finger12", "hand_finger21",
                   "hand_finger31", "hand_finger41", "hand_finger51"]
PASSIVE_JOINTS = {
    "hand_finger13": "hand_finger12",
    "hand_finger22": "hand_finger21",
    "hand_finger32": "hand_finger31",
    "hand_finger42": "hand_finger41",
    "hand_finger52": "hand_finger51",
}
FINGERTIP_BODIES = ["hand_finger14_link", "hand_finger23_link", "hand_finger33_link",
                    "hand_finger43_link", "hand_finger53_link"]

# Hand base frame (from the URDF): +X is the palm normal, +Z runs along the
# fingers, +Y towards the thumb/index side.  Palm up with the fingers along
# world +X is the 180 deg turn about (1, 0, 1)/sqrt(2): xyzw (0.7071, 0, 0.7071, 0).
HAND_BASE_POS = (0.0, 0.0, 0.5)
HAND_BASE_ROT_XYZW = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
# Palm surface where a pen actually comes to rest (measured in
# flick_test.py: a 5 mm capsule settles with its centre at x = 0.019 and
# slides to z ~ 0.08, just below the finger knuckles at z = 0.10).  The
# base_link mesh's x = 0.028 maximum is the thumb mount, not the palm.
PALM_SURFACE_X = 0.014
# Where the pen comes to rest (reference for the fall check and the
# observations) and where it is spawned: a little higher and nearer the
# wrist, so it drops onto the palm and slides into place instead of
# spawning inside the knuckle covers (that launched it at 1.7 m/s).
# The palm slopes towards the fingers: a pen left alone rolls to the finger
# bases (hand z ~ 0.12) -- where a relaxed hand holds one anyway.  The home
# point is the centre of the palm/finger area, only used as the reference
# for the fall check and the position observation.
PEN_HOME_HAND = (0.0, 0.0, 0.10)
PEN_SPAWN_HAND = (PALM_SURFACE_X + 0.014, 0.0, 0.07)
PEN_REST_HAND = PEN_HOME_HAND


def hand_to_world(point_hand: tuple[float, float, float]) -> tuple[float, float, float]:
    """Hand-frame point -> world, for the palm-up base pose above."""
    x, y, z = point_hand
    return (HAND_BASE_POS[0] + z, HAND_BASE_POS[1] - y, HAND_BASE_POS[2] + x)


DH116_HAND_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(HAND_USD),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # The importer wrote a world-anchored root_joint (fix_base=True).
            fix_root_link=None,
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=HAND_BASE_POS,
        # xyzw: Isaac Lab 3.0's root-pose write path and data API use that
        # order (see xarm5_dh116_lab.robot_cfg).
        rot=HAND_BASE_ROT_XYZW,
        joint_pos={"hand_finger(11|12|21|31|41|51)": 0.0,
                   # Keep the imported defaults a few microradians inside the
                   # limits; URDF -> USD converts through degrees/float32.
                   "hand_finger13": THUMB_PASSIVE_MIN + 1.0e-5,
                   "hand_finger(22|32|42|52)": FINGER_PASSIVE_MIN + 1.0e-5},
    ),
    actuators={
        "active_fingers": ImplicitActuatorCfg(
            joint_names_expr=ACTUATED_JOINTS,
            stiffness=3.0,
            damping=0.1,
            effort_limit_sim=2.0,
            # datasheet 3 rad/s; the env cfg may raise it for experiments
            velocity_limit_sim=3.0,
        ),
        # PhysX mimic joints only support a linear relation.  The environment
        # supplies exact quadratic targets to this compliant coupling drive.
        "passive_coupling": ImplicitActuatorCfg(
            joint_names_expr=list(PASSIVE_JOINTS),
            stiffness=3.0,
            damping=0.1,
            effort_limit_sim=2.0,
            velocity_limit_sim=3.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
