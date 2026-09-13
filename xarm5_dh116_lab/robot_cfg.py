"""Isaac Lab articulation config for the merged xArm5 + DH116 asset."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from dh116_joint_model import (
    FINGER_FLEXION_MAX, FINGER_PASSIVE_MAX, FINGER_PASSIVE_MIN,
    THUMB_ABDUCTION_MAX, THUMB_FLEXION_MAX, THUMB_PASSIVE_MAX,
    THUMB_PASSIVE_MIN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_USD = (PROJECT_ROOT / "assets" / "xarm5_dh116" / "usd_manual_2026_04_self_collision" / "xarm5_dh116"
             / "xarm5_dh116.usda")

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]

# DH116: finger1 is the thumb (abduction 11, flexion 12, passive IP 13); the
# other fingers each have an actuated MCP (x1) and a passive PIP (x2).  finger2
# sits on the thumb side of the palm (+Y of the hand base) and is the index.
HAND_JOINTS = [
    "hand_finger11", "hand_finger12", "hand_finger13",
    "hand_finger21", "hand_finger22",
    "hand_finger31", "hand_finger32",
    "hand_finger41", "hand_finger42",
    "hand_finger51", "hand_finger52",
]
ALL_JOINTS = ARM_JOINTS + HAND_JOINTS

# Leadshine 2026-04-09 ranges, mirrored in the calibrated URDF/USD.
THUMB_IP_MAX = THUMB_PASSIVE_MAX
FINGER_MCP_MAX = FINGER_FLEXION_MAX
FINGER_PIP_MAX = FINGER_PASSIVE_MAX
ACTIVE_HAND_JOINTS = ["hand_finger11", "hand_finger12", "hand_finger21",
                      "hand_finger31", "hand_finger41", "hand_finger51"]
PASSIVE_HAND_JOINTS = ["hand_finger13", "hand_finger22", "hand_finger32",
                       "hand_finger42", "hand_finger52"]

# "Presenting" pose: forearm horizontal towards +X, wrist bent up 90 deg so the
# fingers point at +Z and the palm faces +X (joint5 = pi turns the palm to the
# front).  joints 2-4 share a parallel axis; their sum of -pi is what points
# the fingers up.  From xarm5_dh116_lab.kinematics: hand base at
# (0.44, 0, 0.69) m above the robot base, fingertips ~0.85 m.
PRESENT_J2 = 0.45
ARM_PRESENT = {
    "joint1": 0.0,
    "joint2": PRESENT_J2,
    "joint3": -PRESENT_J2 - math.pi / 2.0,
    "joint4": -math.pi / 2.0,
    "joint5": math.pi,
}

XARM5_DH116_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # The imported asset already owns a world-anchored
            # /Physics/root_joint; asking the spawner for another fixed
            # joint at the articulation root is invalid (same as SO-101).
            fix_root_link=None,
            # Shared collider offsets and stable solver settings are authored
            # in the corrected asset; fingers must stop at physical contact.
            enabled_self_collisions=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # Identity, written in (x, y, z, w) order.  Same importer layout as
        # the SO-101 asset (world-anchored root_joint, ArticulationRootAPI on
        # the non-rigid /Geometry Xform) and the same trap: the root-pose
        # write path consumes this field shifted by one component, so the
        # documented (1, 0, 0, 0) lands as a 180 deg turn about X and hangs
        # the arm upside down under the pedestal (record_gestures.py --probe
        # showed every link at -z of the FK).  See isaac-lab clone notes.
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={**ARM_PRESENT, "hand_finger(11|12|21|31|41|51)": 0.0,
                   "hand_finger13": THUMB_PASSIVE_MIN + 1.0e-5,
                   "hand_finger(22|32|42|52)": FINGER_PASSIVE_MIN + 1.0e-5},
    ),
    actuators={
        # The USD carries 400 N*m/rad stiffness with very heavy damping
        # (10 N*m*s/deg); a critically damped arm settles the same but does
        # not drag through the wrist bobs.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-5]"],
            stiffness=400.0,
            damping=40.0,
            effort_limit_sim={"joint1": 50.0, "joint2": 50.0, "joint3": 30.0,
                              "joint4": 20.0, "joint5": 20.0},
        ),
        # Finger links weigh 7-20 g: a few N*m/rad already leaves them
        # millidegrees from target under gravity, and the 3 rad/s joint
        # velocity cap in the USD is what sets the closing speed.
        "hand_active": ImplicitActuatorCfg(
            joint_names_expr=ACTIVE_HAND_JOINTS,
            stiffness=3.0,
            damping=0.15,
            effort_limit_sim=2.0,
        ),
        "hand_passive_coupling": ImplicitActuatorCfg(
            joint_names_expr=PASSIVE_HAND_JOINTS,
            stiffness=3.0,
            damping=0.15,
            effort_limit_sim=2.0,
        ),
    },
)
