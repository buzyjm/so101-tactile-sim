"""Manager-based RL environment configuration for SO-101 tactile policies."""

from __future__ import annotations

import isaaclab.envs.mdp as lab_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from scene_config import TACTILE_MVP_PHYSICS_FREQUENCY_HZ
from so101_isaac_lab import mdp
from so101_isaac_lab.scene_cfg import TactileSmokeSceneCfg


SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@configclass
class ActionsCfg:
    """Six normalized joint-position commands around the default pose."""

    joint_pos = lab_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=SO101_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Policy-ready tactile tensor plus non-flattened diagnostic channels."""

    @configclass
    class PolicyCfg(ObsGroup):
        # The policy receives the normalized 2 x 52 x 3 tensor flattened in
        # stable [finger, taxel, xyz] order.
        tactile = ObsTerm(func=mdp.tactile_policy_features)

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class TactileDebugCfg(ObsGroup):
        # These tensors keep their physical units and original dimensions.
        taxel_forces = ObsTerm(func=mdp.tactile_taxel_forces)
        force_magnitudes = ObsTerm(func=mdp.tactile_force_magnitudes)
        total_forces = ObsTerm(func=mdp.tactile_total_forces)

        def __post_init__(self) -> None:
            self.concatenate_terms = False
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    tactile_debug: TactileDebugCfg = TactileDebugCfg()


@configclass
class RewardsCfg:
    """Generic tactile terms; task-specific training may reweight/replace them."""

    bilateral_contact = RewTerm(func=mdp.tactile_contact_reward, weight=1.0)
    force_balance = RewTerm(
        func=mdp.tactile_balance_reward,
        weight=0.1,
        params={"force_scale_n": 25.0},
    )
    overload = RewTerm(
        func=mdp.tactile_overload_penalty,
        weight=-0.1,
        params={"safe_normal_force_n": 20.0},
    )


@configclass
class TerminationsCfg:
    """Minimal termination used by the integration environment."""

    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)


@configclass
class SO101TactileEnvCfg(ManagerBasedRLEnvCfg):
    """Ready-to-instantiate Isaac Lab wrapper for the current tactile scene.

    This is deliberately a task-neutral interface environment.  It validates
    the sensor, observation, action, reward and reset plumbing; a manipulation
    task can subclass it and add task state, commands and success conditions.
    """

    scene: TactileSmokeSceneCfg = TactileSmokeSceneCfg(
        num_envs=2,
        env_spacing=1.0,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Four 240 Hz physics substeps per 60 Hz policy action.  The custom
        # tactile sensor still publishes independently at 83.3 Hz.
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ
        self.sim.render_interval = self.decimation

