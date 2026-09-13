"""Config for the DH116 pen-spinning environment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from dh116_hand_lab.hand_cfg import (
    ACTUATED_JOINTS, DH116_HAND_CFG, FINGERTIP_BODIES, PEN_HOME_HAND, PEN_SPAWN_HAND,
    hand_to_world)

PEN_RADIUS = 0.005
PEN_LENGTH = 0.14
PEN_MASS = 0.015
PEN_REST_WORLD = hand_to_world(PEN_HOME_HAND)
PEN_SPAWN_WORLD = hand_to_world(PEN_SPAWN_HAND)

PEN_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Pen",
    spawn=sim_utils.CapsuleCfg(
        radius=PEN_RADIUS,
        height=PEN_LENGTH - 2.0 * PEN_RADIUS,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=PEN_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.1), roughness=0.5),
    ),
    # lying across the palm: capsule Z -> world Y (90 deg about X), xyzw
    init_state=RigidObjectCfg.InitialStateCfg(pos=PEN_SPAWN_WORLD, rot=(0.7071068, 0.0, 0.0, 0.7071068)),
)


@configclass
class DH116PenSpinEnvCfg(DirectRLEnvCfg):
    """Spin a pen lying on the up-facing palm about the palm normal."""

    decimation = 2
    episode_length_s = 8.0
    action_space = len(ACTUATED_JOINTS)   # 6
    observation_space = 41
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysxCfg(
            solve_articulation_contact_last=True,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**22,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.6, replicate_physics=True, clone_in_fabric=True)

    robot_cfg: ArticulationCfg = DH116_HAND_CFG
    pen_cfg: RigidObjectCfg = PEN_CFG
    # optional per-env camera for recordings (needs --enable_cameras)
    camera_cfg: CameraCfg | None = None
    actuated_joint_names = ACTUATED_JOINTS
    fingertip_body_names = FINGERTIP_BODIES

    # reset noise
    reset_position_noise = 0.01      # pen xy (m)
    reset_dof_pos_noise = 0.1        # fraction of joint range
    randomize_pen_yaw = True
    train_yaw_center = 0.0
    train_yaw_range = math.pi
    # Evaluation only: equally spaced yaw bins, repeated across environments.
    eval_yaw_bins: int = 0
    eval_yaw_center = 0.0
    eval_yaw_range = math.pi
    capture_step_state: bool = False

    # reward
    spin_sign = -1.0                 # index push / thumb sweep both turn the pen this way (world -Z)
    spin_reward_scale = 1.0
    forward_spin_reward_scale = 0.0
    target_spin_reward_scale = 0.0
    target_spin_rate = 3.0
    target_spin_sigma = 1.5
    # Curriculum reward: only newly achieved net heading progress is paid.
    # This cannot be farmed by oscillating back and forth.
    best_progress_reward_scale = 0.0
    progress_goal_bonus_scale = 0.0
    progress_target_angle = math.pi / 2
    progress_observation = False
    spin_reward_mode = "omega_z"    # v1 compatibility; v2 uses long-axis heading change
    spin_clip = (-2.0, 10.0)         # rad/s, before scaling by 1/10
    alive_bonus = 0.1
    fall_penalty = -10.0
    fall_dist = 0.10                 # horizontal distance from the home point
    fall_height = 0.06               # below the home height (off the hand)
    action_penalty_scale = -0.001
    action_rate_scale = -0.01
    palm_center_penalty_scale = 0.0
    # Optional plane-alignment cost; zero preserves the original v1 objective.
    tilt_penalty_scale = 0.0
    vel_obs_scale = 0.2
    act_moving_average = 0.5
