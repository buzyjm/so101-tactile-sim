"""DH116 pen spinning: a Direct RL environment.

The hand is fixed palm-up and a pen lies across the palm. v1 rewards world-Z
angular velocity; v2 rewards the pen long-axis azimuth change. Six commanded
joints drive five passive PIP/IP joints through the vendor's nonlinear curves.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from dh116_hand_lab.hand_cfg import PASSIVE_JOINTS
from dh116_hand_lab.pen_spin_env_cfg import DH116PenSpinEnvCfg, PEN_REST_WORLD
from dh116_hand_lab.rewards import projected_spin_rate, update_best_progress
from dh116_joint_model import FINGER_COUPLING, THUMB_COUPLING


class DH116PenSpinEnv(DirectRLEnv):
    cfg: DH116PenSpinEnvCfg

    def __init__(self, cfg: DH116PenSpinEnvCfg, render_mode: str | None = None, **kwargs):
        # Old checkpoints retain their 41-value observation. Progress-curriculum
        # runs append one normalized cumulative-heading value.
        cfg.observation_space = 42 if cfg.progress_observation else 41
        super().__init__(cfg, render_mode, **kwargs)
        self.num_hand_dofs = self.hand.num_joints
        names = list(self.hand.joint_names)
        self.actuated_dof_indices = [names.index(n) for n in cfg.actuated_joint_names]
        self.passive_dof_indices = [names.index(n) for n in PASSIVE_JOINTS]
        self.passive_parent_indices = [names.index(PASSIVE_JOINTS[n]) for n in PASSIVE_JOINTS]
        self.finger_bodies = sorted(self.hand.body_names.index(n) for n in cfg.fingertip_body_names)

        limits = self.hand.data.joint_limits.torch.to(self.device)
        self.dof_lower = limits[..., 0]
        self.dof_upper = limits[..., 1]

        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        self.prev_targets = torch.zeros_like(self.cur_targets)
        self.prev_actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device)

        self.pen_rest = torch.tensor(PEN_REST_WORLD, device=self.device).repeat(self.num_envs, 1)
        self.z_unit = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.spin_rate = torch.zeros(self.num_envs, device=self.device)
        self.on_palm = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.initial_yaw = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_heading = torch.zeros(self.num_envs, device=self.device)
        self.best_heading_progress = torch.zeros(self.num_envs, device=self.device)
        self.progress_goal_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_state = {}
        if cfg.spin_reward_mode not in ("omega_z", "heading"):
            raise ValueError(f"Unknown spin reward mode: {cfg.spin_reward_mode}")

        self._set_joint_pos_target = self.hand.set_joint_position_target_index
        self._write_pen_pose = self.pen.write_root_pose_to_sim_index
        self._write_pen_vel = self.pen.write_root_velocity_to_sim_index
        self._write_hand_joint_pos = self.hand.write_joint_position_to_sim_index
        self._write_hand_joint_vel = self.hand.write_joint_velocity_to_sim_index

    # -- scene ---------------------------------------------------------------

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.pen = RigidObject(self.cfg.pen_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["pen"] = self.pen
        self.camera = None
        if self.cfg.camera_cfg is not None:
            self.camera = Camera(self.cfg.camera_cfg)
            self.scene.sensors["camera"] = self.camera
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -- actions -------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.previous_pen_axis = quat_apply(self.pen.data.root_quat_w.torch, self.z_unit).clone()

    def _apply_action(self) -> None:
        idx = self.actuated_dof_indices
        target = scale(self.actions, self.dof_lower[:, idx], self.dof_upper[:, idx])
        target = (self.cfg.act_moving_average * target
                  + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, idx])
        self.cur_targets[:, idx] = saturate(target, self.dof_lower[:, idx], self.dof_upper[:, idx])
        # The first pair is the thumb; the other four share one finger curve.
        parents = self.cur_targets[:, self.passive_parent_indices]
        passive = torch.empty_like(parents)
        for column, coefficients in enumerate((THUMB_COUPLING,) + (FINGER_COUPLING,) * 4):
            offset, linear, quadratic = coefficients
            passive[:, column] = (
                offset + linear * parents[:, column]
                + quadratic * parents[:, column].square())
        self.cur_targets[:, self.passive_dof_indices] = saturate(
            passive, self.dof_lower[:, self.passive_dof_indices],
            self.dof_upper[:, self.passive_dof_indices])
        self.prev_targets[:] = self.cur_targets
        self._set_joint_pos_target(target=self.cur_targets)

    # -- observations / rewards / dones ---------------------------------------

    def _compute_intermediate_values(self):
        self.hand_dof_pos = self.hand.data.joint_pos.torch
        self.hand_dof_vel = self.hand.data.joint_vel.torch
        self.pen_pos = self.pen.data.root_pos_w.torch - self.scene.env_origins
        self.pen_rot = self.pen.data.root_quat_w.torch
        self.pen_linvel = self.pen.data.root_lin_vel_w.torch
        self.pen_angvel = self.pen.data.root_ang_vel_w.torch
        self.spin_rate = self.cfg.spin_sign * self.pen_angvel[:, 2]
        offset = self.pen_pos - self.pen_rest
        horizontal = torch.linalg.norm(offset[:, :2], dim=-1)
        self.on_palm = (horizontal < self.cfg.fall_dist) & (offset[:, 2] > -self.cfg.fall_height)

    def _get_observations(self) -> dict:
        parts = [
                unscale(self.hand_dof_pos, self.dof_lower, self.dof_upper),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.pen_pos - self.pen_rest,
                self.pen_rot,
                self.pen_linvel,
                self.cfg.vel_obs_scale * self.pen_angvel,
                self.actions,
        ]
        if self.cfg.progress_observation:
            parts.append(torch.clamp(
                self.cumulative_heading / self.cfg.progress_target_angle, -1.0, 2.0
            ).unsqueeze(-1))
        obs = torch.cat(parts, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        axis = quat_apply(self.pen_rot, self.z_unit)
        heading_rate = projected_spin_rate(self.previous_pen_axis, axis, self.step_dt, self.cfg.spin_sign)
        heading_delta = heading_rate * self.step_dt * self.on_palm.float()
        self.cumulative_heading, new_best, best_progress, reached_goal = update_best_progress(
            self.cumulative_heading, self.best_heading_progress, heading_delta,
            self.cfg.progress_target_angle)
        newly_reached_goal = (~self.progress_goal_achieved) & reached_goal
        # DirectRLEnv resets terminated environments before returning from step.
        # Capture values here so evaluation cannot mistake a reset for survival.
        if self.cfg.capture_step_state:
            self.step_state = {
                "spin_rate": self.spin_rate.clone(),
                "on_palm": self.on_palm.clone(),
                "pen_pos": self.pen_pos.clone(),
                "pen_axis": axis.clone(),
                "pen_angvel": self.pen_angvel.clone(),
            }
        objective_rate = self.spin_rate if self.cfg.spin_reward_mode == "omega_z" else heading_rate
        spin = torch.clamp(objective_rate, self.cfg.spin_clip[0], self.cfg.spin_clip[1]) / 10.0
        forward_spin = torch.clamp(objective_rate, 0.0, self.cfg.target_spin_rate) / self.cfg.target_spin_rate
        target_spin = torch.exp(
            -0.5 * ((objective_rate - self.cfg.target_spin_rate) / self.cfg.target_spin_sigma).square()
        ) * self.on_palm.float()
        horizontal = torch.linalg.norm(self.pen_pos[:, :2] - self.pen_rest[:, :2], dim=-1)
        terms = {
            "spin": self.cfg.spin_reward_scale * spin,
            "forward_spin": self.cfg.forward_spin_reward_scale * forward_spin,
            "target_spin": self.cfg.target_spin_reward_scale * target_spin,
            "best_progress": self.cfg.best_progress_reward_scale * best_progress,
            "progress_goal": self.cfg.progress_goal_bonus_scale * newly_reached_goal.float(),
            "alive": self.cfg.alive_bonus * self.on_palm.float(),
            "action": self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1),
            "action_rate": self.cfg.action_rate_scale * torch.sum((self.actions - self.prev_actions) ** 2, dim=-1),
            "fall": self.cfg.fall_penalty * (~self.on_palm).float(),
            "palm_center": self.cfg.palm_center_penalty_scale * (horizontal / self.cfg.fall_dist).square(),
        }
        if self.cfg.tilt_penalty_scale:
            terms["tilt"] = self.cfg.tilt_penalty_scale * axis[:, 2].square()
        reward = sum(terms.values())
        self.best_heading_progress[:] = new_best
        self.progress_goal_achieved |= newly_reached_goal
        self.prev_actions[:] = self.actions
        log = self.extras.setdefault("log", {})
        log["Metrics/spin_rate_rad_s"] = self.spin_rate.mean()
        log["Metrics/on_palm_fraction"] = self.on_palm.float().mean()
        log["Metrics/heading_rate_rad_s"] = heading_rate.mean()
        for name, value in terms.items():
            log[f"Reward/{name}"] = value.mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return ~self.on_palm, time_out

    # -- reset ---------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        # pen: rest pose, xy noise, random yaw about the palm normal
        pose = self.pen.data.default_root_pose.torch.clone()[env_ids]
        pose[:, 0:2] += self.cfg.reset_position_noise * sample_uniform(-1.0, 1.0, (n, 2), device=self.device)
        pose[:, 0:3] += self.scene.env_origins[env_ids]
        if self.cfg.eval_yaw_bins:
            ids = torch.as_tensor(env_ids, device=self.device)
            yaw = (self.cfg.eval_yaw_center - self.cfg.eval_yaw_range
                   + (ids % self.cfg.eval_yaw_bins + 0.5)
                   * (2 * self.cfg.eval_yaw_range / self.cfg.eval_yaw_bins))
            pose[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, self.z_unit[env_ids]), pose[:, 3:7])
        elif self.cfg.randomize_pen_yaw:
            yaw = sample_uniform(
                self.cfg.train_yaw_center - self.cfg.train_yaw_range,
                self.cfg.train_yaw_center + self.cfg.train_yaw_range,
                (n,), device=self.device)
            pose[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, self.z_unit[env_ids]), pose[:, 3:7])
        else:
            yaw = torch.full((n,), self.cfg.train_yaw_center, device=self.device)
            pose[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, self.z_unit[env_ids]), pose[:, 3:7])
        self.initial_yaw[env_ids] = yaw
        self.cumulative_heading[env_ids] = 0.0
        self.best_heading_progress[env_ids] = 0.0
        self.progress_goal_achieved[env_ids] = False
        vel = torch.zeros((n, 6), device=self.device)
        self._write_pen_pose(root_pose=pose, env_ids=env_ids)
        self._write_pen_vel(root_velocity=vel, env_ids=env_ids)
        # hand: open with a little noise
        default = self.hand.data.default_joint_pos.torch[env_ids]
        noise = sample_uniform(
            0.0, 1.0, (n, len(self.actuated_dof_indices)), device=self.device)
        dof_pos = default.clone()
        active = self.actuated_dof_indices
        dof_pos[:, active] += self.cfg.reset_dof_pos_noise * noise * (
            self.dof_upper[env_ids][:, active] - self.dof_lower[env_ids][:, active])
        parents = dof_pos[:, self.passive_parent_indices]
        for column, coefficients in enumerate((THUMB_COUPLING,) + (FINGER_COUPLING,) * 4):
            offset, linear, quadratic = coefficients
            dof_pos[:, self.passive_dof_indices[column]] = (
                offset + linear * parents[:, column]
                + quadratic * parents[:, column].square())
        dof_pos = saturate(dof_pos, self.dof_lower[env_ids], self.dof_upper[env_ids])
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.prev_actions[env_ids] = 0.0
        self._set_joint_pos_target(target=dof_pos, env_ids=env_ids)
        self._write_hand_joint_pos(position=dof_pos, env_ids=env_ids)
        self._write_hand_joint_vel(velocity=torch.zeros_like(dof_pos), env_ids=env_ids)
        self._compute_intermediate_values()


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
