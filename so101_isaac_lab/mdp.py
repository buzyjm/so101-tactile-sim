"""Observation, reward and debug terms for the SO-101 tactile sensors."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from so101_isaac_lab.sensors import DPS2015TactileCore
from so101_isaac_lab.tactile_tensor import TactileTensorBatch


def _sensor(
    env: ManagerBasedEnv, cfg: SceneEntityCfg
) -> DPS2015TactileCore:
    sensor = env.scene[cfg.name]
    if not isinstance(sensor, DPS2015TactileCore):
        raise TypeError(
            f"Scene entity '{cfg.name}' must be a DPS2015 tactile sensor, got "
            f"{type(sensor).__name__}"
        )
    return sensor


def _paired_data(
    env: ManagerBasedEnv,
    fixed_sensor_cfg: SceneEntityCfg,
    moving_sensor_cfg: SceneEntityCfg,
) -> tuple[TactileTensorBatch, TactileTensorBatch]:
    return (
        _sensor(env, fixed_sensor_cfg).tactile_data,
        _sensor(env, moving_sensor_cfg).tactile_data,
    )


def tactile_taxel_forces(
    env: ManagerBasedEnv,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_fixed"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_moving"),
) -> torch.Tensor:
    """Return unnormalized device output with shape ``(N, 2, 52, 3)``."""
    fixed, moving = _paired_data(env, fixed_sensor_cfg, moving_sensor_cfg)
    return torch.cat((fixed.taxel_forces, moving.taxel_forces), dim=1)


def tactile_policy_features(
    env: ManagerBasedEnv,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_fixed"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_moving"),
) -> torch.Tensor:
    """Return normalized, flattened policy input with shape ``(N, 312)``."""
    fixed, moving = _paired_data(env, fixed_sensor_cfg, moving_sensor_cfg)
    return torch.cat((fixed.policy_features(), moving.policy_features()), dim=1)


def tactile_force_magnitudes(
    env: ManagerBasedEnv,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_fixed"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_moving"),
) -> torch.Tensor:
    """Return per-taxel force magnitudes with shape ``(N, 2, 52)``."""
    fixed, moving = _paired_data(env, fixed_sensor_cfg, moving_sensor_cfg)
    return torch.cat((fixed.force_magnitudes, moving.force_magnitudes), dim=1)


def tactile_total_forces(
    env: ManagerBasedEnv,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_fixed"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_moving"),
) -> torch.Tensor:
    """Return independent aggregate channels with shape ``(N, 2, 3)``."""
    fixed, moving = _paired_data(env, fixed_sensor_cfg, moving_sensor_cfg)
    return torch.cat((fixed.total_forces, moving.total_forces), dim=1)


def tactile_contact_reward(
    env: ManagerBasedRLEnv,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_fixed"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_moving"),
) -> torch.Tensor:
    """Reward simultaneous contact on both fingertips."""
    fixed, moving = _paired_data(env, fixed_sensor_cfg, moving_sensor_cfg)
    return (fixed.contacts[:, 0] & moving.contacts[:, 0]).to(torch.float32)


def tactile_balance_reward(
    env: ManagerBasedRLEnv,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_fixed"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_moving"),
    force_scale_n: float = 25.0,
) -> torch.Tensor:
    """Reward equal normal load, gated by bilateral contact."""
    if force_scale_n <= 0.0:
        raise ValueError("force_scale_n must be positive")
    fixed, moving = _paired_data(env, fixed_sensor_cfg, moving_sensor_cfg)
    difference = (fixed.total_forces[:, 0, 2] - moving.total_forces[:, 0, 2]).abs()
    bilateral = fixed.contacts[:, 0] & moving.contacts[:, 0]
    return torch.exp(-difference / force_scale_n) * bilateral


def tactile_overload_penalty(
    env: ManagerBasedRLEnv,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_fixed"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_moving"),
    safe_normal_force_n: float = 20.0,
) -> torch.Tensor:
    """Return a positive excess-load value for use with a negative weight."""
    if safe_normal_force_n < 0.0:
        raise ValueError("safe_normal_force_n must be non-negative")
    total = tactile_total_forces(env, fixed_sensor_cfg, moving_sensor_cfg)
    return torch.relu(total[..., 2] - safe_normal_force_n).sum(dim=1)


def tactile_debug_tensors(
    env: ManagerBasedEnv,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_fixed"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_moving"),
) -> dict[str, torch.Tensor]:
    """Return raw tensors for logging/HUD without adding them to policy input."""
    fixed, moving = _paired_data(env, fixed_sensor_cfg, moving_sensor_cfg)
    return {
        "raw_taxel_forces": torch.cat(
            (fixed.raw_taxel_forces, moving.raw_taxel_forces), dim=1
        ),
        "raw_total_forces": torch.cat(
            (fixed.raw_total_forces, moving.raw_total_forces), dim=1
        ),
        "taxel_forces": torch.cat(
            (fixed.taxel_forces, moving.taxel_forces), dim=1
        ),
        "force_magnitudes": torch.cat(
            (fixed.force_magnitudes, moving.force_magnitudes), dim=1
        ),
        "total_forces": torch.cat(
            (fixed.total_forces, moving.total_forces), dim=1
        ),
        "taxel_total_forces": torch.cat(
            (fixed.taxel_total_forces, moving.taxel_total_forces), dim=1
        ),
        "contacts": torch.cat((fixed.contacts, moving.contacts), dim=1),
        "accepted_contact_counts": torch.cat(
            (fixed.accepted_contact_counts, moving.accepted_contact_counts),
            dim=1,
        ),
        "rejected_contact_counts": torch.cat(
            (fixed.rejected_contact_counts, moving.rejected_contact_counts),
            dim=1,
        ),
        "nearest_taxel_distances_m": torch.cat(
            (
                fixed.nearest_taxel_distances_m,
                moving.nearest_taxel_distances_m,
            ),
            dim=1,
        ),
    }
