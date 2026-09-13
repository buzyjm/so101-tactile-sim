"""Offline metrics, independent of the training reward and Isaac Sim.

Each row of a rollout is a pre-reset state. Only the first episode of each
environment is scored; a new pen after a drop cannot improve that trial.
"""
from __future__ import annotations

import math
import numpy as np


def summarize_rollout(spin, axes, positions, on_palm, initial_axis, initial_yaw,
                      home, dt, spin_sign, yaw_bins=0):
    spin, axes, positions = map(np.asarray, (spin, axes, positions))
    on_palm = np.asarray(on_palm, dtype=bool)
    steps, num_envs = spin.shape
    horizon = steps * dt
    # Include the terminal state for the drop count, exclude it from motion credit.
    valid = np.logical_and.accumulate(on_palm, axis=0)
    previous = np.concatenate([np.asarray(initial_axis)[None], axes[:-1]], axis=0)
    angle_delta = np.arctan2(previous[..., 0] * axes[..., 1] - previous[..., 1] * axes[..., 0],
                            (previous[..., :2] * axes[..., :2]).sum(axis=-1))
    # The projected direction is ill-conditioned when the pen is nearly vertical.
    projection_valid = (np.linalg.norm(previous[..., :2], axis=-1) > 0.2) & (
        np.linalg.norm(axes[..., :2], axis=-1) > 0.2)
    projected_rate = spin_sign * angle_delta / dt
    credit = valid & projection_valid
    projected_rate = np.where(credit, projected_rate, 0.0)
    tilt = np.degrees(np.arcsin(np.clip(np.abs(axes[..., 2]), 0, 1)))
    offset = positions - np.asarray(home)
    warmup = min(round(1.0 / dt), steps - 1)
    window = max(1, round(1.0 / dt))
    # Predeclared startup criterion: a full second averaging >= 3 rad/s,
    # with an observable projection and no drop throughout that second.
    cumulative = np.concatenate([np.zeros((1, num_envs)), np.cumsum(projected_rate, axis=0)])
    observed = np.concatenate([np.zeros((1, num_envs)), np.cumsum(credit, axis=0)])
    rolling = (cumulative[window:] - cumulative[:-window]) / window
    full_window = (observed[window:] - observed[:-window]) == window
    started = (rolling >= 3.0) & full_window
    records = []
    for i in range(num_envs):
        start_indices = np.flatnonzero(started[:, i])
        startup = float((start_indices[0] + window) * dt) if len(start_indices) else None
        drop_indices = np.flatnonzero(~on_palm[:, i])
        dropped = bool(len(drop_indices))
        mask = valid[:, i]
        rate = float(projected_rate[warmup:, i].mean())
        success = not dropped and startup is not None and startup <= 5.0 and rate >= 3.0
        records.append({
            "env": i, "yaw_bin": i % yaw_bins if yaw_bins else None,
            "initial_yaw_deg": float(np.degrees(initial_yaw[i])),
            "dropped": dropped,
            "first_drop_s": float((drop_indices[0] + 1) * dt) if dropped else None,
            "startup_s": startup, "success": success,
            "projected_rate_rad_s": rate,
            "projected_turns": float(projected_rate[:, i].sum() * dt / (2 * math.pi)),
            "omega_z_rate_rad_s": float(np.where(mask, spin[:, i], 0).mean()),
            "omega_z_turns": float(np.where(mask, spin[:, i], 0).sum() * dt / (2 * math.pi)),
            "tilt_deg": float(tilt[mask, i].mean()) if mask.any() else None,
            "height_above_home_mm": float(offset[mask, i, 2].mean() * 1000) if mask.any() else None,
            "max_xy_offset_mm": float(np.linalg.norm(offset[mask, i, :2], axis=-1).max() * 1000) if mask.any() else None,
            "projection_invalid_fraction": float((mask & ~projection_valid[:, i]).sum() / max(mask.sum(), 1)),
        })
    rates = [r["projected_rate_rad_s"] for r in records]
    projected_turns = [r["projected_turns"] for r in records]
    success_rate = float(np.mean([r["success"] for r in records]))
    summary = {
        "num_envs": num_envs, "seconds": horizon,
        "success_rate": success_rate,
        "drop_rate": float(np.mean([r["dropped"] for r in records])),
        "started_rate": float(np.mean([r["startup_s"] is not None for r in records])),
        "projected_rate_mean_rad_s": float(np.mean(rates)),
        "projected_rate_median_rad_s": float(np.median(rates)),
        "projected_turns_mean": float(np.mean(projected_turns)),
        "projected_turns_max": float(np.max(projected_turns)),
        "quarter_turn_rate": float(np.mean([turns >= 0.125 for turns in projected_turns])),
        "half_turn_rate": float(np.mean([turns >= 0.25 for turns in projected_turns])),
        "omega_z_rate_mean_rad_s": float(np.mean([r["omega_z_rate_rad_s"] for r in records])),
        "omega_z_rate_median_rad_s": float(np.median([r["omega_z_rate_rad_s"] for r in records])),
        "tilt_mean_deg": float(np.mean([r["tilt_deg"] for r in records if r["tilt_deg"] is not None])) if valid.any() else None,
    }
    bins = []
    for b in range(yaw_bins):
        members = [r for r in records if r["yaw_bin"] == b]
        if members:
            bins.append({"bin": b, "yaw_deg": members[0]["initial_yaw_deg"], "n": len(members),
                         "success_rate": float(np.mean([r["success"] for r in members])),
                         "drop_rate": float(np.mean([r["dropped"] for r in members])),
                         "projected_rate_rad_s": float(np.mean([r["projected_rate_rad_s"] for r in members]))})
    return {"schema_version": 1, "criteria": {
        "startup": "1 s window of signed projected rate >= 3 rad/s, by 5 s",
        "success": "startup criterion AND no drop AND mean projected rate after 1 s >= 3 rad/s",
        "drop": "first exit of the environment's fixed position envelope; no contact claim",
        "aggregation": "first episode only; zero motion credit after first drop; invalid projection earns zero",
    }, "summary": summary, "yaw_bins": bins, "trials": records}
