"""End-to-end smoke test for the SO-101 ManagerBasedRLEnv wrapper."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument(
    "--contact",
    action="store_true",
    help="Drive a probe into the fixed fingertip before each policy step.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from scene_config import TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME
from so101_isaac_lab.env_cfg import SO101TactileEnvCfg
from so101_isaac_lab.mdp import tactile_debug_tensors


def _assert_observations(observations: dict, num_envs: int) -> None:
    assert observations["policy"].shape == (num_envs, 312)
    debug = observations["tactile_debug"]
    assert debug["taxel_forces"].shape == (num_envs, 2, 52, 3)
    assert debug["force_magnitudes"].shape == (num_envs, 2, 52)
    assert debug["total_forces"].shape == (num_envs, 2, 3)
    assert torch.isfinite(observations["policy"]).all()
    assert torch.isfinite(debug["taxel_forces"]).all()


def _drive_probe_into_fixed_sensor(env: ManagerBasedRLEnv) -> None:
    sensor = env.scene["tactile_fixed"]
    probe = env.scene["probe"]
    apex = torch.tensor(
        TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
        dtype=torch.float32,
        device=env.device,
    ).repeat(env.num_envs, 1)
    apex[:, 2] += 0.002
    position_w = sensor.sensor_points_to_world(apex)
    quaternion_w = torch.zeros(
        (env.num_envs, 4), dtype=torch.float32, device=env.device
    )
    quaternion_w[:, 3] = 1.0
    probe.write_root_pose_to_sim_index(
        root_pose=torch.cat((position_w, quaternion_w), dim=1)
    )
    velocity_sensor = torch.zeros(
        (env.num_envs, 3), dtype=torch.float32, device=env.device
    )
    velocity_sensor[:, 2] = -0.75
    velocity_w = sensor.sensor_vectors_to_world(velocity_sensor)
    probe.write_root_velocity_to_sim_index(
        root_velocity=torch.cat(
            (velocity_w, torch.zeros_like(velocity_w)), dim=1
        )
    )


def main() -> None:
    if args.num_envs < 1:
        raise ValueError("--num_envs must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    cfg = SO101TactileEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    # This synthetic fixture teleports a tiny dynamic probe on each env step.
    # For contact mode, make that cadence equal to the physics cadence so the
    # probe is not expelled during hidden decimation substeps.  The production
    # environment remains at its configured decimation of four.
    if args.contact:
        cfg.decimation = 1
        cfg.sim.render_interval = 1
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        observations, _ = env.reset()
        _assert_observations(observations, args.num_envs)

        actions = torch.zeros(
            (args.num_envs, env.action_manager.total_action_dim),
            dtype=torch.float32,
            device=env.device,
        )
        observed_max_taxel_force_n = 0.0
        observed_contacts = torch.zeros(
            (args.num_envs, 2), dtype=torch.bool, device=env.device
        )
        for _ in range(args.steps):
            if args.contact:
                _drive_probe_into_fixed_sensor(env)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            _assert_observations(observations, args.num_envs)
            assert rewards.shape == (args.num_envs,)
            assert terminated.shape == (args.num_envs,)
            assert truncated.shape == (args.num_envs,)
            assert torch.isfinite(rewards).all()
            step_debug = tactile_debug_tensors(env)
            observed_max_taxel_force_n = max(
                observed_max_taxel_force_n,
                float(step_debug["taxel_forces"].abs().max()),
            )
            observed_contacts |= step_debug["contacts"]

        debug = observations["tactile_debug"]
        sensor_debug = tactile_debug_tensors(env)
        max_taxel_force_n = float(debug["taxel_forces"].abs().max())
        if args.contact:
            assert observed_max_taxel_force_n > 0.0
        print(
            json.dumps(
                {
                    "passed": True,
                    "num_envs": args.num_envs,
                    "decimation": cfg.decimation,
                    "action_shape": list(actions.shape),
                    "policy_shape": list(observations["policy"].shape),
                    "taxel_shape": list(debug["taxel_forces"].shape),
                    "magnitude_shape": list(debug["force_magnitudes"].shape),
                    "total_force_shape": list(debug["total_forces"].shape),
                    "reward_shape": list(rewards.shape),
                    "reward": rewards.tolist(),
                    "final_max_taxel_force_n": max_taxel_force_n,
                    "observed_max_taxel_force_n": (
                        observed_max_taxel_force_n
                    ),
                    "final_contacts": sensor_debug["contacts"].tolist(),
                    "observed_contacts": observed_contacts.tolist(),
                    "device": str(observations["policy"].device),
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
