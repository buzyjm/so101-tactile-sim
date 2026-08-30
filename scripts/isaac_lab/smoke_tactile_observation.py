"""Launch a small Isaac Lab scene and validate the tactile observation API."""

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
parser.add_argument("--steps", type=int, default=8)
parser.add_argument(
    "--contact",
    action="store_true",
    help="Drive the dynamic probe into the fixed fingertip during the test.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

from scene_config import (
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
)
from so101_isaac_lab.mdp import (
    tactile_debug_tensors,
    tactile_policy_features,
    tactile_taxel_forces,
)
from so101_isaac_lab.scene_cfg import TactileSmokeSceneCfg


class _ObservationEnv:
    def __init__(self, scene: InteractiveScene, num_envs: int, device: str) -> None:
        self.scene = scene
        self.num_envs = num_envs
        self.device = device


def _drive_probe_into_fixed_sensor(scene: InteractiveScene) -> None:
    sensor = scene["tactile_fixed"]
    probe = scene["probe"]
    apex = torch.tensor(
        TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
        dtype=torch.float32,
        device=args.device,
    ).repeat(args.num_envs, 1)
    # A 4 mm probe with 2 mm penetration, matching the standalone regression.
    apex[:, 2] += 0.002
    position_w = sensor.sensor_points_to_world(apex)
    quaternion_w = torch.zeros(
        (args.num_envs, 4), dtype=torch.float32, device=args.device
    )
    quaternion_w[:, 3] = 1.0
    probe.write_root_pose_to_sim_index(
        root_pose=torch.cat((position_w, quaternion_w), dim=1)
    )
    velocity_sensor = torch.zeros(
        (args.num_envs, 3), dtype=torch.float32, device=args.device
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
    print("[TACTILE_SMOKE] creating simulation", flush=True)
    sim_cfg = SimulationCfg(
        dt=1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
        device=args.device,
        render_interval=max(args.steps, 1),
    )
    sim = SimulationContext(sim_cfg)
    print("[TACTILE_SMOKE] creating scene", flush=True)
    scene = InteractiveScene(
        TactileSmokeSceneCfg(
            num_envs=args.num_envs,
            env_spacing=1.0,
        )
    )
    print("[TACTILE_SMOKE] resetting simulation", flush=True)
    sim.reset()
    print("[TACTILE_SMOKE] resetting scene", flush=True)
    scene.reset()

    print("[TACTILE_SMOKE] stepping", flush=True)
    for _ in range(args.steps):
        if args.contact:
            _drive_probe_into_fixed_sensor(scene)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_cfg.dt)

    print("[TACTILE_SMOKE] reading observations", flush=True)
    env = _ObservationEnv(scene, args.num_envs, args.device)
    fixed_cfg = SceneEntityCfg("tactile_fixed")
    moving_cfg = SceneEntityCfg("tactile_moving")
    taxel_forces = tactile_taxel_forces(env, fixed_cfg, moving_cfg)
    policy = tactile_policy_features(env, fixed_cfg, moving_cfg)
    debug = tactile_debug_tensors(env, fixed_cfg, moving_cfg)

    assert taxel_forces.shape == (args.num_envs, 2, 52, 3)
    assert policy.shape == (args.num_envs, 312)
    assert taxel_forces.device == torch.device(args.device)
    assert torch.isfinite(taxel_forces).all()
    assert torch.isfinite(policy).all()
    assert set(debug) == {
        "raw_taxel_forces",
        "raw_total_forces",
        "taxel_forces",
        "force_magnitudes",
        "total_forces",
        "taxel_total_forces",
        "contacts",
        "accepted_contact_counts",
        "rejected_contact_counts",
        "nearest_taxel_distances_m",
    }
    raw_conservation_error = (
        debug["raw_taxel_forces"].sum(dim=2)
        - debug["raw_total_forces"]
    ).abs().max()
    assert raw_conservation_error <= 1.0e-5
    active_taxels = debug["force_magnitudes"].gt(0.0).sum(dim=2)
    dominant_taxels = debug["force_magnitudes"].argmax(dim=2)
    per_finger_peaks = debug["force_magnitudes"].amax(dim=2)
    print(
        json.dumps(
            {
                "passed": True,
                "num_envs": args.num_envs,
                "physics_hz": TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
                "taxel_shape": list(taxel_forces.shape),
                "policy_shape": list(policy.shape),
                "device": str(taxel_forces.device),
                "max_force_n": float(taxel_forces.abs().max()),
                "per_finger_peak_magnitude_n": per_finger_peaks.tolist(),
                "dominant_taxel_zero_based": dominant_taxels.tolist(),
                "active_taxel_counts": active_taxels.tolist(),
                "contacts": debug["contacts"].tolist(),
                "accepted_contact_counts": debug[
                    "accepted_contact_counts"
                ].tolist(),
                "rejected_contact_counts": debug[
                    "rejected_contact_counts"
                ].tolist(),
                "raw_conservation_max_error_n": float(
                    raw_conservation_error
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close() may use fast shutdown, so print before closing.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
