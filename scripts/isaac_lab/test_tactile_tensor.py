"""Isaac-independent regression for the batched Torch tactile interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_config import TACTILE_MVP_PHYSICS_FREQUENCY_HZ
from so101_isaac_lab.tactile_tensor import (
    TactileOutputClock,
    TactileTensorMapper,
)


def run_device(device: torch.device) -> dict[str, object]:
    mapper = TactileTensorMapper(device=device)
    taxels = mapper.taxel_positions_m

    positions = torch.stack(
        (
            taxels[25],
            taxels[0],
            taxels[51],
            taxels[12],
            taxels[25].new_tensor((0.0, 0.0, -0.05)),
        )
    )
    forces = torch.tensor(
        (
            (0.7, -0.4, 3.2),
            (0.3, 0.0, 1.0),
            (-0.2, 0.4, 2.5),
            (0.0, -0.1, 0.7),
            (1.0, 2.0, 3.0),
        ),
        dtype=torch.float32,
        device=device,
    )
    env_ids = torch.tensor((0, 0, 2, 2, 1), device=device)
    fingertip_ids = torch.tensor((0, 1, 0, 1, 1), device=device)
    batch = mapper.map_contacts(
        positions,
        forces,
        env_ids,
        fingertip_ids,
        num_envs=3,
    )

    assert batch.taxel_forces.shape == (3, 2, 52, 3)
    assert batch.force_magnitudes.shape == (3, 2, 52)
    assert batch.total_forces.shape == (3, 2, 3)
    assert batch.policy_features().shape == (3, 312)
    assert batch.taxel_forces.device == device
    assert batch.policy_features().abs().max() <= 1.0

    expected_raw_totals = torch.zeros((3, 2, 3), device=device)
    expected_raw_totals[0, 0] = forces[0]
    expected_raw_totals[0, 1] = forces[1]
    expected_raw_totals[2, 0] = forces[2]
    expected_raw_totals[2, 1] = forces[3]
    torch.testing.assert_close(
        batch.raw_total_forces, expected_raw_totals, rtol=1.0e-6, atol=1.0e-6
    )
    torch.testing.assert_close(
        batch.raw_taxel_forces.sum(dim=2),
        expected_raw_totals,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert int(batch.rejected_contact_counts[1, 1]) == 1
    assert int(batch.accepted_contact_counts.sum()) == 4
    assert int(batch.force_magnitudes[0, 0].argmax()) == 25
    assert int(batch.force_magnitudes[0, 1].argmax()) == 0
    assert int(batch.force_magnitudes[2, 0].argmax()) == 51
    assert int(batch.force_magnitudes[2, 1].argmax()) == 12

    clock = TactileOutputClock(num_envs=3, device=device)
    raw = mapper.map_raw_contacts(
        positions,
        forces,
        env_ids,
        fingertip_ids,
        num_envs=3,
    )
    physics_dt = 1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ
    emissions = 0
    held_frames = 0
    for step in range(int(TACTILE_MVP_PHYSICS_FREQUENCY_HZ)):
        emitted = clock.update(
            raw, physics_dt=physics_dt, simulation_time=(step + 1) * physics_dt
        )
        if emitted is None:
            held_frames += 1
        else:
            emissions += 1
    assert emissions in (83, 84), emissions
    assert held_frames > 0
    torch.testing.assert_close(
        clock.latest.raw_total_forces,
        expected_raw_totals,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    clock.reset(torch.tensor((2,), device=device))
    assert not clock.latest.taxel_forces[2].any()
    assert clock.latest.taxel_forces[0].any()

    return {
        "device": str(device),
        "output_shape": list(batch.taxel_forces.shape),
        "policy_shape": list(batch.policy_features().shape),
        "accepted_contacts": int(batch.accepted_contact_counts.sum()),
        "rejected_contacts": int(batch.rejected_contact_counts.sum()),
        "emissions_per_simulated_second": emissions,
        "force_conservation_max_error_n": float(
            (batch.raw_taxel_forces.sum(dim=2) - expected_raw_totals)
            .abs()
            .max()
        ),
    }


def main() -> None:
    results = [run_device(torch.device("cpu"))]
    if torch.cuda.is_available():
        results.append(run_device(torch.device("cuda:0")))
    print(json.dumps({"passed": True, "devices": results}, indent=2))


if __name__ == "__main__":
    main()
