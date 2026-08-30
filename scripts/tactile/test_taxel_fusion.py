"""Fast, Isaac-independent regression for the 52-taxel spatial decoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tactile_fusion import TactileArrayMapper


def main() -> None:
    mapper = TactileArrayMapper()
    taxels = mapper.taxel_positions_m

    single_force = np.array([0.7, -0.4, 3.2], dtype=np.float64)
    single = mapper.map_contacts(taxels[[25]], single_force[None, :])
    np.testing.assert_allclose(
        single.total_force, single_force, rtol=1.0e-6, atol=1.0e-7
    )
    loaded = np.flatnonzero(
        np.linalg.norm(single.taxel_forces, axis=1) > 0.0
    )
    if not 1 <= loaded.size <= mapper.neighbor_count:
        raise AssertionError(f"Unexpected loaded-taxel count: {loaded.size}")
    dominant_taxel = int(
        np.argmax(np.linalg.norm(single.taxel_forces, axis=1))
    )
    if dominant_taxel != 25:
        raise AssertionError(
            f"Contact at taxel 26 mapped most strongly to {dominant_taxel + 1}"
        )

    multiple_positions = taxels[[0, 25, 51]]
    multiple_forces = np.array(
        [
            [0.3, 0.0, 1.0],
            [-0.2, 0.4, 2.5],
            [0.0, -0.1, 0.7],
        ],
        dtype=np.float64,
    )
    multiple = mapper.map_contacts(multiple_positions, multiple_forces)
    np.testing.assert_allclose(
        multiple.total_force,
        np.sum(multiple_forces, axis=0),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    far = mapper.map_contacts(
        np.array([[0.0, 0.0, -0.05]], dtype=np.float64),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        far.taxel_forces, np.zeros((52, 3), dtype=np.float32)
    )
    if far.accepted_contacts != 0 or far.rejected_contacts != 1:
        raise AssertionError("Far-contact rejection accounting is incorrect")

    result = {
        "passed": True,
        "taxel_count": int(taxels.shape[0]),
        "output_shape": [2, int(taxels.shape[0]), 3],
        "neighbor_count": mapper.neighbor_count,
        "kernel_sigma_m": mapper.kernel_sigma_m,
        "max_contact_distance_m": mapper.max_contact_distance_m,
        "single_contact_loaded_taxels": (loaded + 1).tolist(),
        "single_contact_dominant_taxel": dominant_taxel + 1,
        "single_contact_force_error_n": (
            single.total_force - single_force
        ).tolist(),
        "multiple_contact_force_error_n": (
            multiple.total_force - np.sum(multiple_forces, axis=0)
        ).tolist(),
        "far_contact_rejected": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
