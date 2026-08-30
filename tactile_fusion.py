"""Pure-numpy spatial decoder for the DP-S2015-Elite 52-taxel output.

The real device exposes one fingertip module and a firmware-produced 52x3
distributed-force grid. Isaac follows the same architecture: one PhysX contact
sensor per fingertip, followed by this spatial decoder. This module contains no
Isaac imports so its force-conservation behaviour can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scene_config import (
    TACTILE_TAXEL_KERNEL_SIGMA_PLACEHOLDER_M,
    TACTILE_TAXEL_MAX_CONTACT_DISTANCE_M,
    TACTILE_TAXEL_NEIGHBOR_COUNT,
)
from tactile_taxels import TAXEL_COUNT, load_taxel_positions_m


@dataclass(frozen=True)
class TaxelMappingResult:
    """One spatially decoded contact frame in sensor-local coordinates."""

    taxel_forces: np.ndarray
    accepted_contacts: int
    rejected_contacts: int
    nearest_taxel_distances_m: np.ndarray

    @property
    def total_force(self) -> np.ndarray:
        return np.sum(self.taxel_forces, axis=0, dtype=np.float64).astype(
            np.float32
        )


class TactileArrayMapper:
    """Distribute point-contact vectors over the measured 52-taxel layout.

    A Gaussian kernel is evaluated only on the nearest taxels and normalized
    per contact. Consequently each accepted contact conserves its full vector
    force before the later hardware quantization/deadband stage.
    """

    def __init__(
        self,
        taxel_positions_m: np.ndarray | None = None,
        *,
        kernel_sigma_m: float = TACTILE_TAXEL_KERNEL_SIGMA_PLACEHOLDER_M,
        neighbor_count: int = TACTILE_TAXEL_NEIGHBOR_COUNT,
        max_contact_distance_m: float | None = (
            TACTILE_TAXEL_MAX_CONTACT_DISTANCE_M
        ),
    ) -> None:
        positions = (
            np.asarray(load_taxel_positions_m(), dtype=np.float64)
            if taxel_positions_m is None
            else np.asarray(taxel_positions_m, dtype=np.float64)
        )
        if positions.shape != (TAXEL_COUNT, 3):
            raise ValueError(
                f"Expected taxel positions shape ({TAXEL_COUNT}, 3), "
                f"got {positions.shape}"
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("Taxel positions must be finite")
        if kernel_sigma_m <= 0.0:
            raise ValueError("kernel_sigma_m must be positive")
        if not 1 <= neighbor_count <= TAXEL_COUNT:
            raise ValueError(
                f"neighbor_count must be in [1, {TAXEL_COUNT}], "
                f"got {neighbor_count}"
            )
        if max_contact_distance_m is not None and max_contact_distance_m <= 0.0:
            raise ValueError("max_contact_distance_m must be positive or None")

        self.taxel_positions_m = positions
        self.kernel_sigma_m = float(kernel_sigma_m)
        self.neighbor_count = int(neighbor_count)
        self.max_contact_distance_m = (
            None
            if max_contact_distance_m is None
            else float(max_contact_distance_m)
        )

    def weights_for_position(self, position_sensor_m: np.ndarray) -> np.ndarray:
        """Return 52 normalized weights, or all zeros outside the active pad."""
        position = np.asarray(position_sensor_m, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(
                f"Expected contact position shape (3,), got {position.shape}"
            )
        if not np.all(np.isfinite(position)):
            raise ValueError("Contact position must be finite")

        distances = np.linalg.norm(self.taxel_positions_m - position, axis=1)
        nearest_distance = float(np.min(distances))
        if (
            self.max_contact_distance_m is not None
            and nearest_distance > self.max_contact_distance_m
        ):
            return np.zeros(TAXEL_COUNT, dtype=np.float64)

        nearest = np.argpartition(
            distances, self.neighbor_count - 1
        )[: self.neighbor_count]
        local_distances = distances[nearest]
        local_weights = np.exp(
            -0.5 * np.square(local_distances / self.kernel_sigma_m)
        )
        weight_sum = float(np.sum(local_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            # Numerically safe fallback for a very small custom sigma.
            local_weights = np.zeros_like(local_weights)
            local_weights[int(np.argmin(local_distances))] = 1.0
        else:
            local_weights /= weight_sum

        weights = np.zeros(TAXEL_COUNT, dtype=np.float64)
        weights[nearest] = local_weights
        return weights

    def map_contacts(
        self,
        contact_positions_sensor_m: np.ndarray,
        contact_forces_sensor_n: np.ndarray,
    ) -> TaxelMappingResult:
        """Map N point contacts to a force-conserving (52, 3) tensor."""
        positions = np.asarray(contact_positions_sensor_m, dtype=np.float64)
        forces = np.asarray(contact_forces_sensor_n, dtype=np.float64)
        if positions.size == 0:
            positions = np.empty((0, 3), dtype=np.float64)
        if forces.size == 0:
            forces = np.empty((0, 3), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError(
                f"Expected contact positions shape (N, 3), got {positions.shape}"
            )
        if forces.shape != positions.shape:
            raise ValueError(
                "Contact forces must have the same (N, 3) shape as positions; "
                f"got {forces.shape} and {positions.shape}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(forces)):
            raise ValueError("Contact positions and forces must be finite")

        output = np.zeros((TAXEL_COUNT, 3), dtype=np.float64)
        accepted = 0
        rejected = 0
        nearest_distances = []
        for position, force in zip(positions, forces):
            distances = np.linalg.norm(self.taxel_positions_m - position, axis=1)
            nearest_distances.append(float(np.min(distances)))
            weights = self.weights_for_position(position)
            if not np.any(weights):
                rejected += 1
                continue
            output += weights[:, None] * force[None, :]
            accepted += 1

        return TaxelMappingResult(
            taxel_forces=output.astype(np.float32),
            accepted_contacts=accepted,
            rejected_contacts=rejected,
            nearest_taxel_distances_m=np.asarray(
                nearest_distances, dtype=np.float32
            ),
        )
