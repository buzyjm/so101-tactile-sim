"""Batched Torch implementation of the DP-S2015-Elite tactile decoder.

This module intentionally has no Isaac Lab or Isaac Sim imports.  PhysX-facing
code supplies contact points and force vectors in the native sensor frame; the
decoder keeps the complete calculation on the input tensor's device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from scene_config import (
    TACTILE_FORCE_RESOLUTION_N,
    TACTILE_MIN_FORCE_N,
    TACTILE_NORMAL_FORCE_RANGE_N,
    TACTILE_OUTPUT_FREQUENCY_HZ,
    TACTILE_SHEAR_FORCE_RANGE_N,
    TACTILE_TAXEL_KERNEL_SIGMA_PLACEHOLDER_M,
    TACTILE_TAXEL_MAX_CONTACT_DISTANCE_M,
    TACTILE_TAXEL_NEIGHBOR_COUNT,
)
from tactile_taxels import TAXEL_COUNT, load_taxel_positions_m


FINGERTIP_COUNT = 2


@dataclass(frozen=True)
class RawTactileTensorBatch:
    """Unquantized, force-conserving output for a batch of environments."""

    raw_total_forces: torch.Tensor
    raw_taxel_forces: torch.Tensor
    accepted_contact_counts: torch.Tensor
    rejected_contact_counts: torch.Tensor
    nearest_taxel_distances_m: torch.Tensor


@dataclass(frozen=True)
class TactileTensorBatch:
    """Hardware-shaped tactile output held on the Torch device."""

    raw_total_forces: torch.Tensor
    total_forces: torch.Tensor
    raw_taxel_forces: torch.Tensor
    taxel_forces: torch.Tensor
    taxel_total_forces: torch.Tensor
    force_magnitudes: torch.Tensor
    contacts: torch.Tensor
    accepted_contact_counts: torch.Tensor
    rejected_contact_counts: torch.Tensor
    nearest_taxel_distances_m: torch.Tensor

    @property
    def num_envs(self) -> int:
        return int(self.taxel_forces.shape[0])

    @property
    def num_fingertips(self) -> int:
        return int(self.taxel_forces.shape[1])

    def policy_features(self) -> torch.Tensor:
        """Return normalized taxel XYZ forces flattened per environment."""
        normalized = normalize_taxel_forces(self.taxel_forces)
        return normalized.reshape(self.num_envs, -1)


def apply_dp_s2015_output_model_torch(force_sensor: torch.Tensor) -> torch.Tensor:
    """Apply product range, deadband and 0.1 N/LSB without leaving Torch."""
    if force_sensor.ndim < 1 or force_sensor.shape[-1] != 3:
        raise ValueError(
            f"Expected force shape (..., 3), got {tuple(force_sensor.shape)}"
        )
    if not torch.is_floating_point(force_sensor):
        raise TypeError("force_sensor must be a floating-point tensor")

    output = force_sensor.clone()
    shear_min, shear_max = TACTILE_SHEAR_FORCE_RANGE_N
    normal_min, normal_max = TACTILE_NORMAL_FORCE_RANGE_N
    output[..., :2].clamp_(float(shear_min), float(shear_max))
    output[..., 2].clamp_(float(normal_min), float(normal_max))
    output = torch.where(
        output.abs() < float(TACTILE_MIN_FORCE_N),
        torch.zeros((), dtype=output.dtype, device=output.device),
        output,
    )
    output = torch.round(output / float(TACTILE_FORCE_RESOLUTION_N))
    return output * float(TACTILE_FORCE_RESOLUTION_N)


def normalize_taxel_forces(taxel_forces: torch.Tensor) -> torch.Tensor:
    """Scale shear and normal channels to the hardware range used by policy."""
    if taxel_forces.ndim < 1 or taxel_forces.shape[-1] != 3:
        raise ValueError(
            f"Expected taxel force shape (..., 3), got {tuple(taxel_forces.shape)}"
        )
    shear_scale = max(abs(value) for value in TACTILE_SHEAR_FORCE_RANGE_N)
    normal_scale = max(abs(value) for value in TACTILE_NORMAL_FORCE_RANGE_N)
    scale = taxel_forces.new_tensor((shear_scale, shear_scale, normal_scale))
    return (taxel_forces / scale).clamp(-1.0, 1.0)


def finalize_tactile_batch(raw: RawTactileTensorBatch) -> TactileTensorBatch:
    """Apply the firmware-like output model to an integrated raw frame."""
    total_forces = apply_dp_s2015_output_model_torch(raw.raw_total_forces)
    taxel_forces = apply_dp_s2015_output_model_torch(raw.raw_taxel_forces)
    taxel_total_forces = taxel_forces.sum(dim=2)
    force_magnitudes = torch.linalg.vector_norm(taxel_forces, dim=-1)
    contacts = taxel_forces.ne(0.0).any(dim=(-1, -2))
    return TactileTensorBatch(
        raw_total_forces=raw.raw_total_forces,
        total_forces=total_forces,
        raw_taxel_forces=raw.raw_taxel_forces,
        taxel_forces=taxel_forces,
        taxel_total_forces=taxel_total_forces,
        force_magnitudes=force_magnitudes,
        contacts=contacts,
        accepted_contact_counts=raw.accepted_contact_counts,
        rejected_contact_counts=raw.rejected_contact_counts,
        nearest_taxel_distances_m=raw.nearest_taxel_distances_m,
    )


class TactileTensorMapper:
    """Vectorized Gaussian spatial decoder for one or more environments."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        fingertip_count: int = FINGERTIP_COUNT,
        taxel_positions_m: Sequence[Sequence[float]] | torch.Tensor | None = None,
        kernel_sigma_m: float = TACTILE_TAXEL_KERNEL_SIGMA_PLACEHOLDER_M,
        neighbor_count: int = TACTILE_TAXEL_NEIGHBOR_COUNT,
        max_contact_distance_m: float | None = (
            TACTILE_TAXEL_MAX_CONTACT_DISTANCE_M
        ),
    ) -> None:
        if fingertip_count < 1:
            raise ValueError("fingertip_count must be positive")
        if kernel_sigma_m <= 0.0:
            raise ValueError("kernel_sigma_m must be positive")
        if not 1 <= neighbor_count <= TAXEL_COUNT:
            raise ValueError(
                f"neighbor_count must be in [1, {TAXEL_COUNT}], got "
                f"{neighbor_count}"
            )
        if max_contact_distance_m is not None and max_contact_distance_m <= 0.0:
            raise ValueError("max_contact_distance_m must be positive or None")

        positions = (
            load_taxel_positions_m()
            if taxel_positions_m is None
            else taxel_positions_m
        )
        self.taxel_positions_m = torch.as_tensor(
            positions, dtype=dtype, device=device
        )
        if self.taxel_positions_m.shape != (TAXEL_COUNT, 3):
            raise ValueError(
                f"Expected taxel positions shape ({TAXEL_COUNT}, 3), got "
                f"{tuple(self.taxel_positions_m.shape)}"
            )
        if not torch.isfinite(self.taxel_positions_m).all():
            raise ValueError("Taxel positions must be finite")

        self.device = self.taxel_positions_m.device
        self.dtype = self.taxel_positions_m.dtype
        self.fingertip_count = int(fingertip_count)
        self.kernel_sigma_m = float(kernel_sigma_m)
        self.neighbor_count = int(neighbor_count)
        self.max_contact_distance_m = (
            None
            if max_contact_distance_m is None
            else float(max_contact_distance_m)
        )

    def map_raw_contacts(
        self,
        contact_positions_sensor_m: torch.Tensor,
        contact_forces_sensor_n: torch.Tensor,
        contact_env_ids: torch.Tensor,
        contact_fingertip_ids: torch.Tensor,
        *,
        num_envs: int,
    ) -> RawTactileTensorBatch:
        """Map flat contacts to ``(num_envs, fingertips, 52, 3)``."""
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self._validate_contacts(
            contact_positions_sensor_m,
            contact_forces_sensor_n,
            contact_env_ids,
            contact_fingertip_ids,
            num_envs,
        )

        pair_count = num_envs * self.fingertip_count
        raw_taxels_flat = torch.zeros(
            (pair_count * TAXEL_COUNT, 3),
            dtype=self.dtype,
            device=self.device,
        )
        raw_totals_flat = torch.zeros(
            (pair_count, 3), dtype=self.dtype, device=self.device
        )
        accepted_counts = torch.zeros(
            pair_count, dtype=torch.int32, device=self.device
        )
        rejected_counts = torch.zeros_like(accepted_counts)
        nearest_by_pair = torch.full(
            (pair_count,), float("inf"), dtype=self.dtype, device=self.device
        )

        if contact_positions_sensor_m.shape[0] == 0:
            nearest_by_pair.fill_(float("nan"))
            return RawTactileTensorBatch(
                raw_total_forces=raw_totals_flat.view(
                    num_envs, self.fingertip_count, 3
                ),
                raw_taxel_forces=raw_taxels_flat.view(
                    num_envs, self.fingertip_count, TAXEL_COUNT, 3
                ),
                accepted_contact_counts=accepted_counts.view(
                    num_envs, self.fingertip_count
                ),
                rejected_contact_counts=rejected_counts.view(
                    num_envs, self.fingertip_count
                ),
                nearest_taxel_distances_m=nearest_by_pair.view(
                    num_envs, self.fingertip_count
                ),
            )

        positions = contact_positions_sensor_m.to(
            device=self.device, dtype=self.dtype
        )
        forces = contact_forces_sensor_n.to(device=self.device, dtype=self.dtype)
        env_ids = contact_env_ids.to(device=self.device, dtype=torch.long)
        fingertip_ids = contact_fingertip_ids.to(
            device=self.device, dtype=torch.long
        )
        pair_ids = env_ids * self.fingertip_count + fingertip_ids

        distance_sq = torch.sum(
            (positions[:, None, :] - self.taxel_positions_m[None, :, :]) ** 2,
            dim=-1,
        )
        neighbor_distance_sq, neighbor_ids = torch.topk(
            distance_sq,
            k=self.neighbor_count,
            dim=1,
            largest=False,
            sorted=True,
        )
        nearest_distance = torch.sqrt(neighbor_distance_sq[:, 0])
        nearest_by_pair.scatter_reduce_(
            0, pair_ids, nearest_distance, reduce="amin", include_self=True
        )

        if self.max_contact_distance_m is None:
            accepted = torch.ones_like(nearest_distance, dtype=torch.bool)
        else:
            accepted = nearest_distance <= self.max_contact_distance_m

        accepted_counts.add_(
            torch.bincount(pair_ids[accepted], minlength=pair_count).to(torch.int32)
        )
        rejected_counts.add_(
            torch.bincount(pair_ids[~accepted], minlength=pair_count).to(
                torch.int32
            )
        )

        if accepted.any():
            accepted_pair_ids = pair_ids[accepted]
            accepted_forces = forces[accepted]
            accepted_neighbor_ids = neighbor_ids[accepted]
            accepted_distance_sq = neighbor_distance_sq[accepted]
            weights = torch.exp(
                -0.5
                * accepted_distance_sq
                / (self.kernel_sigma_m * self.kernel_sigma_m)
            )
            weight_sum = weights.sum(dim=1, keepdim=True)
            underflow = (~torch.isfinite(weight_sum)) | (weight_sum <= 0.0)
            weights = weights / weight_sum.clamp_min(torch.finfo(self.dtype).tiny)
            if underflow.any():
                weights[underflow.squeeze(1)] = 0.0
                weights[underflow.squeeze(1), 0] = 1.0

            flat_taxel_ids = (
                accepted_pair_ids[:, None] * TAXEL_COUNT
                + accepted_neighbor_ids
            )
            contributions = weights[..., None] * accepted_forces[:, None, :]
            raw_taxels_flat.index_add_(
                0,
                flat_taxel_ids.reshape(-1),
                contributions.reshape(-1, 3),
            )
            raw_totals_flat.index_add_(
                0, accepted_pair_ids, accepted_forces
            )

        nearest_by_pair.masked_fill_(torch.isinf(nearest_by_pair), float("nan"))
        return RawTactileTensorBatch(
            raw_total_forces=raw_totals_flat.view(
                num_envs, self.fingertip_count, 3
            ),
            raw_taxel_forces=raw_taxels_flat.view(
                num_envs, self.fingertip_count, TAXEL_COUNT, 3
            ),
            accepted_contact_counts=accepted_counts.view(
                num_envs, self.fingertip_count
            ),
            rejected_contact_counts=rejected_counts.view(
                num_envs, self.fingertip_count
            ),
            nearest_taxel_distances_m=nearest_by_pair.view(
                num_envs, self.fingertip_count
            ),
        )

    def map_contacts(
        self,
        contact_positions_sensor_m: torch.Tensor,
        contact_forces_sensor_n: torch.Tensor,
        contact_env_ids: torch.Tensor,
        contact_fingertip_ids: torch.Tensor,
        *,
        num_envs: int,
    ) -> TactileTensorBatch:
        """Map and apply the product output model in one call."""
        return finalize_tactile_batch(
            self.map_raw_contacts(
                contact_positions_sensor_m,
                contact_forces_sensor_n,
                contact_env_ids,
                contact_fingertip_ids,
                num_envs=num_envs,
            )
        )

    def _validate_contacts(
        self,
        positions: torch.Tensor,
        forces: torch.Tensor,
        env_ids: torch.Tensor,
        fingertip_ids: torch.Tensor,
        num_envs: int,
    ) -> None:
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError(
                f"Expected contact positions shape (N, 3), got "
                f"{tuple(positions.shape)}"
            )
        if forces.shape != positions.shape:
            raise ValueError(
                "Contact forces must have the same (N, 3) shape as positions; "
                f"got {tuple(forces.shape)} and {tuple(positions.shape)}"
            )
        contact_count = positions.shape[0]
        if env_ids.shape != (contact_count,):
            raise ValueError(
                f"Expected contact_env_ids shape ({contact_count},), got "
                f"{tuple(env_ids.shape)}"
            )
        if fingertip_ids.shape != (contact_count,):
            raise ValueError(
                f"Expected contact_fingertip_ids shape ({contact_count},), got "
                f"{tuple(fingertip_ids.shape)}"
            )
        if not torch.is_floating_point(positions) or not torch.is_floating_point(
            forces
        ):
            raise TypeError("Contact positions and forces must be floating point")
        if not torch.isfinite(positions).all() or not torch.isfinite(forces).all():
            raise ValueError("Contact positions and forces must be finite")
        if contact_count:
            if int(env_ids.min()) < 0 or int(env_ids.max()) >= num_envs:
                raise ValueError("contact_env_ids contains an out-of-range index")
            if (
                int(fingertip_ids.min()) < 0
                or int(fingertip_ids.max()) >= self.fingertip_count
            ):
                raise ValueError(
                    "contact_fingertip_ids contains an out-of-range index"
                )


class TactileOutputClock:
    """Integrate 240 Hz raw frames and hold product-rate tactile samples."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        fingertip_count: int = FINGERTIP_COUNT,
        output_frequency_hz: float = TACTILE_OUTPUT_FREQUENCY_HZ,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if fingertip_count < 1:
            raise ValueError("fingertip_count must be positive")
        if output_frequency_hz <= 0.0:
            raise ValueError("output_frequency_hz must be positive")
        self.num_envs = int(num_envs)
        self.fingertip_count = int(fingertip_count)
        self.device = torch.device(device)
        self.dtype = dtype
        self.output_frequency_hz = float(output_frequency_hz)
        self._sample_period = 1.0 / self.output_frequency_hz
        self._time = 0.0
        self._next_sample_time = 0.0
        self._allocate()

    @property
    def latest(self) -> TactileTensorBatch:
        return self._latest

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        if env_ids is None:
            indices = slice(None)
            self._time = 0.0
            self._next_sample_time = 0.0
        else:
            indices = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
        self._duration[indices] = 0.0
        self._raw_total_sum[indices] = 0.0
        self._raw_taxel_sum[indices] = 0.0
        self._accepted_sum[indices] = 0
        self._rejected_sum[indices] = 0
        self._nearest_min[indices] = float("inf")
        self._zero_latest(indices)

    def update(
        self,
        raw: RawTactileTensorBatch,
        *,
        physics_dt: float,
        simulation_time: float | None = None,
    ) -> TactileTensorBatch | None:
        """Accumulate a physics frame and emit only on the device clock."""
        if physics_dt <= 0.0:
            raise ValueError("physics_dt must be positive")
        self._validate_raw(raw)
        self._duration += physics_dt
        self._raw_total_sum += raw.raw_total_forces * physics_dt
        self._raw_taxel_sum += raw.raw_taxel_forces * physics_dt
        self._accepted_sum += raw.accepted_contact_counts
        self._rejected_sum += raw.rejected_contact_counts
        finite = torch.isfinite(raw.nearest_taxel_distances_m)
        self._nearest_min = torch.where(
            finite,
            torch.minimum(self._nearest_min, raw.nearest_taxel_distances_m),
            self._nearest_min,
        )

        if simulation_time is None:
            self._time += physics_dt
        else:
            self._time = float(simulation_time)
        if self._time + 1.0e-12 < self._next_sample_time:
            return None

        duration = self._duration.clamp_min(torch.finfo(self.dtype).tiny)
        raw_average = RawTactileTensorBatch(
            raw_total_forces=self._raw_total_sum / duration[:, None, None],
            raw_taxel_forces=(
                self._raw_taxel_sum / duration[:, None, None, None]
            ),
            accepted_contact_counts=self._accepted_sum.clone(),
            rejected_contact_counts=self._rejected_sum.clone(),
            nearest_taxel_distances_m=torch.where(
                torch.isinf(self._nearest_min),
                torch.full_like(self._nearest_min, float("nan")),
                self._nearest_min,
            ),
        )
        self._latest = finalize_tactile_batch(raw_average)
        self._duration.zero_()
        self._raw_total_sum.zero_()
        self._raw_taxel_sum.zero_()
        self._accepted_sum.zero_()
        self._rejected_sum.zero_()
        self._nearest_min.fill_(float("inf"))
        while self._next_sample_time <= self._time + 1.0e-12:
            self._next_sample_time += self._sample_period
        return self._latest

    def _allocate(self) -> None:
        pair_shape = (self.num_envs, self.fingertip_count)
        self._duration = torch.zeros(
            self.num_envs, dtype=self.dtype, device=self.device
        )
        self._raw_total_sum = torch.zeros(
            (*pair_shape, 3), dtype=self.dtype, device=self.device
        )
        self._raw_taxel_sum = torch.zeros(
            (*pair_shape, TAXEL_COUNT, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self._accepted_sum = torch.zeros(
            pair_shape, dtype=torch.int32, device=self.device
        )
        self._rejected_sum = torch.zeros_like(self._accepted_sum)
        self._nearest_min = torch.full(
            pair_shape, float("inf"), dtype=self.dtype, device=self.device
        )
        zero_raw = RawTactileTensorBatch(
            raw_total_forces=torch.zeros_like(self._raw_total_sum),
            raw_taxel_forces=torch.zeros_like(self._raw_taxel_sum),
            accepted_contact_counts=torch.zeros_like(self._accepted_sum),
            rejected_contact_counts=torch.zeros_like(self._rejected_sum),
            nearest_taxel_distances_m=torch.full_like(
                self._nearest_min, float("nan")
            ),
        )
        self._latest = finalize_tactile_batch(zero_raw)

    def _zero_latest(self, indices: slice | torch.Tensor) -> None:
        for tensor in (
            self._latest.raw_total_forces,
            self._latest.total_forces,
            self._latest.raw_taxel_forces,
            self._latest.taxel_forces,
            self._latest.taxel_total_forces,
            self._latest.force_magnitudes,
            self._latest.accepted_contact_counts,
            self._latest.rejected_contact_counts,
        ):
            tensor[indices] = 0
        self._latest.contacts[indices] = False
        self._latest.nearest_taxel_distances_m[indices] = float("nan")

    def _validate_raw(self, raw: RawTactileTensorBatch) -> None:
        pair_shape = (self.num_envs, self.fingertip_count)
        expected = {
            "raw_total_forces": (*pair_shape, 3),
            "raw_taxel_forces": (*pair_shape, TAXEL_COUNT, 3),
            "accepted_contact_counts": pair_shape,
            "rejected_contact_counts": pair_shape,
            "nearest_taxel_distances_m": pair_shape,
        }
        for field, shape in expected.items():
            value = getattr(raw, field)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Expected {field} shape {shape}, got {tuple(value.shape)}"
                )
            if value.device != self.device:
                raise ValueError(
                    f"Expected {field} on {self.device}, got {value.device}"
                )
