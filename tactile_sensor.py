"""DP-S2015-Elite aggregate and 52-taxel output for Isaac Sim 6.0.

Each physical fingertip uses one Isaac ContactSensor. PhysX contact positions
and impulses are transformed into the native sensor frame, then a firmware-like
spatial decoder produces the same (2, 52, 3) layout as the real device while
retaining the existing aggregate fields.

Import this module only after creating SimulationApp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from isaacsim.sensors.experimental.physics import ContactSensor
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom

from scene_config import (
    TACTILE_BODY_PRIM_PATHS,
    TACTILE_FORCE_RESOLUTION_N,
    TACTILE_MIN_FORCE_N,
    TACTILE_NORMAL_FORCE_FROM_PHYSICAL_SIGN,
    TACTILE_NORMAL_FORCE_RANGE_N,
    TACTILE_OUTPUT_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS,
    TACTILE_SENSOR_PRIM_PATHS,
    TACTILE_SHEAR_FORCE_RANGE_N,
)
from tactile_fusion import TactileArrayMapper
from tactile_taxels import TAXEL_COUNT


SENSOR_NAMES = ("fixed", "moving")


@dataclass(frozen=True)
class TactileSample:
    """One sample held at the simulated hardware output rate."""

    timestamp: float
    physics_step: int
    raw_total_forces: np.ndarray
    total_forces: np.ndarray
    raw_taxel_forces: np.ndarray
    taxel_forces: np.ndarray
    taxel_total_forces: np.ndarray
    contacts: np.ndarray
    number_of_contacts: np.ndarray
    rejected_contact_counts: np.ndarray
    nearest_taxel_distances_m: np.ndarray
    backend_force_magnitudes: np.ndarray

    @property
    def taxel_force_magnitudes(self) -> np.ndarray:
        return np.linalg.norm(self.taxel_forces, axis=-1).astype(np.float32)

    def to_dict(self, include_taxels: bool = True) -> dict[str, object]:
        nearest = [
            None if np.isnan(value) else float(value)
            for value in self.nearest_taxel_distances_m
        ]
        result: dict[str, object] = {
            "timestamp": self.timestamp,
            "physics_step": self.physics_step,
            "sensor_names": list(SENSOR_NAMES),
            "raw_total_forces": self.raw_total_forces.tolist(),
            "total_forces": self.total_forces.tolist(),
            "taxel_total_forces": self.taxel_total_forces.tolist(),
            "contacts": self.contacts.tolist(),
            "number_of_contacts": self.number_of_contacts.tolist(),
            "rejected_contact_counts": self.rejected_contact_counts.tolist(),
            "nearest_taxel_distances_m": nearest,
            "backend_force_magnitudes": self.backend_force_magnitudes.tolist(),
        }
        if include_taxels:
            result.update(
                {
                    "raw_taxel_forces": self.raw_taxel_forces.tolist(),
                    "taxel_forces": self.taxel_forces.tolist(),
                    "taxel_force_magnitudes": (
                        self.taxel_force_magnitudes.tolist()
                    ),
                }
            )
        return result


def apply_dp_s2015_output_model(force_sensor: np.ndarray) -> np.ndarray:
    """Apply range, deadband and 0.1 N/LSB to one or more XYZ vectors."""
    force = np.asarray(force_sensor, dtype=np.float64).copy()
    if force.ndim < 1 or force.shape[-1] != 3:
        raise ValueError(f"Expected force shape (..., 3), got {force.shape}")

    shear_min, shear_max = TACTILE_SHEAR_FORCE_RANGE_N
    normal_min, normal_max = TACTILE_NORMAL_FORCE_RANGE_N
    force[..., 0:2] = np.clip(force[..., 0:2], shear_min, shear_max)
    force[..., 2] = np.clip(force[..., 2], normal_min, normal_max)
    force[np.abs(force) < TACTILE_MIN_FORCE_N] = 0.0
    force = np.round(force / TACTILE_FORCE_RESOLUTION_N)
    force *= TACTILE_FORCE_RESOLUTION_N
    return force.astype(np.float32)


class DPS2015EliteMvp:
    """Decode two Isaac contact sensors into aggregate and 52-taxel forces."""

    def __init__(
        self,
        stage: Usd.Stage,
        contact_partner_paths: tuple[str, str] | None = None,
    ) -> None:
        self._stage = stage
        self._sensors = [ContactSensor(path) for path in TACTILE_SENSOR_PRIM_PATHS]
        self._mapper = TactileArrayMapper()
        self._contact_partner_paths = (
            contact_partner_paths if contact_partner_paths is not None else (None, None)
        )
        self._sample_period = 1.0 / TACTILE_OUTPUT_FREQUENCY_HZ
        self._next_sample_time = 0.0
        self._last_update_time = 0.0
        self.latest: TactileSample | None = None
        self._reset_accumulators()

    @property
    def output_frequency_hz(self) -> float:
        return TACTILE_OUTPUT_FREQUENCY_HZ

    def reset(self, start_time: float = 0.0) -> None:
        self._next_sample_time = float(start_time)
        self._last_update_time = float(start_time)
        self.latest = None
        self._reset_accumulators()

    def close(self) -> None:
        for sensor in self._sensors:
            sensor.reset()

    def maybe_sample(
        self,
        simulation_time: float,
        physics_step: int,
    ) -> TactileSample | None:
        """Accumulate every physics step and emit on the device clock.

        Call this once after every physics step. PhysX contact forces are
        integrated over all substeps in the output window; callers reuse
        latest between 83.3 Hz device samples.
        """
        simulation_time = float(simulation_time)
        frame = self._read(simulation_time, int(physics_step))
        step_duration = max(
            0.0, simulation_time - self._last_update_time
        )
        self._last_update_time = simulation_time
        self._accumulate_frame(frame, step_duration)

        if simulation_time + 1.0e-12 < self._next_sample_time:
            return None

        sample = self._finalize_accumulated_sample(
            simulation_time, int(physics_step), frame
        )
        while self._next_sample_time <= simulation_time + 1.0e-12:
            self._next_sample_time += self._sample_period
        self.latest = sample
        self._reset_accumulators()
        return sample

    def _reset_accumulators(self) -> None:
        self._accumulated_duration = 0.0
        self._accumulated_raw_forces = np.zeros(
            (2, 3), dtype=np.float64
        )
        self._accumulated_raw_taxel_forces = np.zeros(
            (2, TAXEL_COUNT, 3), dtype=np.float64
        )
        self._accumulated_contact_counts = np.zeros(2, dtype=np.int32)
        self._accumulated_rejected_counts = np.zeros(2, dtype=np.int32)
        self._accumulated_nearest_distances = np.full(
            2, np.nan, dtype=np.float32
        )
        self._latest_backend_magnitudes = np.zeros(2, dtype=np.float32)

    def _accumulate_frame(
        self,
        frame: TactileSample,
        step_duration: float,
    ) -> None:
        if step_duration > 0.0:
            self._accumulated_duration += step_duration
            self._accumulated_raw_forces += (
                frame.raw_total_forces * step_duration
            )
            self._accumulated_raw_taxel_forces += (
                frame.raw_taxel_forces * step_duration
            )
        self._accumulated_contact_counts += frame.number_of_contacts
        self._accumulated_rejected_counts += frame.rejected_contact_counts
        finite = np.isfinite(frame.nearest_taxel_distances_m)
        for index in np.flatnonzero(finite):
            previous = self._accumulated_nearest_distances[index]
            current = frame.nearest_taxel_distances_m[index]
            if not np.isfinite(previous) or current < previous:
                self._accumulated_nearest_distances[index] = current
        self._latest_backend_magnitudes[:] = frame.backend_force_magnitudes

    def _finalize_accumulated_sample(
        self,
        timestamp: float,
        physics_step: int,
        fallback: TactileSample,
    ) -> TactileSample:
        if self._accumulated_duration > 0.0:
            raw_forces = (
                self._accumulated_raw_forces
                / self._accumulated_duration
            ).astype(np.float32)
            raw_taxel_forces = (
                self._accumulated_raw_taxel_forces
                / self._accumulated_duration
            ).astype(np.float32)
        else:
            raw_forces = fallback.raw_total_forces.copy()
            raw_taxel_forces = fallback.raw_taxel_forces.copy()

        total_forces = apply_dp_s2015_output_model(raw_forces)
        taxel_forces = apply_dp_s2015_output_model(raw_taxel_forces)
        taxel_total_forces = np.sum(
            taxel_forces, axis=1, dtype=np.float64
        ).astype(np.float32)
        contacts = np.any(taxel_forces != 0.0, axis=(1, 2))
        return TactileSample(
            timestamp=timestamp,
            physics_step=physics_step,
            raw_total_forces=raw_forces,
            total_forces=total_forces,
            raw_taxel_forces=raw_taxel_forces,
            taxel_forces=taxel_forces,
            taxel_total_forces=taxel_total_forces,
            contacts=contacts,
            number_of_contacts=self._accumulated_contact_counts.copy(),
            rejected_contact_counts=(
                self._accumulated_rejected_counts.copy()
            ),
            nearest_taxel_distances_m=(
                self._accumulated_nearest_distances.copy()
            ),
            backend_force_magnitudes=(
                self._latest_backend_magnitudes.copy()
            ),
        )

    def _read(self, timestamp: float, physics_step: int) -> TactileSample:
        raw_forces = np.zeros((2, 3), dtype=np.float32)
        device_forces = np.zeros((2, 3), dtype=np.float32)
        raw_taxel_forces = np.zeros(
            (2, TAXEL_COUNT, 3), dtype=np.float32
        )
        taxel_forces = np.zeros((2, TAXEL_COUNT, 3), dtype=np.float32)
        taxel_total_forces = np.zeros((2, 3), dtype=np.float32)
        contacts = np.zeros(2, dtype=bool)
        counts = np.zeros(2, dtype=np.int32)
        rejected_counts = np.zeros(2, dtype=np.int32)
        nearest_distances = np.full(2, np.nan, dtype=np.float32)
        backend_magnitudes = np.zeros(2, dtype=np.float32)

        for index, sensor in enumerate(self._sensors):
            reading = sensor.get_sensor_reading()
            # Isaac Sim 6.0 can retain the previous raw-contact buffer for one
            # or more frames after contact ends. The scalar reading carries
            # the authoritative current-frame contact flag.
            raw_contacts = (
                sensor.get_raw_data()
                if reading.is_valid and reading.in_contact
                else []
            )
            positions_sensor, forces_sensor = (
                self._contact_samples_on_sensor_body(
                    raw_contacts,
                    TACTILE_BODY_PRIM_PATHS[index],
                    TACTILE_ROOT_PRIM_PATHS[index],
                    self._contact_partner_paths[index],
                )
            )
            mapping = self._mapper.map_contacts(
                positions_sensor, forces_sensor
            )
            raw_taxel_forces[index] = mapping.taxel_forces
            raw_forces[index] = mapping.total_force
            device_forces[index] = apply_dp_s2015_output_model(
                raw_forces[index]
            )
            taxel_forces[index] = apply_dp_s2015_output_model(
                raw_taxel_forces[index]
            )
            taxel_total_forces[index] = np.sum(
                taxel_forces[index], axis=0, dtype=np.float64
            )
            contacts[index] = bool(np.any(taxel_forces[index] != 0.0))
            counts[index] = mapping.accepted_contacts
            rejected_counts[index] = mapping.rejected_contacts
            if mapping.nearest_taxel_distances_m.size:
                nearest_distances[index] = float(
                    np.min(mapping.nearest_taxel_distances_m)
                )
            if reading.is_valid:
                backend_magnitudes[index] = float(reading.value)

        return TactileSample(
            timestamp=timestamp,
            physics_step=physics_step,
            raw_total_forces=raw_forces,
            total_forces=device_forces,
            raw_taxel_forces=raw_taxel_forces,
            taxel_forces=taxel_forces,
            taxel_total_forces=taxel_total_forces,
            contacts=contacts,
            number_of_contacts=counts,
            rejected_contact_counts=rejected_counts,
            nearest_taxel_distances_m=nearest_distances,
            backend_force_magnitudes=backend_magnitudes,
        )

    def _contact_samples_on_sensor_body(
        self,
        raw_contacts: list[dict[str, object]],
        sensor_body_path: str,
        sensor_root_path: str,
        contact_partner_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return accepted-candidate positions and forces in device axes."""
        positions = []
        forces = []
        for contact in raw_contacts:
            body0 = str(PhysicsSchemaTools.intToSdfPath(int(contact["body0"])))
            body1 = str(PhysicsSchemaTools.intToSdfPath(int(contact["body1"])))
            if sensor_body_path not in (body0, body1):
                continue
            partner_path = body1 if body0 == sensor_body_path else body0
            if (
                contact_partner_path is not None
                and partner_path != contact_partner_path
            ):
                continue

            dt = float(contact["dt"])
            if dt <= 0.0:
                continue
            impulse_world = self._raw_xyz(contact["impulse"])
            sign = 1.0 if body0 == sensor_body_path else -1.0
            force_world = sign * impulse_world / dt
            force_sensor = self._world_vector_to_sensor(
                force_world, sensor_root_path
            ).astype(np.float64)
            force_sensor[2] *= TACTILE_NORMAL_FORCE_FROM_PHYSICAL_SIGN

            position_world = self._raw_xyz(contact["position"])
            position_sensor = self._world_point_to_sensor(
                position_world, sensor_root_path
            )
            positions.append(position_sensor)
            forces.append(force_sensor)

        return (
            np.asarray(positions, dtype=np.float64).reshape((-1, 3)),
            np.asarray(forces, dtype=np.float64).reshape((-1, 3)),
        )

    @staticmethod
    def _raw_xyz(value: object) -> np.ndarray:
        """Convert a raw-contact XYZ dictionary to a NumPy vector."""
        return np.array(
            [value["x"], value["y"], value["z"]],
            dtype=np.float64,
        )

    @staticmethod
    def _sum_force_on_sensor_body(
        raw_contacts: list[dict[str, object]],
        sensor_body_path: str,
        contact_partner_path: str | None = None,
    ) -> np.ndarray:
        force_world = np.zeros(3, dtype=np.float64)
        for contact in raw_contacts:
            body0 = str(PhysicsSchemaTools.intToSdfPath(int(contact["body0"])))
            body1 = str(PhysicsSchemaTools.intToSdfPath(int(contact["body1"])))
            if sensor_body_path not in (body0, body1):
                continue
            partner_path = body1 if body0 == sensor_body_path else body0
            if (
                contact_partner_path is not None
                and partner_path != contact_partner_path
            ):
                continue

            impulse_data = contact["impulse"]
            impulse = np.array(
                [impulse_data["x"], impulse_data["y"], impulse_data["z"]],
                dtype=np.float64,
            )
            dt = float(contact["dt"])
            if dt <= 0.0:
                continue

            # PhysX reports the impulse applied to body0. The equal and
            # opposite impulse acts on body1.
            sign = 1.0 if body0 == sensor_body_path else -1.0
            force_world += sign * impulse / dt
        return force_world

    def _world_point_to_sensor(
        self,
        point_world: np.ndarray,
        sensor_root_path: str,
    ) -> np.ndarray:
        prim = self._stage.GetPrimAtPath(sensor_root_path)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        local = transform.GetInverse().Transform(
            Gf.Vec3d(*np.asarray(point_world, dtype=np.float64))
        )
        return np.array(local, dtype=np.float64)

    def _world_vector_to_sensor(
        self,
        vector_world: np.ndarray,
        sensor_root_path: str,
    ) -> np.ndarray:
        prim = self._stage.GetPrimAtPath(sensor_root_path)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        local = transform.GetInverse().TransformDir(
            Gf.Vec3d(*np.asarray(vector_world, dtype=np.float64))
        )
        return np.array(local, dtype=np.float32)
