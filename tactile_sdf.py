"""Per-taxel tactile readout from analytic sphere geometry.

The contact-report path asks PhysX where the bodies touch and spreads the two
or three points it returns across the array with a Gaussian.  That is a point
contact dressed up as a patch: it lights 4-6 taxels with a sharp peak, while
the recorded hardware lights 16-23 at a near-uniform 0.78 of peak.  The count
does not move when the contact stiffness is changed over three decades,
because the number of contact points is set by the collision algorithm, not by
how hard or soft the contact is.

So the readout is computed from geometry instead.  Each taxel independently
measures how far the ball has pressed past it, which is what produces a
distributed load.  The sensing points sit one soft-layer thickness proud of the
collision surface, standing in for the plush ball and the compliant pad face
that PhysX does not simulate: the ball rests on the collider while the sensing
plane is already inside it.  Sphere geometry is closed form, so no SDF mesh,
no PhysX SDF view, and no GPU dynamics are involved.

Dynamics are untouched.  PhysX still holds the ball with real contact forces;
only the number the sensor reports changes.  That decoupling is deliberate --
the observation model can then be calibrated and randomised on its own -- but
it does mean the reported force and the force actually holding the ball are
two different quantities, so :meth:`consistency` reports their ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pxr import Gf, UsdGeom

from scene_config import (
    TACTILE_FORCE_RESOLUTION_N,
    TACTILE_MIN_FORCE_N,
    TACTILE_NORMAL_FORCE_RANGE_N,
    TACTILE_OUTPUT_FREQUENCY_HZ,
    TACTILE_SHEAR_FORCE_RANGE_N,
)
from tactile_taxels import TAXEL_COUNT


@dataclass(frozen=True)
class SdfTactileParameters:
    """The three quantities that set what the array reports.

    Each is identified by a different hardware statistic, so the fit is well
    posed rather than one knob traded against another:

    * ``layer_thickness_m`` fixes how many taxels load  -> active-taxel count;
    * ``stiffness_n_per_m`` fixes the magnitude         -> total force;
    * ``depth_exponent`` fixes how flat the patch is    -> uniformity.
    """

    layer_thickness_m: float = 0.0009
    stiffness_n_per_m: float = 900.0
    depth_exponent: float = 0.6
    shear_coefficient: float = 0.35


@dataclass
class SdfTactileSample:
    timestamp: float
    physics_step: int
    taxel_forces: np.ndarray        # (2, 52, 3) after the device model
    raw_taxel_forces: np.ndarray    # (2, 52, 3) before clipping/quantising
    total_forces: np.ndarray        # (2, 3)
    contacts: np.ndarray            # (2,) bool
    penetration_m: np.ndarray       # (2, 52)

    @property
    def active_taxels(self) -> np.ndarray:
        return (np.linalg.norm(self.taxel_forces, axis=-1) > 0.0).sum(axis=1)


def _apply_device_model(forces: np.ndarray) -> np.ndarray:
    """Range, deadband and 0.1 N/LSB, matching the recorded device output."""
    out = np.asarray(forces, dtype=np.float64).copy()
    shear_min, shear_max = TACTILE_SHEAR_FORCE_RANGE_N
    normal_min, normal_max = TACTILE_NORMAL_FORCE_RANGE_N
    out[..., 0:2] = np.clip(out[..., 0:2], shear_min, shear_max)
    out[..., 2] = np.clip(out[..., 2], normal_min, normal_max)
    out[np.abs(out) < TACTILE_MIN_FORCE_N] = 0.0
    out = np.round(out / TACTILE_FORCE_RESOLUTION_N) * TACTILE_FORCE_RESOLUTION_N
    return out.astype(np.float32)


class AnalyticSphereTactile:
    """Read both pads against one spherical object."""

    def __init__(self, stage, ball_prim_path: str, ball_radius_m: float,
                 pad_prim_paths, taxel_positions_m: np.ndarray,
                 parameters: SdfTactileParameters | None = None) -> None:
        self._stage = stage
        self._ball_path = ball_prim_path
        self._radius = float(ball_radius_m)
        self._pad_paths = tuple(pad_prim_paths)
        self._taxels = np.asarray(taxel_positions_m, dtype=float)
        if self._taxels.shape != (TAXEL_COUNT, 3):
            raise ValueError(f"expected ({TAXEL_COUNT}, 3) taxels, "
                             f"got {self._taxels.shape}")
        self.parameters = parameters or SdfTactileParameters()
        self._sample_period = 1.0 / TACTILE_OUTPUT_FREQUENCY_HZ
        self._next_sample_time = 0.0
        self._previous_ball_position = None
        self._previous_time = None
        self.latest: SdfTactileSample | None = None

    def reset(self, start_time: float = 0.0) -> None:
        self._next_sample_time = float(start_time)
        self._previous_ball_position = None
        self._previous_time = None
        self.latest = None

    def _world_transform(self, path: str) -> Gf.Matrix4d:
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"missing prim: {path}")
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)

    def _read(self, timestamp: float, physics_step: int) -> SdfTactileSample:
        parameters = self.parameters
        ball_world = np.asarray(
            self._world_transform(self._ball_path).ExtractTranslation(),
            dtype=float)

        velocity = np.zeros(3)
        if (self._previous_ball_position is not None
                and self._previous_time is not None
                and timestamp > self._previous_time):
            velocity = ((ball_world - self._previous_ball_position)
                        / (timestamp - self._previous_time))

        raw = np.zeros((2, TAXEL_COUNT, 3), dtype=np.float64)
        penetration = np.zeros((2, TAXEL_COUNT), dtype=np.float64)

        for index, pad_path in enumerate(self._pad_paths):
            matrix = self._world_transform(pad_path)
            rotation = np.array([[matrix[i][j] for j in range(3)]
                                 for i in range(3)]).T
            origin = np.asarray(matrix.ExtractTranslation(), dtype=float)
            # Sensing points sit proud of the collision surface by the soft
            # layer, so the ball is already inside them when PhysX has it
            # resting on the collider.
            proud = self._taxels + np.array(
                [0.0, 0.0, parameters.layer_thickness_m])
            world_points = proud @ rotation.T + origin

            offsets = world_points - ball_world
            distances = np.linalg.norm(offsets, axis=1)
            depth = np.clip(self._radius - distances, 0.0, None)
            penetration[index] = depth
            loaded = depth > 0.0
            if not loaded.any():
                continue

            normals = offsets[loaded] / distances[loaded, None]
            magnitude = (parameters.stiffness_n_per_m
                         * np.power(depth[loaded], parameters.depth_exponent))
            normal_force = normals * magnitude[:, None]

            tangential = velocity - np.einsum(
                "ij,j->i", normals, velocity)[:, None] * normals
            speed = np.linalg.norm(tangential, axis=1)
            moving = speed > 1e-6
            shear_force = np.zeros_like(normal_force)
            if moving.any():
                direction = np.zeros_like(tangential)
                direction[moving] = (tangential[moving]
                                     / speed[moving, None])
                shear_force = (direction * parameters.shear_coefficient
                               * magnitude[:, None])

            world_force = normal_force + shear_force
            local = world_force @ rotation
            raw[index][loaded] = local

        self._previous_ball_position = ball_world
        self._previous_time = timestamp
        device = _apply_device_model(raw)
        return SdfTactileSample(
            timestamp=timestamp, physics_step=physics_step,
            taxel_forces=device, raw_taxel_forces=raw.astype(np.float32),
            total_forces=device.sum(axis=1).astype(np.float32),
            contacts=(np.linalg.norm(device, axis=-1) > 0.0).any(axis=1),
            penetration_m=penetration.astype(np.float32))

    def maybe_sample(self, simulation_time: float, physics_step: int):
        """Emit on the device clock; call once per physics step."""
        simulation_time = float(simulation_time)
        if simulation_time + 1.0e-12 < self._next_sample_time:
            return None
        sample = self._read(simulation_time, int(physics_step))
        while self._next_sample_time <= simulation_time + 1.0e-12:
            self._next_sample_time += self._sample_period
        self.latest = sample
        return sample
