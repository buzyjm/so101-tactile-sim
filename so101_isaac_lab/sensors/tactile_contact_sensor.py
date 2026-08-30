"""PhysX-backed, product-rate 52-taxel ContactSensor for Isaac Lab 3.0.

Each instance monitors exactly one fingertip rigid body.  Isaac Lab calls the
sensor at every physics step, so the class can integrate at 240 Hz and hold its
latest sample at the DP-S2015-Elite 83.3 Hz device rate.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_physx.sensors import ContactSensor as PhysXContactSensor

from scene_config import TACTILE_NORMAL_FORCE_FROM_PHYSICAL_SIGN
from so101_isaac_lab.tactile_tensor import (
    TactileOutputClock,
    TactileTensorBatch,
    TactileTensorMapper,
)


def _as_torch(value: torch.Tensor | wp.array) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)


def _quat_rotate_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., :3]
    real = quaternion[..., 3:4]
    cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + real * cross + torch.linalg.cross(xyz, cross, dim=-1)


def _quat_rotate_inverse_xyzw(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    xyz = quaternion[..., :3]
    real = quaternion[..., 3:4]
    cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector - real * cross + torch.linalg.cross(xyz, cross, dim=-1)


def _quat_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    )
    real = left_w * right_w - (left_xyz * right_xyz).sum(
        dim=-1, keepdim=True
    )
    return torch.cat((xyz, real), dim=-1)


@configclass
class DPS2015ContactSensorCfg(ContactSensorCfg):
    """Configuration for one DP-S2015-Elite fingertip sensor."""

    class_type: type = None
    sensor_frame_prim_name: str = ""
    """Direct child Xform of the sensed rigid body defining device XYZ axes."""

    normal_force_sign: float = TACTILE_NORMAL_FORCE_FROM_PHYSICAL_SIGN
    """Conversion from physical local Z force to positive device normal force."""

    def __post_init__(self) -> None:
        self.class_type = DPS2015ContactSensor
        if not self.sensor_frame_prim_name:
            raise ValueError("sensor_frame_prim_name must be specified")
        if self.update_period != 0.0:
            raise ValueError(
                "DPS2015ContactSensor requires update_period=0.0 for 240 Hz "
                "substep integration"
            )
        if not self.filter_prim_paths_expr:
            raise ValueError(
                "At least one filter_prim_paths_expr is required to obtain "
                "per-contact positions"
            )
        self.track_contact_points = True
        self.track_friction_forces = True
        if self.max_contact_data_count_per_prim is None:
            self.max_contact_data_count_per_prim = 32


class DPS2015ContactSensor(PhysXContactSensor):
    """One-fingertip PhysX contact sensor with a 52-taxel output buffer."""

    cfg: DPS2015ContactSensorCfg

    def __init__(self, cfg: DPS2015ContactSensorCfg) -> None:
        super().__init__(cfg)
        self._mapper: TactileTensorMapper | None = None
        self._output_clock: TactileOutputClock | None = None
        self._physics_time = 0.0

    @property
    def tactile_data(self) -> TactileTensorBatch:
        """Latest held 83.3 Hz device sample."""
        if self._output_clock is None:
            raise RuntimeError("Tactile sensor has not been initialized")
        return self._output_clock.latest

    def sensor_frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return device-frame position and XYZW quaternion for every env."""
        if self._mapper is None:
            raise RuntimeError("Tactile sensor has not been initialized")
        body_pose = _as_torch(self.body_physx_view.get_transforms()).view(
            self._num_envs, self.num_sensors, 7
        )[:, 0]
        body_pos_w = body_pose[:, :3]
        body_quat_w = body_pose[:, 3:7]
        local_pos = self._frame_pos_b.expand(self._num_envs, -1)
        local_quat = self._frame_quat_b.expand(self._num_envs, -1)
        sensor_pos_w = body_pos_w + _quat_rotate_xyzw(body_quat_w, local_pos)
        sensor_quat_w = _quat_multiply_xyzw(body_quat_w, local_quat)
        sensor_quat_w /= torch.linalg.vector_norm(
            sensor_quat_w, dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        return sensor_pos_w, sensor_quat_w

    def sensor_points_to_world(self, points_sensor: torch.Tensor) -> torch.Tensor:
        """Transform one sensor-local point per environment to world space."""
        if points_sensor.shape != (self._num_envs, 3):
            raise ValueError(
                f"Expected points_sensor shape ({self._num_envs}, 3), got "
                f"{tuple(points_sensor.shape)}"
            )
        sensor_pos_w, sensor_quat_w = self.sensor_frame_pose_w()
        return sensor_pos_w + _quat_rotate_xyzw(sensor_quat_w, points_sensor)

    def sensor_vectors_to_world(self, vectors_sensor: torch.Tensor) -> torch.Tensor:
        """Rotate one sensor-local vector per environment to world space."""
        if vectors_sensor.shape != (self._num_envs, 3):
            raise ValueError(
                f"Expected vectors_sensor shape ({self._num_envs}, 3), got "
                f"{tuple(vectors_sensor.shape)}"
            )
        _, sensor_quat_w = self.sensor_frame_pose_w()
        return _quat_rotate_xyzw(sensor_quat_w, vectors_sensor)

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        super().reset(env_ids=env_ids, env_mask=env_mask)
        if self._output_clock is None:
            return
        clock_env_ids: Sequence[int] | torch.Tensor | None = env_ids
        if clock_env_ids is None and env_mask is not None:
            clock_env_ids = _as_torch(env_mask).nonzero().flatten()
        self._output_clock.reset(clock_env_ids)
        if clock_env_ids is None:
            self._physics_time = 0.0

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        if self.num_sensors != 1:
            raise RuntimeError(
                "DPS2015ContactSensor must match exactly one fingertip body per "
                f"environment, got {self.num_sensors}"
            )
        self._read_sensor_frame_offset()
        self._mapper = TactileTensorMapper(
            device=self.device, fingertip_count=1
        )
        self._output_clock = TactileOutputClock(
            num_envs=self._num_envs,
            device=self.device,
            fingertip_count=1,
        )

    def _read_sensor_frame_offset(self) -> None:
        body_path = self.body_physx_view.prim_paths[0]
        frame_path = f"{body_path}/{self.cfg.sensor_frame_prim_name}"
        frame_prim = sim_utils.get_current_stage().GetPrimAtPath(frame_path)
        if not frame_prim.IsValid():
            raise RuntimeError(
                f"Sensor frame prim '{frame_path}' does not exist. The tactile "
                "CAD layer must be present in the spawned robot asset."
            )
        local_matrix = UsdGeom.Xformable(frame_prim).GetLocalTransformation()
        transform = Gf.Transform(local_matrix)
        translation = transform.GetTranslation()
        quaternion = transform.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        self._frame_pos_b = torch.tensor(
            tuple(translation), dtype=torch.float32, device=self.device
        )
        self._frame_quat_b = torch.tensor(
            (*tuple(imaginary), quaternion.GetReal()),
            dtype=torch.float32,
            device=self.device,
        )
        self._frame_quat_b /= torch.linalg.vector_norm(self._frame_quat_b)

    def _update_buffers_impl(self, env_mask: wp.array | None = None) -> None:
        super()._update_buffers_impl(env_mask)
        if self._mapper is None or self._output_clock is None:
            return

        positions_w, forces_w, env_ids = self._read_point_contacts()
        if positions_w.numel():
            positions_sensor, forces_sensor = self._world_to_sensor(
                positions_w, forces_w, env_ids
            )
            finite = torch.isfinite(positions_sensor).all(dim=1) & torch.isfinite(
                forces_sensor
            ).all(dim=1)
            nonzero = torch.linalg.vector_norm(forces_sensor, dim=1) > 0.0
            keep = finite & nonzero
            positions_sensor = positions_sensor[keep]
            forces_sensor = forces_sensor[keep]
            env_ids = env_ids[keep]
        else:
            positions_sensor = torch.empty(
                (0, 3), dtype=torch.float32, device=self.device
            )
            forces_sensor = torch.empty_like(positions_sensor)
            env_ids = torch.empty((0,), dtype=torch.long, device=self.device)

        raw = self._mapper.map_raw_contacts(
            positions_sensor,
            forces_sensor,
            env_ids,
            torch.zeros_like(env_ids),
            num_envs=self._num_envs,
        )
        self._physics_time += self._sim_physics_dt
        self._output_clock.update(
            raw,
            physics_dt=self._sim_physics_dt,
            simulation_time=self._physics_time,
        )

    def _read_point_contacts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            normal_force,
            normal_position,
            normal_direction,
            _,
            normal_count,
            normal_start,
        ) = self.contact_view.get_contact_data(dt=self._sim_physics_dt)
        normal_position, normal_env_ids = self._unpack_raw_buffer(
            normal_position, normal_count, normal_start
        )
        normal_force_values, _ = self._unpack_raw_buffer(
            normal_force, normal_count, normal_start
        )
        normal_directions, _ = self._unpack_raw_buffer(
            normal_direction, normal_count, normal_start
        )
        if normal_force_values.numel():
            normal_forces_w = normal_directions * normal_force_values.reshape(
                -1, 1
            )
            normal_forces_w = self._align_force_sign(
                normal_forces_w,
                normal_env_ids,
                self._data.force_matrix_w.torch.sum(dim=(1, 2)),
            )
        else:
            normal_forces_w = torch.empty(
                (0, 3), dtype=torch.float32, device=self.device
            )

        (
            friction_force,
            friction_position,
            friction_count,
            friction_start,
        ) = self.contact_view.get_friction_data(dt=self._sim_physics_dt)
        friction_positions, friction_env_ids = self._unpack_raw_buffer(
            friction_position, friction_count, friction_start
        )
        friction_forces_w, _ = self._unpack_raw_buffer(
            friction_force, friction_count, friction_start
        )
        if friction_forces_w.numel():
            friction_forces_w = friction_forces_w.reshape(-1, 3)
            friction_forces_w = self._align_force_sign(
                friction_forces_w,
                friction_env_ids,
                self._data.friction_forces_w.torch.sum(dim=(1, 2)),
            )

        return (
            torch.cat((normal_position.reshape(-1, 3), friction_positions.reshape(-1, 3))),
            torch.cat((normal_forces_w.reshape(-1, 3), friction_forces_w.reshape(-1, 3))),
            torch.cat((normal_env_ids, friction_env_ids)),
        )

    def _unpack_raw_buffer(
        self,
        values: torch.Tensor | wp.array,
        counts: torch.Tensor | wp.array,
        starts: torch.Tensor | wp.array,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values_torch = _as_torch(values)
        counts_flat = _as_torch(counts).reshape(-1).to(torch.long)
        starts_flat = _as_torch(starts).reshape(-1).to(torch.long)
        row_ids = torch.repeat_interleave(
            torch.arange(counts_flat.numel(), device=self.device), counts_flat
        )
        if row_ids.numel() == 0:
            return values_torch[:0], torch.empty(
                (0,), dtype=torch.long, device=self.device
            )
        block_starts = counts_flat.cumsum(0) - counts_flat
        deltas = torch.arange(row_ids.numel(), device=self.device) - torch.repeat_interleave(
            block_starts, counts_flat
        )
        value_indices = starts_flat[row_ids] + deltas
        body_rows = row_ids // self.contact_view.filter_count
        env_ids = body_rows // self.num_sensors
        return values_torch.index_select(0, value_indices), env_ids

    def _align_force_sign(
        self,
        forces_w: torch.Tensor,
        env_ids: torch.Tensor,
        target_force_w: torch.Tensor,
    ) -> torch.Tensor:
        summed = torch.zeros(
            (self._num_envs, 3), dtype=forces_w.dtype, device=self.device
        )
        summed.index_add_(0, env_ids, forces_w)
        flip = (summed * target_force_w).sum(dim=1) < 0.0
        sign = torch.where(flip, -1.0, 1.0).to(forces_w.dtype)
        return forces_w * sign[env_ids, None]

    def _world_to_sensor(
        self,
        positions_w: torch.Tensor,
        forces_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sensor_pos_w, sensor_quat_w = self.sensor_frame_pose_w()
        selected_pos = sensor_pos_w[env_ids]
        selected_quat = sensor_quat_w[env_ids]
        positions_sensor = _quat_rotate_inverse_xyzw(
            selected_quat, positions_w - selected_pos
        )
        forces_sensor = _quat_rotate_inverse_xyzw(selected_quat, forces_w)
        forces_sensor[:, 2] *= float(self.cfg.normal_force_sign)
        return positions_sensor, forces_sensor
