"""Newton-backed DP-S2015 fingertip sensor.

Isaac Lab's Newton ``ContactSensor`` only exposes net forces and the
per-partner force matrix; ``contact_pos_w`` raises ``NotImplementedError``.
The taxel readout needs every contact point, so this class keeps the Isaac
Lab sensor for its registration, filtering and net-force buffers and reads
the per-contact positions, normals and solved forces straight from
``NewtonManager.get_contacts()`` -- the same ``Contacts`` object the solver
fills every step (``rigid_contact_point0/1`` in body frame, ``force`` as a
spatial vector whose linear part is the force on shape 0's body).

Works for MuJoCo's own contacts and for Newton's CollisionPipeline, including
the SDF hydroelastic generator, which is what the port is for.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sensors.contact_sensor import ContactSensor as NewtonContactSensor

from .tactile_contact_sensor import (
    DPS2015ContactSensorCfg,
    DPS2015TactileCore,
    _as_torch,
    _quat_rotate_xyzw,
)


class DPS2015NewtonContactSensor(DPS2015TactileCore, NewtonContactSensor):
    """One-fingertip Newton contact sensor with a 52-taxel output buffer."""

    def __init__(self, cfg: DPS2015ContactSensorCfg) -> None:
        # The Newton sensor rebuilds its cfg from the *base* ContactSensorCfg
        # fields and rejects unknown ones, so hand it a stripped copy and keep
        # the DPS cfg on the core.
        from isaaclab.sensors import ContactSensorCfg as BaseContactSensorCfg

        base_fields = {
            name: getattr(cfg, name)
            for name in BaseContactSensorCfg.__dataclass_fields__
            if name != "class_type"
        }
        super().__init__(BaseContactSensorCfg(**base_fields))
        self._init_tactile_state(cfg)
        self._lookup_ready = False

    # -- backend hooks ------------------------------------------------------

    def _sensing_body_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_q = _as_torch(NewtonManager._state_0.body_q)
        pose = body_q[self._sensing_body_idx]
        return pose[:, :3], pose[:, 3:7]

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        super().reset(env_ids=env_ids, env_mask=env_mask)
        self._reset_tactile(env_ids, env_mask)

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        from isaaclab.sim.utils.queries import find_matching_prim_paths

        paths = find_matching_prim_paths(self.cfg.prim_path)
        if not paths:
            raise RuntimeError(f"No prim matches {self.cfg.prim_path!r}")
        self._setup_tactile(paths[0])
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Body-index tables that turn raw contacts into (env, point, force)."""
        model = NewtonManager._model
        view = self.contact_view
        sensing = list(view.sensing_obj_idx)
        if view.sensing_obj_type != "body":
            raise RuntimeError("DPS2015NewtonContactSensor expects body-level sensing")
        if len(sensing) != self._num_envs:
            raise RuntimeError(
                f"Expected one sensed body per env ({self._num_envs}), Newton "
                f"matched {len(sensing)}: {self.cfg.prim_path}"
            )
        n_bodies = int(model.body_count)
        device = self.device
        self._sensing_body_idx = torch.tensor(sensing, dtype=torch.long, device=device)
        self._body_env_row = torch.full((n_bodies,), -1, dtype=torch.long, device=device)
        self._body_env_row[self._sensing_body_idx] = torch.arange(
            self._num_envs, dtype=torch.long, device=device)
        counterpart_env = torch.full((n_bodies,), -1, dtype=torch.long)
        for row, partners in enumerate(view.counterpart_indices):
            for body in partners:
                counterpart_env[int(body)] = row
        self._counterpart_env = counterpart_env.to(device)
        self._shape_body = _as_torch(model.shape_body).long()
        self._lookup_ready = True

    @staticmethod
    def _solver_is_kamino() -> bool:
        return type(NewtonManager._solver).__name__ == "SolverKamino"

    def _update_buffers_impl(self, env_mask: wp.array | None = None) -> None:
        super()._update_buffers_impl(env_mask)
        if self._mapper is None or self._output_clock is None or not self._lookup_ready:
            return
        positions_w, forces_w, env_ids = self._read_raw_contacts()
        self._publish_contacts(positions_w, forces_w, env_ids)

    # -- raw contact buffer -------------------------------------------------

    def _contact_forces_on_shape0(self, contacts, count: int) -> torch.Tensor:
        """World-frame force on shape 0's body for the first ``count`` contacts.

        MuJoCo-warp and XPBD write it into ``contacts.force``.  Kamino's
        ``update_contacts`` only converts geometry, so its contact impulses
        are read from the P-ADMM solution instead: for contact ``k`` in world
        ``w`` the three multipliers sit at ``vio[w] + ccgo[w] + 3*cid[k]`` in
        the contact frame (z = normal, A -> B), scaled by ``inv_dt``; the
        force on body B is ``+R(frame) lambda`` and on A (shape 0) its
        negative.  Newton's converter keeps the contact order, so index
        ``k`` is the same on both sides.
        """
        solver = NewtonManager._solver
        if type(solver).__name__ != "SolverKamino":
            return _as_torch(contacts.force)[:count][:, :3]
        impl = solver._solver_kamino
        ck = solver._contacts_kamino.data
        wid = _as_torch(ck.wid)[:count].long()
        cid = _as_torch(ck.cid)[:count].long()
        frame = _as_torch(ck.frame)[:count]
        vio = _as_torch(impl._problem_fd.data.vio).long()
        ccgo = _as_torch(impl._data.info.contact_cts_group_offset).long()
        inv_dt = _as_torch(impl._model.time.inv_dt)
        lambdas = _as_torch(impl._solver_fd.data.solution.lambdas)
        base = vio[wid] + ccgo[wid] + 3 * cid
        lam = torch.stack((lambdas[base], lambdas[base + 1], lambdas[base + 2]), dim=-1)
        lam = lam * inv_dt[wid].unsqueeze(-1)
        # Empirically (ball resting on the table: 19.62 N on the 2 kg ball,
        # pointing up) the force on shape 0 is +R lambda in this build.
        return _quat_rotate_xyzw(frame, lam)

    def _empty(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        return empty, empty.clone(), torch.empty((0,), dtype=torch.long, device=self.device)

    def _read_raw_contacts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contacts between this env's sensed body and its filter partners.

        Returns world-frame points and forces acting on the sensed body.
        """
        contacts = NewtonManager.get_contacts()
        if contacts is None or contacts.force is None:
            return self._empty()
        count = int(_as_torch(contacts.rigid_contact_count)[0].item())
        if count == 0:
            return self._empty()

        shape0 = _as_torch(contacts.rigid_contact_shape0)[:count].long()
        shape1 = _as_torch(contacts.rigid_contact_shape1)[:count].long()
        valid = (shape0 >= 0) & (shape1 >= 0)
        body0 = torch.where(valid, self._shape_body[shape0.clamp_min(0)], -1)
        body1 = torch.where(valid, self._shape_body[shape1.clamp_min(0)], -1)
        row0 = torch.where(body0 >= 0, self._body_env_row[body0.clamp_min(0)], -1)
        row1 = torch.where(body1 >= 0, self._body_env_row[body1.clamp_min(0)], -1)
        cp0 = torch.where(body0 >= 0, self._counterpart_env[body0.clamp_min(0)], -1)
        cp1 = torch.where(body1 >= 0, self._counterpart_env[body1.clamp_min(0)], -1)
        # Sensed body is shape 0 and the partner (same env) is shape 1, or
        # the other way round.
        sel0 = (row0 >= 0) & (cp1 == row0)
        sel1 = (row1 >= 0) & (cp0 == row1)
        if not bool((sel0 | sel1).any()):
            return self._empty()

        body_q = _as_torch(NewtonManager._state_0.body_q)
        force = self._contact_forces_on_shape0(contacts, count)
        point0 = _as_torch(contacts.rigid_contact_point0)[:count]
        point1 = _as_torch(contacts.rigid_contact_point1)[:count]

        b0 = body0[sel0]
        b1 = body1[sel1]
        if self._solver_is_kamino():
            # Kamino's converter derived the body-local points from the stale
            # body_q; its own world-space contact points are exact.
            ck = NewtonManager._solver._contacts_kamino.data
            pos0_w = _as_torch(ck.position_A)[:count][sel0]
            pos1_w = _as_torch(ck.position_B)[:count][sel1]
        else:
            pos0_w = body_q[b0, :3] + _quat_rotate_xyzw(body_q[b0, 3:7], point0[sel0])
            pos1_w = body_q[b1, :3] + _quat_rotate_xyzw(body_q[b1, 3:7], point1[sel1])
        # ``force`` is the force on shape 0's body; the sensed body gets the
        # opposite when it is shape 1.
        forces_w = torch.cat((force[sel0], -force[sel1]))
        positions_w = torch.cat((pos0_w, pos1_w))
        env_ids = torch.cat((row0[sel0], row1[sel1]))

        # Guard the sign convention against the sensor's own force matrix
        # (net force on the sensed body from the filter partners).
        matrix = self._data.force_matrix_w
        if matrix is not None:
            target = matrix.torch.sum(dim=(1, 2))
            forces_w = self._align_force_sign(forces_w, env_ids, target)
        return positions_w, forces_w, env_ids
