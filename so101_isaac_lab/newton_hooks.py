"""MODEL_INIT hooks that retune the Newton model before it is finalized.

Isaac Lab's Newton backend builds a ``newton.ModelBuilder`` from the USD
stage, dispatches ``PhysicsEvent.MODEL_INIT`` and only then finalizes the
model.  That window is the one place where per-shape parameters that the USD
importer does not expose (contact stiffness, MuJoCo impedance, hydroelastic
flags, convex-decomposition fidelity) can still be edited.  Everything here
is opt-in; nothing runs unless a script calls one of the ``register_*``
functions before the environment is created.

Why the contact retune exists: with Newton's defaults (``ke=2.5e3``,
``kd=100`` -> MuJoCo ``solref=(0.02, 1)``, ``solimp=(0.9, 0.95, ...)``) the
4.8 g ball is pushed 1-2 cm into the table by a 0.2 N pad touch and the pinch
never builds up -- MuJoCo's soft, mass-scaled contact model against a nearly
massless body.  PhysX resolves the same pinch rigidly at 4 N per pad.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Keep strong references: PhysicsManager may store callbacks weakly.
_ACTIVE_CALLBACKS: list[Callable[..., None]] = []


def _register(callback: Callable[..., None], name: str) -> None:
    from isaaclab.physics.physics_manager import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    _ACTIVE_CALLBACKS.append(callback)
    NewtonManager.register_callback(callback, PhysicsEvent.MODEL_INIT, name=name, wrap_weak_ref=False)
    logger.info("Registered Newton MODEL_INIT hook %s", name)


def _shape_indices(builder, shape_regex: str | None) -> list[int]:
    labels = list(builder.shape_label)
    if shape_regex is None:
        return list(range(len(labels)))
    pattern = re.compile(shape_regex)
    return [i for i, label in enumerate(labels) if pattern.search(label)]


def register_contact_tuning(
    *,
    ke: float | None = None,
    kd: float | None = None,
    solimp: tuple[float, float, float, float, float] | None = None,
    shape_regex: str | None = None,
) -> None:
    """Override contact stiffness/damping and MuJoCo impedance on matching shapes.

    ``ke``/``kd`` feed Newton's ``convert_solref``: ``timeconst = 2 / kd`` and
    ``dampratio = kd / 2 / sqrt(ke)`` (MuJoCo clamps ``timeconst`` to two
    physics steps).  ``solimp`` is MuJoCo's ``(d0, dmax, width, midpoint,
    power)``; ``d`` close to 1 makes the constraint nearly rigid.
    """

    def _apply(payload: Any = None) -> None:
        from isaaclab_newton.physics import NewtonManager

        builder = NewtonManager._builder
        idx = _shape_indices(builder, shape_regex)
        if ke is not None:
            for i in idx:
                builder.shape_material_ke[i] = float(ke)
        if kd is not None:
            for i in idx:
                builder.shape_material_kd[i] = float(kd)
        if solimp is not None:
            key = next((k for k, a in builder.custom_attributes.items()
                        if getattr(a, "name", "") == "geom_solimp"), None)
            if key is None:
                logger.warning("MuJoCo geom_solimp custom attribute not registered; solimp ignored")
            else:
                attr = builder.custom_attributes[key]
                values = dict(attr.values or {})
                for i in idx:
                    values[i] = tuple(float(v) for v in solimp)
                builder.custom_attributes[key] = replace(attr, values=values)
        logger.info("Contact tuning applied to %d shapes (ke=%s kd=%s solimp=%s regex=%s)",
                    len(idx), ke, kd, solimp, shape_regex)
        print(f"[newton_hooks] contact tuning: {len(idx)} shapes ke={ke} kd={kd} "
              f"solimp={solimp} regex={shape_regex}", flush=True)

    _register(_apply, "so101_contact_tuning")


def register_hydroelastic(
    *,
    shape_regex: str,
    kh: float = 1.0e6,
    mesh_sdf_max_resolution: int = 64,
    mesh_sdf_narrow_band_m: float = 0.005,
    mesh_sdf_margin_m: float = 0.004,
    compliant_layer_m: float = 0.0,
    primitive_sdf_max_resolution: int = 32,
) -> None:
    """Switch matching shapes to SDF hydroelastic contact before finalize.

    Hydroelastic contacts only fire between two shapes that both carry the
    flag, so match the fingertip pads *and* the ball; everything else keeps
    rigid point contacts.  Mesh shapes get an SDF built here (the importer
    does not build any); primitives get their SDF resolution set so
    ``finalize`` voxelises them.  ``kh`` is the hydroelastic stiffness
    (force per unit penetration volume, effectively), ``compliant_layer_m``
    shrinks the mesh SDF surface inward to model a soft skin of that
    thickness.
    """

    def _apply(payload: Any = None) -> None:
        from newton import GeoType, ShapeFlags
        from isaaclab.physics.physics_manager import PhysicsManager
        from isaaclab_newton.physics import NewtonManager

        builder = NewtonManager._builder
        # SDFs must live on the model's device; the default Warp device may
        # be a different GPU than the one the simulation was asked to use.
        device = PhysicsManager._device
        idx = _shape_indices(builder, shape_regex)
        built_sdfs = 0
        primitives = 0
        for i in idx:
            builder.shape_flags[i] |= int(ShapeFlags.HYDROELASTIC)
            builder.shape_material_kh[i] = float(kh)
            stype = builder.shape_type[i]
            src = builder.shape_source[i]
            if stype in (GeoType.MESH, GeoType.CONVEX_MESH):
                if src is None:
                    raise RuntimeError(f"shape {builder.shape_label[i]} has no mesh source")
                # finalize() only picks up an attached mesh SDF for GeoType.MESH;
                # a CONVEX_MESH would be sent down the primitive-voxelisation
                # path, which has no generator for it.  The convex hull is a
                # closed mesh, so demote it.
                builder.shape_type[i] = GeoType.MESH
                if getattr(src, "sdf", None) is None:
                    scale = tuple(float(v) for v in builder.shape_scale[i])
                    src.build_sdf(
                        device=device,
                        max_resolution=int(mesh_sdf_max_resolution),
                        narrow_band_range=(-float(mesh_sdf_narrow_band_m), float(mesh_sdf_narrow_band_m)),
                        margin=float(mesh_sdf_margin_m),
                        shape_margin=float(compliant_layer_m),
                        # Hydroelastic requires scale_baked SDFs, unit scale included.
                        scale=scale,
                    )
                    built_sdfs += 1
            else:
                builder.shape_sdf_max_resolution[i] = int(primitive_sdf_max_resolution)
                builder.shape_sdf_target_voxel_size[i] = None
                primitives += 1
        print(f"[newton_hooks] hydroelastic on {len(idx)} shapes (kh={kh:g}, "
              f"{built_sdfs} mesh SDFs built on {device}, {primitives} primitives voxelised at "
              f"{primitive_sdf_max_resolution}, compliant layer {compliant_layer_m*1e3:.1f} mm)",
              flush=True)
        for i in idx[:4]:
            src = builder.shape_source[i]
            sdf = getattr(src, "sdf", None) if src is not None else None
            print(f"[newton_hooks]   shape {i} {builder.shape_label[i].split('/')[-1]} "
                  f"type={builder.shape_type[i]} flags={builder.shape_flags[i]} "
                  f"sdf={'yes' if sdf is not None else 'no'} "
                  f"res={builder.shape_sdf_max_resolution[i]}", flush=True)

    _register(_apply, "so101_hydroelastic")


def register_joint_gain_scale(*, ke_scale: float = 1.0, kd_scale: float = 1.0) -> None:
    """Multiply every joint drive's target stiffness/damping before finalize.

    Isaac Lab writes the actuator gains (17.8 N*m/rad, 0.6 N*m*s/rad here)
    into ``joint_target_ke/kd`` unchanged; MuJoCo-warp tracks with them,
    XPBD's compliance-based joint target does not.  This hook lets the XPBD
    gain semantics be probed without touching the shared actuator config.
    """

    def _apply(payload: Any = None) -> None:
        from isaaclab_newton.physics import NewtonManager

        builder = NewtonManager._builder
        n = 0
        for i, ke in enumerate(builder.joint_target_ke):
            if ke > 0.0:
                builder.joint_target_ke[i] = float(ke) * ke_scale
                builder.joint_target_kd[i] = float(builder.joint_target_kd[i]) * kd_scale
                n += 1
        print(f"[newton_hooks] joint gains scaled on {n} dofs (ke x{ke_scale:g}, kd x{kd_scale:g})",
              flush=True)

    _register(_apply, "so101_joint_gain_scale")


def register_disable_collision(*, shape_regex: str) -> None:
    """Turn off shape collision for matching shapes (they still render).

    Used to take the tactile-mount adapters out of the Newton contact set:
    the importer's coarse CoACD pass turns each concave adapter into one
    hull that fills the pad's pocket, and shapes cannot be added after
    replication without breaking MuJoCo's per-world geom layout.
    """

    def _apply(payload: Any = None) -> None:
        from newton import ShapeFlags
        from isaaclab_newton.physics import NewtonManager

        builder = NewtonManager._builder
        idx = _shape_indices(builder, shape_regex)
        for i in idx:
            builder.shape_flags[i] &= ~int(ShapeFlags.COLLIDE_SHAPES)
        print(f"[newton_hooks] collision disabled on {len(idx)} shapes matching {shape_regex!r}",
              flush=True)

    _register(_apply, "so101_disable_collision")


def register_redecomposition(
    *,
    shape_regex: str,
    threshold: float = 0.1,
    usd_stage_getter: Callable[[], Any] | None = None,
) -> None:
    """Re-run CoACD on matching mesh shapes with a finer threshold.

    Newton's USD importer decomposes ``convexDecomposition`` meshes with
    CoACD at ``threshold=0.5``, which collapses the concave tactile adapters
    into a single hull.  The original triangles are re-read from the stage
    (the builder only keeps the coarse hull) and decomposed again.
    """

    def _apply(payload: Any = None) -> None:
        import numpy as np
        from newton import Mesh
        from isaaclab_newton.physics import NewtonManager
        from pxr import UsdGeom

        if usd_stage_getter is not None:
            stage = usd_stage_getter()
        else:
            import isaaclab.sim as sim_utils

            stage = sim_utils.get_current_stage()
        builder = NewtonManager._builder
        idx = _shape_indices(builder, shape_regex)
        cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        replaced = 0
        for i in idx:
            path = builder.shape_label[i]
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
                # Cloned envs may reference the prototype; try the env_0 path.
                alt = re.sub(r"/env_\d+/", "/env_0/", path)
                prim = stage.GetPrimAtPath(alt)
                if not prim.IsValid():
                    logger.warning("No USD mesh for shape %s", path)
                    continue
                key = alt
            else:
                key = path
            if key not in cache:
                mesh = UsdGeom.Mesh(prim)
                pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                tris = []
                o = 0
                for c in counts:
                    for k in range(1, int(c) - 1):
                        tris.append((fvi[o], fvi[o + k], fvi[o + k + 1]))
                    o += int(c)
                cache[key] = (pts, np.asarray(tris, dtype=np.int32).reshape(-1))
            pts, tri = cache[key]
            old = builder.shape_source[i]
            builder.shape_source[i] = Mesh(pts, tri, maxhullvert=getattr(old, "maxhullvert", 64))
            replaced += 1
        if replaced:
            builder.approximate_meshes(method="coacd", shape_indices=idx, threshold=threshold)
        print(f"[newton_hooks] re-decomposed {replaced} shapes with coacd threshold={threshold}; "
              f"shape_count now {builder.shape_count}", flush=True)

    _register(_apply, "so101_redecompose")
