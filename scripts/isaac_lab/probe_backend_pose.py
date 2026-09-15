"""Replay the scripted pick-and-place on a chosen physics backend and report.

Phase-0/1 check for the Newton port.  Bring up the pick-and-place scene on
``--physics {physx,newton_mjwarp,newton_pipeline,newton_hydro}`` and either

* hold one IK waypoint (``--probe_pose grasp``), or
* replay the nominal trajectory up to the end of a phase
  (``--until grip_hold``, ``--until lift_hold``, ...),

then write a JSON report: gripper/jaw poses (env-relative), joint tracking
error, ball position vs its start, and -- with ``--dump_contacts`` -- the
pad/ball contact forces seen by plain Isaac Lab contact sensors plus, on
Newton, the raw per-contact buffer (positions, normals, solved forces).  Run
once per backend and diff the reports; PhysX is the calibrated reference.

The project's DP-S2015 sensors are PhysX-only until phase 1 lands, so the
scene runs with plain contact sensors (or none).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from so101_isaac_lab.physics_presets import PRESET_NAMES  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics", choices=PRESET_NAMES, default="physx")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--env_spacing", type=float, default=1.4)
parser.add_argument("--probe_pose", default=None,
                    help="Hold one waypoint label (mutually exclusive with --until).")
parser.add_argument("--until", default=None,
                    help="Replay the phase plan through this phase label "
                         "(approach, descend, close, grip_hold, lift, lift_hold, ...).")
parser.add_argument("--gripper_deg", type=float, default=None,
                    help="Jaw command for --probe_pose (default: open angle).")
parser.add_argument("--hold_s", type=float, default=2.5)
parser.add_argument("--waypoints", type=Path,
                    default=PROJECT_ROOT / "calibration" / "pick_place_waypoints_nominal.json")
parser.add_argument("--init_rot", type=float, nargs=4, default=None,
                    help="Override the robot init_state.rot (4 floats, as authored).")
parser.add_argument("--dump_contacts", action="store_true",
                    help="Attach plain contact sensors to both fingertips and "
                         "report their forces; on Newton also dump the raw "
                         "per-contact buffer for env 0.")
parser.add_argument("--dps_sensors", action="store_true",
                    help="Keep the scene's DP-S2015 tactile sensors (backend-"
                         "dispatched) and the tactile observation terms; report "
                         "active taxels and pad forces at every phase end.")
parser.add_argument("--taxel_threshold_n", type=float, default=0.1,
                    help="Taxel force magnitude counted as active [N].")
parser.add_argument("--dump_shapes", action="store_true",
                    help="Newton only: list env 0's collision shapes (type, scale, "
                         "world AABB, flags), every contact pair, and the MuJoCo "
                         "geom parameters behind them.")
parser.add_argument("--trace_contacts", action="store_true",
                    help="Newton only, with --until: every 0.05 s from the close "
                         "phase on, print env 0's ball height and every contact "
                         "pair carrying force on the ball.")
parser.add_argument("--newton_ke", type=float, default=None,
                    help="Newton only: override contact stiffness on all shapes.")
parser.add_argument("--newton_kd", type=float, default=None,
                    help="Newton only: override contact damping on all shapes.")
parser.add_argument("--newton_solimp", type=float, nargs=5, default=None,
                    help="Newton/MuJoCo only: geom solimp (d0 dmax width midpoint power).")
parser.add_argument("--newton_tune_regex", default=None,
                    help="Restrict the contact tuning to shape labels matching this regex.")
parser.add_argument("--hydro", action="store_true",
                    help="Newton pipeline only: flag the fingertip pads and the "
                         "ball as SDF hydroelastic shapes (use with "
                         "--physics newton_hydro).")
parser.add_argument("--hydro_kh", type=float, default=1.0e6)
parser.add_argument("--hydro_layer_mm", type=float, default=0.0,
                    help="Compliant layer thickness baked into the pad SDFs.")
parser.add_argument("--hydro_mesh_res", type=int, default=64)
parser.add_argument("--hydro_ball_res", type=int, default=32)
parser.add_argument("--hydro_regex", default="TaxelAlignedCollision|TargetBall/geometry")
parser.add_argument("--kinematic_ball", action="store_true",
                    help="Pin the ball in place (kinematic rigid body) so the "
                         "jaw squeezes a stationary target: isolates the contact "
                         "model from the grasp dynamics.")
parser.add_argument("--ball_mass", type=float, default=None,
                    help="Override the ball mass [kg]; a heavy ball stays put under "
                         "the pinch on every backend (friction with the table).")
parser.add_argument("--gripper_effort", type=float, default=None,
                    help="Effort limit [N*m] for the gripper joint only, giving a "
                         "force-controlled squeeze.")
parser.add_argument("--close_deg", type=float, default=None,
                    help="Override the close-phase jaw angle of the nominal plan.")
parser.add_argument("--no_adapter", action="store_true",
                    help="Newton only: disable collision on the tactile-mount "
                         "adapter shapes (their coarse single hull hits the ball).")
parser.add_argument("--asset", default=None,
                    help="Override the robot USD (path relative to assets/isaac_lab, "
                         "e.g. so101_tactile_noadapter.usda).")
parser.add_argument("--coacd_threshold", type=float, default=None,
                    help="Newton only: re-decompose shapes matching --coacd_regex with CoACD.")
parser.add_argument("--coacd_regex", default="AdapterCollision")
parser.add_argument("--joint_gain_scale", type=float, nargs=2, default=None,
                    metavar=("KE_SCALE", "KD_SCALE"),
                    help="Newton only: scale every joint drive's target ke/kd "
                         "before finalize (XPBD gain-semantics probe).")
parser.add_argument("--dump_bodies", action="store_true",
                    help="Newton only: print env 0's body poses from state_0.body_q "
                         "next to poses recomputed by eval_fk from joint_q (into a "
                         "scratch state), to locate solver-specific layout differences.")
parser.add_argument("--dump_control", action="store_true",
                    help="Newton only: print the model's joint drive gains and the "
                         "control struct (targets, joint_f) for env 0 after the run.")
parser.add_argument("--render_check", action="store_true",
                    help="Keep the overview camera, run a 3 s streaming lead-in "
                         "and save one RGB frame (needs --enable_cameras).")
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.probe_pose is None) == (args.until is None):
    parser.error("give exactly one of --probe_pose or --until")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from scene_config import TABLE_TOP_Z, TARGET_BALL_RADIUS  # noqa: E402
from so101_isaac_lab.env_cfg import SO101_JOINT_NAMES, SO101TactileEnvCfg  # noqa: E402
from so101_isaac_lab.physics_presets import make_physics_cfg  # noqa: E402
from so101_isaac_lab.scene_cfg import (  # noqa: E402
    GRIPPER_LINK_RELATIVE_PATH,
    MOVING_JAW_RELATIVE_PATH,
    BallPickPlaceSceneCfg,
)

PHYSICS_HZ = 240.0


def quintic_blend(phase: float) -> float:
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def build_timeline(dump: dict) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Same drive-target sequence as record_parallel_pick_place.py."""
    solutions = {k: np.asarray(v) for k, v in dump["solutions_rad"].items()}
    command = np.concatenate([solutions["approach"], [np.radians(dump["open_gripper_deg"])]])
    rows = [command.copy()]
    phase_ends = []
    for phase in dump["phase_plan"]:
        steps = max(1, round(phase["duration_s"] * PHYSICS_HZ))
        if phase["arm"] is None:
            rows.extend([command.copy()] * steps)
            phase_ends.append((phase["label"], len(rows) - 1))
            continue
        start = command.copy()
        target = np.concatenate([solutions[phase["arm"]], [np.radians(phase["gripper_deg"])]])
        for step in range(1, steps + 1):
            command = start + quintic_blend(step / steps) * (target - start)
            rows.append(command.copy())
        phase_ends.append((phase["label"], len(rows) - 1))
    return np.asarray(rows), phase_ends


def bare_observation_cfgs():
    """Observation/reward configs that do not touch the tactile sensors."""
    import isaaclab.envs.mdp as lab_mdp
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.utils import configclass

    @configclass
    class BareObsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=lab_mdp.joint_pos)

            def __post_init__(self) -> None:
                self.concatenate_terms = True
                self.enable_corruption = False

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class BareRewardsCfg:
        pass

    return BareObsCfg(), BareRewardsCfg()


def plain_contact_sensor_cfgs():
    """Backend-dispatched Isaac Lab contact sensors on both fingertip bodies."""
    from isaaclab.sensors import ContactSensorCfg

    def make(rel_path: str) -> ContactSensorCfg:
        return ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{rel_path}",
            update_period=0.0,
            history_length=0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/TargetBall"],
            debug_vis=False,
        )

    return make(GRIPPER_LINK_RELATIVE_PATH), make(MOVING_JAW_RELATIVE_PATH)


def _quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz, w = q[..., :3], q[..., 3:4]
    c = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * c + torch.linalg.cross(xyz, c, dim=-1)


def _r(t, nd=4):
    return [round(float(v), nd) for v in torch.as_tensor(t).flatten().tolist()]


def tactile_summary(scene, threshold_n: float) -> dict:
    """Active taxels and total force per pad from the DPS2015 sensors."""
    out = {}
    for name in ("tactile_fixed", "tactile_moving"):
        sensor = scene[name]
        data = sensor.tactile_data
        taxel = data.taxel_forces  # (N, 1, 52, 3) device output
        mag = torch.linalg.norm(taxel, dim=-1)[:, 0]  # (N, 52)
        active = (mag > threshold_n).sum(dim=-1)
        total = data.total_forces[:, 0] if hasattr(data, "total_forces") else taxel[:, 0].sum(dim=1)
        out[name] = {
            "type": type(sensor).__name__,
            "active_taxels_per_env": [int(v) for v in active.tolist()],
            "taxel_sum_norm_per_env": _r(torch.linalg.norm(taxel[:, 0].sum(dim=1), dim=-1), 3),
            "total_force_env0": _r(total[0], 3),
            "max_taxel_n_env0": round(float(mag[0].max()), 3),
        }
    return out


def sensor_summary(scene) -> dict:
    """Net and filtered forces from the plain sensors, env 0 plus per-env norms."""
    out = {}
    for name in ("tactile_fixed", "tactile_moving"):
        sensor = scene[name]
        data = sensor.data
        net = data.net_forces_w
        net = net.torch if hasattr(net, "torch") else torch.as_tensor(net)
        names = getattr(sensor, "sensor_names", None) or getattr(sensor, "body_names", None) or []
        entry = {
            "sensor_names": list(names),
            "net_force_env0": _r(net[0, 0]),
            "net_force_norm_per_env": _r(torch.linalg.norm(net[:, 0], dim=-1), 3),
        }
        fm = data.force_matrix_w
        if fm is not None:
            fm = fm.torch if hasattr(fm, "torch") else torch.as_tensor(fm)
            entry["filter_object_names"] = list(getattr(sensor, "filter_object_names", []) or [])
            entry["ball_force_env0"] = _r(fm[0, 0, 0])
            entry["ball_force_norm_per_env"] = _r(torch.linalg.norm(fm[:, 0, 0], dim=-1), 3)
        cv = getattr(sensor, "contact_view", None)
        if cv is not None and hasattr(cv, "sensing_obj_idx"):
            entry["newton_sensing_obj_idx"] = list(cv.sensing_obj_idx)[:8]
            entry["newton_sensing_obj_type"] = cv.sensing_obj_type
            entry["newton_counterpart_type"] = cv.counterpart_type
            entry["newton_counterpart_row0"] = list(cv.counterpart_indices[0]) if cv.counterpart_indices else []
        out[name] = entry
    return out


def dump_newton_contacts(ball) -> dict:
    """Read Newton's raw contact buffer and summarise the pad/ball contacts.

    Verifies, before any sensor code is written, that per-contact positions,
    normals and solved forces are reachable on this backend and that the
    body-frame points land on the fingertip pads.
    """
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    contacts = NewtonManager.get_contacts()
    model = NewtonManager._model
    state = NewtonManager._state_0
    if contacts is None:
        return {"error": "NewtonManager has no Contacts object"}
    if contacts.force is None:
        return {"error": "contacts.force not allocated (no contact sensor registered)"}
    wp.synchronize()

    n = int(wp.to_torch(contacts.rigid_contact_count)[0].item())
    shape_label = list(model.shape_label)
    body_label = list(model.body_label)
    shape_body = wp.to_torch(model.shape_body).long()
    body_q = wp.to_torch(state.body_q)
    shape0 = wp.to_torch(contacts.rigid_contact_shape0)[:n].long()
    shape1 = wp.to_torch(contacts.rigid_contact_shape1)[:n].long()
    point0 = wp.to_torch(contacts.rigid_contact_point0)[:n]
    point1 = wp.to_torch(contacts.rigid_contact_point1)[:n]
    normal = wp.to_torch(contacts.rigid_contact_normal)[:n]
    force = wp.to_torch(contacts.force)[:n]

    summary = {
        "rigid_contact_max": int(contacts.rigid_contact_max),
        "rigid_contact_count": n,
        "num_shapes": len(shape_label),
        "num_bodies": len(body_label),
        "force_shape": list(force.shape),
        "body_label_examples": body_label[:3] + body_label[-2:],
        "env0_pad_ball": [],
        "per_env_pad_ball_counts": {},
    }

    def is_pad(label: str) -> bool:
        return "TactileFixedMount" in label or "TactileMovingMount" in label

    def is_ball(label: str) -> bool:
        return "TargetBall" in label

    def env_of(label: str) -> str:
        for p in label.split("/"):
            if p.startswith("env_"):
                return p
        return "?"

    summary["env0_pad_shape_labels"] = [
        l.split("/Robot/")[-1] for l in shape_label if env_of(l) == "env_0" and is_pad(l)]
    summary["env0_ball_shape_labels"] = [
        l for l in shape_label if env_of(l) == "env_0" and is_ball(l)]
    if n == 0:
        return summary

    counts: dict[str, int] = {}
    b0 = shape_body[shape0]
    b1 = shape_body[shape1]
    ball_c = ball.data.root_pos_w[0]
    for i in range(n):
        l0, l1 = shape_label[shape0[i]], shape_label[shape1[i]]
        pair_ok = (is_pad(l0) and is_ball(l1)) or (is_ball(l0) and is_pad(l1))
        if not pair_ok:
            continue
        e = env_of(l0)
        counts[e] = counts.get(e, 0) + 1
        if e != "env_0":
            continue
        f3 = force[i][:3]

        def to_world(b, p):
            if b < 0:
                return p
            return body_q[b, :3] + _quat_rotate_xyzw(body_q[b, 3:7], p)

        p0w = to_world(b0[i], point0[i])
        p1w = to_world(b1[i], point1[i])
        summary["env0_pad_ball"].append({
            "shape0": "/".join(l0.split("/")[-2:]),
            "shape1": "/".join(l1.split("/")[-2:]),
            "point0_w": _r(p0w),
            "point1_w": _r(p1w),
            "dist_point0_to_ball_centre": round(float(torch.linalg.norm(p0w - ball_c)), 4),
            "normal": _r(normal[i], 3),
            "force": _r(f3, 3),
            "force_norm": round(float(torch.linalg.norm(f3)), 3),
            "force_tail": _r(force[i][3:], 3),
        })
    summary["per_env_pad_ball_counts"] = counts
    summary["env0_ball_centre_w"] = _r(ball_c)
    return summary


def dump_newton_shapes(ball) -> dict:
    """Env-0 shape inventory, all contact pairs, and MuJoCo geom parameters."""
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager._model
    state = NewtonManager._state_0
    solver = NewtonManager._solver
    contacts = NewtonManager.get_contacts()
    wp.synchronize()

    geo_names: dict[int, str] = {}
    flag_collide = 1
    try:
        import newton
        try:
            geo_names = {int(m): m.name for m in newton.GeoType}
        except TypeError:
            geo_names = {int(v): k for k, v in vars(newton.GeoType).items() if isinstance(v, int)}
        flag_collide = int(getattr(newton.ShapeFlags, "COLLIDE_SHAPES", 1))
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[dump_shapes] enum lookup failed: {exc}", flush=True)

    shape_label = list(model.shape_label)
    body_label = list(model.body_label)
    shape_type = wp.to_torch(model.shape_type).cpu().numpy()
    shape_scale = wp.to_torch(model.shape_scale).cpu().numpy()
    shape_body = wp.to_torch(model.shape_body).cpu().numpy()
    shape_flags = wp.to_torch(model.shape_flags).cpu().numpy()
    shape_xf = wp.to_torch(model.shape_transform).cpu().numpy()
    groups = wp.to_torch(model.shape_collision_group).cpu().numpy() if model.shape_collision_group is not None else None
    body_q = wp.to_torch(state.body_q).cpu().numpy()

    def env_of(label: str) -> str:
        for p in label.split("/"):
            if p.startswith("env_"):
                return p
        return "global"

    def q_rot(q, v):
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        xyz, w = q[:3], q[3]
        c = 2.0 * np.cross(xyz, v)
        return v + w * c + np.cross(xyz, c)

    def world_aabb(i):
        src = model.shape_source[i] if i < len(model.shape_source) else None
        s = shape_scale[i]
        if src is not None and hasattr(src, "vertices"):
            pts = np.asarray(src.vertices, dtype=np.float64) * s
        else:
            r = float(max(abs(s[0]), 1e-6))
            pts = np.array([[-r, -r, -r], [r, r, r]])
        pts = np.stack([q_rot(shape_xf[i, 3:7], p) + shape_xf[i, :3] for p in pts])
        b = int(shape_body[i])
        if b >= 0:
            pts = np.stack([q_rot(body_q[b, 3:7], p) + body_q[b, :3] for p in pts])
        return pts.min(0), pts.max(0)

    shapes = []
    for i, label in enumerate(shape_label):
        e = env_of(label)
        if e not in ("env_0", "global"):
            continue
        lo, hi = world_aabb(i)
        shapes.append({
            "i": i,
            "label": label.replace("/World/envs/env_0/", "").replace(
                "Robot/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/", "Robot/.../"),
            "type": geo_names.get(int(shape_type[i]), int(shape_type[i])),
            "scale": [round(float(v), 5) for v in shape_scale[i]],
            "body": int(shape_body[i]),
            "collide": bool(int(shape_flags[i]) & flag_collide),
            "flags": int(shape_flags[i]),
            "group": int(groups[i]) if groups is not None else None,
            "aabb_lo": [round(float(v), 4) for v in lo],
            "aabb_hi": [round(float(v), 4) for v in hi],
            "extent": [round(float(v), 4) for v in (hi - lo)],
        })

    out = {"env0_shapes": shapes, "num_filter_pairs": len(model.shape_collision_filter_pairs)}
    out["env0_ball_centre_w"] = _r(ball.data.root_pos_w[0])

    # Every contact touching an env-0 shape.
    if contacts is not None:
        n = int(wp.to_torch(contacts.rigid_contact_count)[0].item())
        s0 = wp.to_torch(contacts.rigid_contact_shape0)[:n].cpu().numpy()
        s1 = wp.to_torch(contacts.rigid_contact_shape1)[:n].cpu().numpy()
        force = wp.to_torch(contacts.force)[:n].cpu().numpy() if contacts.force is not None else None
        pairs = []
        for k in range(n):
            l0, l1 = shape_label[s0[k]], shape_label[s1[k]]
            if env_of(l0) != "env_0" and env_of(l1) != "env_0":
                continue
            pairs.append({
                "shape0": "/".join(l0.split("/")[-2:]),
                "shape1": "/".join(l1.split("/")[-2:]),
                "force_norm": round(float(np.linalg.norm(force[k][:3])), 3) if force is not None else None,
            })
        out["env0_contact_pairs"] = pairs
        out["rigid_contact_count_total"] = n

    # MuJoCo side (MJWarp solver): geom parameters of interesting shapes.
    mj_model = getattr(solver, "mj_model", None)
    s2g = getattr(solver, "newton_shape_to_mjc_geom", None)
    if mj_model is not None and s2g is not None:
        import mujoco
        s2g = wp.to_torch(s2g).cpu().numpy()
        geoms = []
        for sh in shapes:
            i = sh["i"]
            key = sh["label"]
            if not any(t in key for t in ("Tactile", "TargetBall", "TableTop", "Adapter", "Bowl", "Table")):
                continue
            g = int(s2g[i])
            entry = {"shape": key, "geom": g}
            if g >= 0:
                entry.update({
                    "name": mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g),
                    "type": int(mj_model.geom_type[g]),
                    "size": [round(float(v), 5) for v in mj_model.geom_size[g]],
                    "contype": int(mj_model.geom_contype[g]),
                    "conaffinity": int(mj_model.geom_conaffinity[g]),
                    "condim": int(mj_model.geom_condim[g]),
                    "margin": round(float(mj_model.geom_margin[g]), 5),
                    "gap": round(float(mj_model.geom_gap[g]), 5),
                    "solref": [round(float(v), 5) for v in mj_model.geom_solref[g]],
                    "solimp": [round(float(v), 4) for v in mj_model.geom_solimp[g]],
                    "bodyid": int(mj_model.geom_bodyid[g]),
                })
            geoms.append(entry)
        out["mujoco_geoms"] = geoms
        out["mujoco_ngeom"] = int(mj_model.ngeom)
        out["mujoco_opt"] = {
            "timestep": float(mj_model.opt.timestep),
            "o_solref": [float(v) for v in mj_model.opt.o_solref],
            "cone": int(mj_model.opt.cone),
            "impratio": float(mj_model.opt.impratio),
        }
    return out


def newton_ball_contact_trace(ball, min_force: float = 1e-3, force_fn=None) -> list[dict]:
    """Force-carrying contact pairs on env 0's ball, from Newton's raw buffer."""
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    contacts = NewtonManager.get_contacts()
    model = NewtonManager._model
    if contacts is None or contacts.force is None:
        return [{"error": "no contact buffer"}]
    wp.synchronize()
    n = int(wp.to_torch(contacts.rigid_contact_count)[0].item())
    if n == 0:
        return []
    shape_label = list(model.shape_label)
    s0 = wp.to_torch(contacts.rigid_contact_shape0)[:n].cpu().numpy()
    s1 = wp.to_torch(contacts.rigid_contact_shape1)[:n].cpu().numpy()
    if force_fn is not None:
        force = force_fn(contacts, n).cpu().numpy()[:, :3]
    else:
        force = wp.to_torch(contacts.force)[:n].cpu().numpy()[:, :3]
    normal = wp.to_torch(contacts.rigid_contact_normal)[:n].cpu().numpy()
    out = []
    for k in range(n):
        l0, l1 = shape_label[s0[k]], shape_label[s1[k]]
        if "/env_0/" not in l0 and "/env_0/" not in l1:
            continue
        if "TargetBall" not in l0 and "TargetBall" not in l1:
            continue
        fn = float(np.linalg.norm(force[k]))
        if fn < min_force:
            continue
        other = l1 if "TargetBall" in l0 else l0
        out.append({
            "other": "/".join(other.split("/")[-2:]),
            "f": [round(float(v), 2) for v in force[k]],
            "fn": round(fn, 2),
            "n": [round(float(v), 2) for v in normal[k]],
        })
    return out


def pad_geometry_gap(ball, radius: float) -> dict:
    """Env-0 gap between each fingertip pad hull and the ball surface (Newton).

    Uses the actual collision hull Newton simulates (vertices, shape
    transform, body pose) so a placement error of the collider relative to
    the link shows up directly.  Negative gap = penetration.
    """
    import numpy as np
    import warp as wp
    from scipy.spatial import ConvexHull
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager._model
    state = NewtonManager._state_0
    wp.synchronize()
    labels = list(model.shape_label)
    shape_body = wp.to_torch(model.shape_body).cpu().numpy()
    shape_xf = wp.to_torch(model.shape_transform).cpu().numpy()
    shape_scale = wp.to_torch(model.shape_scale).cpu().numpy()
    body_q = wp.to_torch(state.body_q).cpu().numpy()
    centre = ball.data.root_pos_w[0].cpu().numpy().astype(np.float64)

    def q_rot(q, v):
        q = np.asarray(q, dtype=np.float64)
        xyz, w = q[:3], q[3]
        c = 2.0 * np.cross(xyz, v)
        return v + w * c + np.cross(xyz, c)

    out = {}
    for i, label in enumerate(labels):
        if "/env_0/" not in label or "TaxelAlignedCollision" not in label:
            continue
        src = model.shape_source[i]
        if src is None:
            continue
        verts = np.asarray(src.vertices, dtype=np.float64) * shape_scale[i]
        verts = np.stack([q_rot(shape_xf[i, 3:7], v) + shape_xf[i, :3] for v in verts])
        b = int(shape_body[i])
        if b >= 0:
            verts = np.stack([q_rot(body_q[b, 3:7], v) + body_q[b, :3] for v in verts])
        hull = ConvexHull(verts)
        plane_dist = hull.equations[:, :3] @ centre + hull.equations[:, 3]
        face_gap = float(plane_dist.max()) - radius
        vert_gap = float(np.linalg.norm(verts - centre, axis=1).min()) - radius
        out[label.split("/")[-1]] = {
            "vertices": int(len(verts)),
            "face_gap_mm": round(face_gap * 1e3, 3),
            "vertex_gap_mm": round(vert_gap * 1e3, 3),
            "hull_centroid": [round(float(v), 5) for v in verts.mean(0)],
            "body_pos": [round(float(v), 5) for v in body_q[b, :3]] if b >= 0 else None,
        }
    return out


def hydro_surface_summary(ball) -> dict:
    """Contact-patch statistics from the hydroelastic iso-surface (env 0)."""
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    pipeline = getattr(NewtonManager, "_collision_pipeline", None)
    hydro = getattr(pipeline, "hydroelastic_sdf", None) if pipeline is not None else None
    if hydro is None:
        return {"error": "no hydroelastic generator active"}
    surface = hydro.get_contact_surface()
    if surface is None:
        return {"error": "output_contact_surface disabled"}
    wp.synchronize()
    n = int(surface.face_contact_count.numpy()[0])
    out = {"face_contacts_total": n, "max_faces": int(surface.max_num_face_contacts), "env0_pairs": {}}
    if n == 0:
        return out
    pts = surface.contact_surface_point.numpy()[: 3 * n].reshape(n, 3, 3)
    depth = surface.contact_surface_depth.numpy()[:n]
    pairs = surface.contact_surface_shape_pair.numpy()[:n]
    labels = list(NewtonManager._model.shape_label)
    ball_c = ball.data.root_pos_w[0].cpu().numpy()
    area = 0.5 * np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=-1)
    for key in np.unique(pairs, axis=0):
        la, lb = labels[int(key[0])], labels[int(key[1])]
        if "/env_0/" not in la and "/env_0/" not in lb:
            continue
        sel = np.all(pairs == key, axis=1)
        p = pts[sel].reshape(-1, 3)
        name = "/".join((lb if "TargetBall" in la else la).split("/")[-2:])
        out["env0_pairs"][name] = {
            "faces": int(sel.sum()),
            "area_mm2": round(float(area[sel].sum() * 1e6), 2),
            "depth_mean_mm": round(float(depth[sel].mean() * 1e3), 3),
            "depth_max_mm": round(float(depth[sel].max() * 1e3), 3),
            "extent_mm": [round(float(v), 2) for v in (p.max(0) - p.min(0)) * 1e3],
            "centroid_to_ball_centre_mm": round(float(np.linalg.norm(p.mean(0) - ball_c) * 1e3), 2),
        }
    return out


def main() -> int:
    dump = json.loads(args.waypoints.read_text())
    if args.close_deg is not None:
        old_close = dump["close_angle_deg"]
        dump["close_angle_deg"] = args.close_deg
        for phase in dump["phase_plan"]:
            if phase["gripper_deg"] == old_close:
                phase["gripper_deg"] = args.close_deg

    cfg = SO101TactileEnvCfg()
    cfg.sim.device = args.device
    cfg.sim.physics = make_physics_cfg(args.physics)
    cfg.scene = BallPickPlaceSceneCfg(
        num_envs=args.num_envs, env_spacing=args.env_spacing,
        replicate_physics=True)
    if args.init_rot is not None:
        cfg.scene.robot.init_state.rot = tuple(args.init_rot)
    if args.asset is not None:
        asset_path = PROJECT_ROOT / "assets" / "isaac_lab" / args.asset
        if not asset_path.exists():
            raise SystemExit(f"asset not found: {asset_path}")
        cfg.scene.robot.spawn.usd_path = str(asset_path)
    if args.render_check:
        cfg.scene.overview_camera.width = 960
        cfg.scene.overview_camera.height = 540
    else:
        cfg.scene.overview_camera = None
    cfg.episode_length_s = 120.0
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    if args.physics == "physx":
        # Same solver settings the calibrated replay uses.
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 4
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
    cfg.scene.ball.spawn.rigid_props.max_depenetration_velocity = None
    if args.kinematic_ball:
        cfg.scene.ball.spawn.rigid_props.kinematic_enabled = True
    if args.ball_mass is not None:
        cfg.scene.ball.spawn.mass_props.mass = float(args.ball_mass)
    if args.gripper_effort is not None:
        from isaaclab.actuators import ImplicitActuatorCfg

        base = cfg.scene.robot.actuators["all_joints"]
        cfg.scene.robot.actuators = {
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[n for n in SO101_JOINT_NAMES if n != "gripper"],
                effort_limit_sim=base.effort_limit_sim,
                stiffness=base.stiffness, damping=base.damping),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=float(args.gripper_effort),
                stiffness=base.stiffness, damping=base.damping),
        }
    if args.dps_sensors:
        # Scene keeps its DPS2015ContactSensorCfg pair and the env keeps the
        # tactile observation/reward terms that consume them.
        pass
    elif args.dump_contacts:
        cfg.scene.tactile_fixed, cfg.scene.tactile_moving = plain_contact_sensor_cfgs()
        cfg.observations, cfg.rewards = bare_observation_cfgs()
    else:
        cfg.scene.tactile_fixed = None
        cfg.scene.tactile_moving = None
        cfg.observations, cfg.rewards = bare_observation_cfgs()

    if args.physics.startswith("newton"):
        from so101_isaac_lab import newton_hooks

        if any(v is not None for v in (args.newton_ke, args.newton_kd, args.newton_solimp)):
            newton_hooks.register_contact_tuning(
                ke=args.newton_ke, kd=args.newton_kd,
                solimp=tuple(args.newton_solimp) if args.newton_solimp else None,
                shape_regex=args.newton_tune_regex)
        if args.no_adapter:
            newton_hooks.register_disable_collision(shape_regex="AdapterCollision")
        if args.joint_gain_scale is not None:
            newton_hooks.register_joint_gain_scale(
                ke_scale=args.joint_gain_scale[0], kd_scale=args.joint_gain_scale[1])
        if args.coacd_threshold is not None:
            newton_hooks.register_redecomposition(
                shape_regex=args.coacd_regex, threshold=args.coacd_threshold)
        if args.hydro:
            if args.physics not in ("newton_hydro", "newton_featherstone_hydro", "newton_xpbd_hydro"):
                raise SystemExit("--hydro needs --physics newton_hydro (Newton pipeline "
                                 "with the hydroelastic generator enabled)")
            newton_hooks.register_hydroelastic(
                shape_regex=args.hydro_regex, kh=args.hydro_kh,
                mesh_sdf_max_resolution=args.hydro_mesh_res,
                primitive_sdf_max_resolution=args.hydro_ball_res,
                compliant_layer_m=args.hydro_layer_mm * 1e-3)

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    scene = env.scene
    robot = scene["robot"]
    ball = scene["ball"]
    origins = scene.env_origins
    device = env.device
    count = scene.num_envs

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    print(f"[{args.physics}] joints={joint_names}", flush=True)
    print(f"[{args.physics}] bodies={body_names}", flush=True)

    # Action term order is SO101_JOINT_NAMES (preserve_order=True); the
    # default offset is the robot's default joint position in that order.
    robot_idx = [joint_names.index(n) for n in SO101_JOINT_NAMES]
    default_pos = robot.data.default_joint_pos[:, robot_idx]
    scale = cfg.actions.joint_pos.scale

    def act(target_row: np.ndarray) -> torch.Tensor:
        target = torch.tensor(target_row, dtype=torch.float32, device=device).repeat(count, 1)
        return (target - default_pos) / scale, target

    ball_start = (ball.data.root_pos_w - origins).clone()
    ball_z_trace = []
    t_wall = time.perf_counter()
    if args.probe_pose is not None:
        if args.probe_pose not in dump["solutions_rad"]:
            raise SystemExit(f"unknown waypoint {args.probe_pose!r}; have {sorted(dump['solutions_rad'])}")
        gripper_deg = dump["open_gripper_deg"] if args.gripper_deg is None else args.gripper_deg
        row = np.concatenate([dump["solutions_rad"][args.probe_pose], [np.radians(gripper_deg)]])
        action, target = act(row)
        steps = round(args.hold_s * PHYSICS_HZ)
        for step in range(steps):
            env.step(action)
            if step % 60 == 0:
                ball_z_trace.append(float((ball.data.root_pos_w[0, 2] - origins[0, 2]).item()))
        mode = {"probe_pose": args.probe_pose, "gripper_deg": gripper_deg, "hold_s": args.hold_s}
    else:
        timeline, phase_ends = build_timeline(dump)
        ends = dict(phase_ends)
        if args.until not in ends:
            raise SystemExit(f"unknown phase {args.until!r}; have {list(ends)}")
        steps = ends[args.until] + 1
        phase_of_step = {}
        last = 0
        for label, end in phase_ends:
            for s in range(last, end + 1):
                phase_of_step[s] = label
            last = end + 1
        phase_ball = {}
        trace_from = ends.get("descend", 0) + 1 if args.trace_contacts else None
        contact_trace = []
        for step in range(steps):
            action, target = act(timeline[step])
            env.step(action)
            if step % 60 == 0:
                ball_z_trace.append(float((ball.data.root_pos_w[0, 2] - origins[0, 2]).item()))
            if trace_from is not None and step >= trace_from and (step - trace_from) % 12 == 0 \
                    and args.physics.startswith("newton"):
                bp = (ball.data.root_pos_w[0] - origins[0]).cpu().numpy()
                force_fn = getattr(scene["tactile_fixed"], "_contact_forces_on_shape0", None) \
                    if args.dps_sensors else None
                pairs = newton_ball_contact_trace(ball, force_fn=force_fn)
                rec = {"t": round(step / PHYSICS_HZ, 3), "phase": phase_of_step[step],
                       "ball": [round(float(v), 4) for v in bp],
                       "grip": round(float(robot.data.joint_pos[0, robot_idx[5]]), 4),
                       "pairs": pairs}
                if args.dps_sensors:
                    tac = tactile_summary(scene, args.taxel_threshold_n)
                    rec["taxels"] = {k: {"active_env0": v["active_taxels_per_env"][0],
                                         "sum_n_env0": v["taxel_sum_norm_per_env"][0],
                                         "max_taxel_n_env0": v["max_taxel_n_env0"]}
                                     for k, v in tac.items()}
                contact_trace.append(rec)
                grouped: dict[str, list] = {}
                for p in pairs:
                    grouped.setdefault(p.get("other", "?"), []).append(p)
                parts = []
                for other, items in grouped.items():
                    total = [sum(p["f"][k] for p in items) for k in range(3)]
                    parts.append(f"{other.split('/')[-1][:34]} x{len(items)} "
                                 f"F=({total[0]:.2f},{total[1]:.2f},{total[2]:.2f})")
                tax = ""
                if "taxels" in rec:
                    tf, tm = rec["taxels"]["tactile_fixed"], rec["taxels"]["tactile_moving"]
                    tax = (f" || taxels fixed {tf['active_env0']}@{tf['sum_n_env0']:.2f}N "
                           f"moving {tm['active_env0']}@{tm['sum_n_env0']:.2f}N")
                print(f"[trace] t={rec['t']:.3f} {rec['phase']:9s} ball={rec['ball']} grip={rec['grip']} "
                      + " | ".join(parts) + tax, flush=True)
            if step in ends.values():
                label = phase_of_step[step]
                bp = ball.data.root_pos_w - origins
                phase_ball[label] = {
                    "ball_env0": _r(bp[0]),
                    "ball_z_min_over_envs": round(float(bp[:, 2].min()), 4),
                    "ball_z_max_over_envs": round(float(bp[:, 2].max()), 4),
                    "gripper_rad_env0": round(float(robot.data.joint_pos[0, robot_idx[5]]), 4),
                }
                if args.dump_contacts or args.dps_sensors:
                    phase_ball[label]["sensors"] = {
                        k: {"net_force_norm_per_env": v["net_force_norm_per_env"],
                            "ball_force_norm_per_env": v.get("ball_force_norm_per_env")}
                        for k, v in sensor_summary(scene).items()}
                if args.dps_sensors:
                    phase_ball[label]["tactile"] = tactile_summary(scene, args.taxel_threshold_n)
                    print(f"[{args.physics}] tactile @ {label}: "
                          + json.dumps(phase_ball[label]["tactile"]), flush=True)
                if args.hydro:
                    phase_ball[label]["hydro_surface"] = hydro_surface_summary(ball)
                    print(f"[{args.physics}] hydro surface @ {label}: "
                          + json.dumps(phase_ball[label]["hydro_surface"]), flush=True)
                print(f"[{args.physics}] end of {label}: ball_env0={phase_ball[label]['ball_env0']}",
                      flush=True)
        mode = {"until": args.until, "steps": steps, "sim_s": steps / PHYSICS_HZ,
                "phase_end_ball": phase_ball}
        if args.trace_contacts:
            mode["contact_trace"] = contact_trace
    wall = time.perf_counter() - t_wall

    joints = robot.data.joint_pos[:, robot_idx]
    joint_err = (joints - target).abs()
    body_pos = robot.data.body_pos_w - origins[:, None, :]
    body_quat = robot.data.body_quat_w
    grip_i = body_names.index("gripper_link")
    jaw_i = body_names.index("moving_jaw_so101_v1_link")
    base_i = body_names.index("base_link")
    ball_pos = ball.data.root_pos_w - origins
    resting_z = TABLE_TOP_Z + TARGET_BALL_RADIUS

    ref = {}
    if args.probe_pose == "retreat":
        # The standalone dump records these frames at the end of its episode,
        # i.e. resting in the retreat waypoint with the jaw open (row-major
        # 4x4, translation in the last row).
        ref = {
            "gripper_link": dump["gripper_link_world"][3][:3],
            "moving_jaw": dump["moving_jaw_world"][3][:3],
        }

    def rows(t: torch.Tensor) -> list[list[float]]:
        return [[round(float(v), 5) for v in r] for r in t.cpu().numpy()]

    report = {
        "physics": args.physics,
        "num_envs": count,
        "mode": mode,
        "wall_s": round(wall, 1),
        "steps_per_s": round(steps * count / wall, 1),
        "joint_names_robot_order": joint_names,
        "body_names": body_names,
        "target_rad": _r(target[0], 5),
        "joint_pos_env0": _r(joints[0], 5),
        "joint_err_max_rad": round(float(joint_err.max().item()), 5),
        "joint_err_max_per_joint_rad": _r(joint_err.max(dim=0).values, 5),
        "gripper_link_pos": rows(body_pos[:, grip_i]),
        "gripper_link_quat_env0": _r(body_quat[0, grip_i], 5),
        "moving_jaw_pos": rows(body_pos[:, jaw_i]),
        "moving_jaw_quat_env0": _r(body_quat[0, jaw_i], 5),
        "base_link_pos_env0": _r(body_pos[0, base_i], 5),
        "base_link_quat_env0": _r(body_quat[0, base_i], 5),
        "ball_start_env0": _r(ball_start[0], 5),
        "ball_pos": rows(ball_pos),
        "ball_lift_m_per_env": _r(ball_pos[:, 2] - ball_start[:, 2], 4),
        "ball_xy_drift_mm_per_env": _r(torch.linalg.norm(ball_pos[:, :2] - ball_start[:, :2], dim=-1) * 1e3, 1),
        "ball_resting_z": round(resting_z, 5),
        "ball_z_trace_env0": [round(v, 5) for v in ball_z_trace],
        "env_spread_gripper_mm": round(
            float((body_pos[:, grip_i] - body_pos[0:1, grip_i]).norm(dim=-1).max().item() * 1e3), 3),
        "reference_standalone": ref,
    }
    if ref:
        g = body_pos[0, grip_i].cpu().numpy()
        j = body_pos[0, jaw_i].cpu().numpy()
        report["gripper_vs_standalone_mm"] = round(float(np.linalg.norm(g - np.asarray(ref["gripper_link"])) * 1e3), 3)
        report["jaw_vs_standalone_mm"] = round(float(np.linalg.norm(j - np.asarray(ref["moving_jaw"])) * 1e3), 3)

    if args.dump_contacts or args.dps_sensors:
        report["sensors"] = sensor_summary(scene)
        if args.physics.startswith("newton"):
            report["contacts"] = dump_newton_contacts(ball)
    if args.dps_sensors:
        report["tactile"] = tactile_summary(scene, args.taxel_threshold_n)
    if args.hydro:
        report["hydro_surface"] = hydro_surface_summary(ball)
    if args.dump_bodies and args.physics.startswith("newton"):
        import warp as wp
        from newton import eval_fk
        from isaaclab_newton.physics import NewtonManager

        model, state = NewtonManager._model, NewtonManager._state_0
        scratch = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, scratch)
        wp.synchronize()
        bq = wp.to_torch(state.body_q).cpu().numpy()
        fq = wp.to_torch(scratch.body_q).cpu().numpy()
        labels = list(model.body_label)
        rows = []
        for i, lab in enumerate(labels):
            if "/env_0/" not in lab:
                continue
            rows.append({"i": i, "body": lab.split("/")[-1],
                         "state_pos": [round(float(v), 4) for v in bq[i, :3]],
                         "state_quat": [round(float(v), 3) for v in bq[i, 3:7]],
                         "fk_pos": [round(float(v), 4) for v in fq[i, :3]]})
        report["bodies"] = rows
        for r_ in rows:
            print("[bodies]", json.dumps(r_), flush=True)

        # Joint frames as the solver sees them.  Kamino's ``from_newton``
        # rewrites model.joint_X_p / joint_X_c in place (it absorbs
        # non-identity child frames into rotated body frames), so dumping
        # them under two solvers shows exactly what it changed.
        jx_p = wp.to_torch(model.joint_X_p).cpu().numpy()
        jx_c = wp.to_torch(model.joint_X_c).cpu().numpy()
        jtype = wp.to_torch(model.joint_type).cpu().numpy()
        jpar = wp.to_torch(model.joint_parent).cpu().numpy()
        jchild = wp.to_torch(model.joint_child).cpu().numpy()
        jrows = []
        for j in range(min(len(jtype), 16)):
            c = int(jchild[j])
            if c < 0 or c >= len(labels) or "/env_0/" not in labels[c]:
                continue
            jrows.append({
                "j": j, "type": int(jtype[j]),
                "parent": labels[int(jpar[j])].split("/")[-1] if int(jpar[j]) >= 0 else "world",
                "child": labels[c].split("/")[-1],
                "X_p_pos": [round(float(v), 4) for v in jx_p[j, :3]],
                "X_p_quat": [round(float(v), 3) for v in jx_p[j, 3:7]],
                "X_c_pos": [round(float(v), 4) for v in jx_c[j, :3]],
                "X_c_quat": [round(float(v), 3) for v in jx_c[j, 3:7]],
            })
        report["joints"] = jrows
        for r_ in jrows:
            print("[joints]", json.dumps(r_), flush=True)

    if args.dump_control and args.physics.startswith("newton"):
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        model, control, state = NewtonManager._model, NewtonManager._control, NewtonManager._state_0
        wp.synchronize()

        def head(arr, n=8):
            if arr is None:
                return None
            v = wp.to_torch(arr).flatten()[:n].cpu().tolist()
            return [round(float(x), 4) for x in v]

        ctrl = {
            "joint_dof_count": int(getattr(model, "joint_dof_count", -1)),
            "joint_target_ke": head(getattr(model, "joint_target_ke", None)),
            "joint_target_kd": head(getattr(model, "joint_target_kd", None)),
            "joint_dof_mode": head(getattr(model, "joint_dof_mode", None)),
            "joint_effort_limit": head(getattr(model, "joint_effort_limit", None)),
            "joint_armature": head(getattr(model, "joint_armature", None)),
            "control_joint_target_pos": head(getattr(control, "joint_target_pos", None)),
            "control_joint_target_vel": head(getattr(control, "joint_target_vel", None)),
            "control_joint_f": head(getattr(control, "joint_f", None)),
            "state_joint_q": head(getattr(state, "joint_q", None)),
            "joint_type": head(getattr(model, "joint_type", None)),
        }
        report["control"] = ctrl
        print("[control]", json.dumps(ctrl), flush=True)

    if args.physics.startswith("newton"):
        try:
            report["pad_geometry"] = pad_geometry_gap(ball, TARGET_BALL_RADIUS)
            print("[pad_geometry]", json.dumps(report["pad_geometry"]), flush=True)
        except Exception as exc:  # diagnostics only
            report["pad_geometry"] = {"error": repr(exc)}
    if args.dump_shapes and args.physics.startswith("newton"):
        report["shapes"] = dump_newton_shapes(ball)

    if args.render_check:
        cam = scene["overview_camera"]
        eye = origins + torch.tensor([0.9, -0.7, 0.75], device=device)
        look = origins + torch.tensor([0.15, 0.25, 0.05], device=device)
        cam.set_world_poses_from_view(eye, look)
        # RTX streams static meshes in lazily; hold the last command and keep
        # rendering for ~3 s of sim time before judging the frame.
        hold_action = action
        for _ in range(round(3.0 * PHYSICS_HZ)):
            env.step(hold_action)
        rgb = cam.data.output["rgb"][0].cpu().numpy()
        import cv2
        out_png = PROJECT_ROOT / "tactile_logs" / "backend_probe" / \
            f"render_{args.physics}_{args.probe_pose or args.until}.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR))
        report["render_png"] = str(out_png)
        report["render_mean_rgb"] = [round(float(v), 2) for v in rgb[..., :3].reshape(-1, 3).mean(0)]
        report["render_std_rgb"] = round(float(rgb[..., :3].std()), 2)

    out = args.output or (PROJECT_ROOT / "tactile_logs" / "backend_probe" /
                          f"probe_{args.physics}_{args.probe_pose or args.until}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("gripper_link_pos", "moving_jaw_pos", "ball_pos", "contacts", "shapes")},
                     indent=1), flush=True)
    if "contacts" in report:
        print("[contacts]", json.dumps(report["contacts"], indent=1), flush=True)
    if "shapes" in report:
        print("[shapes]", json.dumps(report["shapes"], indent=None), flush=True)
    print(f"[probe] env0 gripper={report['gripper_link_pos'][0]} "
          f"jaw={report['moving_jaw_pos'][0]} ball={report['ball_pos'][0]}", flush=True)
    print(f"[probe] wrote {out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
