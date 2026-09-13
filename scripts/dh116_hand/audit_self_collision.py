"""Exercise hand gestures under physics and report self-contact separation.

This diagnostic never teleports joints after initialization. Missing contacts
with self-collisions disabled are not evidence that visual meshes do not overlap.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd_path", type=Path)
parser.add_argument("--robot", choices=("hand", "arm"), default="hand")
parser.add_argument("--self_collisions", choices=("on", "off"), default="on")
parser.add_argument("--position_iterations", type=int, default=None,
                    help="override the robot profile (hand=32, arm=64)")
parser.add_argument("--velocity_iterations", type=int, default=8)
parser.add_argument("--dt", type=float, default=1/120)
parser.add_argument("--contact_last", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--max_depenetration_velocity", type=float, default=None)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--record", type=Path)
parser.add_argument("--seconds_per_gesture", type=float, default=3.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = args.record is not None
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_physx.physics import PhysxCfg
from pxr import PhysxSchema, UsdPhysics, Usd
from dh116_hand_lab.hand_cfg import DH116_HAND_CFG
from xarm5_dh116_lab.robot_cfg import XARM5_DH116_CFG, HAND_JOINTS
from xarm5_dh116_lab.gestures import hand_targets, quintic


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=args.dt, device=args.device,
        physics=PhysxCfg(solve_articulation_contact_last=args.contact_last)))
    cfg = (DH116_HAND_CFG if args.robot == "hand" else XARM5_DH116_CFG).copy()
    cfg.prim_path = "/World/Robot"
    if args.usd_path:
        cfg.spawn.usd_path = str(args.usd_path.resolve())
    position_iterations = (args.position_iterations if args.position_iterations is not None
                           else cfg.spawn.articulation_props.solver_position_iteration_count)
    cfg.spawn.activate_contact_sensors = True
    cfg.spawn.articulation_props.enabled_self_collisions = args.self_collisions == "on"
    cfg.spawn.articulation_props.solver_position_iteration_count = position_iterations
    cfg.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
    if args.max_depenetration_velocity is not None:
        cfg.spawn.rigid_props.max_depenetration_velocity = args.max_depenetration_velocity
    robot = Articulation(cfg)
    camera = None
    if args.record:
        camera_cfg = CameraCfg(prim_path="/World/Camera", height=720, width=1280,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=30., horizontal_aperture=20.955))
        camera = Camera(camera_cfg)
        light = sim_utils.DomeLightCfg(intensity=1500., color=(.75,.75,.75), visible_in_primary_ray=False)
        light.func("/World/Light", light)
    active = {"gesture": "initial", "time": 0.}
    pairs = {}
    traces = []

    # Importer uses nested rigid-body prims with reset transforms. The stock
    # activation helper stops at the first rigid body, so enable every link.
    for prim in sim.stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    sim.set_setting("/physics/disableContactProcessing", False)
    sim.reset()
    sim.set_setting("/physics/disableContactProcessing", False)
    from isaaclab_physx.physics.physx_manager import PhysxManager as SimulationManager
    colliders = [p for p in Usd.PrimRange(sim.stage.GetPrimAtPath("/World/Robot"), Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.CollisionAPI)]
    collider_paths = [str(p.GetPath()) for p in colliders]
    collider_body = {}
    collider_config = []
    for p in colliders:
        parent = p
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        collider_body[str(p.GetPath())] = parent.GetName()
        api = PhysxSchema.PhysxCollisionAPI(p)
        collider_config.append({"path": str(p.GetPath()), "body": parent.GetName(),
                               "contact_offset": api.GetContactOffsetAttr().Get(),
                               "rest_offset": api.GetRestOffsetAttr().Get()})
    body_paths = [str(p.GetPath()) for p in sim.stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    contact_view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
        body_paths, filter_patterns=[body_paths for _ in body_paths], max_contact_data_count=16384)
    if contact_view.sensor_count != len(body_paths) or contact_view.filter_count != len(body_paths):
        raise RuntimeError("Contact view did not resolve all expected bodies")
    print(f"Contact audit: {contact_view.sensor_count} bodies, {len(colliders)} colliders", flush=True)
    names = list(robot.joint_names)
    indices = [names.index(n) for n in HAND_JOINTS]
    targets = robot.data.default_joint_pos.torch.clone()
    targets[0, indices] = torch.tensor(hand_targets("open"), device=args.device)
    robot.write_joint_position_to_sim_index(position=targets)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(targets))
    robot.set_joint_position_target_index(target=targets)
    sim.forward()
    writer = None
    if camera:
        base_index = robot.body_names.index("hand_base_link")
        base = robot.data.body_pos_w.torch[0, base_index].cpu().numpy()
        if args.robot == "hand":
            look = base + np.array([.08, 0., 0.])
            eye = look + np.array([.32, .36, .28])
        else:
            look = base + np.array([0., 0., .07])
            eye = look + np.array([.30, .24, .15])
        camera.set_world_poses_from_view(eyes=torch.tensor(eye[None],device=args.device,dtype=torch.float32),
                                         targets=torch.tensor(look[None],device=args.device,dtype=torch.float32))
        for _ in range(30):
            sim.render()
        import cv2
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280,720))
        if not writer.isOpened():
            raise RuntimeError("Cannot open video writer")
    sequence = ["open", "rock", "open", "scissors", "one", "thumbs_up", "open"]
    phases = []
    previous = targets[0, indices].cpu().numpy()
    frame = 0
    for segment, gesture in enumerate(sequence):
        label = f"{segment}_{gesture}"
        goal = hand_targets(gesture)
        q, qd = [], []
        for k in range(round(args.seconds_per_gesture / args.dt)):
            active.update(gesture=label, time=k*args.dt)
            desired = previous + quintic(k*args.dt / 1.) * (goal-previous)
            targets[0, indices] = torch.tensor(desired, device=args.device)
            robot.set_joint_position_target_index(target=targets)
            robot.write_data_to_sim()
            sim.step(render=args.record is not None)
            robot.update(args.dt)
            _, _, _, separation, counts, starts = contact_view.get_contact_data(args.dt)
            sep, cnt, begin = separation.numpy().reshape(-1), counts.numpy(), starts.numpy()
            if int(cnt.sum()) >= contact_view.max_contact_data_count:
                raise RuntimeError("Contact buffer may be saturated; increase capacity")
            for row, col in np.argwhere(cnt > 0):
                sensor = str(contact_view.sensor_paths[row]).split("/")[-1]
                other = body_paths[col].split("/")[-1]
                key = (label, " / ".join(sorted([sensor, other])))
                vals = sep[int(begin[row,col]):int(begin[row,col]+cnt[row,col])]
                entry = pairs.setdefault(key, {"samples":0,"min_separation_m":0.,"settled_min_separation_m":0.})
                entry["samples"] += len(vals)
                entry["min_separation_m"] = min(entry["min_separation_m"],float(vals.min()))
                if active["time"] >= args.seconds_per_gesture-.5:
                    entry["settled_min_separation_m"] = min(entry["settled_min_separation_m"],float(vals.min()))
            positions = robot.data.joint_pos.torch[0].cpu().numpy().copy()
            velocities = robot.data.joint_vel.torch[0].cpu().numpy().copy()
            if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                raise RuntimeError(f"Non-finite state at {label} step {k}")
            q.append(positions[indices]); qd.append(velocities[indices])
            if k % max(1,round(.1/args.dt)) == 0:
                traces.append({"phase": label, "t": k*args.dt,
                    "body_pos": robot.data.body_pos_w.torch[0].cpu().numpy().tolist(),
                    "body_quat": robot.data.body_quat_w.torch[0].cpu().numpy().tolist()})
            if writer and frame % max(1,round(1/(30*args.dt))) == 0:
                camera.update(args.dt, force_recompute=True)
                pixels = camera.data.output["rgb"][0,:,:,:3].cpu().numpy().astype(np.uint8)
                pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                text = f"{gesture}   self-collision {args.self_collisions}"
                cv2.putText(pixels,text,(24,685),cv2.FONT_HERSHEY_SIMPLEX,.85,(0,0,0),4,cv2.LINE_AA)
                cv2.putText(pixels,text,(24,685),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2,cv2.LINE_AA)
                writer.write(pixels)
            frame += 1
        q, qd = np.asarray(q), np.asarray(qd)
        tail = round(.5/args.dt)
        limits = robot.data.joint_limits.torch[0, indices].cpu().numpy()
        violation = np.maximum(np.maximum(limits[:,0] - q, q - limits[:,1]), 0.)
        phases.append({"phase": label, "max_joint_limit_violation_deg":float(np.degrees(violation).max()), "max_joint_speed_rad_s":float(np.abs(qd).max()),
            "settled_max_joint_speed_rad_s":float(np.abs(qd[-tail:]).max()),
            "max_tracking_error_deg":float(np.degrees(np.abs(q[-tail:]-goal)).max()),
            "settled_joint_std_deg":float(np.degrees(q[-tail:].std(axis=0)).max()),
            "final_joint_deg": np.degrees(q[-1]).tolist()})
        previous = goal
    if writer:
        writer.release()
    report = {"schema_version":1, "asset_path": cfg.spawn.usd_path,
        "asset_sha256":hashlib.sha256(Path(cfg.spawn.usd_path).read_bytes()).hexdigest(),
        "robot":args.robot, "self_collisions":args.self_collisions,
        "dt": args.dt, "position_iterations":position_iterations,
        "velocity_iterations":args.velocity_iterations, "contact_last":args.contact_last,
        "max_depenetration_velocity":cfg.spawn.rigid_props.max_depenetration_velocity,
        "collider_config":collider_config, "joint_names":HAND_JOINTS, "body_names":list(robot.body_names), "phases":phases,
        "contacts":[{"phase":g,"pair":p,**v} for (g,p),v in sorted(pairs.items())]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    depth = -1000*min([v["min_separation_m"] for v in pairs.values()]+[0.])
    report["acceptance"] = {"sampled_max_penetration_mm":depth,
        "max_joint_limit_violation_deg":max(p["max_joint_limit_violation_deg"] for p in phases),
        "max_settled_joint_std_deg":max(p["settled_joint_std_deg"] for p in phases),
        "scope":"7 deterministic gesture phases, sampled at each physics step; no real-hand calibration"}
    report["acceptance"]["passed"] = (args.self_collisions=="on" and bool(pairs) and depth < .5
        and report["acceptance"]["max_joint_limit_violation_deg"] < 1.
        and report["acceptance"]["max_settled_joint_std_deg"] < .5)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    args.output.with_suffix(".poses.json").write_text(json.dumps(traces)+"\n")
    print(json.dumps({"contact_pairs":len(pairs),"worst_depth_mm":
        -1000*min([v["min_separation_m"] for v in pairs.values()]+[0.]),"phases":phases},indent=2),flush=True)
    sim.stop()
    sim.clear_instance()
    return 0 if report["acceptance"]["passed"] or args.self_collisions == "off" else 2


if __name__ == "__main__":
    status = 0
    try:
        status = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        status = 1
    finally:
        app.close()
    raise SystemExit(status)
