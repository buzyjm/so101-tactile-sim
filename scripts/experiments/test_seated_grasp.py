"""Validate the seated grasp: descend onto the ball, then close gently.

Three changes from the swept grasp, each from an offline geometry result:
  * the grasp pose is solved so the ball lands on the fixed pad's centroid
    rather than wherever a lateral sweep shoves it;
  * the approach descends vertically, which is collision-free at 15 deg tilt
    (min clearance 29.96 mm) whereas the lateral sweep pushed the ball out;
  * the close command stops just past first contact (46.4 deg) instead of
    driving to 5 deg, which was 41 deg of position error and pinned the servo
    at its torque limit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--usd", type=Path, default=ROOT / "lab_scene_task.usda")
    p.add_argument("--tilt", type=float, default=15.0)
    p.add_argument("--approach-rise", type=float, default=0.06)
    p.add_argument(
        "--close-deg", type=float, nargs="*",
        default=[44.0, 42.0, 40.0, 36.0, 30.0, 5.0],
        help="Gripper close targets to try; first contact is near 46.4 deg.",
    )
    p.add_argument(
        "--compliant-stiffness", type=float, nargs="*", default=[None],
        help="physxMaterial:compliantContactStiffness on the ball. A rigid "
             "sphere loads 4-5 taxels with a sharp peak; the recorded plush "
             "ball loads 21-34 at a near-uniform 0.46-0.58 of peak.")
    p.add_argument("--compliant-damping", type=float, default=0.0)
    p.add_argument(
        "--contact-offset", type=float, nargs="*", default=[None],
        help="physxCollision:contactOffset on the ball and both pads. The "
             "scene authors none, so PhysX defaults apply; taxel pitch is "
             "about 2 mm, so this is worth measuring rather than assuming.")
    p.add_argument(
        "--rest-offset", type=float, default=0.0)
    p.add_argument(
        "--friction", type=float, nargs="*", default=[None],
        help="Static friction for the ball and both jaws; the scene ships "
             "with no physics material at all, so the default is PhysX's.",
    )
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "tactile_logs" / "seated_grasp_test.json",
    )
    return p.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "active_gpu": ARGS.gpu,
                     "physics_gpu": ARGS.gpu, "multi_gpu": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from tactile_sensor import DPS2015EliteMvp
from tactile_taxels import load_taxel_positions_m
from scripts.tasks.grasp_geometry import RadialToolGeometry
from scene_config import (
    GRIPPER_CONTACT_PRIM_PATHS,
    GRIPPER_LINK_PRIM_PATH, PROJECT_ROOT, ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG, TABLE_TOP_Z, TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS, TARGET_BALL_RADIUS,
    TARGET_BALL_XY_TASK_READY_PLACEHOLDER,
)

JOINT_ROOT = "/World/Robot/Physics"
BALL_PATH = "/World/TargetBall"
IK_FRAME = "gripper_frame_link"
GRIPPER_FRAME_PRIM = GRIPPER_LINK_PRIM_PATH + "/gripper_frame_link"
HZ = TACTILE_MVP_PHYSICS_FREQUENCY_HZ
OPEN_DEG = 70.0
BALL_START = np.array([TARGET_BALL_XY_TASK_READY_PLACEHOLDER[0],
                       TARGET_BALL_XY_TASK_READY_PLACEHOLDER[1],
                       TABLE_TOP_Z + TARGET_BALL_RADIUS])
TAXELS = np.asarray(load_taxel_positions_m())


def matrix(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)])


def rotation_of(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(3)] for i in range(3)]).T


def seated_ball_in_pad_frame() -> np.ndarray:
    """Ball centre resting on the pad centroid, in the pad's own frame."""
    centroid = TAXELS.mean(axis=0)
    for offset in np.linspace(0.020, 0.050, 3001):
        candidate = centroid + np.array([0.0, 0.0, offset])
        if abs(np.linalg.norm(TAXELS - candidate, axis=1).min()
               - TARGET_BALL_RADIUS) < 2.0e-5:
            return candidate
    raise RuntimeError("could not solve the seated ball centre")


def seated_ball_in_gripper_link(stage) -> np.ndarray:
    """Ball centre, in gripper_link axes, when it rests on the pad centroid."""
    world = lambda p: UsdGeom.Xformable(
        stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)
    pad_in_link = world(TACTILE_ROOT_PRIM_PATHS[0]) * world(
        GRIPPER_LINK_PRIM_PATH).GetInverse()
    centroid = TAXELS.mean(axis=0)
    for offset in np.linspace(0.020, 0.050, 3001):
        candidate = centroid + np.array([0.0, 0.0, offset])
        if abs(np.linalg.norm(TAXELS - candidate, axis=1).min()
               - TARGET_BALL_RADIUS) < 2.0e-5:
            break
    else:
        raise RuntimeError("could not solve the seated ball centre")
    return (np.append(candidate, 1.0) @ matrix(pad_in_link))[:3]


def seated_grasp_frame(stage, geometry, ball_world, tilt_deg, iterations=6):
    """IK frame pose that puts the ball on the fixed pad's centroid.

    The SO-101 has five joints, so the tool orientation and the tool position
    must share one base-yaw plane.  Seating the ball fixes the position, which
    in turn fixes the required yaw, so solve that fixed point instead of
    orienting toward the ball -- that mismatch was 6.6 deg and made the IK
    infeasible.
    """
    world = lambda p: UsdGeom.Xformable(
        stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)
    w_link, w_frame = world(GRIPPER_LINK_PRIM_PATH), world(GRIPPER_FRAME_PRIM)
    link_in_frame = w_link * w_frame.GetInverse()
    frame_origin_in_link = np.asarray(
        (w_frame * w_link.GetInverse()).ExtractTranslation())
    seated = seated_ball_in_gripper_link(stage)
    ball_world = np.asarray(ball_world, dtype=float)

    virtual_object = ball_world.copy()
    frame = orientation = None
    for _ in range(iterations):
        pose = geometry.tool_pose(virtual_object, tilt_deg)
        tool = np.asarray(pose["tool_rotation"])
        frame_rot = Gf.Matrix4d()
        frame_rot.SetRotate(Gf.Matrix3d(*tool.T.flatten()))
        link_rot = rotation_of(link_in_frame * frame_rot)
        frame = link_rot @ frame_origin_in_link + (
            ball_world - link_rot @ seated)
        orientation = pose["orientation"]
        frame_yaw = math.atan2(frame[1], frame[0])
        if abs(frame_yaw - geometry.motion_yaw(virtual_object)) < 3.0e-4:
            break
        radius = float(np.linalg.norm(virtual_object[:2]))
        virtual_object = np.array([radius * math.cos(frame_yaw),
                                   radius * math.sin(frame_yaw),
                                   virtual_object[2]])
    return frame, orientation


def quintic(t: float) -> float:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def apply_compliant_contact(stage, stiffness: float, damping: float) -> None:
    """Make the ball's contact a spring instead of a hard constraint."""
    from pxr import PhysxSchema
    material = UsdShade.Material.Define(
        stage, "/World/PhysicsMaterials/SeatedGrasp")
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateCompliantContactStiffnessAttr().Set(float(stiffness))
    physx.CreateCompliantContactDampingAttr().Set(float(damping))


def apply_contact_offset(stage, contact_offset: float, rest_offset: float):
    """Author contact/rest offsets on the ball and both jaw colliders."""
    from pxr import PhysxSchema
    targets = ["/World/TargetBall/Collision"]
    for root in GRIPPER_CONTACT_PRIM_PATHS:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                targets.append(prim.GetPath().pathString)
    for path in targets:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr().Set(float(contact_offset))
        api.CreateRestOffsetAttr().Set(float(rest_offset))
    return len(targets)


def apply_contact_friction(stage, static_friction: float) -> None:
    """Bind one high-friction physics material to the ball and both jaws."""
    material = UsdShade.Material.Define(
        stage, "/World/PhysicsMaterials/SeatedGrasp")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr().Set(float(static_friction))
    physics.CreateDynamicFrictionAttr().Set(float(static_friction) * 0.85)
    physics.CreateRestitutionAttr().Set(0.0)
    for path in (BALL_PATH, *GRIPPER_CONTACT_PRIM_PATHS):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"cannot bind friction, missing prim: {path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics")


def solve_chain(solver, grasp_frame, approach_frame, orientation):
    """IK for the descend / grasp / lift chain, or None if any waypoint fails."""
    solutions, warm = {}, None
    for label, position in (("grasp", grasp_frame),
                            ("approach", approach_frame),
                            ("lift", grasp_frame + np.array([0, 0, 0.12]))):
        joints, ok = solver.compute_inverse_kinematics(
            frame_name=IK_FRAME, target_position=position,
            target_orientation=orientation, warm_start=warm,
            position_tolerance=0.010, orientation_tolerance=0.3)
        if not ok:
            return None
        solutions[label] = joints
        warm = joints
    return solutions


def settle_at_grasp(world_sim, ball, drives, arm, solutions, ball_position):
    """Drive to the grasp pose with the gripper open and let it settle."""
    world_sim.reset()
    ball.set_world_poses(positions=np.array([ball_position]),
                         orientations=np.array([[1.0, 0, 0, 0]]))
    ball.set_velocities(np.zeros((1, 6)))
    for name, value in zip(arm, np.degrees(solutions["grasp"])):
        drives[name].CreateTargetPositionAttr().Set(float(value))
    drives["gripper"].CreateTargetPositionAttr().Set(OPEN_DEG)
    for _ in range(round(1.5 * HZ)):
        world_sim.step(render=False)


def main() -> int:
    solver = LulaKinematicsSolver(
        robot_description_path=str(PROJECT_ROOT / "so101_descriptor.yaml"),
        urdf_path=str(PROJECT_ROOT.parent / "SO-ARM100" / "Simulation"
                      / "SO101" / "so101_new_calib.urdf"))
    yaw = math.radians(ROBOT_BASE_YAW_DEG)
    solver.set_robot_base_pose(
        np.asarray(ROBOT_BASE_POSITION),
        np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
    arm = solver.get_joint_names()
    geometry = RadialToolGeometry(ROBOT_BASE_POSITION[:2])

    omni.usd.get_context().open_stage(str(ARGS.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    drives = {n: UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath(f"{JOINT_ROOT}/{n}"), "angular")
        for n in arm + ["gripper"]}
    world_sim = World(stage_units_in_meters=1.0, physics_dt=1.0 / HZ)
    world_sim.reset()
    ball = RigidPrim(BALL_PATH)
    tactile = DPS2015EliteMvp(stage)

    grasp_frame, orientation = seated_grasp_frame(
        stage, geometry, BALL_START, ARGS.tilt)
    approach_frame = grasp_frame + np.array([0.0, 0.0, ARGS.approach_rise])
    print(f"seated grasp IK frame : {np.round(grasp_frame, 4)}")
    print(f"descend from          : {np.round(approach_frame, 4)}")

    # The seated pose needs a 10 mm convergence tolerance; at 5 mm Lula reports
    # failure even though scripts/diagnostics/probe_seated_grasp_reach.py shows
    # the pose and its whole neighbourhood are reachable.  The tolerance is a
    # solver stopping criterion, so check the pose actually achieved via FK.
    solutions, warm = {}, None
    for label, position in (("grasp", grasp_frame),
                            ("approach", approach_frame),
                            ("lift", grasp_frame + np.array([0, 0, 0.12]))):
        joints, ok = solver.compute_inverse_kinematics(
            frame_name=IK_FRAME, target_position=position,
            target_orientation=orientation, warm_start=warm,
            position_tolerance=0.010, orientation_tolerance=0.3)
        achieved, achieved_rot = solver.compute_forward_kinematics(
            IK_FRAME, joints)
        error_mm = float(np.linalg.norm(np.asarray(achieved) - position)) * 1000
        print(f"[IK {label:9s}] ok={ok}  achieved error {error_mm:6.2f} mm")
        if not ok:
            raise RuntimeError(f"IK failed for {label}")
        solutions[label] = joints
        if label == "grasp":
            achieved_grasp = (np.asarray(achieved), np.asarray(achieved_rot))
        warm = joints

    # A 5-DOF arm cannot hit a full 6-DOF pose exactly, so the tool lands ~8 mm
    # from the commanded grasp.  Rather than fight that, read back where the
    # jaw actually ends up and put the ball where it will be seated there.  The
    # ball start is a scene placeholder, not a calibrated measurement.
    world = lambda p: UsdGeom.Xformable(
        stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)
    link_in_frame = matrix(world(GRIPPER_LINK_PRIM_PATH)
                           * world(GRIPPER_FRAME_PRIM).GetInverse())
    seated_in_link = seated_ball_in_gripper_link(stage)
    frame_position, frame_rotation = achieved_grasp
    link_rotation = frame_rotation @ link_in_frame[:3, :3].T
    link_origin = frame_rotation @ link_in_frame[3, :3] + frame_position
    seated_world = link_rotation @ seated_in_link + link_origin
    table_z = TABLE_TOP_Z + TARGET_BALL_RADIUS
    print(f"seated ball for the ACHIEVED pose: {np.round(seated_world, 4)}  "
          f"(table rest height {table_z:.4f}, off by "
          f"{(seated_world[2] - table_z) * 1000:+.2f} mm)")
    ball_target = np.array([seated_world[0], seated_world[1], table_z])
    print(f"nominal ball position {np.round(ball_target, 4)}")

    # Forward kinematics is not where the arm actually goes: the position
    # drives are limited to 3.35 N m, so gravity leaves a steady-state error
    # that put the ball 8.9 mm off the pad.  Calibrate against the pose the
    # simulation really reaches -- descend with the ball parked off the table,
    # read the pad's actual transform, and shift the tool by the residual.
    seated_in_pad = seated_ball_in_pad_frame()
    parked = np.array([2.0, 2.0, 0.03])
    for attempt in range(5):
        solutions = solve_chain(solver, grasp_frame, approach_frame, orientation)
        if solutions is None:
            raise RuntimeError("IK failed during calibration")
        settle_at_grasp(world_sim, ball, drives, arm, solutions, parked)
        pad_world = UsdGeom.Xformable(
            stage.GetPrimAtPath(TACTILE_ROOT_PRIM_PATHS[0])
        ).ComputeLocalToWorldTransform(0)
        seat_world = np.asarray(
            pad_world.Transform(Gf.Vec3d(*seated_in_pad)), dtype=float)
        residual = ball_target - seat_world
        print(f"[calib {attempt}] actual seat {np.round(seat_world, 4)}  "
              f"residual {np.round(residual * 1000, 2)} mm", flush=True)
        if np.linalg.norm(residual) < 0.0015:
            break
        grasp_frame = grasp_frame + residual
        approach_frame = grasp_frame + np.array([0.0, 0.0, ARGS.approach_rise])
    solutions = solve_chain(solver, grasp_frame, approach_frame, orientation)
    if solutions is None:
        raise RuntimeError("IK failed after calibration")
    ball_start = ball_target

    print("  real reference: fixed 21 taxels uniformity 0.46, "
          "moving 34 taxels uniformity 0.58")
    print(f"\n{'close':>6s} | {'ftax':>5s} {'funi':>5s} {'fN':>6s} | "
          f"{'mtax':>5s} {'muni':>5s} {'mN':>6s} | {'peakN':>6s} {'lift mm':>7s}"
          "  seated?")
    rows = []
    for offset in ARGS.contact_offset:
        if offset is not None:
            count = apply_contact_offset(stage, offset, ARGS.rest_offset)
            print(f"--- contactOffset {offset*1000:.1f} mm on {count} colliders "
                  f"(rest {ARGS.rest_offset*1000:.1f} mm) ---", flush=True)
        for friction in ARGS.friction:
            if friction is not None:
                apply_contact_friction(stage, friction)
                print(f"--- static friction {friction:.2f} ---", flush=True)
            for stiffness in ARGS.compliant_stiffness:
                if stiffness is not None:
                    apply_compliant_contact(
                        stage, stiffness, ARGS.compliant_damping)
                    print(f"--- compliant stiffness {stiffness:g} ---",
                          flush=True)
                for close_deg in ARGS.close_deg:
                    row = run_once(stage, world_sim, ball, tactile, drives,
                                   arm, solutions, close_deg, ball_start)
                    row["static_friction"] = friction
                    row["contact_offset_m"] = offset
                    row["compliant_stiffness"] = stiffness
                    rows.append(row)
                    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
                    ARGS.output.write_text(json.dumps(
                        {"grasp_frame_m": grasp_frame.tolist(),
                         "ball_start_m": ball_start.tolist(),
                         "tilt_deg": ARGS.tilt, "trials": rows},
                        indent=2) + "\n", encoding="utf-8")
    good = [r for r in rows if r["seated"] and r["lift_delta_mm"] > 50]
    print(f"\nseated on BOTH pads and lifted: {len(good)} / {len(rows)}")
    return 0 if good else 2


def run_once(stage, world_sim, ball, tactile, drives, arm, solutions, close_deg,
             ball_start):
    world_sim.reset()
    ball.set_world_poses(positions=np.array([ball_start]),
                         orientations=np.array([[1.0, 0, 0, 0]]))
    ball.set_velocities(np.zeros((1, 6)))
    command = np.degrees(solutions["approach"])
    for n, v in zip(arm, command):
        drives[n].CreateTargetPositionAttr().Set(float(v))
    drives["gripper"].CreateTargetPositionAttr().Set(OPEN_DEG)
    for _ in range(round(0.6 * HZ)):
        world_sim.step(render=False)
    tactile.reset(world_sim.current_time)
    grip, peak = OPEN_DEG, 0.0

    def segment(target, gripper_target, seconds):
        nonlocal command, grip, peak
        start, start_grip, goal = command.copy(), grip, np.degrees(target)
        steps = max(1, round(seconds * HZ))
        for i in range(1, steps + 1):
            b = quintic(i / steps)
            command = start + b * (goal - start)
            grip = start_grip + b * (gripper_target - start_grip)
            for n, v in zip(arm, command):
                drives[n].CreateTargetPositionAttr().Set(float(v))
            drives["gripper"].CreateTargetPositionAttr().Set(float(grip))
            world_sim.step(render=False)
            s = tactile.maybe_sample(world_sim.current_time,
                                     world_sim.current_time_step_index)
            if s is not None:
                peak = max(peak, float(np.abs(s.raw_total_forces[..., 2]).max()))

    segment(solutions["approach"], OPEN_DEG, 0.3)   # settle open
    segment(solutions["grasp"], OPEN_DEG, 0.9)      # vertical descent

    # Where did the ball actually end up relative to the pad once the tool has
    # descended?  If this is not the seated point, the fault is the arm's pose
    # tracking, not the closing.
    pad_world = UsdGeom.Xformable(
        stage.GetPrimAtPath(TACTILE_ROOT_PRIM_PATHS[0])
    ).ComputeLocalToWorldTransform(0)
    ball_world = np.asarray(UsdGeom.Xformable(
        stage.GetPrimAtPath(BALL_PATH)
    ).ComputeLocalToWorldTransform(0).ExtractTranslation())
    after_descent = np.asarray(
        pad_world.GetInverse().Transform(Gf.Vec3d(*ball_world)), dtype=float)
    seated_target = seated_ball_in_pad_frame()
    print(f"       after descent: ball_in_pad={np.round(after_descent*1000,2)} mm "
          f"(seated target {np.round(seated_target*1000,2)}, "
          f"off {np.linalg.norm(after_descent-seated_target)*1000:.2f} mm)",
          flush=True)

    segment(solutions["grasp"], close_deg, 0.8)     # close
    segment(solutions["grasp"], close_deg, 0.3)     # hold

    sample = tactile.latest
    def pad(i):
        """accepted contacts, nearest-taxel mm, active taxels, |F| N."""
        if sample is None:
            return 0, None, 0, 0.0
        near = sample.nearest_taxel_distances_m[i]
        magnitudes = np.linalg.norm(sample.taxel_forces[i], axis=-1)
        loaded = magnitudes[magnitudes > 0]
        uniformity = (float(loaded.min() / loaded.max())
                      if loaded.size else 0.0)
        return (int(sample.number_of_contacts[i]),
                float(near) * 1000 if np.isfinite(near) else None,
                int(np.count_nonzero(magnitudes)),
                float(np.linalg.norm(sample.taxel_forces[i].sum(axis=0))),
                uniformity)
    fa, fn, f_active, f_total, f_uniform = pad(0)
    ma, mn, m_active, m_total, m_uniform = pad(1)
    before = float(UsdGeom.Xformable(stage.GetPrimAtPath(BALL_PATH))
                   .ComputeLocalToWorldTransform(0).ExtractTranslation()[2])
    segment(solutions["lift"], close_deg, 0.9)
    segment(solutions["lift"], close_deg, 0.3)
    after = float(UsdGeom.Xformable(stage.GetPrimAtPath(BALL_PATH))
                  .ComputeLocalToWorldTransform(0).ExtractTranslation()[2])
    seated = fa > 0 and ma > 0
    fmt = lambda v: "      -" if v is None else f"{v:7.2f}"
    print(f"{close_deg:6.1f} | {f_active:5d} {f_uniform:5.2f} {f_total:6.2f} | "
          f"{m_active:5d} {m_uniform:5.2f} {m_total:6.2f} | "
          f"{peak:6.1f} {(after-before)*1000:7.1f}  "
          f"{'YES' if seated else 'no'}", flush=True)
    return {"close_deg": close_deg,
            "fixed_accepted": fa, "fixed_nearest_mm": fn,
            "fixed_active_taxels": f_active, "fixed_total_n": f_total,
            "fixed_uniformity": f_uniform, "moving_uniformity": m_uniform,
            "moving_accepted": ma, "moving_nearest_mm": mn,
            "moving_active_taxels": m_active, "moving_total_n": m_total,
            "peak_raw_normal_n": peak, "lift_delta_mm": (after - before) * 1000,
            "seated": seated}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
