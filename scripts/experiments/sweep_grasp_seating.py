"""Find the grasp offset that seats the ball on both tactile pads.

The scripted pick-and-place lifts the ball, but the video and the pad-frame
logs show it pinched at the jaw tips: contacts on the moving pad land ~20 mm
from the nearest taxel and are discarded, and the fixed pad only catches the
ball on its edge taxels.  This sweep varies the grasp reference offset in the
motion frame and reports, per trial, where the ball actually sits in each pad's
own frame and whether the contact decoder accepts it.

Everything runs in one Isaac session; trials are separated by world.reset().
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))


DEFAULT_TOOL_TILT_DEG_HINT = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--usd", type=Path, default=PROJECT_ROOT_HINT / "lab_scene_task.usda"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT_HINT / "tactile_logs" / "grasp_seating_sweep.json",
    )
    parser.add_argument(
        "--dy",
        type=float,
        nargs="*",
        default=[0.0055, 0.0115, 0.0175, 0.0235, 0.0315, 0.0375],
        help="Motion-frame y offset of the grasp reference from the ball.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        nargs="*",
        default=[-0.004, 0.0022, 0.008],
        help="Motion-frame x offset of the grasp reference from the ball.",
    )
    parser.add_argument(
        "--dz",
        type=float,
        nargs="*",
        default=[0.0],
        help="Motion-frame z offset; shifts which latitude of the ball the "
             "jaws close on.",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        nargs="*",
        default=[DEFAULT_TOOL_TILT_DEG_HINT],
        help="Tool tilt in degrees for the grasp approach.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        nargs="*",
        default=[None],
        help=(
            "Static friction applied at runtime to the ball and both gripper "
            "contact bodies.  Omit to leave the scene's PhysX defaults, which "
            "is what every earlier sweep used."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()

# Kit re-parses whatever is left in sys.argv and stalls at startup on the
# flags meant for this script, so hand it a clean argv before booting.
sys.argv = sys.argv[:1]

from isaacsim import SimulationApp

APP = SimulationApp(
    {
        "headless": True,
        "active_gpu": ARGS.gpu,
        "physics_gpu": ARGS.gpu,
        "multi_gpu": False,
    }
)

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

from tactile_sensor import DPS2015EliteMvp
from scripts.tasks.grasp_geometry import (
    DEFAULT_TOOL_TILT_DEG,
    RadialToolGeometry,
)
from scene_config import (
    GRIPPER_CONTACT_PRIM_PATHS,
    PROJECT_ROOT,
    ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG,
    TABLE_TOP_Z,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS,
    TARGET_BALL_RADIUS,
    TARGET_BALL_XY_TASK_READY_PLACEHOLDER,
)
from tactile_taxels import load_taxel_positions_m

JOINT_ROOT = "/World/Robot/Physics"
BALL_PATH = "/World/TargetBall"
IK_FRAME = "gripper_frame_link"
PHYSICS_HZ = TACTILE_MVP_PHYSICS_FREQUENCY_HZ
OPEN_GRIPPER_DEG = 70.0
CLOSED_GRIPPER_DEG = 5.0

BALL_START = np.array(
    [
        TARGET_BALL_XY_TASK_READY_PLACEHOLDER[0],
        TARGET_BALL_XY_TASK_READY_PLACEHOLDER[1],
        TABLE_TOP_Z + TARGET_BALL_RADIUS,
    ]
)
TAXELS = np.asarray(load_taxel_positions_m())
TAXEL_CENTROID = TAXELS.mean(axis=0)


def apply_contact_friction(stage, static_friction: float) -> None:
    """Bind a high-friction physics material to the ball and both jaws.

    The scene ships with no physics material at all, so PhysX defaults apply.
    The spec calls for friction well above smooth plastic for a plush ball;
    this makes that tunable without rebuilding the USD.
    """
    dynamic_friction = static_friction * 0.85
    material_path = "/World/PhysicsMaterials/SweepContact"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(float(static_friction))
    physics_material.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    physics_material.CreateRestitutionAttr().Set(0.0)

    for path in (BALL_PATH, *GRIPPER_CONTACT_PRIM_PATHS):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Cannot bind friction, missing prim: {path}")
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(
            material,
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )


def quintic(phase: float) -> float:
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def world_to_local(stage, prim_path: str, point_world) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return np.asarray(
        transform.GetInverse().Transform(Gf.Vec3d(*point_world)), dtype=float
    )


def main() -> int:
    solver = LulaKinematicsSolver(
        robot_description_path=str(PROJECT_ROOT / "so101_descriptor.yaml"),
        urdf_path=str(
            PROJECT_ROOT.parent
            / "SO-ARM100"
            / "Simulation"
            / "SO101"
            / "so101_new_calib.urdf"
        ),
    )
    base_yaw = math.radians(ROBOT_BASE_YAW_DEG)
    solver.set_robot_base_pose(
        np.asarray(ROBOT_BASE_POSITION),
        np.array([math.cos(base_yaw / 2), 0.0, 0.0, math.sin(base_yaw / 2)]),
    )
    arm_joints = solver.get_joint_names()
    geometry = RadialToolGeometry(ROBOT_BASE_POSITION[:2])

    omni.usd.get_context().open_stage(str(ARGS.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    drives = {
        name: UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}"), "angular"
        )
        for name in arm_joints + ["gripper"]
    }

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ)
    world.reset()
    ball = RigidPrim(BALL_PATH)
    tactile = DPS2015EliteMvp(stage)

    print(
        f"{'tilt':>5s} {'dx mm':>6s} {'dy mm':>6s} {'dz mm':>6s} {'IK':>3s} | "
        f"{'fixacc':>6s} {'fix near':>8s} | "
        f"{'movacc':>6s} {'movrej':>6s} {'mov near':>8s} | "
        f"{'peakN':>6s} {'lift mm':>7s}",
        flush=True,
    )
    rows = []
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        # Written after every trial: a long sweep that gets interrupted still
        # leaves usable results behind.
        ARGS.output.write_text(
            json.dumps(
                {
                    "ball_start_m": BALL_START.tolist(),
                    "taxel_centroid_m": TAXEL_CENTROID.tolist(),
                    "taxel_extent_min_m": TAXELS.min(axis=0).tolist(),
                    "taxel_extent_max_m": TAXELS.max(axis=0).tolist(),
                    "trials": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for friction in ARGS.friction:
        if friction is not None:
            apply_contact_friction(stage, friction)
            print(f"--- static friction {friction:.2f} ---", flush=True)
        for tilt in ARGS.tilt:
            for dx in ARGS.dx:
                for dy in ARGS.dy:
                    for dz in ARGS.dz:
                        row = run_trial(
                            stage, world, ball, tactile, solver, geometry,
                            drives, arm_joints, dx, dy, dz, tilt,
                        )
                        row["static_friction"] = friction
                        rows.append(row)
                        flush()
    print("sweep written:", ARGS.output)

    seated = [
        r for r in rows
        if r["ik_ok"] and r["lift_delta_m"] > 0.05
        and r["fixed"]["accepted"] > 0 and r["moving"]["accepted"] > 0
    ]
    print(f"trials seating both pads: {len(seated)} / {len(rows)}")
    for r in seated:
        print(
            f"  tilt={r['tilt_deg']:.0f} dx={r['dx_m']*1000:+.1f} "
            f"dy={r['dy_m']*1000:+.1f} dz={r['dz_m']*1000:+.1f} mm  "
            f"peak={r['peak_raw_normal_n']:.1f} N  lift={r['lift_delta_m']*1000:.0f} mm"
        )
    return 0 if seated else 2


def run_trial(
    stage, world, ball, tactile, solver, geometry, drives, arm_joints,
    dx, dy, dz, tilt,
):
    yaw = geometry.motion_yaw(BALL_START)
    ik_object = BALL_START + geometry.motion_rotation(yaw) @ np.array([dx, dy, dz])
    pose = geometry.tool_pose(ik_object, tilt)
    frame, orientation, axis = (
        pose["frame_position"], pose["orientation"], pose["insertion_axis"]
    )

    solutions = {}
    warm = None
    ok = True
    for label, position in (
        ("grasp", frame),
        ("pregrasp", frame - 0.055 * axis),
        ("approach", frame - 0.075 * axis),
        ("lift", frame + np.array([0.0, 0.0, 0.12])),
    ):
        joints, success = solver.compute_inverse_kinematics(
            frame_name=IK_FRAME,
            target_position=position,
            target_orientation=orientation,
            warm_start=warm,
            position_tolerance=0.005,
            orientation_tolerance=0.3,
        )
        if not success:
            ok = False
            break
        solutions[label] = joints
        warm = joints

    empty = {"accepted": 0, "rejected": 0, "nearest_mm": None, "ball_in_pad_mm": None}
    if not ok:
        print(
            f"{tilt:5.0f} {dx*1000:6.1f} {dy*1000:6.1f} {dz*1000:6.1f} "
            f"{'no':>3s} | (IK failed)"
        )
        return {
            "dx_m": dx, "dy_m": dy, "dz_m": dz, "tilt_deg": tilt,
            "ik_ok": False,
            "fixed": empty, "moving": empty,
            "peak_raw_normal_n": None, "lift_delta_m": 0.0,
        }

    world.reset()
    ball.set_world_poses(
        positions=np.array([BALL_START]),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    ball.set_velocities(np.zeros((1, 6)))

    command = np.degrees(solutions["approach"])
    for name, value in zip(arm_joints, command):
        drives[name].CreateTargetPositionAttr().Set(float(value))
    drives["gripper"].CreateTargetPositionAttr().Set(OPEN_GRIPPER_DEG)
    for _ in range(round(0.6 * PHYSICS_HZ)):
        world.step(render=False)
    tactile.reset(world.current_time)

    gripper_command = OPEN_GRIPPER_DEG
    peak_normal = 0.0

    def run_segment(target_joints, gripper_target, duration_s):
        nonlocal command, gripper_command, peak_normal
        start = command.copy()
        start_gripper = gripper_command
        goal = np.degrees(target_joints)
        steps = max(1, round(duration_s * PHYSICS_HZ))
        for index in range(1, steps + 1):
            blend = quintic(index / steps)
            command = start + blend * (goal - start)
            gripper_command = start_gripper + blend * (
                gripper_target - start_gripper
            )
            for name, value in zip(arm_joints, command):
                drives[name].CreateTargetPositionAttr().Set(float(value))
            drives["gripper"].CreateTargetPositionAttr().Set(
                float(gripper_command)
            )
            world.step(render=False)
            sample = tactile.maybe_sample(
                world.current_time, world.current_time_step_index
            )
            if sample is not None:
                peak_normal = max(
                    peak_normal, float(np.abs(sample.raw_total_forces[..., 2]).max())
                )

    run_segment(solutions["pregrasp"], OPEN_GRIPPER_DEG, 0.4)
    run_segment(solutions["grasp"], OPEN_GRIPPER_DEG, 0.4)
    run_segment(solutions["grasp"], CLOSED_GRIPPER_DEG, 0.7)
    run_segment(solutions["grasp"], CLOSED_GRIPPER_DEG, 0.3)

    ball_world = np.asarray(
        UsdGeom.Xformable(
            stage.GetPrimAtPath(BALL_PATH)
        ).ComputeLocalToWorldTransform(0).ExtractTranslation()
    )
    sample = tactile.latest
    pads = []
    for index in range(2):
        in_pad = world_to_local(stage, TACTILE_ROOT_PRIM_PATHS[index], ball_world)
        nearest = (
            None
            if sample is None or not np.isfinite(sample.nearest_taxel_distances_m[index])
            else float(sample.nearest_taxel_distances_m[index]) * 1000.0
        )
        pads.append(
            {
                "accepted": 0 if sample is None else int(sample.number_of_contacts[index]),
                "rejected": 0 if sample is None else int(sample.rejected_contact_counts[index]),
                "nearest_mm": nearest,
                "ball_in_pad_mm": (in_pad * 1000.0).tolist(),
            }
        )

    before = ball_world[2]
    run_segment(solutions["lift"], CLOSED_GRIPPER_DEG, 0.8)
    run_segment(solutions["lift"], CLOSED_GRIPPER_DEG, 0.3)
    after = float(
        UsdGeom.Xformable(
            stage.GetPrimAtPath(BALL_PATH)
        ).ComputeLocalToWorldTransform(0).ExtractTranslation()[2]
    )
    lift_delta = after - before

    def fmt(value, spec):
        return "    n/a" if value is None else format(value, spec)

    print(
        f"{tilt:5.0f} {dx*1000:6.1f} {dy*1000:6.1f} {dz*1000:6.1f} "
        f"{'ok':>3s} | "
        f"{pads[0]['accepted']:6d} {fmt(pads[0]['nearest_mm'], '8.2f')} | "
        f"{pads[1]['accepted']:6d} {pads[1]['rejected']:6d} "
        f"{fmt(pads[1]['nearest_mm'], '8.2f')} | "
        f"{peak_normal:6.1f} {lift_delta*1000:7.1f}",
        flush=True,
    )
    return {
        "dx_m": dx, "dy_m": dy, "dz_m": dz, "tilt_deg": tilt, "ik_ok": True,
        "fixed": pads[0], "moving": pads[1],
        "peak_raw_normal_n": peak_normal, "lift_delta_m": lift_delta,
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
