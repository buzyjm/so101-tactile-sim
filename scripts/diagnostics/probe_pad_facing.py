"""At which opening angle do the two tactile pads actually face each other?

The moving jaw is a single revolute finger, so its pad swings as the gripper
closes and only presents its active face squarely at one opening angle.  The
object diameter that this gripper can grip *on both pads* is the pad-centre
separation at that angle.  This probe sweeps the gripper joint with no object
in the way and reports separation and facing error per angle.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--usd", type=Path, default=PROJECT_ROOT_HINT / "lab_scene_task.usda"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT_HINT / "tactile_logs" / "pad_facing_probe.json",
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
from pxr import Gf, UsdGeom, UsdPhysics

from scene_config import (
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS,
)

JOINT_ROOT = "/World/Robot/Physics"
BALL_PATH = "/World/TargetBall"
ANGLES_DEG = list(range(-8, 76, 2))


def pad_frame(stage, prim_path):
    """Pad centre and outward surface normal in world coordinates."""
    prim = stage.GetPrimAtPath(prim_path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    centre = np.asarray(
        transform.Transform(Gf.Vec3d(*TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME)),
        dtype=float,
    )
    matrix = np.array([[transform[r][c] for c in range(3)] for r in range(3)])
    normal = matrix[2] / np.linalg.norm(matrix[2])
    return centre, normal


def main() -> int:
    omni.usd.get_context().open_stage(str(ARGS.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    drive = UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath(f"{JOINT_ROOT}/gripper"), "angular"
    )
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    )
    world.reset()

    # Park the ball well clear so the jaws close freely.
    ball = RigidPrim(BALL_PATH)
    ball.set_world_poses(
        positions=np.array([[0.0, 0.0, -5.0]]),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    ball.set_velocities(np.zeros((1, 6)))

    print(
        f"{'gripper deg':>11s} {'pad gap mm':>10s} "
        f"{'fixed facing deg':>16s} {'moving facing deg':>17s} "
        f"{'worst deg':>9s}"
    )
    rows = []
    for angle in ANGLES_DEG:
        drive.CreateTargetPositionAttr().Set(float(angle))
        for _ in range(round(0.35 * TACTILE_MVP_PHYSICS_FREQUENCY_HZ)):
            world.step(render=False)

        fixed_centre, fixed_normal = pad_frame(stage, TACTILE_ROOT_PRIM_PATHS[0])
        moving_centre, moving_normal = pad_frame(stage, TACTILE_ROOT_PRIM_PATHS[1])
        separation = moving_centre - fixed_centre
        gap = float(np.linalg.norm(separation))
        if gap < 1e-9:
            continue
        direction = separation / gap
        # A pad faces the other one when its outward normal points at it.
        fixed_facing = math.degrees(
            math.acos(float(np.clip(np.dot(fixed_normal, direction), -1.0, 1.0)))
        )
        moving_facing = math.degrees(
            math.acos(float(np.clip(np.dot(moving_normal, -direction), -1.0, 1.0)))
        )
        worst = max(fixed_facing, moving_facing)
        rows.append(
            {
                "gripper_deg": angle,
                "pad_gap_m": gap,
                "fixed_facing_deg": fixed_facing,
                "moving_facing_deg": moving_facing,
                "worst_facing_deg": worst,
            }
        )
        print(
            f"{angle:11d} {gap * 1000:10.2f} {fixed_facing:16.1f} "
            f"{moving_facing:17.1f} {worst:9.1f}"
        )

    # If the moving pad is simply mis-mounted, the rotation needed to make it
    # face the other pad is the SAME in its own local frame at every opening
    # angle.  A drifting correction would instead mean a kinematic artefact.
    print()
    print("correction that would make the moving pad face the fixed pad,")
    print("expressed in the moving mount's own local frame:")
    print(
        f"{'gripper deg':>11s} {'axis_x':>8s} {'axis_y':>8s} {'axis_z':>8s} "
        f"{'angle deg':>9s}"
    )
    corrections = []
    for angle in (6, 10, 12, 16, 20, 30, 40):
        drive.CreateTargetPositionAttr().Set(float(angle))
        for _ in range(round(0.35 * TACTILE_MVP_PHYSICS_FREQUENCY_HZ)):
            world.step(render=False)
        fixed_centre, _ = pad_frame(stage, TACTILE_ROOT_PRIM_PATHS[0])
        moving_centre, moving_normal = pad_frame(
            stage, TACTILE_ROOT_PRIM_PATHS[1]
        )
        target = fixed_centre - moving_centre
        target /= np.linalg.norm(target)

        axis_world = np.cross(moving_normal, target)
        sin_a = float(np.linalg.norm(axis_world))
        cos_a = float(np.dot(moving_normal, target))
        if sin_a < 1e-9:
            continue
        axis_world /= sin_a
        correction_deg = math.degrees(math.atan2(sin_a, cos_a))

        prim = stage.GetPrimAtPath(TACTILE_ROOT_PRIM_PATHS[1])
        matrix_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        rotation = np.array(
            [[matrix_world[r][c] for c in range(3)] for r in range(3)]
        )
        axis_local = rotation @ axis_world
        axis_local /= np.linalg.norm(axis_local)
        corrections.append(
            {
                "gripper_deg": angle,
                "axis_local": axis_local.tolist(),
                "angle_deg": correction_deg,
            }
        )
        print(
            f"{angle:11d} {axis_local[0]:8.3f} {axis_local[1]:8.3f} "
            f"{axis_local[2]:8.3f} {correction_deg:9.2f}"
        )

    if corrections:
        angles = np.array([c["angle_deg"] for c in corrections])
        axes = np.array([c["axis_local"] for c in corrections])
        spread_deg = float(angles.max() - angles.min())
        axis_spread = float(
            np.max(np.linalg.norm(axes - axes.mean(axis=0), axis=1))
        )
        print()
        print(
            f"correction angle spread across openings: {spread_deg:.2f} deg; "
            f"axis spread: {axis_spread:.3f}"
        )
        print(
            "constant correction => static mount error; "
            "drifting => kinematic artefact"
        )

    best = min(rows, key=lambda r: r["worst_facing_deg"])
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(
        json.dumps(
            {
                "rows": rows,
                "best_facing": best,
                "moving_pad_corrections": corrections,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print(
        f"best facing at gripper {best['gripper_deg']} deg: "
        f"worst pad off by {best['worst_facing_deg']:.1f} deg, "
        f"pad gap {best['pad_gap_m'] * 1000:.1f} mm"
    )
    if best["moving_facing_deg"] > 10.0:
        print(
            "NOTE: the moving pad never faces the fixed pad "
            f"(best {best['moving_facing_deg']:.1f} deg), so no opening angle "
            "grips an object on both pads.  Fix the mount before reading any "
            "object-size conclusion from this table."
        )
    else:
        print(
            "=> an object this gripper can press onto BOTH pads is about "
            f"{best['pad_gap_m'] * 1000:.0f} mm across."
        )
    print("probe written:", ARGS.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
