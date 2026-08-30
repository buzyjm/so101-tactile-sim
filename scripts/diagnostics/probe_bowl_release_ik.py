"""Map the reachable SO-101 release poses above the bowl.

The scripted pick-and-place aborted because the bowl-hover waypoint had no IK
solution at the grasp-side tool tilt.  This probe boots Lula only (no PhysX, no
RTX) and sweeps tool tilt, release clearance, and warm start so the task can be
retuned against measured reachability instead of guesses.
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
        "--grasp-result",
        type=Path,
        default=PROJECT_ROOT_HINT
        / "tactile_logs"
        / "scripted_ball_grasp_result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT_HINT / "tactile_logs" / "bowl_release_ik_probe.json",
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

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

from scene_config import (
    BOWL_CENTER_XY_PLACEHOLDER,
    BOWL_HEIGHT,
    PROJECT_ROOT,
    ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG,
    TABLE_TOP_Z,
    TARGET_BALL_RADIUS,
)
from scripts.tasks.grasp_geometry import (
    DEFAULT_TOOL_TILT_DEG,
    RadialToolGeometry,
)

IK_FRAME = "gripper_frame_link"
IK_GRASP_REFERENCE = np.array([0.08, 0.31, TABLE_TOP_Z + TARGET_BALL_RADIUS])
LIFT_HEIGHT_M = 0.12
BOWL_CENTER = np.asarray(BOWL_CENTER_XY_PLACEHOLDER, dtype=float)

TILTS_DEG = [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]
CLEARANCES_M = [0.012, 0.030, 0.050, 0.080]
GRID_TILT_DEG = 15.0
GRID_CLEARANCE_M = 0.030


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
        np.array([math.cos(base_yaw / 2.0), 0.0, 0.0, math.sin(base_yaw / 2.0)]),
    )

    geometry = RadialToolGeometry(ROBOT_BASE_POSITION[:2])

    def solve(position, orientation, warm_start):
        joints, ok = solver.compute_inverse_kinematics(
            frame_name=IK_FRAME,
            target_position=position,
            target_orientation=orientation,
            warm_start=warm_start,
            position_tolerance=0.005,
            orientation_tolerance=0.3,
        )
        return joints, bool(ok)

    # Reproduce the grasp-side chain exactly as the task solves it.
    grasp_pose = geometry.tool_pose(IK_GRASP_REFERENCE, DEFAULT_TOOL_TILT_DEG)
    lift_ik_object = IK_GRASP_REFERENCE + np.array([0.0, 0.0, LIFT_HEIGHT_M])
    lift_pose = geometry.tool_pose(lift_ik_object, DEFAULT_TOOL_TILT_DEG)

    warm = None
    for label, position in (
        ("grasp", grasp_pose["frame_position"]),
        ("pregrasp", grasp_pose["frame_position"] - 0.055 * grasp_pose["insertion_axis"]),
        ("approach", grasp_pose["frame_position"] - 0.075 * grasp_pose["insertion_axis"]),
        ("lift", lift_pose["frame_position"]),
    ):
        joints, ok = solve(position, grasp_pose["orientation"], warm)
        print(f"[grasp chain] {label:9s} ok={ok}")
        if not ok:
            raise RuntimeError(f"grasp-side IK regressed at {label}")
        warm = joints
    lift_solution = warm

    # Measured in-hand ball offset, taken from the validated grasp run.
    grasp_result = json.loads(ARGS.grasp_result.read_text(encoding="utf-8"))
    lifted_ball = np.asarray(grasp_result["lifted_ball_position_m"], dtype=float)
    held_offset_tool = geometry.held_offset_tool(
        lift_ik_object, lifted_ball, DEFAULT_TOOL_TILT_DEG
    )
    print(f"[handoff] held ball offset (tool frame) = {np.round(held_offset_tool, 4)}")

    rows = []
    for tilt in TILTS_DEG:
        for clearance in CLEARANCES_M:
            ball_target = np.array(
                [
                    BOWL_CENTER[0],
                    BOWL_CENTER[1],
                    TABLE_TOP_Z + BOWL_HEIGHT + TARGET_BALL_RADIUS + clearance,
                ]
            )
            ik_object = geometry.object_for_held_target(
                ball_target, held_offset_tool, tilt
            )
            pose = geometry.tool_pose(ik_object, tilt)
            attempts = {}
            for warm_label, warm_start in (
                ("lift", lift_solution),
                ("none", None),
            ):
                _, ok = solve(
                    pose["frame_position"], pose["orientation"], warm_start
                )
                attempts[warm_label] = ok
            rows.append(
                {
                    "tilt_deg": tilt,
                    "release_clearance_m": clearance,
                    "ball_target_m": ball_target.tolist(),
                    "ik_frame_m": pose["frame_position"].tolist(),
                    "frame_radius_m": float(
                        np.linalg.norm(pose["frame_position"][:2])
                    ),
                    "reachable": attempts,
                }
            )
            flags = (
                f"lift={'Y' if attempts['lift'] else '.'} "
                f"cold={'Y' if attempts['none'] else '.'}"
            )
            print(
                f"tilt={tilt:5.1f} clr={clearance:.3f} "
                f"frame={np.round(pose['frame_position'], 4)} "
                f"r={np.linalg.norm(pose['frame_position'][:2]):.4f} {flags}"
            )

    # Where on the table can the arm release the ball at all?  If the current
    # bowl position is simply outside the workspace, the fix is to move the
    # bowl in the task scene, not to keep retuning the tool pose.
    grid_x = np.round(np.arange(0.04, 0.3001, 0.02), 4)
    grid_y = np.round(np.arange(0.08, 0.3601, 0.02), 4)
    release_z = TABLE_TOP_Z + BOWL_HEIGHT + TARGET_BALL_RADIUS + GRID_CLEARANCE_M
    grid_rows = []
    print()
    print(
        f"release reachability map (tilt={GRID_TILT_DEG:.0f} deg, "
        f"ball z={release_z:.3f} m); rows = y, cols = x"
    )
    header = "      " + " ".join(f"{value:5.2f}" for value in grid_x)
    print(header)
    for y in grid_y:
        cells = []
        for x in grid_x:
            ball_target = np.array([x, y, release_z])
            ik_object = geometry.object_for_held_target(
                ball_target, held_offset_tool, GRID_TILT_DEG
            )
            pose = geometry.tool_pose(ik_object, GRID_TILT_DEG)
            _, ok_warm = solve(
                pose["frame_position"], pose["orientation"], lift_solution
            )
            _, ok_cold = solve(
                pose["frame_position"], pose["orientation"], None
            )
            reachable = bool(ok_warm or ok_cold)
            grid_rows.append(
                {
                    "ball_target_m": ball_target.tolist(),
                    "reachable": reachable,
                    "warm_start_ok": bool(ok_warm),
                    "cold_start_ok": bool(ok_cold),
                }
            )
            cells.append("  #  " if reachable else "  .  ")
        print(f"{y:5.2f} " + " ".join(cells))

    reachable_targets = [
        np.asarray(row["ball_target_m"][:2])
        for row in grid_rows
        if row["reachable"]
    ]
    bowl_ok = any(
        bool(np.allclose(target, BOWL_CENTER, atol=0.011))
        for target in reachable_targets
    )
    nearest = None
    if reachable_targets:
        distances = [
            float(np.linalg.norm(target - BOWL_CENTER))
            for target in reachable_targets
        ]
        nearest = reachable_targets[int(np.argmin(distances))].tolist()

    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(
        json.dumps(
            {
                "bowl_center_xy_m": BOWL_CENTER.tolist(),
                "held_ball_offset_tool_frame_m": held_offset_tool.tolist(),
                "grasp_tilt_deg": DEFAULT_TOOL_TILT_DEG,
                "tilt_clearance_sweep": rows,
                "release_grid_tilt_deg": GRID_TILT_DEG,
                "release_grid_clearance_m": GRID_CLEARANCE_M,
                "release_grid": grid_rows,
                "current_bowl_center_reachable": bowl_ok,
                "nearest_reachable_release_xy_m": nearest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print("probe written:", ARGS.output)

    feasible = [r for r in rows if r["reachable"]["lift"] or r["reachable"]["none"]]
    print(f"feasible tilt/clearance release poses: {len(feasible)} / {len(rows)}")
    print(f"current bowl center reachable: {bowl_ok}")
    print(f"nearest reachable release xy: {nearest}")
    return 0 if (feasible or bowl_ok) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
