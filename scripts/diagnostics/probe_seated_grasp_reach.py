"""Map which seated-grasp poses the 5-DOF SO-101 can actually reach.

seated_grasp_frame() solves where the tool must be for the ball to rest on the
fixed pad's centroid, but that pose has no IK solution at 15 deg tilt.  This
sweeps tool tilt and a neighbourhood of the seated frame so the answer is
"unreachable everywhere" or "reachable at tilt X", not a single failed call.
"""

from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--usd", type=Path, default=ROOT / "lab_scene_task.usda")
ap.add_argument("--output", type=Path,
                default=ROOT / "tactile_logs" / "seated_grasp_reach.json")
ARGS = ap.parse_args()

from isaacsim import SimulationApp
APP = SimulationApp({"headless": True, "active_gpu": ARGS.gpu,
                     "physics_gpu": ARGS.gpu, "multi_gpu": False})

import omni.usd
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import Gf, UsdGeom

from tactile_taxels import load_taxel_positions_m
from scripts.tasks.grasp_geometry import RadialToolGeometry
from scene_config import (GRIPPER_LINK_PRIM_PATH, PROJECT_ROOT,
                          ROBOT_BASE_POSITION, ROBOT_BASE_YAW_DEG, TABLE_TOP_Z,
                          TACTILE_ROOT_PRIM_PATHS, TARGET_BALL_RADIUS,
                          TARGET_BALL_XY_TASK_READY_PLACEHOLDER)

IK_FRAME = "gripper_frame_link"
GF = GRIPPER_LINK_PRIM_PATH + "/gripper_frame_link"
TAX = np.asarray(load_taxel_positions_m())
BALL = np.array([TARGET_BALL_XY_TASK_READY_PLACEHOLDER[0],
                 TARGET_BALL_XY_TASK_READY_PLACEHOLDER[1],
                 TABLE_TOP_Z + TARGET_BALL_RADIUS])
mat = lambda m: np.array([[m[i][j] for j in range(4)] for i in range(4)])
rot = lambda m: np.array([[m[i][j] for j in range(3)] for i in range(3)]).T


def main() -> int:
    omni.usd.get_context().open_stage(str(ARGS.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    W = lambda p: UsdGeom.Xformable(
        stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)
    Wg, Wf = W(GRIPPER_LINK_PRIM_PATH), W(GF)
    link_in_frame = Wg * Wf.GetInverse()
    frame_origin_in_link = np.asarray((Wf * Wg.GetInverse()).ExtractTranslation())
    pad_in_link = W(TACTILE_ROOT_PRIM_PATHS[0]) * Wg.GetInverse()
    centroid = TAX.mean(axis=0)
    for off in np.linspace(0.020, 0.050, 3001):
        c = centroid + np.array([0.0, 0.0, off])
        if abs(np.linalg.norm(TAX - c, axis=1).min() - TARGET_BALL_RADIUS) < 2e-5:
            break
    seated = (np.append(c, 1.0) @ mat(pad_in_link))[:3]

    solver = LulaKinematicsSolver(
        robot_description_path=str(PROJECT_ROOT / "so101_descriptor.yaml"),
        urdf_path=str(PROJECT_ROOT.parent / "SO-ARM100" / "Simulation"
                      / "SO101" / "so101_new_calib.urdf"))
    yaw = math.radians(ROBOT_BASE_YAW_DEG)
    solver.set_robot_base_pose(
        np.asarray(ROBOT_BASE_POSITION),
        np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
    geometry = RadialToolGeometry(ROBOT_BASE_POSITION[:2])

    def seated_frame(tilt):
        obj = BALL.copy()
        for _ in range(6):
            pose = geometry.tool_pose(obj, tilt)
            T = np.asarray(pose["tool_rotation"])
            M = Gf.Matrix4d(); M.SetRotate(Gf.Matrix3d(*T.T.flatten()))
            Rl = rot(link_in_frame * M)
            f = Rl @ frame_origin_in_link + (BALL - Rl @ seated)
            fy = math.atan2(f[1], f[0])
            if abs(fy - geometry.motion_yaw(obj)) < 3e-4:
                break
            r = float(np.linalg.norm(obj[:2]))
            obj = np.array([r * math.cos(fy), r * math.sin(fy), obj[2]])
        return f, pose["orientation"]

    def reachable(pos, orient):
        for warm in (None,):
            for tol in (0.005, 0.010, 0.020):
                _, ok = solver.compute_inverse_kinematics(
                    frame_name=IK_FRAME, target_position=pos,
                    target_orientation=orient, warm_start=warm,
                    position_tolerance=tol, orientation_tolerance=0.3)
                if ok:
                    return True, tol
        return False, None

    rows = []
    print(f"{'tilt':>5s} {'frame (x,y,z)':>26s} {'radius':>7s}  IK")
    for tilt in range(0, 65, 5):
        f, q = seated_frame(float(tilt))
        ok, tol = reachable(f, q)
        rows.append({"tilt_deg": tilt, "frame_m": f.tolist(),
                     "radius_m": float(np.linalg.norm(f[:2])),
                     "reachable": ok, "tolerance_m": tol})
        print(f"{tilt:5d} ({f[0]:+.4f},{f[1]:+.4f},{f[2]:+.4f}) {np.linalg.norm(f[:2]):7.4f}  "
              f"{'OK tol=%.3f' % tol if ok else 'no'}", flush=True)

    # How far from the seated frame does reachability begin?
    f15, q15 = seated_frame(15.0)
    print("\nneighbourhood of the 15 deg seated frame (radial / vertical shift):")
    print(f"{'dr mm':>6s} {'dz mm':>6s}  IK")
    unit = f15[:2] / np.linalg.norm(f15[:2])
    grid = []
    for dr in (0, 10, 20, 30, 40):
        for dz in (0, 10, 20, 30):
            p = f15 + np.array([unit[0] * dr / 1000, unit[1] * dr / 1000, dz / 1000])
            ok, tol = reachable(p, q15)
            grid.append({"dr_mm": dr, "dz_mm": dz, "reachable": ok})
            if ok:
                print(f"{dr:6d} {dz:6d}  OK")
    if not any(g["reachable"] for g in grid):
        print("  none reachable in the sampled neighbourhood")

    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(
        {"seated_ball_in_gripper_link_mm": (seated * 1000).tolist(),
         "tilt_sweep": rows, "neighbourhood": grid}, indent=2) + "\n",
        encoding="utf-8")
    print("\nwritten:", ARGS.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
