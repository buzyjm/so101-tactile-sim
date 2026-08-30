"""Recover where the ball and bowl actually sat, from joint data alone.

The scene places them from placeholders read off a photograph, and the dataset
carries no object pose, so this was assumed to need camera calibration.  It
does not: forward kinematics on the recorded joints puts the gripper in the
world, the tactile array says where on the pad the ball is pressing, and the
ball centre is one radius along the pad normal from there.  Averaged over the
stable part of each grasp that gives a per-episode ball position, and the pose
at release gives the bowl.

Everything is measured except the pad-to-gripper transform, which is read from
the scene's own CAD.
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

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--episodes", type=int, nargs="*", default=list(range(21)))
ap.add_argument("--usd", type=Path, default=ROOT / "lab_scene_task.usda")
ap.add_argument("--output", type=Path,
                default=ROOT / "tactile_logs" / "recovered_object_poses.json")
ARGS = ap.parse_args()

from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "active_gpu": ARGS.gpu,
                     "physics_gpu": ARGS.gpu, "multi_gpu": False})

import omni.usd
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import Gf, UsdGeom

from scripts.tactile.analyze_real_arm_tracking import to_degrees
from tactile_taxels import load_taxel_positions_m
from scene_config import (
    GRIPPER_LINK_PRIM_PATH, MOVING_JAW_LINK_PRIM_PATH, PROJECT_ROOT,
    ROBOT_BASE_POSITION, ROBOT_BASE_YAW_DEG, TACTILE_ROOT_PRIM_PATHS,
    TARGET_BALL_RADIUS,
)

# Gripper joint, from the scene's own articulation.
JAW_PIVOT_POSITION = Gf.Vec3d(0.0202, 0.0188, -0.0234)
JAW_PIVOT_ROTATION = Gf.Quatd(
    -0.70710546, Gf.Vec3d(-0.7071081, 1.8536205e-8, -1.8536273e-8))

DATASET = "Jingyi-Z/sotac"
GRIPPER_FRAME_PRIM = GRIPPER_LINK_PRIM_PATH + "/gripper_frame_link"


def matrix_of(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)])


def main() -> int:
    omni.usd.get_context().open_stage(str(ARGS.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    world_of = lambda p: UsdGeom.Xformable(
        stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)
    # Pads are rigid on the gripper, so their offset from the IK frame is a
    # constant we can read straight out of the scene.
    frame_inverse = world_of(GRIPPER_FRAME_PRIM).GetInverse()
    # The fixed pad rides on gripper_link, so its offset from the IK frame is
    # genuinely constant.  The moving pad does not: it sits on the jaw that the
    # gripper joint rotates, and at a 44 deg grasp that carries it 55 mm from
    # where the rest pose puts it -- exactly the discrepancy seen when both
    # pads were treated as fixed.
    fixed_pad_in_frame = matrix_of(
        world_of(TACTILE_ROOT_PRIM_PATHS[0]) * frame_inverse)
    pad_in_jaw = world_of(TACTILE_ROOT_PRIM_PATHS[1]) * world_of(
        MOVING_JAW_LINK_PRIM_PATH).GetInverse()
    jaw_pivot = Gf.Matrix4d().SetRotate(JAW_PIVOT_ROTATION)
    jaw_pivot.SetTranslateOnly(JAW_PIVOT_POSITION)
    gripper_link_in_frame = world_of(GRIPPER_LINK_PRIM_PATH) * frame_inverse

    def moving_pad_in_frame(gripper_deg: float) -> np.ndarray:
        """Moving pad relative to the IK frame at a given jaw opening."""
        swing = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(0, 0, 1), float(gripper_deg)))
        return matrix_of(pad_in_jaw * swing * jaw_pivot * gripper_link_in_frame)

    solver = LulaKinematicsSolver(
        robot_description_path=str(PROJECT_ROOT / "so101_descriptor.yaml"),
        urdf_path=str(PROJECT_ROOT.parent / "SO-ARM100" / "Simulation"
                      / "SO101" / "so101_new_calib.urdf"))
    yaw = math.radians(ROBOT_BASE_YAW_DEG)
    solver.set_robot_base_pose(
        np.asarray(ROBOT_BASE_POSITION),
        np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))

    taxels = np.asarray(load_taxel_positions_m())
    from huggingface_hub import hf_hub_download, HfApi
    import pyarrow.parquet as pq
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(DATASET, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet"))
    tables = [pq.read_table(hf_hub_download(DATASET, f, repo_type="dataset"))
              for f in files]

    results, ball_estimates, bowl_estimates = [], [], []
    for episode in ARGS.episodes:
        state = tactile = None
        for table in tables:
            index = np.asarray(table.column("episode_index"))
            keep = index == episode
            if not keep.any():
                continue
            state = to_degrees(
                np.array(table.column("observation.state").to_pylist())[keep])
            tactile = np.array(table.column(
                "observation.sensors.paxini_fingertip").to_pylist())[keep]
            break
        if state is None:
            continue
        tactile = tactile.reshape(len(tactile), 2, 52, 3)
        magnitudes = np.linalg.norm(tactile, axis=-1)
        holding = magnitudes.sum(axis=(1, 2)) > 0

        # Each pad gives its own estimate of the centre, one radius along its
        # normal from where the force is centred.  Both describe the same ball,
        # so their disagreement measures whether the chain is trustworthy --
        # using one pad alone hides that entirely.
        centres, disagreements = [], []
        for frame in np.flatnonzero(holding):
            if magnitudes[frame, 0].sum() <= 0 or magnitudes[frame, 1].sum() <= 0:
                continue
            position, rotation = solver.compute_forward_kinematics(
                "gripper_frame_link", np.radians(state[frame, :5]))
            frame_matrix = np.eye(4)
            frame_matrix[:3, :3] = np.asarray(rotation).T
            frame_matrix[3, :3] = np.asarray(position)
            pads = [fixed_pad_in_frame,
                    moving_pad_in_frame(state[frame, 5])]
            estimates = []
            for pad_index in (0, 1):
                pad = pads[pad_index] @ frame_matrix
                weights = magnitudes[frame, pad_index]
                contact_local = (weights / weights.sum()) @ taxels
                ball_local = contact_local + np.array(
                    [0.0, 0.0, TARGET_BALL_RADIUS])
                estimates.append((np.append(ball_local, 1.0) @ pad)[:3])
            disagreements.append(float(np.linalg.norm(
                estimates[0] - estimates[1])))
            centres.append(np.mean(estimates, axis=0))
        if len(centres) < 10:
            continue
        centres = np.array(centres)
        disagreement = float(np.median(disagreements)) * 1000
        core = centres[len(centres) // 4: -len(centres) // 4 or None]
        ball = core.mean(axis=0)
        release = centres[-1]
        # Before the lift the ball is still on the table, so its centre must be
        # one radius up.  That is an independent check on the whole chain.
        resting_z = float(np.median(centres[:10, 2]))
        ball_estimates.append(ball)
        bowl_estimates.append(release)
        results.append({"episode": episode, "hold_frames": int(len(centres)),
                        "ball_xyz_m": ball.tolist(),
                        "release_xyz_m": release.tolist(),
                        "pad_disagreement_mm": disagreement,
                        "resting_z_m": resting_z,
                        "ball_spread_mm": float(
                            np.linalg.norm(core.std(axis=0)) * 1000)})
        print(f"ep {episode:2d}  ball ({ball[0]:+.4f},{ball[1]:+.4f},"
              f"{ball[2]:+.4f})  spread {results[-1]['ball_spread_mm']:5.1f}"
              f"  pads disagree {disagreement:6.1f} mm  resting z "
              f"{resting_z * 1000:6.1f} mm (expect 30.0)", flush=True)

    if ball_estimates:
        ball = np.array(ball_estimates)
        bowl = np.array(bowl_estimates)
        print(f"\nball   x {ball[:,0].mean():+.4f}±{ball[:,0].std():.4f}   "
              f"y {ball[:,1].mean():+.4f}±{ball[:,1].std():.4f}   "
              f"z {ball[:,2].mean():+.4f}±{ball[:,2].std():.4f}")
        print(f"bowl   x {bowl[:,0].mean():+.4f}±{bowl[:,0].std():.4f}   "
              f"y {bowl[:,1].mean():+.4f}±{bowl[:,1].std():.4f}")
        print(f"\nscene placeholders: ball (0.1100, 0.3000, 0.0300)   "
              f"bowl (0.2200, 0.2300)")
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        ARGS.output.write_text(json.dumps({
            "episodes": results,
            "ball_mean_m": ball.mean(axis=0).tolist(),
            "ball_std_m": ball.std(axis=0).tolist(),
            "bowl_mean_m": bowl.mean(axis=0).tolist(),
            "bowl_std_m": bowl.std(axis=0).tolist(),
        }, indent=2) + "\n", encoding="utf-8")
        print("written:", ARGS.output)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        status = 1
    finally:
        APP.close()
    raise SystemExit(status)
