#!/usr/bin/env python3
"""Solve SO-101 wrist-camera eye-in-hand calibration.

The ChArUco board must remain fixed while all samples are captured. Joint
positions are converted to ``T_base_from_gripper`` with the same calibrated
SO-101 URDF used by the simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import (
    CALIBRATION_ROOT,
    DEFAULT_BOARD_SPEC,
    PROJECT_ROOT,
    average_transforms,
    detect_charuco,
    invert_transform,
    load_board_spec,
    load_intrinsics,
    load_yaml,
    make_charuco_board,
    make_charuco_detector,
    match_charuco_points,
    matrix_list,
    reprojection_error,
    rotation_error_deg,
    save_yaml,
    transform_from_rvec_tvec,
)


DEFAULT_URDF = (
    PROJECT_ROOT.parent
    / "SO-ARM100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.urdf"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=CALIBRATION_ROOT / "raw" / "wrist_handeye" / "samples.yaml",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=CALIBRATION_ROOT / "results" / "wrist_intrinsics.yaml",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_BOARD_SPEC)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--descriptor", type=Path, default=PROJECT_ROOT / "so101_descriptor.yaml"
    )
    parser.add_argument("--gripper-frame", default="gripper_link")
    parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_ROOT / "results" / "wrist_handeye.yaml",
    )
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--max-reprojection-error", type=float, default=1.5)
    return parser.parse_args()


def make_transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def evaluate_candidate(base_from_gripper, camera_from_board, gripper_from_camera):
    base_from_board = [
        base_gripper @ gripper_from_camera @ camera_board
        for base_gripper, camera_board in zip(
            base_from_gripper, camera_from_board
        )
    ]
    reference = average_transforms(base_from_board)
    translation_errors = np.asarray(
        [
            np.linalg.norm(value[:3, 3] - reference[:3, 3])
            for value in base_from_board
        ]
    )
    rotation_errors = np.asarray(
        [rotation_error_deg(reference, value) for value in base_from_board]
    )
    return {
        "reference": reference,
        "translation_rms_m": float(
            np.sqrt(np.mean(translation_errors**2))
        ),
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_errors**2))),
        "rotation_max_deg": float(np.max(rotation_errors)),
        "score": float(
            np.sqrt(np.mean(translation_errors**2))
            + 0.001 * np.sqrt(np.mean(rotation_errors**2))
        ),
    }


def solve_candidates(base_from_gripper, camera_from_board):
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    rotations_gripper_to_base = [value[:3, :3] for value in base_from_gripper]
    translations_gripper_to_base = [value[:3, 3] for value in base_from_gripper]
    rotations_target_to_camera = [value[:3, :3] for value in camera_from_board]
    translations_target_to_camera = [value[:3, 3] for value in camera_from_board]

    candidates = {}
    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                rotations_gripper_to_base,
                translations_gripper_to_base,
                rotations_target_to_camera,
                translations_target_to_camera,
                method=method,
            )
            transform = make_transform(rotation, translation)
            if not np.all(np.isfinite(transform)):
                raise ValueError("non-finite result")
            metrics = evaluate_candidate(
                base_from_gripper, camera_from_board, transform
            )
            candidates[name] = {"transform": transform, **metrics}
        except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
            candidates[name] = {"error": str(error)}
    valid = {
        name: value for name, value in candidates.items() if "score" in value
    }
    if not valid:
        raise RuntimeError(f"Every hand-eye method failed: {candidates}")
    best_name = min(valid, key=lambda name: valid[name]["score"])
    return best_name, candidates


def run(args):
    if not args.urdf.is_file():
        raise FileNotFoundError(f"SO-101 URDF not found: {args.urdf}")
    manifest = load_yaml(args.samples)
    sample_dir = args.samples.parent
    _, camera_matrix, distortion, resolution = load_intrinsics(args.intrinsics)
    spec = load_board_spec(args.spec)
    board = make_charuco_board(spec)
    detector = make_charuco_detector(board)

    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

    solver = LulaKinematicsSolver(
        robot_description_path=str(args.descriptor),
        urdf_path=str(args.urdf),
    )
    solver.set_robot_base_pose(
        np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0])
    )
    solver_joint_names = list(solver.get_joint_names())

    base_from_gripper = []
    camera_from_board = []
    used = []
    skipped = []
    for sample in manifest.get("samples", []):
        path = sample_dir / sample["file"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append({"file": path.name, "reason": "unreadable"})
            continue
        if (image.shape[1], image.shape[0]) != resolution:
            skipped.append({"file": path.name, "reason": "resolution mismatch"})
            continue
        corners, ids, _, _ = detect_charuco(image, detector)
        count = 0 if ids is None else len(ids)
        if count < args.min_corners:
            skipped.append({"file": path.name, "reason": f"only {count} corners"})
            continue
        object_points, image_points = match_charuco_points(board, corners, ids)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            skipped.append({"file": path.name, "reason": "solvePnP failed"})
            continue
        error = reprojection_error(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        if error > args.max_reprojection_error:
            skipped.append(
                {"file": path.name, "reason": f"PnP RMS {error:.4f} px"}
            )
            continue

        joints = sample["joint_positions"]
        joint_units = sample.get("joint_units", manifest.get("joint_units"))
        if joint_units != "degrees":
            raise ValueError(
                f"Expected degree joint values, got {joint_units!r} in {path.name}"
            )
        missing = [name for name in solver_joint_names if name not in joints]
        if missing:
            raise ValueError(f"{path.name} is missing joints: {missing}")
        joint_positions = np.radians(
            np.asarray([joints[name] for name in solver_joint_names], dtype=float)
        )
        position, rotation = solver.compute_forward_kinematics(
            args.gripper_frame, joint_positions
        )
        base_from_gripper.append(make_transform(rotation, position))
        camera_from_board.append(transform_from_rvec_tvec(rvec, tvec))
        used.append(
            {
                "file": path.name,
                "corner_count": int(count),
                "pnp_reprojection_rms_px": error,
                "joint_positions_degrees": {
                    name: float(joints[name]) for name in solver_joint_names
                },
            }
        )

    if len(used) < args.min_samples:
        raise RuntimeError(
            f"Only {len(used)} usable synchronized samples; "
            f"need at least {args.min_samples}"
        )

    best_name, candidates = solve_candidates(
        base_from_gripper, camera_from_board
    )
    best = candidates[best_name]
    gripper_from_camera = best["transform"]
    result_candidates = {}
    for name, candidate in candidates.items():
        if "error" in candidate:
            result_candidates[name] = {"error": candidate["error"]}
            continue
        result_candidates[name] = {
            "T_gripper_from_camera_cv": matrix_list(candidate["transform"]),
            "translation_rms_m": candidate["translation_rms_m"],
            "translation_max_m": candidate["translation_max_m"],
            "rotation_rms_deg": candidate["rotation_rms_deg"],
            "rotation_max_deg": candidate["rotation_max_deg"],
            "score": candidate["score"],
        }

    result = {
        "schema_version": 1,
        "transform_convention": "T_A_from_B maps coordinates in B into A",
        "camera_frame": "OpenCV: +X right, +Y down, +Z forward",
        "gripper_frame": args.gripper_frame,
        "joint_source_units": "degrees",
        "solver_joint_order": solver_joint_names,
        "selected_method": best_name,
        "T_gripper_from_wrist_camera_cv": matrix_list(gripper_from_camera),
        "T_wrist_camera_cv_from_gripper": matrix_list(
            invert_transform(gripper_from_camera)
        ),
        "estimated_T_base_from_board": matrix_list(best["reference"]),
        "validation": {
            "translation_rms_m": best["translation_rms_m"],
            "translation_max_m": best["translation_max_m"],
            "rotation_rms_deg": best["rotation_rms_deg"],
            "rotation_max_deg": best["rotation_max_deg"],
        },
        "candidate_methods": result_candidates,
        "samples_used": used,
        "samples_skipped": skipped,
    }
    save_yaml(args.output, result)
    print(f"saved: {args.output}")
    print(f"selected method: {best_name}")
    print(
        f"board consistency: translation RMS={best['translation_rms_m'] * 1000:.2f} mm, "
        f"rotation RMS={best['rotation_rms_deg']:.3f} deg"
    )
    print("T_gripper_from_wrist_camera_cv:")
    print(np.array2string(gripper_from_camera, precision=6, suppress_small=True))


if __name__ == "__main__":
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        run(args)
    finally:
        app.close()
