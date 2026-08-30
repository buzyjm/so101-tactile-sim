#!/usr/bin/env python3
"""Estimate the fixed top-camera pose in the SO-101 base frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import (
    CALIBRATION_ROOT,
    DEFAULT_BOARD_SPEC,
    average_transforms,
    detect_charuco,
    invert_transform,
    load_board_spec,
    load_intrinsics,
    make_charuco_board,
    make_charuco_detector,
    match_charuco_points,
    matrix_list,
    reprojection_error,
    rotation_error_deg,
    save_yaml,
    transform_from_rvec_tvec,
    transform_from_xyz_rpy,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "The printed O/X/Y board frame must be measured in the robot base "
            "frame. T_A_from_B maps points in B into A."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=CALIBRATION_ROOT / "raw" / "top_extrinsic",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=CALIBRATION_ROOT / "results" / "top_intrinsics.yaml",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_BOARD_SPEC)
    parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_ROOT / "results" / "top_extrinsic.yaml",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=CALIBRATION_ROOT / "reports" / "top_extrinsic",
    )
    parser.add_argument("--board-x", type=float, required=True)
    parser.add_argument("--board-y", type=float, required=True)
    parser.add_argument("--board-z", type=float, default=0.0)
    parser.add_argument("--board-roll-deg", type=float, default=0.0)
    parser.add_argument("--board-pitch-deg", type=float, default=0.0)
    parser.add_argument("--board-yaw-deg", type=float, default=0.0)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--max-reprojection-error", type=float, default=1.5)
    args = parser.parse_args()

    _, camera_matrix, distortion, resolution = load_intrinsics(args.intrinsics)
    spec = load_board_spec(args.spec)
    board = make_charuco_board(spec)
    detector = make_charuco_detector(board)

    camera_from_board_samples = []
    reports = []
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.images.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            reports.append({"file": path.name, "status": "unreadable"})
            continue
        if (image.shape[1], image.shape[0]) != resolution:
            reports.append({"file": path.name, "status": "resolution mismatch"})
            continue
        corners, ids, _, _ = detect_charuco(image, detector)
        count = 0 if ids is None else len(ids)
        if count < args.min_corners:
            reports.append(
                {"file": path.name, "status": f"only {count} corners"}
            )
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
            reports.append({"file": path.name, "status": "solvePnP failed"})
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
            reports.append(
                {
                    "file": path.name,
                    "status": "rejected",
                    "reprojection_rms_px": error,
                }
            )
            continue
        camera_from_board_samples.append(transform_from_rvec_tvec(rvec, tvec))
        reports.append(
            {
                "file": path.name,
                "status": "used",
                "corner_count": int(count),
                "reprojection_rms_px": error,
            }
        )
        cv2.drawFrameAxes(
            image,
            camera_matrix,
            distortion,
            rvec,
            tvec,
            float(spec["square_length_m"]) * 2.0,
            2,
        )
        cv2.imwrite(str(args.report_dir / path.name), image)

    if not camera_from_board_samples:
        raise RuntimeError(f"No usable top-camera images in {args.images}")

    camera_from_board = average_transforms(camera_from_board_samples)
    base_from_board = transform_from_xyz_rpy(
        (args.board_x, args.board_y, args.board_z),
        (
            args.board_roll_deg,
            args.board_pitch_deg,
            args.board_yaw_deg,
        ),
    )
    base_from_camera_cv = base_from_board @ invert_transform(camera_from_board)
    camera_cv_from_base = invert_transform(base_from_camera_cv)

    translation_spread = [
        float(np.linalg.norm(sample[:3, 3] - camera_from_board[:3, 3]))
        for sample in camera_from_board_samples
    ]
    rotation_spread = [
        rotation_error_deg(camera_from_board, sample)
        for sample in camera_from_board_samples
    ]
    result = {
        "schema_version": 1,
        "transform_convention": "T_A_from_B maps coordinates in B into A",
        "camera_frame": "OpenCV: +X right, +Y down, +Z forward",
        "base_frame": "+X table long edge, +Y into table, +Z up",
        "board_pose_input": {
            "xyz_m": [args.board_x, args.board_y, args.board_z],
            "rpy_deg_xyz": [
                args.board_roll_deg,
                args.board_pitch_deg,
                args.board_yaw_deg,
            ],
            "note": "Pose of the O/X/Y frame printed beside the board",
        },
        "T_camera_cv_from_board": matrix_list(camera_from_board),
        "T_base_from_board": matrix_list(base_from_board),
        "T_base_from_top_camera_cv": matrix_list(base_from_camera_cv),
        "T_top_camera_cv_from_base": matrix_list(camera_cv_from_base),
        "sample_count": len(camera_from_board_samples),
        "sample_translation_spread_max_m": max(translation_spread),
        "sample_rotation_spread_max_deg": max(rotation_spread),
        "per_view": reports,
    }
    save_yaml(args.output, result)
    print(f"saved: {args.output}")
    print(f"used samples: {len(camera_from_board_samples)}")
    print("T_base_from_top_camera_cv:")
    print(np.array2string(base_from_camera_cv, precision=6, suppress_small=True))
    print(
        "IMPORTANT: OpenCV-to-USD camera-axis conversion is intentionally "
        "applied later when importing this result into the scene."
    )


if __name__ == "__main__":
    main()
