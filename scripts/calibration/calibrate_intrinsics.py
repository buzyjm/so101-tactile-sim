#!/usr/bin/env python3
"""Solve OpenCV pinhole intrinsics from captured ChArUco images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import (
    CALIBRATION_ROOT,
    DEFAULT_BOARD_SPEC,
    detect_charuco,
    load_board_spec,
    make_charuco_board,
    make_charuco_detector,
    match_charuco_points,
    reprojection_error,
    save_yaml,
)


def collect_detections(image_dir, board, detector, min_corners):
    detections = []
    skipped = []
    image_size = None
    for path in sorted(Path(image_dir).glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append({"file": path.name, "reason": "unreadable"})
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            skipped.append({"file": path.name, "reason": "resolution mismatch"})
            continue
        corners, ids, _, _ = detect_charuco(image, detector)
        count = 0 if ids is None else len(ids)
        if count < min_corners:
            skipped.append(
                {"file": path.name, "reason": f"only {count} ChArUco corners"}
            )
            continue
        object_points, image_points = match_charuco_points(board, corners, ids)
        detections.append(
            {
                "path": path,
                "object_points": object_points,
                "image_points": image_points,
                "corner_count": int(count),
            }
        )
    return detections, skipped, image_size


def solve(detections, image_size):
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [item["object_points"] for item in detections],
        [item["image_points"] for item in detections],
        image_size,
        None,
        None,
        flags=0,
    )
    errors = [
        reprojection_error(
            item["object_points"],
            item["image_points"],
            rvec,
            tvec,
            matrix,
            distortion,
        )
        for item, rvec, tvec in zip(detections, rvecs, tvecs)
    ]
    return float(rms), matrix, distortion, rvecs, tvecs, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=("top", "wrist"))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_BOARD_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--min-views", type=int, default=15)
    parser.add_argument(
        "--reject-view-error",
        type=float,
        default=1.5,
        help="Re-solve once without views above this pixel RMS; <=0 disables",
    )
    args = parser.parse_args()

    image_dir = args.images or (
        CALIBRATION_ROOT / "raw" / f"{args.name}_intrinsics"
    )
    output = args.output or (
        CALIBRATION_ROOT / "results" / f"{args.name}_intrinsics.yaml"
    )
    report_dir = args.report_dir or (
        CALIBRATION_ROOT / "reports" / f"{args.name}_intrinsics"
    )

    spec = load_board_spec(args.spec)
    board = make_charuco_board(spec)
    detector = make_charuco_detector(board)
    detections, skipped, image_size = collect_detections(
        image_dir, board, detector, args.min_corners
    )
    if image_size is None:
        raise RuntimeError(f"No readable PNG images in {image_dir}")
    if len(detections) < args.min_views:
        raise RuntimeError(
            f"Only {len(detections)} usable views; need at least {args.min_views}"
        )

    first_solution = solve(detections, image_size)
    _, _, _, _, _, first_errors = first_solution
    rejected_by_error = []
    if args.reject_view_error > 0:
        keep = []
        candidate_rejections = []
        for item, error in zip(detections, first_errors):
            if error <= args.reject_view_error:
                keep.append(item)
            else:
                candidate_rejections.append(
                    {
                        "file": item["path"].name,
                        "reason": f"view RMS {error:.4f} px",
                    }
                )
        if len(keep) >= args.min_views and len(keep) < len(detections):
            detections = keep
            rejected_by_error = candidate_rejections

    rms, matrix, distortion, rvecs, tvecs, errors = solve(
        detections, image_size
    )
    per_view = [
        {
            "file": item["path"].name,
            "corner_count": item["corner_count"],
            "reprojection_rms_px": float(error),
        }
        for item, error in zip(detections, errors)
    ]

    result = {
        "schema_version": 1,
        "camera_name": args.name,
        "model": "opencv_pinhole",
        "resolution": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": matrix.tolist(),
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "rms_reprojection_error_px": rms,
        "mean_view_reprojection_error_px": float(np.mean(errors)),
        "max_view_reprojection_error_px": float(np.max(errors)),
        "board_spec": spec,
        "images_directory": str(Path(image_dir).resolve()),
        "views_used": len(detections),
        "per_view": per_view,
        "skipped": skipped + rejected_by_error,
    }
    save_yaml(output, result)

    report_dir.mkdir(parents=True, exist_ok=True)
    for item, rvec, tvec, error in zip(detections, rvecs, tvecs, errors):
        image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
        cv2.drawFrameAxes(
            image,
            matrix,
            distortion,
            rvec,
            tvec,
            float(spec["square_length_m"]) * 2.0,
            2,
        )
        cv2.putText(
            image,
            f"RMS {error:.3f} px",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(report_dir / item["path"].name), image)

    quality = "GOOD" if rms < 0.5 else "USABLE" if rms < 1.0 else "REVIEW"
    print(f"saved: {output}")
    print(f"views: {len(detections)}")
    print(f"RMS: {rms:.4f} px [{quality}]")
    print(f"debug overlays: {report_dir}")


if __name__ == "__main__":
    main()
