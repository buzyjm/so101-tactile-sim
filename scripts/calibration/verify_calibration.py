#!/usr/bin/env python3
"""Create visual ChArUco pose overlays using saved camera intrinsics."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import (
    CALIBRATION_ROOT,
    DEFAULT_BOARD_SPEC,
    detect_charuco,
    load_board_spec,
    load_intrinsics,
    make_charuco_board,
    make_charuco_detector,
    match_charuco_points,
    reprojection_error,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_BOARD_SPEC)
    parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_ROOT / "reports" / "verification",
    )
    parser.add_argument("--min-corners", type=int, default=8)
    args = parser.parse_args()

    _, camera_matrix, distortion, resolution = load_intrinsics(args.intrinsics)
    spec = load_board_spec(args.spec)
    board = make_charuco_board(spec)
    detector = make_charuco_detector(board)
    args.output.mkdir(parents=True, exist_ok=True)

    used = 0
    for path in sorted(args.images.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or (image.shape[1], image.shape[0]) != resolution:
            continue
        corners, ids, _, _ = detect_charuco(image, detector)
        if ids is None or len(ids) < args.min_corners:
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
            continue
        error = reprojection_error(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        cv2.aruco.drawDetectedCornersCharuco(image, corners, ids)
        cv2.drawFrameAxes(
            image,
            camera_matrix,
            distortion,
            rvec,
            tvec,
            float(spec["square_length_m"]) * 2.0,
            2,
        )
        cv2.putText(
            image,
            f"PnP RMS {error:.3f} px",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(args.output / path.name), image)
        used += 1
    print(f"saved {used} verification overlays to {args.output}")


if __name__ == "__main__":
    main()
