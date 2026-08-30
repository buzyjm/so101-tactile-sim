#!/usr/bin/env python3
"""Interactively capture ChArUco images from one real camera."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from common import (
    CALIBRATION_ROOT,
    DEFAULT_BOARD_SPEC,
    detect_charuco,
    draw_detection,
    load_board_spec,
    make_charuco_board,
    make_charuco_detector,
    open_camera,
    save_yaml,
)


def main():
    parser = argparse.ArgumentParser(
        description="Capture lossless ChArUco frames. Press SPACE to save."
    )
    parser.add_argument("--camera", required=True, help="Index or /dev/video path")
    parser.add_argument("--name", required=True, choices=("top", "wrist"))
    parser.add_argument("--purpose", default="intrinsics")
    parser.add_argument("--spec", type=Path, default=DEFAULT_BOARD_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--min-blur", type=float, default=60.0)
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()

    output = args.output or (
        CALIBRATION_ROOT / "raw" / f"{args.name}_{args.purpose}"
    )
    output.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output.glob("frame_*.png"))
    if existing_frames or (output / "capture.yaml").exists():
        raise FileExistsError(
            f"Capture directory is not empty: {output}. "
            "Use a fresh --output directory so sessions are not mixed or overwritten."
        )

    spec = load_board_spec(args.spec)
    board = make_charuco_board(spec)
    detector = make_charuco_detector(board)
    capture, actual = open_camera(
        args.camera, args.width, args.height, args.fps
    )
    records = []
    metadata_path = output / "capture.yaml"
    print(f"camera={args.camera} actual={actual}")
    print(f"output={output}")
    print("Move the board around the full image. SPACE saves; Q quits.")

    try:
        while len(records) < args.max_images:
            ok, frame = capture.read()
            if not ok:
                print("WARNING: frame read failed")
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            corners, ids, _, _ = detect_charuco(frame, detector)
            corner_count = 0 if ids is None else int(len(ids))
            valid = corner_count >= args.min_corners and blur_score >= args.min_blur
            message = (
                f"saved {len(records):02d}/{args.max_images} | "
                f"corners {corner_count:02d} | blur {blur_score:.0f}"
            )
            preview = draw_detection(frame, corners, ids, valid, message)
            cv2.imshow(f"SO-101 {args.name} camera calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != 32:
                continue
            if not valid:
                print(
                    "rejected: need at least "
                    f"{args.min_corners} corners and blur >= {args.min_blur:.0f}"
                )
                continue

            filename = f"frame_{len(records):03d}.png"
            path = output / filename
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Failed to write {path}")
            record = {
                "file": filename,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "charuco_corners": corner_count,
                "blur_score": blur_score,
            }
            records.append(record)
            save_yaml(
                metadata_path,
                {
                    "schema_version": 1,
                    "camera_name": args.name,
                    "purpose": args.purpose,
                    "camera_source": str(args.camera),
                    "requested": {
                        "width": args.width,
                        "height": args.height,
                        "fps": args.fps,
                    },
                    "actual": actual,
                    "board_spec": str(Path(args.spec).resolve()),
                    "frames": records,
                },
            )
            print(f"saved {path} ({corner_count} corners, blur={blur_score:.0f})")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"complete: {len(records)} images in {output}")
    if len(records) < 20:
        print("WARNING: fewer than 20 images; collect more before calibration.")


if __name__ == "__main__":
    main()
