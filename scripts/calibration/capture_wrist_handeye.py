#!/usr/bin/env python3
"""Capture synchronized wrist-camera images and SO-101 joint positions.

Run this inside the official teleop Docker image. The script owns both the
camera and follower serial port; do not run another teleop/control process at
the same time. Passive mode disables torque, so physically support the arm.
"""

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


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--passive", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_ROOT / "raw" / "wrist_handeye",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_BOARD_SPEC)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--min-blur", type=float, default=60.0)
    parser.add_argument("--max-samples", type=int, default=30)
    args = parser.parse_args()

    try:
        from lerobot.robots.so101_follower import (
            SO101Follower,
            SO101FollowerConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            "LeRobot was not found. Run this script inside teleop-docker."
        ) from error

    args.output.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(args.output.glob("sample_*.png"))
    if existing_frames or (args.output / "samples.yaml").exists():
        raise FileExistsError(
            f"Capture directory is not empty: {args.output}. "
            "Use a fresh --output directory so synchronized samples are not overwritten."
        )
    spec = load_board_spec(args.spec)
    board = make_charuco_board(spec)
    detector = make_charuco_detector(board)
    capture, actual = open_camera(
        args.camera, args.width, args.height, args.fps
    )

    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=True,
        disable_torque_on_disconnect=True,
    )
    robot = SO101Follower(config)
    print(f"Connecting to {args.robot_id} on {args.robot_port}...")
    try:
        robot.connect(calibrate=True)
    except Exception:
        capture.release()
        cv2.destroyAllWindows()
        raise
    if args.passive:
        print("WARNING: torque will be disabled. Hold/support the arm first.")
        confirmation = input("Type DISABLE to continue: ").strip()
        if confirmation != "DISABLE":
            capture.release()
            robot.disconnect()
            raise SystemExit("Passive mode cancelled")
        robot.bus.disable_torque()
    else:
        print(
            "Torque remains enabled. This script does not command motion; "
            "restart with --passive for careful hand-guiding."
        )

    records = []
    manifest_path = args.output / "samples.yaml"
    try:
        while len(records) < args.max_samples:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            corners, ids, _, _ = detect_charuco(frame, detector)
            corner_count = 0 if ids is None else int(len(ids))
            valid = corner_count >= args.min_corners and blur_score >= args.min_blur
            preview = draw_detection(
                frame,
                corners,
                ids,
                valid,
                (
                    f"samples {len(records):02d}/{args.max_samples} | "
                    f"corners {corner_count:02d} | blur {blur_score:.0f}"
                ),
            )
            cv2.imshow("SO-101 wrist hand-eye capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != 32:
                continue
            if not valid:
                print("rejected: board detection or sharpness is insufficient")
                continue

            observation = robot.get_observation()
            joints = {
                name: float(observation[f"{name}.pos"])
                for name in JOINT_NAMES
            }
            filename = f"sample_{len(records):03d}.png"
            path = args.output / filename
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Failed to write {path}")
            records.append(
                {
                    "file": filename,
                    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                    "captured_at_monotonic_ns": time.monotonic_ns(),
                    "joint_units": "degrees",
                    "joint_positions": joints,
                    "charuco_corners": corner_count,
                    "blur_score": blur_score,
                }
            )
            save_yaml(
                manifest_path,
                {
                    "schema_version": 1,
                    "robot_type": "so101_follower",
                    "robot_id": args.robot_id,
                    "robot_port": args.robot_port,
                    "camera_source": str(args.camera),
                    "resolution": [actual["width"], actual["height"]],
                    "fps": actual["fps"],
                    "joint_units": "degrees",
                    "board_must_remain_fixed": True,
                    "samples": records,
                },
            )
            print(f"saved {filename}: {joints}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        robot.disconnect()

    print(f"saved {len(records)} synchronized samples to {args.output}")
    if len(records) < 15:
        print("WARNING: collect at least 15 diverse poses; 20-30 is preferred.")


if __name__ == "__main__":
    main()
