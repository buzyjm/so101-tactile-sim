#!/usr/bin/env python3
"""Generate a physically sized A4 ChArUco calibration sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import (
    CALIBRATION_ROOT,
    DEFAULT_BOARD_SPEC,
    load_board_spec,
    make_charuco_board,
    make_charuco_detector,
    match_charuco_points,
    save_yaml,
)


MM_PER_INCH = 25.4
A4_LANDSCAPE_MM = (297.0, 210.0)


def mm_to_px(value_mm: float, dpi: int) -> int:
    return int(round(value_mm / MM_PER_INCH * dpi))


def project_board_axes(board_image, board, detector):
    corners, ids, _, _ = detector.detectBoard(board_image)
    if corners is None or ids is None:
        raise RuntimeError("Generated board could not be detected")
    object_points, image_points = match_charuco_points(board, corners, ids)
    homography, _ = cv2.findHomography(object_points[:, :2], image_points)
    if homography is None:
        raise RuntimeError("Could not recover the generated board coordinate frame")
    square = float(board.getSquareLength())
    model_points = np.array(
        [[[0.0, 0.0], [2.0 * square, 0.0], [0.0, 2.0 * square]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(model_points, homography)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_BOARD_SPEC)
    parser.add_argument(
        "--output-dir", type=Path, default=CALIBRATION_ROOT / "board"
    )
    args = parser.parse_args()

    spec = load_board_spec(args.spec)
    board = make_charuco_board(spec)
    detector = make_charuco_detector(board)
    dpi = int(spec.get("print", {}).get("dpi", 300))

    board_width_mm = float(spec["squares_x"]) * float(spec["square_length_m"]) * 1000
    board_height_mm = float(spec["squares_y"]) * float(spec["square_length_m"]) * 1000
    board_size_px = (
        mm_to_px(board_width_mm, dpi),
        mm_to_px(board_height_mm, dpi),
    )
    board_image = board.generateImage(board_size_px, marginSize=0, borderBits=1)

    page_size_px = (
        mm_to_px(A4_LANDSCAPE_MM[0], dpi),
        mm_to_px(A4_LANDSCAPE_MM[1], dpi),
    )
    page = np.full((page_size_px[1], page_size_px[0]), 255, dtype=np.uint8)
    offset = (
        (page_size_px[0] - board_size_px[0]) // 2,
        (page_size_px[1] - board_size_px[1]) // 2,
    )
    page[
        offset[1] : offset[1] + board_size_px[1],
        offset[0] : offset[0] + board_size_px[0],
    ] = board_image

    # Recover the OpenCV board frame from detections and draw its origin/axes in
    # the surrounding page margin, never over the usable calibration pattern.
    origin, x_point, y_point = project_board_axes(board_image, board, detector)
    x_dir = x_point - origin
    y_dir = y_point - origin
    x_dir /= np.linalg.norm(x_dir)
    y_dir /= np.linalg.norm(y_dir)
    margin_offset = mm_to_px(7.0, dpi)
    axis_length = mm_to_px(5.0, dpi)
    board_origin = origin + np.asarray(offset, dtype=float)
    legend_origin = board_origin - margin_offset * (x_dir + y_dir)
    legend_origin_i = tuple(np.round(legend_origin).astype(int))
    board_origin_i = tuple(np.round(board_origin).astype(int))
    x_end = tuple(np.round(legend_origin + axis_length * x_dir).astype(int))
    y_end = tuple(np.round(legend_origin + axis_length * y_dir).astype(int))
    cv2.arrowedLine(
        page, legend_origin_i, x_end, 0, 3, cv2.LINE_AA, tipLength=0.25
    )
    cv2.arrowedLine(
        page, legend_origin_i, y_end, 0, 3, cv2.LINE_AA, tipLength=0.25
    )
    # The axes are only a direction legend. A separate leader identifies the
    # actual OpenCV board origin at the outer top-left pattern corner.
    leader_start = tuple(
        np.round(board_origin - 0.58 * margin_offset * (x_dir + y_dir)).astype(int)
    )
    cv2.arrowedLine(
        page, leader_start, board_origin_i, 0, 2, cv2.LINE_AA, tipLength=0.22
    )
    cv2.circle(page, board_origin_i, 5, 0, 2, cv2.LINE_AA)

    pil_page = Image.fromarray(page).convert("RGB")
    draw = ImageDraw.Draw(pil_page)
    font = ImageFont.load_default(size=24)
    draw.text((x_end[0] + 8, x_end[1] - 12), "X+", fill="black", font=font)
    draw.text((y_end[0] + 8, y_end[1] - 12), "Y+", fill="black", font=font)
    draw.text(
        (leader_start[0] - 48, leader_start[1] - 24),
        "O ->",
        fill="black",
        font=font,
    )
    footer = (
        f"ChArUco {spec['squares_x']}x{spec['squares_y']} | "
        f"square={float(spec['square_length_m']) * 1000:.1f} mm | "
        f"marker={float(spec['marker_length_m']) * 1000:.1f} mm | "
        "O=TOP-LEFT PATTERN CORNER | PRINT 100% / ACTUAL SIZE"
    )
    draw.text((40, page_size_px[1] - 42), footer, fill="black", font=font)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "charuco_a4_8x6_25mm.png"
    pdf_path = args.output_dir / "charuco_a4_8x6_25mm.pdf"
    crop_path = args.output_dir / "charuco_pattern.png"
    used_spec_path = args.output_dir / "charuco_spec.yaml"
    pil_page.save(png_path, dpi=(dpi, dpi))
    pil_page.save(pdf_path, "PDF", resolution=dpi)
    Image.fromarray(board_image).save(crop_path, dpi=(dpi, dpi))
    save_yaml(used_spec_path, spec)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"spec: {used_spec_path}")
    print(f"page: {A4_LANDSCAPE_MM[0]:.0f} x {A4_LANDSCAPE_MM[1]:.0f} mm at {dpi} dpi")
    print(f"board: {board_width_mm:.1f} x {board_height_mm:.1f} mm")
    print("Print using Actual size / 100%; disable Fit to page.")


if __name__ == "__main__":
    main()
