"""Shared utilities for real-camera calibration.

Transform names use ``T_A_from_B`` throughout: multiplying a homogeneous point
expressed in frame B by the matrix produces the point expressed in frame A.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_ROOT = PROJECT_ROOT / "calibration"
DEFAULT_BOARD_SPEC = Path(__file__).with_name("board_spec.yaml")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def save_yaml(path: str | Path, data: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
    return output


def load_board_spec(path: str | Path = DEFAULT_BOARD_SPEC) -> dict:
    spec = load_yaml(path)
    required = (
        "dictionary",
        "squares_x",
        "squares_y",
        "square_length_m",
        "marker_length_m",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Board spec is missing: {', '.join(missing)}")
    if float(spec["marker_length_m"]) >= float(spec["square_length_m"]):
        raise ValueError("marker_length_m must be smaller than square_length_m")
    return spec


def make_charuco_board(spec: dict):
    dictionary_name = str(spec["dictionary"])
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown OpenCV ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name)
    )
    board = cv2.aruco.CharucoBoard(
        (int(spec["squares_x"]), int(spec["squares_y"])),
        float(spec["square_length_m"]),
        float(spec["marker_length_m"]),
        dictionary,
    )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(spec.get("legacy_pattern", False)))
    return board


def make_charuco_detector(board):
    detector_parameters = cv2.aruco.DetectorParameters()
    detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco_parameters = cv2.aruco.CharucoParameters()
    return cv2.aruco.CharucoDetector(
        board,
        charuco_parameters,
        detector_parameters,
    )


def detect_charuco(image: np.ndarray, detector):
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if corners is None or ids is None:
        return None, None, marker_corners, marker_ids
    return corners, ids, marker_corners, marker_ids


def match_charuco_points(board, corners, ids):
    object_points, image_points = board.matchImagePoints(corners, ids)
    return (
        np.asarray(object_points, dtype=np.float32).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
    )


def parse_camera_source(value: str):
    stripped = str(value).strip()
    return int(stripped) if stripped.lstrip("-").isdigit() else stripped


def open_camera(source, width: int, height: int, fps: int):
    capture = cv2.VideoCapture(parse_camera_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {source!r}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(15):
        capture.read()
    actual = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    if actual["width"] != width or actual["height"] != height:
        capture.release()
        raise RuntimeError(
            f"Camera returned {actual['width']}x{actual['height']}, "
            f"expected {width}x{height}"
        )
    return capture, actual


def draw_detection(image, corners, ids, valid: bool, message: str):
    preview = image.copy()
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)
    color = (40, 210, 40) if valid else (30, 30, 230)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 62), (0, 0, 0), -1)
    cv2.putText(
        preview,
        message,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "SPACE save | Q quit",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return preview


def load_intrinsics(path: str | Path):
    data = load_yaml(path)
    matrix = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(
        data["distortion_coefficients"], dtype=np.float64
    ).reshape(-1, 1)
    resolution = tuple(int(v) for v in data["resolution"])
    return data, matrix, distortion, resolution


def transform_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def transform_from_xyz_rpy(
    xyz: Iterable[float], rpy_deg: Iterable[float]
) -> np.ndarray:
    roll, pitch, yaw = np.radians(np.asarray(tuple(rpy_deg), dtype=float))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = np.asarray(tuple(xyz), dtype=float)
    return transform


def average_rotations(rotations: Iterable[np.ndarray]) -> np.ndarray:
    accumulator = np.sum(np.asarray(list(rotations), dtype=float), axis=0)
    u, _, vt = np.linalg.svd(accumulator)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    transforms = [np.asarray(value, dtype=float) for value in transforms]
    if not transforms:
        raise ValueError("Cannot average an empty transform collection")
    result = np.eye(4)
    result[:3, :3] = average_rotations([value[:3, :3] for value in transforms])
    result[:3, 3] = np.median(
        np.asarray([value[:3, 3] for value in transforms]), axis=0
    )
    return result


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def reprojection_error(
    object_points,
    image_points,
    rvec,
    tvec,
    camera_matrix,
    distortion,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    projected = projected.reshape(-1, 2)
    observed = np.asarray(image_points).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).tolist()
