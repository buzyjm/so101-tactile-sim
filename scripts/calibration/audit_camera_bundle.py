#!/usr/bin/env python3
"""Audit an imported two-camera calibration bundle and normalize intrinsics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


INTRINSIC_FILES = {
    "top": "top_intrinsics.yaml",
    "wrist": "wrist_intrinsics.yaml",
}
EXTRINSIC_FILES = {
    "top": "top_eye_to_hand_extrinsics.yaml",
    "wrist": "wrist_eye_in_hand_extrinsics.yaml",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


def audit_intrinsics(camera_name: str, source: dict) -> tuple[dict, dict]:
    width = int(source["image_width"])
    height = int(source["image_height"])
    matrix = np.asarray(source["camera_matrix"], dtype=float).reshape(3, 3)
    distortion = np.asarray(source["dist_coeffs"], dtype=float).reshape(-1)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    rms = float(source["rms_reproj_error_px"])
    per_image = list(source.get("per_image", []))
    per_view_errors = [
        float(item["reproj_error_px"])
        for item in per_image
        if "reproj_error_px" in item
    ]

    checks = {
        "matrix_shape_valid": matrix.shape == (3, 3),
        "finite": bool(
            np.all(np.isfinite(matrix)) and np.all(np.isfinite(distortion))
        ),
        "positive_focal_lengths": fx > 0.0 and fy > 0.0,
        "principal_point_inside_image": 0.0 <= cx < width and 0.0 <= cy < height,
        "at_least_15_views": int(source.get("num_images_used", 0)) >= 15,
        "rms_below_1px": rms < 1.0,
    }
    accepted = all(checks.values())
    audit = {
        "accepted": accepted,
        "resolution": [width, height],
        "focal_length_px": [fx, fy],
        "principal_point_px": [cx, cy],
        "horizontal_fov_deg": math.degrees(2.0 * math.atan(width / (2.0 * fx))),
        "vertical_fov_deg": math.degrees(2.0 * math.atan(height / (2.0 * fy))),
        "distortion_coefficients": distortion.tolist(),
        "rms_reprojection_error_px": rms,
        "max_per_view_reprojection_error_px": (
            max(per_view_errors) if per_view_errors else None
        ),
        "views_used": int(source.get("num_images_used", 0)),
        "checks": checks,
        "constraint": "Use only at the calibrated 640x480 resolution.",
    }
    normalized = {
        "schema_version": 1,
        "camera_name": camera_name,
        "model": "opencv_pinhole",
        "resolution": [width, height],
        "camera_matrix": matrix.tolist(),
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "distortion_coefficients": distortion.tolist(),
        "rms_reprojection_error_px": rms,
        "mean_view_reprojection_error_px": (
            float(np.mean(per_view_errors)) if per_view_errors else None
        ),
        "max_view_reprojection_error_px": (
            max(per_view_errors) if per_view_errors else None
        ),
        "views_used": int(source.get("num_images_used", 0)),
        "per_view": [
            {
                "file": item["file"],
                "corner_count": int(item["num_corners"]),
                "reprojection_rms_px": float(item["reproj_error_px"]),
            }
            for item in per_image
        ],
        "source_calibrated_at": source.get("calibrated_at"),
        "source_camera_index": source.get("camera_index"),
        "import_status": "accepted_intrinsics" if accepted else "rejected",
    }
    return audit, normalized


def audit_extrinsics(camera_name: str, source: dict) -> dict:
    transform = np.asarray(source["transform_4x4"], dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    determinant = float(np.linalg.det(rotation))
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    bottom_row_error = float(
        np.linalg.norm(transform[3] - np.array([0.0, 0.0, 0.0, 1.0]))
    )
    spread_mm = float(source.get("translation_spread_across_methods_mm", math.inf))
    rigid_transform_valid = (
        np.all(np.isfinite(transform))
        and abs(determinant - 1.0) < 1e-6
        and orthogonality_error < 1e-6
        and bottom_row_error < 1e-9
    )
    method_agreement_acceptable = spread_mm <= 50.0
    wrist_mount_distance_plausible = (
        True if camera_name != "wrist" else np.linalg.norm(translation) <= 0.20
    )
    accepted = bool(
        rigid_transform_valid
        and method_agreement_acceptable
        and wrist_mount_distance_plausible
    )
    rejection_reasons = []
    if not rigid_transform_valid:
        rejection_reasons.append("matrix is not a valid rigid transform")
    if not method_agreement_acceptable:
        rejection_reasons.append(
            f"cross-method translation spread is {spread_mm:.1f} mm (>50 mm gate)"
        )
    if not wrist_mount_distance_plausible:
        rejection_reasons.append(
            "wrist-link to camera translation norm exceeds 0.20 m"
        )
    return {
        "accepted_for_metric_replay": accepted,
        "status": "accepted" if accepted else "provisional_rejected",
        "parent_frame": source.get("parent_frame"),
        "child_frame": source.get("child_frame"),
        "transform_semantics": source.get("note"),
        "selected_method": source.get("primary_method"),
        "sample_count": int(source.get("num_samples", 0)),
        "translation_xyz_m": translation.tolist(),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "cross_method_translation_spread_mm": spread_mm,
        "rotation_determinant": determinant,
        "rotation_orthogonality_frobenius_error": orthogonality_error,
        "bottom_row_error": bottom_row_error,
        "checks": {
            "rigid_transform_valid": bool(rigid_transform_valid),
            "cross_method_spread_at_most_50mm": method_agreement_acceptable,
            "wrist_mount_translation_at_most_0p20m": bool(
                wrist_mount_distance_plausible
            ),
        },
        "rejection_reasons": rejection_reasons,
        "source_transform_4x4": transform.tolist(),
        "per_method_translation_xyz_m": source.get(
            "per_method_translation_xyz_m", {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--normalized-dir", type=Path)
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    output = args.output or bundle_dir / "calibration_audit.json"
    normalized_dir = args.normalized_dir or bundle_dir / "normalized"

    intrinsic_audits = {}
    normalized_paths = {}
    for camera_name, filename in INTRINSIC_FILES.items():
        source_path = bundle_dir / filename
        audit, normalized = audit_intrinsics(camera_name, load_yaml(source_path))
        intrinsic_audits[camera_name] = audit
        normalized_path = normalized_dir / filename
        save_yaml(normalized_path, normalized)
        normalized_paths[camera_name] = str(normalized_path.resolve())

    extrinsic_audits = {
        camera_name: audit_extrinsics(
            camera_name, load_yaml(bundle_dir / filename)
        )
        for camera_name, filename in EXTRINSIC_FILES.items()
    }
    metric_replay_ready = all(
        item["accepted"] for item in intrinsic_audits.values()
    ) and all(
        item["accepted_for_metric_replay"] for item in extrinsic_audits.values()
    )
    report = {
        "bundle_dir": str(bundle_dir),
        "intrinsics": intrinsic_audits,
        "extrinsics": extrinsic_audits,
        "normalized_intrinsics": normalized_paths,
        "metric_replay_ready": metric_replay_ready,
        "decision": (
            "ready"
            if metric_replay_ready
            else "intrinsics accepted; extrinsics must be recalibrated or verified"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()
