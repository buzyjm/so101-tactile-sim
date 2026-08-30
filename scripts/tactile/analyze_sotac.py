#!/usr/bin/env python3
"""Summarize SoTac DP-S2015-Elite raw sidecars for simulator calibration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tactile_taxels import load_taxel_positions_mm


AXES = ("fx", "fy", "fz")
TAXEL_COUNT = 52
DATASET_URL = "https://huggingface.co/datasets/Jingyi-Z/sotac"


def _expected_header() -> list[str]:
    header = [
        "timestamp_ns",
        "frame_status",
        "time_calibration_offset_ns",
        "calibrated_timestamp_ns",
        *AXES,
    ]
    for taxel in range(TAXEL_COUNT):
        header.extend(f"p_{taxel:02d}_{axis}" for axis in AXES)
    return header


def _read_sidecar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    expected = _expected_header()
    if header != expected:
        raise ValueError(
            f"Unexpected SoTac schema in {path}: expected {len(expected)} "
            f"columns in p_00..p_51 order, got {len(header)}"
        )
    timestamps_ns = np.loadtxt(
        path, delimiter=",", skiprows=1, usecols=3, dtype=np.int64
    )
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=range(4, len(expected)),
        dtype=np.float64,
    )
    if values.ndim == 1:
        values = values[None, :]
    timestamps_ns = np.atleast_1d(timestamps_ns)
    resultant_n = values[:, :3]
    taxels_n = values[:, 3:].reshape(-1, TAXEL_COUNT, 3)
    return timestamps_ns, resultant_n, taxels_n


def _axis_stats(values: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        axis: {
            "min": float(np.min(values[..., index])),
            "max": float(np.max(values[..., index])),
            "mean": float(np.mean(values[..., index])),
            "std": float(np.std(values[..., index])),
        }
        for index, axis in enumerate(AXES)
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _analyze_sensor(
    path: Path, baseline_seconds: float, active_threshold_n: float
) -> tuple[dict, np.ndarray]:
    timestamps_ns, resultant_n, taxels_n = _read_sidecar(path)
    if len(timestamps_ns) < 2:
        raise ValueError(f"Need at least two raw rows in {path}")

    dt_ms = np.diff(timestamps_ns).astype(np.float64) / 1e6
    duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    baseline_limit = timestamps_ns[0] + int(baseline_seconds * 1e9)
    baseline = taxels_n[timestamps_ns < baseline_limit]
    if not len(baseline):
        baseline = taxels_n[:1]

    baseline_bias = np.mean(baseline, axis=0)
    baseline_std = np.std(baseline, axis=0)
    magnitudes = np.linalg.norm(taxels_n, axis=2)
    active_counts = np.count_nonzero(
        magnitudes >= active_threshold_n, axis=1
    )
    summed_n = np.sum(taxels_n, axis=1)
    difference_n = summed_n - resultant_n
    peak_flat = int(np.argmax(magnitudes))
    peak_row, peak_taxel = np.unravel_index(peak_flat, magnitudes.shape)
    quantization_residual = np.abs(taxels_n / 0.1 - np.rint(taxels_n / 0.1))

    baseline_summary = {}
    comparison = {}
    for axis_index, axis in enumerate(AXES):
        abs_bias = np.abs(baseline_bias[:, axis_index])
        temporal_std = baseline_std[:, axis_index]
        delta = difference_n[:, axis_index]
        baseline_summary[axis] = {
            "absolute_bias_n": {
                "median": float(np.median(abs_bias)),
                "p95": float(np.percentile(abs_bias, 95)),
                "max": float(np.max(abs_bias)),
            },
            "temporal_std_n": {
                "median": float(np.median(temporal_std)),
                "p95": float(np.percentile(temporal_std, 95)),
                "max": float(np.max(temporal_std)),
            },
        }
        comparison[axis] = {
            "mean_difference_n": float(np.mean(delta)),
            "mae_n": float(np.mean(np.abs(delta))),
            "rmse_n": float(np.sqrt(np.mean(delta**2))),
            "max_absolute_difference_n": float(np.max(np.abs(delta))),
            "correlation": _correlation(
                summed_n[:, axis_index], resultant_n[:, axis_index]
            ),
        }

    return (
        {
            "path": str(path.resolve()),
            "rows": int(len(timestamps_ns)),
            "start_timestamp_ns": int(timestamps_ns[0]),
            "end_timestamp_ns": int(timestamps_ns[-1]),
            "duration_s": float(duration_s),
            "effective_rate_hz": float((len(timestamps_ns) - 1) / duration_s),
            "sample_interval_ms": {
                "median": float(np.median(dt_ms)),
                "p01": float(np.percentile(dt_ms, 1)),
                "p99": float(np.percentile(dt_ms, 99)),
            },
            "hardware_resultant_n": _axis_stats(resultant_n),
            "distributed_taxels_n": _axis_stats(taxels_n),
            "baseline": {
                "requested_seconds": float(baseline_seconds),
                "rows": int(len(baseline)),
                "per_taxel": baseline_summary,
            },
            "active_taxel_count": {
                "criterion": f"vector magnitude >= {active_threshold_n:g} N",
                **_percentiles(active_counts),
            },
            "global_peak": {
                "row": int(peak_row),
                "taxel_id_one_based": int(peak_taxel + 1),
                "magnitude_n": float(magnitudes[peak_row, peak_taxel]),
                "force_n": taxels_n[peak_row, peak_taxel].tolist(),
            },
            "taxel_sum_minus_hardware_resultant": comparison,
            "quantization_0p1_n": {
                "max_lsb_residual": float(np.max(quantization_residual)),
                "matches": bool(np.max(quantization_residual) < 1e-6),
            },
        },
        timestamps_ns,
    )


def _coordinate_check(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sdk = np.asarray(
        [[point["x_mm"], point["y_mm"], point["z_mm"]] for point in data["points"]],
        dtype=np.float64,
    )
    local = np.asarray(load_taxel_positions_mm(), dtype=np.float64)
    if sdk.shape != local.shape:
        raise ValueError(f"SDK coordinate shape {sdk.shape}, expected {local.shape}")
    difference = np.abs(sdk - local)
    return {
        "sdk_path": str(path.resolve()),
        "taxel_count": int(len(sdk)),
        "index_mapping": "p_00 -> taxel 1, ..., p_51 -> taxel 52",
        "max_absolute_difference_mm": float(np.max(difference)),
        "exact_match": bool(np.array_equal(sdk, local)),
    }


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.sensor1 and args.sensor2:
        return args.sensor1, args.sensor2
    if args.dataset_root:
        episode = f"episode_{args.episode:06d}"
        candidates = list(args.dataset_root.rglob(f"{episode}/sensor1.csv"))
        if not candidates:
            candidates = list(args.dataset_root.rglob(f"{episode}*sensor1*.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Could not resolve one sensor1 CSV for {episode} under "
                f"{args.dataset_root}; pass --sensor1 and --sensor2 explicitly"
            )
        sensor1 = candidates[0]
        sensor2 = sensor1.with_name("sensor2.csv")
        if not sensor2.is_file():
            raise FileNotFoundError(sensor2)
        return sensor1, sensor2
    raise ValueError("Pass --sensor1/--sensor2 or --dataset-root")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor1", type=Path)
    parser.add_argument("--sensor2", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--active-threshold-n", type=float, default=0.1)
    parser.add_argument("--sdk-coordinates", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tactile_logs" / "sotac_calibration_report.json",
    )
    args = parser.parse_args()
    sensor1_path, sensor2_path = _resolve_paths(args)
    sensor1, timestamps1 = _analyze_sensor(
        sensor1_path, args.baseline_seconds, args.active_threshold_n
    )
    sensor2, timestamps2 = _analyze_sensor(
        sensor2_path, args.baseline_seconds, args.active_threshold_n
    )
    aligned_rows = min(len(timestamps1), len(timestamps2))
    timestamp_delta = (
        timestamps1[:aligned_rows].astype(np.float64)
        - timestamps2[:aligned_rows].astype(np.float64)
    )

    report = {
        "schema_version": 1,
        "source": {
            "dataset": "Jingyi-Z/sotac",
            "url": DATASET_URL,
            "force_units": "N",
            "raw_layout": "one hardware resultant (3) plus 52 distributed vectors (52, 3) per sensor",
        },
        "sensor1_fixed_gripper_side": sensor1,
        "sensor2_moving_wrist_roll_side": sensor2,
        "pair_timestamp_alignment": {
            "compared_rows": int(aligned_rows),
            "max_absolute_difference_ns": float(np.max(np.abs(timestamp_delta))),
            "exact": bool(np.all(timestamp_delta == 0.0)),
        },
        "sdk_coordinate_check": (
            _coordinate_check(args.sdk_coordinates)
            if args.sdk_coordinates
            else None
        ),
        "calibration_use": {
            "supported": [
                "raw output rate and jitter",
                "per-axis baseline bias and noise",
                "0.1 N quantization validation",
                "active-taxel and spatial-force statistics",
                "domain-randomization distributions",
            ],
            "not_identifiable_from_sotac_alone": [
                "contact-location-to-taxel spatial kernel",
                "rubber constitutive deformation",
                "force hysteresis versus known ground truth",
            ],
            "important": "The hardware resultant is an independent channel; do not treat the 52-vector sum as its definition.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sensor1_rate_hz": sensor1["effective_rate_hz"],
        "sensor2_rate_hz": sensor2["effective_rate_hz"],
        "timestamps_exact": report["pair_timestamp_alignment"]["exact"],
        "coordinates_exact": (
            report["sdk_coordinate_check"]["exact_match"]
            if report["sdk_coordinate_check"] else None
        ),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
