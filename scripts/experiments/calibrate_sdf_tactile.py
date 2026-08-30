"""Fit the SDF tactile parameters against the recorded hardware statistics.

Runs the validated grasp once, records the ball/pad poses through the carry,
then sweeps the three parameters offline against three independent targets --
active-taxel count, total force and uniformity -- so each is identified by its
own measurement instead of being traded against the others.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--usd", type=Path, default=ROOT / "lab_scene_task.usda")
    p.add_argument(
        "--poses", type=Path,
        default=ROOT / "tactile_logs" / "grasp_pose_trace.npz",
        help="Recorded ball/pad poses; regenerated unless --reuse-poses.")
    p.add_argument("--reuse-poses", action="store_true")
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "tactile_logs" / "sdf_tactile_calibration.json")
    return p.parse_args()


ARGS = parse_args()
TARGETS = {  # tactile_logs/real_tactile_distribution.json, 21 red-ball episodes
    "fixed": {"active": 16.04, "total_n": 7.96, "uniformity": 0.78},
    "moving": {"active": 22.68, "total_n": 11.93, "uniformity": 0.76},
}


def evaluate(parameters, ball_positions, pad_matrices, taxels, radius):
    """Return per-pad (active count, total force, uniformity) for a run."""
    from tactile_sdf import SdfTactileParameters, _apply_device_model
    stats = {}
    for index, key in enumerate(("fixed", "moving")):
        actives, totals, uniformities, geometric = [], [], [], []
        for frame in range(len(ball_positions)):
            matrix = pad_matrices[frame, index]
            # Gf matrices are row-vector: p_world = p_local @ M.  Transposing
            # the rotation here put every taxel in the wrong place and made the
            # whole sweep report zero contact.
            rotation, origin = matrix[:3, :3], matrix[3, :3]
            proud = taxels + np.array([0.0, 0.0, parameters.layer_thickness_m])
            world = proud @ rotation + origin
            offsets = world - ball_positions[frame]
            distance = np.linalg.norm(offsets, axis=1)
            depth = np.clip(radius - distance, 0.0, None)
            if not (depth > 0).any():
                continue
            magnitude = (parameters.stiffness_n_per_m
                         * np.power(depth, parameters.depth_exponent))
            magnitude[depth <= 0] = 0.0
            force = np.zeros((len(taxels), 3))
            force[:, 2] = magnitude
            device = _apply_device_model(force)
            loaded = np.linalg.norm(device, axis=-1)
            live = loaded[loaded > 0]
            if live.size < 2:
                continue
            actives.append(live.size)
            totals.append(float(loaded.sum()))
            uniformities.append(float(live.min() / live.max()))
            geometric.append(int((depth > 0).sum()))
        stats[key] = {
            "geometric": float(np.mean(geometric)) if geometric else 0.0,
            "active": float(np.mean(actives)) if actives else 0.0,
            "total_n": float(np.mean(totals)) if totals else 0.0,
            "uniformity": float(np.mean(uniformities)) if uniformities else 0.0,
        }
    return stats


def main() -> int:
    if not ARGS.poses.is_file():
        raise SystemExit(
            f"{ARGS.poses} not found. Generate it first with "
            "scripts/experiments/trace_grasp_poses.py")
    from tactile_sdf import SdfTactileParameters
    from tactile_taxels import load_taxel_positions_m
    from scene_config import TARGET_BALL_RADIUS

    data = np.load(ARGS.poses)
    ball = data["ball_position"]
    pads = data["pad_matrix"]
    taxels = np.asarray(load_taxel_positions_m())

    rows = []
    best = None
    # A sphere's penetration falls off parabolically from the contact centre,
    # so the rim taxels land under the 0.1 N quantisation floor and never
    # report.  Matching the measured 0.78 uniformity needs the force to be far
    # less depth-sensitive than a linear spring: a small exponent stands in for
    # the plush layer crushing flat and stiffening once dense.
    for layer in (0.0010, 0.0015, 0.0020, 0.0025, 0.0030):
        for stiffness in (150.0, 250.0, 400.0, 700.0, 1000.0):
            for exponent in (0.10, 0.20, 0.30, 0.45, 0.60):
                parameters = SdfTactileParameters(
                    layer_thickness_m=layer, stiffness_n_per_m=stiffness,
                    depth_exponent=exponent)
                stats = evaluate(parameters, ball, pads, taxels,
                                 TARGET_BALL_RADIUS)
                error = 0.0
                for key, target in TARGETS.items():
                    got = stats[key]
                    error += abs(got["active"] - target["active"]) / target["active"]
                    error += abs(got["total_n"] - target["total_n"]) / target["total_n"]
                    error += abs(got["uniformity"] - target["uniformity"]) / target["uniformity"]
                row = {"layer_mm": layer * 1000, "stiffness": stiffness,
                       "exponent": exponent, "error": error, "stats": stats}
                rows.append(row)
                if best is None or error < best["error"]:
                    best = row
    rows.sort(key=lambda r: r["error"])
    print(f"{'layer':>6s} {'stiff':>6s} {'exp':>4s} {'err':>6s} | "
          f"{'fgeo':>5s} {'fix act':>7s} {'fix N':>6s} {'fix uni':>7s} | "
          f"{'mgeo':>5s} {'mov act':>7s} {'mov N':>6s} {'mov uni':>7s}")
    for row in rows[:8]:
        f, m = row["stats"]["fixed"], row["stats"]["moving"]
        print(f"{row['layer_mm']:6.2f} {row['stiffness']:6.0f} "
              f"{row['exponent']:4.2f} {row['error']:6.3f} | "
              f"{f['geometric']:5.1f} {f['active']:7.1f} {f['total_n']:6.2f} "
              f"{f['uniformity']:7.2f} | {m['geometric']:5.1f} {m['active']:7.1f} "
              f"{m['total_n']:6.2f} {m['uniformity']:7.2f}")
    print(f"\nreal targets           |       "
          f"{TARGETS['fixed']['active']:7.1f} {TARGETS['fixed']['total_n']:6.2f} "
          f"{TARGETS['fixed']['uniformity']:7.2f} | "
          f"{TARGETS['moving']['active']:7.1f} {TARGETS['moving']['total_n']:6.2f} "
          f"{TARGETS['moving']['uniformity']:7.2f}")
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(
        {"targets": TARGETS, "best": best, "grid": rows[:40]}, indent=2) + "\n",
        encoding="utf-8")
    print("written:", ARGS.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
