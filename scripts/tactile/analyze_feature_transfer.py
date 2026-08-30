"""Which tactile features actually transfer from this simulation to hardware.

Matching every statistic is the wrong target: the recorded array itself spans
16.0 +/- 5.2 active taxels on the fixed pad, so "real" is a wide distribution
rather than a point.  What matters for zero-shot is whether the feature the
policy consumes lands in the same place on both sides.

Each candidate feature is scored three ways:

* z: how many hardware standard deviations the simulated mean sits away;
* coverage: the share of simulated frames inside the hardware 5-95 band;
* overlap: histogram overlap coefficient, 1.0 identical, 0.0 disjoint.

Features that score well are safe to feed a policy.  The rest have to be left
out of the observation, or covered by randomisation, before any zero-shot claim
is reasonable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tactile_taxels import load_taxel_positions_m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, nargs="*", default=list(range(21)))
    p.add_argument(
        "--sim-log", type=Path, nargs="*",
        default=[ROOT / "tactile_logs" / "scripted_ball_pick_place_tactile.npz"],
        help="One or more tactile logs; pooling randomised episodes shows "
             "whether randomisation widens the simulated distribution enough.")
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "tactile_logs" / "feature_transfer.json")
    return p.parse_args()


def frame_features(forces: np.ndarray, taxels_mm: np.ndarray) -> dict:
    """Per-frame features for one (2, 52, 3) reading, or None without contact."""
    magnitudes = np.linalg.norm(forces, axis=-1)          # (2, 52)
    if magnitudes.sum() <= 0.0:
        return None
    out = {}
    totals = magnitudes.sum(axis=1)
    out["total_force_fixed"] = float(totals[0])
    out["total_force_moving"] = float(totals[1])
    out["total_force_both"] = float(totals.sum())
    denominator = totals.sum()
    out["balance"] = float(abs(totals[0] - totals[1]) / denominator)
    for index, name in ((0, "fixed"), (1, "moving")):
        live = magnitudes[index][magnitudes[index] > 0]
        out[f"active_{name}"] = float(live.size)
        out[f"uniformity_{name}"] = float(
            live.min() / live.max()) if live.size else 0.0
        if magnitudes[index].sum() > 0:
            weights = magnitudes[index] / magnitudes[index].sum()
            centre = weights @ taxels_mm
            out[f"centroid_x_{name}"] = float(centre[0])
            out[f"centroid_y_{name}"] = float(centre[1])
            spread = np.sqrt(
                weights @ np.sum((taxels_mm - centre) ** 2, axis=1))
            out[f"spread_{name}"] = float(spread)
    normal = np.abs(forces[..., 2]).sum()
    shear = np.linalg.norm(forces[..., :2], axis=-1).sum()
    out["shear_to_normal"] = float(shear / normal) if normal > 0 else 0.0
    return out


def collect(stream, taxels_mm) -> dict:
    rows = [frame_features(frame, taxels_mm) for frame in stream]
    rows = [r for r in rows if r]
    if not rows:
        return {}
    return {key: np.array([r[key] for r in rows if key in r])
            for key in rows[0]}


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, edges)[0].astype(float)
    pb = np.histogram(b, edges)[0].astype(float)
    pa /= max(pa.sum(), 1.0)
    pb /= max(pb.sum(), 1.0)
    return float(np.minimum(pa, pb).sum())


def main() -> int:
    args = parse_args()
    taxels_mm = np.asarray(load_taxel_positions_m())[:, :2] * 1000.0

    from scripts.tactile.analyze_real_distribution import load_episode
    real_frames = []
    for episode in args.episodes:
        try:
            _, forces = load_episode(episode)
        except Exception:
            continue
        real_frames.append(forces)
    real = collect(np.concatenate(real_frames, axis=0), taxels_mm)
    print(f"hardware: {len(real['total_force_both'])} contact frames "
          f"from {len(real_frames)} episodes")

    sim_frames = []
    for path in args.sim_log:
        for match in ([path] if path.is_file() else sorted(path.glob("*.npz"))):
            sim_frames.append(np.load(match)["taxel_forces"])
    if not sim_frames:
        raise SystemExit("no simulation logs found")
    simulated = collect(np.concatenate(sim_frames, axis=0), taxels_mm)
    print(f"simulation: {len(simulated['total_force_both'])} contact frames "
          f"from {len(sim_frames)} logs\n")

    print(f"{'feature':>22s} {'real mean±sd':>18s} {'sim mean±sd':>18s} "
          f"{'z':>6s} {'cover':>6s} {'olap':>6s}  verdict")
    report = {}
    for key in sorted(real):
        if key not in simulated or simulated[key].size == 0:
            continue
        r, s = real[key], simulated[key]
        deviation = r.std() if r.std() > 1e-9 else 1e-9
        z = abs(s.mean() - r.mean()) / deviation
        low, high = np.percentile(r, [5, 95])
        coverage = float(((s >= low) & (s <= high)).mean())
        overlap = overlap_coefficient(r, s)
        verdict = ("transfers" if overlap >= 0.5 and z <= 1.0 else
                   "marginal" if overlap >= 0.25 or z <= 2.0 else
                   "does not transfer")
        report[key] = {"real_mean": float(r.mean()), "real_std": float(r.std()),
                       "sim_mean": float(s.mean()), "sim_std": float(s.std()),
                       "z": float(z), "coverage": coverage,
                       "overlap": overlap, "verdict": verdict}
        print(f"{key:>22s} {r.mean():9.2f}±{r.std():<7.2f} "
              f"{s.mean():9.2f}±{s.std():<7.2f} {z:6.2f} {coverage:6.2f} "
              f"{overlap:6.2f}  {verdict}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print("\nwritten:", args.output)
    good = [k for k, v in report.items() if v["verdict"] == "transfers"]
    print(f"\nsafe to put in an observation now: {good if good else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
