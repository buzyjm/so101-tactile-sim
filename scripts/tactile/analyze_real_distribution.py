"""Characterise the recorded tactile observation distribution across episodes.

Everything downstream -- which tactile model to build, how wide to randomise
it, and what "close enough" means for zero-shot -- needs the target measured
over the whole task, not a single episode.

It also tests whether the 52 taxel values live in a fixed low-dimensional
subspace.  If the same subspace appears in episodes whose contacts land in
different places, the values are being expanded from fewer measurements by the
device; if the subspace rotates with the contact, they carry real spatial
information.  That distinction decides what has to be reproduced in sim.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tactile_taxels import load_taxel_positions_m

DATASET = "Jingyi-Z/sotac"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, nargs="*",
                   default=list(range(21)), help="Default: the red-ball task.")
    p.add_argument("--rank-components", type=int, default=3)
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "tactile_logs" / "real_tactile_distribution.json")
    return p.parse_args()


def load_episode(episode: int):
    from huggingface_hub import hf_hub_download
    base = f"sensors/paxini_fingertip/episode_{episode:06d}/"
    pads, stamps = [], None
    for name in ("sensor_1.csv", "sensor_2.csv"):
        path = hf_hub_download(DATASET, base + name, repo_type="dataset")
        with open(path, encoding="utf-8-sig") as handle:
            table = np.array(list(csv.reader(handle))[1:], dtype=np.float64)
        if stamps is None:
            stamps = table[:, 3]
        pads.append(table[:, 7:7 + 156].reshape(-1, 52, 3))
    count = min(len(pads[0]), len(pads[1]))
    return (stamps[:count] - stamps[0]) / 1e9, np.stack(
        [pads[0][:count], pads[1][:count]], axis=1)


def longest_contact(seconds: np.ndarray, magnitudes: np.ndarray):
    """Index range of the longest run where the pad reports force."""
    active = magnitudes.sum(axis=1) > 0.0
    if not active.any():
        return None
    edges = np.diff(active.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, len(active)]
    best = int(np.argmax(seconds[ends - 1] - seconds[starts]))
    return int(starts[best]), int(ends[best])


def subspace_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal angles in degrees between two orthonormal bases."""
    qa = np.linalg.qr(a.T)[0]
    qb = np.linalg.qr(b.T)[0]
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))


def main() -> int:
    args = parse_args()
    taxels_mm = np.asarray(load_taxel_positions_m())[:, :2] * 1000.0
    per_episode, bases = [], {0: [], 1: []}

    for episode in args.episodes:
        try:
            seconds, forces = load_episode(episode)
        except Exception as error:
            print(f"episode {episode}: skipped ({type(error).__name__})")
            continue
        entry = {"episode": episode, "duration_s": float(seconds[-1]),
                 "rate_hz": float(len(seconds) / (seconds[-1] - seconds[0]))}
        for pad in (0, 1):
            magnitudes = np.linalg.norm(forces[:, pad], axis=-1)
            window = longest_contact(seconds, magnitudes)
            key = "fixed" if pad == 0 else "moving"
            if window is None:
                entry[key] = None
                continue
            start, stop = window
            block = magnitudes[start:stop]
            block = block[block.sum(axis=1) > 0]
            active = (block > 0).sum(axis=1)
            uniformity = np.array([
                row[row > 0].min() / row.max() for row in block])
            weights = block / block.sum(axis=1, keepdims=True)
            centroid = weights @ taxels_mm
            centred = block - block.mean(axis=0)
            _, singular, right = np.linalg.svd(centred, full_matrices=False)
            energy = singular ** 2 / (singular ** 2).sum()
            bases[pad].append((episode, right[:args.rank_components]))
            entry[key] = {
                "contact_s": float(seconds[stop - 1] - seconds[start]),
                "active_taxels_mean": float(active.mean()),
                "active_taxels_max": int(active.max()),
                "taxels_ever_active": int((block > 0).any(axis=0).sum()),
                "uniformity_mean": float(uniformity.mean()),
                "total_force_max_n": float(block.sum(axis=1).max()),
                "per_taxel_max_n": float(block.max()),
                "centroid_mm": [float(centroid[:, 0].mean()),
                                float(centroid[:, 1].mean())],
                "pc1_energy": float(energy[0]),
                "pc3_energy": float(energy[:3].sum()),
            }
        per_episode.append(entry)
        summary = entry.get("fixed")
        if summary:
            print(f"ep {episode:2d}  fixed: {summary['active_taxels_mean']:5.1f} "
                  f"taxels  uni {summary['uniformity_mean']:.2f}  "
                  f"Fmax {summary['total_force_max_n']:6.2f} N  "
                  f"pc1 {summary['pc1_energy']:.2f}  "
                  f"centroid ({summary['centroid_mm'][0]:+5.1f},"
                  f"{summary['centroid_mm'][1]:+5.1f})", flush=True)

    # Does the subspace move with the contact, or stay put?
    angles = {}
    for pad in (0, 1):
        entries = bases[pad]
        pairs = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                pairs.append(subspace_angles(entries[i][1], entries[j][1]))
        if pairs:
            stacked = np.array(pairs)
            angles["fixed" if pad == 0 else "moving"] = {
                "pairs": len(pairs),
                "largest_principal_angle_deg": {
                    "p10": float(np.percentile(stacked[:, -1], 10)),
                    "p50": float(np.percentile(stacked[:, -1], 50)),
                    "p90": float(np.percentile(stacked[:, -1], 90))},
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"episodes": per_episode, "subspace_angles": angles,
         "rank_components": args.rank_components}, indent=2) + "\n",
        encoding="utf-8")
    print("\nwritten:", args.output)
    for name, value in angles.items():
        band = value["largest_principal_angle_deg"]
        print(f"{name}: largest principal angle between episode subspaces "
              f"p10 {band['p10']:.1f} deg, p50 {band['p50']:.1f}, "
              f"p90 {band['p90']:.1f}  ({value['pairs']} pairs)")
    print("  small angles => one fixed device mapping; "
          "large => genuine spatial information")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
