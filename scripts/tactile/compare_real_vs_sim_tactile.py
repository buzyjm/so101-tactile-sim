"""Put the recorded SO-101 tactile field next to the simulated one.

Real data comes from the SoTac raw sidecars (Jingyi-Z/sotac), which are the
device's own output in newtons, so it is directly comparable to the device
model the simulator runs.  Both streams are aligned on first contact rather
than on episode start, because the real episode contains a long idle lead-in.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tactile_taxels import load_taxel_positions_m
from scripts.tactile.render_taxel_force_field import (
    BACKGROUND, INACTIVE, TEXT, draw_pad,
)

DATASET = "Jingyi-Z/sotac"
PANELS = ("real  fixed (sensor_1)", "real  moving (sensor_2)",
          "sim   fixed", "sim   moving")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument(
        "--sim-log", type=Path,
        default=ROOT / "tactile_logs" / "scripted_ball_pick_place_tactile.npz")
    p.add_argument(
        "--sim-result", type=Path,
        default=ROOT / "tactile_logs" / "scripted_ball_pick_place_result.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "renders")
    p.add_argument("--stem", default="tactile_real_vs_sim")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


def load_real(episode: int):
    """Return (relative seconds, (N, 2, 52, 3) forces) for one episode."""
    from huggingface_hub import hf_hub_download
    base = f"sensors/paxini_fingertip/episode_{episode:06d}/"
    streams, stamps = [], []
    for name in ("sensor_1.csv", "sensor_2.csv"):
        path = hf_hub_download(DATASET, base + name, repo_type="dataset")
        with open(path, encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
        table = np.array(rows[1:], dtype=np.float64)
        stamps.append(table[:, 3])
        streams.append(table[:, 7:7 + 156].reshape(-1, 52, 3))
    count = min(len(streams[0]), len(streams[1]))
    forces = np.stack([streams[0][:count], streams[1][:count]], axis=1)
    alignment = json.loads(Path(hf_hub_download(
        DATASET, base + "alignment.json", repo_type="dataset")).read_text())
    seconds = (stamps[0][:count] - alignment["episode_start_timestamp_ns"]) / 1e9
    return seconds, forces.astype(np.float32)


def contact_onset(seconds: np.ndarray, forces: np.ndarray) -> float:
    """Start of the longest sustained contact.

    Not simply the first contact: the recorded episode brushes the ball once
    at 2.19 s for half a second before the real grasp starts at 5.01 s, so
    aligning on first touch would put the two streams five seconds apart.
    """
    active = np.linalg.norm(forces, axis=-1).sum(axis=(1, 2)) > 0.0
    if not active.any():
        raise SystemExit("no contact in this stream")
    edges = np.diff(active.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, len(active)]
    longest = int(np.argmax(seconds[ends - 1] - seconds[starts]))
    return float(seconds[starts[longest]])


def main() -> int:
    args = parse_args()
    real_seconds, real_forces = load_real(args.episode)
    data = np.load(args.sim_log)
    sim_forces = data["taxel_forces"]
    sim_seconds = data["timestamps"] - data["timestamps"][0]

    real_zero = contact_onset(real_seconds, real_forces)
    sim_zero = contact_onset(sim_seconds, sim_forces)
    real_relative = real_seconds - real_zero
    sim_relative = sim_seconds - sim_zero
    print(f"real contact onset {real_zero:.2f} s, sim {sim_zero:.2f} s")

    taxels_mm = np.asarray(load_taxel_positions_m())[:, :2] * 1000.0
    span = taxels_mm.max(axis=0) - taxels_mm.min(axis=0)
    centre = (taxels_mm.max(axis=0) + taxels_mm.min(axis=0)) / 2.0
    gaps = np.linalg.norm(taxels_mm[:, None, :] - taxels_mm[None, :, :], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    cell_mm = float(np.median(gaps.min(axis=1))) * 0.9
    margin = cell_mm * 3.0
    limits = ((centre[0] - span[0] / 2 - margin, centre[0] + span[0] / 2 + margin),
              (centre[1] - span[1] / 2 - margin, centre[1] + span[1] / 2 + margin))

    # One shear scale for both sources, so arrow lengths mean the same thing.
    both = np.concatenate([
        np.linalg.norm(real_forces[..., :2], axis=-1).reshape(-1),
        np.linalg.norm(sim_forces[..., :2], axis=-1).reshape(-1)])
    present = both[both > 0]
    reference = float(np.percentile(present, 90)) if present.size else 0.0
    shear_scale = (cell_mm * 2.2 / reference) if reference else 0.0
    full_scale = 25.0

    figure, axes = plt.subplots(
        2, 2, figsize=(args.width / 100, args.height / 100), dpi=100)
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08,
                           wspace=0.08, hspace=0.22)
    axes = axes.reshape(-1)
    title = figure.text(0.5, 0.955, "", ha="center", color=TEXT, fontsize=15)
    subtitle = figure.text(0.5, 0.915, "", ha="center", color=INACTIVE,
                           fontsize=10)
    readouts = [figure.text(0.28 + 0.44 * (i % 2), 0.505 - 0.455 * (i // 2), "",
                            ha="center", color=TEXT, fontsize=10)
                for i in range(4)]

    def render(relative_time: float):
        real_index = int(np.argmin(np.abs(real_relative - relative_time)))
        sim_index = int(np.argmin(np.abs(sim_relative - relative_time)))
        frames = [real_forces[real_index, 0], real_forces[real_index, 1],
                  sim_forces[sim_index, 0], sim_forces[sim_index, 1]]
        for axis, label, force, readout in zip(axes, PANELS, frames, readouts):
            draw_pad(axis, taxels_mm, force, full_scale, shear_scale, cell_mm,
                     limits)
            axis.set_title(label, color=TEXT, fontsize=12, pad=6)
            magnitudes = np.linalg.norm(force, axis=-1)
            readout.set_text(
                f"{int(np.count_nonzero(magnitudes)):2d}/52 taxels    "
                f"|F| = {float(np.linalg.norm(force.sum(axis=0))):5.2f} N")
        title.set_text("tactile field: recorded SO-101 vs simulation")
        subtitle.set_text(
            f"t = {relative_time:+5.2f} s from grasp onset    "
            f"real episode {args.episode} @ 90.9 Hz    sim @ 90.9 Hz    "
            "arrow = shear, fill = normal")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = max(real_relative[0], sim_relative[0])
    stop = min(real_relative[-1], sim_relative[-1])
    print(f"shared window {start:+.2f} .. {stop:+.2f} s")

    # Still at the moment the simulation grips hardest.
    peak = float(sim_relative[int(np.argmax(
        np.linalg.norm(sim_forces, axis=-1).sum(axis=(1, 2))))])
    render(peak)
    still = args.out_dir / f"{args.stem}.png"
    figure.savefig(still, facecolor=BACKGROUND)
    print(f"still at t={peak:+.2f} s -> {still}")

    if not args.no_video:
        import cv2
        times = np.arange(start, stop, 1.0 / args.fps)
        video = args.out_dir / f"{args.stem}.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
            (args.width, args.height))
        for count, moment in enumerate(times):
            render(float(moment))
            figure.canvas.draw()
            frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if count % 50 == 0:
                print(f"  frame {count}/{len(times)}", flush=True)
        writer.release()
        print(f"{len(times)} frames -> {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
