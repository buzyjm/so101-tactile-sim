"""Draw the 52-taxel force field of both tactile pads from a recorded episode.

Reads the (N, 2, 52, 3) tensor written by the scripted pick-and-place task and
renders it in each pad's own sensor plane: one marker per taxel, shaded by the
normal component, with an arrow for the shear component.  Pure matplotlib, so
it needs neither Isaac nor a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

from tactile_taxels import load_taxel_positions_m
from scene_config import (
    TACTILE_MIN_FORCE_N,
    TACTILE_NORMAL_FORCE_RANGE_N,
    TACTILE_OUTPUT_FREQUENCY_HZ,
)

BACKGROUND = "#0e1117"
INACTIVE = "#6b7280"
ACTIVE_LOW = "#bbf7a0"
ACTIVE_HIGH = "#4ade50"
TEXT = "#d7dce3"
SENSOR_LABELS = ("fixed jaw", "moving jaw")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log", type=Path,
        default=ROOT / "tactile_logs" / "scripted_ball_pick_place_tactile.npz")
    p.add_argument(
        "--result", type=Path,
        default=ROOT / "tactile_logs" / "scripted_ball_pick_place_result.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "renders")
    p.add_argument("--stem", default="taxel_force_field")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--raw", action="store_true",
        help="Plot the pre-clip PhysX forces instead of the device output.")
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


def force_colour(normal: float, full_scale: float) -> str:
    """Blend between the pale and saturated greens by normal force."""
    weight = float(np.clip(normal / max(full_scale * 0.25, 1e-6), 0.0, 1.0))
    low = np.array([int(ACTIVE_LOW[i:i + 2], 16) for i in (1, 3, 5)])
    high = np.array([int(ACTIVE_HIGH[i:i + 2], 16) for i in (1, 3, 5)])
    rgb = (low + (high - low) * weight).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def draw_pad(axis, taxels_mm, forces, full_scale, shear_scale, cell_mm, limits):
    # clear() drops the axis limits, so they have to be re-applied here or the
    # view autoscales onto whichever arrow happens to be longest.
    axis.clear()
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal")
    magnitudes = np.linalg.norm(forces, axis=-1)
    for (x, y), force, magnitude in zip(taxels_mm, forces, magnitudes):
        active = magnitude > 0.0
        colour = force_colour(abs(force[2]), full_scale) if active else INACTIVE
        size = cell_mm * (1.0 if active else 0.55)
        axis.add_patch(Rectangle(
            (x - size / 2, y - size / 2), size, size,
            facecolor=colour, edgecolor="none", zorder=2,
            alpha=1.0 if active else 0.55))
        shear = float(np.hypot(force[0], force[1]))
        if active and shear > 0.0:
            # Start the arrow at the marker edge, otherwise the shaft is hidden
            # under the square it belongs to.
            direction = np.array([force[0], force[1]]) / shear
            start = np.array([x, y]) + direction * (cell_mm * 0.45)
            length = min(shear * shear_scale, cell_mm * 4.0)
            axis.add_patch(FancyArrow(
                start[0], start[1], direction[0] * length,
                direction[1] * length,
                width=cell_mm * 0.14, head_width=cell_mm * 0.5,
                head_length=cell_mm * 0.5, length_includes_head=True,
                facecolor=colour, edgecolor="none", alpha=0.95, zorder=3))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> int:
    args = parse_args()
    data = np.load(args.log)
    key = "raw_taxel_forces" if args.raw else "taxel_forces"
    if key not in data:
        raise SystemExit(f"{args.log} has no '{key}'; re-run the task to log it")
    forces = data[key]                      # (N, 2, 52, 3)
    timestamps = data["timestamps"]
    phase_ids = data["phase_ids"] if "phase_ids" in data else None

    phase_names = None
    if args.result.is_file():
        import json
        phase_names = json.loads(
            args.result.read_text(encoding="utf-8")).get("tactile_phase_names")

    taxels_mm = np.asarray(load_taxel_positions_m())[:, :2] * 1000.0
    span = taxels_mm.max(axis=0) - taxels_mm.min(axis=0)
    centre = (taxels_mm.max(axis=0) + taxels_mm.min(axis=0)) / 2.0
    # Nearest-neighbour spacing sets a marker size that never overlaps.
    gaps = np.linalg.norm(taxels_mm[:, None, :] - taxels_mm[None, :, :], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    cell_mm = float(np.median(gaps.min(axis=1))) * 0.9

    full_scale = float(TACTILE_NORMAL_FORCE_RANGE_N[1])
    # Scale on the 90th percentile of the shear that is actually present; the
    # outright maximum makes every typical arrow invisibly short.
    shear_magnitudes = np.linalg.norm(forces[..., :2], axis=-1)
    present = shear_magnitudes[shear_magnitudes > 0.0]
    reference_shear = float(np.percentile(present, 90)) if present.size else 0.0
    shear_scale = (cell_mm * 2.2 / reference_shear) if reference_shear > 0 else 0.0

    figure, axes = plt.subplots(
        1, 2, figsize=(args.width / 100, args.height / 100), dpi=100)
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.04, right=0.96, top=0.84, bottom=0.10,
                           wspace=0.10)
    margin = cell_mm * 3.0
    limits = (
        (centre[0] - span[0] / 2 - margin, centre[0] + span[0] / 2 + margin),
        (centre[1] - span[1] / 2 - margin, centre[1] + span[1] / 2 + margin),
    )
    for axis, label in zip(axes, SENSOR_LABELS):
        axis.set_title(label, color=TEXT, fontsize=13, pad=10)

    title = figure.text(0.5, 0.945, "", ha="center", color=TEXT, fontsize=15)
    subtitle = figure.text(0.5, 0.895, "", ha="center", color=INACTIVE,
                           fontsize=11)
    readout = [figure.text(0.28 + 0.44 * i, 0.045, "", ha="center",
                           color=TEXT, fontsize=11) for i in range(2)]

    def render(index: int):
        for pad in range(2):
            draw_pad(axes[pad], taxels_mm, forces[index, pad], full_scale,
                     shear_scale, cell_mm, limits)
            axes[pad].set_title(SENSOR_LABELS[pad], color=TEXT, fontsize=13,
                                pad=10)
            magnitudes = np.linalg.norm(forces[index, pad], axis=-1)
            total = float(np.linalg.norm(forces[index, pad].sum(axis=0)))
            readout[pad].set_text(
                f"{int(np.count_nonzero(magnitudes))} / 52 taxels    "
                f"|F| = {total:5.2f} N")
        phase = ""
        if phase_ids is not None and phase_names is not None:
            identifier = int(phase_ids[index])
            if 0 <= identifier < len(phase_names):
                phase = phase_names[identifier]
        title.set_text("SO-101 tactile force field"
                       + (f"  |  {phase}" if phase else ""))
        subtitle.set_text(
            f"t = {timestamps[index] - timestamps[0]:5.2f} s   "
            f"{TACTILE_OUTPUT_FREQUENCY_HZ:.1f} Hz   "
            f"{'raw PhysX' if args.raw else 'DP-S2015 output'}   "
            f"arrow = shear, fill = normal (min {TACTILE_MIN_FORCE_N} N)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Still frame at the hardest contact.
    peak_index = int(np.argmax(np.linalg.norm(forces, axis=-1).sum(axis=(1, 2))))
    render(peak_index)
    still = args.out_dir / f"{args.stem}.png"
    figure.savefig(still, facecolor=BACKGROUND)
    print(f"peak contact at t={timestamps[peak_index] - timestamps[0]:.2f} s "
          f"-> {still}")

    if not args.no_video:
        import cv2
        step = max(1, int(round(TACTILE_OUTPUT_FREQUENCY_HZ / args.fps)))
        indices = range(0, len(forces), step)
        video = args.out_dir / f"{args.stem}.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
            (args.width, args.height))
        for count, index in enumerate(indices):
            render(index)
            figure.canvas.draw()
            frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if count % 50 == 0:
                print(f"  frame {count}", flush=True)
        writer.release()
        print(f"{len(list(indices))} frames -> {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
