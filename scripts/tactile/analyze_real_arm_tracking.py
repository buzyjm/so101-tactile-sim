"""Characterise how the recorded SO-101 follows its own commands.

The tactile side has had all the attention, but a policy that transfers has to
survive the arm as well: it emits joint targets, and if the simulated arm
answers a command differently from the real one, nothing downstream transfers.
The dataset carries the commanded ``action`` and the measured
``observation.state`` for every frame, so the real arm's lag and steady-state
error can simply be measured rather than assumed.

Simulation shows roughly 9 mm of end-effector droop under gravity, coming from
position drives limited to 3.35 N m -- a figure copied from the URDF and never
checked.  These numbers are what it should be checked against.
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

DATASET = "Jingyi-Z/sotac"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")
# The dataset stores LeRobot's normalised units, not degrees: the first five
# joints span -100..100 and the gripper 0..100, each mapped onto its own joint
# range.  Comparing those numbers to URDF limits directly is meaningless, so
# convert first.  Ranges from
# official_so101_workshop/.../lerobot_interface.py:SO101_USD_MAPPING.
JOINT_RANGE_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}


def to_degrees(raw: np.ndarray) -> np.ndarray:
    """LeRobot normalised units -> joint degrees, column by column."""
    out = np.empty_like(raw, dtype=float)
    for index, name in enumerate(JOINTS):
        low, high = JOINT_RANGE_DEG[name]
        if name == "gripper":
            normalised = raw[:, index] / 100.0
        else:
            normalised = (raw[:, index] + 100.0) / 200.0
        out[:, index] = low + normalised * (high - low)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, nargs="*", default=list(range(21)))
    p.add_argument("--max-lag", type=int, default=12,
                   help="Frames of lag to search, at 30 Hz.")
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "tactile_logs" / "real_arm_tracking.json")
    return p.parse_args()


def load_frames(episodes):
    from huggingface_hub import hf_hub_download, HfApi
    import pyarrow.parquet as pq
    api = HfApi()
    files = [f for f in api.list_repo_files(DATASET, repo_type="dataset")
             if f.startswith("data/") and f.endswith(".parquet")]
    wanted = set(episodes)
    action, state, index = [], [], []
    for name in sorted(files):
        table = pq.read_table(hf_hub_download(DATASET, name,
                                              repo_type="dataset"))
        episode = np.asarray(table.column("episode_index"))
        keep = np.isin(episode, list(wanted))
        if not keep.any():
            continue
        action.append(np.array(table.column("action").to_pylist())[keep])
        state.append(np.array(
            table.column("observation.state").to_pylist())[keep])
        index.append(episode[keep])
    return (np.concatenate(action), np.concatenate(state),
            np.concatenate(index))


def best_lag(command: np.ndarray, measured: np.ndarray, max_lag: int) -> int:
    """Frames the measurement trails the command, by residual minimisation."""
    best, best_error = 0, np.inf
    for lag in range(max_lag + 1):
        if lag == 0:
            error = np.mean((measured - command) ** 2)
        else:
            error = np.mean((measured[lag:] - command[:-lag]) ** 2)
        if error < best_error:
            best, best_error = lag, error
    return best


def main() -> int:
    args = parse_args()
    action, state, episode = load_frames(args.episodes)
    action, state = to_degrees(action), to_degrees(state)
    print(f"{len(action)} frames from {len(set(episode.tolist()))} episodes, "
          f"converted to degrees\n")

    report = {}
    print(f"{'joint':>14s} {'lag':>4s} {'lag ms':>7s} | {'residual err deg':>18s} "
          f"| {'still-frame err deg':>20s}")
    for j, name in enumerate(JOINTS):
        command, measured = action[:, j], state[:, j]
        lag = best_lag(command, measured, args.max_lag)
        aligned = (measured[lag:] - command[:-lag]) if lag else measured - command
        # Steady state: only frames where the command is essentially still, so
        # the residual is holding error rather than tracking lag.
        speed = np.abs(np.diff(command, prepend=command[0]))
        still = speed < 0.05
        still = still[lag:] if lag else still
        settled = aligned[still] if still.any() else aligned
        report[name] = {
            "lag_frames": int(lag), "lag_ms": float(lag / 30.0 * 1000),
            "residual_mean_deg": float(aligned.mean()),
            "residual_std_deg": float(aligned.std()),
            "residual_abs_p95_deg": float(np.percentile(np.abs(aligned), 95)),
            "still_mean_deg": float(settled.mean()),
            "still_std_deg": float(settled.std()),
            "still_abs_p95_deg": float(np.percentile(np.abs(settled), 95)),
            "still_frames": int(still.sum()),
        }
        r = report[name]
        print(f"{name:>14s} {lag:4d} {r['lag_ms']:7.1f} | "
              f"{r['residual_mean_deg']:+7.2f} ± {r['residual_std_deg']:5.2f} "
              f"p95 {r['residual_abs_p95_deg']:4.2f} | "
              f"{r['still_mean_deg']:+8.2f} ± {r['still_std_deg']:5.2f} "
              f"p95 {r['still_abs_p95_deg']:4.2f}")

    print("\nreachable range vs URDF limit (degrees):")
    for index, name in enumerate(JOINTS):
        low, high = JOINT_RANGE_DEG[name]
        measured_low, measured_high = state[:, index].min(), state[:, index].max()
        report[name].update({
            "state_min_deg": float(measured_low),
            "state_max_deg": float(measured_high),
            "limit_low_deg": low, "limit_high_deg": high,
            "headroom_high_deg": float(high - measured_high),
        })
        print(f"  {name:>14s} reached {measured_low:+7.2f}..{measured_high:+7.2f}"
              f"   limit {low:+7.1f}..{high:+7.1f}"
              f"   headroom {high - measured_high:+6.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print("\nwritten:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
