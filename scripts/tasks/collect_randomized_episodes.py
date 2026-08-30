"""Run the scripted expert over randomised episodes and collect the results.

Each episode is one subprocess so a crash or a physics blow-up costs a single
sample rather than the batch.  Every episode is reproducible from its seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--out-dir", type=Path, default=ROOT / "tactile_logs" / "randomized")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = args.seeds or list(range(args.start_seed, args.start_seed + args.count))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = ROOT / "tactile_logs" / "scripted_ball_pick_place_result.json"
    tactile_path = ROOT / "tactile_logs" / "scripted_ball_pick_place_tactile.npz"

    summary = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        completed = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", "isaacsim",
             "python", "scripts/tasks/scripted_ball_pick_place.py",
             "--mode", "pick-place", "--gpu", str(args.gpu),
             "--seed", str(seed)],
            cwd=ROOT, capture_output=True, text=True,
            env={**__import__("os").environ,
                 "CUDA_VISIBLE_DEVICES": str(args.gpu),
                 "PYTHONUNBUFFERED": "1"},
        )
        entry = {"seed": seed, "returncode": completed.returncode}
        if result_path.is_file():
            report = json.loads(result_path.read_text(encoding="utf-8"))
            entry.update({
                "task_success": report.get("task_success"),
                "lift_delta_m": report.get("lift_delta_m"),
                "bowl_distance_m": report.get("final_bowl_distance_xy_m"),
                "grasp_contact_fraction": report.get("grasp_contact_fraction"),
                "peak_raw_normal_n": report.get("tactile_peak_raw_normal_force_n"),
                "saturated_fraction": report.get("tactile_saturated_fraction"),
                "randomization": report.get("randomization"),
            })
            (args.out_dir / f"seed_{seed:04d}_result.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if tactile_path.is_file():
                (args.out_dir / f"seed_{seed:04d}_tactile.npz").write_bytes(
                    tactile_path.read_bytes())
        else:
            entry["task_success"] = None
        summary.append(entry)
        print(f"seed {seed}: success={entry.get('task_success')} "
              f"contact={entry.get('grasp_contact_fraction')} "
              f"peak={entry.get('peak_raw_normal_n')}", flush=True)
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    ok = [e for e in summary if e.get("task_success")]
    print(f"\n{len(ok)} / {len(summary)} episodes succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
