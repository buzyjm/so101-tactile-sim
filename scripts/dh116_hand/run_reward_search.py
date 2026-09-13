"""Run one deterministic, Eureka-style reward-search generation.

The original Eureka uses an LLM to propose executable rewards, then trains and
ranks them with a task fitness.  This local variant replaces the unavailable
LLM with a declared candidate set while preserving train -> independent eval ->
reward reflection.  Candidates start from the same random seed and are never
ranked by their training reward.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

CANDIDATES = [
    {
        "name": "signed_x8",
        "spin_reward_scale": 8.0,
        "forward_spin_reward_scale": 0.0,
        "target_spin_reward_scale": 0.0,
        "entropy_coef": 0.001,
    },
    {
        "name": "forward_bootstrap",
        "spin_reward_scale": 2.0,
        "forward_spin_reward_scale": 3.0,
        "target_spin_reward_scale": 0.0,
        "entropy_coef": 0.0005,
    },
    {
        "name": "target_rate_3",
        "spin_reward_scale": 2.0,
        "forward_spin_reward_scale": 0.5,
        "target_spin_reward_scale": 0.5,
        "entropy_coef": 0.0005,
    },
]

COMMON = {
    "episode_length_s": 12.0,
    "target_spin_rate": 3.0,
    "target_spin_sigma": 2.0,
    "alive_bonus": 0.0,
    "fall_penalty": -2.0,
    "action_penalty_scale": -0.0001,
    "action_rate_scale": -0.0005,
    "palm_center_penalty_scale": -0.05,
    "tilt_penalty_scale": -0.05,
    "reset_position_noise": 0.002,
    "reset_dof_pos_noise": 0.02,
    "yaw_center_deg": 0.0,
    "yaw_range_deg": 15.0,
    "init_std": 0.6,
    "std_type": "log",
}


def run(command: list[str], log_path: Path, timeout: int) -> None:
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise
        if code:
            raise RuntimeError(f"process exited {code}; see {log_path}")


def reward_curves(folder: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(folder)).Reload()
    curves = {}
    for tag in events.Tags()["scalars"]:
        if "Reward/" not in tag and "Metrics/" not in tag:
            continue
        values = events.Scalars(tag)
        stride = max(1, len(values) // 12)
        curves[tag] = {
            "first": values[0].value,
            "last": values[-1].value,
            "min": min(value.value for value in values),
            "max": max(value.value for value in values),
            "samples": [[value.step, value.value] for value in values[::stride]],
        }
    return curves


def add_option(command: list[str], name: str, value) -> None:
    command.extend(["--" + name, str(value)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--eval_envs", type=int, default=64)
    parser.add_argument("--eval_seconds", type=float, default=12.0)
    parser.add_argument("--eval_seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    if args.iterations < 1 or args.num_envs < 4 or args.eval_envs < 16:
        parser.error("invalid training/evaluation size")
    output = args.output.resolve()
    output.mkdir(parents=True)
    plan = {
        "schema_version": 1,
        "method": "declared reward candidates with Eureka-style train/evaluate/reflect loop; no LLM",
        "fitness": "lexicographic: success rate, startup rate, projected rate, negative drop rate",
        "candidates": CANDIDATES,
        "common": COMMON,
        "run": {**vars(args), "output": str(output)},
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for candidate in CANDIDATES:
        folder = output / candidate["name"]
        train_dir = folder / "train"
        folder.mkdir()
        result = {"candidate": candidate, "status": "failed"}
        try:
            command = [
                sys.executable, "-u", "scripts/dh116_hand/train_pen_spin.py", "--info",
                "--log_dir", str(train_dir), "--run_name", candidate["name"],
                "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
                "--seed", str(args.train_seed), "--device", args.device,
                "--spin_reward_mode", "heading", "--collision_profile", "current",
            ]
            for name, value in {**COMMON, **candidate}.items():
                if name != "name":
                    add_option(command, name, value)
            print(f"Training {candidate['name']}", flush=True)
            run(command, folder / "train.log", args.timeout)
            trained = json.loads((train_dir / "result.json").read_text())
            summaries = []
            for seed in args.eval_seeds:
                report = folder / f"eval_seed{seed}.json"
                eval_command = [
                    sys.executable, "-u", "scripts/dh116_hand/eval_pen_spin.py", "--info",
                    "--checkpoint", trained["checkpoint"], "--collision_profile", "current",
                    "--num_envs", str(args.eval_envs), "--seconds", str(args.eval_seconds),
                    "--seed", str(seed), "--yaw_bins", "8",
                    "--yaw_center_deg", str(COMMON["yaw_center_deg"]),
                    "--yaw_range_deg", str(COMMON["yaw_range_deg"]),
                    "--device", args.device, "--output", str(report),
                ]
                run(eval_command, folder / f"eval_seed{seed}.log", args.timeout)
                summaries.append(json.loads(report.read_text())["summary"])

            def mean(key: str) -> float:
                return sum(summary[key] for summary in summaries) / len(summaries)

            result.update(
                status="complete", checkpoint=trained["checkpoint"], evaluations=summaries,
                success_rate=mean("success_rate"), started_rate=mean("started_rate"),
                projected_rate_rad_s=mean("projected_rate_mean_rad_s"),
                drop_rate=mean("drop_rate"), tilt_deg=mean("tilt_mean_deg"),
                reward_curves=reward_curves(train_dir),
            )
        except Exception as exc:
            result["error"] = str(exc)
        (folder / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        results.append(result)
        print(f"Finished {candidate['name']}: {result['status']}", flush=True)

    complete = [result for result in results if result["status"] == "complete"]
    ranking = sorted(
        complete,
        key=lambda result: (
            result["success_rate"], result["started_rate"],
            result["projected_rate_rad_s"], -result["drop_rate"],
        ),
        reverse=True,
    )
    summary = {
        "results": results,
        "ranking": [result["candidate"]["name"] for result in ranking],
        "successful_candidate": (
            ranking[0]["candidate"]["name"] if ranking and ranking[0]["success_rate"] > 0 else None
        ),
        "reflection_seed": ranking[0]["candidate"]["name"] if ranking else None,
        "note": "A reflection seed is the next candidate parent, not a solved policy.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("ranking", "successful_candidate", "reflection_seed")}), flush=True)
    return 0 if len(complete) == len(CANDIDATES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
