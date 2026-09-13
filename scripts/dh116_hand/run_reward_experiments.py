"""Train explicit reward variants, evaluate fixed starts, and compare them.

No LLM or API is used. This is an experiment runner, not Eureka reward search.
All variants use the same starting checkpoint, seed, and training budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = set()
LOCK = threading.Lock()


def run(command, log_path, timeout):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        with LOCK:
            ACTIVE.add(process)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Process exited {code}; see {log_path}")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            with LOCK:
                ACTIVE.discard(process)


def reward_curves(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(folder)).Reload()
    curves = {}
    for tag in events.Tags()["scalars"]:
        if "Reward/" in tag or "Metrics/" in tag:
            values = events.Scalars(tag)
            stride = max(1, len(values) // 10)
            curves[tag] = {"first": values[0].value, "last": values[-1].value,
                           "min": min(v.value for v in values), "max": max(v.value for v in values),
                           "samples": [[v.step, v.value] for v in values[::stride]]}
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--eval_envs", type=int, default=256)
    parser.add_argument("--eval_seconds", type=float, default=32.)
    parser.add_argument("--eval_seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200, help="Per subprocess, seconds")
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 4 or args.eval_envs < 16 or args.eval_seconds < 2:
        parser.error("need positive iterations, num_envs >= 4, eval_envs >= 16, eval_seconds >= 2")
    if len(set(args.devices)) != len(args.devices):
        parser.error("devices must be unique")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    variants = [{"name": "omega_z_control", "mode": "omega_z"},
                {"name": "heading_v2", "mode": "heading"}]
    plan = {**vars(args), "checkpoint": str(checkpoint), "output": str(output), "variants": variants,
            "common": {"episode_length_s": 16.0, "entropy_coef": 0.005, "tilt_penalty_scale": 0.0},
            "comparison": "matched warm-start pilot; 1 training seed, held-out evaluation starts; not Eureka"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    def experiment(variant, device):
        folder = output / variant["name"]
        folder.mkdir()
        train_dir = folder / "train"
        result = {"variant": variant["name"], "device": device, "status": "failed"}
        try:
            train = [sys.executable, "-u", "scripts/dh116_hand/train_pen_spin.py", "--info",
                     "--resume", str(checkpoint), "--log_dir", str(train_dir),
                     "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
                     "--seed", str(args.train_seed), "--device", device,
                     "--episode_length_s", "16", "--entropy_coef", "0.005",
                     "--tilt_penalty_scale", "0", "--spin_reward_mode", variant["mode"]]
            print(f"Training {variant['name']} on {device}", flush=True)
            run(train, folder / "train.log", args.timeout)
            trained = json.loads((train_dir / "result.json").read_text())
            summaries = []
            for seed in args.eval_seeds:
                report_path = folder / f"eval_seed{seed}.json"
                run([sys.executable, "-u", "scripts/dh116_hand/eval_pen_spin.py", "--info",
                     "--checkpoint", trained["checkpoint"], "--num_envs", str(args.eval_envs),
                     "--seconds", str(args.eval_seconds), "--seed", str(seed), "--yaw_bins", "16",
                     "--device", device, "--output", str(report_path)], folder / f"eval_seed{seed}.log", args.timeout)
                report = json.loads(report_path.read_text())
                summaries.append(report["summary"])
            def mean(key):
                values = [s[key] for s in summaries if s[key] is not None]
                return sum(values) / len(values) if values else None
            result.update(status="complete", checkpoint=trained["checkpoint"], evaluations=summaries,
                          success_rate=mean("success_rate"), drop_rate=mean("drop_rate"),
                          projected_rate_rad_s=mean("projected_rate_mean_rad_s"),
                          tilt_deg=mean("tilt_mean_deg"), reward_curves=reward_curves(train_dir))
        except Exception as exc:
            result["error"] = str(exc)
        (folder / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Finished {variant['name']}: {result['status']}", flush=True)
        return result

    # One sequential queue per device; no concurrent training jobs on the same GPU.
    def worker(index, device):
        return [experiment(v, device) for v in variants[index::len(args.devices)]]

    pool = ThreadPoolExecutor(max_workers=len(args.devices))
    try:
        futures = [pool.submit(worker, i, d) for i, d in enumerate(args.devices)]
        results = [r for future in futures for r in future.result()]
    except BaseException:
        with LOCK:
            for process in ACTIVE:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown()
    complete = [r for r in results if r["status"] == "complete"]
    ranking = sorted(complete, key=lambda r: (r["success_rate"], -r["drop_rate"], r["projected_rate_rad_s"]), reverse=True)
    summary = {"results": results, "ranking": [r["variant"] for r in ranking],
               "successful_candidate": ranking[0]["variant"] if ranking and ranking[0]["success_rate"] > 0 else None,
               "note": "Pilot ranking is descriptive; no candidate is promoted if success_rate is zero."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"ranking": summary["ranking"], "successful_candidate": summary["successful_candidate"]}), flush=True)
    return 0 if len(complete) == len(variants) else 1


if __name__ == "__main__":
    raise SystemExit(main())
