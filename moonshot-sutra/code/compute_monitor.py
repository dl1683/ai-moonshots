"""Compute Monitor & Experiment Scheduler.

Monitors GPU utilization and schedules experiments to maintain 70-80% usage.
Logs utilization history. Launches queued experiments when resources are available.

Usage:
    python code/compute_monitor.py --watch          # Monitor mode: log utilization
    python code/compute_monitor.py --status         # One-shot status
    python code/compute_monitor.py --run-queue      # Run queued experiments when GPU is free
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_PATH = REPO_ROOT / "results" / "compute_log.jsonl"

# Experiment queue: ordered list of scripts to run
EXPERIMENT_QUEUE = [
    {"name": "probe_a", "script": "code/probe_a_compression_capability.py", "est_vram_gb": 2, "est_hours": 1},
    {"name": "probe_f", "script": "code/probe_f_stigmergic.py", "est_vram_gb": 2, "est_hours": 1.5},
    # Probes B-E will be added as they're implemented
]


def get_gpu_stats():
    """Get GPU utilization, memory, temperature via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(", ")
        return {
            "gpu_util_pct": int(parts[0]),
            "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]),
            "mem_used_pct": round(int(parts[1]) / int(parts[2]) * 100, 1),
            "temp_c": int(parts[3]),
            "power_w": float(parts[4]),
        }
    except Exception as e:
        return {"error": str(e)}


def get_gpu_processes():
    """Get list of processes using the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        processes = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    processes.append({"pid": parts[0], "name": parts[1],
                                     "mem_mb": parts[2] if len(parts) > 2 else "?"})
        return processes
    except Exception:
        return []


def print_status():
    """Print current GPU status."""
    stats = get_gpu_stats()
    if not stats or "error" in stats:
        print(f"GPU not available: {stats}")
        return stats

    procs = get_gpu_processes()

    print(f"\n{'='*50}")
    print(f"GPU STATUS — {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    print(f"  Utilization:  {stats['gpu_util_pct']}%")
    print(f"  Memory:       {stats['mem_used_mb']}MB / {stats['mem_total_mb']}MB ({stats['mem_used_pct']}%)")
    print(f"  Temperature:  {stats['temp_c']}C")
    print(f"  Power:        {stats['power_w']}W")
    print(f"  Processes:    {len(procs)}")
    for p in procs:
        print(f"    PID {p['pid']}: {Path(p['name']).name} ({p['mem_mb']}MB)")

    # Assessment
    free_mem_gb = (stats['mem_total_mb'] - stats['mem_used_mb']) / 1024
    print(f"\n  Free VRAM:    {free_mem_gb:.1f}GB")

    if stats['gpu_util_pct'] < 30:
        print(f"  STATUS: IDLE — GPU underutilized! Queue experiments.")
    elif stats['gpu_util_pct'] < 70:
        print(f"  STATUS: LIGHT — Room for more concurrent work.")
    elif stats['gpu_util_pct'] < 90:
        print(f"  STATUS: GOOD — Target utilization range (70-80%).")
    else:
        print(f"  STATUS: HEAVY — Near capacity. Don't add more.")

    # Can we fit more?
    if free_mem_gb > 4:
        print(f"  RECOMMENDATION: {free_mem_gb:.0f}GB free VRAM. Can run additional experiments!")
    elif free_mem_gb > 2:
        print(f"  RECOMMENDATION: {free_mem_gb:.0f}GB free. Small experiments can run concurrently.")

    return stats


def log_stats(stats):
    """Append stats to compute log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(), **stats}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def watch_mode(interval=30):
    """Continuously monitor GPU and log stats."""
    print(f"Monitoring GPU every {interval}s. Ctrl+C to stop.\n")
    try:
        while True:
            stats = get_gpu_stats()
            if stats and "error" not in stats:
                log_stats(stats)
                ts = datetime.now().strftime("%H:%M:%S")
                bar = "#" * (stats["gpu_util_pct"] // 5) + "." * (20 - stats["gpu_util_pct"] // 5)
                print(f"[{ts}] GPU:{stats['gpu_util_pct']:3d}% [{bar}] "
                      f"MEM:{stats['mem_used_mb']:5d}MB/{stats['mem_total_mb']}MB "
                      f"T:{stats['temp_c']}C P:{stats['power_w']:.0f}W", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def run_queue():
    """Run queued experiments, checking GPU availability between each."""
    for exp in EXPERIMENT_QUEUE:
        stats = get_gpu_stats()
        if not stats or "error" in stats:
            print(f"GPU not available, skipping {exp['name']}")
            continue

        free_mem_gb = (stats["mem_total_mb"] - stats["mem_used_mb"]) / 1024
        if free_mem_gb < exp["est_vram_gb"]:
            print(f"Not enough VRAM for {exp['name']} (need {exp['est_vram_gb']}GB, have {free_mem_gb:.1f}GB)")
            continue

        print(f"\nLaunching {exp['name']}: {exp['script']}")
        print(f"  Est VRAM: {exp['est_vram_gb']}GB, Est time: {exp['est_hours']}h")

        proc = subprocess.Popen(
            [sys.executable, "-u", str(REPO_ROOT / exp["script"])],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for completion
        stdout, _ = proc.communicate()
        print(stdout.decode("utf-8", errors="replace")[-2000:])

        if proc.returncode != 0:
            print(f"  FAILED with exit code {proc.returncode}")
        else:
            print(f"  COMPLETED successfully")


def main():
    parser = argparse.ArgumentParser(description="Sutra Compute Monitor")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--status", action="store_true", help="One-shot status check")
    parser.add_argument("--run-queue", action="store_true", help="Run queued experiments")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in seconds")
    args = parser.parse_args()

    if args.status or (not args.watch and not args.run_queue):
        print_status()
    elif args.watch:
        watch_mode(args.interval)
    elif args.run_queue:
        run_queue()


if __name__ == "__main__":
    main()
