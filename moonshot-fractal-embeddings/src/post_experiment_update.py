"""Post-experiment update: run all statistical analyses and regenerate figures.

Run after deep hierarchy experiments complete to get updated paper statistics.
Handles both the original 8 datasets and the 4 new deep hierarchy datasets.

Usage: python src/post_experiment_update.py
"""

import subprocess
import sys
import os
import json
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path("results")

print("=" * 70)
print("POST-EXPERIMENT ANALYSIS PIPELINE")
print("=" * 70)

# Check what benchmark files exist
benchmark_files = sorted(RESULTS_DIR.glob("benchmark_bge-small_*.json"))
print(f"\nFound {len(benchmark_files)} benchmark files:")
for f in benchmark_files:
    ds = f.stem.replace("benchmark_bge-small_", "")
    data = json.load(open(f))
    n_seeds = len(data.get("seeds", []))
    print(f"  {ds:<20} {n_seeds} seeds")

steps = [
    ("Paper stats (Holm-Bonferroni)", [sys.executable, "-u", "src/compute_paper_stats.py"]),
    ("Random-effects meta-analysis", [sys.executable, "-u", "src/meta_analysis.py"]),
    ("Scaling trend robustness", [sys.executable, "-u", "src/scaling_robustness.py"]),
    ("Regenerate paper figures", [sys.executable, "-u", "src/paper_figures.py"]),
    ("Pre-registered prediction validation", [sys.executable, "-u", "src/prediction_validation.py"]),
]

for i, (desc, cmd) in enumerate(steps, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/{len(steps)}] {desc}")
    print("=" * 70)
    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        print(f"  WARNING: {desc} exited with code {result.returncode}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY: Deep Hierarchy Results")
print("=" * 70)

stats_file = RESULTS_DIR / "paper_statistics_holm.json"
if stats_file.exists():
    stats = json.load(open(stats_file))
    n = stats.get("n_tests", 0)
    n_sig = sum(1 for r in stats.get("results", {}).values() if r.get("significant"))
    print(f"  Total datasets: {n}, Significant (Holm p<0.05): {n_sig}/{n}")

    deep = ["hupd_sec_cls", "hupd_sec_sub", "hwv_l0_l2", "hwv_l0_l3"]
    for ds in deep:
        if ds in stats.get("results", {}):
            r = stats["results"][ds]
            sig = "*" if r.get("significant") else "ns"
            print(f"  {ds:<16} H={r.get('H_L1_L0', '?'):.2f}  "
                  f"V5={r['v5_mean']:+.4f}  MRL={r['mrl_mean']:+.4f}  "
                  f"d={r['d']:.1f}  p_adj={r['p_adjusted']:.4f} {sig}")

meta_file = RESULTS_DIR / "meta_analysis_results.json"
if meta_file.exists():
    meta = json.load(open(meta_file))
    d_val = meta.get('pooled_d', 0)
    ci_lo = meta.get('ci_lower', 0)
    ci_hi = meta.get('ci_upper', 0)
    print(f"\n  Meta-analysis: d={d_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], "
          f"p={meta.get('p', 0):.6f}, I^2={meta.get('I_squared', 0):.1f}%")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print("=" * 70)
