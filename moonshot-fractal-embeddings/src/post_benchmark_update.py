"""Post-benchmark update: Run all analysis scripts after new datasets added.

Usage: python src/post_benchmark_update.py
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"

scripts = [
    ("Compute paper statistics (Holm-Bonferroni)", "compute_paper_stats.py"),
    ("Meta-analysis (DerSimonian-Laird)", "meta_analysis.py"),
    ("Metric robustness battery", "metric_robustness.py"),
    ("Scaling trend robustness (LOO)", "scaling_robustness.py"),
    ("Regenerate paper figures", "paper_figures.py"),
]

def main():
    print("=" * 70)
    print("POST-BENCHMARK UPDATE: Rerunning all analysis scripts")
    print("=" * 70)

    for desc, script in scripts:
        path = SRC / script
        if not path.exists():
            print(f"\n  SKIP: {script} not found")
            continue
        print(f"\n{'=' * 70}")
        print(f"  {desc} ({script})")
        print(f"{'=' * 70}")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  WARNING: {script} exited with code {result.returncode}")

    print(f"\n{'=' * 70}")
    print("POST-BENCHMARK UPDATE COMPLETE!")
    print(f"{'=' * 70}")
    print("\nNext steps:")
    print("  1. Check new results in results/")
    print("  2. Update paper tables and text")
    print("  3. Run Codex review")


if __name__ == "__main__":
    main()
