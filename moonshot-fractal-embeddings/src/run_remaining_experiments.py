"""
Orchestrator: Run all remaining experiments sequentially.

Runs after prospective_validation completes:
1. Causal stress tests (permutation + depth collapse + synthetic)
2. Cascade retrieval benchmark (DBPedia Classes + CLINC)
3. Multi-backbone validation (BGE-Base on 4 key datasets)
4. Update all stats and figures

Usage: python -u src/run_remaining_experiments.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"


def run_step(name, cmd, timeout=3600):
    """Run a step and report timing."""
    print(f"\n{'='*70}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] STARTING: {name}")
    print(f"{'='*70}")
    start = time.time()

    result = subprocess.run(
        cmd, shell=True, cwd=str(ROOT),
        timeout=timeout,
    )

    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] {status} in {elapsed:.0f}s: {name}")
    return result.returncode == 0


def main():
    print("="*70)
    print("  EXPERIMENT ORCHESTRATOR")
    print(f"  Started: {datetime.now().isoformat()}")
    print("="*70)

    results = []

    # 1. Causal stress tests
    ok = run_step(
        "Causal Stress Tests (CLINC)",
        f"python -u {SRC}/causal_stress_tests.py --test all --dataset clinc",
        timeout=2400,
    )
    results.append(("Causal Stress Tests", ok))

    # 2. Cascade retrieval - DBPedia Classes
    ok = run_step(
        "Cascade Retrieval (DBPedia Classes)",
        f"python -u {SRC}/adaptive_retrieval.py --dataset dbpedia_classes --seed 42 --max-samples 5000",
        timeout=1800,
    )
    results.append(("Retrieval: DBPedia Classes", ok))

    # 3. Cascade retrieval - CLINC
    ok = run_step(
        "Cascade Retrieval (CLINC)",
        f"python -u {SRC}/adaptive_retrieval.py --dataset clinc --seed 42 --max-samples 10000",
        timeout=1800,
    )
    results.append(("Retrieval: CLINC", ok))

    # 4. Multi-backbone: BGE-Base on 4 key datasets (3 seeds each)
    backbone_datasets = ["clinc", "dbpedia_classes", "trec", "yahoo"]
    seeds = [42, 123, 456]

    for ds in backbone_datasets:
        for seed in seeds:
            ok = run_step(
                f"BGE-Base V5 on {ds} (seed {seed})",
                f"python -u {SRC}/run_single_experiment.py --dataset {ds} --model bge-base --seed {seed} --method v5",
                timeout=600,
            )
            results.append((f"BGE-Base V5 {ds} s{seed}", ok))

            ok = run_step(
                f"BGE-Base MRL on {ds} (seed {seed})",
                f"python -u {SRC}/run_single_experiment.py --dataset {ds} --model bge-base --seed {seed} --method mrl",
                timeout=600,
            )
            results.append((f"BGE-Base MRL {ds} s{seed}", ok))

    # 5. Post-processing
    ok = run_step(
        "Post-experiment stats update",
        f"python -u {SRC}/post_experiment_update.py",
        timeout=300,
    )
    results.append(("Post-processing", ok))

    # Summary
    print(f"\n{'='*70}")
    print(f"  ORCHESTRATOR SUMMARY")
    print(f"  Finished: {datetime.now().isoformat()}")
    print(f"{'='*70}")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  {n_pass}/{len(results)} steps completed successfully")


if __name__ == "__main__":
    main()
