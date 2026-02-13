"""
Unified runner for all external baselines.
==========================================

Usage:
    python src/run_external_baselines.py                     # smoke test (clinc+trec, seed 42)
    python src/run_external_baselines.py --full              # full 12x5 run
    python src/run_external_baselines.py --method heal       # single method, all datasets
    python src/run_external_baselines.py --dataset clinc     # all methods, single dataset
    python src/run_external_baselines.py --cache-only        # just cache embeddings

Runs HEAL, CSR, and SMEC baselines on all 12 datasets with 5 seeds each.
"""

import sys
import os
import json
import time
import argparse
import gc
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from external_baselines.common import (
    ALL_DATASETS, ALL_SEEDS, DATASET_LIMITS, RESULTS_DIR,
    load_cached_embeddings, set_all_seeds,
)
from external_baselines.heal_baseline import run_heal
from external_baselines.csr_baseline import run_csr
from external_baselines.smec_baseline import run_smec

METHODS = {
    "heal": run_heal,
    "csr": run_csr,
    "smec": run_smec,
}

SMOKE_DATASETS = ["clinc", "trec"]
SMOKE_SEEDS = [42]


def cache_all_embeddings(datasets=None, model_key="bge-small"):
    """Pre-cache frozen embeddings for all datasets."""
    datasets = datasets or ALL_DATASETS
    print("=" * 60)
    print("  Pre-caching frozen embeddings")
    print("=" * 60)

    for ds in datasets:
        for split in ["train", "test"]:
            try:
                emb, l0, l1 = load_cached_embeddings(model_key, ds, split)
                print(f"  {ds}/{split}: {emb.shape}")
            except Exception as e:
                print(f"  ERROR {ds}/{split}: {e}")


def run_all(
    methods=None,
    datasets=None,
    seeds=None,
    skip_existing=True,
):
    """Run all specified baseline experiments."""
    methods = methods or list(METHODS.keys())
    datasets = datasets or ALL_DATASETS
    seeds = seeds or ALL_SEEDS

    total = len(methods) * len(datasets) * len(seeds)
    done = 0
    start_time = time.time()

    print("=" * 60)
    print(f"  External Baselines: {len(methods)} methods x "
          f"{len(datasets)} datasets x {len(seeds)} seeds = {total} runs")
    print("=" * 60)

    for method in methods:
        runner = METHODS[method]

        for ds in datasets:
            for seed in seeds:
                done += 1
                result_path = RESULTS_DIR / method / f"{ds}_seed{seed}.json"

                if skip_existing and result_path.exists():
                    print(f"\n[{done}/{total}] SKIP {method}/{ds}/seed{seed} (exists)")
                    continue

                print(f"\n[{done}/{total}] Running {method}/{ds}/seed{seed}...")

                try:
                    result = runner(ds, seed)
                    elapsed = time.time() - start_time
                    rate = done / elapsed * 3600 if elapsed > 0 else 0
                    print(f"  S={result['steerability']:+.4f} "
                          f"(elapsed: {elapsed/60:.1f}min, rate: {rate:.0f}/hr)")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

                # Clean up between runs
                gc.collect()
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Complete! {done} runs in {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")


def summarize_results():
    """Print summary table of all completed results."""
    print("\n" + "=" * 80)
    print("  EXTERNAL BASELINES SUMMARY")
    print("=" * 80)

    # Load V5/MRL results for comparison
    v5_steer = {}
    mrl_steer = {}
    results_root = Path(__file__).parent.parent / "results"

    for ds in ALL_DATASETS:
        v5_steers = []
        mrl_steers = []
        for seed in ALL_SEEDS:
            # V5 results
            v5_path = results_root / f"v5_bge-small_{ds}.json"
            bench_path = results_root / f"benchmark_bge-small_{ds}.json"

            if bench_path.exists():
                bench = json.load(open(bench_path))
                if str(seed) in bench.get("v5", {}):
                    pa = bench["v5"][str(seed)].get("prefix_accuracy", {})
                    s = (pa.get("j1_l0", 0) - pa.get("j4_l0", 0)) + \
                        (pa.get("j4_l1", 0) - pa.get("j1_l1", 0))
                    v5_steers.append(s)
                if str(seed) in bench.get("mrl", {}):
                    pa = bench["mrl"][str(seed)].get("prefix_accuracy", {})
                    s = (pa.get("j1_l0", 0) - pa.get("j4_l0", 0)) + \
                        (pa.get("j4_l1", 0) - pa.get("j1_l1", 0))
                    mrl_steers.append(s)

        if v5_steers:
            v5_steer[ds] = np.mean(v5_steers)
        if mrl_steers:
            mrl_steer[ds] = np.mean(mrl_steers)

    # Print header
    methods = ["heal", "csr", "smec"]
    header = f"{'Dataset':<18} {'V5':>8} {'MRL':>8}"
    for m in methods:
        header += f" {m.upper():>8}"
    print(header)
    print("-" * len(header))

    for ds in ALL_DATASETS:
        line = f"{ds:<18}"

        # V5 and MRL
        line += f" {v5_steer.get(ds, float('nan')):>+8.4f}"
        line += f" {mrl_steer.get(ds, float('nan')):>+8.4f}"

        # External baselines
        for method in methods:
            steers = []
            for seed in ALL_SEEDS:
                path = RESULTS_DIR / method / f"{ds}_seed{seed}.json"
                if path.exists():
                    r = json.load(open(path))
                    steers.append(r["steerability"])
            if steers:
                line += f" {np.mean(steers):>+8.4f}"
            else:
                line += f" {'--':>8}"

        print(line)

    print()


def main():
    parser = argparse.ArgumentParser(description="Run external baselines")
    parser.add_argument("--full", action="store_true", help="Full 12x5 run")
    parser.add_argument("--method", type=str, choices=list(METHODS.keys()),
                        help="Single method")
    parser.add_argument("--dataset", type=str, help="Single dataset")
    parser.add_argument("--seed", type=int, help="Single seed")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only cache embeddings")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of completed results")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip existing results")
    args = parser.parse_args()

    if args.summary:
        summarize_results()
        return

    # Determine what to run
    methods = [args.method] if args.method else None
    datasets = [args.dataset] if args.dataset else (
        ALL_DATASETS if args.full else SMOKE_DATASETS
    )
    seeds = [args.seed] if args.seed else (
        ALL_SEEDS if args.full else SMOKE_SEEDS
    )

    # Cache embeddings first
    cache_all_embeddings(datasets)

    if args.cache_only:
        return

    # Run experiments
    run_all(
        methods=methods,
        datasets=datasets,
        seeds=seeds,
        skip_existing=not args.no_skip,
    )

    # Print summary
    summarize_results()


if __name__ == "__main__":
    main()
