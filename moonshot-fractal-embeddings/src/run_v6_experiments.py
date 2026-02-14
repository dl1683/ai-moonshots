"""
Run V6 experiments across multiple datasets and seeds.
Uses same framework as run_deep_hierarchy_experiments.py.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fractal_v6 import run_v6_experiment
from external_baselines.common import DATASET_LIMITS


# Configuration
DATASETS = {
    # Original 8 datasets (match V5 benchmarks)
    "yahoo":          {"max_train": 10000, "max_test": 2000},
    "goemotions":     {"max_train": 10000, "max_test": 2000},
    "newsgroups":     {"max_train": 10000, "max_test": 2000},
    "trec":           {"max_train": 10000, "max_test": 2000},
    "arxiv":          {"max_train": 10000, "max_test": 2000},
    "dbpedia_classes": {"max_train": 10000, "max_test": 2000},
    "clinc":          {"max_train": 10000, "max_test": 2000},
    "wos":            {"max_train": 10000, "max_test": 2000},
}

# Deep hierarchy datasets
DEEP_DATASETS = {
    "hupd_sec_cls":   {"max_train": 15000, "max_test": 3000},
    "hupd_sec_sub":   {"max_train": 30000, "max_test": 5000},
    "hwv_l0_l2":      {"max_train": 5635,  "max_test": 2000},
    "hwv_l0_l3":      {"max_train": 3402,  "max_test": 1500},
}

SEEDS = [42, 123, 456, 789, 1024]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_all(
    datasets=None,
    seeds=None,
    lambda_adv=0.3,
    stage1_epochs=5,
    stage2_epochs=0,
):
    """Run V6 on specified datasets and seeds."""
    if datasets is None:
        datasets = list(DATASETS.keys())
    if seeds is None:
        seeds = SEEDS

    all_results = []
    total = len(datasets) * len(seeds)
    done = 0

    for ds_name in datasets:
        limits = DATASET_LIMITS.get(ds_name, {"max_train": 10000, "max_test": 2000})

        for seed in seeds:
            result_path = RESULTS_DIR / f"v6_bge-small_{ds_name}.json"

            # Check if seed already done (multi-seed results stored in one file)
            done += 1
            print(f"\n{'#' * 70}")
            print(f"# V6 RUN {done}/{total}: {ds_name} seed={seed} "
                  f"lambda={lambda_adv}")
            print(f"{'#' * 70}")

            t0 = time.time()
            try:
                result = run_v6_experiment(
                    model_key="bge-small",
                    dataset_name=ds_name,
                    stage1_epochs=stage1_epochs,
                    stage2_epochs=stage2_epochs,
                    seed=seed,
                    lambda_adv=lambda_adv,
                    max_train_samples=limits["max_train"],
                    max_test_samples=limits["max_test"],
                )
                all_results.append(result)
                elapsed = time.time() - t0
                print(f"\n  Completed in {elapsed:.0f}s, "
                      f"S={result['steerability']:+.4f}")
            except Exception as e:
                print(f"\n  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("V6 EXPERIMENT SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['dataset']:<20} seed={r.get('training_config', {}).get('lambda_adv', '?'):<5} "
              f"S={r['steerability']:+.4f}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["quick", "main", "deep", "all"],
                        help="quick=clinc+dbpedia seed42, main=8 datasets 5 seeds, "
                             "deep=4 deep datasets, all=everything")
    parser.add_argument("--lambda-adv", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.mode == "quick":
        run_all(
            datasets=["clinc", "dbpedia_classes"],
            seeds=[42],
            lambda_adv=args.lambda_adv,
        )
    elif args.mode == "main":
        run_all(
            datasets=list(DATASETS.keys()),
            seeds=args.seeds or SEEDS,
            lambda_adv=args.lambda_adv,
        )
    elif args.mode == "deep":
        run_all(
            datasets=list(DEEP_DATASETS.keys()),
            seeds=args.seeds or SEEDS,
            lambda_adv=args.lambda_adv,
        )
    elif args.mode == "all":
        all_ds = list(DATASETS.keys()) + list(DEEP_DATASETS.keys())
        run_all(
            datasets=all_ds,
            seeds=args.seeds or SEEDS,
            lambda_adv=args.lambda_adv,
        )
