"""Full Benchmark Suite: All datasets, 5 seeds, V5+MRL.

Runs all remaining experiments needed for NeurIPS submission.
Automatically skips already-completed seeds.

Run: python src/run_full_benchmark_suite.py [dataset_name]
  e.g., python src/run_full_benchmark_suite.py goemotions
  or:   python src/run_full_benchmark_suite.py all

Estimated time on RTX 5090 (bge-small):
  - Each V5+MRL pair: ~4 min
  - 5 seeds per dataset: ~20 min per dataset
  - 7 datasets total: ~2.5 hours max (with caching of completed seeds)
"""

import sys
import os
import json
import torch
import gc
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = Path(__file__).parent.parent / "results"
ALL_SEEDS = [42, 123, 456, 789, 1024]
ALL_DATASETS = ['yahoo', 'newsgroups', 'trec', 'clinc', 'goemotions', 'arxiv', 'dbpedia_classes', 'wos']


def compute_steer(pa):
    return (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
           (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))


def run_dataset(ds_name, model_key="bge-small", seeds=None):
    """Run V5 + MRL benchmark for a dataset with given seeds."""
    from fractal_v5 import run_v5_experiment
    from mrl_v5_baseline import run_mrl_experiment

    if seeds is None:
        seeds = ALL_SEEDS

    bench_file = RESULTS_DIR / f"benchmark_{model_key}_{ds_name}.json"

    # Load existing results
    existing = {}
    if bench_file.exists():
        existing = json.load(open(bench_file))

    all_v5 = existing.get('v5', {})
    all_mrl = existing.get('mrl', {})

    for seed in seeds:
        seed_key = str(seed)

        # V5
        if seed_key in all_v5 and isinstance(all_v5[seed_key], dict) and 'prefix_accuracy' in all_v5[seed_key]:
            pa = all_v5[seed_key]['prefix_accuracy']
            steer = compute_steer(pa)
            print(f"  V5 {ds_name} seed={seed} -> CACHED, steer={steer:+.4f}")
        else:
            print(f"\n  V5 {ds_name} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                r = run_v5_experiment(
                    model_key=model_key, dataset_name=ds_name,
                    stage1_epochs=5, stage2_epochs=0, batch_size=16,
                    device="cuda", seed=seed,
                )
                all_v5[seed_key] = r
                pa = r.get('prefix_accuracy', {})
                steer = compute_steer(pa)
                print(f"    V5 seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"    V5 ERROR: {e}")
                import traceback; traceback.print_exc()

        # MRL
        if seed_key in all_mrl and isinstance(all_mrl[seed_key], dict) and 'prefix_accuracy' in all_mrl[seed_key]:
            pa = all_mrl[seed_key]['prefix_accuracy']
            steer = compute_steer(pa)
            print(f"  MRL {ds_name} seed={seed} -> CACHED, steer={steer:+.4f}")
        else:
            print(f"\n  MRL {ds_name} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                r = run_mrl_experiment(
                    model_key=model_key, dataset_name=ds_name,
                    stage1_epochs=5, stage2_epochs=0, batch_size=16,
                    device="cuda", seed=seed,
                )
                all_mrl[seed_key] = r
                pa = r.get('prefix_accuracy', {})
                steer = compute_steer(pa)
                print(f"    MRL seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"    MRL ERROR: {e}")
                import traceback; traceback.print_exc()

        # Save after each seed pair (incremental)
        out = {
            "model": model_key, "dataset": ds_name,
            "seeds": sorted([int(k) for k in set(list(all_v5.keys()) + list(all_mrl.keys()))]),
            "v5": all_v5, "mrl": all_mrl
        }
        with open(bench_file, 'w') as f:
            json.dump(out, f, indent=2,
                      default=lambda x: float(x) if hasattr(x, 'item') else x)

    # Final summary
    v5_steers = [compute_steer(v['prefix_accuracy'])
                 for v in all_v5.values()
                 if isinstance(v, dict) and 'prefix_accuracy' in v]
    mrl_steers = [compute_steer(v['prefix_accuracy'])
                  for v in all_mrl.values()
                  if isinstance(v, dict) and 'prefix_accuracy' in v]

    import numpy as np
    if v5_steers:
        print(f"\n  {ds_name} V5 steer: {np.mean(v5_steers):+.4f} ± {np.std(v5_steers):.4f} (n={len(v5_steers)})")
    if mrl_steers:
        print(f"  {ds_name} MRL steer: {np.mean(mrl_steers):+.4f} ± {np.std(mrl_steers):.4f} (n={len(mrl_steers)})")
    print(f"  Saved to {bench_file}")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    print("=" * 70)
    print("FULL BENCHMARK SUITE: 5-seed V5+MRL on all datasets")
    print("=" * 70)

    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target == 'all':
            datasets_to_run = ALL_DATASETS
        elif target in ALL_DATASETS:
            datasets_to_run = [target]
        else:
            print(f"Unknown dataset: {target}. Available: {ALL_DATASETS + ['all']}")
            sys.exit(1)
    else:
        datasets_to_run = ALL_DATASETS

    for ds in datasets_to_run:
        print(f"\n{'=' * 70}")
        print(f"  DATASET: {ds}")
        print(f"{'=' * 70}")
        try:
            run_dataset(ds)
        except Exception as e:
            print(f"  DATASET {ds} FAILED: {e}")
            import traceback; traceback.print_exc()
            print("  Continuing to next dataset...")

    print(f"\n{'=' * 70}")
    print("FULL BENCHMARK SUITE COMPLETE!")
    print(f"{'=' * 70}")
