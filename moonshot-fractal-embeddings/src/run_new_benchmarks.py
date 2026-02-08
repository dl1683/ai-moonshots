"""Run V5 + MRL benchmarks on 4 new datasets for scaling law expansion.

Datasets: GoEmotions, arXiv, DBPedia Classes, WOS
Each gets 1 seed (quick validation), then 3 seeds if results look good.
"""

import sys
import os
import json
import torch
import gc
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from fractal_v5 import run_v5_experiment
from mrl_v5_baseline import run_mrl_experiment

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Ordered by H(L1|L0) to see the scaling law emerge in real-time
DATASETS = ["goemotions", "arxiv", "dbpedia_classes", "wos"]


def compute_steerability(prefix_accuracy):
    """Steer = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)."""
    return (prefix_accuracy.get('j1_l0', 0) - prefix_accuracy.get('j4_l0', 0)) + \
           (prefix_accuracy.get('j4_l1', 0) - prefix_accuracy.get('j1_l1', 0))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seeds = [42]  # Quick 1-seed validation first
    all_results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {ds_name}")
        print(f"{'='*70}")

        ds_results = {"dataset": ds_name, "model": "bge-small", "seeds": seeds}

        # V5
        v5_results = {}
        for seed in seeds:
            print(f"\n  V5 {ds_name} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()

            try:
                r = run_v5_experiment(
                    model_key="bge-small",
                    dataset_name=ds_name,
                    stage1_epochs=5,
                    stage2_epochs=0,
                    batch_size=32,
                    device=device,
                    seed=seed,
                )
                v5_results[str(seed)] = r
                pa = r.get('prefix_accuracy', {})
                steer = compute_steerability(pa)
                print(f"  V5 seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"  V5 ERROR: {e}")
                import traceback
                traceback.print_exc()
                v5_results[str(seed)] = {"error": str(e)}

        ds_results['v5'] = v5_results

        # MRL
        mrl_results = {}
        for seed in seeds:
            print(f"\n  MRL {ds_name} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()

            try:
                r = run_mrl_experiment(
                    model_key="bge-small",
                    dataset_name=ds_name,
                    stage1_epochs=5,
                    stage2_epochs=0,
                    batch_size=32,
                    device=device,
                    seed=seed,
                )
                mrl_results[str(seed)] = r
                pa = r.get('prefix_accuracy', {})
                steer = compute_steerability(pa)
                print(f"  MRL seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"  MRL ERROR: {e}")
                import traceback
                traceback.print_exc()
                mrl_results[str(seed)] = {"error": str(e)}

        ds_results['mrl'] = mrl_results
        all_results[ds_name] = ds_results

        # Save per-dataset
        out_path = RESULTS_DIR / f"benchmark_bge-small_{ds_name}.json"
        with open(out_path, 'w') as f:
            json.dump(ds_results, f, indent=2,
                     default=lambda x: float(x) if hasattr(x, 'item') else x)
        print(f"  Saved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  NEW BENCHMARKS SUMMARY")
    print(f"{'='*70}")

    # Load hierarchy profiles for H values
    profiles_path = RESULTS_DIR / "hierarchy_profiles.json"
    profiles = json.load(open(profiles_path)) if profiles_path.exists() else {}

    print(f"\n  {'Dataset':<18} {'H(L1|L0)':<10} {'V5 Steer':<12} {'MRL Steer':<12} {'Gap'}")
    print(f"  {'-'*60}")

    for ds_name in DATASETS:
        h = profiles.get(ds_name, {}).get('h_l1_given_l0', 'N/A')
        dr = all_results.get(ds_name, {})
        v5_steers = []
        mrl_steers = []
        for seed_data in dr.get('v5', {}).values():
            if isinstance(seed_data, dict) and 'prefix_accuracy' in seed_data:
                v5_steers.append(compute_steerability(seed_data['prefix_accuracy']))
        for seed_data in dr.get('mrl', {}).values():
            if isinstance(seed_data, dict) and 'prefix_accuracy' in seed_data:
                mrl_steers.append(compute_steerability(seed_data['prefix_accuracy']))

        v5_mean = np.mean(v5_steers) if v5_steers else float('nan')
        mrl_mean = np.mean(mrl_steers) if mrl_steers else float('nan')

        h_str = f"{h:.3f}" if isinstance(h, float) else str(h)
        print(f"  {ds_name:<18} {h_str:<10} {v5_mean:+.4f}      {mrl_mean:+.4f}      {v5_mean-mrl_mean:+.4f}")


if __name__ == "__main__":
    main()
