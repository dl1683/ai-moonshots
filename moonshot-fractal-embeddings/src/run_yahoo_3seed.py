"""Run Yahoo V5 + MRL with 3 seeds to get proper per-seed prefix accuracy."""

import sys
import os
import json
import torch
import gc
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from fractal_v5 import run_v5_experiment
from mrl_v5_baseline import run_mrl_experiment

RESULTS_DIR = Path(__file__).parent.parent / "results"
SEEDS = [42, 123, 456]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {"dataset": "yahoo", "model": "bge-small", "seeds": SEEDS}

    # Run V5
    v5_results = {}
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  V5 Yahoo seed={seed}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        gc.collect()

        r = run_v5_experiment(
            model_key="bge-small",
            dataset_name="yahoo",
            stage1_epochs=5,
            stage2_epochs=0,
            batch_size=32,
            device=device,
            seed=seed,
        )
        v5_results[str(seed)] = r

        # Print steerability
        pa = r.get('prefix_accuracy', {})
        steer = (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))
        print(f"  V5 seed {seed}: steer={steer:+.4f}")

    results['v5'] = v5_results

    # Run MRL
    mrl_results = {}
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  MRL Yahoo seed={seed}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        gc.collect()

        r = run_mrl_experiment(
            model_key="bge-small",
            dataset_name="yahoo",
            stage1_epochs=5,
            stage2_epochs=0,
            batch_size=32,
            device=device,
            seed=seed,
        )
        mrl_results[str(seed)] = r

        pa = r.get('prefix_accuracy', {})
        steer = (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))
        print(f"  MRL seed {seed}: steer={steer:+.4f}")

    results['mrl'] = mrl_results

    # Save
    out_path = RESULTS_DIR / "benchmark_bge-small_yahoo.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")

    # Summary
    import numpy as np
    print(f"\n{'='*60}")
    print(f"  YAHOO 3-SEED SUMMARY")
    print(f"{'='*60}")

    v5_steers = []
    mrl_steers = []
    for seed in SEEDS:
        v5_pa = v5_results[str(seed)].get('prefix_accuracy', {})
        v5_s = (v5_pa.get('j1_l0', 0) - v5_pa.get('j4_l0', 0)) + (v5_pa.get('j4_l1', 0) - v5_pa.get('j1_l1', 0))
        v5_steers.append(v5_s)

        mrl_pa = mrl_results[str(seed)].get('prefix_accuracy', {})
        mrl_s = (mrl_pa.get('j1_l0', 0) - mrl_pa.get('j4_l0', 0)) + (mrl_pa.get('j4_l1', 0) - mrl_pa.get('j1_l1', 0))
        mrl_steers.append(mrl_s)

    print(f"  V5 steerability: {np.mean(v5_steers):+.4f} +/- {np.std(v5_steers):.4f}")
    print(f"  MRL steerability: {np.mean(mrl_steers):+.4f} +/- {np.std(mrl_steers):.4f}")
    print(f"  Gap: {np.mean(v5_steers) - np.mean(mrl_steers):+.4f}")


if __name__ == "__main__":
    main()
