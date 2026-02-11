"""
Capacity Sweep: Goldilocks Validation
======================================

Tests whether the steerability peak SHIFTS with hierarchy complexity.
This is the key experiment to elevate the Goldilocks effect from
observation to predictive law.

Theory: prefix capacity (scale_dim) should match H(L0) for optimal
steerability. Datasets with higher H(L1|L0) should peak at larger
prefix dimensions.

Sweep: scale_dim in [16, 32, 48, 64, 96, 128]
Datasets: yahoo (H=1.23), trec (H=2.21), dbpedia_classes (H=3.17), clinc (H=3.90)
Seeds: 3 per config (for speed), V5 only (MRL has ~0 steerability)

Expected: ~72 training runs, ~2.5 hours on RTX 5090.
"""

import sys
import os
import json
import gc
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from ablation_steerability import evaluate_prefix_steerability

# Configuration
SCALE_DIMS = [16, 32, 48, 64, 96, 128]
DATASETS = ['yahoo', 'trec', 'dbpedia_classes', 'clinc']
SEEDS = [42, 123, 456]
MODEL_KEY = "bge-small"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Known conditional entropies
H_L1_L0 = {
    'yahoo': 1.23,
    'trec': 2.21,
    'dbpedia_classes': 3.17,
    'clinc': 3.90,
}


def run_single(dataset_name, scale_dim, seed, stage1_epochs=5, batch_size=16, device="cuda"):
    """Train V5 with given scale_dim and return steerability."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"CAPACITY SWEEP: {dataset_name}, scale_dim={scale_dim}, seed={seed}")
    print(f"  Prefix j=1 dim: {scale_dim}d, total dim: {scale_dim * 4}d")
    print(f"{'='*60}")

    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15, seed=42)

    class TempDataset:
        def __init__(self, samples, level0_names, level1_names):
            self.samples = samples
            self.level0_names = level0_names
            self.level1_names = level1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data_obj = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)
    config = MODELS[MODEL_KEY]

    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=scale_dim,
        device=device,
    ).to(device)

    trainer = V5Trainer(
        model=model,
        train_dataset=train_data_obj,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=0,
    )
    trainer.train(batch_size=batch_size, patience=5)

    steerability = evaluate_prefix_steerability(model, test_data.samples)
    print(f"  Steerability: {steerability['steerability_score']:+.4f}")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'dataset': dataset_name,
        'scale_dim': scale_dim,
        'prefix_dim': scale_dim,
        'total_dim': scale_dim * 4,
        'seed': seed,
        'H_L1_L0': H_L1_L0.get(dataset_name, None),
        **steerability,
    }


def convert(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj


if __name__ == "__main__":
    out_path = RESULTS_DIR / "capacity_sweep_goldilocks.json"

    # Load existing results for incremental saving
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    all_results = existing.get('results', {})

    print("=" * 70)
    print("  CAPACITY SWEEP: Goldilocks Validation")
    print("  Does the steerability peak shift with hierarchy complexity?")
    print(f"  Datasets: {DATASETS}")
    print(f"  Scale dims: {SCALE_DIMS}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    total_runs = len(DATASETS) * len(SCALE_DIMS) * len(SEEDS)
    completed = 0

    for ds in DATASETS:
        if ds not in all_results:
            all_results[ds] = {}

        for sd in SCALE_DIMS:
            sd_key = str(sd)
            if sd_key not in all_results[ds]:
                all_results[ds][sd_key] = []

            # Check which seeds are already done
            done_seeds = {r['seed'] for r in all_results[ds][sd_key]}

            for seed in SEEDS:
                completed += 1
                if seed in done_seeds:
                    print(f"  [{completed}/{total_runs}] {ds} sd={sd} seed={seed} -> CACHED")
                    continue

                print(f"\n  [{completed}/{total_runs}] Running {ds} sd={sd} seed={seed}...")
                try:
                    result = run_single(ds, sd, seed)
                    all_results[ds][sd_key].append(result)

                    # Incremental save
                    output = {
                        'experiment': 'capacity_sweep_goldilocks',
                        'model': MODEL_KEY,
                        'scale_dims': SCALE_DIMS,
                        'datasets': DATASETS,
                        'seeds': SEEDS,
                        'H_L1_L0': H_L1_L0,
                        'timestamp': datetime.now().isoformat(),
                        'results': convert(all_results),
                    }
                    with open(out_path, 'w') as f:
                        json.dump(output, f, indent=2)

                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # ===== ANALYSIS =====
    print("\n" + "=" * 70)
    print("CAPACITY SWEEP ANALYSIS")
    print("=" * 70)

    from scipy import stats

    # For each dataset, find the peak scale_dim
    peak_dims = {}
    peak_steers = {}

    for ds in DATASETS:
        if ds not in all_results:
            continue

        print(f"\n  {ds} (H(L1|L0) = {H_L1_L0.get(ds, '?')}):")
        print(f"  {'ScaleDim':<10} {'PrefixDim':<10} {'Steerability':>15}")
        print("  " + "-" * 35)

        ds_means = {}
        for sd in SCALE_DIMS:
            sd_key = str(sd)
            if sd_key not in all_results[ds] or not all_results[ds][sd_key]:
                continue
            steers = [r['steerability_score'] for r in all_results[ds][sd_key]]
            mean_s = np.mean(steers)
            std_s = np.std(steers)
            ds_means[sd] = mean_s
            print(f"  {sd:<10} {sd:<10} {mean_s:>+11.4f} +/- {std_s:.4f}")

        if ds_means:
            best_sd = max(ds_means, key=ds_means.get)
            peak_dims[ds] = best_sd
            peak_steers[ds] = ds_means[best_sd]
            print(f"  -> Peak at scale_dim={best_sd} (S={ds_means[best_sd]:+.4f})")

    # Correlation: does peak dim correlate with H(L1|L0)?
    if len(peak_dims) >= 3:
        ds_list = [ds for ds in DATASETS if ds in peak_dims]
        h_values = [H_L1_L0[ds] for ds in ds_list]
        peak_values = [peak_dims[ds] for ds in ds_list]

        rho, p_rho = stats.spearmanr(h_values, peak_values)
        r, p_r = stats.pearsonr(h_values, peak_values)

        print(f"\n  SCALING LAW VALIDATION:")
        print(f"  Peak prefix dim vs H(L1|L0):")
        print(f"    Spearman rho = {rho:.3f} (p = {p_rho:.4f})")
        print(f"    Pearson r    = {r:.3f} (p = {p_r:.4f})")

        if rho > 0.7 and p_rho < 0.1:
            print("  *** GOLDILOCKS LAW CONFIRMED ***")
            print("  Peak prefix capacity shifts predictably with hierarchy complexity!")
        elif rho > 0.4:
            print("  ** Positive trend but noisy — need more datasets/seeds **")
        else:
            print("  * No clear trend — Goldilocks may be dataset-specific *")

    # Also check: does the SHAPE of the curve (inverted U) exist per dataset?
    print(f"\n  GOLDILOCKS SHAPE CHECK (inverted-U per dataset):")
    for ds in DATASETS:
        if ds not in all_results:
            continue
        steers_by_dim = {}
        for sd in SCALE_DIMS:
            sd_key = str(sd)
            if sd_key in all_results[ds] and all_results[ds][sd_key]:
                steers_by_dim[sd] = np.mean([r['steerability_score'] for r in all_results[ds][sd_key]])

        if len(steers_by_dim) >= 4:
            dims = sorted(steers_by_dim.keys())
            vals = [steers_by_dim[d] for d in dims]
            # Check if there's a peak (not monotonic)
            peak_idx = np.argmax(vals)
            is_inverted_u = (0 < peak_idx < len(vals) - 1)
            print(f"    {ds}: peak at dim={dims[peak_idx]}, inverted-U={is_inverted_u}")
        else:
            print(f"    {ds}: insufficient data")

    # Final save
    output = {
        'experiment': 'capacity_sweep_goldilocks',
        'model': MODEL_KEY,
        'scale_dims': SCALE_DIMS,
        'datasets': DATASETS,
        'seeds': SEEDS,
        'H_L1_L0': H_L1_L0,
        'timestamp': datetime.now().isoformat(),
        'results': convert(all_results),
        'analysis': {
            'peak_dims': convert(peak_dims),
            'peak_steers': convert(peak_steers),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
