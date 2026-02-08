"""
Capacity Sweep Ablation: Inverted V5 with varying prefix dimensions.

Tests the hypothesis that inverted steerability magnitude increases with capacity.
If confirmed, this explains why |inverted| < |V5|: the 64d prefix is too small
to learn 150 L1 classes, creating a capacity ceiling on inversion magnitude.

Sweeps: scale_dim = 32, 64, 128 (prefix j=1 = 32d, 64d, 128d)
Each with 3 seeds on CLINC bge-small.
"""

import sys
import os
import json
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from ablation_steerability import InvertedV5Trainer, evaluate_prefix_steerability

SEEDS = [42, 123, 456]
SCALE_DIMS = [32, 64, 128]
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_capacity_sweep_single(scale_dim, seed, model_key="bge-small", dataset_name="clinc",
                               stage1_epochs=5, batch_size=32, device="cuda"):
    """Run inverted ablation with a specific scale_dim."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"CAPACITY SWEEP: inverted, scale_dim={scale_dim}, seed={seed}")
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
    config = MODELS[model_key]

    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=scale_dim,
        device=device,
    ).to(device)

    trainer = InvertedV5Trainer(
        model=model,
        train_dataset=train_data_obj,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=0,
    )
    trainer.train(batch_size=batch_size, patience=5)

    steerability = evaluate_prefix_steerability(model, test_data.samples)
    print(f"  Steerability results:")
    for j in [1, 2, 3, 4]:
        jr = steerability['prefix_results'][f'j{j}']
        print(f"    j={j}: L0={jr['l0']:.4f}, L1={jr['l1']:.4f}")
    print(f"  SteerabilityScore: {steerability['steerability_score']:+.4f}")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'scale_dim': scale_dim,
        'prefix_j1_dim': scale_dim,
        'total_dim': scale_dim * 4,
        'seed': seed,
        **steerability,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  CAPACITY SWEEP: Inverted V5 with varying prefix dimensions")
    print("  Testing: does inversion magnitude increase with capacity?")
    print("=" * 70)

    all_results = {}

    for sd in SCALE_DIMS:
        all_results[sd] = []
        for seed in SEEDS:
            result = run_capacity_sweep_single(sd, seed)
            all_results[sd].append(result)

    # Summary
    print(f"\n{'='*70}")
    print("CAPACITY SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'PrefixDim':<12} {'TotalDim':<10} {'Steerability':>15} {'Abs(Steer)':>12}")
    print("-" * 50)

    for sd in SCALE_DIMS:
        results = all_results[sd]
        ss = [r['steerability_score'] for r in results]
        mean_ss = np.mean(ss)
        std_ss = np.std(ss)
        print(f"{sd:<12} {sd*4:<10} {mean_ss:>+11.4f}+/-{std_ss:.4f} {abs(mean_ss):>12.4f}")

    # Check trend: does |steerability| increase with capacity?
    means = [np.mean([r['steerability_score'] for r in all_results[sd]]) for sd in SCALE_DIMS]
    abs_means = [abs(m) for m in means]
    trend_increasing = all(abs_means[i] <= abs_means[i+1] for i in range(len(abs_means)-1))
    print(f"\nTrend check: |steerability| increases with capacity? {trend_increasing}")

    if trend_increasing:
        print("  *** CAPACITY HYPOTHESIS CONFIRMED ***")
        print("  Inversion magnitude is limited by prefix capacity, not training signal.")
    else:
        print("  Trend not monotonic — may need more seeds or investigation.")

    # Save
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

    output = {
        'experiment': 'capacity_sweep_inverted',
        'model': 'bge-small',
        'dataset': 'clinc',
        'scale_dims': SCALE_DIMS,
        'seeds': SEEDS,
        'timestamp': datetime.now().isoformat(),
        'results': convert(all_results),
    }
    out_path = RESULTS_DIR / "capacity_sweep_inverted_clinc.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
