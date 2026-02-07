"""
V5 Statistical Validation - Test if 128d > 256d is real or noise.

Methodology:
- Run 8-10 seeds with identical settings
- Compare j=2 (128d) vs j=4 (256d) accuracy
- Use paired bootstrap test for significance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from scipy import stats
import gc

from multi_model_pipeline import MODELS
from hierarchical_datasets import load_hierarchical_dataset
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val


def evaluate_at_prefix(model, samples, prefix_len, batch_size=32) -> Dict:
    """Evaluate model at specific prefix length."""
    texts = [s.text for s in samples]
    l0_labels = np.array([s.level0_label for s in samples])
    l1_labels = np.array([s.level1_label for s in samples])

    # Get embeddings at prefix_len (None = full)
    emb = model.encode(texts, batch_size=batch_size, prefix_len=prefix_len).numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # KNN predictions for paired test
    l0_preds = []
    l1_preds = []

    for i in range(len(emb)):
        sims = emb @ emb[i]
        sims[i] = -float('inf')
        top_k = np.argsort(-sims)[:5]

        # L0 prediction
        neighbor_l0 = l0_labels[top_k]
        unique, counts = np.unique(neighbor_l0, return_counts=True)
        l0_preds.append(unique[np.argmax(counts)])

        # L1 prediction
        neighbor_l1 = l1_labels[top_k]
        unique, counts = np.unique(neighbor_l1, return_counts=True)
        l1_preds.append(unique[np.argmax(counts)])

    l0_preds = np.array(l0_preds)
    l1_preds = np.array(l1_preds)

    l0_correct = (l0_preds == l0_labels).astype(int)
    l1_correct = (l1_preds == l1_labels).astype(int)

    return {
        'l0_accuracy': l0_correct.mean(),
        'l1_accuracy': l1_correct.mean(),
        'l0_correct': l0_correct,  # For paired test
        'l1_correct': l1_correct,
    }


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict:
    """
    McNemar's test for paired nominal data.

    Tests if the marginal frequencies are equal (same as testing
    if A and B have the same error rate).
    """
    # Contingency table
    # b=0 | b=1
    # a=0  n00 | n01
    # a=1  n10 | n11

    n01 = ((correct_a == 0) & (correct_b == 1)).sum()  # A wrong, B right
    n10 = ((correct_a == 1) & (correct_b == 0)).sum()  # A right, B wrong

    # McNemar statistic with continuity correction
    if n01 + n10 == 0:
        return {'statistic': 0, 'p_value': 1.0, 'n01': n01, 'n10': n10}

    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(stat, df=1)

    return {
        'statistic': stat,
        'p_value': p_value,
        'n01': int(n01),  # A wrong, B right
        'n10': int(n10),  # A right, B wrong
        'advantage_b': n01 > n10,  # B is better if more cases where B right, A wrong
    }


def paired_bootstrap_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_bootstrap: int = 10000
) -> Dict:
    """
    Paired bootstrap test for accuracy difference.

    Returns confidence interval and p-value.
    """
    n = len(correct_a)
    observed_diff = correct_b.mean() - correct_a.mean()  # B - A

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diff = correct_b[idx].mean() - correct_a[idx].mean()
        diffs.append(diff)

    diffs = np.array(diffs)

    # Confidence interval
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)

    # Two-sided p-value (H0: diff = 0)
    p_value = 2 * min(
        (diffs <= 0).mean(),
        (diffs >= 0).mean()
    )

    return {
        'observed_diff': observed_diff,
        'ci_95': (ci_low, ci_high),
        'p_value': p_value,
        'significant': ci_low > 0 or ci_high < 0,  # CI doesn't include 0
        'b_better': observed_diff > 0,
    }


def run_single_seed(
    seed: int,
    model_key: str = "bge-small",
    dataset_name: str = "yahoo",
    stage1_epochs: int = 5,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict:
    """Run V5 training for a single seed, return j=2 and j=4 results."""

    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data (fixed split)
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15, seed=42)  # Fixed split

    class TempDataset:
        def __init__(self, samples, level0_names, level1_names):
            self.samples = samples
            self.level0_names = level0_names
            self.level1_names = level1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data.samples = train_samples

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)

    # Create model
    config = MODELS[model_key]
    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    # Train (head-only)
    trainer = V5Trainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=0,  # Head-only
    )

    trainer.train(batch_size=batch_size, patience=3)

    # Evaluate at j=2 (128d) and j=4 (256d)
    model.eval()

    results_j2 = evaluate_at_prefix(model, test_data.samples, prefix_len=2)
    results_j4 = evaluate_at_prefix(model, test_data.samples, prefix_len=None)  # Full

    result = {
        'seed': seed,
        'j2': {
            'l0_accuracy': results_j2['l0_accuracy'],
            'l1_accuracy': results_j2['l1_accuracy'],
            'l0_correct': results_j2['l0_correct'],
            'l1_correct': results_j2['l1_correct'],
        },
        'j4': {
            'l0_accuracy': results_j4['l0_accuracy'],
            'l1_accuracy': results_j4['l1_accuracy'],
            'l0_correct': results_j4['l0_correct'],
            'l1_correct': results_j4['l1_correct'],
        },
    }

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return result


def run_validation(
    num_seeds: int = 8,
    model_key: str = "bge-small",
    dataset_name: str = "yahoo",
    stage1_epochs: int = 5,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Run full statistical validation across multiple seeds."""

    print("=" * 70)
    print(f"V5 STATISTICAL VALIDATION: {model_key}")
    print("=" * 70)
    print(f"Testing: j=2 (128d) vs j=4 (256d)")
    print(f"Seeds: {num_seeds}")
    print()

    all_results = []

    for seed in range(num_seeds):
        print(f"\n[Seed {seed+1}/{num_seeds}]")
        result = run_single_seed(
            seed=seed,
            model_key=model_key,
            dataset_name=dataset_name,
            stage1_epochs=stage1_epochs,
            batch_size=batch_size,
            device=device,
        )

        print(f"  j=2 (128d): L0={result['j2']['l0_accuracy']:.4f}, L1={result['j2']['l1_accuracy']:.4f}")
        print(f"  j=4 (256d): L0={result['j4']['l0_accuracy']:.4f}, L1={result['j4']['l1_accuracy']:.4f}")

        all_results.append(result)

    # Aggregate results
    j2_l0 = np.array([r['j2']['l0_accuracy'] for r in all_results])
    j2_l1 = np.array([r['j2']['l1_accuracy'] for r in all_results])
    j4_l0 = np.array([r['j4']['l0_accuracy'] for r in all_results])
    j4_l1 = np.array([r['j4']['l1_accuracy'] for r in all_results])

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print(f"\nL0 Accuracy:")
    print(f"  j=2 (128d): {j2_l0.mean():.4f} ± {j2_l0.std():.4f}")
    print(f"  j=4 (256d): {j4_l0.mean():.4f} ± {j4_l0.std():.4f}")
    print(f"  Diff (j2-j4): {(j2_l0 - j4_l0).mean():.4f} ± {(j2_l0 - j4_l0).std():.4f}")

    print(f"\nL1 Accuracy:")
    print(f"  j=2 (128d): {j2_l1.mean():.4f} ± {j2_l1.std():.4f}")
    print(f"  j=4 (256d): {j4_l1.mean():.4f} ± {j4_l1.std():.4f}")
    print(f"  Diff (j2-j4): {(j2_l1 - j4_l1).mean():.4f} ± {(j2_l1 - j4_l1).std():.4f}")

    # Wins count
    l0_wins_j2 = (j2_l0 > j4_l0).sum()
    l1_wins_j2 = (j2_l1 > j4_l1).sum()
    print(f"\nWin counts (j2 > j4):")
    print(f"  L0: {l0_wins_j2}/{num_seeds}")
    print(f"  L1: {l1_wins_j2}/{num_seeds}")

    # Paired tests (aggregate all predictions)
    all_j2_l0_correct = np.concatenate([r['j2']['l0_correct'] for r in all_results])
    all_j4_l0_correct = np.concatenate([r['j4']['l0_correct'] for r in all_results])
    all_j2_l1_correct = np.concatenate([r['j2']['l1_correct'] for r in all_results])
    all_j4_l1_correct = np.concatenate([r['j4']['l1_correct'] for r in all_results])

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (paired on all predictions)")
    print("=" * 70)

    # McNemar test
    print("\nMcNemar Test (j2 vs j4):")
    mcnemar_l0 = mcnemar_test(all_j4_l0_correct, all_j2_l0_correct)  # j2 is "B"
    mcnemar_l1 = mcnemar_test(all_j4_l1_correct, all_j2_l1_correct)

    print(f"  L0: chi2={mcnemar_l0['statistic']:.2f}, p={mcnemar_l0['p_value']:.4f}")
    print(f"      j2 right, j4 wrong: {mcnemar_l0['n01']}, j4 right, j2 wrong: {mcnemar_l0['n10']}")
    print(f"      j2 better: {mcnemar_l0['advantage_b']}")

    print(f"  L1: χ²={mcnemar_l1['statistic']:.2f}, p={mcnemar_l1['p_value']:.4f}")
    print(f"      j2 right, j4 wrong: {mcnemar_l1['n01']}, j4 right, j2 wrong: {mcnemar_l1['n10']}")
    print(f"      j2 better: {mcnemar_l1['advantage_b']}")

    # Bootstrap test
    print("\nPaired Bootstrap Test (j2 - j4 accuracy):")
    bootstrap_l0 = paired_bootstrap_test(all_j4_l0_correct, all_j2_l0_correct)
    bootstrap_l1 = paired_bootstrap_test(all_j4_l1_correct, all_j2_l1_correct)

    print(f"  L0: diff={bootstrap_l0['observed_diff']:.4f}, 95% CI=[{bootstrap_l0['ci_95'][0]:.4f}, {bootstrap_l0['ci_95'][1]:.4f}], p={bootstrap_l0['p_value']:.4f}")
    print(f"      Significant: {bootstrap_l0['significant']}, j2 better: {bootstrap_l0['b_better']}")

    print(f"  L1: diff={bootstrap_l1['observed_diff']:.4f}, 95% CI=[{bootstrap_l1['ci_95'][0]:.4f}, {bootstrap_l1['ci_95'][1]:.4f}], p={bootstrap_l1['p_value']:.4f}")
    print(f"      Significant: {bootstrap_l1['significant']}, j2 better: {bootstrap_l1['b_better']}")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    l0_significant = mcnemar_l0['p_value'] < 0.05 or bootstrap_l0['significant']
    l1_significant = mcnemar_l1['p_value'] < 0.05 or bootstrap_l1['significant']
    j2_better_l0 = mcnemar_l0['advantage_b'] or bootstrap_l0['b_better']
    j2_better_l1 = mcnemar_l1['advantage_b'] or bootstrap_l1['b_better']

    if l0_significant and l1_significant and j2_better_l0 and j2_better_l1:
        print("✓ SIGNIFICANT: 128d (j=2) beats 256d (j=4) on BOTH L0 and L1")
        print("  → PURSUE COMPRESSION ANGLE")
    elif (l0_significant and j2_better_l0) or (l1_significant and j2_better_l1):
        print("~ PARTIAL: 128d wins on some metrics")
        print("  → INVESTIGATE FURTHER")
    else:
        print("✗ NOT SIGNIFICANT: Treat 128d>256d as noise")
        print("  → RETURN TO BASELINE-BEATING")

    # Save results
    results = {
        'model': model_key,
        'dataset': dataset_name,
        'num_seeds': num_seeds,
        'stage1_epochs': stage1_epochs,
        'aggregate': {
            'j2_l0_mean': float(j2_l0.mean()),
            'j2_l0_std': float(j2_l0.std()),
            'j2_l1_mean': float(j2_l1.mean()),
            'j2_l1_std': float(j2_l1.std()),
            'j4_l0_mean': float(j4_l0.mean()),
            'j4_l0_std': float(j4_l0.std()),
            'j4_l1_mean': float(j4_l1.mean()),
            'j4_l1_std': float(j4_l1.std()),
        },
        'wins': {
            'l0_wins_j2': int(l0_wins_j2),
            'l1_wins_j2': int(l1_wins_j2),
        },
        'mcnemar': {
            'l0': {k: v for k, v in mcnemar_l0.items()},
            'l1': {k: v for k, v in mcnemar_l1.items()},
        },
        'bootstrap': {
            'l0': {
                'observed_diff': bootstrap_l0['observed_diff'],
                'ci_95_low': bootstrap_l0['ci_95'][0],
                'ci_95_high': bootstrap_l0['ci_95'][1],
                'p_value': bootstrap_l0['p_value'],
                'significant': bootstrap_l0['significant'],
            },
            'l1': {
                'observed_diff': bootstrap_l1['observed_diff'],
                'ci_95_low': bootstrap_l1['ci_95'][0],
                'ci_95_high': bootstrap_l1['ci_95'][1],
                'p_value': bootstrap_l1['p_value'],
                'significant': bootstrap_l1['significant'],
            },
        },
        'per_seed': [
            {
                'seed': r['seed'],
                'j2_l0': float(r['j2']['l0_accuracy']),
                'j2_l1': float(r['j2']['l1_accuracy']),
                'j4_l0': float(r['j4']['l0_accuracy']),
                'j4_l1': float(r['j4']['l1_accuracy']),
            }
            for r in all_results
        ],
    }

    results_path = Path(__file__).parent.parent / "results" / "v5_statistical_validation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def run_simple_multi_seed(
    num_seeds: int = 5,
    model_key: str = "qwen3-0.6b",
    dataset_name: str = "yahoo",
    stage1_epochs: int = 5,
    batch_size: int = 24,
    device: str = "cuda",
):
    """Simpler multi-seed validation - just compare V5 to baseline."""
    from multi_model_pipeline import load_model

    print("=" * 70)
    print(f"MULTI-SEED VALIDATION: {model_key}")
    print("=" * 70)
    print(f"Seeds: {num_seeds}")
    print()

    # First, get baseline (only need once)
    print("Computing baseline...")
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

    base_model = load_model(model_key, use_fractal=False, device=device)

    def evaluate_baseline(model, samples):
        texts = [s.text for s in samples]
        l0_labels = np.array([s.level0_label for s in samples])
        l1_labels = np.array([s.level1_label for s in samples])

        emb = model.encode(texts, batch_size=32).numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        def knn_acc(emb, labels, k=5):
            correct = 0
            for i in range(len(emb)):
                sims = emb @ emb[i]
                sims[i] = -float('inf')
                top_k = np.argsort(-sims)[:k]
                neighbor_labels = labels[top_k]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                pred = unique[np.argmax(counts)]
                if pred == labels[i]:
                    correct += 1
            return correct / len(emb)

        return knn_acc(emb, l0_labels), knn_acc(emb, l1_labels)

    base_l0, base_l1 = evaluate_baseline(base_model, test_data.samples)
    print(f"Baseline: L0={base_l0:.4f}, L1={base_l1:.4f}")

    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # Run V5 with multiple seeds
    v5_l0_results = []
    v5_l1_results = []

    for seed in range(num_seeds):
        print(f"\n[Seed {seed+1}/{num_seeds}]")

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15, seed=42)

        class TempDataset:
            def __init__(self, samples, level0_names, level1_names):
                self.samples = samples
                self.level0_names = level0_names
                self.level1_names = level1_names

        val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
        train_data_seed = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

        num_l0 = len(train_data.level0_names)
        num_l1 = len(train_data.level1_names)

        config = MODELS[model_key]
        model = FractalModelV5(
            config=config,
            num_l0_classes=num_l0,
            num_l1_classes=num_l1,
            num_scales=4,
            scale_dim=64,
            device=device,
        ).to(device)

        trainer = V5Trainer(
            model=model,
            train_dataset=train_data_seed,
            val_dataset=val_data,
            device=device,
            stage1_epochs=stage1_epochs,
            stage2_epochs=0,
        )

        trainer.train(batch_size=batch_size, patience=3)

        # Evaluate
        model.eval()
        v5_l0, v5_l1 = evaluate_baseline(model, test_data.samples)
        v5_l0_results.append(v5_l0)
        v5_l1_results.append(v5_l1)

        print(f"  V5: L0={v5_l0:.4f} (delta={v5_l0-base_l0:+.4f}), L1={v5_l1:.4f} (delta={v5_l1-base_l1:+.4f})")

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    v5_l0 = np.array(v5_l0_results)
    v5_l1 = np.array(v5_l1_results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline: L0={base_l0:.4f}, L1={base_l1:.4f}")
    print(f"V5 (mean±std):")
    print(f"  L0: {v5_l0.mean():.4f} ± {v5_l0.std():.4f}")
    print(f"  L1: {v5_l1.mean():.4f} ± {v5_l1.std():.4f}")
    print(f"Delta vs baseline:")
    print(f"  L0: {(v5_l0.mean()-base_l0)*100:+.2f}% ± {v5_l0.std()*100:.2f}%")
    print(f"  L1: {(v5_l1.mean()-base_l1)*100:+.2f}% ± {v5_l1.std()*100:.2f}%")

    # Save
    results = {
        'model': model_key,
        'num_seeds': num_seeds,
        'baseline': {'l0': base_l0, 'l1': base_l1},
        'v5_mean': {'l0': float(v5_l0.mean()), 'l1': float(v5_l1.mean())},
        'v5_std': {'l0': float(v5_l0.std()), 'l1': float(v5_l1.std())},
        'delta': {
            'l0': float(v5_l0.mean() - base_l0),
            'l1': float(v5_l1.mean() - base_l1),
        },
        'per_seed': [
            {'seed': i, 'l0': float(v5_l0_results[i]), 'l1': float(v5_l1_results[i])}
            for i in range(num_seeds)
        ]
    }

    results_path = Path(__file__).parent.parent / "results" / f"v5_multiseed_{model_key}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--simple", action="store_true", default=True)
    args = parser.parse_args()

    if args.simple:
        run_simple_multi_seed(
            num_seeds=args.seeds,
            model_key=args.model,
            dataset_name=args.dataset,
            stage1_epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        run_validation(
            num_seeds=args.seeds,
            model_key=args.model,
            dataset_name=args.dataset,
            stage1_epochs=args.epochs,
            batch_size=args.batch_size,
        )
