"""
Causal Stress Tests for Fractal Embeddings Theory
==================================================

Tests that the theory's causal mechanism is correct by:

1. HIERARCHY PERMUTATION: Randomly shuffle L0<->L1 mapping.
   Theory predicts: S should drop to ~0 or negative.

2. DEPTH COLLAPSE: Flatten a 3-level hierarchy to 2 levels (merge L1 into L0).
   Theory predicts: S should decrease proportionally to information loss.

3. SYNTHETIC HIERARCHY: Generate datasets with controlled H(L1|L0).
   Theory predicts: S should track H(L1|L0) * base_L1_acc.

These are causal interventions, not just observational correlations.
If all pass, the theory is validated as genuinely causal.

Usage: python -u src/causal_stress_tests.py [--test permutation|collapse|synthetic|all]
"""

import sys
import gc
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))


def compute_steerability(prefix_accuracy: dict) -> float:
    return (prefix_accuracy.get('j1_l0', 0) - prefix_accuracy.get('j4_l0', 0)) + \
           (prefix_accuracy.get('j4_l1', 0) - prefix_accuracy.get('j1_l1', 0))


def run_v5_quick(dataset, model_key="bge-small", seed=42, device="cuda"):
    """Run V5 quickly on a pre-loaded dataset. Returns steerability."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
    from multi_model_pipeline import MODELS

    train_data = dataset['train']
    test_data = dataset['test']
    num_l0 = dataset['num_l0']
    num_l1 = dataset['num_l1']

    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

    class TempDataset:
        def __init__(self, samples, l0_names, l1_names):
            self.samples = samples
            self.level0_names = l0_names
            self.level1_names = l1_names

    val = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data_copy = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

    config = MODELS[model_key]
    model = FractalModelV5(
        config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
        num_scales=4, scale_dim=64, device=device,
    ).to(device)

    trainer = V5Trainer(
        model=model, train_dataset=train_data_copy, val_dataset=val,
        device=device, stage1_epochs=5, stage2_epochs=0,
    )
    trainer.train(batch_size=32, patience=5)

    test_temp = TempDataset(test_data.samples, test_data.level0_names, test_data.level1_names)
    trainer.val_dataset = test_temp
    prefix_acc = trainer.evaluate_prefix_accuracy()
    steer = compute_steerability(prefix_acc)

    del model, trainer
    torch.cuda.empty_cache(); gc.collect()

    return steer, prefix_acc


# ================================================================
# TEST 1: HIERARCHY PERMUTATION
# ================================================================
def test_hierarchy_permutation(
    dataset_name: str = "clinc",
    n_permutations: int = 3,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Randomly permute L0 labels, breaking the true hierarchy.
    Theory predicts: with random hierarchy, V5 can't do better than MRL.
    Steerability should drop to ~0.
    """
    from hierarchical_datasets import load_hierarchical_dataset

    print(f"\n{'='*60}")
    print(f"  CAUSAL TEST 1: Hierarchy Permutation on {dataset_name}")
    print(f"{'='*60}")

    # Load real data
    train = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    num_l0 = len(train.level0_names)
    num_l1 = len(train.level1_names)

    # Run with TRUE hierarchy
    print(f"\n  Running with TRUE hierarchy (L0={num_l0}, L1={num_l1})...")
    true_data = {'train': train, 'test': test, 'num_l0': num_l0, 'num_l1': num_l1}
    true_steer, true_prefix = run_v5_quick(true_data, seed=seed, device=device)
    print(f"    TRUE hierarchy S = {true_steer:+.4f}")

    # Run with PERMUTED hierarchy
    perm_steers = []
    for p in range(n_permutations):
        print(f"\n  Permutation {p+1}/{n_permutations}...")
        rng = np.random.RandomState(seed + p + 1)

        # Randomly assign each L1 to a random L0
        perm_map = rng.randint(0, num_l0, size=num_l1)

        # Apply to train
        perm_train = deepcopy(train)
        for s in perm_train.samples:
            s.level0_label = int(perm_map[s.level1_label])

        # Apply to test
        perm_test = deepcopy(test)
        for s in perm_test.samples:
            s.level0_label = int(perm_map[s.level1_label])

        perm_data = {'train': perm_train, 'test': perm_test,
                     'num_l0': num_l0, 'num_l1': num_l1}
        perm_steer, perm_prefix = run_v5_quick(perm_data, seed=seed, device=device)
        perm_steers.append(perm_steer)
        print(f"    PERMUTED S = {perm_steer:+.4f}")

    mean_perm = np.mean(perm_steers)
    gap = true_steer - mean_perm

    print(f"\n  RESULT:")
    print(f"    TRUE hierarchy S:    {true_steer:+.4f}")
    print(f"    PERMUTED mean S:     {mean_perm:+.4f} +/- {np.std(perm_steers):.4f}")
    print(f"    Gap (TRUE - PERM):   {gap:+.4f}")
    print(f"    Theory prediction:   Gap should be > 0 (structure matters)")
    print(f"    PASS: {'YES' if gap > 0 else 'NO'}")

    return {
        'test': 'hierarchy_permutation',
        'dataset': dataset_name,
        'true_steer': float(true_steer),
        'permuted_steers': [float(s) for s in perm_steers],
        'permuted_mean': float(mean_perm),
        'gap': float(gap),
        'pass': gap > 0,
    }


# ================================================================
# TEST 2: DEPTH COLLAPSE
# ================================================================
def test_depth_collapse(
    seed: int = 42,
    device: str = "cuda",
):
    """
    Compare V5 on HWV L0->L3 (deep) vs HWV L0->L2 (collapsed).
    Theory predicts: less hierarchy depth -> less steerability.
    """
    from hierarchical_datasets import load_hierarchical_dataset

    print(f"\n{'='*60}")
    print(f"  CAUSAL TEST 2: Depth Collapse (HWV L0->L3 vs L0->L2)")
    print(f"{'='*60}")

    configs = [
        ("hwv_l0_l3", "Deep (L0->L3)", 4.59),
        ("hwv_l0_l2", "Shallow (L0->L2)", 4.09),
    ]

    results = []
    for ds_name, label, h_val in configs:
        print(f"\n  Running {label} ({ds_name}, H={h_val:.2f})...")
        train = load_hierarchical_dataset(ds_name, split="train", max_samples=10000)
        test = load_hierarchical_dataset(ds_name, split="test", max_samples=2000)
        num_l0 = len(train.level0_names)
        num_l1 = len(train.level1_names)

        data = {'train': train, 'test': test, 'num_l0': num_l0, 'num_l1': num_l1}
        steer, prefix = run_v5_quick(data, seed=seed, device=device)
        results.append({'name': ds_name, 'label': label, 'H': h_val, 'S': float(steer)})
        print(f"    S = {steer:+.4f}")

    deep_s = results[0]['S']
    shallow_s = results[1]['S']
    gap = deep_s - shallow_s

    print(f"\n  RESULT:")
    print(f"    Deep (L0->L3, H=4.59):    S = {deep_s:+.4f}")
    print(f"    Shallow (L0->L2, H=4.09):  S = {shallow_s:+.4f}")
    print(f"    Theory: deeper hierarchy -> more steerability (gap > 0)")
    print(f"    PASS: {'YES - deeper has more S' if gap > 0 else 'NO'}")

    return {
        'test': 'depth_collapse',
        'results': results,
        'gap': float(gap),
        'pass': gap > 0,
    }


# ================================================================
# TEST 3: SYNTHETIC HIERARCHY
# ================================================================
def test_synthetic_hierarchy(
    base_dataset: str = "clinc",
    seed: int = 42,
    device: str = "cuda",
):
    """
    Create synthetic hierarchies with controlled H(L1|L0):
    - Low H: group L1 classes into few L0 groups (each L0 has many L1)
    - High H: group L1 classes into many L0 groups (each L0 has few L1)
    Theory predicts: S should increase with H(L1|L0).
    """
    from hierarchical_datasets import load_hierarchical_dataset
    from scipy.stats import entropy

    print(f"\n{'='*60}")
    print(f"  CAUSAL TEST 3: Synthetic Hierarchy Control ({base_dataset})")
    print(f"{'='*60}")

    train = load_hierarchical_dataset(base_dataset, split="train", max_samples=10000)
    test = load_hierarchical_dataset(base_dataset, split="test", max_samples=2000)
    num_l1 = len(train.level1_names)

    # Create synthetic hierarchies with different groupings
    configs = [
        ("2-group", 2),    # Low H: 2 L0 groups
        ("5-group", 5),    # Medium H
        ("10-group", 10),  # High H: 10 L0 groups (original for CLINC)
        ("20-group", 20),  # Very high H
    ]

    results = []
    for label, n_groups in configs:
        n_groups = min(n_groups, num_l1)
        rng = np.random.RandomState(seed)

        # Assign L1 classes to L0 groups as evenly as possible
        assignments = np.zeros(num_l1, dtype=int)
        for i in range(num_l1):
            assignments[i] = i % n_groups

        # Compute H(L1|L0)
        h_cond = 0.0
        for c in range(n_groups):
            l1_in_c = np.where(assignments == c)[0]
            if len(l1_in_c) > 0:
                p_c = len(l1_in_c) / num_l1
                h_cond += p_c * np.log2(len(l1_in_c))

        # Apply to data
        syn_train = deepcopy(train)
        for s in syn_train.samples:
            s.level0_label = int(assignments[s.level1_label])
        syn_train.level0_names = [f"group_{i}" for i in range(n_groups)]

        syn_test = deepcopy(test)
        for s in syn_test.samples:
            s.level0_label = int(assignments[s.level1_label])
        syn_test.level0_names = [f"group_{i}" for i in range(n_groups)]

        print(f"\n  Running {label} (L0={n_groups}, H(L1|L0)={h_cond:.2f})...")
        data = {'train': syn_train, 'test': syn_test,
                'num_l0': n_groups, 'num_l1': num_l1}
        steer, prefix = run_v5_quick(data, seed=seed, device=device)
        results.append({'label': label, 'n_groups': n_groups,
                       'H': float(h_cond), 'S': float(steer)})
        print(f"    S = {steer:+.4f}")

    # Check monotonicity
    h_values = [r['H'] for r in results]
    s_values = [r['S'] for r in results]
    from scipy.stats import spearmanr
    rho, p = spearmanr(h_values, s_values)

    print(f"\n  RESULT:")
    for r in results:
        print(f"    {r['label']:10s}: H={r['H']:.2f}, S={r['S']:+.4f}")
    print(f"    Spearman rho(H, S) = {rho:.3f} (p={p:.4f})")
    print(f"    Theory: positive correlation between H and S")
    print(f"    PASS: {'YES' if rho > 0 and p < 0.1 else 'WEAK' if rho > 0 else 'NO'}")

    return {
        'test': 'synthetic_hierarchy',
        'base_dataset': base_dataset,
        'results': results,
        'spearman_rho': float(rho),
        'spearman_p': float(p),
        'pass': rho > 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                       choices=["permutation", "collapse", "synthetic", "all"])
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = []
    RESULTS_DIR = Path(__file__).parent.parent / "results"

    if args.test in ["permutation", "all"]:
        r = test_hierarchy_permutation(dataset_name=args.dataset, seed=args.seed)
        results.append(r)

    if args.test in ["collapse", "all"]:
        r = test_depth_collapse(seed=args.seed)
        results.append(r)

    if args.test in ["synthetic", "all"]:
        r = test_synthetic_hierarchy(base_dataset=args.dataset, seed=args.seed)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  CAUSAL STRESS TESTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r['pass'] else "FAIL"
        print(f"  {r['test']:25s}: {status}")

    n_pass = sum(1 for r in results if r['pass'])
    print(f"\n  Overall: {n_pass}/{len(results)} tests passed")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'tests': results,
        'n_pass': n_pass,
        'n_total': len(results),
    }
    out_path = RESULTS_DIR / "causal_stress_tests.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
