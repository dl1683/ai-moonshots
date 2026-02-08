"""
Synthetic Hierarchy Experiment: The Causal Intervention
========================================================

THE DECISIVE TEST: Directly manipulate H(L1|L0) while holding all else fixed.

Protocol:
  1. Take CLINC's 150 fine-grained classes (same text content across all conditions)
  2. Create RANDOM coarse groupings with K0 = {2, 3, 5, 10, 15, 25, 50, 75}
  3. For each K0: compute H(L1|L0), train V5 + MRL, measure steerability
  4. Plot steerability vs H(L1|L0) across all conditions

Why this is causal:
  - Text content is IDENTICAL across all conditions
  - Only the hierarchy structure changes
  - This isolates H(L1|L0) as the causal variable

Expected from theory (Hierarchical Rate-Allocation Law):
  S_V5(H) = kappa * [H - C(j1)]+
  where C(j1) is prefix capacity. Below threshold: no steerability.
"""

import sys
import os
import json
import torch
import gc
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import HierarchicalDataset, HierarchicalSample, load_hierarchical_dataset
from predict_before_train import compute_entropy, compute_conditional_entropy
from multi_model_pipeline import MODELS
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val

RESULTS_DIR = Path(__file__).parent.parent / "results"


def create_random_coarse_grouping(n_fine, k0, seed=42):
    """Create a random mapping from fine labels to k0 coarse groups."""
    rng = np.random.RandomState(seed)
    fine_labels = list(range(n_fine))
    rng.shuffle(fine_labels)
    mapping = {}
    per_group = n_fine // k0
    remainder = n_fine % k0
    idx = 0
    for coarse in range(k0):
        group_size = per_group + (1 if coarse < remainder else 0)
        for _ in range(group_size):
            mapping[fine_labels[idx]] = coarse
            idx += 1
    return mapping


def create_synthetic_hierarchy(base_dataset, k0, grouping_seed=42):
    """Create synthetic hierarchy: keep fine labels, remap coarse labels."""
    fine_labels = sorted(set(s.level1_label for s in base_dataset.samples))
    n_fine = len(fine_labels)
    fine_to_coarse = create_random_coarse_grouping(n_fine, k0, seed=grouping_seed)

    new_dataset = HierarchicalDataset()
    for s in base_dataset.samples:
        new_coarse = fine_to_coarse[s.level1_label]
        new_dataset.samples.append(HierarchicalSample(
            text=s.text,
            level0_label=new_coarse,
            level1_label=s.level1_label,
            level2_label=s.level2_label,
            level0_name=f"group_{new_coarse}",
            level1_name=s.level1_name,
            level2_name=s.level2_name,
        ))

    new_dataset.level0_names = [f"group_{i}" for i in range(k0)]
    new_dataset.level1_names = base_dataset.level1_names
    new_dataset.level2_names = base_dataset.level2_names
    return new_dataset


def compute_hierarchy_stats(dataset):
    """Compute information-theoretic stats."""
    l0 = [s.level0_label for s in dataset.samples]
    l1 = [s.level1_label for s in dataset.samples]
    h_l0 = compute_entropy(l0)
    h_l1 = compute_entropy(l1)
    h_l1_given_l0 = compute_conditional_entropy(l1, l0)
    return {
        'n_l0': len(set(l0)),
        'n_l1': len(set(l1)),
        'branching': len(set(l1)) / max(len(set(l0)), 1),
        'h_l0': float(h_l0),
        'h_l1': float(h_l1),
        'h_l1_given_l0': float(h_l1_given_l0),
    }


class TempDataset:
    """Minimal dataset wrapper matching V5Trainer's expectations."""
    def __init__(self, samples, level0_names, level1_names):
        self.samples = samples
        self.level0_names = level0_names
        self.level1_names = level1_names


def run_v5_on_synthetic(train_data, test_data, stats, seed, device):
    """Train V5 on synthetic hierarchy and return prefix accuracy."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = MODELS["bge-small"]
    num_l0 = stats['n_l0']
    num_l1 = stats['n_l1']

    # Split train/val
    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)
    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_sub = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

    # Create model
    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    # Create trainer
    trainer = V5Trainer(
        model=model,
        train_dataset=train_sub,
        val_dataset=val_data,
        device=device,
        stage1_epochs=5,
        stage2_epochs=0,
    )

    # Train
    trainer.train(batch_size=32, patience=5)

    # Evaluate prefix accuracy on test set
    model.eval()
    test_wrapper = TempDataset(test_data.samples, test_data.level0_names, test_data.level1_names)
    trainer.val_dataset = test_wrapper
    prefix_accuracy = trainer.evaluate_prefix_accuracy()

    del model, trainer
    return prefix_accuracy


def run_mrl_on_synthetic(train_data, test_data, stats, seed, device):
    """Train MRL on synthetic hierarchy and return prefix accuracy."""
    import random
    from mrl_v5_baseline import MRLTrainerV5

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = MODELS["bge-small"]
    num_l1 = stats['n_l1']

    # Split train/val
    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)
    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_sub = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

    # MRL uses same FractalModelV5 but with L1 for ALL prefix lengths
    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l1,  # MRL: head_top outputs L1 logits
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = MRLTrainerV5(
        model=model,
        train_dataset=train_sub,
        val_dataset=val_data,
        device=device,
        stage1_epochs=5,
        stage2_epochs=0,
    )

    trainer.train(batch_size=32, patience=5)

    # Evaluate
    model.eval()
    test_wrapper = TempDataset(test_data.samples, test_data.level0_names, test_data.level1_names)
    trainer.val_dataset = test_wrapper
    prefix_accuracy = trainer.evaluate_prefix_accuracy()

    del model, trainer
    return prefix_accuracy


def compute_steerability(prefix_accuracy):
    """Steer = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)."""
    return (prefix_accuracy.get('j1_l0', 0) - prefix_accuracy.get('j4_l0', 0)) + \
           (prefix_accuracy.get('j4_l1', 0) - prefix_accuracy.get('j1_l1', 0))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    k0_values = [2, 3, 5, 10, 15, 25, 50, 75]
    grouping_base_seed = 777  # Fixed seed for creating hierarchies

    print("=" * 80)
    print("  SYNTHETIC HIERARCHY EXPERIMENT: Causal H(L1|L0) Intervention")
    print("=" * 80)
    print("  Base: CLINC 150 fine classes | Varying: K0 coarse groups")
    print("  Same text, different hierarchy -> isolates H(L1|L0)")

    # Load base CLINC data ONCE
    print("\n  Loading CLINC...")
    train_base = load_hierarchical_dataset("clinc", split="train", max_samples=10000)
    test_base = load_hierarchical_dataset("clinc", split="test", max_samples=2000)

    # Preview expected H(L1|L0)
    print("\n  Expected hierarchy profiles:")
    for k0 in k0_values:
        synth = create_synthetic_hierarchy(train_base, k0, grouping_seed=grouping_base_seed + k0)
        stats = compute_hierarchy_stats(synth)
        print(f"    K0={k0:>3}: H(L1|L0)={stats['h_l1_given_l0']:.3f} bits, "
              f"branch={stats['branching']:.1f}")

    # Run experiments
    all_results = []
    for k0 in k0_values:
        print(f"\n{'='*60}")
        print(f"  K0 = {k0}")
        print(f"{'='*60}")

        gs = grouping_base_seed + k0
        train_synth = create_synthetic_hierarchy(train_base, k0, grouping_seed=gs)
        test_synth = create_synthetic_hierarchy(test_base, k0, grouping_seed=gs)
        stats = compute_hierarchy_stats(train_synth)
        print(f"  H(L1|L0) = {stats['h_l1_given_l0']:.3f} bits")

        result = {'k0': k0, 'hierarchy_stats': stats}

        try:
            # V5
            print(f"  Training V5...")
            torch.cuda.empty_cache()
            gc.collect()
            v5_pa = run_v5_on_synthetic(train_synth, test_synth, stats, seed=42, device=device)
            v5_steer = compute_steerability(v5_pa)
            result['v5_prefix_accuracy'] = v5_pa
            result['v5_steerability'] = float(v5_steer)
            print(f"    V5 steer = {v5_steer:+.4f}")

            # MRL
            print(f"  Training MRL...")
            torch.cuda.empty_cache()
            gc.collect()
            mrl_pa = run_mrl_on_synthetic(train_synth, test_synth, stats, seed=42, device=device)
            mrl_steer = compute_steerability(mrl_pa)
            result['mrl_prefix_accuracy'] = mrl_pa
            result['mrl_steerability'] = float(mrl_steer)
            print(f"    MRL steer = {mrl_steer:+.4f}")
            print(f"    Gap = {v5_steer - mrl_steer:+.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)

        all_results.append(result)

    # Save results
    output = {
        'experiment': 'synthetic_hierarchy_causal',
        'base_dataset': 'clinc',
        'model': 'bge-small',
        'k0_values': k0_values,
        'results': all_results,
    }

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = RESULTS_DIR / "synthetic_hierarchy_experiment.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert)

    # Summary
    print(f"\n{'='*80}")
    print(f"  SYNTHETIC HIERARCHY RESULTS SUMMARY")
    print(f"{'='*80}")

    valid = [r for r in all_results if 'v5_steerability' in r]
    if not valid:
        print("  No valid results!")
        return

    print(f"\n  {'K0':<5} {'H(L1|L0)':<10} {'Branch':<8} {'V5 Steer':<12} {'MRL Steer':<12} {'Gap'}")
    print(f"  {'-'*55}")

    h_vals, v5_vals, mrl_vals = [], [], []
    for r in valid:
        h = r['hierarchy_stats']['h_l1_given_l0']
        v5s = r['v5_steerability']
        mrls = r.get('mrl_steerability', 0)
        h_vals.append(h)
        v5_vals.append(v5s)
        mrl_vals.append(mrls)
        print(f"  {r['k0']:<5} {h:<10.3f} {r['hierarchy_stats']['branching']:<8.1f} "
              f"{v5s:+.4f}      {mrls:+.4f}      {v5s-mrls:+.4f}")

    # Statistical tests
    if len(h_vals) >= 4:
        from scipy.stats import spearmanr, pearsonr, linregress
        from scipy.optimize import curve_fit

        rho, p = spearmanr(h_vals, v5_vals)
        print(f"\n  V5 Steer vs H(L1|L0): Spearman rho={rho:.4f}, p={p:.6f}")

        rho_mrl, p_mrl = spearmanr(h_vals, mrl_vals)
        print(f"  MRL Steer vs H(L1|L0): Spearman rho={rho_mrl:.4f}, p={p_mrl:.6f}")

        gap_vals = [v - m for v, m in zip(v5_vals, mrl_vals)]
        rho_gap, p_gap = spearmanr(h_vals, gap_vals)
        print(f"  Gap vs H(L1|L0): Spearman rho={rho_gap:.4f}, p={p_gap:.6f}")

        # Linear fit
        slope, intercept, r_value, p_value, std_err = linregress(h_vals, v5_vals)
        print(f"\n  Linear: Steer = {slope:.5f} * H + {intercept:.5f} (R2={r_value**2:.4f}, p={p_value:.6f})")

        # Hinge fit: S = kappa * max(0, H - Hc)
        def hinge(x, kappa, hc):
            return np.array([kappa * max(0, xi - hc) for xi in x])

        try:
            popt, pcov = curve_fit(hinge, h_vals, v5_vals, p0=[0.02, 1.0],
                                    bounds=([0, 0], [1, 10]))
            print(f"  Hinge: Steer = {popt[0]:.5f} * [H - {popt[1]:.3f}]+")
            print(f"    kappa = {popt[0]:.5f} (steerability per bit)")
            print(f"    H_c = {popt[1]:.3f} bits (capacity threshold)")

            # Compute hinge R2
            y_pred = hinge(h_vals, *popt)
            ss_res = np.sum((np.array(v5_vals) - y_pred)**2)
            ss_tot = np.sum((np.array(v5_vals) - np.mean(v5_vals))**2)
            r2_hinge = 1 - ss_res / ss_tot
            print(f"    Hinge R2 = {r2_hinge:.4f}")
        except Exception as e:
            print(f"  Hinge fit failed: {e}")

    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.scatter(h_vals, v5_vals, s=120, color='#2196F3', zorder=5, label='V5')
        ax.scatter(h_vals, mrl_vals, s=120, color='#FF9800', marker='s', zorder=5, label='MRL')

        for r, h, v5s in zip(valid, h_vals, v5_vals):
            ax.annotate(f"K0={r['k0']}", (h, v5s), textcoords="offset points",
                        xytext=(5, 8), fontsize=8, color='#1565C0')

        if len(h_vals) >= 4:
            h_line = np.linspace(0, max(h_vals) * 1.1, 200)
            ax.plot(h_line, slope * h_line + intercept, '--', color='#F44336',
                    linewidth=1.5, label=f'Linear (R2={r_value**2:.3f})')
            try:
                ax.plot(h_line, hinge(h_line, *popt), '-', color='#4CAF50',
                        linewidth=2, label=f'Hinge (R2={r2_hinge:.3f})')
            except:
                pass

        ax.set_xlabel('H(L1|L0) [bits]', fontsize=13)
        ax.set_ylabel('Steerability', fontsize=13)
        ax.set_title('Causal Intervention: V5 Steerability vs Controlled H(L1|L0)\n'
                      '(Same text, different hierarchy structure)',
                      fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.grid(True, alpha=0.3)

        fig_dir = RESULTS_DIR / "figures"
        fig_dir.mkdir(exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(fig_dir / f"synthetic_hierarchy_causal.{ext}", dpi=300, bbox_inches='tight')
        print(f"\n  Figure saved to {fig_dir / 'synthetic_hierarchy_causal.png'}")
        plt.close()
    except ImportError:
        print("  matplotlib not available")

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
