"""
Pareto Analysis + Dual-Encoder Baseline
========================================

Two experiments to demonstrate V5 utility:

1. PARETO ANALYSIS (from existing data):
   - For mixed workloads (alpha L0, (1-alpha) L1), compute mixed accuracy
   - V5-adaptive: use j=1 (64d) for L0 queries, j=4 (256d) for L1 queries
   - MRL baselines: fixed at j=1 or j=4
   - Show V5-adaptive dominates MRL on Pareto frontier

2. DUAL-ENCODER BASELINE (new training):
   - E_L0: trained ONLY on L0 labels at all prefix lengths (256d)
   - E_L1: = MRL (already trained on L1 at all prefix lengths)
   - Compare: V5 single model vs dual-encoder (two models)
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

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from mrl_v5_baseline import MRLTrainerV5

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ============================================================================
# Part 1: Pareto Analysis from Existing Data
# ============================================================================

def load_classification_data():
    """Load existing 5-seed CLINC benchmark data."""
    path = RESULTS_DIR / "benchmark_bge-small_clinc.json"
    with open(path) as f:
        data = json.load(f)
    return data


def load_retrieval_data():
    """Load existing 3-seed CLINC retrieval benchmark data."""
    path = RESULTS_DIR / "retrieval_benchmark_bge-small_clinc.json"
    with open(path) as f:
        data = json.load(f)
    return data


def compute_pareto_classification(data):
    """
    Compute Pareto frontiers for classification accuracy.

    V5-adaptive: j=1 for L0 queries, j=4 for L1 queries
    MRL-fixed-j: use fixed prefix length for all queries
    """
    seeds = data["seeds"]
    alphas = np.linspace(0, 1, 21)  # 0 to 1 in steps of 0.05
    dims = {1: 64, 2: 128, 3: 192, 4: 256}

    results = {
        "v5_adaptive": [],
        "mrl_fixed_64": [],
        "mrl_fixed_128": [],
        "mrl_fixed_192": [],
        "mrl_fixed_256": [],
        "v5_fixed_256": [],
    }

    for alpha in alphas:
        v5_mixed = []
        mrl_j1_mixed = []
        mrl_j2_mixed = []
        mrl_j3_mixed = []
        mrl_j4_mixed = []
        v5_j4_mixed = []

        for seed in [str(s) for s in seeds]:
            v5_pa = data["v5"][seed]["prefix_accuracy"]
            mrl_pa = data["mrl"][seed]["prefix_accuracy"]

            # V5 adaptive: j=1 for L0, j=4 for L1
            v5_l0 = v5_pa["j1_l0"]
            v5_l1 = v5_pa["j4_l1"]
            v5_m = alpha * v5_l0 + (1 - alpha) * v5_l1
            v5_mixed.append(v5_m)

            # V5 fixed 256d (no adaptation)
            v5_j4_l0 = v5_pa["j4_l0"]
            v5_j4_l1 = v5_pa["j4_l1"]
            v5_j4_m = alpha * v5_j4_l0 + (1 - alpha) * v5_j4_l1
            v5_j4_mixed.append(v5_j4_m)

            # MRL at various fixed j
            for j_key, mixed_list in [("j1", mrl_j1_mixed), ("j2", mrl_j2_mixed),
                                       ("j3", mrl_j3_mixed), ("j4", mrl_j4_mixed)]:
                l0 = mrl_pa[f"{j_key}_l0"]
                l1 = mrl_pa[f"{j_key}_l1"]
                mixed_list.append(alpha * l0 + (1 - alpha) * l1)

        v5_dim = alpha * 64 + (1 - alpha) * 256

        results["v5_adaptive"].append({
            "alpha": float(alpha),
            "avg_dim": float(v5_dim),
            "mixed_acc_mean": float(np.mean(v5_mixed)),
            "mixed_acc_std": float(np.std(v5_mixed, ddof=1)),
        })
        results["v5_fixed_256"].append({
            "alpha": float(alpha),
            "avg_dim": 256.0,
            "mixed_acc_mean": float(np.mean(v5_j4_mixed)),
            "mixed_acc_std": float(np.std(v5_j4_mixed, ddof=1)),
        })
        for name, mix, d in [
            ("mrl_fixed_64", mrl_j1_mixed, 64),
            ("mrl_fixed_128", mrl_j2_mixed, 128),
            ("mrl_fixed_192", mrl_j3_mixed, 192),
            ("mrl_fixed_256", mrl_j4_mixed, 256),
        ]:
            results[name].append({
                "alpha": float(alpha),
                "avg_dim": float(d),
                "mixed_acc_mean": float(np.mean(mix)),
                "mixed_acc_std": float(np.std(mix, ddof=1)),
            })

    return results


def compute_pareto_retrieval(data):
    """
    Compute Pareto frontiers for retrieval (Recall@1).
    Uses existing 3-seed retrieval data.
    """
    seeds = list(data["results_by_seed"].keys())
    alphas = np.linspace(0, 1, 21)

    results = {
        "v5_adaptive": [],
        "mrl_fixed_64": [],
        "mrl_fixed_256": [],
    }

    for alpha in alphas:
        v5_mixed = []
        mrl_j1_mixed = []
        mrl_j4_mixed = []

        for seed in seeds:
            sd = data["results_by_seed"][seed]

            # V5 adaptive: j=1 for L0, j=4 for L1
            v5_l0_j1 = sd["v5"]["1"]["L0"]["recall@1"]
            v5_l1_j4 = sd["v5"]["4"]["L1"]["recall@1"]
            v5_mixed.append(alpha * v5_l0_j1 + (1 - alpha) * v5_l1_j4)

            # MRL fixed j=1
            mrl_l0_j1 = sd["mrl"]["1"]["L0"]["recall@1"]
            mrl_l1_j1 = sd["mrl"]["1"]["L1"]["recall@1"]
            mrl_j1_mixed.append(alpha * mrl_l0_j1 + (1 - alpha) * mrl_l1_j1)

            # MRL fixed j=4
            mrl_l0_j4 = sd["mrl"]["4"]["L0"]["recall@1"]
            mrl_l1_j4 = sd["mrl"]["4"]["L1"]["recall@1"]
            mrl_j4_mixed.append(alpha * mrl_l0_j4 + (1 - alpha) * mrl_l1_j4)

        v5_dim = alpha * 64 + (1 - alpha) * 256

        results["v5_adaptive"].append({
            "alpha": float(alpha),
            "avg_dim": float(v5_dim),
            "mixed_recall_mean": float(np.mean(v5_mixed)),
            "mixed_recall_std": float(np.std(v5_mixed, ddof=1)),
        })
        results["mrl_fixed_64"].append({
            "alpha": float(alpha),
            "avg_dim": 64.0,
            "mixed_recall_mean": float(np.mean(mrl_j1_mixed)),
            "mixed_recall_std": float(np.std(mrl_j1_mixed, ddof=1)),
        })
        results["mrl_fixed_256"].append({
            "alpha": float(alpha),
            "avg_dim": 256.0,
            "mixed_recall_mean": float(np.mean(mrl_j4_mixed)),
            "mixed_recall_std": float(np.std(mrl_j4_mixed, ddof=1)),
        })

    return results


def print_pareto_summary(cls_results, ret_results):
    """Print summary of Pareto analysis."""
    print("=" * 70)
    print("PARETO ANALYSIS: CLASSIFICATION (k-NN Accuracy)")
    print("=" * 70)

    print(f"\n{'alpha':>6} | {'V5-adapt':>10} {'dim':>5} | {'MRL-64d':>10} | {'MRL-256d':>10} | {'V5 wins?':>8}")
    print("-" * 70)

    for i, alpha in enumerate([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]):
        idx = int(alpha * 20)
        v5 = cls_results["v5_adaptive"][idx]
        mrl64 = cls_results["mrl_fixed_64"][idx]
        mrl256 = cls_results["mrl_fixed_256"][idx]

        v5_better = v5["mixed_acc_mean"] > mrl256["mixed_acc_mean"]

        print(f"{alpha:>6.1f} | {v5['mixed_acc_mean']:>8.3f} {v5['avg_dim']:>5.0f}d | "
              f"{mrl64['mixed_acc_mean']:>8.3f}   | {mrl256['mixed_acc_mean']:>8.3f}   | "
              f"{'YES' if v5_better else 'no':>8}")

    # Find crossover
    for i in range(len(cls_results["v5_adaptive"])):
        v5 = cls_results["v5_adaptive"][i]["mixed_acc_mean"]
        mrl = cls_results["mrl_fixed_256"][i]["mixed_acc_mean"]
        if v5 > mrl:
            alpha = cls_results["v5_adaptive"][i]["alpha"]
            print(f"\nCrossover: V5 adaptive beats MRL-256d at alpha >= {alpha:.2f}")
            print(f"  (when >= {alpha*100:.0f}% of queries are coarse)")
            break

    # Compute savings at alpha=0.5
    v5_50 = cls_results["v5_adaptive"][10]
    mrl256_50 = cls_results["mrl_fixed_256"][10]
    dim_saving = (256 - v5_50["avg_dim"]) / 256 * 100
    acc_diff = (v5_50["mixed_acc_mean"] - mrl256_50["mixed_acc_mean"]) * 100
    print(f"\nAt alpha=0.5 (equal mix):")
    print(f"  V5 adaptive: {v5_50['mixed_acc_mean']:.3f} at {v5_50['avg_dim']:.0f}d")
    print(f"  MRL-256d:    {mrl256_50['mixed_acc_mean']:.3f} at 256d")
    print(f"  Dim saving: {dim_saving:.0f}%, Acc diff: {acc_diff:+.1f}pp")

    print("\n" + "=" * 70)
    print("PARETO ANALYSIS: RETRIEVAL (Recall@1)")
    print("=" * 70)

    for alpha in [0.3, 0.5, 0.7]:
        idx = int(alpha * 20)
        v5 = ret_results["v5_adaptive"][idx]
        mrl64 = ret_results["mrl_fixed_64"][idx]
        mrl256 = ret_results["mrl_fixed_256"][idx]
        print(f"\nalpha={alpha}: V5-adapt={v5['mixed_recall_mean']:.3f} ({v5['avg_dim']:.0f}d), "
              f"MRL-64d={mrl64['mixed_recall_mean']:.3f}, MRL-256d={mrl256['mixed_recall_mean']:.3f}")


# ============================================================================
# Part 2: Dual-Encoder Baseline
# ============================================================================

def train_l0_only_model(config, train_data, val_data, num_l0, device, seed, stage1_epochs=5):
    """
    Train a model with ONLY L0 supervision at all prefix lengths.
    This is E_L0 in the dual-encoder baseline.

    Implementation: Use MRL trainer but with L0 labels as the target.
    We modify the dataset so l1 = l0 and num_l1 = num_l0.
    """
    # Create modified dataset where L1 labels = L0 labels
    modified_train = []
    for text, l0, l1 in train_data.samples:
        modified_train.append((text, l0, l0))  # L1 replaced with L0

    modified_val = []
    for text, l0, l1 in val_data.samples:
        modified_val.append((text, l0, l0))

    class ModifiedDataset:
        def __init__(self, samples, l0_names, l1_names):
            self.samples = samples
            self.level0_names = l0_names
            self.level1_names = l0_names  # Use L0 names for L1 too

    mod_train = ModifiedDataset(modified_train, train_data.level0_names, train_data.level0_names)
    mod_val = ModifiedDataset(modified_val, val_data.level0_names, val_data.level0_names)

    # Use MRL trainer (all prefixes on same label) with L0 as target
    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l0,  # L1 head also outputs L0 logits
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = MRLTrainerV5(
        model=model,
        train_dataset=mod_train,
        val_dataset=mod_val,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=0,
    )
    trainer.train(batch_size=16, patience=5)
    model.eval()
    return model


def evaluate_knn(model, test_data, device, prefix_len=None, k=5):
    """Evaluate k-NN accuracy at given prefix length for both L0 and L1."""
    texts = [s[0] for s in test_data.samples]
    l0_labels = np.array([s[1] for s in test_data.samples])
    l1_labels = np.array([s[2] for s in test_data.samples])

    embs = model.encode(texts, batch_size=32, prefix_len=prefix_len)
    embs = embs.numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-9)

    # k-NN
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1)

    top_k = np.argsort(-sims, axis=1)[:, :k]

    # L0 accuracy
    l0_preds = []
    for i in range(len(texts)):
        neighbor_labels = l0_labels[top_k[i]]
        counts = np.bincount(neighbor_labels, minlength=len(test_data.level0_names))
        l0_preds.append(np.argmax(counts))
    l0_acc = np.mean(np.array(l0_preds) == l0_labels)

    # L1 accuracy
    l1_preds = []
    for i in range(len(texts)):
        neighbor_labels = l1_labels[top_k[i]]
        counts = np.bincount(neighbor_labels, minlength=len(test_data.level1_names))
        l1_preds.append(np.argmax(counts))
    l1_acc = np.mean(np.array(l1_preds) == l1_labels)

    return float(l0_acc), float(l1_acc)


def run_dual_encoder_experiment(seeds=[42, 123, 456]):
    """
    Train L0-only encoders and compare dual-encoder vs V5.

    Dual-encoder: E_L0 (256d) for L0 queries + E_L1=MRL (256d) for L1 queries
    V5: single model, j=1 (64d) for L0 queries, j=4 (256d) for L1 queries
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODELS["bge-small"]

    print("=" * 70)
    print("DUAL-ENCODER BASELINE EXPERIMENT")
    print("=" * 70)

    all_results = []

    for seed in seeds:
        print(f"\n{'=' * 50}")
        print(f"  SEED {seed}")
        print(f"{'=' * 50}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load data
        print("[1] Loading CLINC...")
        train_data = load_hierarchical_dataset("clinc", split="train", max_samples=10000)
        test_data = load_hierarchical_dataset("clinc", split="test", max_samples=2000)
        train_ds, val_ds = split_train_val(train_data, val_ratio=0.1, seed=seed)

        num_l0 = len(train_data.level0_names)
        num_l1 = len(train_data.level1_names)

        # Train L0-only encoder
        print(f"\n[2] Training L0-only encoder (E_L0)...")
        e_l0_model = train_l0_only_model(config, train_ds, val_ds, num_l0, device, seed)

        # Evaluate E_L0 at full 256d
        print("[3] Evaluating E_L0 at 256d...")
        e_l0_l0_acc, e_l0_l1_acc = evaluate_knn(e_l0_model, test_data, device, prefix_len=None)
        print(f"    E_L0 (256d): L0={e_l0_l0_acc:.3f}, L1={e_l0_l1_acc:.3f}")

        # Also evaluate E_L0 at 64d
        e_l0_64_l0_acc, e_l0_64_l1_acc = evaluate_knn(e_l0_model, test_data, device, prefix_len=1)
        print(f"    E_L0 (64d):  L0={e_l0_64_l0_acc:.3f}, L1={e_l0_64_l1_acc:.3f}")

        del e_l0_model
        gc.collect()
        torch.cuda.empty_cache()

        # Train V5 model
        print(f"\n[4] Training V5 model...")
        from fractal_v5 import FractalModelV5
        v5_model = FractalModelV5(
            config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
            num_scales=4, scale_dim=64, device=device,
        ).to(device)
        v5_trainer = V5Trainer(
            model=v5_model, train_dataset=train_ds, val_dataset=val_ds,
            device=device, stage1_epochs=5, stage2_epochs=0,
        )
        v5_trainer.train(batch_size=16, patience=5)
        v5_model.eval()

        # Evaluate V5 at j=1 (64d) and j=4 (256d)
        print("[5] Evaluating V5...")
        v5_j1_l0, v5_j1_l1 = evaluate_knn(v5_model, test_data, device, prefix_len=1)
        v5_j4_l0, v5_j4_l1 = evaluate_knn(v5_model, test_data, device, prefix_len=None)
        print(f"    V5 (64d):  L0={v5_j1_l0:.3f}, L1={v5_j1_l1:.3f}")
        print(f"    V5 (256d): L0={v5_j4_l0:.3f}, L1={v5_j4_l1:.3f}")

        del v5_model
        gc.collect()
        torch.cuda.empty_cache()

        # Train MRL model (= E_L1)
        print(f"\n[6] Training MRL model (E_L1)...")
        mrl_model = FractalModelV5(
            config=config, num_l0_classes=num_l1, num_l1_classes=num_l1,
            num_scales=4, scale_dim=64, device=device,
        ).to(device)
        mrl_trainer = MRLTrainerV5(
            model=mrl_model, train_dataset=train_ds, val_dataset=val_ds,
            device=device, stage1_epochs=5, stage2_epochs=0,
        )
        mrl_trainer.train(batch_size=16, patience=5)
        mrl_model.eval()

        # Evaluate MRL at j=4 (256d) = E_L1
        mrl_j4_l0, mrl_j4_l1 = evaluate_knn(mrl_model, test_data, device, prefix_len=None)
        print(f"    MRL/E_L1 (256d): L0={mrl_j4_l0:.3f}, L1={mrl_j4_l1:.3f}")

        del mrl_model
        gc.collect()
        torch.cuda.empty_cache()

        result = {
            "seed": seed,
            "e_l0_256d": {"l0": e_l0_l0_acc, "l1": e_l0_l1_acc},
            "e_l0_64d": {"l0": e_l0_64_l0_acc, "l1": e_l0_64_l1_acc},
            "v5_64d": {"l0": v5_j1_l0, "l1": v5_j1_l1},
            "v5_256d": {"l0": v5_j4_l0, "l1": v5_j4_l1},
            "mrl_256d": {"l0": mrl_j4_l0, "l1": mrl_j4_l1},
        }
        all_results.append(result)
        print(f"\n  Seed {seed} complete.")

    return all_results


def print_dual_encoder_summary(results):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("DUAL-ENCODER vs V5 COMPARISON (CLINC, bge-small)")
    print("=" * 70)

    # Aggregate
    e_l0_l0 = [r["e_l0_256d"]["l0"] for r in results]
    e_l0_l1 = [r["e_l0_256d"]["l1"] for r in results]
    mrl_l0 = [r["mrl_256d"]["l0"] for r in results]
    mrl_l1 = [r["mrl_256d"]["l1"] for r in results]
    v5_j1_l0 = [r["v5_64d"]["l0"] for r in results]
    v5_j1_l1 = [r["v5_64d"]["l1"] for r in results]
    v5_j4_l0 = [r["v5_256d"]["l0"] for r in results]
    v5_j4_l1 = [r["v5_256d"]["l1"] for r in results]

    print(f"\n{'System':>25} | {'L0 query':>12} | {'L1 query':>12} | {'Index':>8} | {'Models':>6}")
    print("-" * 75)

    # Dual-encoder: E_L0 for L0 queries, MRL for L1 queries
    dual_l0 = np.mean(e_l0_l0)
    dual_l1 = np.mean(mrl_l1)
    print(f"{'Dual (E_L0+E_L1, 256d)':>25} | {dual_l0:.3f}+/-{np.std(e_l0_l0,ddof=1):.3f} | "
          f"{dual_l1:.3f}+/-{np.std(mrl_l1,ddof=1):.3f} | {'2x256d':>8} | {'2':>6}")

    # V5 adaptive: j=1 for L0, j=4 for L1
    print(f"{'V5 adapt (64d+256d)':>25} | {np.mean(v5_j1_l0):.3f}+/-{np.std(v5_j1_l0,ddof=1):.3f} | "
          f"{np.mean(v5_j4_l1):.3f}+/-{np.std(v5_j4_l1,ddof=1):.3f} | {'1x256d':>8} | {'1':>6}")

    # MRL-256d for everything
    print(f"{'MRL (256d for all)':>25} | {np.mean(mrl_l0):.3f}+/-{np.std(mrl_l0,ddof=1):.3f} | "
          f"{np.mean(mrl_l1):.3f}+/-{np.std(mrl_l1,ddof=1):.3f} | {'1x256d':>8} | {'1':>6}")

    # V5 fixed 256d
    print(f"{'V5 (256d for all)':>25} | {np.mean(v5_j4_l0):.3f}+/-{np.std(v5_j4_l0,ddof=1):.3f} | "
          f"{np.mean(v5_j4_l1):.3f}+/-{np.std(v5_j4_l1,ddof=1):.3f} | {'1x256d':>8} | {'1':>6}")

    print("\nKey insight:")
    print(f"  V5 adaptive L0 at 64d: {np.mean(v5_j1_l0):.3f} vs Dual E_L0 at 256d: {dual_l0:.3f}")
    print(f"  V5 uses 4x less compute for L0 queries")
    print(f"  V5 L1 at 256d: {np.mean(v5_j4_l1):.3f} vs Dual E_L1 at 256d: {dual_l1:.3f}")
    print(f"  V5 uses 1 model+index vs dual's 2 models+2 indexes")


# ============================================================================
# Part 3: Figure Generation
# ============================================================================

def generate_pareto_figure(cls_results):
    """Generate Pareto curve figure for the paper."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Mixed accuracy vs alpha (workload mix)
    ax = axes[0]
    alphas = [r["alpha"] for r in cls_results["v5_adaptive"]]

    for name, color, ls, label in [
        ("v5_adaptive", "#1E88E5", "-", "V5 adaptive (64d L0, 256d L1)"),
        ("mrl_fixed_256", "#E53935", "--", "MRL fixed 256d"),
        ("mrl_fixed_64", "#FB8C00", ":", "MRL fixed 64d"),
    ]:
        means = [r["mixed_acc_mean"] for r in cls_results[name]]
        stds = [r["mixed_acc_std"] for r in cls_results[name]]
        ax.plot(alphas, means, color=color, ls=ls, linewidth=2, label=label)
        ax.fill_between(alphas,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

    ax.set_xlabel("Fraction of L0 (coarse) queries ($\\alpha$)", fontsize=11)
    ax.set_ylabel("Mixed k-NN Accuracy", fontsize=11)
    ax.set_title("Classification: Workload-Adaptive Routing", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Mixed accuracy vs avg dimensionality (Pareto)
    ax = axes[1]

    # V5 adaptive trace (varies with alpha)
    v5_dims = [r["avg_dim"] for r in cls_results["v5_adaptive"]]
    v5_accs = [r["mixed_acc_mean"] for r in cls_results["v5_adaptive"]]
    ax.plot(v5_dims, v5_accs, color="#1E88E5", linewidth=2.5, label="V5 adaptive", zorder=5)

    # MRL operating points (fixed dims)
    for name, dim, color, marker in [
        ("mrl_fixed_64", 64, "#E53935", "s"),
        ("mrl_fixed_128", 128, "#E53935", "D"),
        ("mrl_fixed_192", 192, "#E53935", "^"),
        ("mrl_fixed_256", 256, "#E53935", "o"),
    ]:
        # Average across alphas at alpha=0.5 to show one point
        # Actually show the range as the alpha varies
        accs = [r["mixed_acc_mean"] for r in cls_results[name]]
        # Show at alpha=0.5
        acc_50 = cls_results[name][10]["mixed_acc_mean"]
        ax.scatter([dim], [acc_50], color=color, marker=marker, s=80, zorder=6, edgecolors='black', linewidth=0.5)

    # Annotate MRL points
    ax.annotate("MRL operating\npoints ($\\alpha$=0.5)", xy=(180, cls_results["mrl_fixed_192"][10]["mixed_acc_mean"]),
                fontsize=8, color="#E53935", ha='center')

    ax.set_xlabel("Average Embedding Dimensions", fontsize=11)
    ax.set_ylabel("Mixed k-NN Accuracy ($\\alpha$=0 to 1)", fontsize=11)
    ax.set_title("Pareto Frontier: Accuracy vs Compute", fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig_dir = Path(__file__).parent.parent / "results" / "figures" / "paper"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig11_pareto.png", dpi=150, bbox_inches='tight')
    fig.savefig(fig_dir / "fig11_pareto.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"\nFig 11 saved: {fig_dir / 'fig11_pareto.png'}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PARETO ANALYSIS + DUAL-ENCODER BASELINE")
    print("=" * 70)

    # Part 1: Pareto from existing data
    print("\n[PART 1] Loading existing benchmark data...")
    cls_data = load_classification_data()
    ret_data = load_retrieval_data()

    print("Computing classification Pareto frontiers...")
    cls_pareto = compute_pareto_classification(cls_data)

    print("Computing retrieval Pareto frontiers...")
    ret_pareto = compute_pareto_retrieval(ret_data)

    print_pareto_summary(cls_pareto, ret_pareto)

    # Generate figure
    generate_pareto_figure(cls_pareto)

    # Part 2: Dual-encoder (needs GPU)
    print("\n\n[PART 2] Running dual-encoder baseline experiment...")
    dual_results = run_dual_encoder_experiment(seeds=[42, 123, 456])
    print_dual_encoder_summary(dual_results)

    # Save all results
    output = {
        "timestamp": str(datetime.now().isoformat()),
        "classification_pareto": cls_pareto,
        "retrieval_pareto": ret_pareto,
        "dual_encoder": dual_results,
    }

    out_path = RESULTS_DIR / "pareto_dual_encoder_clinc.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
