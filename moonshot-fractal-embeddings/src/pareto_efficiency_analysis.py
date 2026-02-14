"""
Pareto efficiency analysis: steerability vs compute/parameters.
================================================================

Shows V5 achieves competitive steerability at a fraction of HEAL's compute.

Usage:
    python src/pareto_efficiency_analysis.py
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))

from external_baselines.common import ALL_DATASETS, ALL_SEEDS, RESULTS_DIR

RESULTS_ROOT = Path(__file__).parent.parent / "results"
FIG_DIR = RESULTS_ROOT / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Method metadata
METHOD_META = {
    "v5": {
        "label": "V5 (Ours)",
        "color": "#2196F3",
        "marker": "o",
        "epochs": 5,
        "batch_size": 16,
        "loss_type": "CE",  # O(n) cross-entropy
        "architecture": "Linear projection (384->256) + 2 classifiers",
    },
    "mrl": {
        "label": "MRL",
        "color": "#FF9800",
        "marker": "s",
        "epochs": 5,
        "batch_size": 16,
        "loss_type": "CE",
        "architecture": "Same as V5 (different loss only)",
    },
    "heal": {
        "label": "HEAL",
        "color": "#4CAF50",
        "marker": "D",
        "epochs": 30,
        "batch_size": 128,
        "loss_type": "SupCon",  # O(n^2) contrastive
        "architecture": "Shared trunk MLP + L0/L1 projectors + classifiers",
    },
    "csr": {
        "label": "CSR",
        "color": "#9C27B0",
        "marker": "^",
        "epochs": 20,
        "batch_size": 128,
        "loss_type": "Recon+Con",
        "architecture": "Sparse autoencoder + decoder + classifiers",
    },
    "smec": {
        "label": "SMEC",
        "color": "#F44336",
        "marker": "v",
        "epochs": 24,  # 4 stages x 6 epochs
        "batch_size": 128,
        "loss_type": "CE+InfoNCE",
        "architecture": "MLP + per-stage classifiers + ADS",
    },
}


def estimate_training_flops(method, n_train, head_params):
    """Estimate total training FLOPs for a method."""
    meta = METHOD_META[method]
    epochs = meta["epochs"]
    bs = meta["batch_size"]
    n_batches = (n_train + bs - 1) // bs

    # Forward pass: ~2 * head_params per sample (multiply-add)
    # Backward pass: ~4 * head_params per sample
    # Total per sample: ~6 * head_params
    flops_per_sample = 6 * head_params

    # Contrastive loss adds O(n^2) pairwise computations
    if meta["loss_type"] == "SupCon":
        # HEAL: two contrastive losses (L0 supcon + hierarchy-weighted)
        # Each computes bs x bs similarity matrix in embedding dim ~256
        flops_per_batch_extra = 2 * bs * bs * 256
        flops_per_sample += flops_per_batch_extra / bs
    elif "Con" in meta["loss_type"]:
        # CSR/SMEC: one contrastive loss
        flops_per_batch_extra = bs * bs * 256
        flops_per_sample += flops_per_batch_extra / bs

    total_flops = flops_per_sample * n_train * epochs
    return total_flops


def load_all_steerability():
    """Load steerability data for all methods."""
    data = {}  # method -> dataset -> [steers]

    # V5 and MRL from benchmark files
    for method_key in ["v5", "mrl"]:
        data[method_key] = {}
        for ds in ALL_DATASETS:
            bench_path = RESULTS_ROOT / f"benchmark_bge-small_{ds}.json"
            if not bench_path.exists():
                continue
            bench = json.load(open(bench_path))
            steers = []
            for seed in ALL_SEEDS:
                seed_key = str(seed)
                if seed_key in bench.get(method_key, {}):
                    pa = bench[method_key][seed_key].get("prefix_accuracy", {})
                    s = (pa.get("j1_l0", 0) - pa.get("j4_l0", 0)) + \
                        (pa.get("j4_l1", 0) - pa.get("j1_l1", 0))
                    steers.append(s)
            if steers:
                data[method_key][ds] = steers

    # External baselines
    for method_key in ["heal", "csr", "smec"]:
        data[method_key] = {}
        for ds in ALL_DATASETS:
            steers = []
            params_list = []
            for seed in ALL_SEEDS:
                path = RESULTS_DIR / method_key / f"{ds}_seed{seed}.json"
                if path.exists():
                    r = json.load(open(path))
                    steers.append(r["steerability"])
                    params_list.append(r["training"].get("head_params", 0))
            if steers:
                data[method_key][ds] = steers

    return data


def get_head_params(method, ds):
    """Get head parameters for a method on a dataset."""
    if method in ["v5", "mrl"]:
        # Estimate V5/MRL params: linear(384,256) + L0_head + L1_head
        # Need dataset class counts
        ds_classes = {
            "yahoo": (4, 10), "goemotions": (4, 28), "newsgroups": (6, 20),
            "trec": (6, 50), "hupd_sec_cls": (8, 121), "arxiv": (20, 123),
            "dbpedia_classes": (9, 70), "clinc": (10, 150), "wos": (10, 336),
            "hupd_sec_sub": (8, 587), "hwv_l0_l2": (10, 253), "hwv_l0_l3": (10, 230),
        }
        if ds not in ds_classes:
            return 130000  # estimate
        k0, k1 = ds_classes[ds]
        proj = 384 * 256 + 256  # weight + bias
        l0 = 256 * k0 + k0
        l1 = 256 * k1 + k1
        return proj + l0 + l1
    else:
        # Load from result file
        for seed in ALL_SEEDS:
            path = RESULTS_DIR / method / f"{ds}_seed{seed}.json"
            if path.exists():
                r = json.load(open(path))
                return r["training"].get("head_params", 0)
    return 0


def generate_pareto_plot(data):
    """Generate the Pareto efficiency plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Steerability vs Head Parameters
    ax1 = axes[0]
    for method in ["v5", "mrl", "heal", "csr", "smec"]:
        meta = METHOD_META[method]
        params_list = []
        steer_list = []

        for ds in data.get(method, {}):
            hp = get_head_params(method, ds)
            if hp > 0:
                mean_steer = np.mean(data[method][ds])
                params_list.append(hp / 1000)  # in K
                steer_list.append(mean_steer)

        if params_list:
            ax1.scatter(params_list, steer_list, label=meta["label"],
                       color=meta["color"], marker=meta["marker"], s=60, alpha=0.7)

    ax1.set_xlabel("Head Parameters (K)")
    ax1.set_ylabel("Mean Steerability")
    ax1.set_title("Steerability vs Model Complexity")
    ax1.legend(fontsize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Plot 2: Steerability vs Training Compute (estimated GFLOPs)
    ax2 = axes[1]
    n_train_default = 10000

    for method in ["v5", "mrl", "heal", "csr", "smec"]:
        meta = METHOD_META[method]
        flops_list = []
        steer_list = []

        for ds in data.get(method, {}):
            hp = get_head_params(method, ds)
            if hp > 0:
                flops = estimate_training_flops(method, n_train_default, hp)
                mean_steer = np.mean(data[method][ds])
                flops_list.append(flops / 1e9)  # in GFLOPs
                steer_list.append(mean_steer)

        if flops_list:
            ax2.scatter(flops_list, steer_list, label=meta["label"],
                       color=meta["color"], marker=meta["marker"], s=60, alpha=0.7)

    ax2.set_xlabel("Estimated Training FLOPs (G)")
    ax2.set_ylabel("Mean Steerability")
    ax2.set_title("Steerability vs Training Compute")
    ax2.legend(fontsize=8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    out_path = FIG_DIR / "fig_pareto_efficiency.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def print_efficiency_table(data):
    """Print efficiency comparison table."""
    print("\n" + "=" * 90)
    print("  EFFICIENCY COMPARISON TABLE")
    print("=" * 90)

    print(f"{'Method':<10} {'Params (K)':<12} {'Epochs':<8} {'Est. GFLOPs':<14} "
          f"{'Mean S':<10} {'S/GFLOPs':<12}")
    print("-" * 90)

    n_train = 10000

    for method in ["v5", "mrl", "heal", "csr", "smec"]:
        meta = METHOD_META[method]

        # Aggregate across datasets
        all_steers = []
        all_params = []
        all_flops = []

        for ds in data.get(method, {}):
            hp = get_head_params(method, ds)
            if hp > 0:
                flops = estimate_training_flops(method, n_train, hp)
                mean_steer = np.mean(data[method][ds])
                all_steers.append(mean_steer)
                all_params.append(hp)
                all_flops.append(flops)

        if all_steers:
            avg_params = np.mean(all_params) / 1000
            avg_flops = np.mean(all_flops) / 1e9
            avg_steer = np.mean(all_steers)
            efficiency = avg_steer / avg_flops if avg_flops > 0 else 0

            print(f"{meta['label']:<10} {avg_params:<12.1f} {meta['epochs']:<8} "
                  f"{avg_flops:<14.1f} {avg_steer:<+10.4f} {efficiency:<12.6f}")

    print()


def main():
    print("Loading steerability data...")
    data = load_all_steerability()

    print_efficiency_table(data)
    generate_pareto_plot(data)

    # Also compute the steerability-per-compute ratio
    print("\n=== KEY FINDING ===")

    # V5 mean steerability across all datasets
    v5_steers = [np.mean(s) for s in data.get("v5", {}).values()]
    heal_steers = [np.mean(s) for s in data.get("heal", {}).values()]

    if v5_steers and heal_steers:
        v5_avg = np.mean(v5_steers)
        heal_avg = np.mean(heal_steers)

        # Rough compute ratio: V5 = 5 epochs * 130K params; HEAL = 30 epochs * 600K params
        v5_compute = 5 * 130000
        heal_compute = 30 * 600000
        compute_ratio = heal_compute / v5_compute

        print(f"V5 mean steerability: {v5_avg:+.4f} (across {len(v5_steers)} datasets)")
        print(f"HEAL mean steerability: {heal_avg:+.4f} (across {len(heal_steers)} datasets)")
        print(f"HEAL/V5 steerability ratio: {heal_avg/v5_avg:.1f}x")
        print(f"HEAL/V5 compute ratio: ~{compute_ratio:.0f}x")
        print(f"V5 steerability per unit compute: {v5_avg/v5_compute*1e6:.4f}")
        print(f"HEAL steerability per unit compute: {heal_avg/heal_compute*1e6:.4f}")
        print(f"V5 is {(v5_avg/v5_compute)/(heal_avg/heal_compute):.1f}x more compute-efficient")


if __name__ == "__main__":
    main()
