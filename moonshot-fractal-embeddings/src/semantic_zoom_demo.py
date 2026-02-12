"""Semantic Zoom Demo: Cluster Nesting Purity + Retrieval Overlap.

Demonstrates the KEY property distinguishing fractal embeddings from MRL:
- V5: Truncation changes MEANING (coarse -> fine clusters)
- MRL: Truncation changes FIDELITY (same clusters, worse quality)

Metrics:
1. Cluster Nesting Purity: fine clusters nest inside coarse clusters
2. Retrieval Overlap: how much top-k results change with truncation
3. NMI with true labels at each prefix length

Run: python src/semantic_zoom_demo.py [dataset_name]
"""

import sys
import os
import json
import numpy as np
import torch
import gc
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model configs
MODELS = None  # Will import from fractal_v5


def train_and_get_embeddings(dataset_name, model_key="bge-small",
                              method="v5", seed=42, device="cuda"):
    """Train V5 or MRL, extract embeddings at all 4 prefix lengths."""
    import random
    from fractal_v5 import (FractalModelV5, V5Trainer, MODELS as V5_MODELS,
                             split_train_val)
    from mrl_v5_baseline import MRLModelV5, MRLTrainer
    from hierarchical_datasets import load_hierarchical_dataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load data
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

    class TempDataset:
        def __init__(self, samples, level0_names, level1_names):
            self.samples = samples
            self.level0_names = level0_names
            self.level1_names = level1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data.samples = train_samples

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)
    config = V5_MODELS[model_key]

    print(f"\n  Training {method.upper()} on {dataset_name} (seed={seed})...")
    print(f"  L0 classes: {num_l0}, L1 classes: {num_l1}")

    if method == "v5":
        model = FractalModelV5(
            config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
            num_scales=4, scale_dim=64, device=device,
        ).to(device)
        trainer = V5Trainer(
            model=model, train_dataset=train_data, val_dataset=val_data,
            device=device, stage1_epochs=5, stage2_epochs=0, unfreeze_layers=4,
        )
    else:
        model = MRLModelV5(
            config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
            num_scales=4, scale_dim=64, device=device,
        ).to(device)
        trainer = MRLTrainer(
            model=model, train_dataset=train_data, val_dataset=val_data,
            device=device, stage1_epochs=5, stage2_epochs=0, unfreeze_layers=4,
        )

    trainer.train(batch_size=16, patience=5)

    # Extract embeddings at each prefix length
    test_texts = [s.text for s in test_data.samples]
    test_l0 = np.array([s.level0_label for s in test_data.samples])
    test_l1 = np.array([s.level1_label for s in test_data.samples])

    embeddings = {}
    model.eval()

    # Full embedding (j=4, 256d)
    full_embs = model.encode(test_texts, batch_size=64).cpu().numpy()
    # Normalize
    norms = np.linalg.norm(full_embs, axis=1, keepdims=True)
    full_embs = full_embs / np.maximum(norms, 1e-8)
    embeddings[256] = full_embs

    # Prefix embeddings
    for j in [1, 2, 3]:
        prefix_embs = model.encode(test_texts, batch_size=64, prefix_len=j).cpu().numpy()
        norms = np.linalg.norm(prefix_embs, axis=1, keepdims=True)
        prefix_embs = prefix_embs / np.maximum(norms, 1e-8)
        embeddings[j * 64] = prefix_embs

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings, test_l0, test_l1, num_l0, num_l1


def cluster_nesting_purity(embs_coarse, embs_fine, n_coarse, n_fine):
    """Measure how well fine clusters nest inside coarse clusters."""
    km_coarse = KMeans(n_clusters=n_coarse, random_state=42, n_init=10)
    km_fine = KMeans(n_clusters=n_fine, random_state=42, n_init=10)

    coarse_labels = km_coarse.fit_predict(embs_coarse)
    fine_labels = km_fine.fit_predict(embs_fine)

    purities = []
    for fc in range(n_fine):
        mask = fine_labels == fc
        if mask.sum() < 3:
            continue
        coarse_in_fc = coarse_labels[mask]
        counts = Counter(coarse_in_fc)
        majority = counts.most_common(1)[0][1]
        purity = majority / len(coarse_in_fc)
        purities.append(purity)

    mean_purity = np.mean(purities) if purities else 0
    perfect_nesting = np.mean([p >= 0.9 for p in purities]) if purities else 0
    return float(mean_purity), float(perfect_nesting)


def retrieval_overlap(embs_short, embs_full, k=10):
    """Retrieval overlap between short and full embeddings.

    Low overlap = different results at different scales (semantic zoom).
    High overlap = same results at different scales (fidelity only).
    """
    n = min(len(embs_short), 500)  # Sample for speed
    idx = np.random.choice(len(embs_short), n, replace=False)

    # Compute similarities
    sim_short = embs_short[idx] @ embs_short.T
    sim_full = embs_full[idx] @ embs_full.T

    overlaps = []
    for i in range(n):
        # Exclude self
        sim_s = sim_short[i].copy()
        sim_f = sim_full[i].copy()
        self_idx = idx[i]
        sim_s[self_idx] = -np.inf
        sim_f[self_idx] = -np.inf

        topk_short = set(np.argsort(sim_s)[-k:])
        topk_full = set(np.argsort(sim_f)[-k:])
        overlap = len(topk_short & topk_full) / k
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def run_demo(dataset_name="clinc", seed=42):
    """Run semantic zoom demo."""
    print(f"\n{'='*70}")
    print(f"SEMANTIC ZOOM DEMO: {dataset_name}")
    print(f"{'='*70}")

    results = {}

    for method in ["v5", "mrl"]:
        torch.cuda.empty_cache()
        gc.collect()

        embs, l0_ids, l1_ids, n_l0, n_l1 = train_and_get_embeddings(
            dataset_name, method=method, seed=seed
        )

        result = {'method': method}
        n_fine_clusters = min(n_l1, 50)

        # 1. Cluster Nesting Purity (64d coarse -> 256d fine)
        purity, perfect = cluster_nesting_purity(
            embs[64], embs[256], n_coarse=n_l0, n_fine=n_fine_clusters
        )
        result['cnp_mean'] = purity
        result['cnp_perfect'] = perfect
        print(f"\n  {method.upper()} Cluster Nesting (64d->256d): purity={purity:.3f}, "
              f"perfect={perfect:.1%}")

        # Also 128d -> 256d
        purity128, perf128 = cluster_nesting_purity(
            embs[128], embs[256], n_coarse=min(n_l0 * 3, 30), n_fine=n_fine_clusters
        )
        result['cnp_128_256'] = purity128

        # 2. Retrieval Overlap
        for short_dim in [64, 128]:
            overlap = retrieval_overlap(embs[short_dim], embs[256], k=10)
            key = f'overlap_{short_dim}_256'
            result[key] = overlap
            print(f"  {method.upper()} Retrieval Overlap ({short_dim}d vs 256d): {overlap:.1%}")

        # 3. NMI with true labels at each prefix
        for dim in [64, 128, 192, 256]:
            n_clusters = n_l0 if dim <= 64 else min(n_l1, 50)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            pred = km.fit_predict(embs[dim])

            nmi_l0 = normalized_mutual_info_score(l0_ids, pred)
            nmi_l1 = normalized_mutual_info_score(l1_ids, pred)
            result[f'nmi_l0_{dim}d'] = float(nmi_l0)
            result[f'nmi_l1_{dim}d'] = float(nmi_l1)
            print(f"  {method.upper()} {dim:3d}d: NMI(L0)={nmi_l0:.3f}, NMI(L1)={nmi_l1:.3f}")

        results[method] = result

    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    v5 = results['v5']
    mrl = results['mrl']

    print(f"  {'Metric':<35} {'V5':>8} {'MRL':>8} {'Winner':>8}")
    print(f"  {'-'*60}")

    metrics = [
        ('Cluster Nesting Purity (64->256)', 'cnp_mean', True),
        ('Perfect Nesting Rate', 'cnp_perfect', True),
        ('Retrieval Overlap 64d vs 256d', 'overlap_64_256', False),
        ('Retrieval Overlap 128d vs 256d', 'overlap_128_256', False),
        ('NMI(L0) at 64d', 'nmi_l0_64d', True),
        ('NMI(L1) at 256d', 'nmi_l1_256d', True),
    ]

    for name, key, higher_is_better in metrics:
        v = v5.get(key, 0)
        m = mrl.get(key, 0)
        winner = "V5" if (v > m) == higher_is_better else "MRL"
        print(f"  {name:<35} {v:>8.3f} {m:>8.3f} {winner:>8}")

    print(f"\n  KEY INSIGHT:")
    print(f"    MRL overlap (64d vs 256d): {mrl['overlap_64_256']:.1%} (same results = fidelity only)")
    print(f"     V5 overlap (64d vs 256d): {v5['overlap_64_256']:.1%} (different results = semantic zoom)")

    # Save
    out_path = RESULTS_DIR / f"semantic_zoom_{dataset_name}.json"
    with open(out_path, 'w') as f:
        json.dump({'dataset': dataset_name, 'seed': seed, 'results': results}, f, indent=2)
    print(f"\n  Saved to {out_path}")

    # Generate figure
    make_figure(results, dataset_name)

    return results


def make_figure(results, dataset_name):
    """Generate semantic zoom comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    v5 = results['v5']
    mrl = results['mrl']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: NMI curves
    ax = axes[0]
    prefixes = [64, 128, 192, 256]
    for method, data, color in [('V5', v5, '#2ca02c'), ('MRL', mrl, '#1f77b4')]:
        nmi_l0 = [data.get(f'nmi_l0_{p}d', 0) for p in prefixes]
        nmi_l1 = [data.get(f'nmi_l1_{p}d', 0) for p in prefixes]
        ax.plot(prefixes, nmi_l0, '-o', color=color, label=f'{method} NMI(L0)',
                linewidth=2, markersize=7)
        ax.plot(prefixes, nmi_l1, '--s', color=color, label=f'{method} NMI(L1)',
                linewidth=1.5, markersize=6, alpha=0.7)
    ax.set_xlabel('Prefix dimensions')
    ax.set_ylabel('NMI with true labels')
    ax.set_title('(a) Cluster-label alignment')
    ax.legend(fontsize=8)
    ax.set_xticks(prefixes)

    # Panel B: Cluster Nesting Purity
    ax = axes[1]
    x = np.arange(2)
    vals = [v5['cnp_mean'], mrl['cnp_mean']]
    colors = ['#2ca02c', '#1f77b4']
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['V5 (Fractal)', 'MRL (Standard)'])
    ax.set_ylabel('Cluster Nesting Purity')
    ax.set_title('(b) Do fine clusters nest in coarse?')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    # Panel C: Retrieval Overlap
    ax = axes[2]
    vals = [v5['overlap_64_256'], mrl['overlap_64_256']]
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['V5 (Fractal)', 'MRL (Standard)'])
    ax.set_ylabel('Retrieval Overlap (64d vs 256d)')
    ax.set_title('(c) Same results at different scales?')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=11, fontweight='bold')

    # Annotations
    ax.annotate('Lower = more\nsemantic zoom', xy=(0.5, 0.05),
               fontsize=8, ha='center', style='italic', color='gray',
               transform=ax.transAxes)

    fig.suptitle(f'Semantic Zoom: V5 vs MRL on {dataset_name.upper()}', fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / f"fig_semantic_zoom_{dataset_name}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {out}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "clinc"
    run_demo(ds)
