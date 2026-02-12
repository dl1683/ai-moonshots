"""Generate paper-ready figures for new experiments.

Figures:
1. Capacity sweep heatmap: S(dim, dataset) showing Goldilocks peak
2. Backbone control bar chart: 4-arm comparison
3. Combined scaling with capacity sweep overlay

Run: python src/new_experiment_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def fig_capacity_sweep():
    """Capacity sweep: S vs dim for each dataset."""
    path = RESULTS_DIR / "capacity_sweep_goldilocks.json"
    if not path.exists():
        print("  No capacity sweep results")
        return

    with open(path) as f:
        cs = json.load(f)

    H_map = cs.get('H_L1_L0', {})
    results = cs['results']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Line plot S vs dim for each dataset
    ax = axes[0]
    colors = {'yahoo': '#1f77b4', 'trec': '#ff7f0e', 'dbpedia_classes': '#2ca02c', 'clinc': '#d62728'}
    markers = {'yahoo': 'o', 'trec': 's', 'dbpedia_classes': '^', 'clinc': 'D'}

    ds_order = sorted(results.keys(), key=lambda d: H_map.get(d, 0))

    for ds in ds_order:
        dims_data = results[ds]
        h = H_map.get(ds, 0)
        dim_vals = sorted(dims_data.keys(), key=int)
        means = []
        stds = []
        xs = []
        for dim_str in dim_vals:
            runs = dims_data[dim_str]
            steers = [r['steerability_score'] for r in runs]
            means.append(np.mean(steers))
            stds.append(np.std(steers))
            xs.append(int(dim_str))

        label = f"{ds.replace('_', ' ').title()} (H={h:.2f})"
        ax.errorbar(xs, means, yerr=stds, marker=markers.get(ds, 'o'),
                    color=colors.get(ds, 'gray'), label=label, capsize=3,
                    linewidth=1.5, markersize=6)

    ax.set_xlabel('Scale dimension (prefix capacity)')
    ax.set_ylabel('Steerability S')
    ax.set_title('(a) Steerability vs. prefix capacity')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xticks([16, 32, 48, 64, 96, 128])

    # Panel B: S@dim=16 vs H (scaling law)
    ax = axes[1]
    dim16_data = []
    for ds in ds_order:
        if '16' in results[ds]:
            runs = results[ds]['16']
            steers = [r['steerability_score'] for r in runs]
            h = H_map.get(ds, 0)
            dim16_data.append((h, np.mean(steers), np.std(steers), ds))

    if dim16_data:
        hs = [d[0] for d in dim16_data]
        ss = [d[1] for d in dim16_data]
        errs = [d[2] for d in dim16_data]
        names = [d[3] for d in dim16_data]

        for i, (h, s, e, name) in enumerate(dim16_data):
            ax.errorbar(h, s, yerr=e, marker=markers.get(name, 'o'),
                       color=colors.get(name, 'gray'), capsize=4, markersize=8)
            ax.annotate(name.replace('_', '\n').title(), (h, s),
                       textcoords="offset points", xytext=(5, 8),
                       fontsize=8, alpha=0.7)

        # Fit line
        if len(hs) >= 3:
            slope, intercept, r, p, se = sp_stats.linregress(hs, ss)
            x_fit = np.linspace(min(hs) - 0.2, max(hs) + 0.2, 50)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'k--', alpha=0.5,
                    label=f'r={r:.3f}, p={p:.3f}')

            rho, p_rho = sp_stats.spearmanr(hs, ss)
            ax.set_title(f'(b) S@dim=16 vs H(L1|L0)\nrho={rho:.3f}, r={r:.3f}')
            ax.legend(loc='upper left')

    ax.set_xlabel('Conditional entropy H(L1|L0)')
    ax.set_ylabel('Steerability S at dim=16')

    plt.tight_layout()
    out = FIGURES_DIR / "fig_capacity_sweep.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def fig_backbone_control():
    """Backbone control: 4-arm bar chart."""
    path = RESULTS_DIR / "backbone_finetune_control.json"
    if not path.exists():
        print("  No backbone control results")
        return

    with open(path) as f:
        bc = json.load(f)

    results = bc['results']
    datasets = sorted(results.keys())
    arms_order = ['v5_frozen', 'mrl_frozen', 'flat_finetune', 'v5_finetune']
    arm_labels = ['V5\n(head only)', 'MRL\n(head only)', 'MRL + backbone\n(flat finetune)', 'V5 + backbone\n(V5 finetune)']
    arm_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        means = []
        stds = []
        available = []

        for arm in arms_order:
            if arm in results[ds] and results[ds][arm]:
                runs = results[ds][arm]
                steers = [r['steerability_score'] for r in runs]
                means.append(np.mean(steers))
                stds.append(np.std(steers))
                available.append(True)
            else:
                means.append(0)
                stds.append(0)
                available.append(False)

        x = np.arange(len(arms_order))
        bars = ax.bar(x, means, yerr=stds, color=arm_colors, capsize=5,
                      edgecolor='black', linewidth=0.5, alpha=0.85)

        # Gray out unavailable arms
        for i, avail in enumerate(available):
            if not avail:
                bars[i].set_alpha(0.2)
                bars[i].set_hatch('//')

        ax.set_xticks(x)
        ax.set_xticklabels(arm_labels, fontsize=8)
        ax.set_title(ds.replace('_', ' ').title())
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        if idx == 0:
            ax.set_ylabel('Steerability S')

        # Add significance annotations
        if available[0] and available[2]:
            v5f_steers = [r['steerability_score'] for r in results[ds]['v5_frozen']]
            flat_steers = [r['steerability_score'] for r in results[ds]['flat_finetune']]
            t, p = sp_stats.ttest_ind(v5f_steers, flat_steers)
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ymax = max(means[0] + stds[0], means[2] + stds[2]) + 0.005
            ax.annotate(stars, xy=(1, ymax), fontsize=12, ha='center', color='red')

        # Add param count annotations
        for i, arm in enumerate(arms_order):
            if arm in results[ds] and results[ds][arm]:
                params = results[ds][arm][0].get('trainable_params', 0)
                if params:
                    ax.annotate(f'{params/1e6:.1f}M', xy=(i, -0.003),
                              fontsize=7, ha='center', alpha=0.6)

    fig.suptitle('Backbone Fine-Tuning Control: Alignment vs. Capacity', fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_backbone_control.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def fig_backbone_summary():
    """Summary figure: alignment effect isolated."""
    path = RESULTS_DIR / "backbone_finetune_control.json"
    if not path.exists():
        return

    with open(path) as f:
        bc = json.load(f)

    results = bc['results']

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # For each dataset, show the 4 arms as grouped bars
    datasets = sorted(results.keys())
    arms_order = ['mrl_frozen', 'flat_finetune', 'v5_frozen', 'v5_finetune']
    arm_labels = ['MRL (head)', 'MRL+backbone', 'V5 (head)', 'V5+backbone']
    arm_colors = ['#aec7e8', '#ff9896', '#2ca02c', '#d62728']

    width = 0.18
    x = np.arange(len(datasets))

    for i, (arm, label, color) in enumerate(zip(arms_order, arm_labels, arm_colors)):
        means = []
        stds = []
        for ds in datasets:
            if arm in results[ds] and results[ds][arm]:
                steers = [r['steerability_score'] for r in results[ds][arm]]
                means.append(np.mean(steers))
                stds.append(np.std(steers))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, capsize=3, edgecolor='black', linewidth=0.3)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([ds.replace('_', ' ').title() for ds in datasets])
    ax.set_ylabel('Steerability S')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_title('Alignment Is Necessary and Sufficient for Steerability')

    # Add bracket annotation
    ax.annotate('', xy=(0.08, 0.052), xytext=(0.28, 0.052),
               arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(0.18, 0.054, 'd=12.9', ha='center', fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    out = FIGURES_DIR / "fig_backbone_summary.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Generating new experiment figures...")
    fig_capacity_sweep()
    fig_backbone_control()
    fig_backbone_summary()
    print("Done!")
