"""
Generate paper-ready figures for the fractal embeddings paper.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def fig1_ablation_barplot():
    """Fig 1: Causal ablation — steerability scores for V5, Inverted, No-prefix."""
    with open(RESULTS_DIR / "ablation_steerability_bge-small_clinc.json") as f:
        abl = json.load(f)

    variants = {}
    for variant in ['v5', 'inverted', 'no_prefix']:
        seeds_data = abl['results'][variant]
        steers = [s['steerability_score'] for s in seeds_data]
        variants[variant] = {'mean': np.mean(steers), 'std': np.std(steers)}

    fig, ax = plt.subplots(figsize=(5, 3.5))

    names = ['V5\n(Ours)', 'Inverted\n(short→L1)', 'No-prefix\n(full→L1 only)']
    means = [variants['v5']['mean'], variants['inverted']['mean'], variants['no_prefix']['mean']]
    stds = [variants['v5']['std'], variants['inverted']['std'], variants['no_prefix']['std']]
    colors = ['#2196F3', '#F44336', '#9E9E9E']

    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8, width=0.6)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_ylabel('Steerability Score')
    ax.set_title('Causal Ablation: CLINC150 / bge-small (5 seeds)')

    # Annotate
    for bar, m, s in zip(bars, means, stds):
        y = m + s + 0.003 if m > 0 else m - s - 0.008
        ax.text(bar.get_x() + bar.get_width()/2, y, f'{m:+.003f}',
                ha='center', va='bottom' if m > 0 else 'top', fontsize=9, fontweight='bold')

    # Significance annotation
    ax.annotate('', xy=(0, 0.065), xytext=(1, 0.065),
                arrowprops=dict(arrowstyle='-', color='black', lw=1))
    ax.text(0.5, 0.067, 'p < 0.0001, d = 11.4', ha='center', fontsize=8)

    ax.set_ylim(-0.04, 0.085)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_ablation_barplot.png")
    plt.savefig(FIGURES_DIR / "fig1_ablation_barplot.pdf")
    plt.close()
    print("  Fig 1: Ablation bar plot saved")


def fig2_prefix_curves():
    """Fig 2: L0 and L1 accuracy vs prefix length for V5 vs MRL on CLINC."""
    with open(RESULTS_DIR / "ablation_steerability_bge-small_clinc.json") as f:
        abl = json.load(f)

    # V5 prefix data (average over 5 seeds)
    v5_data = abl['results']['v5']
    js = [1, 2, 3, 4]

    v5_l0 = [np.mean([s['prefix_results'][f'j{j}']['l0'] for s in v5_data]) for j in js]
    v5_l1 = [np.mean([s['prefix_results'][f'j{j}']['l1'] for s in v5_data]) for j in js]
    v5_l0_std = [np.std([s['prefix_results'][f'j{j}']['l0'] for s in v5_data]) for j in js]
    v5_l1_std = [np.std([s['prefix_results'][f'j{j}']['l1'] for s in v5_data]) for j in js]

    # No-prefix (acts like MRL-equivalent: no hierarchy alignment)
    nop_data = abl['results']['no_prefix']
    nop_l0 = [np.mean([s['prefix_results'][f'j{j}']['l0'] for s in nop_data]) for j in js]
    nop_l1 = [np.mean([s['prefix_results'][f'j{j}']['l1'] for s in nop_data]) for j in js]

    prefix_dims = [64, 128, 192, 256]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: V5
    ax1.plot(prefix_dims, v5_l0, 'o-', color='#2196F3', label='L0 (Coarse)', linewidth=2, markersize=6)
    ax1.fill_between(prefix_dims,
                     [m-s for m,s in zip(v5_l0, v5_l0_std)],
                     [m+s for m,s in zip(v5_l0, v5_l0_std)], alpha=0.15, color='#2196F3')
    ax1.plot(prefix_dims, v5_l1, 's-', color='#F44336', label='L1 (Fine)', linewidth=2, markersize=6)
    ax1.fill_between(prefix_dims,
                     [m-s for m,s in zip(v5_l1, v5_l1_std)],
                     [m+s for m,s in zip(v5_l1, v5_l1_std)], alpha=0.15, color='#F44336')
    ax1.set_xlabel('Prefix Dimension')
    ax1.set_ylabel('KNN Accuracy')
    ax1.set_title('V5 (Ours): Hierarchy-Aligned')
    ax1.legend(loc='lower right')
    ax1.set_xticks(prefix_dims)
    ax1.set_ylim(0.90, 1.0)
    ax1.grid(True, alpha=0.3)

    # Annotate the gap
    gap_x = 64
    ax1.annotate('', xy=(gap_x-5, v5_l0[0]), xytext=(gap_x-5, v5_l1[0]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax1.text(gap_x-15, (v5_l0[0]+v5_l1[0])/2, 'Gap', color='green', fontsize=9, ha='right')

    # Right: No-prefix (baseline)
    ax2.plot(prefix_dims, nop_l0, 'o-', color='#2196F3', label='L0 (Coarse)', linewidth=2, markersize=6)
    ax2.plot(prefix_dims, nop_l1, 's-', color='#F44336', label='L1 (Fine)', linewidth=2, markersize=6)
    ax2.set_xlabel('Prefix Dimension')
    ax2.set_ylabel('KNN Accuracy')
    ax2.set_title('No-Prefix Baseline')
    ax2.legend(loc='lower right')
    ax2.set_xticks(prefix_dims)
    ax2.set_ylim(0.90, 1.0)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('CLINC150 / bge-small: Prefix Specialization', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_prefix_curves.png")
    plt.savefig(FIGURES_DIR / "fig2_prefix_curves.pdf")
    plt.close()
    print("  Fig 2: Prefix accuracy curves saved")


def fig3_complexity_steerability():
    """Fig 3: Steerability vs hierarchy complexity across datasets."""
    # Dataset complexity stats
    datasets = {
        'Yahoo': {'branching': 3.0, 'n_l0': 10, 'n_l1': 30, 'steer_v5': 0.011, 'steer_mrl': 0.004},
        'TREC': {'branching': 8.3, 'n_l0': 6, 'n_l1': 50, 'steer_v5': None, 'steer_mrl': None},  # Will fill from prefix data
        'CLINC': {'branching': 15.0, 'n_l0': 10, 'n_l1': 150, 'steer_v5': 0.0534, 'steer_mrl': 0.0091},
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot V5 steerability vs branching factor
    names = []
    branchings = []
    steers = []
    for name, d in datasets.items():
        if d['steer_v5'] is not None:
            names.append(name)
            branchings.append(d['branching'])
            steers.append(d['steer_v5'])

    ax.scatter(branchings, steers, s=120, c='#2196F3', edgecolors='black', linewidth=1, zorder=5)

    for name, b, s in zip(names, branchings, steers):
        ax.annotate(f'{name}\n({int(datasets[name]["n_l0"])}→{int(datasets[name]["n_l1"])})',
                   xy=(b, s), xytext=(10, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Trend line
    if len(branchings) >= 2:
        z = np.polyfit(branchings, steers, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(branchings)-1, max(branchings)+2, 100)
        ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, linewidth=1)

    ax.set_xlabel('Hierarchy Branching Factor (L1/L0)')
    ax.set_ylabel('V5 Steerability Score')
    ax.set_title('Steerability Scales with Hierarchy Depth')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 18)
    ax.set_ylim(-0.01, 0.07)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_complexity_steerability.png")
    plt.savefig(FIGURES_DIR / "fig3_complexity_steerability.pdf")
    plt.close()
    print("  Fig 3: Complexity-steerability plot saved")


def fig4_ablation_prefix_curves():
    """Fig 4: Three-panel prefix curves comparing V5 vs Inverted vs No-prefix."""
    with open(RESULTS_DIR / "ablation_steerability_bge-small_clinc.json") as f:
        abl = json.load(f)

    prefix_dims = [64, 128, 192, 256]
    js = [1, 2, 3, 4]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    variants = [('v5', 'V5 (Ours)', '#2196F3'),
                ('inverted', 'Inverted', '#F44336'),
                ('no_prefix', 'No-Prefix', '#9E9E9E')]

    for ax, (variant, title, color) in zip(axes, variants):
        data = abl['results'][variant]
        l0 = [np.mean([s['prefix_results'][f'j{j}']['l0'] for s in data]) for j in js]
        l1 = [np.mean([s['prefix_results'][f'j{j}']['l1'] for s in data]) for j in js]

        ax.plot(prefix_dims, l0, 'o-', color='#2196F3', label='L0 (Coarse)', linewidth=2)
        ax.plot(prefix_dims, l1, 's-', color='#F44336', label='L1 (Fine)', linewidth=2)

        # Shade the gap at j=1
        ax.fill_between([prefix_dims[0]-5, prefix_dims[0]+5],
                        [l1[0], l1[0]], [l0[0], l0[0]],
                        alpha=0.2, color='green')

        steer = np.mean([s['steerability_score'] for s in data])
        ax.set_title(f'{title}\nSteer={steer:+.003f}')
        ax.set_xlabel('Prefix Dimension')
        ax.set_xticks(prefix_dims)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('KNN Accuracy')
            ax.legend(loc='lower right', fontsize=8)

    axes[0].set_ylim(0.87, 1.0)
    plt.suptitle('CLINC150 / bge-small: Ablation Prefix Curves (5 seeds)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_ablation_prefix_curves.png")
    plt.savefig(FIGURES_DIR / "fig4_ablation_prefix_curves.pdf")
    plt.close()
    print("  Fig 4: Ablation prefix curves saved")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_ablation_barplot()
    fig2_prefix_curves()
    fig3_complexity_steerability()
    fig4_ablation_prefix_curves()
    print(f"\nAll figures saved to {FIGURES_DIR}")
