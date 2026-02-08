"""Paper-Ready Figures for Fractal Embeddings.

Generates all 6 figures recommended by Codex for the paper:
  Fig 1: Teaser — V5 vs MRL same accuracy, different steerability (CLINC)
  Fig 3: Cross-dataset steerability forest plot with CIs/effect sizes
  Fig 4: Causal ablation bar/point plot with sign reversal
  Fig 5: Scaling law scatter S vs H(L1|L0) with linear fit
  Fig 6: Synthetic hierarchy causal curve (generated after experiment)

(Fig 2 is architecture diagram — done in LaTeX/tikz)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "paper"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

V5_COLOR = '#2196F3'
MRL_COLOR = '#FF9800'
INV_COLOR = '#F44336'
NP_COLOR = '#9E9E9E'


def compute_steer(prefix_data):
    """Steer = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)."""
    return (prefix_data.get('j1_l0', 0) - prefix_data.get('j4_l0', 0)) + \
           (prefix_data.get('j4_l1', 0) - prefix_data.get('j1_l1', 0))


def compute_steer_from_ablation(pr):
    """Compute steer from ablation-format prefix_results."""
    return (pr['j1']['l0'] - pr['j4']['l0']) + (pr['j4']['l1'] - pr['j1']['l1'])


def load_benchmark_steers(ds_name):
    """Load V5 and MRL steerability from benchmark file."""
    bench_file = RESULTS_DIR / f"benchmark_bge-small_{ds_name}.json"
    if not bench_file.exists():
        return [], []
    d = json.load(open(bench_file))
    v5_steers, mrl_steers = [], []
    for sk, sv in d.get('v5', {}).items():
        if isinstance(sv, dict) and 'prefix_accuracy' in sv:
            v5_steers.append(compute_steer(sv['prefix_accuracy']))
    for sk, sv in d.get('mrl', {}).items():
        if isinstance(sv, dict) and 'prefix_accuracy' in sv:
            mrl_steers.append(compute_steer(sv['prefix_accuracy']))
    return v5_steers, mrl_steers


def load_clinc_steers():
    """Load CLINC V5 steerability from ablation file (5 seeds)."""
    abl_file = RESULTS_DIR / "ablation_steerability_bge-small_clinc.json"
    if not abl_file.exists():
        return [], []
    d = json.load(open(abl_file))
    v5_steers = [compute_steer_from_ablation(r['prefix_results'])
                 for r in d['results']['v5']]
    # MRL from individual file (single seed)
    mrl_steers = []
    mrl_file = RESULTS_DIR / "mrl_baseline_bge-small_clinc.json"
    if mrl_file.exists():
        md = json.load(open(mrl_file))
        if 'prefix_accuracy' in md:
            mrl_steers.append(compute_steer(md['prefix_accuracy']))
    return v5_steers, mrl_steers


def load_clinc_mrl_prefix():
    """Load CLINC MRL prefix accuracy from individual file."""
    mrl_file = RESULTS_DIR / "mrl_baseline_bge-small_clinc.json"
    if not mrl_file.exists():
        return None
    return json.load(open(mrl_file)).get('prefix_accuracy', None)


# ============================================================================
# Figure 1: Teaser — CLINC V5 vs MRL prefix accuracy curves
# ============================================================================
def fig1_teaser():
    """V5 vs MRL on CLINC: same j=4 accuracy, very different prefix behavior."""
    abl = json.load(open(RESULTS_DIR / "ablation_steerability_bge-small_clinc.json"))

    # Average V5 prefix curves across 5 seeds
    v5_l0 = np.mean([[r['prefix_results'][f'j{j}']['l0'] for j in range(1, 5)]
                      for r in abl['results']['v5']], axis=0)
    v5_l1 = np.mean([[r['prefix_results'][f'j{j}']['l1'] for j in range(1, 5)]
                      for r in abl['results']['v5']], axis=0)

    # MRL from individual file (single seed)
    mrl_pa = load_clinc_mrl_prefix()
    if mrl_pa is None:
        print("  Fig 1: No CLINC MRL prefix data. Skipping.")
        return
    mrl_l0 = np.array([mrl_pa[f'j{j}_l0'] for j in range(1, 5)])
    mrl_l1 = np.array([mrl_pa[f'j{j}_l1'] for j in range(1, 5)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    js = [1, 2, 3, 4]
    dims = [64, 128, 192, 256]

    # Left: L0 (Coarse) accuracy
    ax1.plot(dims, v5_l0 * 100, 'o-', color=V5_COLOR, linewidth=2, markersize=8, label='V5')
    ax1.plot(dims, mrl_l0 * 100, 's--', color=MRL_COLOR, linewidth=2, markersize=8, label='MRL')
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('L0 (Domain) Accuracy (%)')
    ax1.set_title('Coarse Classification')
    ax1.legend()
    ax1.set_ylim([88, 100])
    ax1.set_xticks(dims)
    ax1.grid(alpha=0.3)

    # Annotation showing V5 is better at short prefixes
    ax1.annotate(f'V5: {v5_l0[0]*100:.1f}%\n(64d only!)',
                xy=(64, v5_l0[0]*100), xytext=(100, v5_l0[0]*100-3),
                fontsize=9, color=V5_COLOR,
                arrowprops=dict(arrowstyle='->', color=V5_COLOR))

    # Right: L1 (Fine) accuracy
    ax2.plot(dims, v5_l1 * 100, 'o-', color=V5_COLOR, linewidth=2, markersize=8, label='V5')
    ax2.plot(dims, mrl_l1 * 100, 's--', color=MRL_COLOR, linewidth=2, markersize=8, label='MRL')
    ax2.set_xlabel('Embedding Dimension')
    ax2.set_ylabel('L1 (Intent) Accuracy (%)')
    ax2.set_title('Fine Classification')
    ax2.legend()
    ax2.set_ylim([85, 100])
    ax2.set_xticks(dims)
    ax2.grid(alpha=0.3)

    # Annotation: at j=4, roughly same
    ax2.annotate(f'Same at 256d\n(~94.5%)',
                xy=(256, (v5_l1[3]+mrl_l1[3])/2*100),
                xytext=(180, 87),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'))

    fig.suptitle('CLINC: V5 vs MRL — Same Full Accuracy, Different Prefix Behavior',
                fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_teaser.png")
    fig.savefig(FIGURES_DIR / "fig1_teaser.pdf")
    plt.close(fig)
    print(f"  Fig 1 saved: {FIGURES_DIR / 'fig1_teaser.png'}")


# ============================================================================
# Figure 3: Cross-dataset steerability forest plot
# ============================================================================
def fig3_forest_plot():
    """Forest plot of V5 steerability across all datasets with CIs."""
    profiles = json.load(open(RESULTS_DIR / "hierarchy_profiles.json"))

    datasets = []
    # Load all available datasets
    for ds_name in ['yahoo', 'newsgroups', 'trec', 'clinc']:
        if ds_name == 'clinc':
            v5_s, mrl_s = load_clinc_steers()
        else:
            v5_s, mrl_s = load_benchmark_steers(ds_name)
        if v5_s:
            h = profiles.get(ds_name, {}).get('h_l1_given_l0', 0)
            datasets.append({
                'name': ds_name.upper(),
                'h': h,
                'v5_mean': np.mean(v5_s),
                'v5_std': np.std(v5_s) if len(v5_s) > 1 else 0,
                'v5_n': len(v5_s),
                'mrl_mean': np.mean(mrl_s) if mrl_s else 0,
                'mrl_std': np.std(mrl_s) if len(mrl_s) > 1 else 0,
                'mrl_n': len(mrl_s),
            })

    # Sort by H
    datasets.sort(key=lambda x: x['h'])

    fig, ax = plt.subplots(figsize=(8, 5))
    y_positions = np.arange(len(datasets))
    bar_height = 0.35

    for i, ds in enumerate(datasets):
        # V5
        v5_ci = 1.96 * ds['v5_std'] / np.sqrt(ds['v5_n']) if ds['v5_n'] > 1 else ds['v5_std']
        ax.barh(i + bar_height/2, ds['v5_mean'], bar_height,
               xerr=v5_ci, color=V5_COLOR, alpha=0.8, label='V5' if i == 0 else None,
               capsize=3)
        # MRL
        mrl_ci = 1.96 * ds['mrl_std'] / np.sqrt(ds['mrl_n']) if ds['mrl_n'] > 1 else ds['mrl_std']
        ax.barh(i - bar_height/2, ds['mrl_mean'], bar_height,
               xerr=mrl_ci, color=MRL_COLOR, alpha=0.8, label='MRL' if i == 0 else None,
               capsize=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{ds['name']}\nH={ds['h']:.2f}" for ds in datasets])
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xlabel('Steerability Score')
    ax.set_title('Cross-Dataset Steerability: V5 vs MRL\n(higher = more prefix specialization)',
                fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_forest_plot.png")
    fig.savefig(FIGURES_DIR / "fig3_forest_plot.pdf")
    plt.close(fig)
    print(f"  Fig 3 saved: {FIGURES_DIR / 'fig3_forest_plot.png'}")


# ============================================================================
# Figure 4: Causal ablation bar plot with sign reversal
# ============================================================================
def fig4_ablation():
    """Bar plot showing V5/inverted/no-prefix steerability on CLINC (5 seeds)."""
    abl = json.load(open(RESULTS_DIR / "ablation_steerability_bge-small_clinc.json"))

    conditions = ['v5', 'inverted', 'no_prefix']
    labels = ['V5\n(Aligned)', 'Inverted\n(Reversed)', 'No-Prefix\n(Control)']
    colors = [V5_COLOR, INV_COLOR, NP_COLOR]

    means = []
    stds = []
    all_points = []
    for cond in conditions:
        steers = [r['steerability_score'] for r in abl['results'][cond]]
        means.append(np.mean(steers))
        stds.append(np.std(steers))
        all_points.append(steers)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conditions))

    bars = ax.bar(x, means, 0.6, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)

    # Error bars
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black',
               capsize=5, linewidth=1.5)

    # Individual seed points
    for i, points in enumerate(all_points):
        ax.scatter([x[i]] * len(points), points, color='black',
                  s=30, zorder=5, alpha=0.6)

    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Steerability Score')
    ax.set_title('Causal Ablation: Hierarchy Alignment Determines Steerability\n(CLINC, bge-small, 5 seeds)',
                fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Annotations — positioned to avoid overlap
    ax.annotate(f'+{means[0]:.3f}', xy=(0, means[0]),
               xytext=(0.35, means[0]-0.008), fontsize=10, color=V5_COLOR,
               ha='center', fontweight='bold')
    ax.annotate(f'{means[1]:.3f}', xy=(1, means[1]),
               xytext=(1.35, means[1]+0.005), fontsize=10, color=INV_COLOR,
               ha='center', fontweight='bold')
    ax.annotate(f'+{means[2]:.3f}', xy=(2, means[2]),
               xytext=(2.35, means[2]-0.008), fontsize=10, color='gray',
               ha='center', fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_ablation.png")
    fig.savefig(FIGURES_DIR / "fig4_ablation.pdf")
    plt.close(fig)
    print(f"  Fig 4 saved: {FIGURES_DIR / 'fig4_ablation.png'}")


# ============================================================================
# Figure 5: Scaling law scatter S vs H(L1|L0)
# ============================================================================
def fig5_scaling_law():
    """Scatter plot of steerability vs H(L1|L0) with linear fit."""
    profiles = json.load(open(RESULTS_DIR / "hierarchy_profiles.json"))

    datasets_data = []
    for ds_name in ['yahoo', 'newsgroups', 'trec', 'clinc']:
        if ds_name == 'clinc':
            v5_s, _ = load_clinc_steers()
        else:
            v5_s, _ = load_benchmark_steers(ds_name)
        if v5_s:
            h = profiles[ds_name]['h_l1_given_l0']
            datasets_data.append({
                'name': ds_name.upper(),
                'h': h,
                'steer_mean': np.mean(v5_s),
                'steer_std': np.std(v5_s) if len(v5_s) > 1 else 0,
                'n': len(v5_s),
            })

    h_vals = [d['h'] for d in datasets_data]
    s_vals = [d['steer_mean'] for d in datasets_data]
    s_errs = [d['steer_std'] for d in datasets_data]

    # Linear fit
    slope, intercept, r, p, se = stats.linregress(h_vals, s_vals)
    h_fit = np.linspace(0.5, 4.5, 100)
    s_fit = slope * h_fit + intercept

    # Spearman
    rho, p_spearman = stats.spearmanr(h_vals, s_vals)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Confidence band (approximate)
    n_pts = len(h_vals)
    h_mean = np.mean(h_vals)
    s_residuals = np.array(s_vals) - (slope * np.array(h_vals) + intercept)
    mse = np.sum(s_residuals**2) / (n_pts - 2)
    for h_pt in h_fit:
        se_pred = np.sqrt(mse * (1/n_pts + (h_pt - h_mean)**2 / np.sum((np.array(h_vals) - h_mean)**2)))

    # Fit line
    ax.plot(h_fit, s_fit, '-', color='gray', linewidth=1.5, alpha=0.7,
           label=f'Linear fit (R²={r**2:.3f})')

    # Data points with error bars
    for d in datasets_data:
        ci = 1.96 * d['steer_std'] / np.sqrt(d['n']) if d['n'] > 1 else d['steer_std']
        ax.errorbar(d['h'], d['steer_mean'], yerr=ci,
                   fmt='o', markersize=10, color=V5_COLOR,
                   capsize=5, linewidth=1.5, markeredgecolor='black',
                   markeredgewidth=0.5)
        ax.annotate(d['name'], (d['h'], d['steer_mean']),
                   textcoords="offset points", xytext=(8, 8),
                   fontsize=10, fontweight='bold')

    ax.set_xlabel('H(L1|L0) — Hierarchy Refinement Entropy (bits)')
    ax.set_ylabel('V5 Steerability Score')
    ax.set_title(f'Steerability Scales with Hierarchy Depth\n'
                f'Spearman ρ={rho:.2f} (p={p_spearman:.3f}), '
                f'slope={slope:.4f}±{se:.4f}',
                fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_scaling_law.png")
    fig.savefig(FIGURES_DIR / "fig5_scaling_law.pdf")
    plt.close(fig)
    print(f"  Fig 5 saved: {FIGURES_DIR / 'fig5_scaling_law.png'}")


# ============================================================================
# Figure 6: Synthetic hierarchy causal curve
# ============================================================================
def fig6_synthetic():
    """Synthetic hierarchy experiment results — if available."""
    synth_file = RESULTS_DIR / "synthetic_hierarchy_results.json"
    if not synth_file.exists():
        print("  Fig 6: Synthetic results not yet available. Skipping.")
        return

    data = json.load(open(synth_file))
    results = data.get('results', [])
    if not results:
        print("  Fig 6: No results in synthetic file. Skipping.")
        return

    h_vals = [r['h_l1_given_l0'] for r in results]
    v5_steers = [r['v5_steer'] for r in results]
    mrl_steers = [r.get('mrl_steer', 0) for r in results]
    k0_vals = [r['k0'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(h_vals, v5_steers, 'o-', color=V5_COLOR, linewidth=2,
           markersize=8, label='V5', markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(h_vals, mrl_steers, 's--', color=MRL_COLOR, linewidth=2,
           markersize=8, label='MRL', markeredgecolor='black', markeredgewidth=0.5)

    # Label K0 values
    for h, s, k in zip(h_vals, v5_steers, k0_vals):
        ax.annotate(f'K₀={k}', (h, s),
                   textcoords="offset points", xytext=(5, 8),
                   fontsize=8, color=V5_COLOR)

    # Linear fit to V5
    if len(h_vals) >= 3:
        slope, intercept, r, p, se = stats.linregress(h_vals, v5_steers)
        h_fit = np.linspace(min(h_vals) - 0.2, max(h_vals) + 0.2, 100)
        ax.plot(h_fit, slope * h_fit + intercept, '-', color=V5_COLOR,
               alpha=0.3, linewidth=3, label=f'V5 fit (R²={r**2:.3f})')

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xlabel('H(L1|L0) — Manipulated Hierarchy Entropy (bits)')
    ax.set_ylabel('Steerability Score')
    ax.set_title('Synthetic Hierarchy Experiment: Causal Intervention\n'
                '(Same CLINC text, different coarse groupings)',
                fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_synthetic.png")
    fig.savefig(FIGURES_DIR / "fig6_synthetic.pdf")
    plt.close(fig)
    print(f"  Fig 6 saved: {FIGURES_DIR / 'fig6_synthetic.png'}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"Output: {FIGURES_DIR}\n")

    fig1_teaser()
    fig3_forest_plot()
    fig4_ablation()
    fig5_scaling_law()
    fig6_synthetic()

    print("\nDone!")
