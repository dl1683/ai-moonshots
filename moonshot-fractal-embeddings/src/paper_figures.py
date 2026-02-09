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
UHMT_COLOR = '#9C27B0'  # Purple for UHMT


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
    """Load CLINC V5/MRL steerability. Prefer benchmark file (standard protocol)."""
    # Prefer benchmark file (standard protocol, consistent with other datasets)
    bench_file = RESULTS_DIR / "benchmark_bge-small_clinc.json"
    if bench_file.exists():
        return load_benchmark_steers('clinc')
    # Fallback: ablation file (different protocol — contrastive+margin+CE)
    abl_file = RESULTS_DIR / "ablation_steerability_bge-small_clinc.json"
    if not abl_file.exists():
        return [], []
    d = json.load(open(abl_file))
    v5_steers = [compute_steer_from_ablation(r['prefix_results'])
                 for r in d['results']['v5']]
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
    # Prefer benchmark file (standard protocol)
    bench_file = RESULTS_DIR / "benchmark_bge-small_clinc.json"
    if bench_file.exists():
        d = json.load(open(bench_file))
        # Average V5 prefix curves across seeds
        v5_curves_l0, v5_curves_l1 = [], []
        for sv in d['v5'].values():
            if isinstance(sv, dict) and 'prefix_accuracy' in sv:
                pa = sv['prefix_accuracy']
                v5_curves_l0.append([pa[f'j{j}_l0'] for j in range(1, 5)])
                v5_curves_l1.append([pa[f'j{j}_l1'] for j in range(1, 5)])
        v5_l0 = np.mean(v5_curves_l0, axis=0)
        v5_l1 = np.mean(v5_curves_l1, axis=0)
        # Average MRL prefix curves across seeds
        mrl_curves_l0, mrl_curves_l1 = [], []
        for sv in d['mrl'].values():
            if isinstance(sv, dict) and 'prefix_accuracy' in sv:
                pa = sv['prefix_accuracy']
                mrl_curves_l0.append([pa[f'j{j}_l0'] for j in range(1, 5)])
                mrl_curves_l1.append([pa[f'j{j}_l1'] for j in range(1, 5)])
        mrl_l0 = np.mean(mrl_curves_l0, axis=0)
        mrl_l1 = np.mean(mrl_curves_l1, axis=0)
    else:
        # Fallback: ablation file
        abl = json.load(open(RESULTS_DIR / "ablation_steerability_bge-small_clinc.json"))
        v5_l0 = np.mean([[r['prefix_results'][f'j{j}']['l0'] for j in range(1, 5)]
                          for r in abl['results']['v5']], axis=0)
        v5_l1 = np.mean([[r['prefix_results'][f'j{j}']['l1'] for j in range(1, 5)]
                          for r in abl['results']['v5']], axis=0)
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
    ALL_DS = ['yahoo', 'goemotions', 'newsgroups', 'trec', 'arxiv', 'clinc', 'dbpedia_classes', 'wos']
    DS_DISPLAY = {'goemotions': 'GoEmo', 'dbpedia_classes': 'DBPedia'}
    DS_H_FALLBACK = {'dbpedia_classes': 3.17, 'wos': 5.05}
    for ds_name in ALL_DS:
        if ds_name == 'clinc':
            v5_s, mrl_s = load_clinc_steers()
        else:
            v5_s, mrl_s = load_benchmark_steers(ds_name)
        if v5_s:
            h = profiles.get(ds_name, {}).get('h_l1_given_l0', DS_H_FALLBACK.get(ds_name, 0))
            display_name = DS_DISPLAY.get(ds_name, ds_name.upper())
            datasets.append({
                'name': display_name,
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
    """Grouped bar plot showing V5/inverted/no-prefix/UHMT steerability on CLINC + TREC."""
    datasets_info = [
        ('clinc', 'CLINC (H=3.90, 5s)'),
        ('trec', 'TREC (H=2.21, 3s)'),
    ]
    conditions = ['v5', 'inverted', 'no_prefix', 'uhmt']
    cond_labels = ['Aligned (V5)', 'Inverted', 'No-Prefix', 'UHMT']
    colors = [V5_COLOR, INV_COLOR, NP_COLOR, UHMT_COLOR]

    n_datasets = len(datasets_info)
    n_conds = len(conditions)
    bar_width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))

    for di, (ds_name, ds_label) in enumerate(datasets_info):
        abl_path = RESULTS_DIR / f"ablation_steerability_bge-small_{ds_name}.json"
        uhmt_path = RESULTS_DIR / f"uhmt_ablation_bge-small_{ds_name}.json"

        abl = None
        if abl_path.exists():
            abl = json.load(open(abl_path))

        uhmt_data = None
        if uhmt_path.exists():
            uhmt_data = json.load(open(uhmt_path))

        for ci, cond in enumerate(conditions):
            steers = None
            if cond == 'uhmt' and uhmt_data is not None:
                steers = [r['steerability_score'] for r in uhmt_data['results']]
            elif cond != 'uhmt' and abl is not None and cond in abl.get('results', {}):
                steers = [r['steerability_score'] for r in abl['results'][cond]]

            if steers is None:
                continue

            m = np.mean(steers)
            s = np.std(steers, ddof=1) if len(steers) > 1 else 0
            xpos = di + (ci - 1.5) * bar_width

            ax.bar(xpos, m, bar_width * 0.9, color=colors[ci], alpha=0.8,
                   edgecolor='black', linewidth=0.5,
                   label=cond_labels[ci] if di == 0 else None)
            ax.errorbar(xpos, m, yerr=s, fmt='none', color='black',
                       capsize=3, linewidth=1.2)
            # Individual seeds
            for pt in steers:
                ax.scatter(xpos, pt, color='black', s=20, zorder=5, alpha=0.5)

    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.set_xticks(range(n_datasets))
    ax.set_xticklabels([d[1] for d in datasets_info])
    ax.set_ylabel('Steerability Score')
    ax.set_title('Causal Ablation: Alignment vs. Awareness vs. Inversion',
                fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_ablation.png")
    fig.savefig(FIGURES_DIR / "fig4_ablation.pdf")
    plt.close(fig)
    print(f"  Fig 4 saved: {FIGURES_DIR / 'fig4_ablation.png'}")


# ============================================================================
# Figure 5: Scaling law scatter S vs H(L1|L0)
# ============================================================================
def fig5_scaling_law():
    """Two-panel: (Left) S vs H(L1|L0), (Right) S vs H*L1_acc product predictor."""
    profiles = json.load(open(RESULTS_DIR / "hierarchy_profiles.json"))

    datasets_data = []
    ALL_DS = ['yahoo', 'goemotions', 'newsgroups', 'trec', 'arxiv', 'clinc', 'dbpedia_classes', 'wos']
    DS_DISPLAY = {'goemotions': 'GoEmo', 'dbpedia_classes': 'DBPedia'}
    DS_H_FALLBACK = {'dbpedia_classes': 3.17, 'wos': 5.05}
    for ds_name in ALL_DS:
        if ds_name == 'clinc':
            v5_s, mrl_s = load_clinc_steers()
        else:
            v5_s, mrl_s = load_benchmark_steers(ds_name)
        if v5_s:
            h = profiles.get(ds_name, {}).get('h_l1_given_l0', DS_H_FALLBACK.get(ds_name, 0))
            display_name = DS_DISPLAY.get(ds_name, ds_name.upper())
            # Get baseline (unfinetuned) L1 accuracy for learnability
            bench_file = RESULTS_DIR / f"benchmark_bge-small_{ds_name}.json"
            best_l1 = 0
            if bench_file.exists():
                bd = json.load(open(bench_file))
                # Use unfinetuned baseline L1 accuracy (same across seeds)
                first_seed = list(bd.get('v5', {}).keys())[0] if bd.get('v5') else None
                if first_seed:
                    best_l1 = bd['v5'][first_seed].get('baseline', {}).get('l1_accuracy', 0)
            datasets_data.append({
                'name': display_name,
                'ds_name': ds_name,
                'h': h,
                'steer_mean': np.mean(v5_s),
                'steer_std': np.std(v5_s) if len(v5_s) > 1 else 0,
                'n': len(v5_s),
                'l1_acc': best_l1,
            })

    h_vals = np.array([d['h'] for d in datasets_data])
    s_vals = np.array([d['steer_mean'] for d in datasets_data])
    l1_vals = np.array([d['l1_acc'] for d in datasets_data])
    product = h_vals * l1_vals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: S vs H(L1|L0) ---
    rho_h, p_h = stats.spearmanr(h_vals, s_vals)
    slope_h, intercept_h, r_h, p_lr, se_h = stats.linregress(h_vals, s_vals)
    h_fit = np.linspace(0.5, max(h_vals) + 0.5, 100)
    ax1.plot(h_fit, slope_h * h_fit + intercept_h, '-', color='gray',
             linewidth=1.5, alpha=0.7, label=f'Linear fit (R²={r_h**2:.3f})')

    label_offsets = {'GoEmo': (8, -16), 'NEWSGROUPS': (8, 8), 'ARXIV': (8, -14), 'WOS': (8, -14)}
    for d in datasets_data:
        ci = 1.96 * d['steer_std'] / np.sqrt(d['n']) if d['n'] > 1 else d['steer_std']
        marker = 'D' if d['ds_name'] == 'wos' else 'o'
        ax1.errorbar(d['h'], d['steer_mean'], yerr=ci,
                     fmt=marker, markersize=10, color=V5_COLOR,
                     capsize=5, linewidth=1.5, markeredgecolor='black',
                     markeredgewidth=0.5)
        offset = label_offsets.get(d['name'], (8, 8))
        ax1.annotate(d['name'], (d['h'], d['steer_mean']),
                     textcoords="offset points", xytext=offset,
                     fontsize=10, fontweight='bold')

    ax1.set_xlabel('H(L1|L0) -- Hierarchy Refinement Entropy (bits)')
    ax1.set_ylabel('V5 Steerability Score')
    ax1.set_title(f'Raw Scaling: rho={rho_h:.2f} (p={p_h:.3f})\nWOS deviates due to L1 floor effect',
                  fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

    # --- Right: S vs H * L1_acc (product predictor) ---
    rho_p, p_p = stats.spearmanr(product, s_vals)
    r_p, pr_p = stats.pearsonr(product, s_vals)
    slope_p, intercept_p, _, _, _ = stats.linregress(product, s_vals)
    p_fit = np.linspace(0, max(product) + 0.3, 100)
    ax2.plot(p_fit, slope_p * p_fit + intercept_p, '-', color='gray',
             linewidth=1.5, alpha=0.7, label=f'Linear fit (R²={r_p**2:.3f})')

    for d, prod in zip(datasets_data, product):
        ci = 1.96 * d['steer_std'] / np.sqrt(d['n']) if d['n'] > 1 else d['steer_std']
        marker = 'D' if d['ds_name'] == 'wos' else 'o'
        ax2.errorbar(prod, d['steer_mean'], yerr=ci,
                     fmt=marker, markersize=10, color=V5_COLOR,
                     capsize=5, linewidth=1.5, markeredgecolor='black',
                     markeredgewidth=0.5)
        offset = label_offsets.get(d['name'], (8, 8))
        ax2.annotate(d['name'], (prod, d['steer_mean']),
                     textcoords="offset points", xytext=offset,
                     fontsize=10, fontweight='bold')

    ax2.set_xlabel('H(L1|L0) x Baseline L1 Accuracy')
    ax2.set_ylabel('V5 Steerability Score')
    ax2.set_title(f'Product Predictor: rho={rho_p:.2f} (p={p_p:.4f})\nAccounts for WOS floor effect',
                  fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

    fig.suptitle('Steerability Scaling: Hierarchy Depth x Model Learnability',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_scaling_law.png")
    fig.savefig(FIGURES_DIR / "fig5_scaling_law.pdf")
    plt.close(fig)
    print(f"  Fig 5 saved: {FIGURES_DIR / 'fig5_scaling_law.png'}")


# ============================================================================
# Figure 6: Synthetic hierarchy causal curve
# ============================================================================
def fig6_synthetic():
    """Synthetic hierarchy experiment: S vs H(L0) showing Goldilocks effect."""
    synth_file = RESULTS_DIR / "synthetic_hierarchy_experiment.json"
    if not synth_file.exists():
        print("  Fig 6: Synthetic results not yet available. Skipping.")
        return

    data = json.load(open(synth_file))
    results = [r for r in data.get('results', []) if 'v5_steerability' in r]
    if not results:
        print("  Fig 6: No results in synthetic file. Skipping.")
        return

    # Sort by K0
    results.sort(key=lambda r: r['k0'])
    h_l0 = [np.log2(r['k0']) for r in results]
    v5_steers = [r['v5_steerability'] for r in results]
    mrl_steers = [r.get('mrl_steerability', 0) for r in results]
    k0_vals = [r['k0'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(h_l0, v5_steers, 'o-', color=V5_COLOR, linewidth=2,
           markersize=9, label='V5', markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(h_l0, mrl_steers, 's--', color=MRL_COLOR, linewidth=2,
           markersize=8, label='MRL', markeredgecolor='black', markeredgewidth=0.5)

    # Label K0 values
    for h, s, k in zip(h_l0, v5_steers, k0_vals):
        offset_y = 8 if k != 15 else -15  # peak label below
        ax.annotate(f'K₀={k}', (h, s),
                   textcoords="offset points", xytext=(5, offset_y),
                   fontsize=8, color=V5_COLOR)

    # Quadratic fit to V5 (captures inverted-U)
    if len(h_l0) >= 4:
        coeffs = np.polyfit(h_l0, v5_steers, 2)
        h_fit = np.linspace(min(h_l0) - 0.1, max(h_l0) + 0.1, 100)
        s_fit = np.polyval(coeffs, h_fit)
        r2 = 1 - np.sum((np.array(v5_steers) - np.polyval(coeffs, np.array(h_l0)))**2) / \
                  np.sum((np.array(v5_steers) - np.mean(v5_steers))**2)
        ax.plot(h_fit, s_fit, '-', color=V5_COLOR,
               alpha=0.3, linewidth=3, label=f'Quadratic fit (R²={r2:.3f})')

        # Mark the peak
        peak_h = -coeffs[1] / (2 * coeffs[0])
        peak_s = np.polyval(coeffs, peak_h)
        ax.axvline(x=peak_h, color='gray', linewidth=1, linestyle=':', alpha=0.5)
        ax.annotate(f'Peak: H(L0)={peak_h:.1f} bits\n(~{2**peak_h:.0f} classes)',
                   xy=(peak_h, peak_s), xytext=(peak_h + 0.8, peak_s - 0.04),
                   fontsize=9, color='gray', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='gray'))

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xlabel('H(L0) — Coarse Task Entropy (bits)')
    ax.set_ylabel('Steerability Score')
    ax.set_title('Synthetic Hierarchy: "Goldilocks" Effect\n'
                '(Same CLINC text, varied K₀ coarse groupings)',
                fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_synthetic.png")
    fig.savefig(FIGURES_DIR / "fig6_synthetic.pdf")
    plt.close(fig)
    print(f"  Fig 6 saved: {FIGURES_DIR / 'fig6_synthetic.png'}")


# ============================================================================
# Figure 7: Entropy allocation — S vs H(L0) across real + synthetic
# ============================================================================
def fig7_entropy_allocation():
    """Codex-recommended: S vs H(L0) with real and synthetic datasets.

    Shows the true mechanistic driver: prefix task demand H(L0),
    not H(L1|L0) which was a confounded proxy in observational data.
    """
    profiles = json.load(open(RESULTS_DIR / "hierarchy_profiles.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: S vs H(L0) ---
    # Real datasets
    real_data = []
    for ds_name in ['yahoo', 'goemotions', 'newsgroups', 'trec', 'arxiv', 'clinc']:
        if ds_name == 'clinc':
            v5_s, _ = load_clinc_steers()
        else:
            v5_s, _ = load_benchmark_steers(ds_name)
        if v5_s and 'h_l0' in profiles.get(ds_name, {}):
            display_name = 'GoEmo' if ds_name == 'goemotions' else ds_name.upper()
            real_data.append({
                'name': display_name,
                'h_l0': profiles[ds_name]['h_l0'],
                'h_l1_l0': profiles[ds_name]['h_l1_given_l0'],
                'steer': np.mean(v5_s),
                'std': np.std(v5_s) if len(v5_s) > 1 else 0,
                'n': len(v5_s),
            })

    # Synthetic datasets
    synth_data = []
    synth_file = RESULTS_DIR / "synthetic_hierarchy_experiment.json"
    if synth_file.exists():
        sd = json.load(open(synth_file))
        for r in sd.get('results', []):
            if 'v5_steerability' in r:
                hs = r['hierarchy_stats']
                synth_data.append({
                    'k0': r['k0'],
                    'h_l0': np.log2(r['k0']),  # H(L0) = log2(K0) for uniform
                    'h_l1_l0': hs['h_l1_given_l0'],
                    'steer': r['v5_steerability'],
                    'mrl': r.get('mrl_steerability', 0),
                })

    # Plot real
    for d in real_data:
        ci = 1.96 * d['std'] / np.sqrt(d['n']) if d['n'] > 1 else d['std']
        ax1.errorbar(d['h_l0'], d['steer'], yerr=ci,
                    fmt='o', markersize=10, color=V5_COLOR,
                    capsize=5, markeredgecolor='black', markeredgewidth=0.5)
        ax1.annotate(d['name'], (d['h_l0'], d['steer']),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, fontweight='bold', color=V5_COLOR)

    # Plot synthetic
    if synth_data:
        sh = [d['h_l0'] for d in synth_data]
        ss = [d['steer'] for d in synth_data]
        ax1.plot(sh, ss, 'D-', color='#4CAF50', markersize=7,
                linewidth=1.5, markeredgecolor='black', markeredgewidth=0.5,
                label='Synthetic (varied K₀)', alpha=0.8)
        for d in synth_data:
            ax1.annotate(f'K₀={d["k0"]}', (d['h_l0'], d['steer']),
                        textcoords="offset points", xytext=(5, -12),
                        fontsize=7, color='#4CAF50')

    ax1.set_xlabel('H(L0) — Coarse Task Entropy (bits)')
    ax1.set_ylabel('V5 Steerability Score')
    ax1.set_title('Mechanism: S ~ H(L0)\n(prefix task demand drives steerability)',
                  fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)

    # --- Right panel: S vs H(L1|L0) — shows confound ---
    for d in real_data:
        ci = 1.96 * d['std'] / np.sqrt(d['n']) if d['n'] > 1 else d['std']
        ax2.errorbar(d['h_l1_l0'], d['steer'], yerr=ci,
                    fmt='o', markersize=10, color=V5_COLOR,
                    capsize=5, markeredgecolor='black', markeredgewidth=0.5)
        ax2.annotate(d['name'], (d['h_l1_l0'], d['steer']),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, fontweight='bold', color=V5_COLOR)

    if synth_data:
        sh = [d['h_l1_l0'] for d in synth_data]
        ss = [d['steer'] for d in synth_data]
        ax2.plot(sh, ss, 'D-', color='#4CAF50', markersize=7,
                linewidth=1.5, markeredgecolor='black', markeredgewidth=0.5,
                label='Synthetic (varied K₀)', alpha=0.8)
        for d in synth_data:
            ax2.annotate(f'K₀={d["k0"]}', (d['h_l1_l0'], d['steer']),
                        textcoords="offset points", xytext=(5, -12),
                        fontsize=7, color='#4CAF50')

    ax2.set_xlabel('H(L1|L0) — Refinement Entropy (bits)')
    ax2.set_ylabel('V5 Steerability Score')
    ax2.set_title('Observational Confound: S vs H(L1|L0)\n(positive for real, NEGATIVE for synthetic)',
                  fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)

    fig.suptitle('Disentangling the Scaling Law: H(L0) is the True Driver',
                fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_entropy_allocation.png")
    fig.savefig(FIGURES_DIR / "fig7_entropy_allocation.pdf")
    plt.close(fig)
    print(f"  Fig 7 saved: {FIGURES_DIR / 'fig7_entropy_allocation.png'}")


# ============================================================================
# Figure 8: Prefix Surgery — Causal prefix swap
# ============================================================================
def fig8_prefix_surgery():
    """Information localization: grouped bar chart showing L0/L1 accuracy
    in prefix vs suffix for V5 and MRL."""
    surgery_files = sorted(Path(RESULTS_DIR).glob("prefix_surgery_*.json"))
    if not surgery_files:
        print("  Fig 8: No prefix surgery results yet. Skipping.")
        return

    data = json.load(open(surgery_files[0]))
    v5 = data.get("v5", {})
    mrl = data.get("mrl", {})

    if not v5 or not mrl or "localization" not in v5:
        print("  Fig 8: Incomplete localization data. Skipping.")
        return

    v5_loc = v5["localization"]
    mrl_loc = mrl["localization"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    metrics = ['prefix_l0', 'prefix_l1', 'suffix_l0', 'suffix_l1']
    labels = ['Prefix L0', 'Prefix L1', 'Suffix L0', 'Suffix L1']
    v5_vals = [v5_loc[m] for m in metrics]
    mrl_vals = [mrl_loc[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, v5_vals, width, label='V5', color=V5_COLOR,
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, mrl_vals, width, label='MRL', color=MRL_COLOR,
                   edgecolor='black', linewidth=0.5)

    ax.set_ylabel('kNN Accuracy', fontsize=12)
    ax.set_title('Information Localization: Where Does Each Level Live?',
                fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.9, 1.0)

    for bar, val in zip(bars1, v5_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
               f'{val:.3f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, mrl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
               f'{val:.3f}', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_prefix_surgery.png", dpi=150)
    fig.savefig(FIGURES_DIR / "fig8_prefix_surgery.pdf")
    plt.close(fig)
    print(f"  Fig 8 saved: {FIGURES_DIR / 'fig8_prefix_surgery.png'}")


def fig9_retrieval_benchmark():
    """Retrieval benchmark: Recall@1 at each prefix length for V5 vs MRL.
    Shows V5's L1 Recall ramps from 64d to 256d while MRL is flat."""
    retrieval_files = sorted(Path(RESULTS_DIR).glob("retrieval_benchmark_*.json"))
    if not retrieval_files:
        print("  Fig 9: No retrieval benchmark results yet. Skipping.")
        return

    # Load all available datasets
    all_data = {}
    for f in retrieval_files:
        data = json.load(open(f))
        ds_name = data["dataset"]
        all_data[ds_name] = data

    n_datasets = len(all_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4.5), squeeze=False)

    dims = [64, 128, 192, 256]
    js = ["1", "2", "3", "4"]

    for col, (ds_name, data) in enumerate(sorted(all_data.items())):
        ax = axes[0, col]
        seeds = data["seeds"]
        results = data["results_by_seed"]

        # Aggregate across seeds
        for level, style in [("L0", "--"), ("L1", "-")]:
            v5_means = []
            mrl_means = []
            v5_stds = []
            mrl_stds = []
            for j in js:
                v5_vals = [results[str(s)]["v5"][j][level]["recall@1"] for s in seeds]
                mrl_vals = [results[str(s)]["mrl"][j][level]["recall@1"] for s in seeds]
                v5_means.append(np.mean(v5_vals))
                mrl_means.append(np.mean(mrl_vals))
                v5_stds.append(np.std(v5_vals))
                mrl_stds.append(np.std(mrl_vals))

            label_suffix = " (coarse)" if level == "L0" else " (fine)"
            ax.errorbar(dims, v5_means, yerr=v5_stds, marker='o', linestyle=style,
                       color=V5_COLOR, label=f'V5 {level}{label_suffix}', capsize=3,
                       linewidth=2 if level == "L1" else 1.2,
                       markersize=7 if level == "L1" else 5)
            ax.errorbar(dims, mrl_means, yerr=mrl_stds, marker='s', linestyle=style,
                       color=MRL_COLOR, label=f'MRL {level}{label_suffix}', capsize=3,
                       linewidth=2 if level == "L1" else 1.2,
                       markersize=7 if level == "L1" else 5)

        ax.set_xlabel('Embedding Dimensions')
        ax.set_ylabel('Recall@1')
        ax.set_title(f'{ds_name.upper()}', fontweight='bold')
        ax.set_xticks(dims)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(0.82, 1.01)
        ax.grid(True, alpha=0.3)

        # Annotate the L1 ramp
        v5_l1_64 = np.mean([results[str(s)]["v5"]["1"]["L1"]["recall@1"] for s in seeds])
        v5_l1_256 = np.mean([results[str(s)]["v5"]["4"]["L1"]["recall@1"] for s in seeds])
        mrl_l1_64 = np.mean([results[str(s)]["mrl"]["1"]["L1"]["recall@1"] for s in seeds])
        mrl_l1_256 = np.mean([results[str(s)]["mrl"]["4"]["L1"]["recall@1"] for s in seeds])
        v5_delta = v5_l1_256 - v5_l1_64
        mrl_delta = mrl_l1_256 - mrl_l1_64
        ax.annotate(f'V5 L1 ramp: +{v5_delta:.1%}',
                   xy=(160, (v5_l1_64 + v5_l1_256)/2), fontsize=9,
                   color=V5_COLOR, fontweight='bold')

    fig.suptitle('Retrieval Benchmark: V5 Enables Coarse-to-Fine Recall via Truncation',
                fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_retrieval.png", dpi=150)
    fig.savefig(FIGURES_DIR / "fig9_retrieval.pdf")
    plt.close(fig)
    print(f"  Fig 9 saved: {FIGURES_DIR / 'fig9_retrieval.png'}")


def fig10_three_level():
    """3-level hierarchy: monotonic semantic zoom across 3 granularity levels."""
    result_path = RESULTS_DIR / "three_level_clinc.json"
    if not result_path.exists():
        print("  Fig 10: No 3-level results yet. Skipping.")
        return

    data = json.load(open(result_path))
    results_list = data["results"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    dims = [64, 128, 192, 256]
    js = [1, 2, 3, 4]
    level_colors = {'l0': '#E53935', 'l1': '#FB8C00', 'l2': '#1E88E5'}
    level_labels = {'l0': 'L0 (super-domain)', 'l1': 'L1 (domain)', 'l2': 'L2 (intent)'}

    for panel, method in enumerate(['v5', 'mrl']):
        ax = axes[panel]
        for level in ['l0', 'l1', 'l2']:
            means = []
            stds = []
            for j in js:
                vals = [r[method]['prefix_results'][str(j)][level] for r in results_list]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(dims, means, yerr=stds, marker='o', capsize=3,
                       color=level_colors[level], label=level_labels[level],
                       linewidth=2, markersize=6)

        title = 'V5 (Hierarchy-Aligned)' if method == 'v5' else 'MRL (Flat Supervision)'
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Embedding Dimensions')
        ax.set_ylabel('kNN Accuracy')
        ax.set_xticks(dims)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('3-Level Hierarchy: Monotonic Semantic Zoom (CLINC 5->10->150)',
                fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig10_three_level.png", dpi=150)
    fig.savefig(FIGURES_DIR / "fig10_three_level.pdf")
    plt.close(fig)
    print(f"  Fig 10 saved: {FIGURES_DIR / 'fig10_three_level.png'}")


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
    fig7_entropy_allocation()
    fig8_prefix_surgery()
    fig9_retrieval_benchmark()
    fig10_three_level()

    print("\nDone!")
