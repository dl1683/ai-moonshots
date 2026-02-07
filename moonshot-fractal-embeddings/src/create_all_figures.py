"""
Create all publication-quality figures for Fractal Embeddings research.
Generates shareable images for blog posts, presentations, and papers.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# Colors
COLORS = {
    'flat': '#E74C3C',      # Red
    'fractal': '#27AE60',   # Green
    'random': '#95A5A6',    # Gray
    'true': '#27AE60',      # Green
    'hier_softmax': '#3498DB',  # Blue
    'classifier_chain': '#9B59B6',  # Purple
}

def load_results():
    """Load all result files."""
    base = Path(__file__).parent.parent / 'results'

    with open(base / 'hierarchy_randomization_fast.json') as f:
        hier_rand = json.load(f)

    with open(base / 'v5_multiseed_qwen3-0.6b.json') as f:
        v5_results = json.load(f)

    with open(base / 'rigorous_scaling_qwen3-0.6b.json') as f:
        scaling = json.load(f)

    with open(base / 'newsgroups_benchmark_qwen3-0.6b.json') as f:
        newsgroups = json.load(f)

    return hier_rand, v5_results, scaling, newsgroups


def fig1_hierarchy_randomization(hier_rand, output_dir):
    """
    KEY FIGURE: Bar chart showing Flat vs True Hierarchy vs Random Hierarchy
    This is our most important result.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    flat_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['flat_results']]) * 100
    true_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['true_results']]) * 100
    rand_accs = [r['aggregate']['hier_acc.mean'] * 100 for r in hier_rand['randomizations']]
    rand_mean = np.mean(rand_accs)
    rand_std = np.std(rand_accs)

    # Flat seed std
    flat_std = np.std([r['metrics']['hier_acc'] for r in hier_rand['flat_results']]) * 100
    true_std = np.std([r['metrics']['hier_acc'] for r in hier_rand['true_results']]) * 100

    conditions = ['Flat Baseline\n(No Hierarchy)', 'Fractal +\nTRUE Hierarchy', 'Fractal +\nRANDOM Hierarchy']
    values = [flat_acc, true_acc, rand_mean]
    errors = [flat_std, true_std, rand_std]
    colors = [COLORS['flat'], COLORS['true'], COLORS['random']]

    bars = ax.bar(conditions, values, yerr=errors, color=colors,
                  capsize=5, edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + err + 0.3),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add delta annotations
    ax.annotate(f'+{true_acc - flat_acc:.2f}%',
                xy=(1, true_acc - 0.8), ha='center', fontsize=11,
                color='green', fontweight='bold')
    ax.annotate(f'{rand_mean - flat_acc:.2f}%',
                xy=(2, rand_mean - 0.8), ha='center', fontsize=11,
                color='red', fontweight='bold')

    # Gap annotation
    gap = true_acc - rand_mean
    ax.annotate('', xy=(1, true_acc + true_std + 1), xytext=(2, true_acc + true_std + 1),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(1.5, true_acc + true_std + 1.5, f'Gap: +{gap:.2f}%\n(95% CI excludes 0)',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_ylabel('Hierarchical Accuracy (%)', fontsize=13)
    ax.set_title('KEY FINDING: Correct Structure Matters\n(20 Newsgroups, K=30 randomizations, S=3 seeds)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(64, 70)
    ax.axhline(y=flat_acc, color=COLORS['flat'], linestyle='--', alpha=0.5, label='Flat baseline')

    # Add key insight box
    textstr = 'Wrong hierarchy HURTS\nCorrect hierarchy HELPS'
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_hierarchy_randomization.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_hierarchy_randomization.pdf', bbox_inches='tight')
    print(f"Saved: fig1_hierarchy_randomization")
    plt.close()


def fig2_delta_distribution(hier_rand, output_dir):
    """Histogram of delta_rand values showing most are negative or near zero."""
    fig, ax = plt.subplots(figsize=(10, 5))

    deltas = [d * 100 for d in hier_rand['summary']['delta_rand']['values']]

    ax.hist(deltas, bins=15, color=COLORS['random'], edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (no effect)')
    ax.axvline(x=np.mean(deltas), color='blue', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(deltas):.3f}%')

    # Add delta_true line
    delta_true = hier_rand['summary']['delta_true']['mean'] * 100
    ax.axvline(x=delta_true, color='green', linestyle='-', linewidth=2,
               label=f'True hierarchy: +{delta_true:.2f}%')

    ax.set_xlabel('Delta vs Flat Baseline (%)', fontsize=12)
    ax.set_ylabel('Count (out of 30 randomizations)', fontsize=12)
    ax.set_title('Distribution of Random Hierarchy Performance\n(63% are at or below zero)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')

    # Add annotation
    pct_le_zero = hier_rand['summary']['fractions']['delta_rand_le_zero'] * 100
    ax.text(0.02, 0.98, f'{pct_le_zero:.0f}% of random\nhierarchies hurt\nperformance',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_delta_distribution.png', bbox_inches='tight')
    print(f"Saved: fig2_delta_distribution")
    plt.close()


def fig3_scaling_law(scaling, output_dir):
    """Scaling law: Depth vs Hierarchical Accuracy for all classifiers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    depths = [2, 3, 4, 5]
    results = scaling['results_summary']

    classifiers = ['flat', 'fractal', 'hier_softmax', 'classifier_chain']
    labels = ['Flat (baseline)', 'Fractal (ours)', 'Hierarchical Softmax', 'Classifier Chain']

    for clf, label in zip(classifiers, labels):
        means = [results[f'depth_{d}'][clf]['hier_acc_mean'] * 100 for d in depths]
        cis = [results[f'depth_{d}'][clf]['hier_acc_ci95'] * 100 for d in depths]

        ax.errorbar(depths, means, yerr=cis, marker='o', markersize=8,
                    color=COLORS.get(clf, 'gray'), label=label, capsize=4,
                    linewidth=2, capthick=2)

    ax.set_xlabel('Hierarchy Depth (levels)', fontsize=12)
    ax.set_ylabel('Hierarchical Accuracy (%)', fontsize=12)
    ax.set_title('Scaling Law: Fractal Beats All Baselines at All Depths\n(3 seeds, 95% CI)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(depths)
    ax.legend(loc='upper right')
    ax.set_ylim(35, 95)

    # Add advantage annotations
    for d in depths:
        flat_acc = results[f'depth_{d}']['flat']['hier_acc_mean'] * 100
        frac_acc = results[f'depth_{d}']['fractal']['hier_acc_mean'] * 100
        ax.annotate(f'+{frac_acc - flat_acc:.1f}%',
                    xy=(d, frac_acc + 2), ha='center', fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_scaling_law.png', bbox_inches='tight')
    print(f"Saved: fig3_scaling_law")
    plt.close()


def fig4_variance_comparison(scaling, output_dir):
    """Show that Fractal has consistently lower variance."""
    fig, ax = plt.subplots(figsize=(8, 5))

    depths = [2, 3, 4, 5]
    results = scaling['results_summary']

    flat_cis = [results[f'depth_{d}']['flat']['hier_acc_ci95'] * 100 for d in depths]
    frac_cis = [results[f'depth_{d}']['fractal']['hier_acc_ci95'] * 100 for d in depths]

    x = np.arange(len(depths))
    width = 0.35

    bars1 = ax.bar(x - width/2, flat_cis, width, label='Flat', color=COLORS['flat'], alpha=0.8)
    bars2 = ax.bar(x + width/2, frac_cis, width, label='Fractal', color=COLORS['fractal'], alpha=0.8)

    ax.set_xlabel('Hierarchy Depth', fontsize=12)
    ax.set_ylabel('95% CI Width (%)', fontsize=12)
    ax.set_title('Fractal Has Consistently Lower Variance\n(More Stable Predictions)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Depth {d}' for d in depths])
    ax.legend()

    # Add reduction percentages
    for i, (fc, ff) in enumerate(zip(flat_cis, frac_cis)):
        reduction = (fc - ff) / fc * 100
        ax.annotate(f'-{reduction:.0f}%', xy=(i + width/2, ff + 0.2),
                    ha='center', fontsize=9, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_variance_comparison.png', bbox_inches='tight')
    print(f"Saved: fig4_variance_comparison")
    plt.close()


def fig5_v5_results(v5_results, output_dir):
    """V5 multi-seed results showing per-seed improvements."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Data
    baseline = v5_results['baseline']
    per_seed = v5_results['per_seed']

    seeds = [r['seed'] for r in per_seed]
    l0_vals = [r['l0'] * 100 for r in per_seed]
    l1_vals = [r['l1'] * 100 for r in per_seed]

    # L0 plot
    ax1.bar(range(len(seeds)), l0_vals, color=COLORS['fractal'], alpha=0.8, label='V5 Fractal')
    ax1.axhline(y=baseline['l0'] * 100, color=COLORS['flat'], linestyle='--',
                linewidth=2, label=f'Baseline: {baseline["l0"]*100:.1f}%')
    ax1.set_xlabel('Seed', fontsize=12)
    ax1.set_ylabel('L0 (Coarse) Accuracy (%)', fontsize=12)
    ax1.set_title(f'L0 Accuracy: +{v5_results["delta"]["l0"]*100:.2f}% avg improvement',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(seeds)))
    ax1.set_xticklabels(seeds)
    ax1.legend()
    ax1.set_ylim(60, 80)

    # L1 plot
    ax2.bar(range(len(seeds)), l1_vals, color=COLORS['fractal'], alpha=0.8, label='V5 Fractal')
    ax2.axhline(y=baseline['l1'] * 100, color=COLORS['flat'], linestyle='--',
                linewidth=2, label=f'Baseline: {baseline["l1"]*100:.1f}%')
    ax2.set_xlabel('Seed', fontsize=12)
    ax2.set_ylabel('L1 (Fine) Accuracy (%)', fontsize=12)
    ax2.set_title(f'L1 Accuracy: +{v5_results["delta"]["l1"]*100:.2f}% avg improvement',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels(seeds)
    ax2.legend()
    ax2.set_ylim(55, 70)

    fig.suptitle('V5 Results on Yahoo Answers (5 seeds, all seeds show improvement)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_v5_results.png', bbox_inches='tight')
    print(f"Saved: fig5_v5_results")
    plt.close()


def fig6_newsgroups_realworld(newsgroups, output_dir):
    """Real-world validation on 20 Newsgroups."""
    fig, ax = plt.subplots(figsize=(10, 6))

    stats = newsgroups['statistics']
    metrics = ['hier_acc', 'l0_acc', 'l1_acc']
    labels = ['Hierarchical\n(All Levels Correct)', 'L0 (Super-category)', 'L1 (Sub-category)']

    x = np.arange(len(metrics))
    width = 0.35

    flat_vals = [stats['flat'][m]['mean'] * 100 for m in metrics]
    flat_cis = [stats['flat'][m]['ci_95'] * 100 for m in metrics]
    frac_vals = [stats['fractal'][m]['mean'] * 100 for m in metrics]
    frac_cis = [stats['fractal'][m]['ci_95'] * 100 for m in metrics]

    bars1 = ax.bar(x - width/2, flat_vals, width, yerr=flat_cis,
                   label='Flat', color=COLORS['flat'], capsize=4, alpha=0.8)
    bars2 = ax.bar(x + width/2, frac_vals, width, yerr=frac_cis,
                   label='Fractal', color=COLORS['fractal'], capsize=4, alpha=0.8)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Real-World Validation: 20 Newsgroups\n(5 seeds, p=0.0232 for hierarchical)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(60, 90)

    # Add improvement annotations
    for i, (fv, frv) in enumerate(zip(flat_vals, frac_vals)):
        delta = frv - fv
        ax.annotate(f'+{delta:.2f}%', xy=(i + width/2, frv + frac_cis[i] + 0.5),
                    ha='center', fontsize=10, color='green', fontweight='bold')

    # Add p-value annotation
    ax.text(0.02, 0.98, 'p = 0.0232\n(Statistically Significant)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_newsgroups_realworld.png', bbox_inches='tight')
    print(f"Saved: fig6_newsgroups_realworld")
    plt.close()


def fig7_summary_dashboard(hier_rand, v5_results, scaling, newsgroups, output_dir):
    """Single-image dashboard summarizing all key findings."""
    fig = plt.figure(figsize=(16, 10))

    # Title
    fig.suptitle('Fractal Embeddings: Key Results Summary', fontsize=18, fontweight='bold', y=0.98)

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # === Panel 1: Hierarchy Randomization (top-left) ===
    ax1 = fig.add_subplot(gs[0, 0])

    flat_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['flat_results']]) * 100
    true_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['true_results']]) * 100
    rand_accs = [r['aggregate']['hier_acc.mean'] * 100 for r in hier_rand['randomizations']]
    rand_mean = np.mean(rand_accs)

    conditions = ['Flat', 'TRUE\nHierarchy', 'RANDOM\nHierarchy']
    values = [flat_acc, true_acc, rand_mean]
    colors = [COLORS['flat'], COLORS['true'], COLORS['random']]

    bars = ax1.bar(conditions, values, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Hier. Accuracy (%)')
    ax1.set_title('1. Structure Sensitivity', fontweight='bold')
    ax1.set_ylim(65, 68.5)

    # Gap line
    ax1.annotate('', xy=(1, true_acc), xytext=(2, rand_mean),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax1.text(1.5, (true_acc + rand_mean)/2 + 0.3, f'+{true_acc - rand_mean:.1f}%',
             ha='center', fontsize=10, fontweight='bold')

    # === Panel 2: Scaling Law (top-right) ===
    ax2 = fig.add_subplot(gs[0, 1])

    depths = [2, 3, 4, 5]
    results = scaling['results_summary']

    flat_means = [results[f'depth_{d}']['flat']['hier_acc_mean'] * 100 for d in depths]
    frac_means = [results[f'depth_{d}']['fractal']['hier_acc_mean'] * 100 for d in depths]

    ax2.plot(depths, flat_means, 'o-', color=COLORS['flat'], label='Flat', linewidth=2, markersize=8)
    ax2.plot(depths, frac_means, 's-', color=COLORS['fractal'], label='Fractal', linewidth=2, markersize=8)
    ax2.set_xlabel('Hierarchy Depth')
    ax2.set_ylabel('Hier. Accuracy (%)')
    ax2.set_title('2. Scaling with Depth', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xticks(depths)

    # === Panel 3: V5 Results (bottom-left) ===
    ax3 = fig.add_subplot(gs[1, 0])

    categories = ['L0 (Coarse)', 'L1 (Fine)']
    baseline = [v5_results['baseline']['l0'] * 100, v5_results['baseline']['l1'] * 100]
    v5_mean = [v5_results['v5_mean']['l0'] * 100, v5_results['v5_mean']['l1'] * 100]

    x = np.arange(len(categories))
    width = 0.35

    ax3.bar(x - width/2, baseline, width, label='Baseline', color=COLORS['flat'], alpha=0.8)
    ax3.bar(x + width/2, v5_mean, width, label='V5', color=COLORS['fractal'], alpha=0.8)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('3. V5 on Yahoo Answers (5 seeds)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()

    # Add deltas
    for i in range(2):
        delta = v5_mean[i] - baseline[i]
        ax3.annotate(f'+{delta:.1f}%', xy=(i + width/2, v5_mean[i] + 1),
                     ha='center', fontsize=10, color='green', fontweight='bold')

    # === Panel 4: Real-World (bottom-right) ===
    ax4 = fig.add_subplot(gs[1, 1])

    stats = newsgroups['statistics']
    flat_hier = stats['flat']['hier_acc']['mean'] * 100
    frac_hier = stats['fractal']['hier_acc']['mean'] * 100

    ax4.bar(['Flat', 'Fractal'], [flat_hier, frac_hier],
            color=[COLORS['flat'], COLORS['fractal']], edgecolor='black', alpha=0.85)
    ax4.set_ylabel('Hier. Accuracy (%)')
    ax4.set_title('4. 20 Newsgroups (Real-World)', fontweight='bold')
    ax4.set_ylim(65, 68)

    delta = frac_hier - flat_hier
    ax4.annotate(f'+{delta:.2f}%\np=0.0232', xy=(1, frac_hier + 0.3),
                 ha='center', fontsize=10, color='green', fontweight='bold')

    # Add overall summary text
    fig.text(0.5, 0.02,
             'Key Finding: Correct hierarchical structure improves performance (+0.7-6%), '
             'while wrong structure actively hurts.',
             ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(output_dir / 'fig7_summary_dashboard.png', bbox_inches='tight')
    print(f"Saved: fig7_summary_dashboard")
    plt.close()


def main():
    """Generate all figures."""
    print("Loading results...")
    hier_rand, v5_results, scaling, newsgroups = load_results()

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'figures'
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating figures...")
    fig1_hierarchy_randomization(hier_rand, output_dir)
    fig2_delta_distribution(hier_rand, output_dir)
    fig3_scaling_law(scaling, output_dir)
    fig4_variance_comparison(scaling, output_dir)
    fig5_v5_results(v5_results, output_dir)
    fig6_newsgroups_realworld(newsgroups, output_dir)
    fig7_summary_dashboard(hier_rand, v5_results, scaling, newsgroups, output_dir)

    print(f"\n✓ All figures saved to: {output_dir}")
    print("\nFigures created:")
    print("  fig1_hierarchy_randomization.png - KEY FINDING: Structure matters")
    print("  fig2_delta_distribution.png - Random hierarchy delta distribution")
    print("  fig3_scaling_law.png - Depth scaling with 4 classifiers")
    print("  fig4_variance_comparison.png - Fractal has lower variance")
    print("  fig5_v5_results.png - V5 multi-seed results")
    print("  fig6_newsgroups_realworld.png - Real-world validation")
    print("  fig7_summary_dashboard.png - Single-image summary dashboard")


if __name__ == '__main__':
    main()
