"""
Visualize Scaling Law Results
==============================

Creates publication-quality figures from rigorous scaling experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def plot_scaling_law(results: dict, save_path: str = None):
    """Plot hierarchical accuracy vs depth with confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    depths = sorted([int(k.split('_')[1]) for k in results.keys() if k.startswith('depth_')])
    classifiers = ['flat', 'fractal', 'hier_softmax', 'classifier_chain']
    colors = {'flat': '#1f77b4', 'fractal': '#d62728', 'hier_softmax': '#2ca02c', 'classifier_chain': '#9467bd'}
    labels = {'flat': 'Flat (Baseline)', 'fractal': 'Fractal (Ours)', 'hier_softmax': 'Hier. Softmax', 'classifier_chain': 'Classifier Chain'}

    # Plot 1: Hierarchical Accuracy vs Depth
    ax1 = axes[0]
    for clf in classifiers:
        means = []
        cis = []
        for d in depths:
            stats = results[f'depth_{d}']['statistics'][clf]['hier_acc']
            means.append(stats['mean'] * 100)
            cis.append(stats['ci_95'] * 100)

        ax1.errorbar(depths, means, yerr=cis, marker='o', capsize=5, label=labels[clf], color=colors[clf], linewidth=2, markersize=8)

    ax1.set_xlabel('Hierarchy Depth', fontsize=12)
    ax1.set_ylabel('Hierarchical Accuracy (%)', fontsize=12)
    ax1.set_title('Hierarchical Accuracy vs Depth\n(95% CI across 3 seeds)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(depths)

    # Plot 2: Fractal Advantage (Delta) vs Depth
    ax2 = axes[1]
    baselines = ['flat', 'hier_softmax', 'classifier_chain']
    baseline_labels = {'flat': 'vs Flat', 'hier_softmax': 'vs Hier. Softmax', 'classifier_chain': 'vs Classifier Chain'}
    baseline_colors = {'flat': '#1f77b4', 'hier_softmax': '#2ca02c', 'classifier_chain': '#9467bd'}

    for baseline in baselines:
        deltas = []
        for d in depths:
            frac_mean = results[f'depth_{d}']['statistics']['fractal']['hier_acc']['mean']
            base_mean = results[f'depth_{d}']['statistics'][baseline]['hier_acc']['mean']
            deltas.append((frac_mean - base_mean) * 100)

        ax2.bar([d + 0.2 * (baselines.index(baseline) - 1) for d in depths], deltas,
                width=0.2, label=baseline_labels[baseline], color=baseline_colors[baseline], alpha=0.8)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Hierarchy Depth', fontsize=12)
    ax2.set_ylabel('Fractal Advantage (%)', fontsize=12)
    ax2.set_title('Fractal Advantage Over Baselines', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(depths)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_conditional_accuracy(results: dict, save_path: str = None):
    """Plot conditional accuracy P(L_k | L_0...L_{k-1}) for each classifier."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    depths = sorted([int(k.split('_')[1]) for k in results.keys() if k.startswith('depth_')])
    classifiers = ['flat', 'fractal', 'hier_softmax', 'classifier_chain']
    colors = {'flat': '#1f77b4', 'fractal': '#d62728', 'hier_softmax': '#2ca02c', 'classifier_chain': '#9467bd'}

    for idx, d in enumerate(depths):
        ax = axes[idx // 2, idx % 2]
        levels = list(range(d))

        for clf in classifiers:
            cond_accs = []
            for level in levels:
                stats = results[f'depth_{d}']['statistics'][clf][f'cond_l{level}_acc']
                cond_accs.append(stats['mean'] * 100)

            ax.plot(levels, cond_accs, marker='o', label=clf.replace('_', ' ').title(), color=colors[clf], linewidth=2)

        ax.set_xlabel('Level', fontsize=11)
        ax.set_ylabel('Conditional Accuracy (%)', fontsize=11)
        ax.set_title(f'Depth {d}: P(L_k correct | L_0...L_{{k-1}} correct)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(levels)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def print_summary_table(results: dict):
    """Print a summary table for the paper."""
    print("\n" + "="*100)
    print("SUMMARY TABLE FOR PAPER")
    print("="*100)

    depths = sorted([int(k.split('_')[1]) for k in results.keys() if k.startswith('depth_')])
    classifiers = ['flat', 'fractal', 'hier_softmax', 'classifier_chain']

    print(f"\n{'Depth':<8}", end="")
    for clf in classifiers:
        print(f"{clf:<22}", end="")
    print()
    print("-"*100)

    for d in depths:
        print(f"{d:<8}", end="")
        for clf in classifiers:
            stats = results[f'depth_{d}']['statistics'][clf]['hier_acc']
            print(f"{stats['mean']*100:.2f} ± {stats['ci_95']*100:.2f}%     ", end="")
        print()

    # Print fractal advantage with significance
    print("\n\nFRACTAL ADVANTAGE (vs Flat) with p-values:")
    print("-"*60)
    for d in depths:
        frac_vals = results[f'depth_{d}']['statistics']['fractal']['hier_acc']['values']
        flat_vals = results[f'depth_{d}']['statistics']['flat']['hier_acc']['values']
        delta = np.mean(frac_vals) - np.mean(flat_vals)

        # Manual t-test
        n = len(frac_vals)
        diff = [f - b for f, b in zip(frac_vals, flat_vals)]
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        t_stat = mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else 0
        # Two-tailed p-value approximation (for n=3, use t-distribution)
        from scipy import stats as scipy_stats
        p_val = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), n-1))

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"Depth {d}: Δ = {delta*100:+.2f}%, p = {p_val:.4f} {sig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None, help="Path to results JSON")
    parser.add_argument("--model", type=str, default="qwen3-0.6b", help="Model name for default path")
    args = parser.parse_args()

    # Find results file
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = Path(__file__).parent.parent / "results" / f"rigorous_scaling_{args.model}.json"

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run rigorous_scaling_experiment.py first.")
        return

    results = load_results(results_path)

    # Print summary
    print_summary_table(results)

    # Create visualizations
    fig_dir = results_path.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_scaling_law(results, str(fig_dir / "scaling_law.png"))
    plot_conditional_accuracy(results, str(fig_dir / "conditional_accuracy.png"))

    print(f"\nFigures saved to {fig_dir}")


if __name__ == "__main__":
    main()
