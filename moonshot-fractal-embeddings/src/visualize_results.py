"""
Visualize Fractal Classifier Results
=====================================

Creates publication-quality figures from experiment results.
Works with our simplified JSON format.
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

    summary = results['results_summary']
    depths = sorted([int(k.split('_')[1]) for k in summary.keys()])
    classifiers = ['flat', 'fractal', 'hier_softmax', 'classifier_chain']
    colors = {'flat': '#1f77b4', 'fractal': '#d62728', 'hier_softmax': '#2ca02c', 'classifier_chain': '#9467bd'}
    labels = {'flat': 'Flat (Baseline)', 'fractal': 'Fractal (Ours)', 'hier_softmax': 'Hier. Softmax', 'classifier_chain': 'Classifier Chain'}
    markers = {'flat': 'o', 'fractal': 's', 'hier_softmax': '^', 'classifier_chain': 'D'}

    # Plot 1: Hierarchical Accuracy vs Depth
    ax1 = axes[0]
    for clf in classifiers:
        means = []
        cis = []
        for d in depths:
            data = summary[f'depth_{d}'][clf]
            means.append(data['hier_acc_mean'] * 100)
            cis.append(data['hier_acc_ci95'] * 100)

        ax1.errorbar(depths, means, yerr=cis, marker=markers[clf], capsize=5,
                     label=labels[clf], color=colors[clf], linewidth=2.5, markersize=10)

    ax1.set_xlabel('Hierarchy Depth', fontsize=14)
    ax1.set_ylabel('Hierarchical Accuracy (%)', fontsize=14)
    ax1.set_title('Hierarchical Accuracy vs Depth\n(95% CI across 3 seeds)', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(depths)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Plot 2: Fractal Advantage (Delta) vs Depth
    ax2 = axes[1]
    advantage = results['fractal_advantage']

    baselines = ['vs_flat', 'vs_hier_softmax', 'vs_classifier_chain']
    baseline_labels = {'vs_flat': 'vs Flat', 'vs_hier_softmax': 'vs Hier. Softmax', 'vs_classifier_chain': 'vs Classifier Chain'}
    baseline_colors = {'vs_flat': '#1f77b4', 'vs_hier_softmax': '#2ca02c', 'vs_classifier_chain': '#9467bd'}

    bar_width = 0.25
    for i, baseline in enumerate(baselines):
        deltas = [advantage[f'depth_{d}'][baseline] * 100 for d in depths]
        positions = [d + bar_width * (i - 1) for d in depths]
        ax2.bar(positions, deltas, width=bar_width, label=baseline_labels[baseline],
                color=baseline_colors[baseline], alpha=0.8)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Hierarchy Depth', fontsize=14)
    ax2.set_ylabel('Fractal Advantage (%)', fontsize=14)
    ax2.set_title('Fractal Advantage Over Baselines', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(depths)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_variance_comparison(results: dict, save_path: str = None):
    """Plot variance comparison showing fractal has lower variance."""
    summary = results['results_summary']
    depths = sorted([int(k.split('_')[1]) for k in summary.keys()])

    fig, ax = plt.subplots(figsize=(10, 6))

    classifiers = ['flat', 'fractal']
    colors = {'flat': '#1f77b4', 'fractal': '#d62728'}
    labels = {'flat': 'Flat Classifier', 'fractal': 'Fractal Classifier (Ours)'}

    bar_width = 0.35
    for i, clf in enumerate(classifiers):
        cis = [summary[f'depth_{d}'][clf]['hier_acc_ci95'] * 100 for d in depths]
        positions = [d + bar_width * (i - 0.5) for d in depths]
        ax.bar(positions, cis, width=bar_width, label=labels[clf], color=colors[clf], alpha=0.8)

    ax.set_xlabel('Hierarchy Depth', fontsize=14)
    ax.set_ylabel('95% Confidence Interval Width (%)', fontsize=14)
    ax.set_title('Variance Comparison: Fractal Has Lower Variance', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(depths)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add annotation
    ax.annotate('Lower is better\n(more consistent)', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, ha='left', va='top', style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def print_latex_table(results: dict):
    """Print results as LaTeX table for paper."""
    summary = results['results_summary']
    depths = sorted([int(k.split('_')[1]) for k in summary.keys()])
    classifiers = ['flat', 'fractal', 'hier_softmax', 'classifier_chain']

    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print("""
\\begin{table}[h]
\\centering
\\caption{Hierarchical accuracy (\\%) across depths 2-5. Results shown as mean ± 95\\% CI across 3 seeds. Bold indicates best performance per depth.}
\\label{tab:scaling}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Depth} & \\textbf{Flat} & \\textbf{Fractal (Ours)} & \\textbf{Hier. Softmax} & \\textbf{Classifier Chain} \\\\
\\midrule""")

    for d in depths:
        row = f"{d}"
        best_mean = 0
        for clf in classifiers:
            data = summary[f'depth_{d}'][clf]
            if data['hier_acc_mean'] > best_mean:
                best_mean = data['hier_acc_mean']

        for clf in classifiers:
            data = summary[f'depth_{d}'][clf]
            mean = data['hier_acc_mean'] * 100
            ci = data['hier_acc_ci95'] * 100

            if data['hier_acc_mean'] == best_mean:
                row += f" & \\textbf{{{mean:.1f} ± {ci:.1f}}}"
            else:
                row += f" & {mean:.1f} ± {ci:.1f}"

        row += " \\\\"
        print(row)

    print("""\\bottomrule
\\end{tabular}
\\end{table}""")


def print_summary(results: dict):
    """Print human-readable summary."""
    print("\n" + "="*80)
    print("FRACTAL CLASSIFIER RESULTS SUMMARY")
    print("="*80)

    summary = results['results_summary']
    depths = sorted([int(k.split('_')[1]) for k in summary.keys()])

    print(f"\n{'Depth':<8} {'Flat':<20} {'Fractal (Ours)':<20} {'Delta':<10}")
    print("-"*60)

    for d in depths:
        flat = summary[f'depth_{d}']['flat']
        frac = summary[f'depth_{d}']['fractal']
        delta = (frac['hier_acc_mean'] - flat['hier_acc_mean']) * 100

        flat_str = f"{flat['hier_acc_mean']*100:.2f}% ± {flat['hier_acc_ci95']*100:.2f}%"
        frac_str = f"{frac['hier_acc_mean']*100:.2f}% ± {frac['hier_acc_ci95']*100:.2f}%"

        print(f"{d:<8} {flat_str:<20} {frac_str:<20} {delta:+.2f}%")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    for finding in results.get('key_findings', []):
        print(f"  • {finding}")


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

    # Print summaries
    print_summary(results)
    print_latex_table(results)

    # Create visualizations
    fig_dir = results_path.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_scaling_law(results, str(fig_dir / "scaling_law.png"))
    plot_variance_comparison(results, str(fig_dir / "variance_comparison.png"))

    print(f"\nFigures saved to {fig_dir}")


if __name__ == "__main__":
    main()
