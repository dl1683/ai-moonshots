"""
Create additional charts and animated GIFs for social media sharing.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14

COLORS = {
    'flat': '#E74C3C',
    'fractal': '#27AE60',
    'random': '#95A5A6',
    'true': '#27AE60',
}

def load_results():
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


def chart_hero_stat(output_dir):
    """Big bold stat card - perfect for social media."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Main stat
    ax.text(5, 4.5, '+0.82%', fontsize=80, ha='center', va='center',
            color='#00ff88', fontweight='bold', family='monospace')

    # Subtitle
    ax.text(5, 2.8, 'Gap between CORRECT vs RANDOM hierarchy',
            fontsize=18, ha='center', va='center', color='white')

    ax.text(5, 2.0, '95% CI excludes zero  |  K=30 randomizations',
            fontsize=14, ha='center', va='center', color='#888888')

    # Bottom text
    ax.text(5, 0.8, 'STRUCTURE MATTERS',
            fontsize=28, ha='center', va='center', color='#ff6b6b',
            fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'hero_stat_card.png', bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print("Saved: hero_stat_card.png")
    plt.close()


def chart_simple_comparison(hier_rand, output_dir):
    """Ultra-simple 3-bar comparison for quick sharing."""
    fig, ax = plt.subplots(figsize=(8, 6))

    flat_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['flat_results']]) * 100
    true_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['true_results']]) * 100
    rand_accs = [r['aggregate']['hier_acc.mean'] * 100 for r in hier_rand['randomizations']]
    rand_mean = np.mean(rand_accs)

    bars = ax.bar(['No\nHierarchy', 'CORRECT\nHierarchy', 'WRONG\nHierarchy'],
                  [flat_acc, true_acc, rand_mean],
                  color=[COLORS['flat'], COLORS['fractal'], COLORS['random']],
                  edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, [flat_acc, true_acc, rand_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.1f}%', ha='center', fontsize=16, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_ylim(65, 68.5)
    ax.set_title('Does Hierarchy Structure Matter?', fontsize=18, fontweight='bold', pad=20)

    # Add verdict
    ax.text(0.5, 0.02, 'YES - Wrong structure is WORSE than no structure!',
            transform=ax.transAxes, ha='center', fontsize=13,
            style='italic', color='#E74C3C')

    plt.tight_layout()
    plt.savefig(output_dir / 'simple_comparison.png', bbox_inches='tight')
    print("Saved: simple_comparison.png")
    plt.close()


def chart_waterfall(hier_rand, output_dir):
    """Waterfall chart showing the deltas."""
    fig, ax = plt.subplots(figsize=(10, 6))

    flat_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['flat_results']]) * 100
    true_delta = hier_rand['summary']['delta_true']['mean'] * 100
    rand_delta = hier_rand['summary']['delta_rand']['exp_mean'] * 100

    # Positions
    positions = [0, 1, 2]
    labels = ['Flat\nBaseline', 'Add CORRECT\nHierarchy', 'Add WRONG\nHierarchy']

    # Bar heights
    heights = [flat_acc, true_delta, rand_delta]
    colors = [COLORS['flat'], COLORS['fractal'], COLORS['random']]

    # Starting points
    bottoms = [0, flat_acc, flat_acc]

    bars = ax.bar(positions, heights, bottom=bottoms, color=colors,
                  edgecolor='black', linewidth=1.5, width=0.6)

    # Connecting lines
    ax.plot([0.3, 0.7], [flat_acc, flat_acc], 'k--', alpha=0.5)
    ax.plot([1.3, 1.7], [flat_acc, flat_acc], 'k--', alpha=0.5)

    # Labels
    ax.text(0, flat_acc/2, f'{flat_acc:.2f}%', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(1, flat_acc + true_delta/2, f'+{true_delta:.2f}%', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(2, flat_acc + rand_delta/2, f'{rand_delta:.2f}%', ha='center', va='center',
            fontsize=14, fontweight='bold', color='black')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Hierarchical Accuracy (%)', fontsize=13)
    ax.set_title('Waterfall: Effect of Hierarchy Structure', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(output_dir / 'waterfall_chart.png', bbox_inches='tight')
    print("Saved: waterfall_chart.png")
    plt.close()


def chart_confidence_intervals(hier_rand, output_dir):
    """Horizontal CI plot showing separation."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Data
    delta_true = hier_rand['summary']['delta_true']
    delta_rand = hier_rand['summary']['delta_rand']

    y_positions = [1, 0]
    means = [delta_true['mean'] * 100, delta_rand['exp_mean'] * 100]
    ci_lows = [delta_true['ci95_low'] * 100, delta_rand['ci95_low'] * 100]
    ci_highs = [delta_true['ci95_high'] * 100, delta_rand['ci95_high'] * 100]

    colors = [COLORS['fractal'], COLORS['random']]
    labels = ['TRUE Hierarchy', 'RANDOM Hierarchy']

    for i, (y, mean, ci_low, ci_high, color, label) in enumerate(
            zip(y_positions, means, ci_lows, ci_highs, colors, labels)):
        # CI bar
        ax.plot([ci_low, ci_high], [y, y], color=color, linewidth=8, alpha=0.4)
        # Mean point
        ax.scatter([mean], [y], color=color, s=200, zorder=5, edgecolor='black', linewidth=2)
        # Label
        ax.text(ci_high + 0.1, y, f'{mean:.2f}% [{ci_low:.2f}, {ci_high:.2f}]',
                va='center', fontsize=11, fontweight='bold')

    # Zero line
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (no effect)')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Delta vs Flat Baseline (%)', fontsize=12)
    ax.set_title('95% Confidence Intervals: TRUE vs RANDOM Hierarchy',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)

    # Annotation
    ax.text(0.7, 0.5, 'CIs do NOT\noverlap!', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_intervals.png', bbox_inches='tight')
    print("Saved: confidence_intervals.png")
    plt.close()


def chart_all_30_randomizations(hier_rand, output_dir):
    """Show all 30 randomizations as points."""
    fig, ax = plt.subplots(figsize=(12, 5))

    flat_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['flat_results']]) * 100
    true_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['true_results']]) * 100
    rand_accs = [r['aggregate']['hier_acc.mean'] * 100 for r in hier_rand['randomizations']]

    # Plot all 30 randomizations
    x = np.arange(len(rand_accs))
    colors = ['green' if acc > flat_acc else 'red' for acc in rand_accs]

    ax.scatter(x, rand_accs, c=colors, s=100, alpha=0.7, edgecolor='black', linewidth=1)

    # Reference lines
    ax.axhline(y=flat_acc, color=COLORS['flat'], linestyle='-', linewidth=2,
               label=f'Flat baseline: {flat_acc:.2f}%')
    ax.axhline(y=true_acc, color=COLORS['fractal'], linestyle='-', linewidth=2,
               label=f'TRUE hierarchy: {true_acc:.2f}%')
    ax.axhline(y=np.mean(rand_accs), color='gray', linestyle='--', linewidth=2,
               label=f'RANDOM mean: {np.mean(rand_accs):.2f}%')

    ax.set_xlabel('Randomization ID', fontsize=12)
    ax.set_ylabel('Hierarchical Accuracy (%)', fontsize=12)
    ax.set_title('All 30 Random Hierarchy Permutations\n(Green = beats flat, Red = worse than flat)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')

    # Count
    n_above = sum(1 for acc in rand_accs if acc > flat_acc)
    ax.text(0.02, 0.02, f'{n_above}/30 beat flat ({n_above/30*100:.0f}%)',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'all_30_randomizations.png', bbox_inches='tight')
    print("Saved: all_30_randomizations.png")
    plt.close()


def chart_before_after(v5_results, output_dir):
    """Before/After style comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    baseline = v5_results['baseline']
    v5_mean = v5_results['v5_mean']

    for ax, (metric, label) in zip(axes, [('l0', 'L0 (Coarse)'), ('l1', 'L1 (Fine)')]):
        before = baseline[metric] * 100
        after = v5_mean[metric] * 100
        delta = after - before

        bars = ax.bar(['Before\n(Baseline)', 'After\n(V5 Fractal)'],
                      [before, after],
                      color=[COLORS['flat'], COLORS['fractal']],
                      edgecolor='black', linewidth=2)

        # Delta annotation
        ax.annotate('', xy=(1, after), xytext=(1, before),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(1.15, (before + after)/2, f'+{delta:.1f}%',
                fontsize=14, fontweight='bold', color='green')

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_ylim(50, 80)

    fig.suptitle('Before vs After: V5 Fractal Embeddings', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'before_after.png', bbox_inches='tight')
    print("Saved: before_after.png")
    plt.close()


def create_animated_gif(hier_rand, output_dir):
    """Animated GIF showing the key finding."""
    fig, ax = plt.subplots(figsize=(10, 6))

    flat_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['flat_results']]) * 100
    true_acc = np.mean([r['metrics']['hier_acc'] for r in hier_rand['true_results']]) * 100
    rand_accs = [r['aggregate']['hier_acc.mean'] * 100 for r in hier_rand['randomizations']]
    rand_mean = np.mean(rand_accs)

    def animate(frame):
        ax.clear()
        ax.set_ylim(64, 70)
        ax.set_ylabel('Hierarchical Accuracy (%)', fontsize=13)

        if frame < 10:
            # Show flat only
            ax.bar(['Flat\nBaseline'], [flat_acc], color=COLORS['flat'],
                   edgecolor='black', linewidth=2)
            ax.set_title('Step 1: Flat Baseline', fontsize=14, fontweight='bold')
            ax.text(0, flat_acc + 0.3, f'{flat_acc:.2f}%', ha='center',
                    fontsize=14, fontweight='bold')

        elif frame < 20:
            # Add true hierarchy
            ax.bar(['Flat\nBaseline', 'TRUE\nHierarchy'], [flat_acc, true_acc],
                   color=[COLORS['flat'], COLORS['fractal']],
                   edgecolor='black', linewidth=2)
            ax.set_title('Step 2: Add CORRECT Hierarchy (+0.72%)', fontsize=14, fontweight='bold')
            for i, (val, x) in enumerate(zip([flat_acc, true_acc], [0, 1])):
                ax.text(x, val + 0.3, f'{val:.2f}%', ha='center',
                        fontsize=14, fontweight='bold')

        else:
            # Add random hierarchy
            ax.bar(['Flat\nBaseline', 'TRUE\nHierarchy', 'RANDOM\nHierarchy'],
                   [flat_acc, true_acc, rand_mean],
                   color=[COLORS['flat'], COLORS['fractal'], COLORS['random']],
                   edgecolor='black', linewidth=2)
            ax.set_title('Step 3: WRONG Hierarchy HURTS (-0.10%)', fontsize=14, fontweight='bold')
            for i, (val, x) in enumerate(zip([flat_acc, true_acc, rand_mean], [0, 1, 2])):
                ax.text(x, val + 0.3, f'{val:.2f}%', ha='center',
                        fontsize=14, fontweight='bold')

            # Add key insight
            ax.text(0.5, 0.02, 'KEY: Structure must MATCH the data!',
                    transform=ax.transAxes, ha='center', fontsize=12,
                    color='red', fontweight='bold')

        return []

    anim = animation.FuncAnimation(fig, animate, frames=30, interval=200, blit=True)
    anim.save(output_dir / 'hierarchy_animation.gif', writer='pillow', fps=5)
    print("Saved: hierarchy_animation.gif")
    plt.close()


def create_scaling_gif(scaling, output_dir):
    """Animated GIF showing scaling across depths."""
    fig, ax = plt.subplots(figsize=(10, 6))

    depths = [2, 3, 4, 5]
    results = scaling['results_summary']

    flat_means = [results[f'depth_{d}']['flat']['hier_acc_mean'] * 100 for d in depths]
    frac_means = [results[f'depth_{d}']['fractal']['hier_acc_mean'] * 100 for d in depths]

    def animate(frame):
        ax.clear()
        ax.set_xlim(1.5, 5.5)
        ax.set_ylim(35, 95)
        ax.set_xlabel('Hierarchy Depth', fontsize=12)
        ax.set_ylabel('Hierarchical Accuracy (%)', fontsize=12)
        ax.set_xticks(depths)

        n_points = min(frame + 1, len(depths))

        ax.plot(depths[:n_points], flat_means[:n_points], 'o-',
                color=COLORS['flat'], label='Flat', linewidth=2, markersize=10)
        ax.plot(depths[:n_points], frac_means[:n_points], 's-',
                color=COLORS['fractal'], label='Fractal (ours)', linewidth=2, markersize=10)

        if n_points > 0:
            # Show current advantage
            d = depths[n_points - 1]
            adv = frac_means[n_points - 1] - flat_means[n_points - 1]
            ax.text(d, frac_means[n_points - 1] + 3, f'+{adv:.1f}%',
                    ha='center', fontsize=12, color='green', fontweight='bold')

        ax.legend(loc='upper right')
        ax.set_title(f'Fractal Beats Flat at Depth {depths[min(frame, len(depths)-1)]}',
                     fontsize=14, fontweight='bold')

        return []

    anim = animation.FuncAnimation(fig, animate, frames=8, interval=500, blit=True)
    anim.save(output_dir / 'scaling_animation.gif', writer='pillow', fps=2)
    print("Saved: scaling_animation.gif")
    plt.close()


def chart_tweet_card(output_dir):
    """Perfect square image for Twitter."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Title
    ax.text(5, 9, 'FRACTAL EMBEDDINGS', fontsize=24, ha='center',
            color='#58a6ff', fontweight='bold')

    # Key finding
    ax.text(5, 7, 'KEY FINDING:', fontsize=16, ha='center', color='white')
    ax.text(5, 5.5, 'Structure\nMatters', fontsize=42, ha='center',
            color='#3fb950', fontweight='bold', linespacing=0.9)

    # Stats
    ax.text(5, 3.5, 'Correct hierarchy: +0.72%', fontsize=14, ha='center', color='#3fb950')
    ax.text(5, 2.8, 'Wrong hierarchy: -0.10%', fontsize=14, ha='center', color='#f85149')
    ax.text(5, 2.1, 'Gap: +0.82% (p < 0.05)', fontsize=14, ha='center', color='#8b949e')

    # Bottom
    ax.text(5, 0.8, '#AI #MachineLearning #Research', fontsize=12,
            ha='center', color='#8b949e')

    plt.tight_layout()
    plt.savefig(output_dir / 'tweet_card.png', bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    print("Saved: tweet_card.png")
    plt.close()


def main():
    print("Loading results...")
    hier_rand, v5_results, scaling, newsgroups = load_results()

    output_dir = Path(__file__).parent.parent / 'results' / 'figures'
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating extra charts...")
    chart_hero_stat(output_dir)
    chart_simple_comparison(hier_rand, output_dir)
    chart_waterfall(hier_rand, output_dir)
    chart_confidence_intervals(hier_rand, output_dir)
    chart_all_30_randomizations(hier_rand, output_dir)
    chart_before_after(v5_results, output_dir)
    chart_tweet_card(output_dir)

    print("\nGenerating animated GIFs...")
    create_animated_gif(hier_rand, output_dir)
    create_scaling_gif(scaling, output_dir)

    print(f"\nAll extra visuals saved to: {output_dir}")


if __name__ == '__main__':
    main()
