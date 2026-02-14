"""
Combined Prediction Analysis: Merge Round 1 + Round 2 pre-registered predictions.

Creates definitive prediction validation figure and statistics for the paper.
Produces:
  - Combined correlation plot (calibration + test rounds 1&2)
  - Updated statistics (Spearman, Pearson, RMSE, MAE)
  - Coverage analysis (% in prediction intervals)

Usage: python -u src/combined_prediction_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "paper"


def load_calibration_data():
    """Load the 8 calibration datasets from scaling_robustness.json."""
    scaling = json.load(open(RESULTS_DIR / "scaling_robustness.json"))

    # Product predictor values and steerability gaps
    products = scaling['interaction_analysis']['product_h_l1']
    # Need mean steerability per dataset - get from meta_analysis
    meta = json.load(open(RESULTS_DIR / "meta_analysis.json"))
    gaps = {d['name']: d['gap_mean'] for d in meta['per_dataset']}

    calibration_names = ["yahoo", "goemotions", "newsgroups", "trec",
                         "arxiv", "dbpedia_classes", "clinc", "wos"]

    cal_data = []
    for name in calibration_names:
        cal_data.append({
            'name': name,
            'product': products[name],
            'actual_S': gaps[name],
            'type': 'calibration',
        })

    return cal_data


def load_round1_predictions():
    """Load round 1 pre-registered predictions (deep hierarchy datasets)."""
    # Round 1 predictions are in the paper: HUPD sec_cls, HUPD sec_sub, HWV L0->L2, HWV L0->L3
    deep = json.load(open(RESULTS_DIR / "deep_hierarchy_summary.json"))

    # Pre-registered predictions from the paper
    round1_predictions = {
        'hupd_sec_cls': {'predicted_S': 0.037, 'product': 1.15, 'H': 2.42},
        'hupd_sec_sub': {'predicted_S': 0.054, 'product': 1.48, 'H': 4.44},
        'hwv_l0_l2': {'predicted_S': 0.120, 'product': 2.80, 'H': 4.09},
        'hwv_l0_l3': {'predicted_S': 0.145, 'product': 3.29, 'H': 4.59},
    }

    actual_values = {s['config']: s['v5_mean'] for s in deep['summaries']}

    round1_data = []
    for name, pred in round1_predictions.items():
        round1_data.append({
            'name': name,
            'product': pred['product'],
            'predicted_S': pred['predicted_S'],
            'actual_S': actual_values.get(name, None),
            'type': 'round1_test',
        })

    return round1_data


def load_round2_predictions():
    """Load round 2 pre-registered predictions (4 unseen datasets)."""
    pred_file = RESULTS_DIR / "prospective_predictions_round2.json"
    result_file = RESULTS_DIR / "prospective_validation_round2.json"

    preds = json.load(open(pred_file))

    round2_data = []
    actual_results = None
    if result_file.exists():
        actual_results = json.load(open(result_file))

    for name, pred in preds['predictions'].items():
        actual_S = None
        if actual_results and name in actual_results.get('results', {}):
            actual_S = actual_results['results'][name]['actual_S_mean']

        round2_data.append({
            'name': name,
            'product': pred['product'],
            'predicted_S': pred['predicted_S'],
            'prediction_interval': pred['prediction_interval_95'],
            'actual_S': actual_S,
            'type': 'round2_test',
        })

    return round2_data


def compute_combined_statistics(cal_data, round1_data, round2_data):
    """Compute combined statistics across all rounds."""
    stats = {}

    # Calibration fit
    cal_products = [d['product'] for d in cal_data]
    cal_actuals = [d['actual_S'] for d in cal_data]
    slope, intercept, r, p, se = sp_stats.linregress(cal_products, cal_actuals)
    stats['calibration'] = {
        'n': len(cal_data),
        'slope': slope,
        'intercept': intercept,
        'r_squared': r**2,
        'p': p,
    }

    # Round 1 test stats
    r1_predicted = [d['predicted_S'] for d in round1_data if d['actual_S'] is not None]
    r1_actual = [d['actual_S'] for d in round1_data if d['actual_S'] is not None]
    if len(r1_predicted) > 1:
        r1_residuals = np.array(r1_actual) - np.array(r1_predicted)
        stats['round1'] = {
            'n': len(r1_predicted),
            'rmse': float(np.sqrt(np.mean(r1_residuals**2))),
            'mae': float(np.mean(np.abs(r1_residuals))),
            'sign_correct': sum(1 for a, p in zip(r1_actual, r1_predicted) if (a > 0) == (p > 0)),
        }

    # Round 2 test stats
    r2_predicted = [d['predicted_S'] for d in round2_data if d['actual_S'] is not None]
    r2_actual = [d['actual_S'] for d in round2_data if d['actual_S'] is not None]
    if len(r2_predicted) > 1:
        r2_residuals = np.array(r2_actual) - np.array(r2_predicted)
        r2_in_pi = sum(1 for d in round2_data
                       if d['actual_S'] is not None and
                       d['prediction_interval'][0] <= d['actual_S'] <= d['prediction_interval'][1])
        stats['round2'] = {
            'n': len(r2_predicted),
            'rmse': float(np.sqrt(np.mean(r2_residuals**2))),
            'mae': float(np.mean(np.abs(r2_residuals))),
            'sign_correct': sum(1 for a, p in zip(r2_actual, r2_predicted) if (a > 0) == (p > 0)),
            'in_pi': r2_in_pi,
        }

    # Combined test stats (round 1 + round 2)
    all_predicted = r1_predicted + r2_predicted
    all_actual = r1_actual + r2_actual
    if len(all_predicted) > 2:
        all_residuals = np.array(all_actual) - np.array(all_predicted)
        rho_test, p_rho_test = sp_stats.spearmanr(all_predicted, all_actual)
        r_test, p_r_test = sp_stats.pearsonr(all_predicted, all_actual)
        stats['combined_test'] = {
            'n': len(all_predicted),
            'spearman_rho': float(rho_test),
            'spearman_p': float(p_rho_test),
            'pearson_r': float(r_test),
            'pearson_p': float(p_r_test),
            'rmse': float(np.sqrt(np.mean(all_residuals**2))),
            'mae': float(np.mean(np.abs(all_residuals))),
        }

    # Combined all (calibration + test)
    all_products = cal_products + [d['product'] for d in round1_data + round2_data if d['actual_S'] is not None]
    all_actuals_combined = cal_actuals + [d['actual_S'] for d in round1_data + round2_data if d['actual_S'] is not None]
    if len(all_products) > 2:
        rho_all, p_rho_all = sp_stats.spearmanr(all_products, all_actuals_combined)
        r_all, p_r_all = sp_stats.pearsonr(all_products, all_actuals_combined)
        stats['combined_all'] = {
            'n': len(all_products),
            'spearman_rho': float(rho_all),
            'spearman_p': float(p_rho_all),
            'pearson_r': float(r_all),
            'pearson_p': float(p_r_all),
        }

    return stats


def generate_combined_figure(cal_data, round1_data, round2_data, stats):
    """Generate combined prediction validation figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Product predictor plot with all data
    cal_products = [d['product'] for d in cal_data]
    cal_actuals = [d['actual_S'] for d in cal_data]
    ax1.scatter(cal_products, cal_actuals, c='steelblue', s=60, zorder=3,
                label=f"Calibration (n={len(cal_data)})", alpha=0.8)

    # Round 1 test
    r1_with_actual = [d for d in round1_data if d['actual_S'] is not None]
    if r1_with_actual:
        r1_products = [d['product'] for d in r1_with_actual]
        r1_actuals = [d['actual_S'] for d in r1_with_actual]
        ax1.scatter(r1_products, r1_actuals, c='orangered', s=80, marker='D', zorder=4,
                    label=f"Round 1 test (n={len(r1_with_actual)})", edgecolors='black', linewidth=0.5)

    # Round 2 test
    r2_with_actual = [d for d in round2_data if d['actual_S'] is not None]
    if r2_with_actual:
        r2_products = [d['product'] for d in r2_with_actual]
        r2_actuals = [d['actual_S'] for d in r2_with_actual]
        ax1.scatter(r2_products, r2_actuals, c='forestgreen', s=80, marker='s', zorder=4,
                    label=f"Round 2 test (n={len(r2_with_actual)})", edgecolors='black', linewidth=0.5)

    # Regression line from calibration
    x_range = np.linspace(0, max(cal_products + [d['product'] for d in round1_data + round2_data]) * 1.1, 100)
    slope = stats['calibration']['slope']
    intercept = stats['calibration']['intercept']
    ax1.plot(x_range, slope * x_range + intercept, 'k--', alpha=0.5,
             label=f"Calibration fit (R$^2$={stats['calibration']['r_squared']:.3f})")

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel("Product: H(L$_1$|L$_0$) x A$_{L_1}^{base}$", fontsize=11)
    ax1.set_ylabel("Steerability S", fontsize=11)
    ax1.set_title("Product Predictor with Pre-Registered Tests", fontsize=12)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.2)

    # Right panel: Predicted vs Actual for test datasets
    all_test = r1_with_actual + r2_with_actual
    if all_test:
        predicted = [d['predicted_S'] for d in all_test]
        actual = [d['actual_S'] for d in all_test]

        # Round 1 points
        for d in r1_with_actual:
            ax2.scatter(d['predicted_S'], d['actual_S'], c='orangered', s=80, marker='D',
                        zorder=4, edgecolors='black', linewidth=0.5)
        # Round 2 points
        for d in r2_with_actual:
            ax2.scatter(d['predicted_S'], d['actual_S'], c='forestgreen', s=80, marker='s',
                        zorder=4, edgecolors='black', linewidth=0.5)

        # Identity line
        min_val = min(min(predicted), min(actual)) - 0.01
        max_val = max(max(predicted), max(actual)) + 0.01
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect prediction')

        # Labels
        for d in all_test:
            label = d['name'].replace('_', ' ')
            if len(label) > 12:
                label = label[:12]
            ax2.annotate(label, (d['predicted_S'], d['actual_S']),
                         textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)

        if 'combined_test' in stats:
            ct = stats['combined_test']
            ax2.set_title(f"Predicted vs Actual (n={ct['n']}, "
                          f"rho={ct['spearman_rho']:.2f}, "
                          f"MAE={ct['mae']:.3f})", fontsize=12)

    ax2.set_xlabel("Predicted S", fontsize=11)
    ax2.set_ylabel("Actual S", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f"fig_prediction_combined.{ext}", dpi=150, bbox_inches='tight')
    print(f"  Figure saved to {FIGURES_DIR / 'fig_prediction_combined.pdf'}")
    plt.close()


def main():
    print("=" * 60)
    print("  COMBINED PREDICTION ANALYSIS")
    print("=" * 60)

    # Load data
    cal_data = load_calibration_data()
    round1_data = load_round1_predictions()
    round2_data = load_round2_predictions()

    print(f"\n  Calibration datasets: {len(cal_data)}")
    print(f"  Round 1 test datasets: {len(round1_data)}")
    print(f"  Round 2 test datasets: {len(round2_data)}")
    print(f"  Round 2 with results: {sum(1 for d in round2_data if d['actual_S'] is not None)}")

    # Compute stats
    stats = compute_combined_statistics(cal_data, round1_data, round2_data)

    # Print results
    print(f"\n  CALIBRATION:")
    print(f"    R^2 = {stats['calibration']['r_squared']:.4f}")

    if 'round1' in stats:
        r1 = stats['round1']
        print(f"\n  ROUND 1 TEST:")
        print(f"    n = {r1['n']}, RMSE = {r1['rmse']:.4f}, MAE = {r1['mae']:.4f}")
        print(f"    Sign correct: {r1['sign_correct']}/{r1['n']}")

    if 'round2' in stats:
        r2 = stats['round2']
        print(f"\n  ROUND 2 TEST:")
        print(f"    n = {r2['n']}, RMSE = {r2['rmse']:.4f}, MAE = {r2['mae']:.4f}")
        print(f"    Sign correct: {r2['sign_correct']}/{r2['n']}")
        print(f"    In 95% PI: {r2['in_pi']}/{r2['n']}")

    if 'combined_test' in stats:
        ct = stats['combined_test']
        print(f"\n  COMBINED TEST (Round 1 + Round 2):")
        print(f"    n = {ct['n']}")
        print(f"    Spearman rho = {ct['spearman_rho']:.3f} (p = {ct['spearman_p']:.6f})")
        print(f"    Pearson r = {ct['pearson_r']:.3f} (p = {ct['pearson_p']:.6f})")
        print(f"    RMSE = {ct['rmse']:.4f}, MAE = {ct['mae']:.4f}")

    if 'combined_all' in stats:
        ca = stats['combined_all']
        print(f"\n  COMBINED ALL (Calibration + Test):")
        print(f"    n = {ca['n']}")
        print(f"    Spearman rho = {ca['spearman_rho']:.3f} (p = {ca['spearman_p']:.6f})")
        print(f"    Pearson r = {ca['pearson_r']:.3f} (p = {ca['pearson_p']:.6f})")

    # Generate figure
    generate_combined_figure(cal_data, round1_data, round2_data, stats)

    # Save stats
    out = {
        'calibration_data': cal_data,
        'round1_data': round1_data,
        'round2_data': round2_data,
        'statistics': stats,
    }
    out_path = RESULTS_DIR / "combined_prediction_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
