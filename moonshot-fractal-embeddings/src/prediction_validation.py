"""Pre-registered Prediction Validation for Deep Hierarchy Experiments.

Compares frozen predictions (made BEFORE seeing results) against actual
outcomes. This is the strongest evidence for a predictive theory:
the product predictor (H * base_L1) was calibrated on 8 original datasets,
then used to predict steerability for HUPD and HWV configs without peeking.

Run: python src/prediction_validation.py
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Original 8 calibration datasets (used to fit the predictor)
CALIBRATION_DATASETS = [
    "yahoo", "goemotions", "newsgroups", "trec",
    "arxiv", "dbpedia_classes", "clinc", "wos"
]

# Deep hierarchy test datasets (predictions were frozen before results)
TEST_DATASETS = ["hupd_sec_cls", "hupd_sec_sub", "hwv_l0_l2", "hwv_l0_l3"]

# Known H(L1|L0) values
H_VALUES = {
    "yahoo": 1.2288, "goemotions": 1.8815, "newsgroups": 1.8815,
    "trec": 2.2082, "arxiv": 2.6238, "hupd_sec_cls": 2.42,
    "dbpedia_classes": 3.17, "clinc": 3.9069,
    "hwv_l0_l2": 4.09, "hupd_sec_sub": 4.44, "hwv_l0_l3": 4.59,
    "wos": 5.05,
}


def compute_steer(pa):
    return (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
           (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))


def load_dataset_results(ds_name):
    """Load steerability results for a dataset."""
    f = RESULTS_DIR / f"benchmark_bge-small_{ds_name}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    v5_steers, mrl_steers, base_l1s = [], [], []
    for sk in d.get('v5', {}):
        entry_v5 = d['v5'][sk]
        entry_mrl = d.get('mrl', {}).get(sk, {})
        if not isinstance(entry_v5, dict) or 'prefix_accuracy' not in entry_v5:
            continue
        if not isinstance(entry_mrl, dict) or 'prefix_accuracy' not in entry_mrl:
            continue
        v5_steers.append(compute_steer(entry_v5['prefix_accuracy']))
        mrl_steers.append(compute_steer(entry_mrl['prefix_accuracy']))
        base_l1s.append(entry_v5.get('baseline', {}).get('l1_accuracy', 0))

    if not v5_steers:
        return None

    return {
        'name': ds_name,
        'n_seeds': len(v5_steers),
        'v5_steer_mean': float(np.mean(v5_steers)),
        'v5_steer_std': float(np.std(v5_steers, ddof=1)) if len(v5_steers) > 1 else 0,
        'mrl_steer_mean': float(np.mean(mrl_steers)),
        'mrl_steer_std': float(np.std(mrl_steers, ddof=1)) if len(mrl_steers) > 1 else 0,
        'gap_mean': float(np.mean(np.array(v5_steers) - np.array(mrl_steers))),
        'base_l1': float(np.mean(base_l1s)),
        'H': H_VALUES.get(ds_name, 0),
        'product': float(np.mean(base_l1s)) * H_VALUES.get(ds_name, 0),
        'v5_steers': v5_steers,
        'mrl_steers': mrl_steers,
    }


def main():
    print("=" * 80)
    print("  PRE-REGISTERED PREDICTION VALIDATION")
    print("=" * 80)

    # Load pre-registered predictions
    pred_file = RESULTS_DIR / "deep_hierarchy_predictions.json"
    if not pred_file.exists():
        print("  ERROR: No prediction file found!")
        return
    predictions = json.load(open(pred_file))
    print(f"\n  Predictions timestamp: {predictions['timestamp']}")
    print(f"  Calibration: {predictions['calibration']['method']}")
    print(f"  Fit: {predictions['calibration']['fit']}")

    # === 1. Load calibration data ===
    print(f"\n{'=' * 80}")
    print("  CALIBRATION DATA (8 original datasets)")
    print(f"{'=' * 80}")

    cal_data = []
    for ds in CALIBRATION_DATASETS:
        r = load_dataset_results(ds)
        if r:
            cal_data.append(r)
            print(f"  {ds:15s}: H={r['H']:.2f}, base_L1={r['base_l1']:.3f}, "
                  f"product={r['product']:.3f}, V5_S={r['v5_steer_mean']:+.4f}")

    # Fit calibration regression: S = a * product + b
    cal_products = np.array([d['product'] for d in cal_data])
    cal_steers = np.array([d['v5_steer_mean'] for d in cal_data])
    slope, intercept, r_val, p_val, se = stats.linregress(cal_products, cal_steers)
    print(f"\n  Calibration fit: S = {slope:.4f} * product + {intercept:.4f}")
    print(f"  R^2 = {r_val**2:.3f}, p = {p_val:.4f}")

    # === 2. Load test data ===
    print(f"\n{'=' * 80}")
    print("  TEST DATA (deep hierarchy datasets)")
    print(f"{'=' * 80}")

    test_data = []
    for ds in TEST_DATASETS:
        r = load_dataset_results(ds)
        if r:
            test_data.append(r)
            predicted_s = slope * r['product'] + intercept
            residual = r['v5_steer_mean'] - predicted_s
            pct_error = abs(residual / predicted_s) * 100 if predicted_s != 0 else float('inf')
            print(f"  {ds:15s}: H={r['H']:.2f}, base_L1={r['base_l1']:.3f}, "
                  f"product={r['product']:.3f}")
            print(f"    Predicted S = {predicted_s:+.4f}")
            print(f"    Actual V5 S = {r['v5_steer_mean']:+.4f} +/- {r['v5_steer_std']:.4f} "
                  f"(n={r['n_seeds']})")
            print(f"    Residual    = {residual:+.4f} ({pct_error:.1f}% error)")
            print()
        else:
            print(f"  {ds:15s}: NOT YET AVAILABLE")

    if not test_data:
        print("\n  No test results available yet. Run experiments first.")
        # Save partial results
        output = {
            "status": "incomplete",
            "calibration_datasets": len(cal_data),
            "test_datasets_available": 0,
            "calibration_fit": {
                "slope": float(slope), "intercept": float(intercept),
                "r_squared": float(r_val**2),
            }
        }
        with open(RESULTS_DIR / "prediction_validation.json", 'w') as f:
            json.dump(output, f, indent=2)
        return

    # === 3. Prediction accuracy ===
    print(f"\n{'=' * 80}")
    print("  PREDICTION ACCURACY ANALYSIS")
    print(f"{'=' * 80}")

    all_data = cal_data + test_data
    all_products = np.array([d['product'] for d in all_data])
    all_steers = np.array([d['v5_steer_mean'] for d in all_data])
    all_names = [d['name'] for d in all_data]
    is_test = [d['name'] in TEST_DATASETS for d in all_data]

    # Out-of-sample prediction error
    test_products = np.array([d['product'] for d in test_data])
    test_steers = np.array([d['v5_steer_mean'] for d in test_data])
    test_predicted = slope * test_products + intercept
    test_residuals = test_steers - test_predicted
    rmse = np.sqrt(np.mean(test_residuals**2))
    mae = np.mean(np.abs(test_residuals))

    print(f"\n  Out-of-sample prediction error (n={len(test_data)}):")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    Mean absolute % error: {np.mean(np.abs(test_residuals / test_predicted)) * 100:.1f}%")

    # Correlation including test data
    rho_all, p_all = stats.spearmanr(all_products, all_steers)
    r_all, pr_all = stats.pearsonr(all_products, all_steers)
    print(f"\n  Combined {len(all_data)}-dataset correlation:")
    print(f"    Spearman rho = {rho_all:.3f} (p = {p_all:.6f})")
    print(f"    Pearson r    = {r_all:.3f} (p = {pr_all:.6f})")

    # Re-fit on all data
    slope_all, intercept_all, r_all2, p_all2, se_all = stats.linregress(
        all_products, all_steers)
    print(f"\n  Updated fit (all {len(all_data)} datasets):")
    print(f"    S = {slope_all:.4f} * product + {intercept_all:.4f}")
    print(f"    R^2 = {r_all2**2:.3f}")

    # === 4. Key testable predictions from pre-registration ===
    print(f"\n{'=' * 80}")
    print("  TESTABLE PREDICTIONS CHECK")
    print(f"{'=' * 80}")

    results_check = {}

    # Prediction 1: hupd_sec_sub > hupd_sec_cls
    sec_cls = next((d for d in all_data if d['name'] == 'hupd_sec_cls'), None)
    sec_sub = next((d for d in all_data if d['name'] == 'hupd_sec_sub'), None)
    if sec_cls and sec_sub:
        pred1 = sec_sub['v5_steer_mean'] > sec_cls['v5_steer_mean']
        print(f"\n  P1: hupd_sec_sub S > hupd_sec_cls S")
        print(f"      sec_sub S = {sec_sub['v5_steer_mean']:+.4f}")
        print(f"      sec_cls S = {sec_cls['v5_steer_mean']:+.4f}")
        print(f"      RESULT: {'CONFIRMED' if pred1 else 'REJECTED'}")
        results_check['sec_sub_gt_sec_cls'] = pred1
    else:
        print(f"\n  P1: CANNOT TEST (missing data)")

    # Prediction 2: Ranking follows product predictor
    print(f"\n  P2: Ranking follows product predictor ordering")
    for d in sorted(all_data, key=lambda x: x['product']):
        marker = " [TEST]" if d['name'] in TEST_DATASETS else ""
        print(f"      product={d['product']:.3f} -> S={d['v5_steer_mean']:+.4f}  "
              f"({d['name']}){marker}")

    # Prediction 3: All V5 > MRL (sign test)
    v5_wins = sum(1 for d in all_data if d['v5_steer_mean'] > d['mrl_steer_mean'])
    n_total = len(all_data)
    sign_p = stats.binomtest(v5_wins, n_total, 0.5, alternative='greater').pvalue
    print(f"\n  P3: All V5 > MRL (sign test)")
    print(f"      {v5_wins}/{n_total} V5 > MRL, p = {sign_p:.6f}")
    results_check['sign_test_v5_wins'] = v5_wins
    results_check['sign_test_total'] = n_total
    results_check['sign_test_p'] = float(sign_p)

    # Prediction 4: Effect sizes correlate with product (rho > 0.7)
    products_arr = np.array([d['product'] for d in all_data])
    steers_arr = np.array([d['v5_steer_mean'] for d in all_data])
    rho_test, p_test = stats.spearmanr(products_arr, steers_arr)
    pred4 = rho_test > 0.7
    print(f"\n  P4: Effect sizes correlate with product (rho > 0.7)")
    print(f"      Spearman rho = {rho_test:.3f} (p = {p_test:.6f})")
    print(f"      RESULT: {'CONFIRMED' if pred4 else 'REJECTED'}")
    results_check['rho_gt_07'] = pred4
    results_check['rho_value'] = float(rho_test)

    # === 5. Pre-registered point prediction check (hupd_sec_sub) ===
    if sec_sub and sec_sub['n_seeds'] >= 5:
        print(f"\n{'=' * 80}")
        print("  POINT PREDICTION CHECK: hupd_sec_sub")
        print(f"{'=' * 80}")

        predicted_s = predictions['predictions']['hupd_sec_sub']['predicted_S']
        actual_s = sec_sub['v5_steer_mean']
        actual_se = sec_sub['v5_steer_std'] / np.sqrt(sec_sub['n_seeds'])

        # Is predicted within 1 SE of actual?
        within_1se = abs(predicted_s - actual_s) <= actual_se
        # Is predicted within 95% CI?
        ci_lo = actual_s - 1.96 * actual_se
        ci_hi = actual_s + 1.96 * actual_se
        within_ci = ci_lo <= predicted_s <= ci_hi

        print(f"  Pre-registered predicted S: {predicted_s:+.4f}")
        print(f"  Actual S (5-seed mean):     {actual_s:+.4f} +/- {sec_sub['v5_steer_std']:.4f}")
        print(f"  SE of mean:                 {actual_se:.4f}")
        print(f"  95% CI:                     [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  Prediction within 1 SE:     {'YES' if within_1se else 'NO'}")
        print(f"  Prediction within 95% CI:   {'YES' if within_ci else 'NO'}")
        print(f"  Absolute error:             {abs(predicted_s - actual_s):.4f}")
        print(f"  Relative error:             {abs(predicted_s - actual_s) / abs(actual_s) * 100:.1f}%")

        results_check['point_prediction'] = {
            'predicted': float(predicted_s),
            'actual': float(actual_s),
            'actual_se': float(actual_se),
            'within_1se': bool(within_1se),
            'within_ci': bool(within_ci),
            'absolute_error': float(abs(predicted_s - actual_s)),
            'relative_error': float(abs(predicted_s - actual_s) / abs(actual_s) * 100),
        }

    # === 6. HWV conditional predictions ===
    for hwv_name in ['hwv_l0_l2', 'hwv_l0_l3']:
        hwv = next((d for d in test_data if d['name'] == hwv_name), None)
        if hwv and hwv_name in predictions['predictions']:
            pred = predictions['predictions'][hwv_name]
            pred_range = pred.get('predicted_S_range', [None, None])
            if pred_range[0] is not None:
                in_range = pred_range[0] <= hwv['v5_steer_mean'] <= pred_range[1]
                print(f"\n  {hwv_name}: predicted S in [{pred_range[0]:.3f}, {pred_range[1]:.3f}]")
                print(f"    Actual S = {hwv['v5_steer_mean']:+.4f}")
                print(f"    In range: {'YES' if in_range else 'NO'}")
                results_check[f'{hwv_name}_in_range'] = in_range

    # === Save results ===
    output = {
        "status": "complete" if len(test_data) >= 3 else "partial",
        "n_calibration": len(cal_data),
        "n_test": len(test_data),
        "calibration_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_val**2),
        },
        "combined_fit": {
            "slope": float(slope_all),
            "intercept": float(intercept_all),
            "r_squared": float(r_all2**2),
        },
        "out_of_sample": {
            "rmse": float(rmse),
            "mae": float(mae),
            "n_test": len(test_data),
        },
        "combined_correlation": {
            "spearman_rho": float(rho_all),
            "spearman_p": float(p_all),
            "pearson_r": float(r_all),
            "pearson_p": float(pr_all),
        },
        "predictions_check": results_check,
        "per_dataset": {
            d['name']: {
                'H': d['H'],
                'base_l1': d['base_l1'],
                'product': d['product'],
                'v5_steer': d['v5_steer_mean'],
                'v5_steer_std': d['v5_steer_std'],
                'mrl_steer': d['mrl_steer_mean'],
                'n_seeds': d['n_seeds'],
                'is_test': d['name'] in TEST_DATASETS,
            }
            for d in all_data
        },
    }

    out_path = RESULTS_DIR / "prediction_validation.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(f"\n  Results saved to {out_path}")

    # Generate figure
    try:
        generate_figure(cal_data, test_data, slope, intercept, slope_all, intercept_all)
    except Exception as e:
        print(f"  Figure generation failed: {e}")


def generate_figure(cal_data, test_data, cal_slope, cal_intercept, all_slope, all_intercept):
    """Generate predicted vs actual steerability figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: Product predictor with calibration + test ---
    ax = axes[0]

    cal_products = [d['product'] for d in cal_data]
    cal_steers = [d['v5_steer_mean'] for d in cal_data]
    test_products = [d['product'] for d in test_data]
    test_steers = [d['v5_steer_mean'] for d in test_data]

    # Calibration regression line
    x_range = np.linspace(0, max(cal_products + test_products) * 1.1, 100)
    y_cal = cal_slope * x_range + cal_intercept

    ax.plot(x_range, y_cal, 'b--', alpha=0.5, label='Calibration fit (8 datasets)')

    # Calibration points
    ax.scatter(cal_products, cal_steers, c='steelblue', s=60, zorder=5,
               label=f'Calibration (n={len(cal_data)})', edgecolors='navy', linewidths=0.5)

    # Test points (stars)
    ax.scatter(test_products, test_steers, c='crimson', s=120, marker='*', zorder=6,
               label=f'Pre-registered test (n={len(test_data)})', edgecolors='darkred', linewidths=0.5)

    # Add error bars for test data
    for d in test_data:
        if d['n_seeds'] > 1:
            se = d['v5_steer_std'] / np.sqrt(d['n_seeds'])
            ax.errorbar(d['product'], d['v5_steer_mean'], yerr=1.96*se,
                        color='crimson', capsize=3, capthick=1, linewidth=1, zorder=4)

    # Labels
    name_map = {
        'yahoo': 'Yahoo', 'goemotions': 'GoEmo', 'newsgroups': 'News',
        'trec': 'TREC', 'arxiv': 'arXiv', 'dbpedia_classes': 'DBPedia',
        'clinc': 'CLINC', 'wos': 'WOS', 'hupd_sec_cls': 'HUPD-cls',
        'hupd_sec_sub': 'HUPD-sub', 'hwv_l0_l2': 'HWV-L2', 'hwv_l0_l3': 'HWV-L3',
    }
    for d in cal_data + test_data:
        label = name_map.get(d['name'], d['name'])
        offset_y = 0.003 if d['name'] != 'wos' else -0.006
        ax.annotate(label, (d['product'], d['v5_steer_mean']),
                    textcoords="offset points", xytext=(5, 8 if offset_y > 0 else -12),
                    fontsize=7, color='navy' if d['name'] not in TEST_DATASETS else 'darkred')

    ax.set_xlabel('Product predictor: $H(L_1|L_0) \\times$ base $L_1$ accuracy', fontsize=10)
    ax.set_ylabel('V5 steerability $S$', fontsize=10)
    ax.set_title('(a) Pre-registered prediction test', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Panel B: Predicted vs Actual scatter ---
    ax = axes[1]

    all_data = cal_data + test_data
    all_predicted = [cal_slope * d['product'] + cal_intercept for d in all_data]
    all_actual = [d['v5_steer_mean'] for d in all_data]

    # Perfect prediction line
    min_val = min(min(all_predicted), min(all_actual)) - 0.01
    max_val = max(max(all_predicted), max(all_actual)) + 0.01
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect prediction')

    # Calibration
    for d in cal_data:
        pred = cal_slope * d['product'] + cal_intercept
        ax.scatter(pred, d['v5_steer_mean'], c='steelblue', s=60, zorder=5,
                   edgecolors='navy', linewidths=0.5)

    # Test (stars)
    for d in test_data:
        pred = cal_slope * d['product'] + cal_intercept
        ax.scatter(pred, d['v5_steer_mean'], c='crimson', s=120, marker='*', zorder=6,
                   edgecolors='darkred', linewidths=0.5)
        label = name_map.get(d['name'], d['name'])
        ax.annotate(label, (pred, d['v5_steer_mean']),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=8, color='darkred', fontweight='bold')

    ax.set_xlabel('Predicted steerability $\\hat{S}$', fontsize=10)
    ax.set_ylabel('Actual steerability $S$', fontsize=10)
    ax.set_title('(b) Predicted vs. actual', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add RMSE annotation
    test_residuals = [d['v5_steer_mean'] - (cal_slope * d['product'] + cal_intercept)
                      for d in test_data]
    rmse = np.sqrt(np.mean(np.array(test_residuals)**2))
    ax.text(0.05, 0.95, f'Out-of-sample RMSE: {rmse:.4f}\nTest points: {len(test_data)}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig_dir = RESULTS_DIR / "figures" / "paper"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path_pdf = fig_dir / "fig_prediction_validation.pdf"
    fig_path_png = RESULTS_DIR / "figures" / "fig_prediction_validation.png"
    plt.savefig(fig_path_pdf, dpi=200, bbox_inches='tight')
    plt.savefig(fig_path_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved to {fig_path_pdf}")
    print(f"  Figure saved to {fig_path_png}")


if __name__ == "__main__":
    main()
