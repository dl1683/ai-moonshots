"""Held-Out Prediction Test: Can we predict steerability on unseen datasets?

The ultimate test of a scaling law: fit on N-1 datasets, predict the Nth.
If leave-one-out prediction error is small, the law has predictive power.

This is Codex's recommendation for demonstrating "predictive sharpness" —
one of the 4 requirements for elevating this from observation to law.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_all_steerability_data():
    """Load steerability data from all available benchmark files."""
    profiles = json.load(open(RESULTS_DIR / "hierarchy_profiles.json"))

    def compute_steer(pa):
        return (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
               (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))

    def extract_steers(filepath, method_key):
        if not filepath.exists():
            return []
        d = json.load(open(filepath))
        steers = []
        md = d.get(method_key, {})
        if isinstance(md, dict):
            for sk, sv in md.items():
                if isinstance(sv, dict) and 'prefix_accuracy' in sv:
                    steers.append(compute_steer(sv['prefix_accuracy']))
        return steers

    datasets = {}

    # CLINC (from ablation file, 5 seeds)
    abl_file = RESULTS_DIR / "ablation_steerability_bge-small_clinc.json"
    if abl_file.exists():
        ad = json.load(open(abl_file))
        clinc_steers = []
        for r in ad['results'].get('v5', []):
            if 'prefix_results' in r:
                pr = r['prefix_results']
                pa = {'j1_l0': pr['j1']['l0'], 'j1_l1': pr['j1']['l1'],
                      'j4_l0': pr['j4']['l0'], 'j4_l1': pr['j4']['l1']}
                clinc_steers.append(compute_steer(pa))
        if clinc_steers:
            datasets['clinc'] = {
                'h': profiles['clinc']['h_l1_given_l0'],
                'v5_steers': clinc_steers,
                'v5_mean': float(np.mean(clinc_steers)),
            }

    # TREC, Newsgroups, Yahoo from benchmark files
    for ds_name in ['trec', 'newsgroups', 'yahoo']:
        bench_file = RESULTS_DIR / f"benchmark_bge-small_{ds_name}.json"
        steers = extract_steers(bench_file, 'v5')
        if steers and ds_name in profiles:
            datasets[ds_name] = {
                'h': profiles[ds_name]['h_l1_given_l0'],
                'v5_steers': steers,
                'v5_mean': float(np.mean(steers)),
            }

    # New datasets (if available)
    for ds_name in ['goemotions', 'arxiv', 'dbpedia_classes', 'wos']:
        bench_file = RESULTS_DIR / f"benchmark_bge-small_{ds_name}.json"
        steers = extract_steers(bench_file, 'v5')
        if steers and ds_name in profiles:
            datasets[ds_name] = {
                'h': profiles[ds_name]['h_l1_given_l0'],
                'v5_steers': steers,
                'v5_mean': float(np.mean(steers)),
            }

    return datasets


def leave_one_out_prediction(datasets):
    """Leave-one-out cross-validation of the scaling law."""
    names = sorted(datasets.keys(), key=lambda d: datasets[d]['h'])
    n = len(names)

    print(f"\n{'='*70}")
    print(f"  LEAVE-ONE-OUT PREDICTION TEST (n={n})")
    print(f"{'='*70}")

    if n < 3:
        print("  Need at least 3 datasets for LOO prediction.")
        return None

    predictions = []
    actuals = []
    residuals = []

    print(f"\n  {'Held-out':<15} {'H(L1|L0)':<10} {'Predicted':<12} {'Actual':<12} {'Error':<10} {'RelErr'}")
    print(f"  {'-'*65}")

    for i, held_out in enumerate(names):
        # Fit on all except held_out
        train_names = [n for n in names if n != held_out]
        h_train = [datasets[n]['h'] for n in train_names]
        s_train = [datasets[n]['v5_mean'] for n in train_names]

        # Linear fit
        slope, intercept, _, _, _ = stats.linregress(h_train, s_train)

        # Predict held-out
        h_test = datasets[held_out]['h']
        s_actual = datasets[held_out]['v5_mean']
        s_predicted = slope * h_test + intercept

        error = s_predicted - s_actual
        rel_err = abs(error / s_actual) * 100 if abs(s_actual) > 0.001 else float('inf')

        predictions.append(s_predicted)
        actuals.append(s_actual)
        residuals.append(error)

        print(f"  {held_out:<15} {h_test:<10.3f} {s_predicted:<12.5f} {s_actual:<12.5f} {error:+.5f}   {rel_err:.1f}%")

    # Overall statistics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(np.array(residuals)**2))
    r2_loo = 1 - np.sum(np.array(residuals)**2) / np.sum((np.array(actuals) - np.mean(actuals))**2)

    print(f"\n  LOO Summary:")
    print(f"    MAE  = {mae:.5f}")
    print(f"    RMSE = {rmse:.5f}")
    print(f"    LOO R2 = {r2_loo:.4f}")

    # Compare to full-data fit
    h_all = [datasets[n]['h'] for n in names]
    s_all = [datasets[n]['v5_mean'] for n in names]
    _, _, r_full, _, _ = stats.linregress(h_all, s_all)

    print(f"    Full-data R2 = {r_full**2:.4f}")
    print(f"    Shrinkage = {r_full**2 - r2_loo:.4f}")

    return {
        'n_datasets': n,
        'predictions': {names[i]: float(predictions[i]) for i in range(n)},
        'actuals': {names[i]: float(actuals[i]) for i in range(n)},
        'residuals': {names[i]: float(residuals[i]) for i in range(n)},
        'mae': float(mae),
        'rmse': float(rmse),
        'loo_r2': float(r2_loo),
        'full_r2': float(r_full**2),
    }


def split_half_prediction(datasets):
    """Fit on half, predict other half (for n >= 6)."""
    names = sorted(datasets.keys(), key=lambda d: datasets[d]['h'])
    n = len(names)

    if n < 6:
        print(f"\n  Split-half test requires n>=6 datasets (have {n}). Skipping.")
        return None

    print(f"\n{'='*70}")
    print(f"  SPLIT-HALF PREDICTION TEST (n={n})")
    print(f"{'='*70}")

    # Interleave split: even indices = train, odd = test
    train_names = [names[i] for i in range(0, n, 2)]
    test_names = [names[i] for i in range(1, n, 2)]

    h_train = [datasets[n]['h'] for n in train_names]
    s_train = [datasets[n]['v5_mean'] for n in train_names]

    slope, intercept, r_train, _, _ = stats.linregress(h_train, s_train)

    print(f"\n  Train set ({len(train_names)}): {', '.join(train_names)}")
    print(f"  Test set  ({len(test_names)}):  {', '.join(test_names)}")
    print(f"  Fit: Steer = {slope:.5f} * H + {intercept:.5f} (R2={r_train**2:.4f})")

    print(f"\n  {'Test Dataset':<15} {'H':<10} {'Predicted':<12} {'Actual':<12} {'Error'}")
    print(f"  {'-'*55}")

    residuals = []
    for tn in test_names:
        h = datasets[tn]['h']
        actual = datasets[tn]['v5_mean']
        predicted = slope * h + intercept
        error = predicted - actual
        residuals.append(error)
        print(f"  {tn:<15} {h:<10.3f} {predicted:<12.5f} {actual:<12.5f} {error:+.5f}")

    rmse = np.sqrt(np.mean(np.array(residuals)**2))
    print(f"\n  Test RMSE = {rmse:.5f}")

    return {'train': train_names, 'test': test_names, 'rmse': float(rmse)}


def main():
    datasets = load_all_steerability_data()
    print(f"Loaded {len(datasets)} datasets with steerability data:")
    for name in sorted(datasets.keys(), key=lambda d: datasets[d]['h']):
        d = datasets[name]
        print(f"  {name:<15} H={d['h']:.3f}  V5 steer={d['v5_mean']:+.4f} ({len(d['v5_steers'])} seeds)")

    loo_results = leave_one_out_prediction(datasets)
    split_results = split_half_prediction(datasets)

    # Save
    output = {
        'n_datasets': len(datasets),
        'datasets': {k: {'h': v['h'], 'v5_mean': v['v5_mean'], 'n_seeds': len(v['v5_steers'])}
                     for k, v in datasets.items()},
        'leave_one_out': loo_results,
        'split_half': split_results,
    }

    out_path = RESULTS_DIR / "held_out_prediction_test.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
