#!/usr/bin/env python -u
"""
COMPREHENSIVE REAL-NETWORK UNIVERSALITY ANALYSIS
=================================================
PURPOSE:
  Pool ALL kappa_near_cache_*.json files and test the universal law:
    logit(q) = alpha * kappa_nearest + beta * log(K-1) + C_0

  across 19 diverse architectures (transformers, SSMs, hybrids, encoder-only)
  and 10 datasets spanning K=4 to K=77.

PRE-REGISTERED (Feb 23, 2026):
  From kappa_nearest_extended.json (7 arch LOAO):
    alpha = 3.07, alpha_CV = 3.3% PASS (<25%)
    beta  = -0.72, beta_CV = 2.8% PASS (<25%)
  Extended test adds 12 more architectures and 7 more datasets.

  PR1: Global R2 > 0.60 (already achieved: 0.627 with 192 pts)
  PR2: LOAO alpha_CV < 25% (already: 3.3% with 7 arch)
  PR3: LODO alpha_CV < 30% (cross-dataset)
  PR4: Per-K trend: logit(q) + log(K-1) ~ kappa_nearest (K-corrected correlation > 0.50)
  PRIMARY: PR1 + PR2 + PR3 all pass
"""

import json
import os
import glob
import numpy as np
from scipy import stats

RESULT_PATH = "results/cti_kappa_universal_comprehensive.json"

def load_all_data():
    """Load all kappa_near_cache_*.json files."""
    files = glob.glob("results/kappa_near_cache_*.json")
    all_points = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        for pt in data:
            # Normalize fields
            model = pt.get('model', '')
            dataset = pt.get('dataset', '')
            layer = pt.get('layer', -1)
            K = pt.get('K', -1)
            q = pt.get('q', float('nan'))
            kappa = pt.get('kappa_nearest', float('nan'))
            logit_q = pt.get('logit_q', float('nan'))

            # Compute logKm1
            logKm1 = float(np.log(K - 1)) if K > 1 else float('nan')

            # Validate
            if (np.isnan(q) or np.isnan(kappa) or np.isnan(logit_q)
                    or np.isnan(logKm1) or K < 2 or kappa <= 0
                    or q <= 0 or q >= 1):
                continue

            # Architecture family
            family = classify_family(model)

            all_points.append({
                'model': model,
                'dataset': dataset,
                'layer': layer,
                'K': K,
                'q': float(q),
                'kappa': float(kappa),
                'logit_q': float(logit_q),
                'logKm1': float(logKm1),
                'family': family,
            })
    return all_points


def classify_family(model):
    """Classify model into architecture family."""
    m = model.lower()
    if 'pythia' in m:
        return 'pythia'
    elif 'gpt-neo' in m or 'gpt_neo' in m:
        return 'gpt_neo'
    elif 'gpt2' in m:
        return 'gpt2'
    elif 'qwen3' in m:
        return 'qwen3'
    elif 'qwen2' in m:
        return 'qwen2'
    elif 'falcon' in m:
        return 'falcon'
    elif 'mamba' in m:
        return 'mamba'
    elif 'rwkv' in m:
        return 'rwkv'
    elif 'tinyllama' in m or 'llama' in m:
        return 'llama'
    elif 'olmo' in m:
        return 'olmo'
    elif 'mistral' in m:
        return 'mistral'
    elif 'bert' in m:
        return 'bert'
    elif 'bge' in m:
        return 'bge'
    elif 'deberta' in m:
        return 'deberta'
    elif 'electra' in m:
        return 'electra'
    elif 'phi' in m:
        return 'phi'
    else:
        return 'other'


def ols_fit(kappas, logKm1s, logit_qs):
    """Fit logit_q = alpha * kappa + beta * logKm1 + C using OLS.
    Returns (alpha, beta, C, r2).
    """
    X = np.column_stack([kappas, logKm1s, np.ones(len(kappas))])
    y = np.array(logit_qs)
    try:
        result = np.linalg.lstsq(X, y, rcond=None)
        coeffs = result[0]
        alpha, beta, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        y_pred = X @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        mae = float(np.mean(np.abs(y - y_pred)))
        return alpha, beta, C, r2, mae
    except Exception:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')


def loao_cv(points):
    """Leave-one-architecture-out CV on alpha and beta."""
    models = sorted(set(p['model'] for p in points))
    alphas, betas = [], []
    per_model = {}
    for m in models:
        train = [p for p in points if p['model'] != m]
        test = [p for p in points if p['model'] == m]
        if len(train) < 10 or len(test) < 2:
            continue
        a, b, c, r2, mae = ols_fit(
            [p['kappa'] for p in train],
            [p['logKm1'] for p in train],
            [p['logit_q'] for p in train]
        )
        if np.isnan(a):
            continue
        alphas.append(a)
        betas.append(b)
        # Predict on test set
        y_pred_test = [a * p['kappa'] + b * p['logKm1'] + c for p in test]
        y_true_test = [p['logit_q'] for p in test]
        test_mae = float(np.mean(np.abs(np.array(y_true_test) - np.array(y_pred_test))))
        per_model[m] = {
            'alpha': float(a), 'beta': float(b), 'C': float(c),
            'train_r2': float(r2), 'test_mae': float(test_mae),
            'n_train': len(train), 'n_test': len(test),
        }
    if len(alphas) < 2:
        return float('nan'), float('nan'), float('nan'), float('nan'), per_model
    alpha_arr = np.array(alphas)
    beta_arr = np.array(betas)
    alpha_cv = float(np.std(alpha_arr) / np.abs(np.mean(alpha_arr)))
    beta_cv = float(np.std(beta_arr) / np.abs(np.mean(beta_arr)))
    return float(np.mean(alpha_arr)), float(np.mean(beta_arr)), alpha_cv, beta_cv, per_model


def lodo_cv(points):
    """Leave-one-dataset-out CV on alpha and beta."""
    datasets = sorted(set(p['dataset'] for p in points))
    alphas, betas = [], []
    per_dataset = {}
    for ds in datasets:
        train = [p for p in points if p['dataset'] != ds]
        test = [p for p in points if p['dataset'] == ds]
        if len(train) < 10 or len(test) < 2:
            continue
        a, b, c, r2, mae = ols_fit(
            [p['kappa'] for p in train],
            [p['logKm1'] for p in train],
            [p['logit_q'] for p in train]
        )
        if np.isnan(a):
            continue
        alphas.append(a)
        betas.append(b)
        y_pred_test = [a * p['kappa'] + b * p['logKm1'] + c for p in test]
        y_true_test = [p['logit_q'] for p in test]
        test_mae = float(np.mean(np.abs(np.array(y_true_test) - np.array(y_pred_test))))
        per_dataset[ds] = {
            'alpha': float(a), 'beta': float(b), 'C': float(c),
            'train_r2': float(r2), 'test_mae': float(test_mae),
            'n_train': len(train), 'n_test': len(test),
        }
    if len(alphas) < 2:
        return float('nan'), float('nan'), float('nan'), float('nan'), per_dataset
    alpha_arr = np.array(alphas)
    beta_arr = np.array(betas)
    alpha_cv = float(np.std(alpha_arr) / np.abs(np.mean(alpha_arr)))
    beta_cv = float(np.std(beta_arr) / np.abs(np.mean(beta_arr)))
    return float(np.mean(alpha_arr)), float(np.mean(beta_arr)), alpha_cv, beta_cv, per_dataset


def per_k_analysis(points):
    """For each K value, test if K-corrected logit correlates with kappa."""
    k_groups = {}
    for p in points:
        k = p['K']
        if k not in k_groups:
            k_groups[k] = []
        k_groups[k].append(p)

    per_k = {}
    for K, pts in sorted(k_groups.items()):
        logit_adj = [p['logit_q'] + p['logKm1'] for p in pts]  # logit + log(K-1)
        kappas = [p['kappa'] for p in pts]
        if len(pts) < 5:
            continue
        r, pval = stats.pearsonr(kappas, logit_adj)
        a, _, _, r2, mae = ols_fit(
            kappas,
            [0.0] * len(pts),  # beta=0 (no logKm1 term, already corrected)
            logit_adj
        )
        per_k[K] = {
            'K': K,
            'n_points': len(pts),
            'r': float(r),
            'p_value': float(pval),
            'alpha_k': float(a),
            'r2': float(r2),
            'mae': float(mae),
        }
    return per_k


def main():
    print("Loading all kappa_near_cache files...", flush=True)
    points = load_all_data()
    print(f"Total valid points: {len(points)}", flush=True)

    models = sorted(set(p['model'] for p in points))
    datasets = sorted(set(p['dataset'] for p in points))
    k_values = sorted(set(p['K'] for p in points))
    families = sorted(set(p['family'] for p in points))

    print(f"Models ({len(models)}): {models}", flush=True)
    print(f"Datasets ({len(datasets)}): {datasets}", flush=True)
    print(f"K values: {k_values}", flush=True)
    print(f"Families ({len(families)}): {families}", flush=True)
    print()

    # Global fit
    print("Global OLS fit...", flush=True)
    alpha_g, beta_g, C_g, r2_g, mae_g = ols_fit(
        [p['kappa'] for p in points],
        [p['logKm1'] for p in points],
        [p['logit_q'] for p in points]
    )
    print(f"  alpha={alpha_g:.4f}, beta={beta_g:.4f}, C={C_g:.4f}", flush=True)
    print(f"  R2={r2_g:.4f}, MAE={mae_g:.4f}", flush=True)
    print()

    # K-corrected global (beta forced to -1)
    print("K-corrected fit (logit + log(K-1) = alpha * kappa + C)...", flush=True)
    logit_adj = np.array([p['logit_q'] + p['logKm1'] for p in points])
    kappas = np.array([p['kappa'] for p in points])
    # Simple OLS with just kappa
    X = np.column_stack([kappas, np.ones(len(kappas))])
    res = np.linalg.lstsq(X, logit_adj, rcond=None)
    alpha_corr = float(res[0][0])
    C_corr = float(res[0][1])
    y_pred_corr = X @ res[0]
    ss_res = float(np.sum((logit_adj - y_pred_corr) ** 2))
    ss_tot = float(np.sum((logit_adj - logit_adj.mean()) ** 2))
    r2_corr = float(1 - ss_res / ss_tot)
    r_corr, _ = stats.pearsonr(kappas, logit_adj)
    print(f"  alpha_corr={alpha_corr:.4f}, C_corr={C_corr:.4f}", flush=True)
    print(f"  R2_corr={r2_corr:.4f}, r_corr={r_corr:.4f}", flush=True)
    print()

    # LOAO
    print("LOAO (leave-one-architecture-out)...", flush=True)
    loao_alpha_mean, loao_beta_mean, loao_alpha_cv, loao_beta_cv, loao_per_model = loao_cv(points)
    print(f"  alpha_mean={loao_alpha_mean:.4f}, alpha_CV={loao_alpha_cv*100:.2f}%", flush=True)
    print(f"  beta_mean={loao_beta_mean:.4f}, beta_CV={loao_beta_cv*100:.2f}%", flush=True)
    pr2_pass = loao_alpha_cv < 0.25
    print(f"  PR2 (alpha_CV < 25%): {'PASS' if pr2_pass else 'FAIL'}", flush=True)
    print()

    # LODO
    print("LODO (leave-one-dataset-out)...", flush=True)
    lodo_alpha_mean, lodo_beta_mean, lodo_alpha_cv, lodo_beta_cv, lodo_per_dataset = lodo_cv(points)
    print(f"  alpha_mean={lodo_alpha_mean:.4f}, alpha_CV={lodo_alpha_cv*100:.2f}%", flush=True)
    print(f"  beta_mean={lodo_beta_mean:.4f}, beta_CV={lodo_beta_cv*100:.2f}%", flush=True)
    pr3_pass = lodo_alpha_cv < 0.30
    print(f"  PR3 (alpha_CV < 30%): {'PASS' if pr3_pass else 'FAIL'}", flush=True)
    print()

    # Per-K analysis
    print("Per-K analysis...", flush=True)
    per_k = per_k_analysis(points)
    for K, kres in per_k.items():
        print(f"  K={K:3d}: n={kres['n_points']:3d}, r={kres['r']:.3f}, "
              f"alpha_k={kres['alpha_k']:.3f}, R2={kres['r2']:.3f}", flush=True)
    k_corrs = [v['r'] for v in per_k.values() if v['n_points'] >= 5]
    median_corr = float(np.median(k_corrs)) if k_corrs else float('nan')
    pr4_pass = median_corr > 0.50
    print(f"  Median per-K r={median_corr:.3f}", flush=True)
    print(f"  PR4 (median r > 0.50): {'PASS' if pr4_pass else 'FAIL'}", flush=True)
    print()

    # Summary
    pr1_pass = r2_g > 0.60
    overall_pass = pr1_pass and pr2_pass and pr3_pass
    print("=" * 60, flush=True)
    print("VERDICT SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"PR1 (Global R2 > 0.60): {'PASS' if pr1_pass else 'FAIL'}  R2={r2_g:.4f}", flush=True)
    print(f"PR2 (LOAO alpha_CV < 25%): {'PASS' if pr2_pass else 'FAIL'}  CV={loao_alpha_cv*100:.2f}%", flush=True)
    print(f"PR3 (LODO alpha_CV < 30%): {'PASS' if pr3_pass else 'FAIL'}  CV={lodo_alpha_cv*100:.2f}%", flush=True)
    print(f"PR4 (Median per-K r > 0.50): {'PASS' if pr4_pass else 'FAIL'}  r={median_corr:.3f}", flush=True)
    print(f"PRIMARY (PR1+PR2+PR3): {'PASS' if overall_pass else 'FAIL'}", flush=True)
    print()

    # Save results
    result = {
        "experiment": "comprehensive_real_network_universality",
        "n_total_points": len(points),
        "n_models": len(models),
        "n_datasets": len(datasets),
        "models": models,
        "datasets": datasets,
        "k_values": [int(k) for k in k_values],
        "families": families,
        "global_fit": {
            "alpha": float(alpha_g),
            "beta": float(beta_g),
            "C": float(C_g),
            "r2": float(r2_g),
            "mae": float(mae_g),
        },
        "k_corrected_fit": {
            "alpha": float(alpha_corr),
            "C": float(C_corr),
            "r2": float(r2_corr),
            "r": float(r_corr),
        },
        "loao": {
            "alpha_mean": float(loao_alpha_mean),
            "alpha_cv": float(loao_alpha_cv),
            "beta_mean": float(loao_beta_mean),
            "beta_cv": float(loao_beta_cv),
            "n_architectures": len(loao_per_model),
            "per_model": loao_per_model,
        },
        "lodo": {
            "alpha_mean": float(lodo_alpha_mean),
            "alpha_cv": float(lodo_alpha_cv),
            "beta_mean": float(lodo_beta_mean),
            "beta_cv": float(lodo_beta_cv),
            "n_datasets": len(lodo_per_dataset),
            "per_dataset": lodo_per_dataset,
        },
        "per_k": {str(k): v for k, v in per_k.items()},
        "verdicts": {
            "PR1_global_r2": bool(pr1_pass),
            "PR2_loao_alpha_cv": bool(pr2_pass),
            "PR3_lodo_alpha_cv": bool(pr3_pass),
            "PR4_per_k_corr": bool(pr4_pass),
            "PRIMARY": bool(overall_pass),
        },
        "verdict": "PASS" if overall_pass else "FAIL",
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
