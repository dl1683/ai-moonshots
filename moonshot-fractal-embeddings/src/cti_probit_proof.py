#!/usr/bin/env python
"""
PROBIT PROOF: Rigorous numerical verification of the probit theorem.

We derive and verify:
    q = Phi(mu_M / sigma_M)

where:
    mu_M = c1 * kappa * d - c2 * sqrt(d * log(K))
    sigma_M = c3 * sqrt(d)

and Phi is the standard normal CDF.

The key prediction: logit(q) = a * kappa * d / sqrt(d * f(K)) = a * kappa * sqrt(d) / sqrt(f(K))
where f(K) depends on the correlation structure.

This script:
1. Computes exact margin statistics (mu_M, sigma_M) from Monte Carlo
2. Verifies that q = Phi(mu_M/sigma_M) matches MC accuracy
3. Identifies the exact K-dependence of mu_M and sigma_M
4. Tests the closed-form prediction vs. free fits
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import curve_fit, minimize
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def generate_simplex_centroids(K, d, Delta):
    """Generate K equidistant centroids."""
    centroids = np.random.randn(K, d)
    centroids -= centroids.mean(0)
    norms = np.sqrt(np.sum(centroids ** 2, axis=1, keepdims=True))
    centroids = centroids / norms * Delta
    centroids -= centroids.mean(0)
    actual_dist = np.sqrt(np.mean(np.sum(centroids ** 2, axis=1)))
    if actual_dist > 1e-10:
        centroids *= Delta / actual_dist
    return centroids


def compute_margin_stats(K, d, kappa, n_per_class=100, n_trials=5000):
    """Compute margin distribution statistics and accuracy."""
    sigma = 1.0
    Delta = np.sqrt(kappa * d * sigma ** 2)
    centroids = generate_simplex_centroids(K, d, Delta)

    margins = []
    correct = 0

    for trial in range(n_trials):
        k = trial % K
        x = centroids[k] + sigma * np.random.randn(d)

        # Compute distances to nearest training point in each class
        D_same = float("inf")
        D_diff_min = float("inf")

        for j in range(K):
            X_j = centroids[j] + sigma * np.random.randn(n_per_class, d)
            dists_j = np.sum((X_j - x[None, :]) ** 2, axis=1)
            min_dist_j = np.min(dists_j)

            if j == k:
                D_same = min_dist_j
            else:
                D_diff_min = min(D_diff_min, min_dist_j)

        margin = D_diff_min - D_same
        margins.append(margin)
        if margin > 0:
            correct += 1

    margins = np.array(margins)
    acc = correct / n_trials
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)

    return {
        "mu_M": float(np.mean(margins)),
        "sigma_M": float(np.std(margins)),
        "q_mc": float(q),
        "q_probit": float(norm.cdf(np.mean(margins) / max(np.std(margins), 1e-10))),
        "accuracy": float(acc),
    }


def main():
    print("=" * 70)
    print("PROBIT PROOF: Numerical verification of closed-form prediction")
    print("=" * 70)

    # ================================================================
    # PHASE 1: mu_M and sigma_M as functions of kappa, d, K
    # ================================================================
    print("\nPHASE 1: Margin statistics vs. kappa (fixed d, K)")
    print("-" * 70)

    d = 128
    K = 50
    n_per = 50
    n_trials = 3000
    kappa_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

    kappa_results = []
    for kappa in kappa_values:
        stats = compute_margin_stats(K, d, kappa, n_per, n_trials)
        stats["kappa"] = kappa
        stats["d"] = d
        stats["K"] = K
        kappa_results.append(stats)
        print(f"  kappa={kappa:.3f}: mu_M={stats['mu_M']:>8.2f}, sigma_M={stats['sigma_M']:>7.2f}, "
              f"q_mc={stats['q_mc']:.4f}, q_probit={stats['q_probit']:.4f}")

    # Check: is mu_M linear in kappa?
    kappas_arr = np.array([r["kappa"] for r in kappa_results])
    mus = np.array([r["mu_M"] for r in kappa_results])
    sigmas = np.array([r["sigma_M"] for r in kappa_results])
    q_mc = np.array([r["q_mc"] for r in kappa_results])
    q_probit = np.array([r["q_probit"] for r in kappa_results])

    coef = np.polyfit(kappas_arr, mus, 1)
    r2_linear = 1 - np.sum((mus - np.polyval(coef, kappas_arr))**2) / np.sum((mus - mus.mean())**2)
    print(f"\n  mu_M = {coef[0]:.2f} * kappa + {coef[1]:.2f} (R^2={r2_linear:.6f})")
    print(f"  Theory predicts slope ~ 2*d*K/(K-1) = {2*d*K/(K-1):.1f}")

    # Check: does q_probit match q_mc?
    mae_probit = np.mean(np.abs(q_mc - q_probit))
    r_pq = np.corrcoef(q_mc, q_probit)[0, 1]
    print(f"\n  q_probit vs q_mc: MAE={mae_probit:.4f}, r={r_pq:.6f}")

    # ================================================================
    # PHASE 2: K-dependence of mu_M and sigma_M
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Margin statistics vs. K (fixed kappa, d)")
    print("-" * 70)

    kappa_fixed = 0.2
    d_fixed = 128
    K_values = [2, 5, 10, 20, 50, 100]

    K_results = []
    for K_test in K_values:
        stats = compute_margin_stats(K_test, d_fixed, kappa_fixed, n_per, min(n_trials, K_test * 50))
        stats["kappa"] = kappa_fixed
        stats["d"] = d_fixed
        stats["K"] = K_test
        K_results.append(stats)
        print(f"  K={K_test:>4}: mu_M={stats['mu_M']:>8.2f}, sigma_M={stats['sigma_M']:>7.2f}, "
              f"q_mc={stats['q_mc']:.4f}, q_probit={stats['q_probit']:.4f}")

    Ks = np.array([r["K"] for r in K_results])
    mus_K = np.array([r["mu_M"] for r in K_results])
    sigmas_K = np.array([r["sigma_M"] for r in K_results])

    # Fit mu_M ~ K^alpha
    log_K = np.log(Ks)
    log_mu = np.log(np.maximum(mus_K, 0.1))
    coef_mu = np.polyfit(log_K, log_mu, 1)
    alpha_mu = coef_mu[0]

    log_sigma = np.log(sigmas_K)
    coef_sigma = np.polyfit(log_K, log_sigma, 1)
    alpha_sigma = coef_sigma[0]

    print(f"\n  mu_M ~ K^{alpha_mu:.3f}")
    print(f"  sigma_M ~ K^{alpha_sigma:.3f}")
    print(f"  SNR ~ K^{alpha_mu - alpha_sigma:.3f}")

    # ================================================================
    # PHASE 3: d-dependence of mu_M and sigma_M
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Margin statistics vs. d (fixed kappa, K)")
    print("-" * 70)

    kappa_fixed2 = 0.1
    K_fixed2 = 50
    d_values = [64, 128, 256, 512]

    d_results = []
    for d_test in d_values:
        stats = compute_margin_stats(K_fixed2, d_test, kappa_fixed2, n_per, n_trials)
        stats["kappa"] = kappa_fixed2
        stats["d"] = d_test
        stats["K"] = K_fixed2
        d_results.append(stats)
        print(f"  d={d_test:>4}: mu_M={stats['mu_M']:>8.2f}, sigma_M={stats['sigma_M']:>7.2f}, "
              f"q_mc={stats['q_mc']:.4f}, q_probit={stats['q_probit']:.4f}")

    ds_arr = np.array([r["d"] for r in d_results])
    mus_d = np.array([r["mu_M"] for r in d_results])
    sigmas_d = np.array([r["sigma_M"] for r in d_results])

    log_d = np.log(ds_arr)
    coef_mu_d = np.polyfit(log_d, np.log(np.maximum(mus_d, 0.1)), 1)
    coef_sig_d = np.polyfit(log_d, np.log(sigmas_d), 1)

    print(f"\n  mu_M ~ d^{coef_mu_d[0]:.3f} (theory predicts ~1.0)")
    print(f"  sigma_M ~ d^{coef_sig_d[0]:.3f} (theory predicts ~0.5)")
    print(f"  SNR = mu_M/sigma_M ~ d^{coef_mu_d[0] - coef_sig_d[0]:.3f} "
          f"(theory: ~0.5 = sqrt(d))")

    # ================================================================
    # PHASE 4: Closed-form prediction
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 4: Closed-form prediction quality")
    print("-" * 70)

    # Collect all results
    all_results = kappa_results + K_results + d_results

    kappas_all = np.array([r["kappa"] for r in all_results])
    ds_all = np.array([float(r["d"]) for r in all_results])
    Ks_all = np.array([float(r["K"]) for r in all_results])
    qs_all = np.array([r["q_mc"] for r in all_results])

    # Model A: Phi(a * kappa * d / sqrt(K) + b) -- from old Gaussian theory
    def model_probit_kd(params, kappas, ds, Ks):
        a, b = params
        x = a * kappas * ds / np.sqrt(Ks) + b
        return norm.cdf(x)

    def loss_A(params):
        pred = model_probit_kd(params, kappas_all, ds_all, Ks_all)
        return np.sum((qs_all - pred)**2)

    res_A = minimize(loss_A, [0.01, -1.0], method="Nelder-Mead")
    q_A = model_probit_kd(res_A.x, kappas_all, ds_all, Ks_all)
    r2_A = 1 - np.sum((qs_all - q_A)**2) / np.sum((qs_all - qs_all.mean())**2)
    print(f"  Model A: Phi(a*kappa*d/sqrt(K)+b): R^2={r2_A:.4f}")

    # Model A2: Phi(a * kappa * d / log(K+1) + b) -- CORRECTED theory
    def model_probit_kd_logK(params, kappas, ds, Ks):
        a, b = params
        x = a * kappas * ds / np.log(Ks + 1) + b
        return norm.cdf(x)

    def loss_A2(params):
        pred = model_probit_kd_logK(params, kappas_all, ds_all, Ks_all)
        return np.sum((qs_all - pred)**2)

    res_A2 = minimize(loss_A2, [0.01, -1.0], method="Nelder-Mead")
    q_A2 = model_probit_kd_logK(res_A2.x, kappas_all, ds_all, Ks_all)
    r2_A2 = 1 - np.sum((qs_all - q_A2)**2) / np.sum((qs_all - qs_all.mean())**2)
    print(f"  Model A2: Phi(a*kappa*d/log(K+1)+b): R^2={r2_A2:.4f}  [CORRECTED]")

    # Model B: Phi(a * kappa * sqrt(d) / sqrt(K) + b)
    def model_probit_ksd(params, kappas, ds, Ks):
        a, b = params
        x = a * kappas * np.sqrt(ds) / np.sqrt(Ks) + b
        return norm.cdf(x)

    def loss_B(params):
        pred = model_probit_ksd(params, kappas_all, ds_all, Ks_all)
        return np.sum((qs_all - pred)**2)

    res_B = minimize(loss_B, [1.0, -1.0], method="Nelder-Mead")
    q_B = model_probit_ksd(res_B.x, kappas_all, ds_all, Ks_all)
    r2_B = 1 - np.sum((qs_all - q_B)**2) / np.sum((qs_all - qs_all.mean())**2)
    print(f"  Model B: Phi(a*kappa*sqrt(d)/sqrt(K)+b): R^2={r2_B:.4f}")

    # Model B2: Phi(a * kappa * sqrt(d) / log(K+1) + b)
    def model_probit_ksd_logK(params, kappas, ds, Ks):
        a, b = params
        x = a * kappas * np.sqrt(ds) / np.log(Ks + 1) + b
        return norm.cdf(x)

    def loss_B2(params):
        pred = model_probit_ksd_logK(params, kappas_all, ds_all, Ks_all)
        return np.sum((qs_all - pred)**2)

    res_B2 = minimize(loss_B2, [0.1, -1.0], method="Nelder-Mead")
    q_B2 = model_probit_ksd_logK(res_B2.x, kappas_all, ds_all, Ks_all)
    r2_B2 = 1 - np.sum((qs_all - q_B2)**2) / np.sum((qs_all - qs_all.mean())**2)
    print(f"  Model B2: Phi(a*kappa*sqrt(d)/log(K+1)+b): R^2={r2_B2:.4f}  [CORRECTED]")

    # Model C: Phi(a * kappa * d / (K^gamma) + b) -- free K exponent
    def loss_C(params):
        a, b, gamma = params
        x = a * kappas_all * ds_all / np.power(Ks_all, gamma) + b
        pred = norm.cdf(x)
        return np.sum((qs_all - pred)**2)

    best_C = minimize(loss_C, [0.01, -1.0, 0.5], method="Nelder-Mead",
                      options={"maxiter": 10000})
    a_C, b_C, gamma_C = best_C.x
    q_C = norm.cdf(a_C * kappas_all * ds_all / np.power(Ks_all, gamma_C) + b_C)
    r2_C = 1 - np.sum((qs_all - q_C)**2) / np.sum((qs_all - qs_all.mean())**2)
    print(f"  Model C: Phi(a*kappa*d/K^{gamma_C:.3f}+b): R^2={r2_C:.4f}")

    # Model D: Phi(a * kappa * d^delta / K^gamma + b) -- free both
    def loss_D(params):
        a, b, delta, gamma = params
        x = a * kappas_all * np.power(ds_all, delta) / np.power(Ks_all, gamma) + b
        pred = norm.cdf(x)
        return np.sum((qs_all - pred)**2)

    best_D = minimize(loss_D, [0.1, -1.0, 0.5, 0.5], method="Nelder-Mead",
                      options={"maxiter": 20000})
    a_D, b_D, delta_D, gamma_D = best_D.x
    q_D = norm.cdf(a_D * kappas_all * np.power(ds_all, delta_D) / np.power(Ks_all, gamma_D) + b_D)
    r2_D = 1 - np.sum((qs_all - q_D)**2) / np.sum((qs_all - qs_all.mean())**2)
    print(f"  Model D: Phi(a*kappa*d^{delta_D:.3f}/K^{gamma_D:.3f}+b): R^2={r2_D:.4f}")
    print(f"    (Theory predicts delta=1.0; K-norm: log(K) from EVT)")

    # ================================================================
    # PHASE 5: q_probit vs q_mc across ALL conditions
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 5: Does Phi(mu_M/sigma_M) = q_mc universally?")
    print("-" * 70)

    q_mc_all = np.array([r["q_mc"] for r in all_results])
    q_probit_all = np.array([r["q_probit"] for r in all_results])

    mae = np.mean(np.abs(q_mc_all - q_probit_all))
    r_pq = np.corrcoef(q_mc_all, q_probit_all)[0, 1]
    max_err = np.max(np.abs(q_mc_all - q_probit_all))

    print(f"  N = {len(all_results)} conditions")
    print(f"  Phi(mu_M/sigma_M) vs q_mc: MAE={mae:.4f}, r={r_pq:.6f}, max_err={max_err:.4f}")

    if mae < 0.02:
        print(f"  PROBIT FORMULA VERIFIED: q = Phi(mu_M / sigma_M) universally!")
    else:
        print(f"  PROBIT FORMULA APPROXIMATE: MAE={mae:.4f}")

    # ================================================================
    # SCORECARD
    # ================================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print("=" * 70)

    checks = [
        ("mu_M linear in kappa (R^2 > 0.99)",
         r2_linear > 0.99, f"R^2={r2_linear:.4f}"),
        ("q = Phi(mu_M/sigma_M) universally (MAE < 0.03)",
         mae < 0.03, f"MAE={mae:.4f}"),
        ("mu_M ~ d^1.0 (within 0.2)",
         abs(coef_mu_d[0] - 1.0) < 0.2, f"exponent={coef_mu_d[0]:.3f}"),
        ("sigma_M ~ d^0.5 (within 0.15)",
         abs(coef_sig_d[0] - 0.5) < 0.15, f"exponent={coef_sig_d[0]:.3f}"),
        ("Best K-exponent close to 0.5 (within 0.15)",
         abs(gamma_C - 0.5) < 0.15, f"gamma_K={gamma_C:.3f}"),
        ("Best d-exponent in [0.8, 1.2]",
         0.8 < delta_D < 1.2, f"delta_d={delta_D:.3f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "probit_proof_verification",
        "kappa_linearity_r2": float(r2_linear),
        "probit_mae": float(mae),
        "probit_r": float(r_pq),
        "mu_d_exponent": float(coef_mu_d[0]),
        "sigma_d_exponent": float(coef_sig_d[0]),
        "mu_K_exponent": float(alpha_mu),
        "sigma_K_exponent": float(alpha_sigma),
        "best_K_gamma": float(gamma_C),
        "best_d_delta": float(delta_D),
        "model_fits": {
            "kd_sqrtK": float(r2_A),
            "kd_logK": float(r2_A2),
            "k_sqrtd_sqrtK": float(r2_B),
            "k_sqrtd_logK": float(r2_B2),
            "kd_K_gamma": float(r2_C),
            "k_d_delta_K_gamma": float(r2_D),
        },
    }

    out_path = RESULTS_DIR / "cti_probit_proof.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
