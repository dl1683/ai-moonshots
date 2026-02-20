#!/usr/bin/env python
"""
MARGIN ANATOMY: Empirical measurement of margin statistics.

The zero-param theory predicts:
  q = Phi(mu_M / sigma_M)

where M = min_dist_other - min_dist_same.

This script measures the ACTUAL mu_M and sigma_M from data and checks:
1. How do mu_M and sigma_M depend on kappa and K?
2. Does sigma_M grow with K? (This would explain the divisive interaction)
3. Is the additive form logit(q) = A*kappa - B*f(K) + C correct, or is the
   divisive form q = sigmoid(kappa/g(K)) needed?
4. What is the correct K-normalization: log(K), sqrt(K), K^alpha, etc.?

This is the critical diagnostic for closing the theory-practice gap.
"""

import json
import sys
import numpy as np
from scipy import stats
from scipy.special import ndtri
from scipy.optimize import curve_fit
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def generate_mixture(K, n, d, kappa, seed=42):
    """Generate K-class Gaussian mixture with random means."""
    rng = np.random.RandomState(seed)
    n_per = n // K
    labels = np.repeat(np.arange(K), n_per)[:n]
    if len(labels) < n:
        labels = np.concatenate([labels, rng.randint(0, K, n - len(labels))])
    labels = labels[:n]
    class_means = rng.randn(K, d) * np.sqrt(kappa)
    X = np.zeros((n, d))
    for k in range(K):
        mask = labels == k
        X[mask] = class_means[k] + rng.randn(mask.sum(), d)
    return X, labels


def compute_margins(X, labels):
    """Compute the margin M = min_dist_other - min_dist_same for each point.

    Returns array of margins and the kNN accuracy.
    """
    n = len(X)
    K = len(np.unique(labels))

    # Compute all pairwise distances
    D = pairwise_distances(X, metric='sqeuclidean')
    np.fill_diagonal(D, np.inf)  # exclude self

    margins = []
    correct = 0

    for i in range(n):
        y_i = labels[i]
        same_mask = (labels == y_i)
        same_mask[i] = False  # exclude self
        other_mask = ~(labels == y_i)

        if same_mask.sum() == 0 or other_mask.sum() == 0:
            continue

        d_same_min = D[i, same_mask].min()
        d_other_min = D[i, other_mask].min()

        margin = d_other_min - d_same_min
        margins.append(margin)

        if margin > 0:
            correct += 1

    margins = np.array(margins)
    knn_acc = correct / len(margins)

    return margins, knn_acc


def compute_kappa(X, labels):
    """Compute kappa = tr(S_B)/tr(S_W)."""
    classes = np.unique(labels)
    grand_mean = X.mean(axis=0)
    tr_sb, tr_sw = 0.0, 0.0
    for c in classes:
        Xc = X[labels == c]
        mu_c = Xc.mean(axis=0)
        tr_sb += len(Xc) * np.sum((mu_c - grand_mean)**2)
        tr_sw += np.sum((Xc - mu_c)**2)
    return tr_sb / max(tr_sw, 1e-10)


def main():
    print("=" * 70)
    print("MARGIN ANATOMY: How do mu_M and sigma_M depend on kappa and K?")
    print("=" * 70)

    d = 200  # fixed dimension
    n_total = 2000  # fixed total samples

    all_results = {"d": d, "n_total": n_total}

    # ================================================================
    # EXPERIMENT 1: Fix K, vary kappa -> measure mu_M, sigma_M
    # ================================================================
    print("\n" + "=" * 70)
    print("EXP 1: Fix K=20, vary kappa -> margin statistics")
    print("=" * 70)

    K_fixed = 20
    kappa_range = np.logspace(-1.5, 0.5, 20)

    exp1_results = []
    for kappa_true in kappa_range:
        X, labels = generate_mixture(K_fixed, n_total, d, kappa_true, seed=42)
        kappa_meas = compute_kappa(X, labels)
        margins, knn_acc = compute_margins(X, labels)
        q_obs = (knn_acc - 1.0/K_fixed) / (1.0 - 1.0/K_fixed)

        mu_M = float(np.mean(margins))
        sigma_M = float(np.std(margins))
        median_M = float(np.median(margins))
        skew_M = float(stats.skew(margins))
        kurt_M = float(stats.kurtosis(margins))  # excess kurtosis

        # Theory prediction: q = Phi(mu_M / sigma_M)
        z_empirical = mu_M / sigma_M if sigma_M > 1e-10 else 0
        q_probit = float(stats.norm.cdf(z_empirical))

        exp1_results.append({
            "kappa": float(kappa_meas),
            "mu_M": mu_M,
            "sigma_M": sigma_M,
            "median_M": median_M,
            "skew": skew_M,
            "kurtosis": kurt_M,
            "z_score": float(z_empirical),
            "q_obs": float(q_obs),
            "q_probit": float(q_probit),
            "knn_acc": float(knn_acc),
        })

        print(f"  kappa={kappa_meas:.4f}: mu_M={mu_M:.1f}, sigma_M={sigma_M:.1f}, "
              f"z={z_empirical:.3f}, q_obs={q_obs:.4f}, q_probit={q_probit:.4f}")

    # Check: is q_probit = Phi(mu_M/sigma_M) accurate?
    q_obs_arr = np.array([r["q_obs"] for r in exp1_results])
    q_probit_arr = np.array([r["q_probit"] for r in exp1_results])
    probit_mae = float(np.mean(np.abs(q_obs_arr - q_probit_arr)))
    probit_corr = float(np.corrcoef(q_obs_arr, q_probit_arr)[0, 1])

    print(f"\n  Probit approximation: MAE={probit_mae:.4f}, r={probit_corr:.4f}")

    # Fit mu_M = a * kappa + b
    kappas = np.array([r["kappa"] for r in exp1_results])
    mu_Ms = np.array([r["mu_M"] for r in exp1_results])
    sigma_Ms = np.array([r["sigma_M"] for r in exp1_results])

    # Linear fit mu_M vs kappa
    slope_mu, intercept_mu, r_mu, _, _ = stats.linregress(kappas, mu_Ms)
    print(f"\n  mu_M = {slope_mu:.1f} * kappa + {intercept_mu:.1f}  (r={r_mu:.4f})")

    # sigma_M vs kappa
    slope_sig, intercept_sig, r_sig, _, _ = stats.linregress(kappas, sigma_Ms)
    print(f"  sigma_M = {slope_sig:.1f} * kappa + {intercept_sig:.1f}  (r={r_sig:.4f})")

    all_results["exp1_fix_K"] = {
        "K": K_fixed,
        "results": exp1_results,
        "probit_mae": probit_mae,
        "probit_corr": probit_corr,
        "mu_M_fit": {"slope": float(slope_mu), "intercept": float(intercept_mu), "r": float(r_mu)},
        "sigma_M_fit": {"slope": float(slope_sig), "intercept": float(intercept_sig), "r": float(r_sig)},
    }

    # ================================================================
    # EXPERIMENT 2: Fix kappa, vary K -> the CRITICAL test
    # ================================================================
    print("\n" + "=" * 70)
    print("EXP 2: Fix kappa, vary K -> how do mu_M and sigma_M scale with K?")
    print("=" * 70)

    K_values = [3, 5, 10, 20, 50, 100]
    kappa_targets = [0.05, 0.1, 0.2, 0.5]

    exp2_results = []
    for kappa_target in kappa_targets:
        print(f"\n  --- kappa_target = {kappa_target} ---")
        for K in K_values:
            n_per = n_total // K
            if n_per < 5:
                print(f"  K={K}: SKIP (n_per={n_per} too small)")
                continue

            X, labels = generate_mixture(K, n_total, d, kappa_target, seed=42)
            kappa_meas = compute_kappa(X, labels)
            margins, knn_acc = compute_margins(X, labels)
            q_obs = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

            mu_M = float(np.mean(margins))
            sigma_M = float(np.std(margins))

            z_empirical = mu_M / sigma_M if sigma_M > 1e-10 else 0

            exp2_results.append({
                "kappa_target": float(kappa_target),
                "K": int(K),
                "kappa_meas": float(kappa_meas),
                "n_per": int(n_per),
                "mu_M": mu_M,
                "sigma_M": sigma_M,
                "z_score": float(z_empirical),
                "q_obs": float(q_obs),
                "knn_acc": float(knn_acc),
                "log_K": float(np.log(K)),
                "sqrt_K": float(np.sqrt(K)),
            })

            print(f"  K={K:4d}: kappa={kappa_meas:.4f}, mu_M={mu_M:8.1f}, "
                  f"sigma_M={sigma_M:8.1f}, z={z_empirical:6.3f}, q={q_obs:.4f}")

    # ================================================================
    # KEY ANALYSIS: How does the z-score z = mu_M/sigma_M relate to K?
    # ================================================================
    print("\n" + "=" * 70)
    print("KEY ANALYSIS: Z-score vs K scaling")
    print("=" * 70)

    # For each kappa level, fit z vs f(K) for different f
    for kappa_target in kappa_targets:
        subset = [r for r in exp2_results if r["kappa_target"] == kappa_target and r["K"] >= 3]
        if len(subset) < 3:
            continue

        Ks = np.array([r["K"] for r in subset])
        zs = np.array([r["z_score"] for r in subset])
        mu_Ms = np.array([r["mu_M"] for r in subset])
        sigma_Ms = np.array([r["sigma_M"] for r in subset])
        kappas_meas = np.array([r["kappa_meas"] for r in subset])

        print(f"\n  kappa_target={kappa_target}:")
        print(f"    K:       {[r['K'] for r in subset]}")
        mu_strs = [f"{r['mu_M']:.1f}" for r in subset]
        sig_strs = [f"{r['sigma_M']:.1f}" for r in subset]
        z_strs = [f"{r['z_score']:.3f}" for r in subset]
        print(f"    mu_M:    {mu_strs}")
        print(f"    sigma_M: {sig_strs}")
        print(f"    z:       {z_strs}")

        # Fit sigma_M vs K
        log_Ks = np.log(Ks)
        log_sigmas = np.log(sigma_Ms)
        slope_s, intercept_s, r_s, _, _ = stats.linregress(log_Ks, log_sigmas)
        print(f"    sigma_M ~ K^{slope_s:.3f}  (log-log r={r_s:.3f})")

        # Fit mu_M vs K
        log_mus = np.log(np.abs(mu_Ms) + 1e-10)
        slope_m, intercept_m, r_m, _, _ = stats.linregress(log_Ks, log_mus)
        print(f"    |mu_M| ~ K^{slope_m:.3f}  (log-log r={r_m:.3f})")

        # Fit z vs different K normalizations
        for name, f_K in [("log(K)", np.log(Ks)), ("sqrt(K)", np.sqrt(Ks)),
                           ("K^0.3", Ks**0.3), ("K^0.4", Ks**0.4)]:
            slope, intercept, r_val, _, _ = stats.linregress(f_K, zs)
            print(f"    z vs {name:8s}: z = {slope:+.4f}*{name} + {intercept:.4f}  (r={r_val:.3f})")

    all_results["exp2_vary_K"] = exp2_results

    # ================================================================
    # EXPERIMENT 3: Full kappa x K grid -> fit the joint law
    # ================================================================
    print("\n" + "=" * 70)
    print("EXP 3: Full kappa x K grid -> determine the joint law")
    print("=" * 70)

    exp3_data = []
    K_grid = [5, 10, 20, 50, 100]
    kappa_grid = np.logspace(-1.5, 0.5, 12)

    for K in K_grid:
        for kappa_true in kappa_grid:
            n_per = n_total // K
            if n_per < 5:
                continue
            X, labels = generate_mixture(K, n_total, d, kappa_true, seed=42)
            kappa_meas = compute_kappa(X, labels)
            margins, knn_acc = compute_margins(X, labels)
            q_obs = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

            mu_M = float(np.mean(margins))
            sigma_M = float(np.std(margins))
            z = mu_M / sigma_M if sigma_M > 1e-10 else 0

            exp3_data.append({
                "K": int(K), "kappa": float(kappa_meas),
                "n_per": int(n_per),
                "mu_M": mu_M, "sigma_M": sigma_M,
                "z": float(z), "q_obs": float(q_obs),
            })

    if len(exp3_data) >= 10:
        # Fit competing models for z(kappa, K)
        K_arr = np.array([r["K"] for r in exp3_data])
        kappa_arr = np.array([r["kappa"] for r in exp3_data])
        z_arr = np.array([r["z"] for r in exp3_data])
        q_arr = np.array([r["q_obs"] for r in exp3_data])

        # Filter out saturated points (q near 0 or 1)
        mask = (q_arr > 0.01) & (q_arr < 0.99)
        K_f = K_arr[mask]
        kappa_f = kappa_arr[mask]
        z_f = z_arr[mask]
        q_f = q_arr[mask]

        print(f"\n  {mask.sum()} non-saturated points out of {len(exp3_data)}")

        # Model 1: z = a*kappa + b*log(K) + c  (additive log)
        def model_add_log(X, a, b, c):
            kap, K = X
            return a * kap + b * np.log(K) + c

        # Model 2: z = a*kappa/sqrt(K) + c  (divisive sqrt)
        def model_div_sqrt(X, a, c):
            kap, K = X
            return a * kap / np.sqrt(K) + c

        # Model 3: z = a*kappa/log(K) + c  (divisive log)
        def model_div_log(X, a, c):
            kap, K = X
            return a * kap / np.log(K) + c

        # Model 4: z = a*kappa + b*sqrt(K) + c  (additive sqrt)
        def model_add_sqrt(X, a, b, c):
            kap, K = X
            return a * kap + b * np.sqrt(K) + c

        # Model 5: z = a*kappa/K^alpha + c  (divisive power)
        def model_div_power(X, a, alpha, c):
            kap, K = X
            return a * kap / (K ** alpha) + c

        # Model 6: z = a*kappa*sqrt(d/K) + b*log(K) + c  (theory-inspired)
        def model_theory(X, a, b, c):
            kap, K = X
            return a * kap * np.sqrt(d / K) + b * np.log(K) + c

        models = {
            "additive_log": (model_add_log, [1.0, -1.0, 0.0]),
            "divisive_sqrt": (model_div_sqrt, [10.0, -1.0]),
            "divisive_log": (model_div_log, [10.0, -1.0]),
            "additive_sqrt": (model_add_sqrt, [1.0, -0.1, 0.0]),
            "divisive_power": (model_div_power, [10.0, 0.5, -1.0]),
            "theory_mixed": (model_theory, [1.0, -1.0, 0.0]),
        }

        print("\n  MODEL COMPARISON (predicting z-score on non-saturated data):")
        model_results = {}

        for name, (func, p0) in models.items():
            try:
                popt, _ = curve_fit(func, (kappa_f, K_f), z_f, p0=p0, maxfev=10000)
                z_pred = func((kappa_f, K_f), *popt)
                ss_res = np.sum((z_f - z_pred)**2)
                ss_tot = np.sum((z_f - z_f.mean())**2)
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                mae = float(np.mean(np.abs(z_f - z_pred)))

                # Also compute q prediction
                q_pred = stats.norm.cdf(z_pred)
                q_mae = float(np.mean(np.abs(q_f - q_pred)))
                q_ss_res = np.sum((q_f - q_pred)**2)
                q_ss_tot = np.sum((q_f - q_f.mean())**2)
                q_r2 = 1 - q_ss_res / max(q_ss_tot, 1e-10)

                params_str = ", ".join(f"{p:.4f}" for p in popt)
                print(f"  {name:20s}: z R^2={r2:.4f}, q R^2={q_r2:.4f}, q MAE={q_mae:.4f}  params=[{params_str}]")

                model_results[name] = {
                    "z_r2": float(r2), "z_mae": float(mae),
                    "q_r2": float(q_r2), "q_mae": float(q_mae),
                    "params": [float(p) for p in popt],
                    "n_params": len(popt),
                }
            except Exception as e:
                print(f"  {name:20s}: FAILED ({e})")
                model_results[name] = {"error": str(e)}

        # Find best by AIC (penalize for params)
        print("\n  MODEL RANKING (by q_R^2 adjusted for params):")
        valid_models = [(n, r) for n, r in model_results.items() if "q_r2" in r]
        valid_models.sort(key=lambda x: x[1]["q_r2"], reverse=True)
        for rank, (name, r) in enumerate(valid_models, 1):
            aic_penalty = 2 * r["n_params"] / len(kappa_f)
            adj_r2 = r["q_r2"] - aic_penalty
            print(f"  #{rank}: {name:20s} q_R^2={r['q_r2']:.4f} (adj={adj_r2:.4f}, {r['n_params']} params)")

        all_results["exp3_model_comparison"] = model_results

    # ================================================================
    # EXPERIMENT 4: Does the theory error come from mu_M or sigma_M?
    # ================================================================
    print("\n" + "=" * 70)
    print("EXP 4: Theory error decomposition")
    print("=" * 70)

    K_test = 20
    n_per_test = n_total // K_test
    kappa_test_range = np.logspace(-1.5, 0, 15)

    exp4_results = []
    for kappa_true in kappa_test_range:
        X, labels = generate_mixture(K_test, n_total, d, kappa_true, seed=42)
        kappa_meas = compute_kappa(X, labels)
        margins, knn_acc = compute_margins(X, labels)
        q_obs = (knn_acc - 1.0/K_test) / (1.0 - 1.0/K_test)

        mu_M_emp = float(np.mean(margins))
        sigma_M_emp = float(np.std(margins))

        # Theory predictions for mu_M and sigma_M
        n = n_per_test
        m = (K_test - 1) * n_per_test

        delta_sq = 2.0 * kappa_meas * d
        mu_s = 2.0 * d
        sigma_s = np.sqrt(8.0 * d)
        mu_o = 2.0 * d + delta_sq
        sigma_o = np.sqrt(8.0 * d + 8.0 * delta_sq)

        p_n = 1.0 / (n + 1)
        z_n = ndtri(p_n)
        phi_z_n = stats.norm.pdf(z_n)
        mu_s_min_th = mu_s + sigma_s * z_n
        tau_s_th = sigma_s / (n * phi_z_n) if phi_z_n > 1e-20 else sigma_s

        p_m = 1.0 / (m + 1)
        z_m = ndtri(p_m)
        phi_z_m = stats.norm.pdf(z_m)
        mu_o_min_th = mu_o + sigma_o * z_m
        tau_o_th = sigma_o / (m * phi_z_m) if phi_z_m > 1e-20 else sigma_o

        mu_M_th = mu_o_min_th - mu_s_min_th
        sigma_M_th = np.sqrt(tau_o_th**2 + tau_s_th**2)

        exp4_results.append({
            "kappa": float(kappa_meas),
            "mu_M_emp": mu_M_emp,
            "mu_M_theory": float(mu_M_th),
            "mu_M_error": float(mu_M_emp - mu_M_th),
            "sigma_M_emp": sigma_M_emp,
            "sigma_M_theory": float(sigma_M_th),
            "sigma_M_ratio": float(sigma_M_emp / sigma_M_th) if sigma_M_th > 0 else float('nan'),
            "q_obs": float(q_obs),
        })

        print(f"  kappa={kappa_meas:.4f}: mu_M emp={mu_M_emp:8.1f} th={mu_M_th:8.1f} "
              f"(err={mu_M_emp-mu_M_th:+.1f}) | sigma_M emp={sigma_M_emp:6.1f} th={sigma_M_th:6.1f} "
              f"(ratio={sigma_M_emp/sigma_M_th:.3f})")

    # Summary: which component has more error?
    mu_errors = np.array([r["mu_M_error"] for r in exp4_results])
    sigma_ratios = np.array([r["sigma_M_ratio"] for r in exp4_results])

    print(f"\n  mu_M error: mean={np.mean(mu_errors):.1f}, std={np.std(mu_errors):.1f}")
    print(f"  sigma_M ratio (emp/theory): mean={np.nanmean(sigma_ratios):.3f}, std={np.nanstd(sigma_ratios):.3f}")
    print(f"  -> Theory {'OVERestimates' if np.mean(mu_errors) < 0 else 'UNDERestimates'} mu_M")
    print(f"  -> Theory {'OVERestimates' if np.nanmean(sigma_ratios) < 1 else 'UNDERestimates'} sigma_M")

    all_results["exp4_theory_decomposition"] = exp4_results

    # ================================================================
    # SCORECARD
    # ================================================================
    print("\n" + "=" * 70)
    print("SCORECARD")
    print("=" * 70)

    checks = []

    # C1: Probit approximation works (MAE < 0.03)
    c1 = probit_mae < 0.03
    checks.append({"criterion": "Probit Phi(mu_M/sigma_M) accurate (MAE<0.03)",
                    "passed": bool(c1), "value": f"MAE={probit_mae:.4f}"})

    # C2: mu_M linear in kappa (r > 0.99)
    c2 = abs(r_mu) > 0.99
    checks.append({"criterion": "mu_M linear in kappa (|r|>0.99)",
                    "passed": bool(c2), "value": f"r={r_mu:.4f}"})

    # C3: Best model is identified
    if valid_models:
        best_name = valid_models[0][0]
        best_r2 = valid_models[0][1]["q_r2"]
        c3 = best_r2 > 0.95
        checks.append({"criterion": f"Best z-model R^2 > 0.95 ({best_name})",
                        "passed": bool(c3), "value": f"R^2={best_r2:.4f}"})
    else:
        checks.append({"criterion": "Best z-model R^2 > 0.95",
                        "passed": False, "value": "no models"})

    # C4: sigma_M ratio near 1 (theory calibrated)
    mean_ratio = float(np.nanmean(sigma_ratios))
    c4 = abs(mean_ratio - 1.0) < 0.2
    checks.append({"criterion": "sigma_M theory calibrated (ratio 0.8-1.2)",
                    "passed": bool(c4), "value": f"ratio={mean_ratio:.3f}"})

    passes = sum(1 for c in checks if c["passed"])
    total = len(checks)

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['criterion']}: {c['value']}")

    print(f"\n  SCORECARD: {passes}/{total}")

    all_results["scorecard"] = {"passes": passes, "total": total, "checks": checks}

    # Save
    out_path = RESULTS_DIR / "cti_margin_anatomy.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
