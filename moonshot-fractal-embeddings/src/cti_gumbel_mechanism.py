#!/usr/bin/env python
"""
GUMBEL MECHANISM: Verify that sigmoid arises from Gumbel extreme-value theory.

Core theoretical argument (from Codex proof strategy):
1. For K-class Gaussian mixture, squared distances to nearest same-class and
   nearest diff-class neighbors both follow Gumbel distributions (as minima of
   chi-squared variables in high dimensions).
2. The difference of two Gumbel variables follows a LOGISTIC distribution.
3. logistic CDF = sigmoid function.
4. The location parameter of the logistic scales as kappa, and the scale
   parameter scales as sqrt(K).
5. Therefore: q = sigmoid(a * kappa / sqrt(K) + c).

This script verifies each step numerically.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.stats import gumbel_l, logistic, norm, kstest
from scipy.optimize import curve_fit, minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def step1_gumbel_convergence(d=256, n_per_class=100, n_trials=10000):
    """
    STEP 1: Verify that min ||x - x_i||^2 for same-class follows Gumbel.

    For x, x_i ~ N(mu, sigma^2 I_d), ||x - x_i||^2 ~ sigma^2 * chi^2(d)
    with mean 2*d*sigma^2 and var 8*d*sigma^4.
    The minimum of n such variables should converge to Gumbel as n->inf.
    """
    print("STEP 1: Gumbel convergence of min distances")
    print("-" * 60)

    sigma = 1.0

    # Generate same-class distances
    min_dists = []
    for _ in range(n_trials):
        # x and x_1, ..., x_n all from N(0, sigma^2 I_d)
        x = sigma * np.random.randn(d)
        X = sigma * np.random.randn(n_per_class, d)
        dists = np.sum((X - x[None, :]) ** 2, axis=1)
        min_dists.append(np.min(dists))

    min_dists = np.array(min_dists)

    # Standardize
    mu_min = np.mean(min_dists)
    std_min = np.std(min_dists)
    standardized = (min_dists - mu_min) / std_min

    # Test against Gumbel (minimum = left Gumbel)
    # The Gumbel for minima has CDF: F(x) = 1 - exp(-exp((x-loc)/scale))
    ks_gumbel, p_gumbel = kstest(standardized, 'gumbel_l')
    ks_normal, p_normal = kstest(standardized, 'norm')

    print(f"  d={d}, n={n_per_class}, trials={n_trials}")
    print(f"  min dist: mean={mu_min:.2f}, std={std_min:.2f}")
    print(f"  Expected: mean~{2*d*sigma**2:.0f} - O(sqrt(d*log(n)))")
    print(f"  KS test vs Gumbel: stat={ks_gumbel:.4f}, p={p_gumbel:.4f}")
    print(f"  KS test vs Normal: stat={ks_normal:.4f}, p={p_normal:.4f}")
    print(f"  Gumbel better: {p_gumbel > p_normal}")

    return {"ks_gumbel": ks_gumbel, "p_gumbel": p_gumbel,
            "ks_normal": ks_normal, "p_normal": p_normal,
            "mean": mu_min, "std": std_min}


def step2_logistic_from_gumbel_difference(n_trials=50000):
    """
    STEP 2: Verify that Gumbel(a) - Gumbel(b) ~ Logistic.

    If G1 ~ Gumbel(mu1, beta1) and G2 ~ Gumbel(mu2, beta2) independently,
    then G1 - G2 ~ Logistic(mu1 - mu2, beta) approximately when beta1 ~ beta2.

    More precisely, for standard Gumbels with same scale:
    G1 - G2 has distribution with CDF = sigmoid((x - loc) / scale).
    """
    print("\nSTEP 2: Gumbel difference -> Logistic distribution")
    print("-" * 60)

    results = []
    for beta in [0.5, 1.0, 2.0]:
        for delta_mu in [0.0, 1.0, 3.0]:
            # Generate two Gumbel samples
            G1 = np.random.gumbel(loc=delta_mu, scale=beta, size=n_trials)
            G2 = np.random.gumbel(loc=0.0, scale=beta, size=n_trials)
            diff = G1 - G2

            # Standardize
            diff_mean = np.mean(diff)
            diff_std = np.std(diff)
            standardized = (diff - diff_mean) / diff_std

            # Test against logistic
            ks_logistic, p_logistic = kstest(standardized, 'logistic')
            ks_normal, p_normal = kstest(standardized, 'norm')

            is_logistic = p_logistic > 0.01

            print(f"  beta={beta}, delta_mu={delta_mu}: "
                  f"KS_logistic={ks_logistic:.4f} (p={p_logistic:.4f}), "
                  f"KS_normal={ks_normal:.4f} (p={p_normal:.4f}) "
                  f"{'LOGISTIC' if is_logistic else 'NOT logistic'}")

            results.append({
                "beta": beta, "delta_mu": delta_mu,
                "ks_logistic": ks_logistic, "p_logistic": p_logistic,
                "ks_normal": ks_normal, "p_normal": p_normal,
            })

    return results


def step3_knn_margin_is_logistic(K=50, d=256, kappa=0.3, n_per_class=100,
                                  n_trials=5000):
    """
    STEP 3: Verify that the kNN classification margin follows logistic distribution.

    For a test point x from class k:
    margin = min_{j != k} D_j - D_k
    where D_k = min_i ||x - x_{k,i}||^2 (nearest same-class)
    and D_j = min_i ||x - x_{j,i}||^2 (nearest class-j)

    If margin > 0, classification is correct.
    """
    print(f"\nSTEP 3: kNN margin distribution (K={K}, d={d}, kappa={kappa})")
    print("-" * 60)

    sigma = 1.0
    Delta = np.sqrt(kappa * d * sigma ** 2)

    # Generate centroids on simplex
    centroids = np.zeros((K, d))
    # Random directions, scaled
    raw = np.random.randn(K, d)
    raw -= raw.mean(0)
    # Make equal distance from origin
    norms = np.sqrt(np.sum(raw ** 2, axis=1, keepdims=True))
    centroids = raw / norms * Delta

    # Re-center
    centroids -= centroids.mean(0)
    actual_dist = np.sqrt(np.mean(np.sum(centroids ** 2, axis=1)))
    centroids *= Delta / max(actual_dist, 1e-10)

    margins = []
    correct_count = 0

    for trial in range(n_trials):
        # Pick a random class
        k = trial % K

        # Generate test point from class k
        x = centroids[k] + sigma * np.random.randn(d)

        # Generate training points for each class
        D_same = float("inf")
        D_diff_min = float("inf")

        for j in range(K):
            # Generate n_per_class points from class j
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
            correct_count += 1

    margins = np.array(margins)
    accuracy = correct_count / n_trials
    q = (accuracy - 1.0 / K) / (1.0 - 1.0 / K)

    # Test if margins are logistic
    m_mean = np.mean(margins)
    m_std = np.std(margins)
    standardized = (margins - m_mean) / m_std

    ks_logistic, p_logistic = kstest(standardized, 'logistic')
    ks_normal, p_normal = kstest(standardized, 'norm')

    print(f"  accuracy={accuracy:.4f}, q={q:.4f}")
    print(f"  margin: mean={m_mean:.2f}, std={m_std:.2f}")
    print(f"  KS logistic: stat={ks_logistic:.4f}, p={p_logistic:.6f}")
    print(f"  KS normal: stat={ks_normal:.4f}, p={p_normal:.6f}")
    print(f"  Margin is LOGISTIC: {p_logistic > 0.01}")

    return {
        "K": K, "d": d, "kappa": kappa,
        "accuracy": accuracy, "q": q,
        "margin_mean": m_mean, "margin_std": m_std,
        "ks_logistic": ks_logistic, "p_logistic": p_logistic,
        "ks_normal": ks_normal, "p_normal": p_normal,
    }


def step4_location_scales_with_kappa(K=50, d=256, n_per_class=100,
                                      n_trials=3000):
    """
    STEP 4: Verify that margin location scales linearly with kappa.

    If margins ~ Logistic(mu, s), then:
    - mu should scale linearly with kappa (the signal)
    - s should be roughly constant or scale with sqrt(K)
    - P(correct) = sigmoid(mu/s) = sigmoid(a * kappa + b)
    """
    print(f"\nSTEP 4: Margin location scales with kappa (K={K})")
    print("-" * 60)

    sigma = 1.0
    kappa_values = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

    results = []
    for kappa in kappa_values:
        Delta = np.sqrt(kappa * d * sigma ** 2)

        # Generate centroids
        centroids = np.random.randn(K, d)
        centroids -= centroids.mean(0)
        norms = np.sqrt(np.sum(centroids ** 2, axis=1, keepdims=True))
        centroids = centroids / norms * Delta
        centroids -= centroids.mean(0)
        actual_dist = np.sqrt(np.mean(np.sum(centroids ** 2, axis=1)))
        centroids *= Delta / max(actual_dist, 1e-10)

        margins = []
        correct = 0

        for trial in range(n_trials):
            k = trial % K
            x = centroids[k] + sigma * np.random.randn(d)

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

        result = {
            "kappa": kappa,
            "margin_mean": float(np.mean(margins)),
            "margin_std": float(np.std(margins)),
            "accuracy": float(acc),
            "q": float(q),
        }
        results.append(result)
        print(f"  kappa={kappa:.3f}: margin_mean={result['margin_mean']:.2f}, "
              f"margin_std={result['margin_std']:.2f}, q={q:.4f}")

    # Check if margin_mean is linear in kappa
    kappas = np.array([r["kappa"] for r in results])
    means = np.array([r["margin_mean"] for r in results])
    stds = np.array([r["margin_std"] for r in results])
    qs = np.array([r["q"] for r in results])

    # Linear fit: margin_mean = a * kappa + b
    coef = np.polyfit(kappas, means, 1)
    pred_means = np.polyval(coef, kappas)
    ss_res = np.sum((means - pred_means) ** 2)
    ss_tot = np.sum((means - means.mean()) ** 2)
    r2_linear = 1 - ss_res / ss_tot

    print(f"\n  margin_mean vs kappa: slope={coef[0]:.2f}, R^2={r2_linear:.4f}")
    print(f"  margin_std variation: {stds.min():.2f} - {stds.max():.2f} "
          f"(CV={stds.std()/stds.mean():.3f})")

    # Fit sigmoid to q vs kappa/sqrt(K)
    x_sig = kappas / np.sqrt(K)
    try:
        def sig_model(x, a, b):
            return expit(a * x + b)
        popt, _ = curve_fit(sig_model, x_sig, qs, p0=[5.0, -1.0], maxfev=10000)
        q_pred = sig_model(x_sig, *popt)
        r2_sig = 1 - np.sum((qs - q_pred) ** 2) / np.sum((qs - qs.mean()) ** 2)
        print(f"  sigmoid(a*kappa/sqrt(K)+b): a={popt[0]:.3f}, b={popt[1]:.3f}, "
              f"R^2={r2_sig:.4f}")
    except:
        r2_sig = 0

    return results, r2_linear, r2_sig


def step5_sqrt_K_scaling(d=256, n_per_class=100, n_trials=2000):
    """
    STEP 5: Verify that the logistic scale parameter grows as sqrt(K).

    For different K values, compute the margin distribution and check
    if the standard deviation scales as sqrt(K).
    """
    print(f"\nSTEP 5: sqrt(K) scaling of margin noise")
    print("-" * 60)

    sigma = 1.0
    kappa_fixed = 0.3  # Fixed kappa
    K_values = [2, 5, 10, 20, 50, 100]

    results = []
    for K in K_values:
        Delta = np.sqrt(kappa_fixed * d * sigma ** 2)

        centroids = np.random.randn(K, d)
        centroids -= centroids.mean(0)
        norms = np.sqrt(np.sum(centroids ** 2, axis=1, keepdims=True))
        centroids = centroids / norms * Delta
        centroids -= centroids.mean(0)
        actual_dist = np.sqrt(np.mean(np.sum(centroids ** 2, axis=1)))
        centroids *= Delta / max(actual_dist, 1e-10)

        margins = []
        correct = 0
        n_trials_k = min(n_trials, K * 50)  # Scale trials with K

        for trial in range(n_trials_k):
            k = trial % K
            x = centroids[k] + sigma * np.random.randn(d)

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
        acc = correct / n_trials_k
        q = (acc - 1.0 / K) / (1.0 - 1.0 / K)

        result = {
            "K": K, "kappa": kappa_fixed,
            "margin_mean": float(np.mean(margins)),
            "margin_std": float(np.std(margins)),
            "margin_mean_over_std": float(np.mean(margins) / max(np.std(margins), 1e-10)),
            "accuracy": float(acc),
            "q": float(q),
        }
        results.append(result)
        print(f"  K={K:>4}: mean={result['margin_mean']:.2f}, "
              f"std={result['margin_std']:.2f}, "
              f"mean/std={result['margin_mean_over_std']:.3f}, "
              f"q={q:.4f}")

    # Check scaling: margin_std ~ K^gamma
    Ks = np.array([r["K"] for r in results])
    stds = np.array([r["margin_std"] for r in results])
    means = np.array([r["margin_mean"] for r in results])

    # Log-log fit for std vs K
    log_K = np.log(Ks)
    log_std = np.log(stds)
    coef_std = np.polyfit(log_K, log_std, 1)
    gamma_std = coef_std[0]

    # Log-log fit for mean vs K (should be roughly constant or weakly K-dependent)
    log_mean = np.log(np.abs(means) + 1e-10)
    coef_mean = np.polyfit(log_K, log_mean, 1)
    gamma_mean = coef_mean[0]

    print(f"\n  margin_std ~ K^{gamma_std:.3f} (theory predicts ~0.5 for sqrt(K))")
    print(f"  margin_mean ~ K^{gamma_mean:.3f}")

    # The key ratio: mean/std should scale as 1/sqrt(K) if std~sqrt(K) and mean~const
    ratios = means / stds
    log_ratio = np.log(np.abs(ratios) + 1e-10)
    coef_ratio = np.polyfit(log_K, log_ratio, 1)
    gamma_ratio = coef_ratio[0]
    print(f"  mean/std ~ K^{gamma_ratio:.3f} (theory predicts ~-0.5)")

    return results, gamma_std, gamma_mean


def main():
    print("=" * 70)
    print("GUMBEL MECHANISM: WHY SIGMOID EMERGES FROM GAUSSIAN MIXTURES")
    print("=" * 70)

    all_results = {}

    # Step 1: Gumbel convergence
    r1 = step1_gumbel_convergence()
    all_results["step1_gumbel"] = r1

    # Step 2: Logistic from Gumbel difference
    r2 = step2_logistic_from_gumbel_difference()
    all_results["step2_logistic"] = r2

    # Step 3: kNN margin is logistic
    r3 = step3_knn_margin_is_logistic(K=20, d=128, kappa=0.3,
                                       n_per_class=50, n_trials=3000)
    all_results["step3_margin_logistic"] = r3

    # Step 4: Location scales with kappa
    r4_results, r4_linear, r4_sig = step4_location_scales_with_kappa(
        K=20, d=128, n_per_class=50, n_trials=2000)
    all_results["step4_kappa_scaling"] = {
        "results": r4_results,
        "r2_linear": r4_linear,
        "r2_sigmoid": r4_sig,
    }

    # Step 5: sqrt(K) scaling
    r5_results, r5_gamma_std, r5_gamma_mean = step5_sqrt_K_scaling(
        d=128, n_per_class=50, n_trials=1000)
    all_results["step5_sqrtK"] = {
        "results": r5_results,
        "gamma_std": r5_gamma_std,
        "gamma_mean": r5_gamma_mean,
    }

    # ================================================================
    # SCORECARD
    # ================================================================
    print(f"\n{'='*70}")
    print("SCORECARD: GUMBEL MECHANISM VERIFICATION")
    print("=" * 70)

    checks = [
        ("Min distances converge to Gumbel (p > 0.01)",
         r1["p_gumbel"] > 0.01,
         f"p={r1['p_gumbel']:.4f}"),
        ("Gumbel difference gives logistic distribution",
         all(r["p_logistic"] > 0.01 for r in r2),
         f"all p > 0.01: {all(r['p_logistic'] > 0.01 for r in r2)}"),
        ("kNN margin is logistic (p > 0.01)",
         r3["p_logistic"] > 0.01,
         f"p={r3['p_logistic']:.6f}"),
        ("Margin location linear in kappa (R^2 > 0.95)",
         r4_linear > 0.95,
         f"R^2={r4_linear:.4f}"),
        ("sigmoid(kappa/sqrt(K)) fits q (R^2 > 0.95)",
         r4_sig > 0.95,
         f"R^2={r4_sig:.4f}"),
        ("Margin std scales as sqrt(K) (gamma in [0.3, 0.7])",
         0.3 < r5_gamma_std < 0.7,
         f"gamma={r5_gamma_std:.3f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    out_path = RESULTS_DIR / "cti_gumbel_mechanism.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "____float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
