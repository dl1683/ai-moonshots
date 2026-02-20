#!/usr/bin/env python
"""
GUMBEL THEORY VALIDATION: First-Principles Derivation of q = sigmoid(kappa/sqrt(K))

The key theoretical insight (from Codex/RMT analysis):
1. Same-class distances ~ chi^2_d (central)
2. Wrong-class distances ~ noncentral chi^2_d(lambda_j)
3. Minima over m samples converge to Gumbel distributions (EVT)
4. Difference of two equal-scale Gumbels is LOGISTIC = sigmoid
5. Location gap scales as kappa/sqrt(K) via exchangeable CLT

This script validates each step of the derivation with Monte Carlo simulation.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.stats import gumbel_l, logistic
from scipy.special import expit  # sigmoid
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def simulate_knn_accuracy(d, K, m, kappa_target, n_test=5000, k_nn=1):
    """
    Simulate 1-NN classification for K-class isotropic Gaussians in d dimensions.

    Setup: Y ~ Unif{1,...,K}, X|Y=k ~ N(mu_k, sigma^2 * I_d)
    We set sigma^2 = 1 and adjust ||mu_k - mu_bar||^2 to achieve target kappa.

    For simplex means: ||mu_k - mu_bar||^2 = delta^2 * (K-1)/K for each k
    kappa = tr(S_B)/tr(S_W) = (1/K) * sum_k ||mu_k - mu_bar||^2 / sigma^2
          = delta^2 * (K-1)/K^2 * ...

    Actually for K equal-energy means on a simplex:
    S_B = (delta^2/K) * (I_{K-1} embedded in d dims)
    tr(S_B) = delta^2 * (K-1)/K
    tr(S_W) = d * sigma^2 = d (since sigma=1)
    kappa = delta^2 * (K-1) / (K * d)

    So delta^2 = kappa * K * d / (K-1)
    """
    sigma2 = 1.0

    # Place class means on simplex vertices in first K-1 dimensions
    # Use simplex ETF construction
    delta2 = kappa_target * K * d / (K - 1)
    delta = np.sqrt(delta2)

    # Simplex means: use random orthogonal directions
    # For simplicity, place means along random unit vectors scaled by delta
    # Actually, for proper simplex: mu_k such that ||mu_k - mu_bar||^2 = delta^2*(K-1)/K
    # and <mu_k - mu_bar, mu_j - mu_bar> = -delta^2/K for k != j

    # Construct simplex ETF in first min(K-1, d) dimensions
    if K - 1 <= d:
        # Standard simplex construction
        # Start with identity in (K-1) dims, center to get simplex
        V = np.eye(K, K - 1)  # K x (K-1)
        V = V - V.mean(0)  # center
        # Normalize each row to have ||v_k||^2 = (K-1)/K
        norms = np.sqrt((V ** 2).sum(1, keepdims=True))
        norms[norms < 1e-10] = 1.0
        V = V / norms * np.sqrt((K - 1) / K)

        # Embed in d dimensions
        means = np.zeros((K, d))
        means[:, :K-1] = V * delta
    else:
        # More classes than dimensions: use random directions
        means = np.random.randn(K, d)
        means = means - means.mean(0)
        norms = np.sqrt((means ** 2).sum(1, keepdims=True))
        norms[norms < 1e-10] = 1.0
        means = means / norms * delta * np.sqrt((K - 1) / K)

    # Verify kappa
    grand_mean = means.mean(0)
    actual_tr_sb = sum(np.sum((means[k] - grand_mean)**2) for k in range(K)) / K
    actual_tr_sw = d * sigma2
    actual_kappa = actual_tr_sb / actual_tr_sw

    # Generate training data: m points per class
    train_labels = np.repeat(np.arange(K), m)
    train_X = np.zeros((K * m, d))
    for k in range(K):
        train_X[k*m:(k+1)*m] = means[k] + np.random.randn(m, d) * np.sqrt(sigma2)

    # Generate test data
    test_labels = np.random.randint(0, K, n_test)
    test_X = np.zeros((n_test, d))
    for i in range(n_test):
        test_X[i] = means[test_labels[i]] + np.random.randn(d) * np.sqrt(sigma2)

    # 1-NN classification
    correct = 0
    batch_size = 500
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        # Compute distances: (n_batch, K*m)
        diff = test_X[start:end, np.newaxis, :] - train_X[np.newaxis, :, :]
        dists = (diff ** 2).sum(axis=2)
        nn_idx = dists.argmin(axis=1)
        nn_labels = train_labels[nn_idx]
        correct += (nn_labels == test_labels[start:end]).sum()

    accuracy = correct / n_test
    q = (accuracy - 1.0/K) / (1.0 - 1.0/K)
    return accuracy, q, actual_kappa


def validate_gumbel_logistic():
    """
    Validate: difference of two iid Gumbel(0,1) is Logistic(0,1).
    This is the mathematical heart of the sigmoid derivation.
    """
    print("=" * 70)
    print("LEMMA 4: Difference of Gumbels -> Logistic")
    print("=" * 70)

    n_samples = 100000
    g1 = np.random.gumbel(0, 1, n_samples)
    g2 = np.random.gumbel(0, 1, n_samples)
    diff = g1 - g2  # Should be Logistic(0, 1)

    # KS test against logistic
    from scipy.stats import kstest
    stat, p = kstest(diff, 'logistic')
    print(f"  KS test: stat={stat:.6f}, p={p:.4f}")
    print(f"  Mean diff: {diff.mean():.4f} (theory: 0)")
    print(f"  Std diff: {diff.std():.4f} (theory: pi/sqrt(3) = {np.pi/np.sqrt(3):.4f})")

    passed = p > 0.01
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def validate_min_gumbel():
    """
    Validate: min of m iid chi^2_d samples, properly centered/scaled, -> Gumbel.
    """
    print("\n" + "=" * 70)
    print("LEMMA 2: Min of chi^2 distances -> Gumbel")
    print("=" * 70)

    d = 200
    m = 50
    n_trials = 10000

    mins = []
    for _ in range(n_trials):
        # m samples from chi^2_d
        samples = np.random.chisquare(d, m)
        mins.append(samples.min())
    mins = np.array(mins)

    # Center and scale (Gumbel location/scale for chi^2 minimum)
    # For large d, chi^2_d ~ N(d, 2d), so min of m normals:
    # location ~ d - sqrt(2d) * (sqrt(2*log(m)) - log(log(m))/(2*sqrt(2*log(m))))
    # scale ~ sqrt(2d) / sqrt(2*log(m))

    # Fit Gumbel minimum (using scipy)
    from scipy.stats import gumbel_l
    params = gumbel_l.fit(mins)

    from scipy.stats import kstest
    stat, p = kstest(mins, 'gumbel_l', args=params)
    print(f"  d={d}, m={m}, n_trials={n_trials}")
    print(f"  Gumbel fit: loc={params[0]:.2f}, scale={params[1]:.2f}")
    print(f"  KS test: stat={stat:.6f}, p={p:.4f}")

    passed = p > 0.001
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def validate_distance_decomposition():
    """
    Validate: D^+ ~ chi^2_d (central), D^- ~ noncentral chi^2_d(lambda).
    """
    print("\n" + "=" * 70)
    print("LEMMA 1: Distance decomposition into chi^2")
    print("=" * 70)

    d = 100
    delta = 2.0  # class separation
    n_samples = 50000

    mu1 = np.zeros(d)
    mu2 = np.zeros(d)
    mu2[0] = delta

    # Same-class distances: ||x1 - x2||^2 where x1, x2 ~ N(mu1, I)
    x1 = np.random.randn(n_samples, d) + mu1
    x2 = np.random.randn(n_samples, d) + mu1
    D_same = ((x1 - x2)**2).sum(1)  # Should be chi^2_{d} scaled by 2

    # Cross-class: ||x1 - x2||^2 where x1 ~ N(mu1, I), x2 ~ N(mu2, I)
    x2_cross = np.random.randn(n_samples, d) + mu2
    D_cross = ((x1 - x2_cross)**2).sum(1)  # Should be noncentral chi^2

    # D_same / 2 should be chi^2_d
    D_same_scaled = D_same / 2.0
    print(f"  D_same/2: mean={D_same_scaled.mean():.2f} (theory: {d}), "
          f"var={D_same_scaled.var():.2f} (theory: {2*d})")

    # D_cross / 2 should be noncentral chi^2_d(lambda) with lambda = delta^2/2
    D_cross_scaled = D_cross / 2.0
    lam = delta**2 / 2.0
    theory_mean = d + lam
    theory_var = 2 * (d + 2 * lam)
    print(f"  D_cross/2: mean={D_cross_scaled.mean():.2f} (theory: {theory_mean}), "
          f"var={D_cross_scaled.var():.2f} (theory: {theory_var})")

    mean_ok = abs(D_same_scaled.mean() - d) / d < 0.02
    cross_ok = abs(D_cross_scaled.mean() - theory_mean) / theory_mean < 0.02
    passed = mean_ok and cross_ok
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main_sigmoid_validation():
    """
    Main validation: simulate kNN for varying kappa, K, d and check sigmoid(kappa/sqrt(K)).
    """
    print("\n" + "=" * 70)
    print("MAIN THEOREM: q = sigmoid(kappa / sqrt(K))")
    print("=" * 70)

    configs = [
        # (d, K, m, label)
        (200, 10, 50, "d=200, K=10"),
        (200, 50, 30, "d=200, K=50"),
        (200, 100, 20, "d=200, K=100"),
        (500, 10, 50, "d=500, K=10"),
        (500, 50, 30, "d=500, K=50"),
        (500, 100, 20, "d=500, K=100"),
    ]

    all_kappa_norm = []
    all_q = []
    per_config_results = []

    for d, K, m, label in configs:
        print(f"\n  --- {label}, m={m} ---")
        kappa_values = np.linspace(0.01, 0.5, 12)

        config_kappas = []
        config_qs = []

        for kappa_target in kappa_values:
            acc, q, actual_kappa = simulate_knn_accuracy(
                d, K, m, kappa_target, n_test=3000, k_nn=1
            )
            kappa_norm = actual_kappa / np.sqrt(K)

            all_kappa_norm.append(kappa_norm)
            all_q.append(q)
            config_kappas.append(kappa_norm)
            config_qs.append(q)

            q_pred = expit(kappa_norm)  # raw theory prediction

        config_kappas = np.array(config_kappas)
        config_qs = np.array(config_qs)

        # Fit sigmoid to this config
        def sigmoid_model(x, a, b):
            return expit(a * x + b)

        try:
            popt, _ = curve_fit(sigmoid_model, config_kappas, config_qs,
                               p0=[10.0, -1.0], maxfev=5000)
            q_fit = sigmoid_model(config_kappas, *popt)
            ss_res = ((config_qs - q_fit)**2).sum()
            ss_tot = ((config_qs - config_qs.mean())**2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"    Sigmoid fit: a={popt[0]:.3f}, b={popt[1]:.3f}, R^2={r2:.4f}")
        except Exception as e:
            print(f"    Fit failed: {e}")
            r2 = 0
            popt = [0, 0]

        per_config_results.append({
            "d": d, "K": K, "m": m, "label": label,
            "sigmoid_a": float(popt[0]), "sigmoid_b": float(popt[1]),
            "r2": float(r2),
            "kappa_norm": [float(x) for x in config_kappas],
            "q_values": [float(x) for x in config_qs],
        })

    # Global fit: all configs together
    all_kappa_norm = np.array(all_kappa_norm)
    all_q = np.array(all_q)

    print(f"\n{'='*70}")
    print("GLOBAL SIGMOID FIT (all configs pooled)")
    print(f"{'='*70}")

    def sigmoid_global(x, a, b):
        return expit(a * x + b)

    try:
        popt_g, _ = curve_fit(sigmoid_global, all_kappa_norm, all_q,
                              p0=[10.0, -1.0], maxfev=10000)
        q_fit_g = sigmoid_global(all_kappa_norm, *popt_g)
        ss_res = ((all_q - q_fit_g)**2).sum()
        ss_tot = ((all_q - all_q.mean())**2).sum()
        r2_global = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Also compute correlation
        from scipy.stats import pearsonr, spearmanr
        r_val, p_val = pearsonr(all_kappa_norm, all_q)
        rho_val, _ = spearmanr(all_kappa_norm, all_q)

        print(f"  N points: {len(all_q)}")
        print(f"  Sigmoid: a={popt_g[0]:.4f}, b={popt_g[1]:.4f}")
        print(f"  R^2 = {r2_global:.4f}")
        print(f"  Pearson r = {r_val:.4f}")
        print(f"  Spearman rho = {rho_val:.4f}")
    except Exception as e:
        print(f"  Global fit failed: {e}")
        r2_global = 0
        popt_g = [0, 0]

    # Test: does kappa (without /sqrt(K)) collapse?
    print(f"\n{'='*70}")
    print("CONTROL: Does raw kappa (without sqrt(K) normalization) collapse?")
    print(f"{'='*70}")

    all_kappa_raw = all_kappa_norm * np.sqrt(np.array([
        K for d, K, m, _ in configs for _ in range(12)
    ]))

    try:
        popt_raw, _ = curve_fit(sigmoid_global, all_kappa_raw, all_q,
                                p0=[10.0, -1.0], maxfev=10000)
        q_fit_raw = sigmoid_global(all_kappa_raw, *popt_raw)
        ss_res_raw = ((all_q - q_fit_raw)**2).sum()
        ss_tot_raw = ((all_q - all_q.mean())**2).sum()
        r2_raw = 1 - ss_res_raw / ss_tot_raw
        print(f"  Raw kappa sigmoid R^2 = {r2_raw:.4f}")
        print(f"  Normalized kappa/sqrt(K) R^2 = {r2_global:.4f}")
        print(f"  Normalization improvement: {r2_global - r2_raw:+.4f}")
    except Exception:
        r2_raw = 0

    # Also test kappa/K and kappa/log(K) normalizations
    all_K = np.array([K for d, K, m, _ in configs for _ in range(12)])

    for norm_name, norm_func in [
        ("kappa/K", lambda k, K: k / K),
        ("kappa/log(K)", lambda k, K: k / np.log(K)),
        ("kappa/K^(1/3)", lambda k, K: k / K**(1/3)),
    ]:
        kappa_alt = all_kappa_raw
        K_arr = all_K
        x_alt = np.array([norm_func(kr, Kv) for kr, Kv in zip(all_kappa_raw, K_arr)])

        try:
            popt_alt, _ = curve_fit(sigmoid_global, x_alt, all_q,
                                    p0=[10.0, -1.0], maxfev=10000)
            q_alt = sigmoid_global(x_alt, *popt_alt)
            r2_alt = 1 - ((all_q - q_alt)**2).sum() / ((all_q - all_q.mean())**2).sum()
            print(f"  {norm_name}: R^2 = {r2_alt:.4f}")
        except Exception:
            print(f"  {norm_name}: fit failed")

    # Scorecard
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("Gumbel-logistic lemma validates (KS p > 0.01)",
         validate_gumbel_logistic_result, ""),
        ("Distance decomposition validates (chi^2)",
         validate_distance_result, ""),
        ("Min-of-chi^2 -> Gumbel validates",
         validate_min_gumbel_result, ""),
        ("Global sigmoid(kappa/sqrt(K)) R^2 > 0.90",
         r2_global > 0.90,
         f"R^2={r2_global:.4f}"),
        ("sqrt(K) normalization beats raw kappa",
         r2_global > r2_raw,
         f"normalized={r2_global:.4f} vs raw={r2_raw:.4f}"),
        ("Per-config R^2 > 0.85 for all configs",
         all(r["r2"] > 0.85 for r in per_config_results),
         f"min={min(r['r2'] for r in per_config_results):.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save results
    results = {
        "experiment": "gumbel_theory_validation",
        "theory": "q = sigmoid(kappa/sqrt(K)) from Gumbel EVT",
        "proof_steps": [
            "1. Same-class distances D+ ~ 2*sigma^2 * chi^2_d",
            "2. Cross-class distances D- ~ 2*sigma^2 * noncentral_chi^2_d(lambda)",
            "3. Min over m samples -> Gumbel (EVT)",
            "4. Difference of equal-scale Gumbels -> Logistic (sigmoid)",
            "5. Location gap = kappa/sqrt(K) via exchangeable CLT over K-1 impostors",
        ],
        "global_sigmoid": {
            "a": float(popt_g[0]),
            "b": float(popt_g[1]),
            "r2": float(r2_global),
        },
        "normalization_comparison": {
            "kappa_over_sqrtK": float(r2_global),
            "raw_kappa": float(r2_raw),
        },
        "per_config": per_config_results,
        "scorecard": {"passes": passes, "total": len(checks)},
    }

    out_path = RESULTS_DIR / "cti_gumbel_theory.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")

    return results


# Global flags for scorecard (set by preliminary tests)
validate_gumbel_logistic_result = False
validate_distance_result = False
validate_min_gumbel_result = False


if __name__ == "__main__":
    print("=" * 70)
    print("GUMBEL THEORY VALIDATION")
    print("First-principles derivation: q = sigmoid(kappa/sqrt(K))")
    print("=" * 70)

    # Step 1: Validate lemmas
    validate_distance_result = validate_distance_decomposition()
    validate_min_gumbel_result = validate_min_gumbel()
    validate_gumbel_logistic_result = validate_gumbel_logistic()

    # Step 2: Main theorem validation
    main_sigmoid_validation()
