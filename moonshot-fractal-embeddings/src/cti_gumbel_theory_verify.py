#!/usr/bin/env python
"""
GUMBEL MECHANISM TEST: Prove the sigmoid comes from extreme value theory.

Setup: K-class Gaussian mixture, kappa = tr(S_B)/tr(S_W) near transition.
Question: Is the margin M = min_dist_other - min_dist_same logistic?

Theory (Codex derivation):
  1. Squared distances to same-class neighbors concentrate around 2*tr(Sigma_W)
  2. NN distances (minima) follow Gumbel distribution
  3. Difference of Gumbels = Logistic
  4. P(correct) = P(M > 0) = sigmoid(location/scale)
  5. Location scales as kappa/sqrt(K)

This script uses parameters in the TRANSITION REGIME (kappa ~ kappa_c)
where classification is neither trivial nor impossible.
"""

import json
import sys
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import expit as sigmoid_fn
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def generate_mixture(K, n, d, kappa, seed=42):
    """Generate K-class Gaussian mixture with controlled kappa.
    mu_k ~ N(0, sqrt(kappa)*I), x|y=k ~ N(mu_k, I)."""
    rng = np.random.RandomState(seed)
    labels = np.repeat(np.arange(K), n // K)[:n]
    if len(labels) < n:
        labels = np.concatenate([labels, rng.randint(0, K, n - len(labels))])
    labels = labels[:n]

    class_means = rng.randn(K, d) * np.sqrt(kappa)
    X = np.zeros((n, d))
    for k in range(K):
        mask = labels == k
        X[mask] = class_means[k] + rng.randn(mask.sum(), d)
    return X, labels, class_means


def compute_kappa_and_knn(X, labels):
    """Compute kappa and LOO-kNN accuracy."""
    classes = np.unique(labels)
    K = len(classes)
    grand_mean = X.mean(axis=0)
    tr_sb, tr_sw = 0.0, 0.0
    for c in classes:
        Xc = X[labels == c]
        mu_c = Xc.mean(axis=0)
        tr_sb += len(Xc) * np.sum((mu_c - grand_mean)**2)
        tr_sw += np.sum((Xc - mu_c)**2)
    kappa = tr_sb / max(tr_sw, 1e-10)

    # LOO-kNN
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, labels)
    _, indices = knn.kneighbors(X)
    correct = sum(labels[indices[i, 1]] == labels[i] for i in range(len(X)))
    knn_acc = correct / len(X)

    return kappa, knn_acc


def collect_margins(X, labels):
    """For each point, compute M = min_dist_other - min_dist_same (LOO)."""
    n = len(X)
    classes = np.unique(labels)
    margins = []

    class_indices = {k: np.where(labels == k)[0] for k in classes}

    for i in range(n):
        y = labels[i]
        xi = X[i]

        same_idx = class_indices[y]
        same_idx = same_idx[same_idx != i]
        if len(same_idx) == 0:
            continue
        same_dists = np.sum((X[same_idx] - xi)**2, axis=1)
        same_min = np.min(same_dists)

        other_min = float('inf')
        for k in classes:
            if k == y:
                continue
            other_idx = class_indices[k]
            if len(other_idx) == 0:
                continue
            other_dists = np.sum((X[other_idx] - xi)**2, axis=1)
            other_min = min(other_min, np.min(other_dists))

        margins.append(other_min - same_min)

    return np.array(margins)


def test_margin_distribution(margins, label=""):
    """Test whether margins follow logistic, normal, or Laplace."""
    loc_l, scale_l = stats.logistic.fit(margins)
    ks_l, p_l = stats.kstest(margins, 'logistic', args=(loc_l, scale_l))

    mu_n, sigma_n = stats.norm.fit(margins)
    ks_n, p_n = stats.kstest(margins, 'norm', args=(mu_n, sigma_n))

    loc_lap, scale_lap = stats.laplace.fit(margins)
    ks_lap, p_lap = stats.kstest(margins, 'laplace', args=(loc_lap, scale_lap))

    ll_logistic = np.sum(stats.logistic.logpdf(margins, loc_l, scale_l))
    ll_normal = np.sum(stats.norm.logpdf(margins, mu_n, sigma_n))
    ll_laplace = np.sum(stats.laplace.logpdf(margins, loc_lap, scale_lap))

    aic_logistic = 2*2 - 2*ll_logistic
    aic_normal = 2*2 - 2*ll_normal
    aic_laplace = 2*2 - 2*ll_laplace

    best_aic = min(aic_logistic, aic_normal, aic_laplace)
    if best_aic == aic_logistic:
        winner = "logistic"
    elif best_aic == aic_normal:
        winner = "normal"
    else:
        winner = "laplace"

    kurt = float(stats.kurtosis(margins, fisher=True))

    result = {
        "logistic": {"loc": float(loc_l), "scale": float(scale_l),
                      "ks_p": float(p_l), "aic": float(aic_logistic), "ll": float(ll_logistic)},
        "normal": {"mu": float(mu_n), "sigma": float(sigma_n),
                    "ks_p": float(p_n), "aic": float(aic_normal), "ll": float(ll_normal)},
        "laplace": {"loc": float(loc_lap), "scale": float(scale_lap),
                     "ks_p": float(p_lap), "aic": float(aic_laplace), "ll": float(ll_laplace)},
        "winner_aic": winner,
        "kurtosis": kurt,
        "theory_kurtosis": 1.2,
    }

    if label:
        print(f"  {label}")
    print(f"    Kurtosis: {kurt:.3f} (logistic=1.2, normal=0, Laplace=3)")
    print(f"    AIC: logistic={aic_logistic:.1f}, normal={aic_normal:.1f}, "
          f"laplace={aic_laplace:.1f}")
    print(f"    KS p: logistic={p_l:.4f}, normal={p_n:.4f}, laplace={p_lap:.4f}")
    print(f"    WINNER (AIC): {winner}")

    return result


def sigmoid_4p(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def main():
    print("=" * 70)
    print("GUMBEL MECHANISM TEST")
    print("Does the sigmoid come from logistic margin distributions?")
    print("=" * 70)

    all_results = {}

    # ============================================================
    # TEST 1: Margin distribution at transition (single K, sweep kappa)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: Margin distribution across kappa values")
    print("=" * 70)

    K, n, d = 20, 2000, 500
    kappa_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

    test1_results = []
    logistic_wins_count = 0
    total_tests = 0

    for kappa in kappa_values:
        X, labels, means = generate_mixture(K, n, d, kappa, seed=42)
        kappa_meas, knn_acc = compute_kappa_and_knn(X, labels)
        q = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

        print(f"\n  kappa_target={kappa:.3f}, kappa_meas={kappa_meas:.4f}, "
              f"kNN={knn_acc:.4f}, q={q:.4f}")

        margins = collect_margins(X, labels)
        dist_result = test_margin_distribution(
            margins, label=f"kappa={kappa:.3f}")

        frac_correct = float(np.mean(margins > 0))

        if dist_result["winner_aic"] == "logistic":
            logistic_wins_count += 1
        total_tests += 1

        test1_results.append({
            "kappa_target": float(kappa),
            "kappa_measured": float(kappa_meas),
            "knn_acc": float(knn_acc),
            "q": float(q),
            "frac_correct_from_margin": float(frac_correct),
            "margin_mean": float(np.mean(margins)),
            "margin_std": float(np.std(margins)),
            "distribution_test": dist_result,
        })

    all_results["test1_margin_distributions"] = test1_results
    all_results["test1_logistic_win_rate"] = logistic_wins_count / total_tests

    print(f"\n  Logistic win rate: {logistic_wins_count}/{total_tests} "
          f"= {logistic_wins_count/total_tests:.1%}")

    # ============================================================
    # TEST 2: Does logistic location scale as kappa?
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: Logistic location vs kappa")
    print("=" * 70)

    locations = []
    scales = []
    kappas_m = []
    qs = []
    for r in test1_results:
        locations.append(r["distribution_test"]["logistic"]["loc"])
        scales.append(r["distribution_test"]["logistic"]["scale"])
        kappas_m.append(r["kappa_measured"])
        qs.append(r["q"])

    locations = np.array(locations)
    scales = np.array(scales)
    kappas_m = np.array(kappas_m)
    qs = np.array(qs)

    rho_loc_kappa, p_loc_kappa = stats.spearmanr(kappas_m, locations)
    r_loc_kappa, _ = stats.pearsonr(kappas_m, locations)

    print(f"  loc vs kappa: rho={rho_loc_kappa:.4f}, r={r_loc_kappa:.4f}")

    logit_ratios = locations / np.clip(scales, 1e-10, None)
    q_from_logistic = sigmoid_fn(logit_ratios)
    rho_q, p_q = stats.spearmanr(q_from_logistic, qs)
    mae_q = float(np.mean(np.abs(q_from_logistic - qs)))

    print(f"  sigmoid(loc/scale) vs observed q: rho={rho_q:.4f}, MAE={mae_q:.4f}")

    all_results["test2_location_scaling"] = {
        "rho_loc_kappa": float(rho_loc_kappa),
        "r_loc_kappa": float(r_loc_kappa),
        "rho_q_prediction": float(rho_q),
        "mae_q_prediction": float(mae_q),
    }

    # ============================================================
    # TEST 3: sqrt(K) scaling of the logistic scale
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: How does the logistic scale change with K?")
    print("=" * 70)

    n, d = 2000, 500
    K_values = [5, 10, 20, 50, 100]
    kappa_fixed = 0.1

    test3_results = []
    for K in K_values:
        X, labels, means = generate_mixture(K, n, d, kappa_fixed, seed=42)
        kappa_meas, knn_acc = compute_kappa_and_knn(X, labels)
        q = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

        margins = collect_margins(X, labels)
        loc_l, scale_l = stats.logistic.fit(margins)

        test3_results.append({
            "K": K,
            "kappa_measured": float(kappa_meas),
            "knn_acc": float(knn_acc),
            "q": float(q),
            "logistic_loc": float(loc_l),
            "logistic_scale": float(scale_l),
            "sqrt_K": float(np.sqrt(K)),
        })
        print(f"  K={K:4d}: q={q:.4f}, loc={loc_l:.2f}, scale={scale_l:.2f}, "
              f"loc/scale={loc_l/scale_l:.3f}, sqrt(K)={np.sqrt(K):.2f}")

    K_arr = np.array([r["K"] for r in test3_results])
    scale_arr = np.array([r["logistic_scale"] for r in test3_results])
    rho_scale_sqrtK, _ = stats.spearmanr(np.sqrt(K_arr), scale_arr)
    rho_scale_K, _ = stats.spearmanr(K_arr, scale_arr)
    rho_scale_logK, _ = stats.spearmanr(np.log(K_arr), scale_arr)

    print(f"\n  scale vs sqrt(K): rho={rho_scale_sqrtK:.4f}")
    print(f"  scale vs K: rho={rho_scale_K:.4f}")
    print(f"  scale vs log(K): rho={rho_scale_logK:.4f}")

    all_results["test3_K_scaling"] = {
        "results": test3_results,
        "rho_scale_sqrtK": float(rho_scale_sqrtK),
        "rho_scale_K": float(rho_scale_K),
        "rho_scale_logK": float(rho_scale_logK),
    }

    # ============================================================
    # TEST 4: Full sigmoid reconstruction from logistic margins
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 4: Reconstruct q from logistic margin parameters")
    print("=" * 70)

    K_sweep = 20
    n_sweep, d_sweep = 2000, 500
    kappa_sweep = np.logspace(-2, 1, 25)

    kappas_all, qs_all, qs_logistic = [], [], []
    for kappa in kappa_sweep:
        X, labels, means = generate_mixture(K_sweep, n_sweep, d_sweep, kappa, seed=42)
        kappa_meas, knn_acc = compute_kappa_and_knn(X, labels)
        q = (knn_acc - 1.0/K_sweep) / (1.0 - 1.0/K_sweep)

        margins = collect_margins(X, labels)
        loc_l, scale_l = stats.logistic.fit(margins)
        q_logistic = float(sigmoid_fn(loc_l / max(scale_l, 1e-10)))

        kappas_all.append(kappa_meas)
        qs_all.append(q)
        qs_logistic.append(q_logistic)

        print(f"  kappa={kappa_meas:.4f}: q_obs={q:.4f}, q_logistic={q_logistic:.4f}, "
              f"diff={abs(q-q_logistic):.4f}")

    kappas_all = np.array(kappas_all)
    qs_all = np.array(qs_all)
    qs_logistic = np.array(qs_logistic)

    rho_recon, _ = stats.spearmanr(qs_all, qs_logistic)
    mae_recon = float(np.mean(np.abs(qs_all - qs_logistic)))
    r2_recon = 1 - np.sum((qs_all - qs_logistic)**2) / max(np.sum((qs_all - qs_all.mean())**2), 1e-10)

    print(f"\n  Reconstruction: rho={rho_recon:.4f}, MAE={mae_recon:.4f}, R^2={r2_recon:.4f}")

    try:
        popt, _ = curve_fit(sigmoid_4p, kappas_all, qs_all,
                            p0=[0.9, 5, 0.07, 0.0], maxfev=10000)
        pred_sigmoid = sigmoid_4p(kappas_all, *popt)
        r2_sigmoid = 1 - np.sum((qs_all - pred_sigmoid)**2) / max(np.sum((qs_all - qs_all.mean())**2), 1e-10)
        print(f"  4-param sigmoid fit: R^2={r2_sigmoid:.6f}")
    except Exception as e:
        r2_sigmoid = 0.0
        print(f"  Sigmoid fit failed: {e}")

    all_results["test4_reconstruction"] = {
        "rho": float(rho_recon),
        "mae": float(mae_recon),
        "r2_logistic_reconstruction": float(r2_recon),
        "r2_sigmoid_fit": float(r2_sigmoid),
    }

    # ============================================================
    # SCORECARD
    # ============================================================
    print("\n" + "=" * 70)
    print("SCORECARD")
    print("=" * 70)

    checks = []

    c1 = all_results["test1_logistic_win_rate"] > 0.5
    checks.append({
        "criterion": "Logistic beats normal (AIC) in >50% of kappa values",
        "passed": bool(c1),
        "value": f"{all_results['test1_logistic_win_rate']:.1%}"
    })

    c2 = all_results["test2_location_scaling"]["rho_loc_kappa"] > 0.9
    checks.append({
        "criterion": "Logistic location correlates with kappa (rho > 0.9)",
        "passed": bool(c2),
        "value": f"rho={all_results['test2_location_scaling']['rho_loc_kappa']:.4f}"
    })

    c3 = all_results["test2_location_scaling"]["mae_q_prediction"] < 0.05
    checks.append({
        "criterion": "sigmoid(loc/scale) predicts q (MAE < 0.05)",
        "passed": bool(c3),
        "value": f"MAE={all_results['test2_location_scaling']['mae_q_prediction']:.4f}"
    })

    c4 = all_results["test4_reconstruction"]["r2_logistic_reconstruction"] > 0.95
    checks.append({
        "criterion": "Logistic reconstruction R^2 > 0.95",
        "passed": bool(c4),
        "value": f"R^2={all_results['test4_reconstruction']['r2_logistic_reconstruction']:.4f}"
    })

    c5 = all_results["test4_reconstruction"]["r2_sigmoid_fit"] > 0.999
    checks.append({
        "criterion": "4-param sigmoid fit R^2 > 0.999 (Gaussian exactness)",
        "passed": bool(c5),
        "value": f"R^2={all_results['test4_reconstruction']['r2_sigmoid_fit']:.6f}"
    })

    transition_kappas = [r for r in test1_results
                         if 0.01 <= r["kappa_target"] <= 0.2]
    if transition_kappas:
        mean_kurt = np.mean([r["distribution_test"]["kurtosis"]
                             for r in transition_kappas])
        c6 = abs(mean_kurt - 1.2) < 1.0
    else:
        mean_kurt = float('nan')
        c6 = False
    checks.append({
        "criterion": "Mean kurtosis near logistic (1.2 +/- 1.0) at transition",
        "passed": bool(c6),
        "value": f"kurtosis={mean_kurt:.3f}"
    })

    passes = sum(1 for c in checks if c["passed"])
    total = len(checks)

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['criterion']}: {c['value']}")

    print(f"\n  SCORECARD: {passes}/{total}")

    all_results["scorecard"] = {
        "passes": passes,
        "total": total,
        "checks": checks,
    }

    out_path = RESULTS_DIR / "cti_gumbel_mechanism.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
