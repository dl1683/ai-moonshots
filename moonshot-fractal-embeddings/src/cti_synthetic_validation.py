#!/usr/bin/env python
"""
SYNTHETIC GAUSSIAN VALIDATION OF kappa_c(K, n, d) THEORY

The Gaussian-cluster kNN theory predicts:
  kappa_c = (c_- - c_+) / (sqrt(d) - c_-)
  where c_+ = 2*sqrt(log(n/K)), c_- = 2*sqrt(log(n*(K-1)/K))

This script creates CONTROLLED synthetic Gaussian mixtures where we
KNOW the true kappa, and tests whether:
1. The sigmoid relationship q = sigmoid(kappa) holds exactly
2. The predicted kappa_c matches the empirical transition point
3. The n, d, K dependencies all match theory

If the theory is exact for Gaussians, it proves the framework is correct
and the gap to real neural networks is just non-Gaussianity.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from scipy.special import expit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def sigmoid(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def kappac_theory(K, n, d):
    """Theoretical kappa_c from Gaussian-cluster model."""
    n_same = max(n / K, 2)
    n_diff = max(n * (K - 1) / K, 2)

    c_plus = 2 * np.sqrt(np.log(n_same))
    c_minus = 2 * np.sqrt(np.log(n_diff))

    denominator = np.sqrt(d) - c_minus
    if denominator <= 0:
        return 10.0

    return (c_minus - c_plus) / denominator


def generate_gaussian_clusters(K, n, d, kappa, seed=42):
    """Generate synthetic Gaussian clusters with controlled kappa.

    kappa = tr(Sigma_B) / tr(Sigma_W)

    We set Sigma_W = I_d (identity), so tr(Sigma_W) = d.
    Then Sigma_B needs tr(Sigma_B) = kappa * d.

    Class means: mu_k ~ N(0, (kappa*d/d)*I_d) = N(0, kappa*I_d)
    This gives E[tr(S_B)] ~ kappa * d (in expectation).
    """
    rng = np.random.RandomState(seed)

    # Class assignments (balanced)
    labels = np.repeat(np.arange(K), n // K)
    if len(labels) < n:
        labels = np.concatenate([labels, rng.randint(0, K, n - len(labels))])
    labels = labels[:n]

    # Class means: each component of mu_k ~ N(0, sqrt(kappa))
    # So ||mu_k||^2 ~ kappa * chi^2(d), and tr(S_B) ~ kappa * d
    class_means = rng.randn(K, d) * np.sqrt(kappa)

    # Generate points: x_i = mu_{y_i} + epsilon, epsilon ~ N(0, I_d)
    X = np.zeros((n, d))
    for k in range(K):
        mask = labels == k
        n_k = mask.sum()
        X[mask] = class_means[k] + rng.randn(n_k, d)

    return X, labels


def compute_knn_accuracy(X, labels, k=1):
    """Compute kNN accuracy with leave-one-out style."""
    knn = KNeighborsClassifier(n_neighbors=k + 1)  # +1 because point is its own neighbor
    knn.fit(X, labels)
    # For each point, find k+1 nearest neighbors (including self),
    # remove self, check if nearest remaining is same class
    dists, indices = knn.kneighbors(X)
    correct = 0
    for i in range(len(X)):
        # Skip self (first neighbor is always self with dist=0)
        neighbors = indices[i, 1:k+1]
        neighbor_labels = labels[neighbors]
        # Majority vote
        if k == 1:
            pred = neighbor_labels[0]
        else:
            counts = np.bincount(neighbor_labels, minlength=max(labels) + 1)
            pred = counts.argmax()
        if pred == labels[i]:
            correct += 1
    return correct / len(X)


def compute_kappa(X, labels):
    """Compute kappa = tr(S_B)/tr(S_W) from data."""
    classes = np.unique(labels)
    grand_mean = X.mean(axis=0)

    S_W = 0.0
    S_B = 0.0
    for c in classes:
        X_c = X[labels == c]
        n_c = len(X_c)
        class_mean = X_c.mean(axis=0)

        # Within-class scatter (trace only)
        diff = X_c - class_mean
        S_W += np.sum(diff ** 2)

        # Between-class scatter (trace only)
        mean_diff = class_mean - grand_mean
        S_B += n_c * np.sum(mean_diff ** 2)

    return S_B / max(S_W, 1e-10)


def main():
    print("=" * 70)
    print("SYNTHETIC GAUSSIAN VALIDATION OF kappa_c(K, n, d) THEORY")
    print("=" * 70)

    all_results = []

    # ============================================================
    # EXPERIMENT 1: Vary kappa for fixed K, n, d
    # ============================================================
    print(f"\n{'='*70}")
    print("EXP 1: Vary kappa (K=20, n=2000, d=500)")
    print(f"{'='*70}")

    K, n, d = 20, 2000, 500
    kappa_range = np.logspace(-2, 1.5, 30)  # 0.01 to ~30
    kc_theory = kappac_theory(K, n, d)
    print(f"Theoretical kappa_c = {kc_theory:.4f}")

    kappas_true = []
    kappas_measured = []
    knns = []

    for kappa_true in kappa_range:
        X, labels = generate_gaussian_clusters(K, n, d, kappa_true)
        knn_acc = compute_knn_accuracy(X, labels, k=1)
        kappa_meas = compute_kappa(X, labels)

        kappas_true.append(kappa_true)
        kappas_measured.append(kappa_meas)
        knns.append(knn_acc)
        print(f"  kappa_true={kappa_true:.4f}, kappa_meas={kappa_meas:.4f}, "
              f"kNN={knn_acc:.4f}")

    kappas_true = np.array(kappas_true)
    kappas_meas = np.array(kappas_measured)
    knns_arr = np.array(knns)

    # Normalize quality
    q = (knns_arr - 1.0 / K) / (1.0 - 1.0 / K)

    # Fit sigmoid
    try:
        popt, _ = curve_fit(sigmoid, kappas_meas, q,
                            p0=[0.9, 5, kc_theory, 0.0], maxfev=10000)
        pred = sigmoid(kappas_meas, *popt)
        ss_tot = np.sum((q - q.mean()) ** 2)
        r2 = 1 - np.sum((q - pred) ** 2) / ss_tot
        kc_empirical = popt[2]
    except Exception as e:
        print(f"  Sigmoid fit failed: {e}")
        r2 = 0.0
        kc_empirical = float('nan')

    rho, _ = spearmanr(kappas_meas, q)
    print(f"\n  Sigmoid R^2 = {r2:.4f}")
    print(f"  Spearman rho = {rho:.4f}")
    print(f"  kappa_c theory = {kc_theory:.4f}")
    print(f"  kappa_c empirical = {kc_empirical:.4f}")
    print(f"  kappa_c error = {abs(kc_theory - kc_empirical):.4f}")

    exp1 = {
        "K": K, "n": n, "d": d,
        "kc_theory": float(kc_theory),
        "kc_empirical": float(kc_empirical),
        "kc_error": float(abs(kc_theory - kc_empirical)),
        "sigmoid_r2": float(r2),
        "rho": float(rho),
    }

    # ============================================================
    # EXPERIMENT 2: Vary K for fixed n, d
    # ============================================================
    print(f"\n{'='*70}")
    print("EXP 2: Vary K (n=2000, d=500) — Theory predicts kappa_c(K)")
    print(f"{'='*70}")

    n, d = 2000, 500
    K_values = [5, 10, 20, 50, 100, 200]
    kc_theories = []
    kc_empiricals = []
    r2s = []

    for K in K_values:
        kc_th = kappac_theory(K, n, d)
        kc_theories.append(kc_th)

        # Scan kappa around the theoretical critical point
        kappa_range_k = np.logspace(
            np.log10(max(kc_th * 0.01, 0.001)),
            np.log10(max(kc_th * 100, 10)),
            25
        )

        kappas_m = []
        qs_k = []
        for kappa_true in kappa_range_k:
            X, labels = generate_gaussian_clusters(K, n, d, kappa_true)
            knn_acc = compute_knn_accuracy(X, labels, k=1)
            kappa_m = compute_kappa(X, labels)
            q_val = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)
            kappas_m.append(kappa_m)
            qs_k.append(q_val)

        kappas_m = np.array(kappas_m)
        qs_k = np.array(qs_k)

        try:
            popt_k, _ = curve_fit(sigmoid, kappas_m, qs_k,
                                  p0=[0.9, 5, kc_th, 0.0], maxfev=10000)
            pred_k = sigmoid(kappas_m, *popt_k)
            ss_tot_k = np.sum((qs_k - qs_k.mean()) ** 2)
            r2_k = 1 - np.sum((qs_k - pred_k) ** 2) / max(ss_tot_k, 1e-10)
            kc_emp = popt_k[2]
        except Exception:
            r2_k = 0.0
            kc_emp = float('nan')

        kc_empiricals.append(kc_emp)
        r2s.append(r2_k)
        print(f"  K={K:>4}: kc_theory={kc_th:.4f}, kc_emp={kc_emp:.4f}, "
              f"err={abs(kc_th-kc_emp):.4f}, R^2={r2_k:.4f}")

    kc_theories = np.array(kc_theories)
    kc_empiricals = np.array(kc_empiricals)
    valid = ~np.isnan(kc_empiricals)

    if valid.sum() >= 3:
        rho_kc, _ = spearmanr(kc_theories[valid], kc_empiricals[valid])
        r_kc, _ = pearsonr(kc_theories[valid], kc_empiricals[valid])
        mae_kc = float(np.mean(np.abs(kc_theories[valid] - kc_empiricals[valid])))
    else:
        rho_kc = r_kc = mae_kc = float('nan')

    print(f"\n  kappa_c prediction: rho={rho_kc:.4f}, r={r_kc:.4f}, MAE={mae_kc:.4f}")

    exp2 = {
        "n": n, "d": d,
        "K_values": [int(k) for k in K_values],
        "kc_theory": [float(x) for x in kc_theories],
        "kc_empirical": [float(x) for x in kc_empiricals],
        "rho": float(rho_kc),
        "r": float(r_kc),
        "mae": float(mae_kc),
        "r2_per_K": [float(x) for x in r2s],
    }

    # ============================================================
    # EXPERIMENT 3: Vary n for fixed K, d
    # ============================================================
    print(f"\n{'='*70}")
    print("EXP 3: Vary n (K=20, d=500) — Theory predicts kappa_c decreases with n")
    print(f"{'='*70}")

    K, d = 20, 500
    n_values = [200, 500, 1000, 2000, 5000, 10000]
    kc_theories_n = []
    kc_empiricals_n = []

    for n in n_values:
        kc_th = kappac_theory(K, n, d)
        kc_theories_n.append(kc_th)

        kappa_range_n = np.logspace(
            np.log10(max(kc_th * 0.01, 0.001)),
            np.log10(max(kc_th * 100, 10)),
            20
        )

        kappas_m = []
        qs_n = []
        for kappa_true in kappa_range_n:
            X, labels = generate_gaussian_clusters(K, n, d, kappa_true)
            knn_acc = compute_knn_accuracy(X, labels, k=1)
            kappa_m = compute_kappa(X, labels)
            q_val = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)
            kappas_m.append(kappa_m)
            qs_n.append(q_val)

        kappas_m = np.array(kappas_m)
        qs_n = np.array(qs_n)

        try:
            popt_n, _ = curve_fit(sigmoid, kappas_m, qs_n,
                                  p0=[0.9, 5, kc_th, 0.0], maxfev=10000)
            kc_emp = popt_n[2]
        except Exception:
            kc_emp = float('nan')

        kc_empiricals_n.append(kc_emp)
        print(f"  n={n:>6}: kc_theory={kc_th:.4f}, kc_emp={kc_emp:.4f}, "
              f"err={abs(kc_th-kc_emp):.4f}")

    kc_theories_n = np.array(kc_theories_n)
    kc_empiricals_n = np.array(kc_empiricals_n)
    valid_n = ~np.isnan(kc_empiricals_n)

    if valid_n.sum() >= 3:
        rho_n, _ = spearmanr(kc_theories_n[valid_n], kc_empiricals_n[valid_n])
        r_n, _ = pearsonr(kc_theories_n[valid_n], kc_empiricals_n[valid_n])
        mae_n = float(np.mean(np.abs(kc_theories_n[valid_n] - kc_empiricals_n[valid_n])))
    else:
        rho_n = r_n = mae_n = float('nan')

    print(f"\n  kappa_c(n) prediction: rho={rho_n:.4f}, r={r_n:.4f}, MAE={mae_n:.4f}")

    exp3 = {
        "K": K, "d": d,
        "n_values": n_values,
        "kc_theory": [float(x) for x in kc_theories_n],
        "kc_empirical": [float(x) for x in kc_empiricals_n],
        "rho": float(rho_n),
        "r": float(r_n),
        "mae": float(mae_n),
    }

    # ============================================================
    # EXPERIMENT 4: Vary d for fixed K, n
    # ============================================================
    print(f"\n{'='*70}")
    print("EXP 4: Vary d (K=20, n=2000) — Theory predicts kappa_c decreases with d")
    print(f"{'='*70}")

    K, n = 20, 2000
    d_values = [50, 100, 200, 500, 1000, 2000]
    kc_theories_d = []
    kc_empiricals_d = []

    for d in d_values:
        kc_th = kappac_theory(K, n, d)
        kc_theories_d.append(kc_th)

        kappa_range_d = np.logspace(
            np.log10(max(kc_th * 0.01, 0.001)),
            np.log10(max(kc_th * 100, 10)),
            20
        )

        kappas_m = []
        qs_d = []
        for kappa_true in kappa_range_d:
            X, labels = generate_gaussian_clusters(K, n, d, kappa_true)
            knn_acc = compute_knn_accuracy(X, labels, k=1)
            kappa_m = compute_kappa(X, labels)
            q_val = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)
            kappas_m.append(kappa_m)
            qs_d.append(q_val)

        kappas_m = np.array(kappas_m)
        qs_d = np.array(qs_d)

        try:
            popt_d, _ = curve_fit(sigmoid, kappas_m, qs_d,
                                  p0=[0.9, 5, kc_th, 0.0], maxfev=10000)
            kc_emp = popt_d[2]
        except Exception:
            kc_emp = float('nan')

        kc_empiricals_d.append(kc_emp)
        print(f"  d={d:>5}: kc_theory={kc_th:.4f}, kc_emp={kc_emp:.4f}, "
              f"err={abs(kc_th-kc_emp):.4f}")

    kc_theories_d = np.array(kc_theories_d)
    kc_empiricals_d = np.array(kc_empiricals_d)
    valid_d = ~np.isnan(kc_empiricals_d)

    if valid_d.sum() >= 3:
        rho_d, _ = spearmanr(kc_theories_d[valid_d], kc_empiricals_d[valid_d])
        r_d, _ = pearsonr(kc_theories_d[valid_d], kc_empiricals_d[valid_d])
        mae_d = float(np.mean(np.abs(kc_theories_d[valid_d] - kc_empiricals_d[valid_d])))
    else:
        rho_d = r_d = mae_d = float('nan')

    print(f"\n  kappa_c(d) prediction: rho={rho_d:.4f}, r={r_d:.4f}, MAE={mae_d:.4f}")

    exp4 = {
        "K": K, "n": n,
        "d_values": d_values,
        "kc_theory": [float(x) for x in kc_theories_d],
        "kc_empirical": [float(x) for x in kc_empiricals_d],
        "rho": float(rho_d),
        "r": float(r_d),
        "mae": float(mae_d),
    }

    # ============================================================
    # EXPERIMENT 5: Full grid — predict kappa_c across (K, n, d)
    # ============================================================
    print(f"\n{'='*70}")
    print("EXP 5: Full grid prediction of kappa_c")
    print(f"{'='*70}")

    grid_configs = [
        (10, 1000, 200),
        (10, 2000, 500),
        (10, 5000, 1000),
        (50, 1000, 200),
        (50, 2000, 500),
        (50, 5000, 1000),
        (100, 1000, 200),
        (100, 2000, 500),
        (100, 5000, 1000),
    ]

    grid_theories = []
    grid_empiricals = []
    grid_labels = []

    for K, n, d in grid_configs:
        kc_th = kappac_theory(K, n, d)
        grid_theories.append(kc_th)

        kappa_range_g = np.logspace(
            np.log10(max(kc_th * 0.01, 0.001)),
            np.log10(max(kc_th * 100, 10)),
            20
        )

        kappas_m = []
        qs_g = []
        for kappa_true in kappa_range_g:
            X, labels_g = generate_gaussian_clusters(K, n, d, kappa_true)
            knn_acc = compute_knn_accuracy(X, labels_g, k=1)
            kappa_m = compute_kappa(X, labels_g)
            q_val = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)
            kappas_m.append(kappa_m)
            qs_g.append(q_val)

        kappas_m = np.array(kappas_m)
        qs_g = np.array(qs_g)

        try:
            popt_g, _ = curve_fit(sigmoid, kappas_m, qs_g,
                                  p0=[0.9, 5, kc_th, 0.0], maxfev=10000)
            kc_emp = popt_g[2]
        except Exception:
            kc_emp = float('nan')

        grid_empiricals.append(kc_emp)
        grid_labels.append(f"K={K},n={n},d={d}")
        print(f"  K={K:>4}, n={n:>5}, d={d:>5}: "
              f"kc_th={kc_th:.4f}, kc_emp={kc_emp:.4f}, err={abs(kc_th-kc_emp):.4f}")

    grid_theories = np.array(grid_theories)
    grid_empiricals = np.array(grid_empiricals)
    valid_g = ~np.isnan(grid_empiricals)

    if valid_g.sum() >= 3:
        rho_g, _ = spearmanr(grid_theories[valid_g], grid_empiricals[valid_g])
        r_g, _ = pearsonr(grid_theories[valid_g], grid_empiricals[valid_g])
        mae_g = float(np.mean(np.abs(grid_theories[valid_g] - grid_empiricals[valid_g])))
    else:
        rho_g = r_g = mae_g = float('nan')

    print(f"\n  Full grid: rho={rho_g:.4f}, r={r_g:.4f}, MAE={mae_g:.4f}")

    exp5 = {
        "configs": [{"K": K, "n": n, "d": d} for K, n, d in grid_configs],
        "kc_theory": [float(x) for x in grid_theories],
        "kc_empirical": [float(x) for x in grid_empiricals],
        "rho": float(rho_g),
        "r": float(r_g),
        "mae": float(mae_g),
    }

    # ============================================================
    # SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("Exp1: Sigmoid R^2 > 0.98 for Gaussian clusters",
         exp1["sigmoid_r2"] > 0.98, f"R^2={exp1['sigmoid_r2']:.4f}"),
        ("Exp1: kappa_c theory-vs-empirical error < 0.02",
         exp1["kc_error"] < 0.02, f"err={exp1['kc_error']:.4f}"),
        ("Exp2: kappa_c(K) rho > 0.95",
         rho_kc > 0.95, f"rho={rho_kc:.4f}"),
        ("Exp3: kappa_c(n) rho > 0.95",
         rho_n > 0.95, f"rho={rho_n:.4f}"),
        ("Exp4: kappa_c(d) rho > 0.95",
         rho_d > 0.95, f"rho={rho_d:.4f}"),
        ("Exp5: Full grid rho > 0.95",
         rho_g > 0.95, f"rho={rho_g:.4f}"),
        ("Exp5: Full grid MAE < 0.05",
         mae_g < 0.05, f"MAE={mae_g:.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "synthetic_gaussian_validation",
        "exp1_vary_kappa": exp1,
        "exp2_vary_K": exp2,
        "exp3_vary_n": exp3,
        "exp4_vary_d": exp4,
        "exp5_full_grid": exp5,
        "scorecard": {
            "passes": passes, "total": len(checks),
            "details": [{"criterion": c, "passed": bool(p), "value": v}
                        for c, p, v in checks],
        },
    }

    out_path = RESULTS_DIR / "cti_synthetic_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
