#!/usr/bin/env python
"""
IMPROVED ZERO-PARAMETER THEORY: Conditional variance correction.

The original theory used UNCONDITIONAL distance variances (8d, 8d+8*delta_sq),
but in LOO-kNN all distances share the same query point x. Conditional on x,
distances are iid with REDUCED variance (6d, 6d+4*delta_sq).

The 25% variance reduction comes from:
  Total Var = E[Var|x] + Var[E|x] = 6d + 2d = 8d
  But conditional Var = 6d (the 2d is shared query noise)

This correction fixes the systematic +12 unit bias in mu_M.
"""

import json
import numpy as np
from scipy import stats
from scipy.special import ndtri
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def theory_predict_q_improved(kappa, K, n_per_class, d_eff):
    """Improved zero-parameter theory using CONDITIONAL variance.

    Key insight: In LOO-kNN, all distances share the same query point x.
    Conditional on x, distances are iid but with REDUCED variance.

    For query x from class y with x = mu_y + eps, eps ~ N(0, I):
    - D_same_i | x ~ ncx2(d, ||eps||^2), ||eps||^2 ~ d
      -> Conditional Mean: 2d, Conditional Var: 6d (was 8d)
    - D_other_i | x ~ ncx2(d, d + delta_sq)
      -> Conditional Mean: 2d + delta_sq, Conditional Var: 6d + 4*delta_sq (was 8d + 8*delta_sq)
    """
    if d_eff < 2:
        return 0.5
    if kappa < 1e-10:
        return 0.0

    n = max(int(n_per_class), 2)
    m = max(int((K - 1) * n_per_class), 2)

    delta_sq = 2.0 * kappa * d_eff

    # CONDITIONAL variances (key correction)
    mu_s = 2.0 * d_eff
    sigma_s = np.sqrt(6.0 * d_eff)          # was sqrt(8*d)

    mu_o = 2.0 * d_eff + delta_sq
    sigma_o = np.sqrt(6.0 * d_eff + 4.0 * delta_sq)  # was sqrt(8*d + 8*delta_sq)

    # Order statistics (Normal approximation)
    p_n = 1.0 / (n + 1)
    z_n = ndtri(p_n)
    phi_z_n = stats.norm.pdf(z_n)
    mu_s_min = mu_s + sigma_s * z_n
    tau_s = sigma_s / (n * phi_z_n) if phi_z_n > 1e-20 else sigma_s

    p_m = 1.0 / (m + 1)
    z_m = ndtri(p_m)
    phi_z_m = stats.norm.pdf(z_m)
    mu_o_min = mu_o + sigma_o * z_m
    tau_o = sigma_o / (m * phi_z_m) if phi_z_m > 1e-20 else sigma_o

    mu_M = mu_o_min - mu_s_min
    sigma_M = np.sqrt(tau_o**2 + tau_s**2)

    if sigma_M < 1e-20:
        return 1.0 if mu_M > 0 else 0.0

    z = mu_M / sigma_M
    q = float(stats.norm.cdf(z))
    return np.clip(q, 0.0, 1.0)


def theory_predict_q_original(kappa, K, n_per_class, d_eff):
    """Original Normal-approximation theory (UNCONDITIONAL variance)."""
    if d_eff < 2:
        return 0.5
    if kappa < 1e-10:
        return 0.0

    n = max(int(n_per_class), 2)
    m = max(int((K - 1) * n_per_class), 2)

    mu_s = 2.0 * d_eff
    sigma_s = np.sqrt(8.0 * d_eff)
    delta_sq = 2.0 * kappa * d_eff
    mu_o = 2.0 * d_eff + delta_sq
    sigma_o = np.sqrt(8.0 * d_eff + 8.0 * delta_sq)

    p_n = 1.0 / (n + 1)
    z_n = ndtri(p_n)
    phi_z_n = stats.norm.pdf(z_n)
    mu_s_min = mu_s + sigma_s * z_n
    tau_s = sigma_s / (n * phi_z_n) if phi_z_n > 1e-20 else sigma_s

    p_m = 1.0 / (m + 1)
    z_m = ndtri(p_m)
    phi_z_m = stats.norm.pdf(z_m)
    mu_o_min = mu_o + sigma_o * z_m
    tau_o = sigma_o / (m * phi_z_m) if phi_z_m > 1e-20 else sigma_o

    mu_M = mu_o_min - mu_s_min
    sigma_M = np.sqrt(tau_o**2 + tau_s**2)

    if sigma_M < 1e-20:
        return 1.0 if mu_M > 0 else 0.0

    z = mu_M / sigma_M
    q = float(stats.norm.cdf(z))
    return np.clip(q, 0.0, 1.0)


def generate_mixture(K, n, d, kappa, seed=42):
    """Generate K-class Gaussian mixture."""
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


def compute_kappa_knn(X, labels):
    """Compute kappa and kNN accuracy."""
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

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, labels)
    _, indices = knn.kneighbors(X)
    correct = sum(labels[indices[i, 1]] == labels[i] for i in range(len(X)))
    knn_acc = correct / len(X)
    n_per = min(np.bincount(labels))

    return kappa, knn_acc, n_per


def main():
    print("=" * 70)
    print("IMPROVED ZERO-PARAMETER THEORY: Conditional Variance Correction")
    print("=" * 70)

    all_results = {}

    # ================================================================
    # TEST 1: Synthetic Gaussians (K=20, n=2000, d=500)
    # ================================================================
    print("\nTEST 1: Synthetic Gaussians (K=20, n=2000, d=500)")
    print("-" * 60)

    K, n, d = 20, 2000, 500
    kappa_range = np.logspace(-2, 1, 30)

    test1_results = []
    for kappa_true in kappa_range:
        X, labels = generate_mixture(K, n, d, kappa_true, seed=42)
        kappa_meas, knn_acc, n_per = compute_kappa_knn(X, labels)
        q_obs = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

        q_imp = theory_predict_q_improved(kappa_meas, K, n_per, d)
        q_orig = theory_predict_q_original(kappa_meas, K, n_per, d)

        test1_results.append({
            "kappa": float(kappa_meas),
            "q_obs": float(q_obs),
            "q_improved": float(q_imp),
            "q_original": float(q_orig),
            "err_improved": float(abs(q_obs - q_imp)),
            "err_original": float(abs(q_obs - q_orig)),
        })

        marker = "***" if abs(q_obs - q_imp) < abs(q_obs - q_orig) else "   "
        print(f"  {marker} kappa={kappa_meas:.4f}: q_obs={q_obs:.4f}, "
              f"improved={q_imp:.4f} (err={abs(q_obs-q_imp):.4f}), "
              f"original={q_orig:.4f} (err={abs(q_obs-q_orig):.4f})")

    # Summary
    q_obs_arr = np.array([r["q_obs"] for r in test1_results])
    q_imp_arr = np.array([r["q_improved"] for r in test1_results])
    q_orig_arr = np.array([r["q_original"] for r in test1_results])

    mae_imp = float(np.mean([r["err_improved"] for r in test1_results]))
    mae_orig = float(np.mean([r["err_original"] for r in test1_results]))

    ss_tot = np.sum((q_obs_arr - q_obs_arr.mean())**2)
    r2_imp = 1 - np.sum((q_obs_arr - q_imp_arr)**2) / max(ss_tot, 1e-10)
    r2_orig = 1 - np.sum((q_obs_arr - q_orig_arr)**2) / max(ss_tot, 1e-10)

    rho_imp, _ = stats.spearmanr(q_obs_arr, q_imp_arr)

    print(f"\n  IMPROVED:  MAE={mae_imp:.4f}, R^2={r2_imp:.4f}")
    print(f"  ORIGINAL:  MAE={mae_orig:.4f}, R^2={r2_orig:.4f}")

    all_results["test1"] = {
        "improved": {"mae": mae_imp, "r2": float(r2_imp)},
        "original": {"mae": mae_orig, "r2": float(r2_orig)},
    }

    # ================================================================
    # TEST 2: Vary K
    # ================================================================
    print("\nTEST 2: Vary K (n=2000, d=500, kappa=0.1)")
    print("-" * 60)

    K_values = [5, 10, 20, 50, 100, 200]
    n_fixed, d_fixed = 2000, 500

    test2_results = []
    for K_val in K_values:
        X, labels = generate_mixture(K_val, n_fixed, d_fixed, 0.1, seed=42)
        kappa_meas, knn_acc, n_per = compute_kappa_knn(X, labels)
        q_obs = (knn_acc - 1.0/K_val) / (1.0 - 1.0/K_val)

        q_imp = theory_predict_q_improved(kappa_meas, K_val, n_per, d_fixed)
        q_orig = theory_predict_q_original(kappa_meas, K_val, n_per, d_fixed)

        test2_results.append({
            "K": int(K_val), "kappa": float(kappa_meas), "n_per": int(n_per),
            "q_obs": float(q_obs),
            "q_improved": float(q_imp), "q_original": float(q_orig),
            "err_improved": float(abs(q_obs - q_imp)),
            "err_original": float(abs(q_obs - q_orig)),
        })

        print(f"  K={K_val:4d}: q_obs={q_obs:.4f}, improved={q_imp:.4f} "
              f"(err={abs(q_obs-q_imp):.4f}), original={q_orig:.4f} "
              f"(err={abs(q_obs-q_orig):.4f})")

    all_results["test2"] = test2_results

    # ================================================================
    # TEST 3: Full kappa x K grid
    # ================================================================
    print("\nTEST 3: Full kappa x K grid")
    print("-" * 60)

    K_grid = [5, 10, 20, 50, 100]
    kappa_grid = np.logspace(-1.5, 0.5, 15)
    n_total = 2000

    test3_results = []
    for K_val in K_grid:
        for kappa_true in kappa_grid:
            n_per = n_total // K_val
            if n_per < 5:
                continue
            X, labels = generate_mixture(K_val, n_total, d_fixed, kappa_true, seed=42)
            kappa_meas, knn_acc, n_per_actual = compute_kappa_knn(X, labels)
            q_obs = (knn_acc - 1.0/K_val) / (1.0 - 1.0/K_val)

            q_imp = theory_predict_q_improved(kappa_meas, K_val, n_per_actual, d_fixed)
            q_orig = theory_predict_q_original(kappa_meas, K_val, n_per_actual, d_fixed)

            test3_results.append({
                "K": int(K_val), "kappa": float(kappa_meas),
                "q_obs": float(q_obs),
                "q_improved": float(q_imp), "q_original": float(q_orig),
            })

    q_obs_all = np.array([r["q_obs"] for r in test3_results])
    q_imp_all = np.array([r["q_improved"] for r in test3_results])
    q_orig_all = np.array([r["q_original"] for r in test3_results])

    mask = (q_obs_all > 0.01) & (q_obs_all < 0.99)
    print(f"  {mask.sum()} non-saturated points out of {len(test3_results)}")

    for label, q_pred in [("Improved", q_imp_all[mask]), ("Original", q_orig_all[mask])]:
        q_o = q_obs_all[mask]
        mae = float(np.mean(np.abs(q_o - q_pred)))
        ss_t = np.sum((q_o - q_o.mean())**2)
        r2 = 1 - np.sum((q_o - q_pred)**2) / max(ss_t, 1e-10)
        rho, _ = stats.spearmanr(q_o, q_pred)
        print(f"  {label:10s}: MAE={mae:.4f}, R^2={r2:.4f}, rho={rho:.4f}")

    # ================================================================
    # TEST 4: Real neural network data
    # ================================================================
    print("\nTEST 4: Real neural network representations")
    print("-" * 60)

    real_results = []
    for cache_file in [
        RESULTS_DIR / "cti_multidata_clinc_cache.json",
        RESULTS_DIR / "cti_multidata_agnews_cache.json",
        RESULTS_DIR / "cti_multidata_dbpedia_classes_cache.json",
    ]:
        if not cache_file.exists():
            print(f"  SKIP: {cache_file.name} not found")
            continue

        with open(cache_file) as f:
            cache = json.load(f)

        entries = cache if isinstance(cache, list) else list(cache.values())
        for entry in entries:
            if "knn" not in entry or "kappa" not in entry:
                continue

            kappa_val = entry["kappa"]
            knn_val = entry["knn"]
            K_val = entry.get("K", entry.get("n_classes", 20))
            d_val = entry.get("d", entry.get("hidden_dim", 768))
            n_total_est = entry.get("n_samples", 2000)
            n_per_val = max(n_total_est // K_val, 2)

            q_obs = (knn_val - 1.0/K_val) / (1.0 - 1.0/K_val)
            q_imp = theory_predict_q_improved(kappa_val, K_val, n_per_val, d_val)
            q_orig = theory_predict_q_original(kappa_val, K_val, n_per_val, d_val)

            real_results.append({
                "kappa": float(kappa_val), "K": int(K_val),
                "q_obs": float(q_obs),
                "q_improved": float(q_imp), "q_original": float(q_orig),
            })

    if real_results:
        q_obs_real = np.array([r["q_obs"] for r in real_results])
        q_imp_real = np.array([r["q_improved"] for r in real_results])
        q_orig_real = np.array([r["q_original"] for r in real_results])

        for label, q_pred in [("Improved", q_imp_real), ("Original", q_orig_real)]:
            mae = float(np.mean(np.abs(q_obs_real - q_pred)))
            ss_t = np.sum((q_obs_real - q_obs_real.mean())**2)
            r2 = 1 - np.sum((q_obs_real - q_pred)**2) / max(ss_t, 1e-10)
            rho, _ = stats.spearmanr(q_obs_real, q_pred)
            print(f"  {label:10s}: MAE={mae:.4f}, R^2={r2:.4f}, rho={rho:.4f}")

    # ================================================================
    # SCORECARD
    # ================================================================
    print("\n" + "=" * 70)
    print("SCORECARD")
    print("=" * 70)

    checks = []

    c1 = mae_imp < 0.03
    checks.append(("Improved synthetic MAE < 0.03", c1, f"MAE={mae_imp:.4f}"))

    c2 = mae_imp < mae_orig
    checks.append(("Improved beats original (MAE)", c2, f"{mae_imp:.4f} vs {mae_orig:.4f}"))

    c3 = r2_imp > 0.98
    checks.append(("Improved R^2 > 0.98 on synthetic", c3, f"R^2={r2_imp:.4f}"))

    k200 = [r for r in test2_results if r["K"] == 200]
    if k200:
        k200_err = k200[0]["err_improved"]
        c4 = k200_err < 0.15
        checks.append(("K=200 error < 0.15", c4, f"err={k200_err:.4f}"))

    if real_results:
        rho_r, _ = stats.spearmanr(q_obs_real, q_imp_real)
        c5 = rho_r > 0.5
        checks.append(("Real data rho > 0.5", c5, f"rho={rho_r:.4f}"))

    passes = sum(1 for _, p, _ in checks if p)
    total = len(checks)

    for name, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {val}")

    print(f"\n  SCORECARD: {passes}/{total}")

    all_results["scorecard"] = {"passes": passes, "total": total}

    out_path = RESULTS_DIR / "cti_improved_theory.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
