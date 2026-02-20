#!/usr/bin/env python
"""
ZERO-PARAMETER THEORY: Derive kNN accuracy from first principles.

The theory (probit mechanism):
  q = Phi(mu_M / sigma_M)

where M = min_dist_other - min_dist_same is the kNN margin.

For K-class isotropic Gaussians in d dimensions with n samples per class:
  Same-class distances: D_s ~ chi^2-like with E=2*d, Var=8*d
  Other-class distances: D_o ~ noncentral with E=2*d + 2*kappa*d, Var=8*d*(1+2*kappa)

  Min of n same-class: mu_s_min = 2*d + sqrt(8*d) * z_n
  Min of m=(K-1)*n other-class: mu_o_min = (2*d + 2*kappa*d) + sqrt(8*d*(1+2*kappa)) * z_m

  where z_t = Phi^{-1}(1/(t+1)) is the expected min quantile

  mu_M = mu_o_min - mu_s_min
  sigma_M = sqrt(tau_o^2 + tau_s^2)

  where tau = sigma / (n * phi(z_n)) is the order-statistic std

  q = Phi(mu_M / sigma_M)

NO FREE PARAMETERS. Everything is derived from (kappa, K, n, d, eta).

Tests:
1. Synthetic Gaussians: should be near-exact
2. Real neural network representations: approximate but predictive
"""

import json
import sys
import numpy as np
from scipy import stats
from scipy.special import ndtri  # probit = Phi^{-1}
from scipy.optimize import curve_fit
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def theory_predict_q(kappa, K, n_per_class, d_eff):
    """Zero-parameter theory prediction of kNN quality.

    q = Phi(mu_M / sigma_M) where M = min_dist_other - min_dist_same

    For isotropic Gaussians with class means at distance delta:
    - kappa = tr(S_B)/tr(S_W) ~ delta^2/sigma^2 (population)
    - Same-class: D ~ chi^2(2d) scaled by sigma^2, E[D]=2d, Var=8d
    - Other-class: noncentral, E[D]=2d+2*kappa*d, Var=8d(1+2*kappa)
    """
    if d_eff < 2:
        return 0.5  # degenerate
    if kappa < 1e-10:
        return 0.0  # no separation -> chance

    n = max(n_per_class, 2)
    m = max((K - 1) * n_per_class, 2)

    # Distance distribution parameters
    # Same-class: two points from same N(mu, sigma^2 I), sigma=1
    # D_s = ||x - x'||^2 ~ sum of 2*lambda_i * chi^2_1
    # For isotropic (lambda_i = 1): E[D_s] = 2*d_eff, Var[D_s] = 8*d_eff
    mu_s = 2.0 * d_eff
    sigma_s = np.sqrt(8.0 * d_eff)

    # Other-class: D_o = ||x - z||^2 where x in class k, z in class j != k
    # E[D_o] = 2*d_eff + ||mu_k - mu_j||^2
    # For the mixture: kappa ~ sum ||mu_k - grand||^2 / (d * sigma^2)
    # Average ||mu_k - mu_j||^2 = 2 * kappa * d_eff / 1 (for isotropic, sigma=1)
    # But in the LOO setup, we're comparing to the nearest point from other classes
    # The mean distance to a random other-class point is:
    delta_sq = 2.0 * kappa * d_eff  # average ||mu_k - mu_j||^2
    mu_o = 2.0 * d_eff + delta_sq
    sigma_o = np.sqrt(8.0 * d_eff + 8.0 * delta_sq)
    # Note: Var[D_o] = 8*tr(Sigma_W^2) + 8*delta'*Sigma_W*delta
    # For isotropic: = 8*d + 8*||delta||^2 = 8*d + 8*delta_sq

    # Order statistics: min of n Gaussians
    # E[min] ~ mu + sigma * z_n where z_n = Phi^{-1}(1/(n+1))
    # Var[min] ~ (sigma / (n * phi(z_n)))^2

    # Quantile for same-class min (n samples)
    p_n = 1.0 / (n + 1)
    z_n = ndtri(p_n)  # negative
    phi_z_n = stats.norm.pdf(z_n)

    mu_s_min = mu_s + sigma_s * z_n
    tau_s = sigma_s / (n * phi_z_n) if phi_z_n > 1e-20 else sigma_s

    # Quantile for other-class min (m = (K-1)*n samples)
    p_m = 1.0 / (m + 1)
    z_m = ndtri(p_m)  # more negative (smaller expected min)
    phi_z_m = stats.norm.pdf(z_m)

    mu_o_min = mu_o + sigma_o * z_m
    tau_o = sigma_o / (m * phi_z_m) if phi_z_m > 1e-20 else sigma_o

    # Margin M = min_D_other - min_D_same
    mu_M = mu_o_min - mu_s_min
    sigma_M = np.sqrt(tau_o**2 + tau_s**2)

    if sigma_M < 1e-20:
        return 1.0 if mu_M > 0 else 0.0

    # q = Phi(mu_M / sigma_M)
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


def compute_kappa_knn_eta(X, labels):
    """Compute kappa, kNN accuracy, and eta from data."""
    classes = np.unique(labels)
    K = len(classes)
    d = X.shape[1]
    grand_mean = X.mean(axis=0)

    tr_sb, tr_sw = 0.0, 0.0
    sw_parts = []
    for c in classes:
        Xc = X[labels == c]
        mu_c = Xc.mean(axis=0)
        tr_sb += len(Xc) * np.sum((mu_c - grand_mean)**2)
        diff = Xc - mu_c
        tr_sw += np.sum(diff**2)
        sw_parts.append(diff)

    kappa = tr_sb / max(tr_sw, 1e-10)

    # eta = tr(S_W)^2 / (d * tr(S_W^2))
    Z = np.concatenate(sw_parts, axis=0)
    s = np.linalg.svd(Z, compute_uv=False)
    s2 = s**2
    tr_sw_sq = np.sum(s2**2)
    eta = float(tr_sw**2 / (d * tr_sw_sq)) if tr_sw_sq > 1e-20 else 1.0

    # LOO-kNN
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, labels)
    _, indices = knn.kneighbors(X)
    correct = sum(labels[indices[i, 1]] == labels[i] for i in range(len(X)))
    knn_acc = correct / len(X)

    n_per = min(np.bincount(labels))

    return kappa, knn_acc, eta, n_per


def main():
    print("=" * 70)
    print("ZERO-PARAMETER THEORY PREDICTION")
    print("q = Phi(mu_M / sigma_M) with NO free parameters")
    print("=" * 70)

    all_results = {}

    # ============================================================
    # TEST 1: Synthetic Gaussians — should be near-exact
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: Synthetic Gaussians (K=20, n=2000, d=500)")
    print("=" * 70)

    K, n, d = 20, 2000, 500
    kappa_range = np.logspace(-2, 1, 30)

    test1_results = []
    for kappa_true in kappa_range:
        X, labels = generate_mixture(K, n, d, kappa_true, seed=42)
        kappa_meas, knn_acc, eta, n_per = compute_kappa_knn_eta(X, labels)
        q_obs = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

        # Theory prediction with d_eff = eta * d
        d_eff = eta * d
        q_theory = theory_predict_q(kappa_meas, K, n_per, d_eff)

        # Also try with d_eff = d (isotropic assumption)
        q_theory_iso = theory_predict_q(kappa_meas, K, n_per, d)

        test1_results.append({
            "kappa_true": float(kappa_true),
            "kappa_meas": float(kappa_meas),
            "eta": float(eta),
            "d_eff": float(d_eff),
            "n_per": int(n_per),
            "knn_acc": float(knn_acc),
            "q_obs": float(q_obs),
            "q_theory": float(q_theory),
            "q_theory_iso": float(q_theory_iso),
            "error": float(abs(q_obs - q_theory)),
            "error_iso": float(abs(q_obs - q_theory_iso)),
        })

        print(f"  kappa={kappa_meas:.4f} eta={eta:.3f}: "
              f"q_obs={q_obs:.4f}, q_theory={q_theory:.4f} "
              f"(err={abs(q_obs-q_theory):.4f}), "
              f"q_iso={q_theory_iso:.4f} (err={abs(q_obs-q_theory_iso):.4f})")

    # Summary statistics
    errors = [r["error"] for r in test1_results]
    errors_iso = [r["error_iso"] for r in test1_results]
    mae = float(np.mean(errors))
    mae_iso = float(np.mean(errors_iso))

    q_obs_arr = np.array([r["q_obs"] for r in test1_results])
    q_th_arr = np.array([r["q_theory"] for r in test1_results])
    q_iso_arr = np.array([r["q_theory_iso"] for r in test1_results])

    ss_tot = np.sum((q_obs_arr - q_obs_arr.mean())**2)
    r2 = 1 - np.sum((q_obs_arr - q_th_arr)**2) / max(ss_tot, 1e-10)
    r2_iso = 1 - np.sum((q_obs_arr - q_iso_arr)**2) / max(ss_tot, 1e-10)

    rho, _ = stats.spearmanr(q_obs_arr, q_th_arr)

    print(f"\n  ZERO-PARAM THEORY: MAE={mae:.4f}, R^2={r2:.4f}, rho={rho:.4f}")
    print(f"  Isotropic (d_eff=d): MAE={mae_iso:.4f}, R^2={r2_iso:.4f}")

    all_results["test1_synthetic"] = {
        "K": K, "n": n, "d": d,
        "results": test1_results,
        "summary": {
            "mae": mae, "mae_iso": mae_iso,
            "r2": float(r2), "r2_iso": float(r2_iso),
            "rho": float(rho),
        }
    }

    # ============================================================
    # TEST 2: Vary K (fixed kappa)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: Vary K (n=2000, d=500, kappa=0.1)")
    print("=" * 70)

    K_values = [5, 10, 20, 50, 100, 200]
    kappa_fixed = 0.1
    n_fixed, d_fixed = 2000, 500

    test2_results = []
    for K in K_values:
        X, labels = generate_mixture(K, n_fixed, d_fixed, kappa_fixed, seed=42)
        kappa_meas, knn_acc, eta, n_per = compute_kappa_knn_eta(X, labels)
        q_obs = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

        d_eff = eta * d_fixed
        q_theory = theory_predict_q(kappa_meas, K, n_per, d_eff)
        q_theory_iso = theory_predict_q(kappa_meas, K, n_per, d_fixed)

        test2_results.append({
            "K": K, "kappa_meas": float(kappa_meas), "eta": float(eta),
            "n_per": int(n_per), "d_eff": float(d_eff),
            "q_obs": float(q_obs), "q_theory": float(q_theory),
            "q_theory_iso": float(q_theory_iso),
            "error": float(abs(q_obs - q_theory)),
        })
        print(f"  K={K:4d}: q_obs={q_obs:.4f}, q_theory={q_theory:.4f} "
              f"(err={abs(q_obs-q_theory):.4f}), n_per={n_per}")

    all_results["test2_vary_K"] = test2_results

    # ============================================================
    # TEST 3: Test on REAL neural network data
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: Real neural network representations")
    print("=" * 70)

    # Load cached results from previous experiments
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

        # Cache is a list of dicts
        entries = cache if isinstance(cache, list) else list(cache.values())
        for i, entry in enumerate(entries):
            if "knn" not in entry or "kappa" not in entry:
                continue

            kappa_val = entry["kappa"]
            knn_val = entry["knn"]
            eta_val = entry.get("eta", 1.0)
            K_val = entry.get("K", entry.get("n_classes", 20))
            # Estimate hidden dim from model name
            d_val = entry.get("d", entry.get("hidden_dim", 768))
            # Estimate n_per from total / K
            n_total = entry.get("n_samples", 2000)
            n_per_val = max(n_total // K_val, 2)

            q_obs = (knn_val - 1.0/K_val) / (1.0 - 1.0/K_val)
            d_eff = max(eta_val * d_val, 2)
            q_theory = theory_predict_q(kappa_val, K_val, n_per_val, d_eff)
            q_theory_iso = theory_predict_q(kappa_val, K_val, n_per_val, d_val)

            real_results.append({
                "key": f"{entry.get('model','?')}_{entry.get('dataset','?')}_a{entry.get('alpha',0)}",
                "kappa": float(kappa_val),
                "eta": float(eta_val),
                "K": int(K_val),
                "d": int(d_val),
                "n_per": int(n_per_val),
                "d_eff": float(d_eff),
                "q_obs": float(q_obs),
                "q_theory": float(q_theory),
                "q_theory_iso": float(q_theory_iso),
                "error": float(abs(q_obs - q_theory)),
            })

    if real_results:
        errors_real = [r["error"] for r in real_results]
        mae_real = float(np.mean(errors_real))
        q_obs_real = np.array([r["q_obs"] for r in real_results])
        q_th_real = np.array([r["q_theory"] for r in real_results])

        ss_tot_real = np.sum((q_obs_real - q_obs_real.mean())**2)
        r2_real = 1 - np.sum((q_obs_real - q_th_real)**2) / max(ss_tot_real, 1e-10)
        rho_real, _ = stats.spearmanr(q_obs_real, q_th_real)

        print(f"\n  REAL DATA: {len(real_results)} points")
        print(f"  ZERO-PARAM THEORY: MAE={mae_real:.4f}, R^2={r2_real:.4f}, rho={rho_real:.4f}")

        all_results["test3_real_data"] = {
            "n_points": len(real_results),
            "mae": mae_real,
            "r2": float(r2_real),
            "rho": float(rho_real),
            "results": real_results[:20],  # save first 20 for brevity
        }
    else:
        print("  No cached real data found")
        all_results["test3_real_data"] = {"n_points": 0}

    # ============================================================
    # SCORECARD
    # ============================================================
    print("\n" + "=" * 70)
    print("SCORECARD")
    print("=" * 70)

    checks = []

    # Check 1: Synthetic MAE < 0.05
    c1 = all_results["test1_synthetic"]["summary"]["mae"] < 0.05
    checks.append({
        "criterion": "Synthetic Gaussian: zero-param MAE < 0.05",
        "passed": bool(c1),
        "value": f"MAE={all_results['test1_synthetic']['summary']['mae']:.4f}"
    })

    # Check 2: Synthetic R^2 > 0.95
    c2 = all_results["test1_synthetic"]["summary"]["r2"] > 0.95
    checks.append({
        "criterion": "Synthetic Gaussian: zero-param R^2 > 0.95",
        "passed": bool(c2),
        "value": f"R^2={all_results['test1_synthetic']['summary']['r2']:.4f}"
    })

    # Check 3: Monotonicity (rho > 0.95)
    c3 = all_results["test1_synthetic"]["summary"]["rho"] > 0.95
    checks.append({
        "criterion": "Synthetic: monotonic (rho > 0.95)",
        "passed": bool(c3),
        "value": f"rho={all_results['test1_synthetic']['summary']['rho']:.4f}"
    })

    # Check 4: K-scaling works (all predictions within 0.15)
    k_errors = [r["error"] for r in all_results.get("test2_vary_K", [])]
    if k_errors:
        max_k_err = max(k_errors)
        c4 = max_k_err < 0.15
    else:
        max_k_err = float('nan')
        c4 = False
    checks.append({
        "criterion": "K-scaling: max prediction error < 0.15",
        "passed": bool(c4),
        "value": f"max_err={max_k_err:.4f}"
    })

    # Check 5: Real data prediction (if available)
    if real_results:
        c5 = all_results["test3_real_data"]["rho"] > 0.7
        checks.append({
            "criterion": "Real NN data: rho > 0.7",
            "passed": bool(c5),
            "value": f"rho={all_results['test3_real_data']['rho']:.4f}"
        })
    else:
        checks.append({
            "criterion": "Real NN data: rho > 0.7",
            "passed": False,
            "value": "no data"
        })

    passes = sum(1 for c in checks if c["passed"])
    total = len(checks)

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['criterion']}: {c['value']}")

    print(f"\n  SCORECARD: {passes}/{total}")

    all_results["scorecard"] = {
        "passes": passes, "total": total, "checks": checks,
    }

    # Save
    out_path = RESULTS_DIR / "cti_zero_param_theory.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
