#!/usr/bin/env python -u
"""
K-CANCELLATION VERIFICATION: Test Theorem 7.5 predictions.

Theorem 7.5 predicts:
1. dist_ratio = 1 + C_1*kappa_nearest + C_2*log(K-1)
   where C_1 ~ 0.760 (from A*C_2 = -C_1 cancellation condition)
   and C_2 < 0 (pool-size effect)

2. The K-cancellation is exact: B ~ 0 in logit(q) = A*(dist_ratio-1) + B*log(K-1) + C

3. When dist_ratio is used, B should be ZERO (K-dependence absorbed)
   When kappa is used, B should be -1 (from Gumbel Race)

This experiment verifies these predictions on synthetic isotropic Gaussians.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit, logit
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

K_VALS = [5, 10, 20, 50, 100, 200]
N_PER = 100       # samples per class (moderate, not too slow)
D = 200           # embedding dimension
N_KAPPA = 8       # sigma_B sweep
N_MC = 20         # Monte Carlo repeats (larger for stability)
SIGMA_W = 1.0     # within-class std


def simulate_one(K, n_per, d, sigma_B, sigma_W=1.0, n_mc=20, seed=None):
    """
    Simulate 1-NN classification on K isotropic Gaussian clusters.
    Returns (kappa_nearest, kappa_spec, dist_ratio, q).
    """
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    kappa_near_list = []
    kappa_spec_list = []
    dr_list = []
    knn_list = []

    for _ in range(n_mc):
        # Generate class means
        means = rng.randn(K, d) * sigma_B   # [K, d]

        # Generate data
        X = np.zeros((K * n_per, d))
        y = np.zeros(K * n_per, dtype=int)
        for k in range(K):
            X[k*n_per:(k+1)*n_per] = means[k] + rng.randn(n_per, d) * sigma_W
            y[k*n_per:(k+1)*n_per] = k

        # kappa_spec = tr(S_B) / tr(S_W)
        N = len(X)
        grand_mean = X.mean(0)
        tr_SB = 0.0
        tr_SW = 0.0
        for k in range(K):
            Xk = X[k*n_per:(k+1)*n_per]
            mu_k = Xk.mean(0)
            tr_SB += n_per * np.dot(mu_k - grand_mean, mu_k - grand_mean)
            tr_SW += np.sum((Xk - mu_k)**2)
        tr_SB /= N
        tr_SW /= N
        kappa_s = tr_SB / (tr_SW + 1e-10)
        kappa_spec_list.append(kappa_s)

        # kappa_nearest = mean pairwise distance / d
        # Use actual class means
        dists_sq = np.zeros(K)
        for k in range(K):
            other_diffs = means - means[k]  # [K, d]
            other_dists = np.sum(other_diffs**2, axis=1)  # [K]
            other_dists[k] = np.inf  # exclude self
            dists_sq[k] = other_dists.min()
        kappa_near = np.mean(dists_sq) / (d * sigma_W**2 * 2)
        kappa_near_list.append(float(kappa_near))

        # 1-NN accuracy (subsample for speed)
        n_sub = min(N, 300)
        idx_sub = rng.choice(N, n_sub, replace=False)
        X_sub = X[idx_sub]
        y_sub = y[idx_sub]

        # Fast pairwise distances
        norms = np.sum(X_sub**2, axis=1)
        D_mat = norms[:, None] + norms[None, :] - 2 * (X_sub @ X_sub.T)
        np.fill_diagonal(D_mat, np.inf)

        nn_idx = np.argmin(D_mat, axis=1)
        knn_acc = float(np.mean(y_sub[nn_idx] == y_sub))
        knn_list.append(knn_acc)

        # dist_ratio = E[D_inter_nearest] / E[D_intra_nearest]
        intra_dists = []
        inter_dists = []
        for i in range(n_sub):
            same_mask = (y_sub == y_sub[i])
            same_mask[i] = False
            diff_mask = ~same_mask
            diff_mask[i] = False
            if same_mask.sum() > 0:
                intra_dists.append(np.sqrt(max(D_mat[i][same_mask].min(), 0)))
            if diff_mask.sum() > 0:
                inter_dists.append(np.sqrt(max(D_mat[i][diff_mask].min(), 0)))

        if len(intra_dists) > 0 and len(inter_dists) > 0:
            dr = np.mean(inter_dists) / (np.mean(intra_dists) + 1e-10)
            dr_list.append(float(dr))

    q_mean = np.mean(knn_list)
    q = max(min((q_mean - 1.0/K) / (1.0 - 1.0/K), 0.999), 0.001)

    return {
        "kappa_nearest": float(np.mean(kappa_near_list)),
        "kappa_spec": float(np.mean(kappa_spec_list)),
        "dist_ratio": float(np.mean(dr_list)) if dr_list else float("nan"),
        "knn_acc": float(q_mean),
        "q": float(q),
    }


def main():
    print("=" * 70)
    print("K-CANCELLATION VERIFICATION (Theorem 7.5)")
    print("=" * 70)
    print(f"  K={K_VALS}, n_per={N_PER}, d={D}, n_mc={N_MC}")
    print()

    # Collect data
    all_data = []
    np.random.seed(42)

    for K in K_VALS:
        print(f"  K={K}:", end="", flush=True)
        sigma_B_vals = np.logspace(-2, 1, N_KAPPA)

        for sigma_B in sigma_B_vals:
            result = simulate_one(K, N_PER, D, sigma_B, SIGMA_W, N_MC)
            if (0.01 < result["q"] < 0.99 and
                not np.isnan(result["dist_ratio"]) and
                result["dist_ratio"] > 0):
                result["K"] = K
                result["sigma_B"] = float(sigma_B)
                all_data.append(result)
                print(".", end="", flush=True)
        print(f" ({len([r for r in all_data if r['K']==K])} pts)")

    print(f"\nTotal valid points: {len(all_data)}")

    if len(all_data) < 20:
        print("[ERROR] Too few valid points!")
        return

    kappa_near = np.array([r["kappa_nearest"] for r in all_data])
    kappa_spec = np.array([r["kappa_spec"] for r in all_data])
    dist_ratios = np.array([r["dist_ratio"] for r in all_data])
    qs = np.array([r["q"] for r in all_data])
    Ks = np.array([float(r["K"]) for r in all_data])
    logit_qs = logit(qs)

    # ================================================================
    # TEST 1: dist_ratio = 1 + C_1*kappa_nearest + C_2*log(K-1)
    # Prediction: C_1 ~ 0.760, C_2 < 0
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: dist_ratio linear decomposition")
    print("  Prediction: C_1 ~ 0.760, C_2 < 0")
    print("=" * 70)

    X_des = np.column_stack([kappa_near, np.log(Ks - 1 + 1e-6), np.ones(len(kappa_near))])
    theta, _, _, _ = np.linalg.lstsq(X_des, dist_ratios, rcond=None)
    C1, C2, C0 = theta
    dr_pred = X_des @ theta
    r2_dr = 1 - np.sum((dist_ratios - dr_pred)**2) / max(np.sum((dist_ratios - dist_ratios.mean())**2), 1e-10)

    print(f"\n  dist_ratio = {C0:.4f} + {C1:.4f}*kappa_nearest + {C2:.4f}*log(K-1)")
    print(f"  R2 = {r2_dr:.4f}")
    print(f"  C_1 = {C1:.4f} [theory: ~0.760]")
    print(f"  C_2 = {C2:.4f} [theory: < 0]")
    pred_C1 = 0.760
    print(f"  |C_1 - prediction| = {abs(C1 - pred_C1):.4f}")

    # ================================================================
    # TEST 2: Cancellation condition A*C_2 = -C_1
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Cancellation condition A*C_2 = -C_1")
    print("  Prediction: A*C_2 ~ -C_1 ~ -0.760")
    print("=" * 70)

    # Fit A from the Gumbel Race: logit(q) = A*kappa_nearest + C
    # (fixing only slope, since B*log(K-1) should cancel)
    X_kappa = np.column_stack([kappa_near, np.ones(len(kappa_near))])
    theta_kappa, _, _, _ = np.linalg.lstsq(X_kappa, logit_qs, rcond=None)
    A_fitted, _ = theta_kappa
    print(f"\n  A (from logit(q) = A*kappa_nearest + C): {A_fitted:.4f}")
    print(f"  A*C_2 = {A_fitted * C2:.4f} [prediction: ~-{C1:.3f}]")
    print(f"  Cancellation error: |A*C_2 + C_1| = {abs(A_fitted * C2 + C1):.4f}")

    # ================================================================
    # TEST 3: B coefficient in logit(q) vs dist_ratio should be ~ 0
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: B coefficient when using dist_ratio")
    print("  Prediction: B ~ 0 (dist_ratio absorbs K-dependence)")
    print("=" * 70)

    # Fit logit(q) = A_dr*(dist_ratio-1) + B_dr*log(K-1) + C_dr
    X_dr = np.column_stack([dist_ratios - 1, np.log(Ks - 1 + 1e-6), np.ones(len(dist_ratios))])
    theta_dr, _, _, _ = np.linalg.lstsq(X_dr, logit_qs, rcond=None)
    A_dr, B_dr, C_dr = theta_dr

    dr_logit_pred = X_dr @ theta_dr
    r2_dr_law = 1 - np.sum((logit_qs - dr_logit_pred)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)

    print(f"\n  logit(q) = {A_dr:.4f}*(dist_ratio-1) + {B_dr:.4f}*log(K-1) + {C_dr:.4f}")
    print(f"  R2 = {r2_dr_law:.4f}")
    print(f"  B_dr = {B_dr:.4f} [prediction: ~ 0.0]")

    # Also fit WITHOUT the K term (to see how much B adds)
    X_dr_simple = np.column_stack([dist_ratios - 1, np.ones(len(dist_ratios))])
    theta_simple, _, _, _ = np.linalg.lstsq(X_dr_simple, logit_qs, rcond=None)
    A_simple, C_simple = theta_simple
    pred_simple = X_dr_simple @ theta_simple
    r2_simple = 1 - np.sum((logit_qs - pred_simple)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)

    print(f"\n  logit(q) = {A_simple:.4f}*(dist_ratio-1) + {C_simple:.4f}  [B=0 forced]")
    print(f"  R2 = {r2_simple:.4f}")
    print(f"  R2 improvement from adding K: {r2_dr_law - r2_simple:.4f}")

    # ================================================================
    # TEST 4: B coefficient when using kappa_nearest (should be ~ -1)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: B coefficient when using kappa_nearest")
    print("  Prediction: B ~ -1.0 (Gumbel Race prediction)")
    print("=" * 70)

    X_kappa_K = np.column_stack([kappa_near, np.log(Ks - 1 + 1e-6), np.ones(len(kappa_near))])
    theta_kappa_K, _, _, _ = np.linalg.lstsq(X_kappa_K, logit_qs, rcond=None)
    A_kK, B_kK, C_kK = theta_kappa_K

    kappa_logit_pred = X_kappa_K @ theta_kappa_K
    r2_kappa_K = 1 - np.sum((logit_qs - kappa_logit_pred)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)

    print(f"\n  logit(q) = {A_kK:.4f}*kappa_nearest + {B_kK:.4f}*log(K-1) + {C_kK:.4f}")
    print(f"  R2 = {r2_kappa_K:.4f}")
    print(f"  B_kappa = {B_kK:.4f} [prediction: ~ -1.0]")

    # ================================================================
    # SCORECARD
    # ================================================================
    print("\n" + "=" * 70)
    print("SCORECARD")
    print("=" * 70)
    checks = [
        ("C_1 ~ 0.760 (within 30%)", abs(C1 - 0.760) / 0.760 < 0.30, f"C_1={C1:.3f}"),
        ("C_2 < 0 (pool-size correction is negative)", C2 < 0, f"C_2={C2:.3f}"),
        ("A*C_2 ~ -C_1 (cancellation condition, within 50%)", abs(A_fitted*C2 + C1) / C1 < 0.50, f"|err|/C_1={abs(A_fitted*C2+C1)/C1:.3f}"),
        ("B_dr ~ 0 when using dist_ratio (|B| < 0.3)", abs(B_dr) < 0.3, f"B_dr={B_dr:.3f}"),
        ("B_kappa ~ -1 when using kappa_nearest (within 50%)", abs(B_kK + 1.0) < 0.50, f"B_kappa={B_kK:.3f}"),
        ("R2 of dist_ratio simple model > 0.90", r2_simple > 0.90, f"R2={r2_simple:.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    output = {
        "theorem": "7.5 K-Cancellation Mechanism",
        "test1_linear_decomposition": {
            "C0": float(C0), "C1": float(C1), "C2": float(C2),
            "r2": float(r2_dr),
            "predicted_C1": 0.760,
        },
        "test2_cancellation_condition": {
            "A_fitted": float(A_fitted),
            "A_times_C2": float(A_fitted * C2),
            "predicted": float(-C1),
            "cancellation_error": float(abs(A_fitted * C2 + C1)),
        },
        "test3_B_when_dist_ratio": {
            "A": float(A_dr), "B": float(B_dr), "C": float(C_dr),
            "r2_with_K": float(r2_dr_law),
            "r2_without_K": float(r2_simple),
            "r2_improvement": float(r2_dr_law - r2_simple),
        },
        "test4_B_when_kappa": {
            "A": float(A_kK), "B": float(B_kK), "C": float(C_kK),
            "r2": float(r2_kappa_K),
            "predicted_B": -1.0,
        },
        "scorecard": {"passes": passes, "total": len(checks)},
        "n_points": len(all_data),
        "config": {"K_vals": K_VALS, "n_per": N_PER, "d": D, "n_mc": N_MC},
    }

    out_path = RESULTS_DIR / "cti_k_cancellation_verify.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
