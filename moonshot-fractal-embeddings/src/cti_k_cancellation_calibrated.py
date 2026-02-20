#!/usr/bin/env python -u
"""
K-CANCELLATION VERIFICATION v2: Calibrated kappa per K (fixes selection bias).

Problem with v1: sweeping fixed kappa values across K creates selection bias.
- High K: only high kappa survives (otherwise q->0, filtered)
- Low K: only low kappa survives (otherwise q->1, filtered)
- Result: K and kappa are CORRELATED in surviving data -> wrong coefficients

Fix: For each K, binary-search kappa_target(K) giving q ~ 0.5.
Then create a calibrated grid: kappa_target * {0.5, 0.7, 1.0, 1.5, 2.0}.
Each MC repeat stored as independent data point.

This ensures K variation is orthogonal to kappa variation in the dataset.

Theorem 7.5 predictions (what we test):
1. dist_ratio = 1 + C_1*kappa_nearest + C_2*log(K-1), C_1 ~ 0.760, C_2 < 0
2. Cancellation: A*C_2 = -C_1 (so K-term vanishes in dist_ratio law)
3. B_dr ~ 0 in logit(q) = A*(dist_ratio-1) + B*log(K-1) + C
4. B_kappa ~ -1 in logit(q) = A*kappa_nearest + B*log(K-1) + C
"""

import json
import numpy as np
from pathlib import Path
from scipy.special import logit
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def generate_gaussian_clusters(K, n_per, d, kappa, rng):
    """Generate K isotropic Gaussian clusters with separation kappa."""
    # Each cluster mean is drawn from N(0, kappa^2/d * I) -> kappa = ||mu||/sigma
    # Use scaled random means so ||mu_i - mu_j||/sigma ~ kappa for nearest pair
    sigma = 1.0
    # Place cluster means so nearest-pair distance ~ kappa
    # Simple: random means scaled so RMS inter-cluster distance ~ kappa
    # For d large, ||mu_i - mu_j||^2 ~ 2*||mu||^2 for random mu_i, mu_j
    # Target ||mu||^2 = kappa^2/2 * d (so ||mu_i||=kappa/sqrt(2) in L2)
    means = rng.standard_normal((K, d)) * (kappa / np.sqrt(2.0 * d)) * np.sqrt(d)
    # Scale: ||mu|| ~ kappa/sqrt(2), ||mu_i - mu_j|| ~ kappa (on average)

    X_list, y_list = [], []
    for k in range(K):
        samples = means[k] + sigma * rng.standard_normal((n_per, d))
        X_list.append(samples)
        y_list.extend([k] * n_per)
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    return X, y, means, sigma


def compute_stats(X, y, means, sigma):
    """Compute dist_ratio, kappa_nearest, kappa_spec, and kNN q."""
    K = len(np.unique(y))
    N = len(X)

    # kNN accuracy (LOO or train/test split)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    q = max(min(q, 0.999), 0.001)

    if q < 0.01 or q > 0.99:
        return None  # too extreme for logit

    # dist_ratio: E[D_intra_min] vs E[D_inter_min]
    n_sub = min(N, 400)
    idx_sub = np.random.choice(N, n_sub, replace=False)
    X_sub = X[idx_sub]
    y_sub = y[idx_sub]

    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X_sub, metric="euclidean")
    np.fill_diagonal(D, np.inf)

    intra_dists, inter_dists = [], []
    for i in range(n_sub):
        same = (y_sub == y_sub[i])
        same[i] = False
        diff = ~same
        diff[i] = False
        if same.sum() > 0:
            intra_dists.append(D[i][same].min())
        if diff.sum() > 0:
            inter_dists.append(D[i][diff].min())

    if not intra_dists or not inter_dists:
        return None

    d_intra = float(np.mean(intra_dists))
    d_inter = float(np.mean(inter_dists))
    dist_ratio = d_inter / (d_intra + 1e-10)

    # kappa_nearest: from true cluster means
    # For each cluster, find nearest other cluster (center-to-center / sigma)
    from sklearn.metrics import pairwise_distances as pd2
    mean_dists = pd2(means, metric="euclidean")
    np.fill_diagonal(mean_dists, np.inf)
    nearest_center_dist = float(mean_dists.min(axis=1).mean())
    kappa_nearest = nearest_center_dist / sigma

    # kappa_spec: trace(S_B)/trace(S_W) spectral
    grand_mean = X.mean(0)
    S_B = sum(
        (n_per * np.outer(means[k] - grand_mean, means[k] - grand_mean))
        for k in range(K)
    ) / N
    S_W = np.cov(X.T) - S_B
    kappa_spec = float(np.trace(S_B) / (np.trace(np.abs(S_W)) + 1e-10))

    return {
        "K": int(K),
        "q": float(q),
        "logit_q": float(logit(q)),
        "dist_ratio": float(dist_ratio),
        "kappa_nearest": float(kappa_nearest),
        "kappa_spec": float(kappa_spec),
        "d_intra": d_intra,
        "d_inter": d_inter,
    }


def compute_q_raw(K, n_per, d, kappa, rng_seed):
    """Compute raw q (no validity filter) for calibration use only."""
    X, y, means, sigma = generate_gaussian_clusters(K, n_per, d, kappa,
                                                     np.random.default_rng(rng_seed))
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return 0.0
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    return (acc - 1.0/K) / (1.0 - 1.0/K)


def find_target_kappa(K, n_per, d, q_target=0.5, n_mc=3, rng_seed=0):
    """Binary search for kappa giving q ~ q_target using raw q (no filter)."""
    lo, hi = 0.05, 5.0

    for _ in range(20):  # binary search iterations
        mid = (lo + hi) / 2.0
        qs = [compute_q_raw(K, n_per, d, mid, rng_seed + trial) for trial in range(n_mc)]
        q_mean = np.mean(qs)
        if q_mean > q_target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.02:
            break

    return (lo + hi) / 2.0


def main():
    print("=" * 70)
    print("K-CANCELLATION VERIFICATION v2 (Calibrated kappa per K)")
    print("=" * 70)

    K_vals = [5, 10, 20, 50, 100, 200]
    n_per = 100
    d = 200
    n_mc = 30  # MC repeats per (K, kappa)
    kappa_multipliers = [0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.5]  # around kappa_target

    print(f"\n  K={K_vals}, n_per={n_per}, d={d}, n_mc={n_mc}")
    print(f"  kappa_multipliers={kappa_multipliers}")
    print()

    # Step 1: Find calibrated kappa_target for each K
    print("STEP 1: Calibrating kappa_target per K (q ~ 0.5)")
    kappa_targets = {}
    for K in K_vals:
        kt = find_target_kappa(K, n_per, d, q_target=0.5, n_mc=5)
        kappa_targets[K] = kt
        print(f"  K={K:3d}: kappa_target = {kt:.3f}")

    # Step 2: Collect data with calibrated kappa grid
    print("\nSTEP 2: Collecting data with calibrated kappa grid")
    all_data = []
    for K in K_vals:
        kappa_target = kappa_targets[K]
        kappas = [kappa_target * m for m in kappa_multipliers]
        n_valid = 0
        for kappa in kappas:
            for mc_i in range(n_mc):
                rng = np.random.default_rng(mc_i * 1000 + K)
                X, y, means, sigma = generate_gaussian_clusters(K, n_per, d, kappa, rng)
                res = compute_stats(X, y, means, sigma)
                if res is not None:
                    res["kappa_generated"] = float(kappa)
                    all_data.append(res)
                    n_valid += 1
        print(f"  K={K:3d}: {n_valid} valid / {len(kappas)*n_mc} total")

    print(f"\nTotal valid data points: {len(all_data)}")

    if len(all_data) < 30:
        print("[ERROR] Too few data points for reliable regression!")
        return

    # Step 3: Regressions
    kappa_near = np.array([r["kappa_nearest"] for r in all_data])
    kappa_spec = np.array([r["kappa_spec"] for r in all_data])
    dist_ratios = np.array([r["dist_ratio"] for r in all_data])
    qs = np.array([r["q"] for r in all_data])
    Ks = np.array([float(r["K"]) for r in all_data])
    logit_qs = np.array([r["logit_q"] for r in all_data])

    # TEST 1: dist_ratio = 1 + C_1*kappa_nearest + C_2*log(K-1)
    print("\n" + "=" * 70)
    print("TEST 1: dist_ratio linear decomposition")
    print("  Prediction: C_1 ~ 0.760, C_2 < 0")
    print("=" * 70)

    X_des = np.column_stack([kappa_near, np.log(Ks + 1e-6), np.ones(len(kappa_near))])
    theta, _, _, _ = np.linalg.lstsq(X_des, dist_ratios, rcond=None)
    C1, C2, C0 = theta
    dr_pred = X_des @ theta
    r2_dr = 1 - np.sum((dist_ratios - dr_pred)**2) / max(np.sum((dist_ratios - dist_ratios.mean())**2), 1e-10)

    print(f"\n  dist_ratio = {C0:.4f} + {C1:.4f}*kappa_nearest + {C2:.4f}*log(K)")
    print(f"  R2 = {r2_dr:.4f}")
    print(f"  C_1 = {C1:.4f} [theory: ~0.760]")
    print(f"  C_2 = {C2:.4f} [theory: < 0]")

    # TEST 2: Cancellation condition
    print("\n" + "=" * 70)
    print("TEST 2: Cancellation condition A*C_2 = -C_1")
    print("  Prediction: A*C_2 ~ -C_1 ~ -0.760")
    print("=" * 70)

    X_kappa = np.column_stack([kappa_near, np.ones(len(kappa_near))])
    theta_kappa, _, _, _ = np.linalg.lstsq(X_kappa, logit_qs, rcond=None)
    A_fitted = float(theta_kappa[0])
    print(f"\n  A (from logit(q) = A*kappa_nearest + C): {A_fitted:.4f}")
    print(f"  A*C_2 = {A_fitted * C2:.4f} [prediction: ~-{C1:.3f}]")
    print(f"  Cancellation error: |A*C_2 + C_1| = {abs(A_fitted * C2 + C1):.4f}")

    # TEST 3: B when dist_ratio
    print("\n" + "=" * 70)
    print("TEST 3: B coefficient when using dist_ratio")
    print("  Prediction: B ~ 0 (dist_ratio absorbs K-dependence)")
    print("=" * 70)

    X_dr = np.column_stack([dist_ratios - 1, np.log(Ks + 1e-6), np.ones(len(dist_ratios))])
    theta_dr, _, _, _ = np.linalg.lstsq(X_dr, logit_qs, rcond=None)
    A_dr, B_dr, C_dr = theta_dr
    pred_dr = X_dr @ theta_dr
    r2_dr_law = 1 - np.sum((logit_qs - pred_dr)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)

    X_dr_simple = np.column_stack([dist_ratios - 1, np.ones(len(dist_ratios))])
    theta_simple, _, _, _ = np.linalg.lstsq(X_dr_simple, logit_qs, rcond=None)
    pred_simple = X_dr_simple @ theta_simple
    r2_simple = 1 - np.sum((logit_qs - pred_simple)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)

    print(f"\n  logit(q) = {A_dr:.4f}*(dist_ratio-1) + {B_dr:.4f}*log(K) + {C_dr:.4f}")
    print(f"  R2 = {r2_dr_law:.4f}")
    print(f"  B_dr = {B_dr:.4f} [prediction: ~ 0.0]")
    print(f"  Simple (B=0): R2 = {r2_simple:.4f}")
    print(f"  R2 improvement from K: {r2_dr_law - r2_simple:.4f}")

    # TEST 4: B when kappa_nearest
    print("\n" + "=" * 70)
    print("TEST 4: B coefficient when using kappa_nearest")
    print("  Prediction: B ~ -1.0 (Gumbel Race)")
    print("=" * 70)

    X_kK = np.column_stack([kappa_near, np.log(Ks + 1e-6), np.ones(len(kappa_near))])
    theta_kK, _, _, _ = np.linalg.lstsq(X_kK, logit_qs, rcond=None)
    A_kK, B_kK, C_kK = theta_kK
    pred_kK = X_kK @ theta_kK
    r2_kK = 1 - np.sum((logit_qs - pred_kK)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)

    print(f"\n  logit(q) = {A_kK:.4f}*kappa_nearest + {B_kK:.4f}*log(K) + {C_kK:.4f}")
    print(f"  R2 = {r2_kK:.4f}")
    print(f"  B_kappa = {B_kK:.4f} [prediction: ~ -1.0]")

    # TEST 5: Direct K effect at fixed kappa_target (cleanest test)
    print("\n" + "=" * 70)
    print("TEST 5: Direct K effect at kappa_target (cleanest)")
    print("  At kappa = kappa_target(K), q should ~ 0.5 for all K")
    print("  Shows the PURE K effect holding kappa constant per-K")
    print("=" * 70)
    print()
    for K in K_vals:
        kt_pts = [r for r in all_data if r["K"] == K and
                  abs(r["kappa_generated"] - kappa_targets[K]) < 1e-6]
        if kt_pts:
            q_vals = [r["q"] for r in kt_pts]
            dr_vals = [r["dist_ratio"] for r in kt_pts]
            print(f"  K={K:3d}: kappa_target={kappa_targets[K]:.2f}, "
                  f"q={np.mean(q_vals):.3f}+/-{np.std(q_vals):.3f}, "
                  f"dist_ratio={np.mean(dr_vals):.3f}+/-{np.std(dr_vals):.3f}")

    # SCORECARD
    print("\n" + "=" * 70)
    print("SCORECARD")
    print("=" * 70)

    checks = [
        ("C_1 > 0 (kappa_nearest has positive effect on dist_ratio)", C1 > 0,
         f"C_1={C1:.3f}"),
        ("C_2 < 0 (pool-size effect: more classes -> smaller inter-class dist)", C2 < 0,
         f"C_2={C2:.3f}"),
        ("R2(dist_ratio model) > 0.70", r2_dr > 0.70, f"R2={r2_dr:.4f}"),
        ("B_dr < B_kappa (dist_ratio absorbs more K than kappa)", abs(B_dr) < abs(B_kK),
         f"B_dr={B_dr:.3f}, B_kK={B_kK:.3f}"),
        ("B_kappa < 0 (Gumbel Race: more classes -> lower q)", B_kK < 0,
         f"B_kappa={B_kK:.3f}"),
        ("R2(simple dist_ratio, no K) > 0.80", r2_simple > 0.80, f"R2_simple={r2_simple:.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    out = {
        "version": "v2_calibrated",
        "n_points": len(all_data),
        "kappa_targets": {str(K): v for K, v in kappa_targets.items()},
        "test1": {"C0": float(C0), "C1": float(C1), "C2": float(C2), "R2": float(r2_dr)},
        "test2": {"A": float(A_fitted), "AC2": float(A_fitted*C2), "cancel_err": float(abs(A_fitted*C2+C1))},
        "test3": {"A": float(A_dr), "B": float(B_dr), "C": float(C_dr),
                  "r2_with_K": float(r2_dr_law), "r2_simple": float(r2_simple)},
        "test4": {"A": float(A_kK), "B": float(B_kK), "C": float(C_kK), "r2": float(r2_kK)},
        "scorecard": {"passes": passes, "total": len(checks)},
    }
    out_path = RESULTS_DIR / "cti_k_cancellation_calibrated.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
