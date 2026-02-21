#!/usr/bin/env python -u
"""
OVERLAP SWEEP: Domain of Validity for kappa_nearest Law (Feb 21 2026)
======================================================================
Hypothesis: logit(q) = alpha * kappa_nearest + C holds ONLY when kappa_nearest > kappa_c.
Below kappa_c (overlapping classes), the theory breaks down.

Design:
- Synthetic K-class Gaussian data with varying sigma (controls kappa)
- Sweep kappa_nearest from 0.0 to 3.0 (very overlapping to very separated)
- For each kappa: compute q_1NN, logit(q), fit the law
- Find kappa_c where R2 drops below 0.5

This directly tests the "domain of validity" for the Gumbel Race theory.
Addresses go_emotions breakdown (kappa_nearest < 0.2 for emotion classification).

Pre-registered:
  - Theory predicts: law holds (R2 > 0.7) for kappa > 1.0
  - Critical kappa kappa_c: R2 drops to 0.5 somewhere in [0.1, 0.5]
  - go_emotions regime: kappa ~ 0.1-0.3 (low overlap), theory should fail there
"""

import json
import numpy as np
from scipy.special import expit  # sigmoid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import time

# Pre-registered thresholds
PRE_REG_KAPPA_C_LOW = 0.1   # expect theory to fail below this
PRE_REG_KAPPA_C_HIGH = 0.5  # expect theory to hold above this
PRE_REG_R2_THRESHOLD = 0.5  # R2 threshold for "theory holds"

# ================================================================
# DATA GENERATION
# ================================================================
def make_kappa_sweep_data(K, d, target_kappa, n_per_class, seed=42):
    """
    Generate K-class Gaussian data with controlled kappa_nearest.

    kappa_nearest = min_{j!=k} ||mu_k - mu_j|| / (sigma_W * sqrt(d))

    We place classes along a 1D line (equidistant) in the first dimension,
    with sigma_W = 1.0 (identity covariance).

    inter-class gap = target_kappa * sigma_W * sqrt(d) / (K - 1)?
    No -- let's use: ||mu_k - mu_{k+1}|| = target_kappa * sigma_W * sqrt(d)
    So kappa_nearest = target_kappa (all pairs equally spaced along first dim).
    """
    rng = np.random.default_rng(seed)
    sigma_W = 1.0

    # Place class means equidistant along dim 0
    inter_gap = target_kappa * sigma_W * np.sqrt(d)
    means = np.zeros((K, d))
    for k in range(K):
        means[k, 0] = k * inter_gap  # evenly spaced along first dim

    # Generate data
    X_list, y_list = [], []
    for k in range(K):
        X_k = rng.normal(means[k], sigma_W, size=(n_per_class, d))
        X_list.append(X_k)
        y_list.extend([k] * n_per_class)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y, means, sigma_W


def compute_kappa_nearest_exact(X, y, K):
    """Exact kappa_nearest from class means."""
    classes = np.unique(y)
    K = len(classes)
    mu = {k: X[y == k].mean(0) for k in classes}

    # Pooled within-class std
    total_var = sum(np.sum((X[y == k] - mu[k])**2) for k in classes)
    n_total = len(X)
    sigma_W = float(np.sqrt(total_var / (n_total * X.shape[1])))

    # kappa for each class
    all_kappa = []
    for k in classes:
        min_dist = min(
            np.linalg.norm(mu[k] - mu[j])
            for j in classes if j != k
        )
        all_kappa.append(min_dist / (sigma_W * np.sqrt(X.shape[1])))

    return float(np.mean(all_kappa)), float(np.min(all_kappa)), sigma_W


def compute_q_1nn(X, y, subsample=2000, seed=42):
    """1-NN quality q = (acc - 1/K) / (1 - 1/K)."""
    rng = np.random.default_rng(seed)
    if len(X) > subsample:
        idx = rng.choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]

    K_eff = len(np.unique(y))
    if K_eff < 2:
        return None, None

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None, None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = (acc - 1.0 / K_eff) / (1.0 - 1.0 / K_eff)
    return float(q), float(acc)


# ================================================================
# MAIN SWEEP
# ================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("OVERLAP SWEEP: Domain of Validity for kappa_nearest Law")
    print(f"Pre-registered: theory holds (R2>0.7) for kappa > {PRE_REG_KAPPA_C_HIGH}")
    print(f"Pre-registered: theory fails for kappa < {PRE_REG_KAPPA_C_LOW}")
    print("=" * 70)

    # Sweep parameters
    kappa_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0,
                              1.5, 2.0, 2.5, 3.0])
    K_values = [4, 14, 20]   # different class counts
    d_values = [50, 100, 200]  # different dimensions (all will give similar kappa)
    n_per_class = 200
    n_seeds = 5

    all_points = []

    for K in K_values:
        print(f"\n--- K={K} ---")
        for d in d_values:
            print(f"  d={d}", end="", flush=True)
            for target_kappa in kappa_values:
                q_list, kappa_list = [], []
                for seed in range(n_seeds):
                    X, y, means, sigma_W = make_kappa_sweep_data(
                        K=K, d=d, target_kappa=target_kappa,
                        n_per_class=n_per_class, seed=seed * 100 + K * 10 + d
                    )
                    kappa_near, kappa_min, sw = compute_kappa_nearest_exact(X, y, K)
                    q, acc = compute_q_1nn(X, y, subsample=n_per_class * K)

                    if q is not None and not np.isnan(q):
                        q_list.append(q)
                        kappa_list.append(kappa_near)

                if q_list:
                    q_mean = float(np.mean(q_list))
                    q_std = float(np.std(q_list))
                    kappa_mean = float(np.mean(kappa_list))
                    logit_q = float(np.log(max(q_mean, 0.001) / max(1 - q_mean, 0.001)))

                    pt = {
                        "K": K, "d": d, "target_kappa": float(target_kappa),
                        "kappa_nearest": kappa_mean,
                        "q_mean": q_mean, "q_std": q_std,
                        "logit_q": logit_q,
                        "logKm1": float(np.log(K - 1)),
                        "n_valid": len(q_list),
                    }
                    all_points.append(pt)
                    print(".", end="", flush=True)
            print()

    print(f"\n\nTotal points: {len(all_points)}")

    # ================================================================
    # ANALYSIS: Per-K within-task sweep (avoid K-confound)
    # The key test: within each K, does kappa predict q?
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: Per-K within-task (kappa sweep with K fixed)")
    print("=" * 70)

    regime_results = {}
    per_k_alphas = {}

    for K in K_values:
        pts_K = [p for p in all_points if p["K"] == K]
        if len(pts_K) < 5:
            continue

        k_arr = np.array([p["kappa_nearest"] for p in pts_K])
        y_arr = np.array([p["logit_q"] for p in pts_K])
        A = np.column_stack([k_arr, np.ones(len(k_arr))])

        try:
            c, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
            y_pred = A @ c
            r2 = 1 - np.sum((y_arr - y_pred)**2) / np.sum((y_arr - y_arr.mean())**2)
            print(f"  K={K:2d} (N={len(pts_K)}): alpha={c[0]:.3f}  C={c[1]:.3f}  "
                  f"R2={r2:.3f}  {'HOLDS' if r2 > PRE_REG_R2_THRESHOLD else 'FAILS'}")
            per_k_alphas[K] = float(c[0])
            regime_results[f"K{K}_all"] = {"K": K, "n": len(pts_K),
                                             "alpha": float(c[0]), "r2": float(r2)}
        except Exception as e:
            print(f"  K={K}: fit failed ({e})")

    # Check alpha consistency across K
    if len(per_k_alphas) >= 2:
        alpha_vals = list(per_k_alphas.values())
        cv_alpha = np.std(alpha_vals) / (abs(np.mean(alpha_vals)) + 1e-10)
        print(f"\n  alpha across K: {[round(a,3) for a in alpha_vals]}")
        print(f"  mean={np.mean(alpha_vals):.3f}  CV={cv_alpha:.3f}  "
              f"{'PASS' if cv_alpha < 0.25 else 'FAIL'}")

    # Per-K by kappa regime (low vs. high kappa, within K)
    print("\n" + "=" * 70)
    print("ANALYSIS: Within-K R2 by kappa regime (low/mid/high kappa)")
    print("=" * 70)

    kappa_thresholds = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    for K in K_values:
        print(f"\n  K={K}:")
        for thresh in kappa_thresholds:
            pts = [p for p in all_points if p["K"] == K and p["kappa_nearest"] >= thresh]
            if len(pts) < 5:
                continue
            k_arr = np.array([p["kappa_nearest"] for p in pts])
            y_arr = np.array([p["logit_q"] for p in pts])
            A = np.column_stack([k_arr, np.ones(len(k_arr))])
            try:
                c, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
                y_pred = A @ c
                ss_res = np.sum((y_arr - y_pred)**2)
                ss_tot = np.sum((y_arr - y_arr.mean())**2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                print(f"    kappa >= {thresh:.2f} (N={len(pts):2d}): "
                      f"alpha={c[0]:.3f}  R2={r2:.3f}  "
                      f"{'HOLDS' if r2 > PRE_REG_R2_THRESHOLD else 'FAILS'}")
                regime_results[f"K{K}_kappa_ge_{thresh}"] = {
                    "K": K, "threshold": float(thresh), "n": len(pts),
                    "alpha": float(c[0]), "r2": float(r2),
                    "holds": bool(r2 > PRE_REG_R2_THRESHOLD)
                }
            except Exception as e:
                print(f"    kappa >= {thresh:.2f}: fit failed ({e})")

    # Find kappa_c per K (within-K sweep to find where theory holds)
    print("\n" + "=" * 70)
    print("CRITICAL KAPPA SWEEP (per K): Find where law first holds R2 > 0.7")
    print("=" * 70)

    kappa_c_candidates = np.arange(0.05, 2.5, 0.1)
    r2_curve = {}
    kappa_c_per_K = {}

    for K in K_values:
        r2_curve[K] = []
        kappa_c_per_K[K] = None
        pts_K = [p for p in all_points if p["K"] == K]

        for kc in kappa_c_candidates:
            pts = [p for p in pts_K if p["kappa_nearest"] >= kc]
            if len(pts) < 5:
                r2_curve[K].append((float(kc), None, 0))
                continue

            k_arr = np.array([p["kappa_nearest"] for p in pts])
            y_arr = np.array([p["logit_q"] for p in pts])
            A = np.column_stack([k_arr, np.ones(len(k_arr))])
            try:
                c, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
                y_pred = A @ c
                ss_tot = np.sum((y_arr - y_arr.mean())**2)
                r2 = float(1 - np.sum((y_arr - y_pred)**2) / ss_tot) if ss_tot > 1e-10 else 0.0
                r2_curve[K].append((float(kc), r2, len(pts)))
            except:
                r2_curve[K].append((float(kc), None, len(pts)))

        # Print R2 curve for this K
        print(f"\n  K={K} R2 curve:")
        for kc, r2, n in r2_curve[K][:15]:
            if r2 is not None and n >= 5:
                print(f"    kappa_c={kc:.2f}: R2={r2:.3f}  N={n}  "
                      f"{'HOLDS' if r2 > 0.7 else 'FAILS'}")

    # ================================================================
    # OVERALL ASSESSMENT
    # ================================================================
    print("\n" + "=" * 70)
    print("PRE-REGISTERED TEST RESULTS")
    print("=" * 70)

    # Check full-range per-K R2
    for K in K_values:
        full_r2 = regime_results.get(f"K{K}_all", {}).get("r2")
        full_alpha = regime_results.get(f"K{K}_all", {}).get("alpha")
        if full_r2 is not None:
            print(f"  K={K}: full-range alpha={full_alpha:.3f}  R2={full_r2:.3f}  "
                  f"{'HOLDS' if full_r2 > PRE_REG_R2_THRESHOLD else 'FAILS'}")

    # Alpha CV across K
    if len(per_k_alphas) >= 2:
        alpha_vals = list(per_k_alphas.values())
        cv_alpha = np.std(alpha_vals) / (abs(np.mean(alpha_vals)) + 1e-10)
        print(f"\n  alpha across K: {[round(a,3) for a in alpha_vals]}")
        print(f"  CV={cv_alpha:.3f}  {'PASS' if cv_alpha < 0.25 else 'FAIL'}")

    # Go-emotions regime check (kappa < 0.4 for first few kappa values per K)
    emotion_regime_K14 = regime_results.get("K14_kappa_ge_0.0", {}).get("r2")
    if emotion_regime_K14 is not None:
        print(f"\n  go_emotions regime (K=14, kappa<0.4): R2={emotion_regime_K14:.3f}")

    theory_has_phase_boundary = len([
        K for K in K_values
        if regime_results.get(f"K{K}_all", {}).get("r2", 0) > PRE_REG_R2_THRESHOLD
    ]) >= 2
    print(f"\n  Theory holds (R2>{PRE_REG_R2_THRESHOLD}) for >= 2 K values: {theory_has_phase_boundary}")

    kappa_c_estimate = None  # No single kappa_c from multi-K analysis

    # Save
    output = {
        "experiment": "overlap_sweep_domain_of_validity",
        "pre_registered": {
            "kappa_c_low": PRE_REG_KAPPA_C_LOW,
            "kappa_c_high": PRE_REG_KAPPA_C_HIGH,
            "r2_threshold": PRE_REG_R2_THRESHOLD
        },
        "all_points": all_points,
        "regime_results": regime_results,
        "per_k_alphas": per_k_alphas,
        "r2_curve": {str(K): r2_curve[K] for K in K_values},
        "kappa_c_per_K": kappa_c_per_K,
        "theory_has_phase_boundary": theory_has_phase_boundary,
        "runtime_s": int(time.time() - t0)
    }

    out_path = "results/cti_overlap_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")
    print(f"Runtime: {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
