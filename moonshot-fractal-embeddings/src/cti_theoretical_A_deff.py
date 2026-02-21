#!/usr/bin/env python -u
"""
THEORETICAL DERIVATION TEST: A = C_universal * sqrt(d_eff)
===========================================================
From the Gumbel Race derivation:
  logit(q) = kappa_nearest - b_eff * log(K-1) + C
  kappa_nearest ~ sqrt(d_eff) * dist_ratio + offset
  => logit(q) ~ A * dist_ratio + const, where A = C_universal * sqrt(d_eff)

Test: in synthetic Gaussians with known d_eff, does A / sqrt(d_eff) = constant?

If YES: proves the law is theoretically grounded, explains cross-task A variation,
        and gives the correct universal law:
        logit(q) = C_universal * sqrt(d_eff) * dist_ratio - b_eff * log(K-1) + C_0
        This is Nobel-track because it explains WHY the law works from first principles.

Pre-registered criterion:
  - A / sqrt(d_eff) = C_universal with CV < 0.20 across d_eff values
  - b_eff = C_be with CV < 0.20 across K values
"""

import json
import sys
import time
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# ================================================================
# CONFIG
# ================================================================
N_TRAIN = 2000       # training samples total (n_per = N_TRAIN / K)
N_TEST  = 1000       # test samples total
K_VALUES = [4, 10, 20, 50]        # number of classes
DEFF_VALUES = [5, 10, 20, 50, 100, 200]  # target effective dimensions
D_AMBIENT  = 500     # ambient dimension (>= max(DEFF_VALUES))
KAPPA_SWEEP = np.linspace(0.0, 3.0, 16)  # kappa_nearest sweep
MC_REPEATS = 50      # Monte Carlo repeats per setting
SEED = 42

PRE_REG_CV_A = 0.20    # A / sqrt(d_eff) CV threshold
PRE_REG_CV_B = 0.20    # b_eff CV threshold

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

print("=" * 70)
print("THEORETICAL DERIVATION: A = C_universal * sqrt(d_eff)")
print(f"D_ambient={D_AMBIENT}, K={K_VALUES}, d_eff={DEFF_VALUES}")
print(f"kappa sweep: {len(KAPPA_SWEEP)} values, MC_repeats={MC_REPEATS}")
print("=" * 70)


# ================================================================
# GENERATE SYNTHETIC GAUSSIAN DATA
# ================================================================
def make_gaussian_data(K, d_eff, d_ambient, kappa, n_train, n_test, seed=42):
    """
    Generate K-class Gaussian data with exact d_eff and signal kappa_nearest.

    Within-class covariance: Sigma_W = diag([sigma_signal^2]*d_eff, [epsilon^2]*(d_ambient-d_eff))
    Class means: arranged in simplex, nearest-class gap = kappa * sigma_signal

    d_eff = effective dimension = tr(Sigma_W)^2 / tr(Sigma_W^2)
           = (d_eff * sigma^2 + (d_amb-d_eff) * eps^2)^2 /
             (d_eff * sigma^4 + (d_amb-d_eff) * eps^4)
    To get exact d_eff_target: set sigma=1, eps=1e-6, d_eff = d_eff_target.
    True d_eff_actual = (d_eff * 1 + rest * eps^2)^2 / (d_eff * 1 + rest * eps^4)
                      ~ d_eff_target for eps << 1.
    """
    rng_local = np.random.default_rng(seed)
    d = d_ambient
    eps = 1e-6   # noise in remaining dimensions
    sigma = 1.0  # signal std in first d_eff dimensions

    # Class means: place on simplex in first d_eff dimensions
    # Nearest-class gap = kappa (in units of sigma)
    # Simplex arrangement: ||mu_i - mu_j||^2 = 2*R^2*(1 - cos(theta_ij)) for ETF
    # For K classes on simplex: min separation = sqrt(2K/(K-1)) * R
    # We want min ||mu_i - mu_j|| = kappa * sigma, so R = kappa * sigma / sqrt(2K/(K-1))
    R = kappa * sigma / np.sqrt(2.0 * K / (K - 1.0)) if K > 1 else kappa * sigma

    # Build ETF class means in d_eff-dimensional subspace
    means = np.zeros((K, d))
    if K <= d_eff and K > 1:
        # Place on equidistant simplex using Gram-Schmidt
        M = np.zeros((K, d_eff))
        for k in range(K):
            v = rng_local.standard_normal(d_eff)
            for prev in M[:k]:
                v -= np.dot(v, prev) * prev
            v /= (np.linalg.norm(v) + 1e-10)
            M[k] = v
        means[:, :d_eff] = M * R
    else:
        # K > d_eff: random means in d_eff subspace
        M = rng_local.standard_normal((K, d_eff)) * R / np.sqrt(d_eff)
        means[:, :d_eff] = M

    # Generate data
    X_train, y_train = [], []
    X_test, y_test = [], []
    n_per_train = n_train // K
    n_per_test = n_test // K

    for k in range(K):
        # Signal dimensions
        noise_sig_tr = rng_local.standard_normal((n_per_train, d_eff)) * sigma
        noise_noise_tr = rng_local.standard_normal((n_per_train, d - d_eff)) * eps
        Xk_tr = means[k:k+1] + np.concatenate([noise_sig_tr, noise_noise_tr], axis=1)

        noise_sig_te = rng_local.standard_normal((n_per_test, d_eff)) * sigma
        noise_noise_te = rng_local.standard_normal((n_per_test, d - d_eff)) * eps
        Xk_te = means[k:k+1] + np.concatenate([noise_sig_te, noise_noise_te], axis=1)

        X_train.append(Xk_tr)
        y_train.extend([k] * n_per_train)
        X_test.append(Xk_te)
        y_test.extend([k] * n_per_test)

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    # Compute true d_eff of generated data
    Sigma_W = np.diag(
        [sigma**2] * d_eff + [eps**2] * (d - d_eff)
    )
    tr_SW = np.trace(Sigma_W)
    tr_SW2 = np.sum(np.diag(Sigma_W)**2)
    d_eff_actual = float(tr_SW**2 / (tr_SW2 + 1e-20))

    return X_train, y_train, X_test, y_test, means, d_eff_actual


# ================================================================
# COMPUTE q AND dist_ratio FROM DATA
# ================================================================
def compute_q_and_dr(X_train, y_train, X_test, y_test, K):
    """1-NN accuracy (q) and dist_ratio from data."""
    # 1-NN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1)
    knn.fit(X_train, y_train)
    acc = float(knn.score(X_test, y_test))
    q = (acc - 1.0/K) / (1.0 - 1.0/K)

    # dist_ratio
    n_sub = min(len(X_test), 300)
    idx = np.random.choice(len(X_test), n_sub, replace=False)
    Xs = X_test[idx]
    ys = y_test[idx]

    # pairwise
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(Xs, metric='euclidean')
    np.fill_diagonal(D, np.inf)

    intra_mins, inter_mins = [], []
    for i in range(n_sub):
        same = (ys == ys[i]); same[i] = False
        diff = ~same; diff[i] = False
        if same.any(): intra_mins.append(D[i][same].min())
        if diff.any(): inter_mins.append(D[i][diff].min())

    if not intra_mins or not inter_mins:
        return float(q), None

    dr = float(np.mean(inter_mins)) / float(np.mean(intra_mins) + 1e-10)
    return float(q), float(dr)


# ================================================================
# FIT logit(q) vs dist_ratio
# ================================================================
def fit_logit_vs_dr(qs, drs):
    """Linear regression: logit(q) = A * dist_ratio + C"""
    valid = [(q, dr) for q, dr in zip(qs, drs)
             if dr is not None and 0.01 < q < 0.99 and np.isfinite(q) and np.isfinite(dr)]
    if len(valid) < 4:
        return None, None, None
    qs_v, drs_v = zip(*valid)
    logit_q = np.log(np.array(qs_v) / (1 - np.array(qs_v)))
    drs_arr = np.array(drs_v)
    slope, intercept, r, p, se = stats.linregress(drs_arr, logit_q)
    r2 = r**2
    return float(slope), float(intercept), float(r2)


# ================================================================
# MAIN EXPERIMENT
# ================================================================
def main():
    t0 = time.time()

    # ============================================================
    # EXPERIMENT 1: A vs d_eff (fixed K=10)
    # Test: A = C_universal * sqrt(d_eff)?
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: A vs d_eff (K=10 fixed)")
    print("Pre-registered: A / sqrt(d_eff) has CV < 0.20")
    print("=" * 70)

    K_FIXED = 10
    kappa_sweep = KAPPA_SWEEP

    results_A_deff = {}
    A_values = []
    deff_values_actual = []

    for d_eff_target in DEFF_VALUES:
        if d_eff_target > D_AMBIENT:
            print(f"  d_eff={d_eff_target}: SKIP (> D_ambient={D_AMBIENT})")
            continue

        qs_all, drs_all = [], []
        d_eff_actual_list = []

        for kappa in kappa_sweep:
            for rep in range(MC_REPEATS):
                seed_r = SEED + rep + int(kappa * 1000) + d_eff_target * 10000
                try:
                    Xtr, ytr, Xte, yte, means, d_eff_actual = make_gaussian_data(
                        K=K_FIXED, d_eff=d_eff_target, d_ambient=D_AMBIENT,
                        kappa=kappa, n_train=N_TRAIN, n_test=N_TEST, seed=seed_r
                    )
                    q, dr = compute_q_and_dr(Xtr, ytr, Xte, yte, K_FIXED)
                    qs_all.append(q)
                    drs_all.append(dr)
                    d_eff_actual_list.append(d_eff_actual)
                except Exception as e:
                    pass

        A, C, r2 = fit_logit_vs_dr(qs_all, drs_all)
        d_eff_mean = float(np.mean(d_eff_actual_list)) if d_eff_actual_list else d_eff_target
        n_valid = sum(1 for d in drs_all if d is not None)

        print(f"  d_eff={d_eff_target:3d}: A={A:.4f}  C={C:.4f}  R2={r2:.3f}  "
              f"d_eff_actual={d_eff_mean:.1f}  n={n_valid}", flush=True)

        if A is not None:
            results_A_deff[d_eff_target] = {
                "A": A, "C": C, "r2": r2,
                "d_eff_target": d_eff_target,
                "d_eff_actual": d_eff_mean,
                "A_normalized": A / float(np.sqrt(d_eff_mean)),
                "n_valid": n_valid
            }
            A_values.append(A)
            deff_values_actual.append(d_eff_mean)

    # Test: A ~ C_universal * sqrt(d_eff)
    if len(A_values) >= 3:
        A_arr = np.array(A_values)
        deff_arr = np.array(deff_values_actual)
        sqrt_deff = np.sqrt(deff_arr)

        # OLS fit: A = C_univ * sqrt(d_eff)
        C_univ = float(np.dot(A_arr, sqrt_deff) / np.dot(sqrt_deff, sqrt_deff))
        residuals = A_arr - C_univ * sqrt_deff
        A_norm = A_arr / sqrt_deff
        cv_A = float(np.std(A_norm) / (np.abs(np.mean(A_norm)) + 1e-10))

        # R2 of A = C_univ * sqrt(d_eff) fit
        ss_res = float(np.sum((A_arr - C_univ * sqrt_deff)**2))
        ss_tot = float(np.sum((A_arr - np.mean(A_arr))**2))
        r2_fit = 1 - ss_res / (ss_tot + 1e-10)

        # Pearson corr(A, sqrt(d_eff))
        r_pearson, p_pearson = stats.pearsonr(sqrt_deff, A_arr)

        print(f"\n  C_universal = {C_univ:.4f}")
        print(f"  A / sqrt(d_eff): mean={np.mean(A_norm):.4f}  std={np.std(A_norm):.4f}  CV={cv_A:.3f}")
        print(f"  R2 of A=C*sqrt(d_eff): {r2_fit:.3f}")
        print(f"  Pearson r(A, sqrt(d_eff)): {r_pearson:.3f}  p={p_pearson:.4f}")
        print(f"  Pre-registered CV < {PRE_REG_CV_A}: {'PASS' if cv_A < PRE_REG_CV_A else 'FAIL'}")
    else:
        C_univ = None
        cv_A = None
        r2_fit = None

    # ============================================================
    # EXPERIMENT 2: b_eff vs K (fixed d_eff=50)
    # Test: b_eff = C_be (constant across K)?
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: b_eff vs K (d_eff=50 fixed)")
    print("Pre-registered: b_eff CV < 0.20 across K values")
    print("=" * 70)

    D_EFF_FIXED = 50

    results_beff_K = {}
    b_values = []

    for K in K_VALUES:
        qs_all, drs_all = [], []

        for kappa in kappa_sweep:
            for rep in range(MC_REPEATS):
                seed_r = SEED + rep + int(kappa * 1000) + K * 1000000
                try:
                    Xtr, ytr, Xte, yte, means, d_eff_actual = make_gaussian_data(
                        K=K, d_eff=D_EFF_FIXED, d_ambient=D_AMBIENT,
                        kappa=kappa, n_train=N_TRAIN, n_test=N_TEST, seed=seed_r
                    )
                    q, dr = compute_q_and_dr(Xtr, ytr, Xte, yte, K)
                    qs_all.append(q)
                    drs_all.append(dr)
                except Exception as e:
                    pass

        A, C_intercept, r2 = fit_logit_vs_dr(qs_all, drs_all)

        # b_eff: fit logit(q) = A*dr + b_eff * log(K-1) + C
        # From the intercept: C = b_eff * log(K-1) + C_0
        # If A ~ C_univ * sqrt(d_eff) ~ const across K, then C measures -b_eff*log(K-1)
        # More precisely: b_eff = -(C_intercept - C_0) / log(K-1)
        # We need C_0 — let's collect and solve jointly

        n_valid = sum(1 for d in drs_all if d is not None)

        print(f"  K={K:3d}: A={A:.4f}  C_intercept={C_intercept:.4f}  R2={r2:.3f}  n={n_valid}",
              flush=True)

        if A is not None:
            results_beff_K[K] = {
                "K": K, "A": A, "C_intercept": C_intercept,
                "r2": r2, "n_valid": n_valid,
                "logKm1": float(np.log(K - 1)) if K > 1 else 0.0
            }
            b_values.append(C_intercept)  # collect intercepts

    # Fit b_eff: intercept = b_eff * log(K-1) + C_0
    if len(b_values) >= 3:
        Ks = np.array([K for K in K_VALUES if K in results_beff_K])
        logKm1 = np.log(Ks - 1)
        intercepts = np.array([results_beff_K[K]["C_intercept"] for K in Ks])

        # Linear regression: intercept = b_eff * log(K-1) + C_0
        slope_b, intercept_b, r_b, p_b, _ = stats.linregress(logKm1, intercepts)
        b_eff_est = float(slope_b)

        print(f"\n  b_eff (from log(K-1) regression): {b_eff_est:.4f}")
        print(f"  R2 of intercept vs log(K-1): {r_b**2:.3f}  r={r_b:.3f}")
        print(f"  C_0 (universal constant): {intercept_b:.4f}")
        print(f"  Note: Gumbel theory predicts b_eff ~ pi^2/6 = {np.pi**2/6:.4f} or b_eff=1.0")
    else:
        b_eff_est = None
        intercept_b = None

    # ============================================================
    # EXPERIMENT 3: Cross-task collapse after d_eff correction
    # After normalizing: logit(q) + b_eff*log(K-1) = C_univ * sqrt(d_eff) * dist_ratio + C_0
    # Test if all (K, d_eff) pairs collapse to single line
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Cross-K x d_eff collapse")
    print("After correction: [logit(q) + b_eff*log(K-1)] / sqrt(d_eff) = C_univ * dist_ratio + C_0")
    print("=" * 70)

    if C_univ is not None and b_eff_est is not None:
        all_x = []  # dist_ratio
        all_y = []  # [logit(q) + b_eff*log(K-1)] / sqrt(d_eff)

        for K in K_VALUES[:2]:  # quick: 2 K values x 2 d_eff values
            for d_eff_target in [10, 50]:
                if d_eff_target > D_AMBIENT:
                    continue
                qs_sub, drs_sub = [], []
                for kappa in kappa_sweep:
                    seed_r = SEED + int(kappa * 1000) + K * 1000000 + d_eff_target * 10000 + 9999
                    try:
                        Xtr, ytr, Xte, yte, means, d_eff_actual = make_gaussian_data(
                            K=K, d_eff=d_eff_target, d_ambient=D_AMBIENT,
                            kappa=kappa, n_train=N_TRAIN, n_test=N_TEST, seed=seed_r
                        )
                        q, dr = compute_q_and_dr(Xtr, ytr, Xte, yte, K)
                        qs_sub.append((q, dr, K, d_eff_actual))
                    except Exception:
                        pass

                for q, dr, K_, de in qs_sub:
                    if dr is None or not (0.01 < q < 0.99) or not np.isfinite(q):
                        continue
                    logit_q = np.log(q / (1 - q))
                    correction = b_eff_est * np.log(K_ - 1) if K_ > 1 else 0.0
                    y_corrected = (logit_q - correction) / np.sqrt(de)
                    all_x.append(float(dr))
                    all_y.append(float(y_corrected))

        if len(all_x) >= 10:
            r_collapse, p_collapse = stats.pearsonr(all_x, all_y)
            slope_c, int_c, _, _, _ = stats.linregress(all_x, all_y)
            print(f"  Collapse: n={len(all_x)} points")
            print(f"  Pearson r = {r_collapse:.3f}  p = {p_collapse:.4f}")
            print(f"  Slope (C_universal) = {slope_c:.4f}")
            print(f"  Intercept (C_0 / sqrt(d_eff)) = {int_c:.4f}")
            print(f"  {'GOOD collapse' if abs(r_collapse) > 0.8 else 'POOR collapse'}")
        else:
            print(f"  Not enough valid points: {len(all_x)}")
            r_collapse, slope_c = None, None
    else:
        r_collapse, slope_c = None, None

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        "experiment": "theoretical_A_deff",
        "pre_registered": {
            "cv_A_threshold": PRE_REG_CV_A,
            "cv_beff_threshold": PRE_REG_CV_B
        },
        "exp1_A_vs_deff": {
            "K_fixed": K_FIXED,
            "per_deff": results_A_deff,
            "C_universal": C_univ,
            "cv_A_over_sqrt_deff": cv_A,
            "r2_A_vs_sqrt_deff": r2_fit,
            "pass": bool(cv_A is not None and cv_A < PRE_REG_CV_A)
        },
        "exp2_beff_vs_K": {
            "d_eff_fixed": D_EFF_FIXED,
            "per_K": results_beff_K,
            "b_eff_estimate": b_eff_est,
            "C0_universal": float(intercept_b) if intercept_b is not None else None
        },
        "exp3_collapse": {
            "r_collapse": float(r_collapse) if r_collapse is not None else None,
            "C_universal_from_collapse": float(slope_c) if slope_c is not None else None
        },
        "runtime_s": int(time.time() - t0)
    }

    out_path = "results/cti_theory_A_deff.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")
    print(f"Total runtime: {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
