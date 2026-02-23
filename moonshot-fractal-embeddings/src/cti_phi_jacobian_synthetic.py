#!/usr/bin/env python -u
"""
Phi Model Jacobian Validation: Synthetic Gaussian Test
========================================================
TESTS: Is d(logit_q)/d(kappa_j) = A * softmax(-kappa_j/tau*)?

In the Gumbel race model for K Gaussian classes:
  logit(q_ci) ≈ A * phi(tau*, kappas) + C
  phi(tau*) = -tau* * log(sum_j exp(-kappa_j/tau*))
  => d(logit_q)/d(kappa_j) = A * softmax(-kappa_j/tau*)_j

KEY PREDICTION: slope ratio w = slope_j2/slope_j1 = exp(-(kappa_j2-kappa_j1)/tau*)
This should follow an EXPONENTIAL DECAY in gap = kappa_j2 - kappa_j1.

DESIGN:
- K=14 Gaussian classes, d=50 dimensions
- For target class ci=0:
  - j1 at kappa=0.40 (fixed, moderate separation)
  - j2 at kappa=0.40+gap, gap in [0.15, 0.25, 0.35, 0.50, 0.70, 1.00]
  - Other 12 classes at kappa=2.0 (irrelevant)
- N=2000 per class (28000 total), large enough for low noise
- Stable-rank dose-response: delta in [0.02, 0.04, 0.06, 0.08, 0.10] kappa units
  (all < gap/3 for gap >= 0.15, ensuring stable rank throughout)

PRE-REGISTERED:
  1. r(log(w_empirical), -gap) > 0.85 (phi is functionally correct)
  2. tau*_fitted from regression in [0.10, 0.40]
  3. tau*_synthetic ≈ 0.2 (matches pooled empirical tau* from phi_upgrade_pooled)
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import pearsonr, linregress

OUT_JSON = Path("results/cti_phi_jacobian_synthetic.json")
RANDOM_STATE = 42
K = 14
D = 50  # embedding dimension
N_PER_CLASS = 2000  # per-class samples (28K total)
KAPPA_J1 = 0.40  # fixed j1 kappa
KAPPA_OTHER = 2.00  # other classes (irrelevant)
GAP_LIST = [0.15, 0.25, 0.35, 0.50, 0.70, 1.00]  # kappa_j2 = kappa_j1 + gap
DELTA_LIST = [0.02, 0.04, 0.06, 0.08, 0.10]  # dose-response (stable-rank for gap >= 0.15)
TEST_SIZE = 0.2
TAU_STAR_POOLED = 0.2  # from phi_upgrade_pooled


def generate_gaussian_embeddings(kappa_j1, gap, sigma_W_target=1.0, seed=0):
    """
    Generate K=14 Gaussian classes in d=50 dims.
    Class 0 (target) at origin.
    Class 1 (j1) at kappa_j1 * sigma_W * sqrt(d) along direction e_1.
    Class 2 (j2) at (kappa_j1+gap) * sigma_W * sqrt(d) along direction e_2.
    Classes 3-13 at kappa_other * sigma_W * sqrt(d) in orthogonal directions.
    All with within-class std = sigma_W_target per dimension.
    """
    rng = np.random.default_rng(seed)
    sigma_W = sigma_W_target

    # Centroid distances: kappa * sigma_W * sqrt(d)
    dist_j1 = kappa_j1 * sigma_W * np.sqrt(D)
    dist_j2 = (kappa_j1 + gap) * sigma_W * np.sqrt(D)
    dist_other = KAPPA_OTHER * sigma_W * np.sqrt(D)

    # Orthogonal centroid directions (e_0, e_1, ..., e_12)
    # Using first K directions of the standard basis (d >= K)
    centroid_directions = np.eye(K, D)

    # Assign centroids relative to class 0 = origin
    centroids = np.zeros((K, D))
    centroids[0] = 0.0  # target class at origin
    centroids[1] = dist_j1 * centroid_directions[1]  # j1
    centroids[2] = dist_j2 * centroid_directions[2]  # j2
    for c in range(3, K):
        centroids[c] = dist_other * centroid_directions[c]

    # Generate samples
    X_list, y_list = [], []
    for c in range(K):
        Xc = rng.normal(loc=centroids[c], scale=sigma_W, size=(N_PER_CLASS, D))
        X_list.append(Xc)
        y_list.append(np.full(N_PER_CLASS, c, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, centroids, sigma_W


def compute_kappa(X, y, ci, cj):
    """Compute kappa(ci, cj) from embeddings."""
    mu_ci = X[y == ci].mean(0)
    mu_cj = X[y == cj].mean(0)
    dist = float(np.linalg.norm(mu_ci - mu_cj))
    # sigma_W from all residuals
    classes = np.unique(y)
    resids = [X[y == c] - X[y == c].mean(0) for c in classes]
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R ** 2)))
    d = X.shape[1]
    return dist / (sigma_W * np.sqrt(d) + 1e-10), sigma_W


def eval_q_ci(X_tr, y_tr, X_te, y_te, ci):
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
    knn.fit(X_tr, y_tr)
    mask = y_te == ci
    if mask.sum() == 0:
        return None
    preds = knn.predict(X_te[mask])
    q_raw = float((preds == ci).mean())
    K_local = len(np.unique(y_tr))
    return float((q_raw - 1.0 / K_local) / (1.0 - 1.0 / K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1 - 1e-5))
    return float(np.log(q / (1.0 - q)))


def fit_slope(x_vals, y_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or x.std() < 1e-8:
        return None, None
    slope, _, r, _, _ = linregress(x, y)
    return float(slope), float(r)


def main():
    print("=" * 70)
    print("PHI JACOBIAN SYNTHETIC: Mechanistic Test on Controlled Gaussians")
    print(f"K={K}, d={D}, N/class={N_PER_CLASS}")
    print(f"kappa_j1={KAPPA_J1} (fixed), gaps={GAP_LIST}")
    print(f"delta_list={DELTA_LIST} (stable-rank for gap >= {min(GAP_LIST):.2f})")
    print("=" * 70)

    records = []

    for gap in GAP_LIST:
        kappa_j2_true = KAPPA_J1 + gap
        print(f"\n--- gap={gap:.2f} (kappa_j1={KAPPA_J1:.2f}, kappa_j2={kappa_j2_true:.2f}) ---")

        # Generate embeddings
        X, y, centroids, sigma_W = generate_gaussian_embeddings(KAPPA_J1, gap, seed=RANDOM_STATE)
        d = X.shape[1]

        # Fixed train/test split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(sss.split(X, y))
        X_tr_base, X_te_base = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Verify actual kappas (from embeddings, not construction)
        kappa_j1_actual, sigma_W_actual = compute_kappa(X, y, 0, 1)
        kappa_j2_actual, _ = compute_kappa(X, y, 0, 2)
        gap_actual = kappa_j2_actual - kappa_j1_actual

        # Baseline q for class 0 (target)
        q_base = eval_q_ci(X_tr_base, y_tr, X_te_base, y_te, 0)
        logit_base = safe_logit(q_base)
        print(f"  kappa_j1_actual={kappa_j1_actual:.4f}, kappa_j2_actual={kappa_j2_actual:.4f}, "
              f"gap_actual={gap_actual:.4f}")
        print(f"  baseline q={q_base:.4f}, logit={logit_base:.4f}")
        n_test_ci = (y_te == 0).sum()
        print(f"  test samples for ci=0: {n_test_ci}")

        # Dose-response for j1 (class 1) and j2 (class 2)
        ci = 0
        j1_class, j2_class = 1, 2

        j1_deltas, j1_dlogits = [0.0], [0.0]
        j2_deltas, j2_dlogits = [0.0], [0.0]

        ci_centroid = X[y == ci].mean(0)
        j1_centroid = X[y == j1_class].mean(0)
        j2_centroid = X[y == j2_class].mean(0)

        for delta in DELTA_LIST:
            # j1-only arm: shift class 1 away from class 0
            X_tr_j1 = X_tr_base.copy()
            X_te_j1 = X_te_base.copy()
            dir_j1 = j1_centroid - ci_centroid
            dir_j1 = dir_j1 / (np.linalg.norm(dir_j1) + 1e-10)
            shift_j1 = delta * sigma_W_actual * np.sqrt(d) * dir_j1
            X_tr_j1[y_tr == j1_class] += shift_j1
            X_te_j1[y_te == j1_class] += shift_j1

            q_j1 = eval_q_ci(X_tr_j1, y_tr, X_te_j1, y_te, ci)
            if q_j1 is not None:
                j1_deltas.append(delta)
                j1_dlogits.append(safe_logit(q_j1) - logit_base)

            # j2-only arm: shift class 2 away from class 0
            X_tr_j2 = X_tr_base.copy()
            X_te_j2 = X_te_base.copy()
            dir_j2 = j2_centroid - ci_centroid
            dir_j2 = dir_j2 / (np.linalg.norm(dir_j2) + 1e-10)
            shift_j2 = delta * sigma_W_actual * np.sqrt(d) * dir_j2
            X_tr_j2[y_tr == j2_class] += shift_j2
            X_te_j2[y_te == j2_class] += shift_j2

            q_j2 = eval_q_ci(X_tr_j2, y_tr, X_te_j2, y_te, ci)
            if q_j2 is not None:
                j2_deltas.append(delta)
                j2_dlogits.append(safe_logit(q_j2) - logit_base)

        slope_j1, r_j1 = fit_slope(j1_deltas, j1_dlogits)
        slope_j2, r_j2 = fit_slope(j2_deltas, j2_dlogits)

        s1_str = f"{slope_j1:.4f}" if slope_j1 is not None else "N/A"
        r1_str = f"{r_j1:.4f}" if r_j1 is not None else "N/A"
        s2_str = f"{slope_j2:.4f}" if slope_j2 is not None else "N/A"
        r2_str = f"{r_j2:.4f}" if r_j2 is not None else "N/A"
        print(f"  slope_j1={s1_str} (r={r1_str}), slope_j2={s2_str} (r={r2_str})")

        w_emp = None
        tau_est = None
        if (slope_j1 is not None and slope_j2 is not None
                and abs(slope_j1) > 0.01 and slope_j1 > 0 and slope_j2 > 0
                and slope_j2 / slope_j1 < 5):  # filter extreme outliers
            w_emp = float(slope_j2 / slope_j1)
            if 0 < w_emp < 1.0:
                tau_est = float(-gap_actual / np.log(w_emp))

        w_phi = float(np.exp(-gap_actual / TAU_STAR_POOLED))

        w_str = f"{w_emp:.3f}" if w_emp is not None else "N/A"
        tau_str = f"{tau_est:.3f}" if tau_est is not None else "N/A"
        print(f"  w_empirical={w_str}, tau*_est={tau_str}, w_phi(0.2)={w_phi:.3f}")

        # j1 dose-response summary
        print(f"  j1 dose-resp: {list(zip([f'{x:.2f}' for x in j1_deltas], [f'{y:.3f}' for y in j1_dlogits]))}")
        print(f"  j2 dose-resp: {list(zip([f'{x:.2f}' for x in j2_deltas], [f'{y:.3f}' for y in j2_dlogits]))}")

        records.append({
            "gap_requested": float(gap),
            "kappa_j1": float(kappa_j1_actual),
            "kappa_j2": float(kappa_j2_actual),
            "gap_actual": float(gap_actual),
            "q_base": float(q_base),
            "logit_base": float(logit_base),
            "n_test_ci": int(n_test_ci),
            "slope_j1": float(slope_j1) if slope_j1 is not None else None,
            "slope_j2": float(slope_j2) if slope_j2 is not None else None,
            "r_j1": float(r_j1) if r_j1 is not None else None,
            "r_j2": float(r_j2) if r_j2 is not None else None,
            "w_empirical": float(w_emp) if w_emp is not None else None,
            "tau_star_est": float(tau_est) if tau_est is not None else None,
            "w_phi_tau02": float(w_phi),
            "j1_deltas": j1_deltas,
            "j1_dlogits": j1_dlogits,
            "j2_deltas": j2_deltas,
            "j2_dlogits": j2_dlogits,
        })

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: Gap-Conditioned Phi Jacobian")
    print("=" * 70)

    valid = [r for r in records
             if r.get("w_empirical") is not None and np.isfinite(r["w_empirical"])]

    print(f"\nValid w estimates: {len(valid)} / {len(records)}")

    # Correlation test
    if len(valid) >= 4:
        gaps = np.array([r["gap_actual"] for r in valid])
        log_w = np.array([np.log(r["w_empirical"]) for r in valid])
        if np.all(np.isfinite(log_w)):
            corr_r, corr_p = pearsonr(-gaps, log_w)
            print(f"\nCorrelation: log(w) vs -gap (n={len(valid)})")
            print(f"  Pearson r = {corr_r:.4f}, p = {corr_p:.4e}")
            print(f"  PASS (r > 0.85): {'PASS' if corr_r > 0.85 else 'FAIL'}")

            # Fit: log(w) = -gap/tau* => tau* = -gap / log(w)
            tau_vals = [-g / lw for g, lw in zip(gaps, log_w) if lw < 0]
            if tau_vals:
                tau_med = float(np.median(tau_vals))
                tau_mean = float(np.mean(tau_vals))
                tau_std = float(np.std(tau_vals))
                print(f"\nFitted tau* from individual estimates: {[f'{t:.3f}' for t in tau_vals]}")
                print(f"  median={tau_med:.4f}, mean={tau_mean:.4f}, std={tau_std:.4f}")
                print(f"  Pooled empirical tau* = {TAU_STAR_POOLED}")
                print(f"  PASS (tau in [0.10, 0.40]): {'PASS' if 0.10 < tau_med < 0.40 else 'FAIL'}")
                print(f"  MATCH (|tau-0.2|<0.10): {'PASS' if abs(tau_med-TAU_STAR_POOLED)<0.10 else 'FAIL'}")

                # Linear regression for tau*
                # log(w) = -1/tau* * gap => slope = -1/tau*
                slope_fit, _, r_fit, _, _ = linregress(gaps, log_w)
                tau_from_slope = float(-1.0 / slope_fit) if abs(slope_fit) > 1e-6 else None
                tau_reg_str = f"{tau_from_slope:.4f}" if tau_from_slope is not None else "N/A"
                print(f"\nFrom regression: log(w) = slope*gap => tau* = -1/slope")
                print(f"  slope = {slope_fit:.4f}, tau*_regression = {tau_reg_str}")
                print(f"  r = {r_fit:.4f}")
        else:
            corr_r, corr_p = None, None
            tau_med, tau_from_slope = None, None
    else:
        corr_r, corr_p = None, None
        tau_med, tau_from_slope = None, None

    # Summary table
    print(f"\nSummary:")
    print(f"{'gap':>6} {'kappa_j1':>10} {'kappa_j2':>10} {'q_base':>8} "
          f"{'slope_j1':>10} {'slope_j2':>10} {'w_emp':>8} {'w_phi02':>8}")
    for r in records:
        w_s = f"{r['w_empirical']:.3f}" if r['w_empirical'] is not None else "N/A"
        s1_s = f"{r['slope_j1']:.4f}" if r['slope_j1'] is not None else "N/A"
        s2_s = f"{r['slope_j2']:.4f}" if r['slope_j2'] is not None else "N/A"
        print(f"{r['gap_actual']:>6.3f} {r['kappa_j1']:>10.4f} {r['kappa_j2']:>10.4f} "
              f"{r['q_base']:>8.4f} {s1_s:>10} {s2_s:>10} {w_s:>8} {r['w_phi_tau02']:>8.3f}")

    result = {
        "experiment": "phi_jacobian_synthetic",
        "K": K, "d": D, "N_per_class": N_PER_CLASS,
        "kappa_j1_fixed": KAPPA_J1,
        "gap_list": GAP_LIST,
        "delta_list": DELTA_LIST,
        "tau_star_pooled": TAU_STAR_POOLED,
        "records": records,
        "n_valid_w": len(valid),
        "corr_log_w_gap_r": float(corr_r) if corr_r is not None else None,
        "corr_pass": bool(corr_r > 0.85 if corr_r is not None else False),
        "tau_star_median": float(tau_med) if tau_med is not None else None,
        "tau_star_from_regression": float(tau_from_slope) if tau_from_slope is not None else None,
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nResults saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
