"""
2-Layer CTI Model: Synthetic Validation
Tests: logit(q_i) = C + A * kappa_i * sqrt(K_eff_i) + C

Approach: generate synthetic Gaussian class distributions with controlled
(kappa, K_eff) pairs. Measure q via Gumbel race simulation and nearest-centroid
classifier. Test 2-layer model fit.

Design:
  - K classes, d dimensions, Gaussian within-class distributions
  - Control kappa: scale centroid separation
  - Control K_eff: control eigenspectrum of Sigma_W in centroid subspace
    * spike (K_eff ~ 1): concentrate all within-class variance in nearest centroid direction
    * flat (K_eff ~ K-1): uniform variance across all centroid directions
    * intermediate: partial concentration
  - For each (kappa, K_eff) combination, measure q and logit(q)
  - Test: R2 of 2-layer fit vs original 1-layer fit (kappa * sqrt(d_eff))
"""

import numpy as np
from scipy.special import logit as logit_fn, expit
from scipy import stats
import json

RESULTS_FILE = "results/cti_2layer_synthetic.json"

# Config
K = 20          # classes
D = 512         # embedding dimension
N_PER_CLASS = 500   # samples per class for kNN test
N_MC = 10000    # Monte Carlo samples for Gumbel race simulation
SEED_BASE = 42

# Kappa range (normalize by sqrt(d) already absorbed)
KAPPA_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

# K_eff control: eigenvalue concentration levels
# 0.0 = full spike (all variance in first direction, K_eff ~ 1)
# 1.0 = fully flat (uniform eigenvalues, K_eff = K-1)
KEFF_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # interpolation between spike and flat

# Pre-registered: A = sqrt(4/pi) from Gumbel race derivation
A_PREREGISTERED = np.sqrt(4.0 / np.pi)  # ~1.128


def make_synthetic_data(kappa, keff_level, K=K, d=D, n_per_class=N_PER_CLASS, seed=0):
    """
    Generate K Gaussian classes with controlled kappa and K_eff.

    kappa = min centroid pair SNR = min_dist / (sigma_W * sqrt(d))
    keff_level = 0 -> K_eff ~ 1 (spike), 1 -> K_eff ~ K-1 (flat)

    Returns: X (n_total, d), y (n_total,), geo (centroids, Sigma_W, kappa, K_eff)
    """
    rng = np.random.default_rng(seed)
    K_minus1 = K - 1

    # --- Construct Sigma_W with controlled K_eff ---
    # Within-class covariance eigenspectrum in centroid subspace:
    # spike: lambda_1 >> lambda_2 = ... = lambda_{K-1}
    # flat:  lambda_1 = ... = lambda_{K-1}
    sigma_W_sq = 1.0  # total variance per direction (sigma_W^2 = tr(Sigma_W)/d)

    # Total trace of Sigma_W = d * sigma_W_sq = 512
    tr_Sigma_W = d * sigma_W_sq

    # Centroid subspace eigenvalues (K-1 of them)
    # keff_level=0 (spike): all tr_sub in first eigen, rest = 0
    # keff_level=1 (flat):  tr_sub / (K-1) in each eigen
    # tr_sub = total variance in centroid subspace = tr_Sigma_W * f_sub
    # We fix f_sub = 0.3 (30% of variance in centroid subspace) to match CIFAR observations
    f_sub = 0.3
    tr_sub = f_sub * tr_Sigma_W

    # Eigenvalues within centroid subspace
    spike_eigs = np.zeros(K_minus1)
    spike_eigs[0] = tr_sub
    flat_eigs = np.ones(K_minus1) * tr_sub / K_minus1

    sub_eigs = (1 - keff_level) * spike_eigs + keff_level * flat_eigs
    sub_eigs = np.maximum(sub_eigs, 0)  # clip negatives

    # Compute K_eff from sub_eigs
    tr_V = np.sum(sub_eigs)
    tr_V2 = np.sum(sub_eigs**2)
    K_eff = (tr_V**2) / (tr_V2 + 1e-12) if tr_V2 > 0 else 1.0

    # Remaining variance in orthogonal subspace (d - K+1 directions)
    d_orth = d - K_minus1
    tr_orth = tr_Sigma_W - tr_sub
    orth_var = tr_orth / d_orth if d_orth > 0 else 0.0

    # Build Sigma_W as block diagonal in centroid subspace + orthogonal
    # We use a random centroid subspace and then build Sigma_W

    # --- Construct centroids with given kappa ---
    # Generate K centroids with controlled minimum pairwise distance
    # Method: ETF-like structure (equal angle tight frame) for uniform competition
    # For K > d, use random placement; for K <= d, ETF

    # Use simplex vertices for ETF structure (K-1 dimensional)
    if K_minus1 <= d:
        # Construct K simplex vertices in R^{K-1} subspace
        # Embedded in R^d
        centroid_basis = rng.standard_normal((d, K_minus1))
        centroid_basis, _ = np.linalg.qr(centroid_basis)  # d x K-1 orthonormal

        # Simplex vertices in R^{K-1}
        simplex = np.zeros((K, K_minus1))
        for k in range(K):
            for j in range(K_minus1):
                if j < k:
                    simplex[k, j] = 1.0 / np.sqrt(j * (j+1) + 1e-10)
                elif j == k:
                    simplex[k, j] = -np.sqrt(j / (j+1) + 1e-10)
                # else 0

        # Scale simplex to get desired kappa
        # min_dist(simplex) depends on structure; normalize first
        dists = []
        for i in range(K):
            for j in range(i+1, K):
                dists.append(np.linalg.norm(simplex[i] - simplex[j]))
        min_dist_simplex = min(dists) if dists else 1.0

        # sigma_W = sqrt(tr_Sigma_W / d) = sqrt(sigma_W_sq) = 1.0
        sigma_W_global = np.sqrt(sigma_W_sq)

        # Desired min_dist = kappa * sigma_W_global * sqrt(d)
        target_min_dist = kappa * sigma_W_global * np.sqrt(d)
        scale = target_min_dist / (min_dist_simplex + 1e-10)

        centroids_lowD = simplex * scale  # K x (K-1)
        centroids = centroids_lowD @ centroid_basis.T  # K x d
    else:
        # K > d: use random centroids
        centroids = rng.standard_normal((K, d))
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        dists = []
        for i in range(K):
            for j in range(i+1, K):
                dists.append(np.linalg.norm(centroids[i] - centroids[j]))
        min_dist = min(dists) if dists else 1.0
        sigma_W_global = np.sqrt(sigma_W_sq)
        target_min_dist = kappa * sigma_W_global * np.sqrt(d)
        centroids = centroids * target_min_dist / (min_dist + 1e-10)

    # --- Build Sigma_W using sub_eigs ---
    # V_i = centroid_basis[:, :K-1].T @ Sigma_W @ centroid_basis[:, :K-1] should have eigenvalues sub_eigs
    # So Sigma_W component in centroid_basis = centroid_basis @ diag(sub_eigs) @ centroid_basis.T

    if K_minus1 <= d:
        Sigma_W_sub = centroid_basis @ np.diag(sub_eigs) @ centroid_basis.T
    else:
        Sigma_W_sub = np.eye(d) * (tr_sub / d)

    # Orthogonal component
    Sigma_W_orth = np.eye(d) * orth_var
    # Subtract centroid_basis contribution from identity
    if K_minus1 <= d:
        Sigma_W_orth = Sigma_W_orth - centroid_basis @ (centroid_basis.T * orth_var)

    Sigma_W = Sigma_W_sub + Sigma_W_orth

    # Verify: tr(Sigma_W) should be close to tr_Sigma_W
    tr_check = float(np.trace(Sigma_W))

    # --- Generate samples ---
    # For efficiency, use only within-class covariance (cholesky-based sampling)
    try:
        L = np.linalg.cholesky(Sigma_W + 1e-6 * np.eye(d))
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma_W)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    all_X = []
    all_y = []
    for k in range(K):
        z = rng.standard_normal((n_per_class, d))
        X_k = centroids[k] + z @ L.T
        all_X.append(X_k)
        all_y.extend([k] * n_per_class)

    X = np.vstack(all_X)
    y = np.array(all_y)

    # Verify kappa
    dists_check = []
    for i in range(K):
        for j in range(i+1, K):
            dists_check.append(np.linalg.norm(centroids[i] - centroids[j]))
    min_dist_check = min(dists_check)
    kappa_check = min_dist_check / (np.sqrt(sigma_W_sq) * np.sqrt(d))

    return X, y, {
        "kappa_target": float(kappa),
        "kappa_actual": float(kappa_check),
        "K_eff_target": float(K_eff),
        "K_eff_actual": float(K_eff),
        "keff_level": float(keff_level),
        "tr_V": float(tr_V),
        "f_sub": float(f_sub),
        "sub_eigs": sub_eigs.tolist(),
        "tr_check": float(tr_check),
        "tr_Sigma_W": float(tr_Sigma_W),
    }


def compute_q_nearest_centroid(X_test, y_test, centroids, K=K):
    """Q = normalized nearest centroid accuracy."""
    # Distances from test points to all centroids
    dists = np.array([[np.linalg.norm(x - centroids[k]) for k in range(K)] for x in X_test])
    preds = np.argmin(dists, axis=1)
    acc = np.mean(preds == y_test)
    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    return float(q)


def run_synthetic_2layer():
    """Run 2x2 factorial test on synthetic Gaussian data."""
    print("=" * 70)
    print("2-Layer CTI Synthetic Validation")
    print("=" * 70)
    print(f"K={K}, d={D}, n_per_class={N_PER_CLASS}")
    print(f"Kappa range: {KAPPA_VALUES}")
    print(f"K_eff levels: {KEFF_LEVELS}")
    print(f"A_preregistered = {A_PREREGISTERED:.4f}")
    print()

    records = []

    for kappa in KAPPA_VALUES:
        for keff_level in KEFF_LEVELS:
            for seed in range(3):
                X, y, geo = make_synthetic_data(kappa, keff_level, seed=SEED_BASE + seed)

                # Split into train/test
                n = len(X)
                n_train = n // 2
                perm = np.random.RandomState(seed).permutation(n)
                X_tr, y_tr = X[perm[:n_train]], y[perm[:n_train]]
                X_te, y_te = X[perm[n_train:]], y[perm[n_train:]]

                # Compute centroids from training data
                centroids_est = np.array([X_tr[y_tr == k].mean(0) for k in range(K)])

                q = compute_q_nearest_centroid(X_te, y_te, centroids_est)
                logit_q = float(logit_fn(np.clip(q, 1e-6, 1-1e-6)))

                kappa_act = geo["kappa_actual"]
                K_eff_act = geo["K_eff_actual"]

                # 2-layer prediction: logit(q) = A * kappa * sqrt(K_eff) + C
                logit_pred_2layer = A_PREREGISTERED * kappa_act * np.sqrt(K_eff_act)

                # 1-layer prediction: logit(q) = A * kappa * sqrt(d_eff) + C
                # d_eff = tr(Sigma_W) / sigma_centroid_dir^2
                # For our synthetic data: sigma_W_sq = 1, d_eff_formula = D / 1 = D
                # (or use sub_eigs[0] as sigma_centroid_dir^2)
                if geo["sub_eigs"][0] > 0:
                    d_eff_1layer = geo["tr_Sigma_W"] / geo["sub_eigs"][0]
                else:
                    d_eff_1layer = D
                logit_pred_1layer = A_PREREGISTERED * kappa_act * np.sqrt(d_eff_1layer)

                records.append({
                    "kappa_target": float(kappa),
                    "kappa_actual": float(kappa_act),
                    "K_eff_target": float(K_eff_act),
                    "keff_level": float(keff_level),
                    "seed": seed,
                    "q": float(q),
                    "logit_q": float(logit_q),
                    "logit_pred_2layer": float(logit_pred_2layer),
                    "logit_pred_1layer": float(logit_pred_1layer),
                    "d_eff_1layer": float(d_eff_1layer),
                    "f_sub": float(geo["f_sub"]),
                })

                print(f"  kappa={kappa:.1f}, keff_lv={keff_level:.1f}, "
                      f"K_eff={K_eff_act:.1f}, q={q:.3f}, logit={logit_q:.3f}, "
                      f"pred_2L={logit_pred_2layer:.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Arrays
    logit_obs = np.array([r["logit_q"] for r in records])
    logit_2layer = np.array([r["logit_pred_2layer"] for r in records])
    logit_1layer = np.array([r["logit_pred_1layer"] for r in records])

    # Fit intercept (C) for each model
    # Since we have logit_obs = A * kappa * sqrt(K_eff) + C, fit C as mean residual
    C_2layer = np.mean(logit_obs - logit_2layer)
    C_1layer = np.mean(logit_obs - logit_1layer)

    residuals_2layer = logit_obs - (logit_2layer + C_2layer)
    residuals_1layer = logit_obs - (logit_1layer + C_1layer)

    # R2
    ss_tot = np.sum((logit_obs - logit_obs.mean())**2)
    r2_2layer = 1.0 - np.sum(residuals_2layer**2) / ss_tot
    r2_1layer = 1.0 - np.sum(residuals_1layer**2) / ss_tot

    # Pearson
    pearson_2layer, _ = stats.pearsonr(logit_2layer, logit_obs)
    pearson_1layer, _ = stats.pearsonr(logit_1layer, logit_obs)

    print(f"\n2-layer model (logit = A*kappa*sqrt(K_eff) + C):")
    print(f"  R2 = {r2_2layer:.4f}")
    print(f"  Pearson r = {pearson_2layer:.4f}")
    print(f"  Fitted C = {C_2layer:.4f}")

    print(f"\n1-layer model (logit = A*kappa*sqrt(d_eff) + C):")
    print(f"  R2 = {r2_1layer:.4f}")
    print(f"  Pearson r = {pearson_1layer:.4f}")
    print(f"  Fitted C = {C_1layer:.4f}")

    # K_eff variation at fixed kappa
    print("\n--- K_eff variation at kappa=1.0 ---")
    kappa1_records = [r for r in records if abs(r["kappa_target"] - 1.0) < 0.01]
    if kappa1_records:
        keff_vals = [r["K_eff_target"] for r in kappa1_records]
        logit_vals = [r["logit_q"] for r in kappa1_records]
        corr_keff, _ = stats.pearsonr(keff_vals, logit_vals)
        print(f"  Pearson r(K_eff, logit_q) at kappa=1.0: {corr_keff:.4f}")

    # kappa variation at fixed K_eff level
    print("\n--- kappa variation at keff_level=0.5 ---")
    keff5_records = [r for r in records if abs(r["keff_level"] - 0.5) < 0.01]
    if keff5_records:
        kappa_vals = [r["kappa_actual"] for r in keff5_records]
        logit_vals = [r["logit_q"] for r in keff5_records]
        if len(kappa_vals) > 2:
            corr_kappa, _ = stats.pearsonr(kappa_vals, logit_vals)
            print(f"  Pearson r(kappa, logit_q) at keff_level=0.5: {corr_kappa:.4f}")

    # Save results
    summary = {
        "r2_2layer": float(r2_2layer),
        "r2_1layer": float(r2_1layer),
        "pearson_2layer": float(pearson_2layer),
        "pearson_1layer": float(pearson_1layer),
        "C_2layer": float(C_2layer),
        "C_1layer": float(C_1layer),
        "n_records": len(records),
        "PASS": bool(r2_2layer > r2_1layer and pearson_2layer > 0.9),
    }

    print(f"\nSUMMARY:")
    print(f"  2-layer R2 = {r2_2layer:.4f} vs 1-layer R2 = {r2_1layer:.4f}")
    print(f"  {'2-layer WINS' if r2_2layer > r2_1layer else '1-layer WINS'}")

    output = {"summary": summary, "records": records}
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    return summary


if __name__ == "__main__":
    run_synthetic_2layer()
