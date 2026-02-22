"""
Synthetic Validation of d_eff_formula Causal Surgery
=====================================================
CPU-only. Validates the mathematical prediction before running on real embeddings.

Generates synthetic Gaussians matching CIFAR-100 CE epoch-60 geometry:
  - K=20 classes, d=512, n=50000 train, n=10000 test
  - d_eff_formula = 1.46 (matching CIFAR CE observations)
  - kappa_nearest = 0.84 (matching CIFAR CE observations)

Then applies causal surgery and verifies:
  actual logit(q) ≈ C + A * kappa * sqrt(r * d_eff_base)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

K = 20
D = 512
N_TRAIN = 50000
N_TEST = 10000
A_RENORM = 1.0535
TARGET_KAPPA = 0.84    # target kappa_nearest
TARGET_D_EFF = 1.46    # target d_eff_formula

SURGERY_LEVELS = [0.5, 0.7, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def construct_geometry(K, D, kappa_nearest, d_eff_formula):
    """
    Construct synthetic Gaussian geometry matching target kappa and d_eff.

    Strategy:
      1. Place K class centroids with inter-centroid distance = delta_min
      2. Build within-class covariance with sigma_centroid_dir such that
         d_eff_formula = trW / sigma_centroid_dir^2 = target
      3. sigma_W_global * sqrt(D) = delta_min / kappa_nearest

    Concretely:
      - All classes share the SAME within-class covariance Sigma_W
      - Sigma_W is anisotropic: one "hard" direction (Delta_hat) has variance sigma_cdir^2
        All other directions share remaining variance uniformly.
      - trW = sigma_cdir^2 + (D-1) * sigma_perp^2
      - sigma_W_global = sqrt(trW / D)
      - kappa_nearest = delta_min / (sigma_W_global * sqrt(D)) = delta_min / sqrt(trW)
    """
    # Set sigma_W_global = 1 for simplicity (we'll scale centroids accordingly)
    # Then trW = D * sigma_W_global^2 = D
    trW = float(D)

    # From d_eff_formula = trW / sigma_cdir^2:
    sigma_cdir_sq = trW / d_eff_formula      # sigma_centroid_dir^2
    sigma_cdir = np.sqrt(sigma_cdir_sq)

    # Remaining variance spread over D-1 perpendicular directions
    sigma_perp_sq = (trW - sigma_cdir_sq) / (D - 1)

    # From kappa_nearest = delta_min / sqrt(trW):
    delta_min = kappa_nearest * np.sqrt(trW)

    # Random class centroids: place K points with min pairwise distance = delta_min
    # Use rejection sampling starting from random normals, scale up
    centroids = np.random.randn(K, D) * delta_min * 2  # start with large spread
    # Find min pairwise distance and rescale
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    actual_min = min(dists)
    # Scale so min is exactly delta_min
    centroids = centroids * (delta_min / actual_min)

    # Nearest pair direction
    min_dist_val = float('inf')
    min_i, min_j = 0, 1
    for i in range(K):
        for j in range(i + 1, K):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_dist_val:
                min_dist_val, min_i, min_j = dist, i, j

    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / np.linalg.norm(Delta)

    # Within-class covariance eigenvectors: Delta_hat + D-1 random orthogonal
    # Generate random basis for the D-1 perpendicular directions
    # Use QR decomposition of random matrix for orthogonal complement
    rand_vecs = np.random.randn(D, D)
    rand_vecs[0] = Delta_hat
    Q, _ = np.linalg.qr(rand_vecs.T)
    # Q columns are orthonormal. First column should align with Delta_hat.
    # Ensure Q[:,0] is aligned with Delta_hat (or flip sign)
    if Q[:, 0] @ Delta_hat < 0:
        Q[:, 0] = -Q[:, 0]

    # Build covariance matrix:
    # Sigma_W = sigma_cdir^2 * (Delta_hat @ Delta_hat^T) + sigma_perp^2 * (I - Delta_hat @ Delta_hat^T)
    # We don't need full covariance matrix - just sample from it.
    # Sample: z = sigma_cdir * (xi_0 * Delta_hat) + sigma_perp * (sum_j xi_j * perp_j)
    # where xi_j are i.i.d. N(0,1)

    return {
        'centroids': centroids,
        'Delta_hat': Delta_hat,
        'sigma_cdir': sigma_cdir,
        'sigma_perp': np.sqrt(sigma_perp_sq),
        'Q': Q,  # orthonormal basis (first column = Delta_hat)
        'trW': trW,
        'sigma_W_global': np.sqrt(trW / D),
        'kappa_nearest_target': kappa_nearest,
        'd_eff_formula_target': d_eff_formula,
        'delta_min_target': delta_min,
        'nearest_pair': (min_i, min_j),
    }


def sample_data(geo, N_per_class, seed=0):
    """Sample N_per_class samples per class from constructed geometry."""
    rng = np.random.default_rng(seed)
    K_actual = len(geo['centroids'])
    D = len(geo['Delta_hat'])
    Q = geo['Q']  # (D, D) orthonormal

    all_X = []
    all_y = []
    for c in range(K_actual):
        # Sample from N(mu_c, Sigma_W)
        # z = sigma_cdir * xi_0 * Q[:,0] + sigma_perp * sum_{j=1}^{D-1} xi_j * Q[:,j]
        xi = rng.standard_normal((N_per_class, D))  # (N_per_class, D)
        # Scale first component by sigma_cdir
        xi_scaled = xi.copy()
        xi_scaled[:, 0] *= geo['sigma_cdir']
        # Scale remaining components by sigma_perp
        xi_scaled[:, 1:] *= geo['sigma_perp']
        # Rotate: z = xi_scaled @ Q^T (since Q columns are basis vectors)
        z = xi_scaled @ Q.T  # (N_per_class, D)
        X_c = geo['centroids'][c] + z
        all_X.append(X_c)
        all_y.extend([c] * N_per_class)

    return np.vstack(all_X).astype(np.float32), np.array(all_y)


def compute_geometry_from_data(X, y):
    """Measure actual geometry from data."""
    classes = np.unique(y)
    N = len(X)
    D = X.shape[1]

    centroids = np.stack([X[y == c].mean(0) for c in classes])

    trW = 0.0
    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        trW += float(np.sum((Xc - centroids[c]) ** 2)) / N

    sigma_W_global = np.sqrt(trW / D)

    min_dist = float('inf')
    min_i, min_j = 0, 1
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j

    kappa_nearest = float(min_dist / (sigma_W_global * np.sqrt(D) + 1e-10))
    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)

    sigma_centroid_sq = 0.0
    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        proj = Xc_c @ Delta_hat
        sigma_centroid_sq += (n_c / N) * float(np.mean(proj ** 2))

    d_eff_formula = float(trW / (sigma_centroid_sq + 1e-10))

    return {
        'centroids': centroids,
        'Delta_hat': Delta_hat,
        'trW': trW,
        'sigma_W_global': float(sigma_W_global),
        'sigma_centroid_sq': sigma_centroid_sq,
        'sigma_centroid_dir': float(np.sqrt(sigma_centroid_sq + 1e-10)),
        'd_eff_formula': d_eff_formula,
        'kappa_nearest': kappa_nearest,
    }


def apply_surgery(X, y, geometry, r):
    """Apply covariance surgery: d_eff_new = r * d_eff_base, trW preserved."""
    centroids = geometry['centroids']
    Delta_hat = geometry['Delta_hat']
    trW = geometry['trW']
    sigma_centroid_sq = geometry['sigma_centroid_sq']
    classes = np.unique(y)

    min_r = float(sigma_centroid_sq / (trW + 1e-10)) * 1.001
    if r < min_r:
        r = min_r

    scale_along = 1.0 / float(np.sqrt(r))
    denom = trW - sigma_centroid_sq
    scale_perp = float(np.sqrt(max(0.0, (trW - sigma_centroid_sq / r) / (denom + 1e-10))))

    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        Xc = X[mask]
        z = Xc - centroids[c]
        proj_scalar = z @ Delta_hat
        z_along = proj_scalar[:, None] * Delta_hat[None, :]
        z_perp = z - z_along
        X_new[mask] = centroids[c] + scale_along * z_along + scale_perp * z_perp

    return X_new


def compute_q(X_tr, y_tr, X_te, y_te, K_classes=K):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1)
    knn.fit(X_tr, y_tr)
    acc = float(knn.score(X_te, y_te))
    return (acc - 1.0 / K_classes) / (1.0 - 1.0 / K_classes)


def main():
    print("=" * 70)
    print("Synthetic Surgery Validation")
    print("=" * 70)
    print(f"Target: K={K}, D={D}, kappa={TARGET_KAPPA}, d_eff={TARGET_D_EFF}")
    print(f"A_renorm={A_RENORM} (pre-registered)")
    print()

    # Step 1: Construct synthetic geometry
    print("Step 1: Constructing geometry...")
    geo_design = construct_geometry(K, D, TARGET_KAPPA, TARGET_D_EFF)
    print(f"  sigma_cdir = {geo_design['sigma_cdir']:.4f}")
    print(f"  sigma_perp = {geo_design['sigma_perp']:.4f}")
    print(f"  trW = {geo_design['trW']:.1f}")

    # Step 2: Sample data
    print("\nStep 2: Sampling data...")
    N_train_per_class = N_TRAIN // K
    N_test_per_class = N_TEST // K
    X_tr, y_tr = sample_data(geo_design, N_train_per_class, seed=0)
    X_te, y_te = sample_data(geo_design, N_test_per_class, seed=1)
    print(f"  X_tr shape: {X_tr.shape}, X_te shape: {X_te.shape}")

    # Step 3: Compute actual geometry from data
    print("\nStep 3: Computing geometry from data...")
    geo_actual = compute_geometry_from_data(X_tr, y_tr)
    print(f"  d_eff_formula: {geo_actual['d_eff_formula']:.4f} (target: {TARGET_D_EFF})")
    print(f"  kappa_nearest: {geo_actual['kappa_nearest']:.4f} (target: {TARGET_KAPPA})")
    print(f"  trW: {geo_actual['trW']:.4f} (target: {D:.1f})")
    print(f"  sigma_centroid_dir: {geo_actual['sigma_centroid_dir']:.4f}")
    print(f"  sigma_W_global: {geo_actual['sigma_W_global']:.4f}")
    print(f"  sigma_cdir/sigma_W = {geo_actual['sigma_centroid_dir']/geo_actual['sigma_W_global']:.2f}x")

    # Step 4: Baseline q
    print("\nStep 4: Computing baseline q...")
    q_base = compute_q(X_tr, y_tr, X_te, y_te)
    logit_q_base = float(np.log(q_base / (1 - q_base + 1e-10) + 1e-10))
    kappa_eff_base = geo_actual['kappa_nearest'] * np.sqrt(geo_actual['d_eff_formula'])
    C_fitted = logit_q_base - A_RENORM * kappa_eff_base
    print(f"  q_base = {q_base:.4f}, logit_q_base = {logit_q_base:.4f}")
    print(f"  kappa_eff_base = {kappa_eff_base:.4f}")
    print(f"  C_fitted = {C_fitted:.4f}")

    # Step 5: Surgery across r levels
    print("\nStep 5: Applying surgery...")
    print(f"{'r':>6} {'d_eff_new':>10} {'kappa_new':>10} {'kappa_chg%':>11} "
          f"{'q_new':>8} {'logit_act':>10} {'logit_pred':>11} {'err%':>7}")

    records = []
    for r in SURGERY_LEVELS:
        X_tr_new = apply_surgery(X_tr, y_tr, geo_actual, r)
        X_te_new = apply_surgery(X_te, y_te, geo_actual, r)

        # Measure new geometry
        geo_new = compute_geometry_from_data(X_tr_new, y_tr)

        # Compute actual q
        q_new = compute_q(X_tr_new, y_tr, X_te_new, y_te)
        logit_q_new = float(np.log(q_new / (1 - q_new + 1e-10) + 1e-10))

        # Predicted logit(q)
        logit_pred = C_fitted + A_RENORM * geo_actual['kappa_nearest'] * np.sqrt(r * geo_actual['d_eff_formula'])
        delta_actual = logit_q_new - logit_q_base
        delta_pred = logit_pred - logit_q_base
        err_pct = abs(delta_actual - delta_pred) / (abs(delta_pred) + 1e-6) * 100

        kappa_chg_pct = abs(geo_new['kappa_nearest'] - geo_actual['kappa_nearest']) / (
            geo_actual['kappa_nearest'] + 1e-10) * 100
        trW_chg_pct = abs(geo_new['trW'] - geo_actual['trW']) / (geo_actual['trW'] + 1e-10) * 100

        print(f"{r:>6.2f} {geo_new['d_eff_formula']:>10.3f} {geo_new['kappa_nearest']:>10.4f} "
              f"{kappa_chg_pct:>11.4f} {q_new:>8.4f} {logit_q_new:>10.4f} "
              f"{logit_pred:>11.4f} {err_pct:>7.1f}")

        records.append({
            'r': r, 'd_eff_new': geo_new['d_eff_formula'],
            'kappa_new': geo_new['kappa_nearest'], 'kappa_chg_pct': kappa_chg_pct,
            'trW_chg_pct': trW_chg_pct,
            'q_new': q_new, 'logit_q_new': logit_q_new, 'logit_pred': logit_pred,
            'delta_actual': delta_actual, 'delta_pred': delta_pred, 'err_pct': err_pct,
        })

    # Step 6: Summary statistics
    print("\nSummary:")
    actual_vals = np.array([r['logit_q_new'] for r in records])
    pred_vals = np.array([r['logit_pred'] for r in records])
    r_pearson = float(np.corrcoef(actual_vals, pred_vals)[0, 1])
    mean_calib = float(np.mean([r['err_pct'] for r in records if abs(r['delta_pred']) > 0.01]))
    max_kappa_chg = float(np.max([r['kappa_chg_pct'] for r in records]))
    max_trW_chg = float(np.max([r['trW_chg_pct'] for r in records]))

    ss_res = float(np.sum((actual_vals - pred_vals) ** 2))
    ss_tot = float(np.sum((actual_vals - actual_vals.mean()) ** 2))
    r2_law = 1.0 - ss_res / (ss_tot + 1e-10)

    print(f"  Pearson r(actual, predicted): {r_pearson:.6f}  [PASS if > 0.99]")
    print(f"  R2 of law: {r2_law:.6f}")
    print(f"  Mean calibration error: {mean_calib:.2f}%  [PASS if < 10%]")
    print(f"  Max kappa_nearest change: {max_kappa_chg:.6f}%  [PASS if < 0.1%]")
    print(f"  Max trW change: {max_trW_chg:.6f}%  [expected ~0%]")

    overall_pass = (r_pearson > 0.99) and (mean_calib < 10.0)
    print(f"\nOVERALL (synthetic): {'PASS' if overall_pass else 'FAIL'}")
    if overall_pass:
        print("  The surgery math is correct. Ready for real embeddings.")
    else:
        print("  WARNING: Surgery math failed on synthetic data. Debug before real run.")

    return overall_pass


if __name__ == "__main__":
    main()
