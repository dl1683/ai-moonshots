"""
Measure d_eff_cls (classification effective dimension) from neural net embeddings.

Theory says: alpha = sqrt(8/pi) * sqrt(d_eff_cls)
Observed: alpha = 1.549
Implies: d_eff_cls = (1.549/sqrt(8/pi))^2 = 0.942 ~= 1.0

Key question: CAN WE MEASURE d_eff_cls DIRECTLY from neural net embeddings
and confirm it is ~= 1?

Method:
d_eff_cls = d_eff_discriminative = (tr(S_B))^2 / tr(S_B^2)
where S_B is the BETWEEN-class scatter matrix (captures discriminative structure)

This is distinct from d_eff_total = tr(S)^2/tr(S^2) (total covariance).

At Neural Collapse:
- S_B -> ETF structure -> d_eff(S_B) -> K-1 (the minimum)
- But from the alpha formula, d_eff_cls ~= 1, not K-1.

This suggests alpha is related to the MINIMUM margin (1 effective competitive class),
not the full K-1 discriminative structure.

Alternative: d_eff_cls = effective dimension of the NEAREST PAIR only:
The pair-specific effective dimension for the (c*, c*') nearest pair.

Let's compute several variants and see which matches theory best.
"""

import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

ALPHA_EMPIRICAL = 1.549
SQRT_8_PI = np.sqrt(8 / np.pi)
D_EFF_CLS_PREDICTED = (ALPHA_EMPIRICAL / SQRT_8_PI) ** 2  # = 0.942


def compute_scatter_matrices(X, y):
    """Compute within-class (S_W) and between-class (S_B) scatter matrices."""
    classes = np.unique(y)
    K = len(classes)
    n, d = X.shape
    grand_mean = X.mean(0)

    # Between-class scatter
    S_B = np.zeros((d, d))
    n_k = {}
    mu_k = {}
    for c in classes:
        mask = y == c
        n_c = mask.sum()
        mu_c = X[mask].mean(0)
        mu_k[c] = mu_c
        n_k[c] = n_c
        diff = (mu_c - grand_mean).reshape(-1, 1)
        S_B += n_c * (diff @ diff.T)
    S_B /= n

    # Within-class scatter
    S_W = np.zeros((d, d))
    for c in classes:
        mask = y == c
        X_c = X[mask] - mu_k[c]
        S_W += X_c.T @ X_c
    S_W /= n

    return S_B, S_W, mu_k, n_k


def effective_rank(M):
    """Effective rank = (tr M)^2 / tr(M^2) — measures dimensional concentration."""
    tr_M = np.trace(M)
    tr_M2 = np.trace(M @ M)
    if tr_M2 < 1e-10:
        return 0.0
    return float(tr_M ** 2 / tr_M2)


def effective_rank_from_eigenvalues(lambdas):
    """Effective rank from eigenvalues (stable for large matrices)."""
    lambdas = lambdas[lambdas > 0]
    if len(lambdas) == 0:
        return 0.0
    tr = lambdas.sum()
    tr2 = (lambdas ** 2).sum()
    return float(tr ** 2 / tr2) if tr2 > 0 else 0.0


def svd_effective_rank(X):
    """Effective rank from SVD of data matrix."""
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum()) if len(s2) > 0 else 0.0


def nearest_pair_d_eff(X, y, nearest_class_pair):
    """d_eff for the nearest class pair only.
    For classes c1, c2: project data onto the axis mu_c1 - mu_c2.
    d_eff_pair = effective dim of projection."""
    c1, c2 = nearest_class_pair
    mask1 = y == c1
    mask2 = y == c2

    X1 = X[mask1]
    X2 = X[mask2]

    if len(X1) == 0 or len(X2) == 0:
        return 0.0

    mu1 = X1.mean(0)
    mu2 = X2.mean(0)
    direction = mu1 - mu2
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-10:
        return 0.0
    direction = direction / d_norm

    # Project all data from both classes onto this axis
    all_X = np.concatenate([X1, X2], axis=0)
    all_y = np.array([0] * len(X1) + [1] * len(X2))

    # Within-class variance along the direction
    proj = all_X @ direction
    var_c1 = np.var(proj[all_y == 0])
    var_c2 = np.var(proj[all_y == 1])
    var_within_axis = (var_c1 + var_c2) / 2

    # Total within-class variance
    S_W1 = np.cov(X1.T)
    S_W2 = np.cov(X2.T)
    sigma_W = 0.5 * (S_W1 + S_W2) if X1.shape[0] > 1 and X2.shape[0] > 1 else np.eye(X.shape[1])

    try:
        total_variance = np.trace(sigma_W)
        # d_eff_cls = total_variance / var_within_axis
        # This measures: how many "axis" dimensions worth of variance falls along the decision boundary
        d_eff_pair = float(total_variance / (var_within_axis + 1e-10))
    except Exception:
        d_eff_pair = float('nan')

    return d_eff_pair


def compute_d_eff_cls(X, y):
    """
    Compute multiple variants of d_eff_cls.

    Returns dict with different estimates.
    """
    classes = np.unique(y)
    K = len(classes)
    d = X.shape[1]

    # Compute scatter matrices
    S_B, S_W, mu_k, n_k = compute_scatter_matrices(X, y)

    # 1. d_eff_cls from BETWEEN-class scatter
    eigvals_B = np.linalg.eigvalsh(S_B)
    d_eff_B = effective_rank_from_eigenvalues(eigvals_B)

    # 2. d_eff_cls from WITHIN-class scatter
    eigvals_W = np.linalg.eigvalsh(S_W)
    d_eff_W = effective_rank_from_eigenvalues(eigvals_W)

    # 3. Ratio: d_eff of S_B / K (should be 1 at NC since S_B has K-1 nonzero eigvals)
    d_eff_B_normalized = d_eff_B / (K - 1)

    # 4. Find nearest class pair
    centroids = np.array([mu_k[c] for c in classes])
    min_dist = np.inf
    nearest_pair = (classes[0], classes[1])
    for i in range(K):
        for j in range(i + 1, K):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_dist:
                min_dist = dist
                nearest_pair = (classes[i], classes[j])

    # 5. d_eff for nearest pair
    d_eff_pair = nearest_pair_d_eff(X, y, nearest_pair)

    # 6. Fisher d_eff: (tr S_B / tr S_W)^2 / (tr S_B^2 / tr S_W^2)
    # This measures how concentrated discriminative info is
    lam_B = np.sort(eigvals_B)[::-1][:K-1]  # Top K-1 eigenvalues
    lam_W = np.sort(eigvals_W)[::-1][:d]
    fisher_ratio = lam_B.sum() / lam_W.sum() if lam_W.sum() > 0 else 0
    d_eff_fisher = effective_rank_from_eigenvalues(lam_B / (lam_W[:K-1] + 1e-10))

    # 7. Theory-implied d_eff_cls from empirical alpha_LOAO
    alpha_implied = ALPHA_EMPIRICAL
    d_eff_from_alpha = (alpha_implied / SQRT_8_PI) ** 2

    return {
        'd_eff_B': float(d_eff_B),
        'd_eff_W': float(d_eff_W),
        'd_eff_B_normalized': float(d_eff_B_normalized),
        'd_eff_pair': float(d_eff_pair),
        'd_eff_fisher': float(d_eff_fisher),
        'd_eff_implied_by_alpha': float(d_eff_from_alpha),
        'K': int(K),
        'd': int(d),
        'nearest_pair': [int(nearest_pair[0]), int(nearest_pair[1])],
    }


def main():
    print("d_eff_cls Measurement from Neural Net Embeddings")
    print("=" * 60)
    print(f"Theory: alpha = sqrt(8/pi) * sqrt(d_eff_cls)")
    print(f"sqrt(8/pi) = {SQRT_8_PI:.4f}")
    print(f"Empirical alpha = {ALPHA_EMPIRICAL}")
    print(f"Implied d_eff_cls = {D_EFF_CLS_PREDICTED:.4f}")
    print()

    results = []

    # Load embedding caches
    cache_files = sorted([f for f in os.listdir('results/') if f.startswith('kappa_near_cache_')])
    print(f"Found {len(cache_files)} kappa cache files")

    for fname in cache_files[:20]:  # Sample 20 to keep fast
        fpath = f'results/{fname}'
        try:
            data = __import__('json').load(open(fpath))
            if not isinstance(data, list) or len(data) == 0:
                continue

            # Use last layer (most developed representations)
            entry = data[-1]
            emb_file = entry.get('emb_file')
            model = entry.get('model', fname)
            dataset = entry.get('dataset', '')
            layer = entry.get('layer', 0)

            # Parse model/dataset from filename
            parts = fname.replace('kappa_near_cache_', '').replace('.json', '').split('_')
            if len(parts) >= 2:
                dataset_name = parts[0]
                model_name = '_'.join(parts[1:])
            else:
                dataset_name = fname
                model_name = fname

            # Load embeddings if available
            npz_file = emb_file if emb_file and os.path.exists(emb_file) else None
            if npz_file is None:
                # Try to find matching npz
                for fn in os.listdir('results/'):
                    if fn.endswith('.npz') and model_name in fn and dataset_name in fn:
                        npz_file = f'results/{fn}'
                        break

            if npz_file is None:
                continue

            npz = np.load(npz_file)
            X = npz['X']
            y = npz['y'].astype(int)

            if len(np.unique(y)) < 2:
                continue

            d_eff_stats = compute_d_eff_cls(X, y)

            print(f"  {fname[:50]:50s}: d_eff_B={d_eff_stats['d_eff_B']:.3f}, "
                  f"d_eff_B_norm={d_eff_stats['d_eff_B_normalized']:.3f}, "
                  f"K={d_eff_stats['K']}")

            results.append({
                'file': fname,
                **d_eff_stats,
            })

        except Exception as e:
            print(f"  SKIP {fname}: {e}")

    # Also analyze do-intervention embeddings
    print("\nDo-intervention embeddings:")
    for fname in sorted(os.listdir('results/')):
        if not fname.startswith('do_int_embs_') or not fname.endswith('.npz'):
            continue
        fpath = f'results/{fname}'
        try:
            npz = np.load(fpath)
            X = npz['X']
            y = npz['y'].astype(int)
            K = len(np.unique(y))
            d_eff_stats = compute_d_eff_cls(X, y)
            print(f"  {fname[:50]:50s}: d_eff_B={d_eff_stats['d_eff_B']:.3f}, "
                  f"d_eff_B_norm={d_eff_stats['d_eff_B_normalized']:.3f}, "
                  f"K={K}")
            results.append({'file': fname, **d_eff_stats})
        except Exception as e:
            print(f"  SKIP {fname}: {e}")

    if not results:
        print("No results computed!")
        return

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    d_eff_Bs = [r['d_eff_B'] for r in results if not np.isnan(r['d_eff_B'])]
    d_eff_norms = [r['d_eff_B_normalized'] for r in results if not np.isnan(r['d_eff_B_normalized'])]

    if d_eff_Bs:
        print(f"d_eff_B (between-class):            {np.mean(d_eff_Bs):.3f} +/- {np.std(d_eff_Bs):.3f}")
        print(f"d_eff_B_normalized (/ K-1):         {np.mean(d_eff_norms):.3f} +/- {np.std(d_eff_norms):.3f}")
        print(f"  (expected ~1.0 at NC, observed implied by alpha: {D_EFF_CLS_PREDICTED:.3f})")
        print()
        nc_condition = np.mean(d_eff_norms) < 1.5
        print(f"NC condition (d_eff_B_norm < 1.5): {'PASS' if nc_condition else 'FAIL'}")
        print(f"Theory d_eff_cls: {D_EFF_CLS_PREDICTED:.3f} -- match with d_eff_B_norm: "
              f"{'GOOD' if abs(np.mean(d_eff_norms) - D_EFF_CLS_PREDICTED) < 0.5 else 'MISMATCH'}")

    output = {
        'theory': {
            'alpha_empirical': ALPHA_EMPIRICAL,
            'sqrt_8_pi': float(SQRT_8_PI),
            'd_eff_cls_implied': float(D_EFF_CLS_PREDICTED),
        },
        'results': results,
        'summary': {
            'mean_d_eff_B': float(np.mean(d_eff_Bs)) if d_eff_Bs else None,
            'std_d_eff_B': float(np.std(d_eff_Bs)) if d_eff_Bs else None,
            'mean_d_eff_B_normalized': float(np.mean(d_eff_norms)) if d_eff_norms else None,
        },
    }

    with open('results/cti_deff_cls_measurement.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nSaved to results/cti_deff_cls_measurement.json")


if __name__ == '__main__':
    main()
