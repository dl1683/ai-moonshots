#!/usr/bin/env python -u
"""
SYNTHETIC w_k INTERVENTION
============================
PURPOSE: Test whether the predicted w_k decay tracks empirical w_k
         using controlled synthetic Gaussian embeddings.

This directly addresses the ICML/NeurIPS reviewer concern:
"The multi-competitor law is empirically supported but w_k decay
mechanism is not shown."

DESIGN:
- Generate K-class Gaussian embeddings with controlled geometry
- Fixed: kappa_nearest (nearest centroid SNR), d_eff (effective dimensionality)
- Varying: spacing of competitor class pairs (controls delta_kappa_k for k=2..K)
- Apply ABCS at m=1,2,...,K-1 and measure slope(m)
- Predicted: w_k prop to slope(m=k)/slope(m=1)
             from theory: w_k = exp(-A_local * (kappa_k - kappa_1) * sqrt(d_eff))
             with A_local = 1.75 (empirical from rank-spectrum session 24)

CONFIGURATIONS:
Three geometric regimes to test:
  1. "Dense": all kappa_k tightly spaced (kappa_k = kappa_1 * (1 + 0.1*k))
     Prediction: w_k all near 1.0, K_eff ~ K-1
  2. "Sparse": widely spaced (kappa_k = kappa_1 * (1 + 0.5*k))
     Prediction: w_k decays quickly, K_eff << K-1
  3. "Geometric decay": kappa_k = kappa_1 * r^(k-1), r=1.3
     Prediction: smooth exponential w_k decay

PRE-REGISTERED ACCEPTANCE (Feb 24, 2026):
  W1: Pearson r(predicted_wk, empirical_wk) > 0.85 in >=2/3 regimes
  W2: dense/sparse ordering preserved (dense K_eff > sparse K_eff)
  W3: A_local consistent across regimes (CV < 0.40)
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_synthetic_wk.json"
LOG_PATH = "results/cti_synthetic_wk_log.txt"

# Synthetic embedding parameters
K = 14         # number of classes (same as DBpedia K=14)
D = 768        # embedding dimension (same as pythia-160m)
N_PER_CLASS = 500  # samples per class (enough for stable KNN)
N_SEEDS = 5    # random seeds for synthetic generation

# Pre-registered law constants
A_LOCAL = 1.75       # empirical from rank-spectrum (session 24)
A_RENORM_K14 = 1.477 / np.sqrt(K - 1)  # global law constant

# Fixed base geometry
KAPPA_1_BASE = 0.30   # kappa of nearest pair (baseline, approx real data)
D_EFF_TARGET = 25.0   # target effective dimensionality

# M_BUNDLE tested (same as cross-arch ABCS)
M_BUNDLE = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13]
R_LEVELS = [0.3, 0.5, 2.0, 5.0, 10.0]

os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')

def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def make_synthetic_embeddings(K, D, n_per_class, kappas, d_eff_target, seed):
    """
    Generate K-class Gaussian embeddings with controlled geometry.

    kappas: array of K-1 kappa values for each pair (sorted, kappas[0] = nearest)
    d_eff_target: target effective dimensionality d_eff = tr(W) / sigma_centroid_dir^2

    Construction:
    1. Set centroids mu_0, ..., mu_{K-1} with controlled inter-centroid distances
    2. Set within-class variance to achieve desired kappa and d_eff
    3. Use anisotropic covariance: high variance in non-centroid directions
    """
    rng = np.random.default_rng(seed)

    # sigma_W_global controls kappa: kappa_ij = delta_ij / (sigma_W * sqrt(D))
    sigma_W_global = 1.0  # baseline

    # Place centroids: mu_0 at origin, mu_1 along e_0 at distance delta_1, etc.
    # For K classes with K-1 competitive pairs, we need K-1 linearly independent directions
    deltas = np.array([kappa * sigma_W_global * np.sqrt(D) for kappa in kappas])

    # Build centroids: each class in a random direction (orthogonalized)
    centroid_directions = []
    for k in range(K):
        v = rng.standard_normal(D)
        for prev in centroid_directions:
            v -= (v @ prev) * prev
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            v = rng.standard_normal(D)
        v /= (np.linalg.norm(v) + 1e-10)
        centroid_directions.append(v)

    # Place mu_i at distance deltas[i-1]/2 from origin along centroid_directions[i]
    # (so that delta between class 0 and class i is ~ deltas[i-1])
    # Actually: place class 0 at origin, class i at distance deltas[i-1] from class 0
    mus = np.zeros((K, D))
    for k in range(1, K):
        delta_k = deltas[min(k-1, len(deltas)-1)]
        mus[k] = delta_k * centroid_directions[k]

    # Within-class covariance: anisotropic to achieve d_eff_target
    # d_eff = tr(Sigma_W) / sigma_centroid_dir^2
    # sigma_centroid_dir = var in nearest-centroid-pair direction
    # tr(Sigma_W) = D * sigma_W_global^2 = D (since sigma_W_global=1)
    # So sigma_centroid_dir^2 = D / d_eff_target
    tr_W = D * sigma_W_global**2
    sigma_centroid_dir_sq = tr_W / d_eff_target

    # Build anisotropic covariance:
    # High variance in most directions: sigma_high^2
    # Low variance in centroid directions: sigma_low^2 = sigma_centroid_dir_sq
    # (K-1) centroid dirs have variance sigma_low^2
    # (D - K + 1) other dirs have variance sigma_high^2
    # tr(W) = (K-1) * sigma_low^2 + (D-K+1) * sigma_high^2 = D * sigma_W_global^2
    n_low = K - 1
    n_high = D - n_low
    sigma_low_sq = sigma_centroid_dir_sq
    sigma_high_sq = (D * sigma_W_global**2 - n_low * sigma_low_sq) / n_high
    sigma_high_sq = max(sigma_high_sq, 1e-6)

    sigma_low = float(np.sqrt(sigma_low_sq))
    sigma_high = float(np.sqrt(sigma_high_sq))

    # Generate samples
    # Project each sample into (centroid dirs, rest)
    U_centroid = np.stack(centroid_directions, axis=1)  # (D, K)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))   # random rotation for null space
    # Build basis: centroid directions first, then null space complement
    null_dirs = []
    for i in range(Q.shape[1]):
        v = Q[:, i].copy()
        for c in centroid_directions:
            v -= (v @ c) * c
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-8:
            null_dirs.append(v / norm_v)
        if len(null_dirs) >= n_high:
            break

    X_list, y_list = [], []
    for c in range(K):
        z_centroid = rng.normal(0, sigma_low, (n_per_class, n_low))
        z_null = rng.normal(0, sigma_high, (n_per_class, min(n_high, len(null_dirs))))

        U_c = np.array(centroid_directions).T  # (D, K)
        U_n = np.array(null_dirs).T if null_dirs else np.zeros((D, 0))

        X_c = mus[c] + z_centroid @ U_c[:, :n_low].T
        if z_null.shape[1] > 0:
            X_c = X_c + z_null @ U_n[:, :z_null.shape[1]].T
        X_list.append(X_c)
        y_list.append(np.full(n_per_class, c))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def compute_geometry_full(X_tr, y_tr, classes):
    K = len(classes)
    N, d = len(X_tr), X_tr.shape[1]
    mu = np.stack([X_tr[y_tr == c].mean(0) for c in classes])
    grand_mean = X_tr.mean(0)
    trW = 0.0
    for i, c in enumerate(classes):
        Xc_c = X_tr[y_tr == c] - mu[i]
        trW += float(np.sum(Xc_c ** 2)) / N
    sigma_W_global = float(np.sqrt(trW / d))
    pair_info = []
    for i in range(K):
        for j in range(i + 1, K):
            Delta = mu[i] - mu[j]
            delta_ij = float(np.linalg.norm(Delta))
            Delta_hat = Delta / (delta_ij + 1e-10)
            kappa_ij = float(delta_ij / (sigma_W_global * np.sqrt(d) + 1e-10))
            sigma_cdir_sq = 0.0
            for k, c in enumerate(classes):
                Xc_c = X_tr[y_tr == c] - mu[k]
                n_c = len(Xc_c)
                proj = Xc_c @ Delta_hat
                sigma_cdir_sq += (n_c / N) * float(np.mean(proj ** 2))
            d_eff_ij = float(trW / (sigma_cdir_sq + 1e-10))
            pair_info.append({
                'i': int(i), 'j': int(j),
                'kappa': float(kappa_ij), 'd_eff': float(d_eff_ij),
                'kappa_eff': float(kappa_ij * np.sqrt(d_eff_ij)),
                'Delta_hat': Delta_hat,
            })
    pair_info.sort(key=lambda x: x['kappa'])
    nearest = pair_info[0]
    return {
        'mu': mu, 'grand_mean': grand_mean, 'trW': trW,
        'sigma_W_global': sigma_W_global, 'K': K, 'd': d,
        'pair_info': pair_info, 'nearest': nearest,
        'kappa': nearest['kappa'], 'd_eff': nearest['d_eff'],
        'kappa_eff': nearest['kappa_eff'], 'Delta_hat': nearest['Delta_hat'],
    }


def build_between_class_basis(mu, grand_mean, Delta_hat_first):
    K = mu.shape[0]
    basis = [Delta_hat_first.copy()]
    mu_c = mu - grand_mean
    for k in range(K):
        v = mu_c[k].copy()
        for b in basis:
            v -= (v @ b) * b
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-8:
            basis.append(v / norm_v)
        if len(basis) == K - 1:
            break
    return np.stack(basis, axis=1)


def build_competitive_bundle(pair_info, m):
    d = pair_info[0]['Delta_hat'].shape[0]
    basis = []
    for pair in pair_info[:m]:
        v = pair['Delta_hat'].copy()
        for b in basis:
            v -= (v @ b) * b
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-8:
            basis.append(v / norm_v)
    return np.stack(basis, axis=1) if basis else None


def compute_subspace_trace(X_tr, y_tr, mu, classes, U):
    N = len(X_tr)
    tr = 0.0
    for k_vec in U.T:
        s = 0.0
        for i, c in enumerate(classes):
            Xc_c = X_tr[y_tr == c] - mu[i]
            n_c = len(Xc_c)
            proj = Xc_c @ k_vec
            s += (n_c / N) * float(np.mean(proj ** 2))
        tr += s
    return tr


def apply_bundle_surgery(X_tr, X_te, y_tr, y_te, geo, U_bundle, U_B, r, tr_W_null):
    mu = geo['mu']
    classes = np.unique(y_tr)
    tr_bundle = compute_subspace_trace(X_tr, y_tr, mu, classes, U_bundle)
    tr_null_new = tr_W_null + tr_bundle * (1.0 - 1.0 / r)
    scale_null = float(np.sqrt(tr_null_new / tr_W_null)) if tr_W_null > 1e-12 and tr_null_new > 0 else 1.0
    scale_bundle = 1.0 / float(np.sqrt(r))

    def transform(X, y):
        X_new = X.copy()
        for i, c in enumerate(classes):
            mask = (y == c)
            z = X[mask] - mu[i]
            z_between = z @ U_B @ U_B.T
            z_bundle = z @ U_bundle @ U_bundle.T
            z_other = z_between - z_bundle
            z_null = z - z_between
            X_new[mask] = mu[i] + scale_bundle * z_bundle + z_other + scale_null * z_null
        return X_new
    return transform(X_tr, y_tr), transform(X_te, y_te)


def eval_q(X_tr, y_tr, X_te, y_te, K):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = float(np.clip((acc - 1.0/K) / (1.0 - 1.0/K), 1e-6, 1-1e-6))
    return acc, q, float(sp.logit(q))


def run_abcs_synthetic(X, y, classes, regime_name):
    """Run ABCS on synthetic data, return slopes and empirical w_k."""
    n_total = len(X)
    n_tr = int(0.7 * n_total)
    idx = np.random.permutation(n_total)
    X_tr, y_tr = X[idx[:n_tr]], y[idx[:n_tr]]
    X_te, y_te = X[idx[n_tr:]], y[idx[n_tr:]]

    K = len(classes)
    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)

    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])
    tr_bundle_all = compute_subspace_trace(X_tr, y_tr, geo['mu'], classes, U_B)
    tr_W_null = geo['trW'] - tr_bundle_all

    # Pre-registered predictions based on global law
    A_keff = A_RENORM_K14 / float(np.sqrt(geo['d_eff']))
    C_keff = logit_base - A_keff * geo['kappa_eff']
    delta_preds = {}
    for r in R_LEVELS:
        kappa_eff_r = geo['kappa_eff'] * float(np.sqrt(r))
        delta_preds[r] = float(C_keff + A_keff * kappa_eff_r) - logit_base

    slopes_by_m = {}
    for m in M_BUNDLE:
        if m > K - 1:
            continue
        U_comp = build_competitive_bundle(geo['pair_info'], m)
        if U_comp is None:
            continue
        results = []
        for r in R_LEVELS:
            X_tr_c, X_te_c = apply_bundle_surgery(
                X_tr, X_te, y_tr, y_te, geo, U_comp, U_B, r, tr_W_null)
            _, _, logit_r = eval_q(X_tr_c, y_tr, X_te_c, y_te, K)
            results.append({'r': float(r), 'delta_logit': float(logit_r - logit_base)})

        nontrivial = [x for x in results if abs(x['r'] - 1.0) > 0.01]
        if len(nontrivial) < 3:
            continue
        d_obs = [x['delta_logit'] for x in nontrivial]
        d_pred = [delta_preds[x['r']] for x in nontrivial]
        if np.std(d_pred) < 1e-8:
            continue
        slope = float(np.polyfit(d_pred, d_obs, 1)[0])
        r_val = float(pearsonr(d_obs, d_pred)[0])
        slopes_by_m[m] = {'slope': slope, 'r': r_val}

    if not slopes_by_m:
        return None

    # Empirical w_k = slope(m=k) / slope(m=1)
    s1 = slopes_by_m.get(1, {}).get('slope', None)
    empirical_wk = {}
    if s1 and abs(s1) > 1e-6:
        for m in sorted(slopes_by_m.keys()):
            empirical_wk[m] = float(slopes_by_m[m]['slope'] / s1)

    # Predicted w_k from theory: exp(-A_local * (kappa_m - kappa_1) * sqrt(d_eff))
    kappas_sorted = [p['kappa'] for p in geo['pair_info'][:max(M_BUNDLE)+1]]
    kappa_1 = kappas_sorted[0]
    d_eff_1 = geo['d_eff']
    predicted_wk = {}
    for m in sorted(slopes_by_m.keys()):
        if m <= len(kappas_sorted):
            kappa_m = kappas_sorted[m-1]
            delta_kappa = kappa_m - kappa_1
            predicted_wk[m] = float(np.exp(-A_LOCAL * delta_kappa * np.sqrt(d_eff_1)))

    # A1 criterion
    all_slopes = [(m, slopes_by_m[m]['slope']) for m in sorted(slopes_by_m.keys())]
    ms = [x[0] for x in all_slopes]
    ss = [x[1] for x in all_slopes]
    r_scale = float(pearsonr(ms, ss)[0]) if len(ms) >= 3 else 0.0

    # Pearson r(predicted_wk, empirical_wk) for W1 criterion
    common_m = sorted(set(predicted_wk.keys()) & set(empirical_wk.keys()))
    if len(common_m) >= 3:
        pred_vals = [predicted_wk[m] for m in common_m]
        emp_vals = [empirical_wk[m] for m in common_m]
        r_wk = float(pearsonr(pred_vals, emp_vals)[0])
    else:
        r_wk = 0.0

    # K_eff estimated
    max_slope = max(s for _, s in all_slopes) if all_slopes else 1.0
    k_eff_est = max((m for m, s in all_slopes if s >= 0.9 * max_slope), default=1)

    # Fit A_local from data
    log_ratios = []
    delta_kappa_sqrt_deff = []
    for m in common_m:
        if m > 1 and empirical_wk[m] > 1e-4:
            log_ratios.append(-np.log(empirical_wk[m]))
            kappa_m = kappas_sorted[m-1] if m-1 < len(kappas_sorted) else kappas_sorted[-1]
            delta_kappa_sqrt_deff.append((kappa_m - kappa_1) * np.sqrt(d_eff_1))
    if len(log_ratios) >= 3 and np.std(delta_kappa_sqrt_deff) > 1e-8:
        A_local_fit = float(np.polyfit(delta_kappa_sqrt_deff, log_ratios, 1)[0])
    else:
        A_local_fit = float('nan')

    return {
        'regime': regime_name,
        'acc_base': float(acc_base), 'q_base': float(q_base),
        'kappa_1': float(kappa_1), 'd_eff': float(d_eff_1),
        'r_scale': r_scale, 'k_eff_est': k_eff_est,
        'r_wk': r_wk, 'pass_W1': bool(r_wk > 0.85),
        'empirical_wk': empirical_wk,
        'predicted_wk': predicted_wk,
        'A_local_fit': A_local_fit,
        'slopes_by_m': {str(k): v for k, v in slopes_by_m.items()},
    }


def build_kappa_regime(K, kappa_1, regime):
    """Build kappa array for K-1 competitor pairs."""
    if regime == 'dense':
        # Tightly spaced: kappa_k = kappa_1 * (1 + 0.15*(k-1))
        kappas = [kappa_1 * (1 + 0.15 * (k)) for k in range(K - 1)]
    elif regime == 'sparse':
        # Widely spaced: kappa_k = kappa_1 * (1 + 0.8*(k-1))
        kappas = [kappa_1 * (1 + 0.8 * (k)) for k in range(K - 1)]
    elif regime == 'geometric':
        # Geometric decay: kappa_k = kappa_1 * 1.4^k
        kappas = [kappa_1 * (1.4 ** k) for k in range(K - 1)]
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return kappas


def main():
    log("=" * 70)
    log("SYNTHETIC w_k INTERVENTION")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log("=" * 70)
    log("PRE-REGISTERED:")
    log("  W1: Pearson r(predicted_wk, empirical_wk) > 0.85 in >=2/3 regimes")
    log("  W2: dense K_eff > sparse K_eff")
    log("  W3: A_local_fit CV < 0.40 across regimes")
    log(f"Parameters: K={K}, D={D}, N_per_class={N_PER_CLASS}, N_seeds={N_SEEDS}")

    all_regime_results = {}
    REGIMES = ['dense', 'sparse', 'geometric']

    for regime in REGIMES:
        log(f"\n{'='*60}")
        log(f"REGIME: {regime.upper()}")
        log(f"{'='*60}")
        kappas = build_kappa_regime(K, KAPPA_1_BASE, regime)
        log(f"kappas[0:5] = {[f'{k:.3f}' for k in kappas[:5]]}")

        regime_results = []
        for seed in range(N_SEEDS):
            log(f"  seed={seed}...", )
            X, y = make_synthetic_embeddings(K, D, N_PER_CLASS, kappas, D_EFF_TARGET, seed)
            classes = np.unique(y)
            np.random.seed(seed + 100)
            res = run_abcs_synthetic(X, y, classes, regime)
            if res:
                regime_results.append(res)
                log(f"    K_eff={res['k_eff_est']}, r_wk={res['r_wk']:.3f} "
                    f"({'PASS' if res['pass_W1'] else 'FAIL'}), "
                    f"A_local_fit={res['A_local_fit']:.3f}")
            else:
                log(f"    SKIP (no valid slopes)")

        if regime_results:
            n_W1 = sum(1 for r in regime_results if r['pass_W1'])
            mean_k_eff = float(np.mean([r['k_eff_est'] for r in regime_results]))
            a_local_fits = [r['A_local_fit'] for r in regime_results if not np.isnan(r['A_local_fit'])]
            mean_a_local = float(np.mean(a_local_fits)) if a_local_fits else float('nan')
            std_a_local = float(np.std(a_local_fits)) if a_local_fits else float('nan')
            log(f"  SUMMARY: W1_pass {n_W1}/{len(regime_results)}, "
                f"K_eff_mean={mean_k_eff:.1f}, "
                f"A_local={mean_a_local:.3f}+/-{std_a_local:.3f}")
            all_regime_results[regime] = {
                'seed_results': regime_results,
                'n_W1_pass': int(n_W1),
                'k_eff_mean': float(mean_k_eff),
                'a_local_mean': float(mean_a_local),
                'a_local_std': float(std_a_local),
            }
        else:
            log(f"  SKIP (no results)")

    # ==================== ANALYSIS ====================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # W1: >=2/3 regimes pass W1 (r_wk > 0.85 in majority of seeds)
    regimes_passing_W1 = []
    for regime, rd in all_regime_results.items():
        if rd['n_W1_pass'] >= N_SEEDS // 2 + 1:  # majority pass
            regimes_passing_W1.append(regime)
    pass_W1 = len(regimes_passing_W1) >= 2
    log(f"W1 (r_wk>0.85 majority in >=2/3 regimes): {'PASS' if pass_W1 else 'FAIL'}")
    log(f"  Passing regimes: {regimes_passing_W1}")

    # W2: dense K_eff > sparse K_eff
    dense_keff = all_regime_results.get('dense', {}).get('k_eff_mean', 0)
    sparse_keff = all_regime_results.get('sparse', {}).get('k_eff_mean', 0)
    pass_W2 = dense_keff > sparse_keff
    log(f"W2 (dense K_eff > sparse K_eff): {'PASS' if pass_W2 else 'FAIL'} "
        f"({dense_keff:.1f} vs {sparse_keff:.1f})")

    # W3: A_local CV < 0.40
    a_locals = [rd['a_local_mean'] for rd in all_regime_results.values()
                if not np.isnan(rd['a_local_mean'])]
    if len(a_locals) >= 2:
        cv_a = float(np.std(a_locals) / (np.mean(a_locals) + 1e-10))
        pass_W3 = cv_a < 0.40
        log(f"W3 (A_local CV<0.40): {'PASS' if pass_W3 else 'FAIL'} "
            f"(A_locals={[f'{a:.3f}' for a in a_locals]}, CV={cv_a:.3f})")
    else:
        pass_W3 = False
        cv_a = float('nan')
        log("W3: INSUFFICIENT DATA")

    overall = pass_W1 and pass_W2 and pass_W3
    verdict = (f"W1={'PASS' if pass_W1 else 'FAIL'}, "
               f"W2={'PASS' if pass_W2 else 'FAIL'}, "
               f"W3={'PASS' if pass_W3 else 'FAIL'}. "
               f"OVERALL: {'PASS' if overall else 'PARTIAL/FAIL'}")
    log(f"\nVERDICT: {verdict}")

    output = {
        'experiment': 'synthetic_wk',
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'K': K, 'D': D, 'N_per_class': N_PER_CLASS, 'N_seeds': N_SEEDS,
            'kappa_1_base': KAPPA_1_BASE, 'd_eff_target': D_EFF_TARGET,
            'A_LOCAL': A_LOCAL, 'A_RENORM_K14': float(A_RENORM_K14),
        },
        'regime_results': all_regime_results,
        'analysis': {
            'pass_W1': bool(pass_W1), 'pass_W2': bool(pass_W2), 'pass_W3': bool(pass_W3),
            'overall': bool(overall), 'a_local_cv': float(cv_a) if not np.isnan(cv_a) else None,
            'dense_keff': float(dense_keff), 'sparse_keff': float(sparse_keff),
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
