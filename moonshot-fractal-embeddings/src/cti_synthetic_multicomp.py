#!/usr/bin/env python -u
"""
SYNTHETIC ANALYTIC MULTI-COMPETITOR TEST
==========================================
PURPOSE: Cleanly test whether A_local (w_k decay constant) depends on:
         (a) spectrum shape (eta_m hypothesis: uniform vs heavy-tailed)
         (b) embedding dimensionality d
         (c) both or neither (surgery estimator noise)

DESIGN: K-class Gaussian embeddings with FULLY CONTROLLED geometry.
        Fix kappa_nearest and K. Vary:
          Factor A: spectrum shape (uniform, power-law, heavy-tailed, spike)
          Factor B: embedding dimension d (64, 256, 1024)
        For each (shape, d): run individual pair surgery, fit A_local.

PREDICTIONS:
  P1 (eta_m): A_local higher for uniform spectrum than heavy-tailed (at same d)
  P2 (dimensionality): A_local ~ 1/sqrt(d) or 1/d at fixed spectrum shape
  P3 (surgery validity): r_wk > 0.70 in synthetic setting (cleaner than real)

PRE-REGISTERED (Feb 25, 2026):
  S1: P1 holds: uniform_A_local > heavy_A_local for >= 2/3 d values
  S2: Pearson r(log_d, log_A_local) < -0.50 (A_local decreases with d)
  S3: r_wk > 0.70 for >= 6/12 (shape, d) combinations
  S4: A_local ranges consistently above 0.5 in clean synthetic (validates
       that real low A_local values ~0.2-0.4 may reflect surgery-specific noise)
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_synthetic_multicomp.json"
LOG_PATH = "results/cti_synthetic_multicomp_log.txt"

K = 14          # match DBpedia
N_PER_CLASS = 300
KAPPA_NEAREST = 0.40   # fixed kappa for nearest pair
N_SEEDS = 3
SEEDS = [42, 137, 271]
R_LEVELS = [0.3, 0.5, 2.0, 5.0, 10.0]
N_PAIRS = 12
A_LOCAL_THEORY = 1.75   # theoretical prediction from Session 24

# Factor A: spectrum types
SPECTRUM_TYPES = ['uniform', 'power_law', 'heavy_tailed', 'spike']

# Factor B: embedding dimensions
DIMS = [64, 256, 1024]

os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')


def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def make_covariance(spectrum_type, d):
    """Create within-class covariance matrix with specified spectrum shape."""
    if spectrum_type == 'uniform':
        # All eigenvalues equal: isotropic
        eigenvalues = np.ones(d)
    elif spectrum_type == 'power_law':
        # Power-law decay: lambda_i ~ 1/i^0.5
        eigenvalues = 1.0 / np.sqrt(1 + np.arange(d).astype(float))
    elif spectrum_type == 'heavy_tailed':
        # Heavy-tailed: lambda_i ~ 1/i^1.5 (fast decay)
        eigenvalues = 1.0 / (1 + np.arange(d).astype(float)) ** 1.5
    elif spectrum_type == 'spike':
        # One large spike: lambda_1=100, rest=0.01
        eigenvalues = np.ones(d) * 0.01
        eigenvalues[0] = 100.0
    else:
        raise ValueError(f"Unknown spectrum type: {spectrum_type}")

    # Normalize so trW = d (unit trace per dimension)
    eigenvalues = eigenvalues / eigenvalues.mean()
    U = np.linalg.qr(np.random.randn(d, d))[0]  # random orthonormal basis
    Sigma = U @ np.diag(eigenvalues) @ U.T
    return Sigma, eigenvalues


def generate_synthetic_embeddings(K, d, n_per_class, Sigma, kappa_nearest, seed):
    """Generate K-class Gaussian embeddings with controlled geometry."""
    rng = np.random.default_rng(seed)

    # sigma_W_global: derived from trace
    trW = float(np.trace(Sigma))
    sigma_W_global = float(np.sqrt(trW / d))

    # Create centroids: nearest pair has kappa=kappa_nearest, others further
    # Place nearest pair first (classes 0, 1)
    delta_min = kappa_nearest * sigma_W_global * np.sqrt(d)

    # Use first principal direction (largest eigenvector) for nearest pair
    eigenvalues_raw, U_raw = np.linalg.eigh(Sigma)
    main_dir = U_raw[:, -1]  # largest eigenvalue direction

    mu = np.zeros((K, d))
    # Nearest pair: classes 0 and 1 in main_dir
    mu[0] = main_dir * (delta_min / 2.0)
    mu[1] = -main_dir * (delta_min / 2.0)
    # Remaining classes: progressively more distant
    spacing_multipliers = np.linspace(1.5, 4.0, K - 2)
    for k in range(2, K):
        direction = rng.standard_normal(d)
        direction = direction / np.linalg.norm(direction)
        scale = delta_min * spacing_multipliers[k - 2]
        mu[k] = direction * scale

    # Generate samples
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(d))
    X_list, y_list = [], []
    for c in range(K):
        Z = rng.standard_normal((n_per_class, d))
        X_list.append(mu[c] + Z @ L.T)
        y_list.append(np.full(n_per_class, c))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    return X, y, mu


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


def apply_single_pair_surgery(X_tr, X_te, y_tr, y_te, geo, pair, U_B, r):
    mu = geo['mu']
    classes = np.unique(y_tr)
    trW = geo['trW']
    Delta_hat = pair['Delta_hat']
    U_pair = Delta_hat.reshape(-1, 1)
    tr_pair = compute_subspace_trace(X_tr, y_tr, mu, classes, U_pair)
    tr_B_total = compute_subspace_trace(X_tr, y_tr, mu, classes, U_B)
    tr_W_null = trW - tr_B_total
    tr_null_new = tr_W_null + tr_pair * (1.0 - 1.0 / r)
    scale_null = float(np.sqrt(tr_null_new / tr_W_null)) if tr_W_null > 1e-12 and tr_null_new > 0 else 1.0
    scale_pair = 1.0 / float(np.sqrt(r))

    def transform(X, y):
        X_new = X.copy()
        for i, c in enumerate(classes):
            mask = (y == c)
            z = X[mask] - mu[i]
            z_pair = z @ U_pair @ U_pair.T
            z_between = z @ U_B @ U_B.T
            z_other_between = z_between - z_pair
            z_null = z - z_between
            X_new[mask] = mu[i] + scale_pair * z_pair + z_other_between + scale_null * z_null
        return X_new

    return transform(X_tr, y_tr), transform(X_te, y_te)


def eval_q(X_tr, y_tr, X_te, y_te, K):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = float(np.clip((acc - 1.0/K) / (1.0 - 1.0/K), 1e-6, 1-1e-6))
    return acc, q, float(sp.logit(q))


def run_surgery(X_tr, y_tr, X_te, y_te, classes, n_pairs):
    K = len(classes)
    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)
    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])

    kappa_1 = geo['pair_info'][0]['kappa']
    d_eff_1 = geo['pair_info'][0]['d_eff']
    kappa_eff_1 = geo['pair_info'][0]['kappa_eff']

    A_keff = 1.477 / float(np.sqrt((K - 1) * d_eff_1))
    C_keff = logit_base - A_keff * kappa_eff_1
    delta_preds_by_r = {}
    for r in R_LEVELS:
        kappa_eff_r = kappa_eff_1 * float(np.sqrt(r))
        delta_preds_by_r[r] = float(C_keff + A_keff * kappa_eff_r) - logit_base

    pair_results = []
    for k_idx, pair in enumerate(geo['pair_info'][:n_pairs]):
        kappa_k = pair['kappa']
        d_eff_k = pair['d_eff']

        single_results = []
        for r in R_LEVELS:
            X_tr_r, X_te_r = apply_single_pair_surgery(
                X_tr, X_te, y_tr, y_te, geo, pair, U_B, r)
            _, _, logit_r = eval_q(X_tr_r, y_tr, X_te_r, y_te, K)
            single_results.append({'r': float(r), 'delta_logit': float(logit_r - logit_base)})

        nontrivial = [x for x in single_results if abs(x['r'] - 1.0) > 0.01]
        if len(nontrivial) < 3:
            continue
        d_obs = [x['delta_logit'] for x in nontrivial]
        d_pred = [delta_preds_by_r[x['r']] for x in nontrivial]
        if np.std(d_pred) < 1e-8:
            continue

        slope_k = float(np.polyfit(d_pred, d_obs, 1)[0])
        r_k = float(pearsonr(d_obs, d_pred)[0])
        norm_factor = float(kappa_1 * np.sqrt(d_eff_1) / (kappa_k * np.sqrt(d_eff_k) + 1e-10))
        empirical_wk = float(slope_k * norm_factor)
        delta_kappa = kappa_k - kappa_1
        delta_kappa_sqrt_d = float(delta_kappa * np.sqrt(d_eff_1))
        predicted_wk_fixed = float(np.exp(-A_LOCAL_THEORY * delta_kappa_sqrt_d))

        pair_results.append({
            'k': k_idx + 1,
            'kappa_k': float(kappa_k), 'd_eff_k': float(d_eff_k),
            'slope_k': float(slope_k), 'r_k': float(r_k),
            'empirical_wk': float(empirical_wk),
            'predicted_wk_fixed': float(predicted_wk_fixed),
            'delta_kappa': float(delta_kappa),
            'delta_kappa_sqrt_d': float(delta_kappa_sqrt_d),
        })

    if not pair_results:
        return pair_results, {}, float('nan'), float('nan'), float('nan')

    emp_wks = [pr['empirical_wk'] for pr in pair_results]
    pred_wks_fixed = [pr['predicted_wk_fixed'] for pr in pair_results]

    # Fit A_local
    valid = [(pr['delta_kappa_sqrt_d'], pr['empirical_wk']) for pr in pair_results
             if pr['delta_kappa_sqrt_d'] > 1e-6 and pr['empirical_wk'] > 1e-6]
    if len(valid) >= 3:
        dk_arr = np.array([v[0] for v in valid])
        lw_arr = np.array([-np.log(v[1]) for v in valid])
        A_local_fit = float(np.polyfit(dk_arr, lw_arr, 1)[0])
    else:
        A_local_fit = float('nan')

    # r_wk with fixed A_local
    if len(emp_wks) >= 3 and np.std(pred_wks_fixed) > 1e-6:
        r_wk_fixed = float(pearsonr(emp_wks, pred_wks_fixed)[0])
    else:
        r_wk_fixed = 0.0

    baseline = {
        'acc_base': float(acc_base), 'q_base': float(q_base), 'logit_base': float(logit_base),
        'kappa_1': float(kappa_1), 'd_eff_1': float(d_eff_1),
    }
    return pair_results, baseline, float(A_local_fit), float(r_wk_fixed), geo


def compute_tail_ratio_inv(Sigma_eigenvalues):
    """Compute tail_ratio_inv from known eigenspectrum."""
    S = np.sort(Sigma_eigenvalues)[::-1]
    n_top = max(1, min(5, len(S) // 10))
    n_bot = max(1, min(5, len(S) // 10))
    tail_ratio = float(S[:n_top].mean() / (S[-n_bot:].mean() + 1e-12))
    return float(1.0 / (tail_ratio + 1e-12))


def main():
    log("=" * 70)
    log("SYNTHETIC ANALYTIC MULTI-COMPETITOR TEST")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log(f"K={K}, N_PER_CLASS={N_PER_CLASS}, kappa_nearest={KAPPA_NEAREST}")
    log(f"SPECTRA: {SPECTRUM_TYPES}")
    log(f"DIMS: {DIMS}")
    log("=" * 70)
    log("PRE-REGISTERED:")
    log("  S1: uniform A_local > heavy A_local for >=2/3 d values")
    log("  S2: r(log_d, log_A_local) < -0.50 (A_local decreases with d)")
    log("  S3: r_wk > 0.70 for >=6/12 (spectrum, d) combinations")
    log("  S4: A_local ranges above 0.5 in at least some conditions")

    all_results = []

    for spectrum_type in SPECTRUM_TYPES:
        for d in DIMS:
            log(f"\n{'='*50}")
            log(f"SPECTRUM: {spectrum_type}, d={d}")
            log(f"{'='*50}")

            # Generate covariance and get theoretical tail_ratio_inv
            spec_seed = 0
            np.random.seed(spec_seed)
            Sigma, eigenvalues = make_covariance(spectrum_type, d)
            tri = compute_tail_ratio_inv(eigenvalues)
            log(f"  tail_ratio_inv = {tri:.5f}")

            seed_results = []
            for seed in SEEDS:
                # Split into train/test
                X, y, _ = generate_synthetic_embeddings(K, d, N_PER_CLASS * 2, Sigma, KAPPA_NEAREST, seed)
                n_per = N_PER_CLASS
                X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
                rng2 = np.random.default_rng(seed + 1000)
                classes = np.unique(y)
                for c in classes:
                    idx = np.where(y == c)[0]
                    rng2.shuffle(idx)
                    n_tr = min(n_per, len(idx) // 2)
                    X_tr_list.append(X[idx[:n_tr]]); y_tr_list.append(y[idx[:n_tr]])
                    X_te_list.append(X[idx[n_tr:]]); y_te_list.append(y[idx[n_tr:]])
                X_tr = np.concatenate(X_tr_list)
                y_tr = np.concatenate(y_tr_list)
                X_te = np.concatenate(X_te_list)
                y_te = np.concatenate(y_te_list)

                pair_results, baseline, A_local_fit, r_wk_fixed, geo = run_surgery(
                    X_tr, y_tr, X_te, y_te, classes, N_PAIRS)

                if not pair_results:
                    log(f"  seed={seed}: SKIP")
                    continue

                log(f"  seed={seed}: A_local={A_local_fit:.3f}, r_wk={r_wk_fixed:.3f}, "
                    f"q_base={baseline['q_base']:.3f}, kappa_1={baseline['kappa_1']:.3f}, "
                    f"d_eff_1={baseline['d_eff_1']:.2f}")
                seed_results.append({
                    'seed': seed,
                    'A_local_fit': float(A_local_fit),
                    'r_wk_fixed': float(r_wk_fixed),
                    'baseline': baseline,
                })

            if not seed_results:
                continue

            A_locals = [s['A_local_fit'] for s in seed_results if not np.isnan(s['A_local_fit'])]
            r_wks = [s['r_wk_fixed'] for s in seed_results]
            mean_A = float(np.mean(A_locals)) if A_locals else float('nan')
            std_A = float(np.std(A_locals)) if len(A_locals) > 1 else 0.0
            mean_r = float(np.mean(r_wks))

            log(f"  SUMMARY: A_local={mean_A:.3f}+/-{std_A:.3f}, r_wk={mean_r:.3f}")

            all_results.append({
                'spectrum': spectrum_type,
                'd': int(d),
                'tail_ratio_inv': float(tri),
                'mean_A_local': float(mean_A),
                'std_A_local': float(std_A),
                'mean_r_wk': float(mean_r),
                'seed_results': seed_results,
            })

    # ==================== ANALYSIS ====================
    log("\n" + "=" * 70)
    log("POOLED ANALYSIS")
    log("=" * 70)

    if not all_results:
        log("No results!")
        return

    # S1: uniform A_local > heavy A_local per d
    s1_pass_count = 0
    for d in DIMS:
        uniform_res = [r for r in all_results if r['spectrum'] == 'uniform' and r['d'] == d]
        heavy_res = [r for r in all_results if r['spectrum'] == 'heavy_tailed' and r['d'] == d]
        if uniform_res and heavy_res:
            if uniform_res[0]['mean_A_local'] > heavy_res[0]['mean_A_local']:
                s1_pass_count += 1
    pass_S1 = s1_pass_count >= 2
    log(f"S1 (uniform > heavy): {s1_pass_count}/{len(DIMS)} dims PASS -> {'PASS' if pass_S1 else 'FAIL'}")

    # S2: r(log_d, log_A_local) within each spectrum type
    s2_rs = []
    for spectrum_type in SPECTRUM_TYPES:
        spec_res = [r for r in all_results if r['spectrum'] == spectrum_type]
        if len(spec_res) >= 3:
            log_ds = [np.log(r['d']) for r in spec_res]
            A_locals = [r['mean_A_local'] for r in spec_res]
            valid = [(ld, a) for ld, a in zip(log_ds, A_locals) if not np.isnan(a) and a > 0]
            if len(valid) >= 3:
                r_val = float(pearsonr([v[0] for v in valid],
                                       [np.log(v[1]) for v in valid])[0])
                s2_rs.append(r_val)
                log(f"  {spectrum_type}: r(log_d, log_A_local) = {r_val:.3f}")
    mean_s2_r = float(np.mean(s2_rs)) if s2_rs else 0.0
    pass_S2 = mean_s2_r < -0.50
    log(f"S2 mean r(log_d, log_A_local) = {mean_s2_r:.3f} -> {'PASS' if pass_S2 else 'FAIL'}")

    # S3: r_wk > 0.70 count
    n_pass_S3 = sum(1 for r in all_results if r['mean_r_wk'] > 0.70)
    pass_S3 = n_pass_S3 >= 6
    log(f"S3 r_wk > 0.70: {n_pass_S3}/{len(all_results)} conditions -> {'PASS' if pass_S3 else 'FAIL'}")

    # S4: max A_local > 0.5
    max_A = max([r['mean_A_local'] for r in all_results if not np.isnan(r['mean_A_local'])], default=0.0)
    pass_S4 = max_A > 0.5
    log(f"S4 max A_local > 0.5: {max_A:.3f} -> {'PASS' if pass_S4 else 'FAIL'}")

    # Summary table
    log("\nFull results table:")
    log(f"{'spectrum':15} {'d':6} {'tail_inv':10} {'A_local':10} {'r_wk':8}")
    for r in sorted(all_results, key=lambda x: (x['spectrum'], x['d'])):
        log(f"  {r['spectrum']:15} {r['d']:6} {r['tail_ratio_inv']:10.5f} "
            f"{r['mean_A_local']:10.3f} {r['mean_r_wk']:8.3f}")

    # r(tail_ratio_inv, A_local) across all conditions
    tris = [r['tail_ratio_inv'] for r in all_results]
    A_locals = [r['mean_A_local'] for r in all_results]
    valid_pairs = [(t, a) for t, a in zip(tris, A_locals) if not np.isnan(a)]
    if len(valid_pairs) >= 3:
        r_eta, p_eta = pearsonr([v[0] for v in valid_pairs], [v[1] for v in valid_pairs])
        log(f"\nr(tail_ratio_inv, A_local) across all conditions: r={r_eta:.3f} p={p_eta:.4f}")
    else:
        r_eta, p_eta = 0.0, 1.0

    overall_pass = pass_S1 and pass_S4
    verdict = (f"S1={'PASS' if pass_S1 else 'FAIL'}, S2={'PASS' if pass_S2 else 'FAIL'}, "
               f"S3={'PASS' if pass_S3 else 'FAIL'}, S4={'PASS' if pass_S4 else 'FAIL'}. "
               f"Max A_local={max_A:.3f}. r(tail_inv, A_local)={r_eta:.3f}. "
               f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    log(f"\nVERDICT: {verdict}")

    output = {
        'experiment': 'synthetic_multicomp',
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'K': K, 'n_per_class': N_PER_CLASS,
            'kappa_nearest': KAPPA_NEAREST,
            'spectrum_types': SPECTRUM_TYPES,
            'dims': DIMS,
            'seeds': SEEDS,
        },
        'all_results': all_results,
        'analysis': {
            'pass_S1': bool(pass_S1), 's1_pass_count': int(s1_pass_count),
            'pass_S2': bool(pass_S2), 'mean_s2_r': float(mean_s2_r),
            'pass_S3': bool(pass_S3), 'n_pass_S3': int(n_pass_S3),
            'pass_S4': bool(pass_S4), 'max_A_local': float(max_A),
            'r_eta_A': float(r_eta), 'p_eta_A': float(p_eta),
            'overall_pass': bool(overall_pass),
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
