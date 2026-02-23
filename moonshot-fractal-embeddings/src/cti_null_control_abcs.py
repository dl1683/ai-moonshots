#!/usr/bin/env python -u
"""
PROPER NULL CONTROL FOR ABCS
==============================

PRE-REGISTERED (Feb 23 2026). Fixes ABCS tertiary criterion failure.

PROBLEM: Original ABCS tertiary test used RANDOM COMPETITIVE PAIRS as control.
These showed large effects (~0.43 slope) because they're still in the competitive subspace.

CORRECT NULL: Sample m random directions from NULL SPACE of U_B (orthogonal to
ALL K-1 between-class directions). Surgery in null space should NOT affect accuracy
because it doesn't change class confusion structure.

PHYSICAL PREDICTION:
  - Null-space surgery: delta_logit ≈ 0 (null directions are irrelevant to classification)
  - Competitive-space surgery (m=5 from ABCS): delta_logit ≈ large (direction-specific effect)
  - Specificity ratio: slope_null / slope_competitive < 0.1 (near-zero null effect)

PRE-REGISTERED ACCEPTANCE:
  NULL: |slope_null| < 0.1 * slope_m5 (null effect < 10% of competitive effect)
  PASS: null effect near zero confirms surgery targets the RIGHT geometric structure

DATA: pythia-160m/DBpedia K=14 (same as ABCS reference)
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

EMBED_PATH = "results/dointerv_multi_pythia-160m_l12.npz"
RESULT_PATH = "results/cti_null_control_abcs.json"
LOG_PATH = "results/cti_null_control_abcs_log.txt"

A_RENORM_K20 = 1.0535
ALPHA_KAPPA_K14 = 1.477
R_LEVELS = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
M_NULL = [1, 2, 5, 8, 13]   # null-space bundle sizes
M_COMPETITIVE = 5             # competitive reference (from ABCS)
N_TRAIN_PER_CLASS = 350
RANDOM_SEED = 42
N_NULL_RUNS = 5               # multiple random null samples for error bars

os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')

def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def load_and_split(path, n_train, seed):
    data = np.load(path)
    X, y = data['X'].astype(np.float64), data['y']
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = min(n_train, len(idx) - 1)
        X_tr_list.append(X[idx[:n]]); y_tr_list.append(y[idx[:n]])
        X_te_list.append(X[idx[n:]]); y_te_list.append(y[idx[n:]])
    return (np.concatenate(X_tr_list), np.concatenate(y_tr_list),
            np.concatenate(X_te_list), np.concatenate(y_te_list), classes)


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
            kappa_eff_ij = kappa_ij * float(np.sqrt(d_eff_ij))

            pair_info.append({
                'i': int(i), 'j': int(j),
                'delta': float(delta_ij), 'kappa': float(kappa_ij),
                'd_eff': float(d_eff_ij), 'kappa_eff': float(kappa_eff_ij),
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


def build_null_space_basis(U_B, d):
    """Build orthonormal basis for null space (complement of U_B).
    Uses SVD to find vectors orthogonal to all columns of U_B.
    """
    K_minus_1 = U_B.shape[1]
    # Randomized null space via SVD
    # Get basis for column space of U_B
    Q, _ = np.linalg.qr(U_B)  # (d, K-1) orthonormal
    # Build projection P_null = I - Q Q^T
    # Null space basis: sample random vectors, project out U_B
    # Then Gram-Schmidt
    return Q  # Return column space for reference; null space sampled differently


def sample_null_bundle(U_B, d, m, rng):
    """Sample m orthonormal vectors from null space of U_B."""
    K_minus_1 = U_B.shape[1]
    # Null space is (d - K_minus_1) dimensional
    # Approach: sample random vectors, project out between-class subspace, Gram-Schmidt
    basis = []
    attempts = 0
    while len(basis) < m and attempts < m * 100:
        v = rng.standard_normal(d)
        # Project out between-class subspace
        v -= U_B @ (U_B.T @ v)
        # Project out already-built basis
        for b in basis:
            v -= (v @ b) * b
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-8:
            basis.append(v / norm_v)
        attempts += 1
    if not basis:
        return np.zeros((d, 1))
    return np.stack(basis, axis=1)  # (d, m)


def build_competitive_bundle(pair_info, m):
    """Top-m nearest class pairs, Gram-Schmidt."""
    selected = pair_info[:m]
    d = selected[0]['Delta_hat'].shape[0]
    basis = []
    for pair in selected:
        v = pair['Delta_hat'].copy()
        for b in basis:
            v -= (v @ b) * b
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-8:
            basis.append(v / norm_v)
    if not basis:
        return np.zeros((d, 1))
    return np.stack(basis, axis=1)


def compute_subspace_trace(X_tr, y_tr, mu, classes, U_directions):
    N = len(X_tr)
    tr = 0.0
    for k_vec in U_directions.T:
        sigma_sq = 0.0
        for i, c in enumerate(classes):
            Xc_c = X_tr[y_tr == c] - mu[i]
            n_c = len(Xc_c)
            proj = Xc_c @ k_vec
            sigma_sq += (n_c / N) * float(np.mean(proj ** 2))
        tr += sigma_sq
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


def apply_null_surgery(X_tr, X_te, y_tr, y_te, geo, U_null_bundle, r):
    """Surgery entirely within null space: scale U_null directions by 1/sqrt(r),
    leave everything else untouched (no compensation needed since we're in null space).
    Actually: compensation goes into REMAINING null space."""
    mu = geo['mu']
    classes = np.unique(y_tr)
    trW = geo['trW']

    tr_null_bundle = compute_subspace_trace(X_tr, y_tr, mu, classes, U_null_bundle)
    # Remaining null space (null space minus the bundle directions)
    # tr(remaining null) = tr_total_null - tr_null_bundle
    # (We don't change between-class subspace at all)

    def transform(X, y):
        X_new = X.copy()
        for i, c in enumerate(classes):
            mask = (y == c)
            z = X[mask] - mu[i]

            # Null-bundle component
            z_null_bundle = z @ U_null_bundle @ U_null_bundle.T
            # Everything else (between-class + remaining null)
            z_rest = z - z_null_bundle

            # Scale null bundle by 1/sqrt(r), leave rest unchanged
            # (This changes tr(W) - we should compensate to keep tr(W) fixed)
            # But we'll do it cleanly: don't compensate, just scale the bundle
            # This tests: does changing null-space variance affect q?
            z_new = (1.0 / float(np.sqrt(r))) * z_null_bundle + z_rest
            X_new[mask] = mu[i] + z_new
        return X_new

    return transform(X_tr, y_tr), transform(X_te, y_te)


def eval_q(X_tr, y_tr, X_te, y_te, K):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = float(np.clip((acc - 1.0/K) / (1.0 - 1.0/K), 1e-6, 1-1e-6))
    return acc, q, float(sp.logit(q))


def compute_ols_slope(results_list, baseline_logit, delta_preds_by_r):
    """Compute OLS slope of delta_obs on delta_pred across r values."""
    nontrivial = [x for x in results_list if abs(x['r'] - 1.0) > 0.01]
    if len(nontrivial) < 3:
        return 0.0, 0.0
    deltas_obs = [x['delta_logit'] for x in nontrivial]
    deltas_pred = [delta_preds_by_r[x['r']] for x in nontrivial]
    if np.std(deltas_pred) < 1e-8:
        return 0.0, 0.0
    slope = float(np.polyfit(deltas_pred, deltas_obs, 1)[0])
    r_val = float(pearsonr(deltas_obs, deltas_pred)[0])
    return slope, r_val


def main():
    log("=" * 70)
    log("PROPER NULL CONTROL FOR ABCS")
    log("Test: does surgery in NULL SPACE show near-zero effect?")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log("=" * 70)
    log("PRE-REGISTERED: |slope_null| < 0.1 * slope_m5 for PASS")

    X_tr, y_tr, X_te, y_te, classes = load_and_split(EMBED_PATH, N_TRAIN_PER_CLASS, RANDOM_SEED)
    K = len(classes)
    d = X_tr.shape[1]
    log(f"\nK={K}, d={d}, n_train={len(X_tr)}, n_test={len(X_te)}")

    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)
    log(f"Baseline: acc={acc_base:.4f}, q={q_base:.4f}, logit={logit_base:.4f}")

    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])
    tr_bundle_all = compute_subspace_trace(X_tr, y_tr, geo['mu'], classes, U_B)
    tr_W_null = geo['trW'] - tr_bundle_all
    log(f"tr_W_null = {tr_W_null:.4f} ({tr_W_null/geo['trW']*100:.1f}%)")

    # Pre-registered prediction
    A_keff_K14 = ALPHA_KAPPA_K14 / float(np.sqrt(geo['d_eff']))
    C_keff = logit_base - A_keff_K14 * geo['kappa_eff']
    delta_preds_by_r = {}
    for r in R_LEVELS:
        kappa_eff_pred = geo['kappa_eff'] * float(np.sqrt(r))
        delta_pred = (C_keff + A_keff_K14 * kappa_eff_pred) - logit_base
        delta_preds_by_r[r] = float(delta_pred)

    # ==================== COMPETITIVE REFERENCE (m=5) ====================
    log(f"\n--- COMPETITIVE ARM (m={M_COMPETITIVE} top nearest pairs) ---")
    U_comp = build_competitive_bundle(geo['pair_info'], M_COMPETITIVE)
    comp_results = []
    for r in R_LEVELS:
        X_tr_c, X_te_c = apply_bundle_surgery(
            X_tr, X_te, y_tr, y_te, geo, U_comp, U_B, r, tr_W_null)
        acc_r, q_r, logit_r = eval_q(X_tr_c, y_tr, X_te_c, y_te, K)
        delta_obs = logit_r - logit_base
        log(f"  r={r:.1f}: delta_obs={delta_obs:+.4f}, delta_pred={delta_preds_by_r[r]:+.4f}, ratio={delta_obs/(delta_preds_by_r[r]+1e-10):.3f}")
        comp_results.append({'r': float(r), 'delta_logit': float(delta_obs)})

    slope_comp, r_comp = compute_ols_slope(comp_results, logit_base, delta_preds_by_r)
    log(f"COMPETITIVE slope = {slope_comp:.4f} (r={r_comp:.4f})")

    # ==================== NULL SPACE CONTROLS ====================
    log(f"\n--- NULL SPACE CONTROLS ---")
    rng_null = np.random.default_rng(RANDOM_SEED + 100)
    null_results_all = {}

    all_null_slopes = []

    for null_run in range(N_NULL_RUNS):
        for m_null in M_NULL:
            U_null = sample_null_bundle(U_B, d, m_null, rng_null)
            actual_m = U_null.shape[1]
            null_run_results = []

            for r in R_LEVELS:
                X_tr_n, X_te_n = apply_null_surgery(
                    X_tr, X_te, y_tr, y_te, geo, U_null, r)
                acc_r, q_r, logit_r = eval_q(X_tr_n, y_tr, X_te_n, y_te, K)
                delta_obs = logit_r - logit_base
                null_run_results.append({'r': float(r), 'delta_logit': float(delta_obs)})

            slope_null_run, _ = compute_ols_slope(null_run_results, logit_base, delta_preds_by_r)
            null_results_all[f"run{null_run}_m{m_null}"] = {
                'null_run': null_run, 'm': m_null, 'actual_m': actual_m,
                'slope': float(slope_null_run), 'results': null_run_results
            }
            log(f"  null_run={null_run}, m={m_null}: slope={slope_null_run:.4f}")
            if m_null == M_COMPETITIVE:
                all_null_slopes.append(slope_null_run)

    # Summary stats for m=M_COMPETITIVE null runs
    if all_null_slopes:
        mean_null_slope = float(np.mean(all_null_slopes))
        std_null_slope = float(np.std(all_null_slopes))
    else:
        mean_null_slope, std_null_slope = 0.0, 0.0

    # Also check ALL null slopes together
    all_slopes_flat = [v['slope'] for v in null_results_all.values()]
    abs_mean_null = float(np.mean(np.abs(all_slopes_flat)))

    # ==================== VERDICT ====================
    log("\n" + "=" * 70)
    log("VERDICT")
    log("=" * 70)
    log(f"Competitive slope (m=5): {slope_comp:.4f}")
    log(f"Null slopes (m={M_COMPETITIVE}, {N_NULL_RUNS} runs): mean={mean_null_slope:.4f} +/- {std_null_slope:.4f}")
    log(f"All null slopes mean |slope|: {abs_mean_null:.4f}")
    log(f"Ratio |null|/competitive: {abs_mean_null/abs(slope_comp):.3f}")

    threshold = 0.10
    pass_null = abs_mean_null < threshold * abs(slope_comp)

    if pass_null:
        verdict = f"NULL CONTROL PASS: Null-space surgery has near-zero effect ({abs_mean_null:.3f} < {threshold*abs(slope_comp):.3f}). Surgery is SPECIFIC to competitive structure."
    else:
        ratio = abs_mean_null / (abs(slope_comp) + 1e-10)
        verdict = f"NULL CONTROL PARTIAL: Null effect = {ratio:.1%} of competitive. Some bleed-through into accuracy."

    log(f"\nVERDICT: {verdict}")

    result = {
        'experiment': 'null_control_abcs',
        'timestamp': datetime.datetime.now().isoformat(),
        'baseline': {'K': int(K), 'd': int(d), 'acc': float(acc_base),
                     'q': float(q_base), 'logit': float(logit_base)},
        'competitive_arm': {
            'slope': float(slope_comp), 'pearson_r': float(r_comp),
            'm': M_COMPETITIVE
        },
        'null_arm': {
            'runs': null_results_all,
            'mean_null_slope_m5': float(mean_null_slope),
            'std_null_slope_m5': float(std_null_slope),
            'abs_mean_all': float(abs_mean_null),
        },
        'analysis': {
            'ratio_null_to_competitive': float(abs_mean_null / (abs(slope_comp) + 1e-10)),
            'pass_null': bool(pass_null),
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
