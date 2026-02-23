#!/usr/bin/env python -u
"""
CONFIRMATORY d_eff_1 SWEEP v2: A_local = B_univ / sqrt(d_eff_1)
================================================================
PURPOSE: Pre-registered confirmatory replication of Session 48 result.
         Key finding: A_local = B_univ/sqrt(d_eff_1), B_univ ~ 6.1, CV ~ 5%.

CHANGES FROM v1 (Session 48):
  - kappa_nearest = 0.20 (vs 0.40 in v1)
    => All alpha levels target q_base in [0.68, 0.79], eliminating saturation
  - ALPHA_LEVELS = [0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]
    => 8 levels all in valid regime, finer resolution
  - Stricter pre-reg: P3 CV_B < 0.15 (was < 0.25 in v1)
  - P4 threshold: r_wk > 0.70 for >=6/8 levels (was >=5/8 in v1)

DESIGN:
  d=768, K=14, kappa=0.20, alpha in [0.625, 1.5] (8 levels, Dalpha=0.125)
  Expected d_eff_1 range: [33, 18.6, 11, 7.65, 4.7, 4.03, 3.3, 2.68] — 12x span
  Expected A_local range: [1.06, 1.41, 1.84, 2.21, 2.82, 3.04, 3.36, 3.73]
  Expected B range: [6.1, 6.1, 6.1, ...] all ~6.1 (theory)

PRE-REGISTERED (Feb 26, 2026):
  P1: r(log d_eff_1, log A_local) < -0.60  (stronger threshold than v1's -0.40)
  P2: slope of log(A_local) on log(d_eff_1) in [-0.60, -0.40]  (slope ~ -0.5)
  P3: B = A_local * sqrt(d_eff_1) CV < 0.15  (stricter than v1's 0.25)
  P4: r_wk > 0.70 for >=6/8 alpha levels

THEORETICAL PREDICTION: A_local = B_univ / sqrt(d_eff_1)
  with B_univ ~ 6.1 (from Session 48 valid range, 4 data points, CV=5.4%)
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_deff1_sweep_v2.json"
LOG_PATH = "results/cti_deff1_sweep_v2_log.txt"

K = 14
N_PER_CLASS = 150
KAPPA_NEAREST = 0.20   # smaller kappa → lower q_base → avoids saturation
D = 768
N_SEEDS = 5
SEEDS = [42, 137, 271, 919, 2345]
R_LEVELS = [2.0, 3.0, 5.0, 7.0, 10.0]
N_PAIRS = 10

# 8 alpha levels, all in valid regime (d_eff_1 in [2.68, 33])
ALPHA_LEVELS = [0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]

os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')


def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def make_covariance_alpha(alpha, d, spec_seed=0):
    i_vals = np.arange(d).astype(float)
    if alpha == 0.0:
        eigenvalues = np.ones(d)
    else:
        eigenvalues = 1.0 / (1.0 + i_vals) ** alpha
    eigenvalues = eigenvalues / eigenvalues.mean()
    np.random.seed(spec_seed)
    U = np.linalg.qr(np.random.randn(d, d))[0]
    Sigma = U @ np.diag(eigenvalues) @ U.T
    return Sigma, eigenvalues


def generate_synthetic_embeddings(K, d, n_per_class, Sigma, kappa_nearest, seed):
    rng = np.random.default_rng(seed)
    trW = float(np.trace(Sigma))
    sigma_W_global = float(np.sqrt(trW / d))
    delta_min = kappa_nearest * sigma_W_global * np.sqrt(d)
    eigenvalues_raw, U_raw = np.linalg.eigh(Sigma)
    main_dir = U_raw[:, -1]

    mu = np.zeros((K, d))
    mu[0] = main_dir * (delta_min / 2.0)
    mu[1] = -main_dir * (delta_min / 2.0)
    spacing_multipliers = np.linspace(1.5, 4.0, K - 2)
    for k in range(2, K):
        direction = rng.standard_normal(d)
        direction /= np.linalg.norm(direction)
        mu[k] = direction * delta_min * spacing_multipliers[k - 2]

    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(d))
    X_list, y_list = [], []
    for c in range(K):
        Z = rng.standard_normal((n_per_class, d))
        X_list.append(mu[c] + Z @ L.T)
        y_list.append(np.full(n_per_class, c))

    return np.concatenate(X_list), np.concatenate(y_list), mu


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
    if tr_W_null > 1e-12 and tr_null_new > 0:
        scale_null = float(np.sqrt(tr_null_new / tr_W_null))
    else:
        scale_null = 1.0
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
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean',
                               algorithm='brute', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = float(np.clip((acc - 1.0 / K) / (1.0 - 1.0 / K), 1e-6, 1 - 1e-6))
    return acc, q, float(sp.logit(q))


def run_surgery_for_seed(X_tr, y_tr, X_te, y_te, classes):
    K = len(classes)
    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)
    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])

    kappa_1 = geo['pair_info'][0]['kappa']
    d_eff_1 = geo['pair_info'][0]['d_eff']
    kappa_eff_1 = geo['pair_info'][0]['kappa_eff']

    A_ref = 1.477 / float(np.sqrt((K - 1) * d_eff_1))
    C_ref = logit_base - A_ref * kappa_eff_1
    delta_preds_by_r = {}
    for r in R_LEVELS:
        kappa_eff_r = kappa_eff_1 * float(np.sqrt(r))
        delta_preds_by_r[r] = float(C_ref + A_ref * kappa_eff_r) - logit_base

    pair_results = []
    for k_idx, pair in enumerate(geo['pair_info'][:N_PAIRS]):
        kappa_k = pair['kappa']
        d_eff_k = pair['d_eff']

        single_results = []
        for r in R_LEVELS:
            X_tr_r, X_te_r = apply_single_pair_surgery(
                X_tr, X_te, y_tr, y_te, geo, pair, U_B, r)
            _, _, logit_r = eval_q(X_tr_r, y_tr, X_te_r, y_te, K)
            single_results.append({'r': float(r), 'delta_logit': float(logit_r - logit_base)})

        d_obs = [x['delta_logit'] for x in single_results]
        d_pred = [delta_preds_by_r[x['r']] for x in single_results]
        if np.std(d_pred) < 1e-8:
            continue

        slope_k = float(np.polyfit(d_pred, d_obs, 1)[0])
        r_k = float(pearsonr(d_obs, d_pred)[0])
        norm_factor = float(kappa_1 * np.sqrt(d_eff_1) /
                            (kappa_k * np.sqrt(d_eff_k) + 1e-10))
        empirical_wk = float(slope_k * norm_factor)
        delta_kappa = kappa_k - kappa_1
        delta_kappa_sqrt_deff = float(delta_kappa * np.sqrt(d_eff_1))

        pair_results.append({
            'k': k_idx + 1,
            'kappa_k': float(kappa_k), 'd_eff_k': float(d_eff_k),
            'slope_k': float(slope_k), 'r_k': float(r_k),
            'empirical_wk': float(empirical_wk),
            'delta_kappa': float(delta_kappa),
            'delta_kappa_sqrt_deff': float(delta_kappa_sqrt_deff),
        })

    if not pair_results:
        return float('nan'), float('nan'), {}, []

    emp_wks = [pr['empirical_wk'] for pr in pair_results]

    # Fit A_local from log-linear decay
    valid = [(pr['delta_kappa_sqrt_deff'], pr['empirical_wk']) for pr in pair_results
             if pr['delta_kappa_sqrt_deff'] > 1e-6 and pr['empirical_wk'] > 1e-6]
    if len(valid) >= 3:
        dk_arr = np.array([v[0] for v in valid])
        lw_arr = np.array([-np.log(v[1]) for v in valid])
        A_local_fit = float(np.polyfit(dk_arr, lw_arr, 1)[0])
    else:
        A_local_fit = float('nan')

    # r_wk with fitted A_local
    if not np.isnan(A_local_fit) and len(emp_wks) >= 3:
        pred_wks = [float(np.exp(-A_local_fit * pr['delta_kappa_sqrt_deff']))
                    for pr in pair_results]
        r_wk = float(pearsonr(emp_wks, pred_wks)[0]) if np.std(pred_wks) > 1e-6 else 0.0
    else:
        r_wk = 0.0

    baseline = {
        'acc_base': float(acc_base), 'q_base': float(q_base),
        'logit_base': float(logit_base),
        'kappa_1': float(kappa_1), 'd_eff_1': float(d_eff_1),
    }
    return A_local_fit, r_wk, baseline, pair_results


def main():
    log("=" * 70)
    log("CONFIRMATORY d_eff_1 SWEEP v2: A_local = B_univ / sqrt(d_eff_1)")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log(f"d={D}, K={K}, N_PER_CLASS={N_PER_CLASS}, kappa_nearest={KAPPA_NEAREST}")
    log(f"Alpha levels: {ALPHA_LEVELS}")
    log(f"R_LEVELS: {R_LEVELS}")
    log("=" * 70)
    log("PRE-REGISTERED (Feb 26, 2026) — CONFIRMATORY of Session 48:")
    log("  P1: r(log d_eff_1, log A_local) < -0.60  (stronger than v1)")
    log("  P2: slope in [-0.60, -0.40]  (slope ~ -0.5, same as v1)")
    log("  P3: B = A_local*sqrt(d_eff_1) CV < 0.15  (stricter than v1)")
    log("  P4: r_wk > 0.70 for >=6/8 alpha levels")
    log(f"PRIOR: B_univ ~ 6.1 (Session 48 valid range, 4 pts, r=-0.993, CV=5.4%)")

    all_results = []

    for alpha in ALPHA_LEVELS:
        log(f"\n{'='*50}")
        log(f"ALPHA = {alpha:.3f}")
        log(f"{'='*50}")

        Sigma, eigenvalues = make_covariance_alpha(alpha, D)
        lambda_max = float(eigenvalues.max())
        d_eff_theoretical = float(D / lambda_max)
        A_predicted = 6.1 / float(np.sqrt(d_eff_theoretical))
        log(f"  lambda_max={lambda_max:.4f}, d_eff_1_theory={d_eff_theoretical:.3f}, "
            f"A_local_predicted={A_predicted:.4f}")

        seed_results = []
        for seed in SEEDS:
            X, y, _ = generate_synthetic_embeddings(
                K, D, N_PER_CLASS * 2, Sigma, KAPPA_NEAREST, seed)
            rng2 = np.random.default_rng(seed + 1000)
            classes = np.unique(y)
            X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
            for c in classes:
                idx = np.where(y == c)[0]
                rng2.shuffle(idx)
                n_tr = min(N_PER_CLASS, len(idx) // 2)
                X_tr_list.append(X[idx[:n_tr]])
                y_tr_list.append(y[idx[:n_tr]])
                X_te_list.append(X[idx[n_tr:n_tr + n_tr]])
                y_te_list.append(y[idx[n_tr:n_tr + n_tr]])
            X_tr = np.concatenate(X_tr_list)
            y_tr = np.concatenate(y_tr_list)
            X_te = np.concatenate(X_te_list)
            y_te = np.concatenate(y_te_list)

            A_local_fit, r_wk, baseline, pair_results = run_surgery_for_seed(
                X_tr, y_tr, X_te, y_te, classes)

            if not baseline:
                log(f"  seed={seed}: SKIP")
                continue

            d_eff_1 = baseline['d_eff_1']
            B = (A_local_fit * np.sqrt(d_eff_1)
                 if not (A_local_fit != A_local_fit) else float('nan'))
            log(f"  seed={seed}: A_local={A_local_fit:.4f} (pred={A_predicted:.4f}), "
                f"r_wk={r_wk:.3f}, d_eff_1={d_eff_1:.3f}, B={B:.4f}, "
                f"q_base={baseline['q_base']:.3f}")
            seed_results.append({
                'seed': int(seed),
                'A_local_fit': float(A_local_fit),
                'r_wk': float(r_wk),
                'B': float(B),
                'baseline': baseline,
            })

        if not seed_results:
            log(f"  alpha={alpha}: ALL SEEDS SKIP")
            continue

        valid_A = [s['A_local_fit'] for s in seed_results
                   if not (s['A_local_fit'] != s['A_local_fit'])]
        valid_B = [s['B'] for s in seed_results
                   if not (s['B'] != s['B'])]
        valid_r = [s['r_wk'] for s in seed_results]
        mean_deff = float(np.mean([s['baseline']['d_eff_1'] for s in seed_results]))
        mean_A = float(np.mean(valid_A)) if valid_A else float('nan')
        std_A = float(np.std(valid_A)) if len(valid_A) > 1 else 0.0
        mean_B = float(np.mean(valid_B)) if valid_B else float('nan')
        std_B = float(np.std(valid_B)) if len(valid_B) > 1 else 0.0
        mean_r = float(np.mean(valid_r))

        log(f"  SUMMARY alpha={alpha:.3f}: A_local={mean_A:.4f}+/-{std_A:.4f}, "
            f"d_eff_1={mean_deff:.3f}, B={mean_B:.4f}+/-{std_B:.4f}, "
            f"r_wk={mean_r:.3f}")

        all_results.append({
            'alpha': float(alpha),
            'd_eff_theoretical': float(d_eff_theoretical),
            'A_predicted': float(A_predicted),
            'mean_d_eff_1': float(mean_deff),
            'mean_A_local': float(mean_A),
            'std_A_local': float(std_A),
            'mean_B': float(mean_B),
            'std_B': float(std_B),
            'mean_r_wk': float(mean_r),
            'n_valid_seeds': int(len(valid_A)),
            'seed_results': seed_results,
        })

    # ==================== ANALYSIS ====================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    if len(all_results) < 3:
        log("Insufficient results!")
        return

    valid = [(r['mean_d_eff_1'], r['mean_A_local']) for r in all_results
             if not (r['mean_A_local'] != r['mean_A_local']) and r['mean_A_local'] > 0
             and r['mean_d_eff_1'] > 0]

    if len(valid) < 3:
        log("Not enough valid data points!")
        return

    log_deff = [np.log(v[0]) for v in valid]
    log_A = [np.log(v[1]) for v in valid]

    # P1: r(log d_eff_1, log A_local) < -0.60
    r_P1, p_P1 = pearsonr(log_deff, log_A)
    pass_P1 = r_P1 < -0.60
    log(f"\nP1 r(log d_eff_1, log A_local) = {r_P1:.4f} (p={p_P1:.6f}) "
        f"-> {'PASS' if pass_P1 else 'FAIL'} (threshold < -0.60)")

    # P2: slope in [-0.60, -0.40]
    slope_P2, intercept_P2 = np.polyfit(log_deff, log_A, 1)
    pass_P2 = -0.60 <= slope_P2 <= -0.40
    B_univ_est = float(np.exp(intercept_P2))
    log(f"P2 slope = {slope_P2:.4f}, B_univ_estimate = {B_univ_est:.4f} "
        f"-> {'PASS' if pass_P2 else 'FAIL'} (target [-0.60, -0.40])")

    # P3: B = A_local * sqrt(d_eff_1) CV < 0.15
    B_vals = [r['mean_B'] for r in all_results
              if not (r['mean_B'] != r['mean_B']) and r['mean_B'] > 0]
    mean_B_overall = float(np.mean(B_vals)) if B_vals else float('nan')
    cv_B = (float(np.std(B_vals) / (mean_B_overall + 1e-10))
            if B_vals and mean_B_overall > 0 else float('nan'))
    pass_P3 = cv_B < 0.15
    log(f"P3 B = A_local*sqrt(d_eff_1): mean={mean_B_overall:.4f}, CV={cv_B:.4f} "
        f"-> {'PASS' if pass_P3 else 'FAIL'} (target CV < 0.15)")

    # P4: r_wk > 0.70 for >=6/8 alpha levels
    n_pass_P4 = sum(1 for r in all_results if r['mean_r_wk'] > 0.70)
    pass_P4 = n_pass_P4 >= 6
    log(f"P4 r_wk > 0.70: {n_pass_P4}/{len(all_results)} alpha levels "
        f"-> {'PASS' if pass_P4 else 'FAIL'} (threshold >=6/8)")

    # Session 48 comparison: B_v1 = 6.11 ± 0.33
    b_v1 = 6.11
    b_diff = abs(mean_B_overall - b_v1) / b_v1 if mean_B_overall > 0 else float('nan')
    log(f"\nComparison to Session 48: B_v1={b_v1:.3f}, B_v2={mean_B_overall:.3f}, "
        f"relative diff = {b_diff:.4f}")

    log("\nFull table (sorted by alpha):")
    log(f"{'alpha':8} {'d_eff':10} {'A_pred':10} {'A_local':10} {'B':10} {'r_wk':8} {'n':5}")
    for r in sorted(all_results, key=lambda x: x['alpha']):
        log(f"  {r['alpha']:8.3f} {r['mean_d_eff_1']:10.3f} {r['A_predicted']:10.4f} "
            f"{r['mean_A_local']:10.4f} {r['mean_B']:10.4f} "
            f"{r['mean_r_wk']:8.3f} {r['n_valid_seeds']:5d}")

    overall_pass = pass_P1 and pass_P2 and pass_P3 and pass_P4
    verdict = (
        f"P1={'PASS' if pass_P1 else 'FAIL'}(r={r_P1:.3f}), "
        f"P2={'PASS' if pass_P2 else 'FAIL'}(slope={slope_P2:.3f}), "
        f"P3={'PASS' if pass_P3 else 'FAIL'}(CV_B={cv_B:.3f}), "
        f"P4={'PASS' if pass_P4 else 'FAIL'}({n_pass_P4}/8 r_wk>0.70). "
        f"B_univ_est={B_univ_est:.3f} (Session48: 6.11). "
        f"OVERALL: {'PASS' if overall_pass else 'FAIL'}"
    )
    log(f"\nVERDICT: {verdict}")

    output = {
        'experiment': 'deff1_sweep_v2',
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'd': D, 'K': K, 'n_per_class': N_PER_CLASS,
            'kappa_nearest': KAPPA_NEAREST,
            'alpha_levels': ALPHA_LEVELS,
            'r_levels': R_LEVELS,
            'seeds': SEEDS,
            'n_pairs': N_PAIRS,
            'prior_B_univ': 6.11,
        },
        'all_results': all_results,
        'analysis': {
            'r_P1': float(r_P1), 'p_P1': float(p_P1), 'pass_P1': bool(pass_P1),
            'slope_P2': float(slope_P2), 'intercept_P2': float(intercept_P2),
            'B_univ_est': float(B_univ_est), 'pass_P2': bool(pass_P2),
            'mean_B_overall': float(mean_B_overall),
            'cv_B': float(cv_B), 'pass_P3': bool(pass_P3),
            'n_pass_P4': int(n_pass_P4), 'pass_P4': bool(pass_P4),
            'overall_pass': bool(overall_pass),
            'B_v1': float(b_v1), 'B_v2': float(mean_B_overall),
            'B_relative_diff': float(b_diff),
        },
        'verdict': verdict,
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
