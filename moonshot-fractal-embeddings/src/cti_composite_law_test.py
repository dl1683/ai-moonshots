#!/usr/bin/env python -u
"""
COMPOSITE LAW CONFIRMATORY TEST: A_local = C / (kappa^p * sqrt(d_eff_1))
==========================================================================
PURPOSE: Held-out kappa=0.30 test of the composite formula derived from:
  v1 (kappa=0.40): B_univ=6.11, CV=5.3%  (4 alpha levels)
  v2 (kappa=0.20): B_univ=10.16, CV=3.9% (3 alpha levels)
  Composite fit: A_local * kappa^0.73 * sqrt(d_eff_1) = C = 3.12, CV=4.8% (7 pts)

COMPOSITE LAW: A_local = C / (kappa^p * sqrt(d_eff_1))
  Fitted: C=3.12, p=0.73 (CV=4.8%)
  Rounded: C=pi=3.14159, p=0.75=3/4

PRE-REGISTERED (Feb 23, 2026 -- committed BEFORE running):
  Held-out kappa = 0.30
  Prior: C=3.12, p=0.75
  Predicted B_univ(kappa=0.30) = 3.12 / 0.30^0.75 = 7.63
  Per-alpha prediction: A_local_pred = 7.63 / sqrt(d_eff_1_empirical)

  PR1: |B_mean_valid - 7.63| / 7.63 < 0.20 (B within 20% of prediction)
  PR2: r(log d_eff_1, log A_local) < -0.50 for valid alpha levels
  PR3: slope of log A_local on log d_eff_1 in [-0.65, -0.40]
  PR4: r_wk > 0.70 for >= 4/8 alpha levels (valid range uncertain)
  PR5: mean relative error |A_local - A_pred| / A_pred < 0.25 across valid alpha

  Valid alpha = alpha levels where r_wk >= 0.70 (same criterion as v1/v2)

ALSO TEST: C=pi=3.14159 -- does pi fit even better?
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_composite_law_test.json"
LOG_PATH = "results/cti_composite_law_test_log.txt"

K = 14
N_PER_CLASS = 150
KAPPA_NEAREST = 0.30          # HELD-OUT kappa (not used in v1=0.40 or v2=0.20)
D = 768
N_SEEDS = 5
SEEDS = [42, 137, 271, 919, 2345]
R_LEVELS = [2.0, 3.0, 5.0, 7.0, 10.0]
N_PAIRS = 10

# Same alpha range as v2 for comparability
ALPHA_LEVELS = [0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]

# Pre-registered predictions
C_PRIOR = 3.12        # composite constant from v1+v2
P_PRIOR = 0.75        # exponent (3/4), from B ~ kappa^(-0.73) fit
B_PRED = C_PRIOR / (KAPPA_NEAREST ** P_PRIOR)   # = 3.12 / 0.30^0.75 = 7.63
C_PI = 3.14159        # alternative: C = pi
B_PRED_PI = C_PI / (KAPPA_NEAREST ** P_PRIOR)   # = 7.67

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
    return Sigma, eigenvalues, U


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

    valid = [(pr['delta_kappa_sqrt_deff'], pr['empirical_wk']) for pr in pair_results
             if pr['delta_kappa_sqrt_deff'] > 1e-6 and pr['empirical_wk'] > 1e-6]
    if len(valid) >= 3:
        dk_arr = np.array([v[0] for v in valid])
        lw_arr = np.array([-np.log(v[1]) for v in valid])
        A_local_fit = float(np.polyfit(dk_arr, lw_arr, 1)[0])
    else:
        A_local_fit = float('nan')

    if not np.isnan(A_local_fit) and len(pair_results) >= 3:
        pred_wks = [float(np.exp(-A_local_fit * pr['delta_kappa_sqrt_deff']))
                    for pr in pair_results]
        emp_wks = [pr['empirical_wk'] for pr in pair_results]
        r_wk = float(pearsonr(emp_wks, pred_wks)[0]) if np.std(pred_wks) > 1e-6 else 0.0
    else:
        r_wk = 0.0

    # Predicted A_local from composite law
    A_local_pred = B_PRED / float(np.sqrt(d_eff_1))
    A_local_pred_pi = B_PRED_PI / float(np.sqrt(d_eff_1))

    baseline = {
        'acc_base': float(acc_base), 'q_base': float(q_base),
        'logit_base': float(logit_base),
        'kappa_1': float(kappa_1), 'd_eff_1': float(d_eff_1),
        'A_local_pred': float(A_local_pred),
        'A_local_pred_pi': float(A_local_pred_pi),
    }
    return A_local_fit, r_wk, baseline, pair_results


def main():
    log("=" * 70)
    log("COMPOSITE LAW TEST: A_local = C/(kappa^p * sqrt(d_eff_1))")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log(f"d={D}, K={K}, N_PER_CLASS={N_PER_CLASS}, kappa_nearest={KAPPA_NEAREST}")
    log(f"Alpha levels: {ALPHA_LEVELS}")
    log(f"R_LEVELS: {R_LEVELS}")
    log("=" * 70)
    log("PRE-REGISTERED (Feb 23, 2026 -- held-out kappa=0.30):")
    log(f"  Prior: C={C_PRIOR}, p={P_PRIOR}")
    log(f"  B_pred = C/kappa^p = {C_PRIOR}/{KAPPA_NEAREST}^{P_PRIOR} = {B_PRED:.4f}")
    log(f"  Also testing C=pi: B_pred_pi = {B_PRED_PI:.4f}")
    log("  PR1: |B_mean_valid - B_pred| / B_pred < 0.20")
    log("  PR2: r(log d_eff_1, log A_local) < -0.50 for valid levels")
    log("  PR3: slope in [-0.65, -0.40]")
    log("  PR4: r_wk > 0.70 for >= 4/8 alpha levels")
    log("  PR5: mean relative error |A_local-A_pred|/A_pred < 0.25")
    log(f"  Valid alpha = levels with mean r_wk >= 0.70")

    all_results = []

    for alpha in ALPHA_LEVELS:
        log(f"\n{'='*50}")
        log(f"ALPHA = {alpha:.3f}")
        log(f"{'='*50}")

        Sigma, eigenvalues, U = make_covariance_alpha(alpha, D)
        lambda_max = float(eigenvalues.max())
        d_eff_theoretical = float(D / lambda_max)
        A_local_theoretical = B_PRED / float(np.sqrt(d_eff_theoretical))
        log(f"  lambda_max={lambda_max:.4f}, d_eff_theory={d_eff_theoretical:.3f}, "
            f"A_local_theory={A_local_theoretical:.4f}, B_pred={B_PRED:.3f}")

        seed_results = []
        for seed in SEEDS:
            try:
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
            except Exception as e:
                log(f"  seed={seed}: EXCEPTION {type(e).__name__}: {e}")
                continue

            if not baseline:
                log(f"  seed={seed}: SKIP")
                continue

            d_eff_1 = baseline['d_eff_1']
            A_pred = baseline['A_local_pred']
            A_pred_pi = baseline['A_local_pred_pi']
            B = (A_local_fit * np.sqrt(d_eff_1)
                 if not np.isnan(A_local_fit) else float('nan'))
            rel_err = (abs(A_local_fit - A_pred) / A_pred
                       if not np.isnan(A_local_fit) else float('nan'))
            log(f"  seed={seed}: A_obs={A_local_fit:.4f}, A_pred={A_pred:.4f} "
                f"(pi:{A_pred_pi:.4f}), rel_err={rel_err:.3f}, "
                f"r_wk={r_wk:.3f}, B={B:.4f}, q_base={baseline['q_base']:.3f}")
            seed_results.append({
                'seed': int(seed),
                'A_local_fit': float(A_local_fit),
                'A_local_pred': float(A_pred),
                'A_local_pred_pi': float(A_pred_pi),
                'rel_error': float(rel_err) if not np.isnan(rel_err) else None,
                'r_wk': float(r_wk),
                'B': float(B) if not np.isnan(B) else None,
                'baseline': baseline,
            })

        if not seed_results:
            log(f"  alpha={alpha}: ALL SEEDS SKIP")
            continue

        valid_A = [s['A_local_fit'] for s in seed_results
                   if not np.isnan(s['A_local_fit'])]
        valid_B = [s['B'] for s in seed_results
                   if s['B'] is not None and not np.isnan(s['B'])]
        valid_r = [s['r_wk'] for s in seed_results]
        valid_rel_err = [s['rel_error'] for s in seed_results
                         if s['rel_error'] is not None and not np.isnan(s['rel_error'])]

        mean_deff = float(np.mean([s['baseline']['d_eff_1'] for s in seed_results]))
        mean_A = float(np.mean(valid_A)) if valid_A else float('nan')
        std_A = float(np.std(valid_A)) if len(valid_A) > 1 else 0.0
        mean_B = float(np.mean(valid_B)) if valid_B else float('nan')
        std_B = float(np.std(valid_B)) if len(valid_B) > 1 else 0.0
        mean_r = float(np.mean(valid_r))
        mean_pred = float(np.mean([s['A_local_pred'] for s in seed_results]))
        mean_rel_err = float(np.mean(valid_rel_err)) if valid_rel_err else float('nan')

        log(f"  SUMMARY alpha={alpha:.3f}: A_obs={mean_A:.4f}+/-{std_A:.4f}, "
            f"A_pred={mean_pred:.4f}, rel_err={mean_rel_err:.3f}, "
            f"d_eff_1={mean_deff:.3f}, B={mean_B:.4f}+/-{std_B:.4f}, "
            f"r_wk={mean_r:.3f}")

        all_results.append({
            'alpha': float(alpha),
            'd_eff_theoretical': float(d_eff_theoretical),
            'mean_d_eff_1': float(mean_deff),
            'mean_A_local': float(mean_A),
            'std_A_local': float(std_A),
            'mean_A_pred': float(mean_pred),
            'mean_rel_error': float(mean_rel_err) if not np.isnan(mean_rel_err) else None,
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
        log("INSUFFICIENT DATA for analysis")
        with open(RESULT_PATH, 'w') as f:
            json.dump({'all_results': all_results, 'error': 'insufficient data'}, f)
        return

    # Select valid levels (mean_r_wk >= 0.70)
    valid_levels = [r for r in all_results if r['mean_r_wk'] >= 0.70]
    log(f"\nValid alpha levels (r_wk>=0.70): {len(valid_levels)}/{len(all_results)}")
    for vl in valid_levels:
        log(f"  alpha={vl['alpha']:.3f}: d_eff={vl['mean_d_eff_1']:.2f}, "
            f"A={vl['mean_A_local']:.4f}, B={vl['mean_B']:.4f}, r_wk={vl['mean_r_wk']:.3f}")

    analysis = {}

    # PR1: B_mean within 20% of B_pred
    if valid_levels:
        B_vals = [vl['mean_B'] for vl in valid_levels if not np.isnan(vl['mean_B'])]
        B_mean_valid = float(np.mean(B_vals)) if B_vals else float('nan')
        B_rel_err = abs(B_mean_valid - B_PRED) / B_PRED if not np.isnan(B_mean_valid) else float('nan')
        B_rel_err_pi = abs(B_mean_valid - B_PRED_PI) / B_PRED_PI if not np.isnan(B_mean_valid) else float('nan')
        pr1 = B_rel_err < 0.20
        log(f"\nPR1 B_mean={B_mean_valid:.4f}, B_pred={B_PRED:.4f}, "
            f"rel_err={B_rel_err:.4f} -> {'PASS' if pr1 else 'FAIL'} (<0.20)")
        log(f"   Also: B_pred_pi={B_PRED_PI:.4f}, rel_err_pi={B_rel_err_pi:.4f}")
        analysis['B_mean_valid'] = float(B_mean_valid)
        analysis['B_pred'] = float(B_PRED)
        analysis['B_rel_err'] = float(B_rel_err)
        analysis['B_rel_err_pi'] = float(B_rel_err_pi)
        analysis['PR1'] = bool(pr1)
    else:
        log(f"\nPR1 SKIP (no valid levels)")
        analysis['PR1'] = False

    # PR2: r(log d_eff_1, log A_local)
    if len(valid_levels) >= 3:
        log_d = np.array([np.log(vl['mean_d_eff_1']) for vl in valid_levels])
        log_A = np.array([np.log(vl['mean_A_local']) for vl in valid_levels
                          if vl['mean_A_local'] > 0])
        if len(log_A) >= 3:
            r_logd_logA, p_r = pearsonr(log_d[:len(log_A)], log_A)
            pr2 = r_logd_logA < -0.50
            log(f"\nPR2 r(log d_eff_1, log A_local) = {r_logd_logA:.4f} (p={p_r:.4f}) "
                f"-> {'PASS' if pr2 else 'FAIL'} (<-0.50)")
            analysis['r_logd_logA'] = float(r_logd_logA)
            analysis['PR2'] = bool(pr2)
        else:
            log("\nPR2 SKIP (insufficient positive A_local values)")
            analysis['PR2'] = False
    else:
        log("\nPR2 SKIP (< 3 valid levels)")
        analysis['PR2'] = False

    # PR3: slope of log A_local on log d_eff_1
    if len(valid_levels) >= 3:
        pos_A = [(vl['mean_d_eff_1'], vl['mean_A_local']) for vl in valid_levels
                 if vl['mean_A_local'] > 0]
        if len(pos_A) >= 3:
            log_d_arr = np.array([np.log(x[0]) for x in pos_A])
            log_A_arr = np.array([np.log(x[1]) for x in pos_A])
            slope = float(np.polyfit(log_d_arr, log_A_arr, 1)[0])
            pr3 = -0.65 <= slope <= -0.40
            log(f"\nPR3 slope = {slope:.4f} -> {'PASS' if pr3 else 'FAIL'} ([-0.65,-0.40])")
            analysis['slope'] = float(slope)
            analysis['PR3'] = bool(pr3)
        else:
            log("\nPR3 SKIP")
            analysis['PR3'] = False
    else:
        log("\nPR3 SKIP")
        analysis['PR3'] = False

    # PR4: r_wk > 0.70 for >= 4/8 alpha levels
    n_r_wk_pass = sum(1 for r in all_results if r['mean_r_wk'] > 0.70)
    pr4 = n_r_wk_pass >= 4
    log(f"\nPR4 r_wk>0.70: {n_r_wk_pass}/{len(all_results)} "
        f"-> {'PASS' if pr4 else 'FAIL'} (>= 4/8)")
    analysis['n_r_wk_pass'] = int(n_r_wk_pass)
    analysis['PR4'] = bool(pr4)

    # PR5: mean relative error < 0.25
    all_rel_errs = [vl['mean_rel_error'] for vl in valid_levels
                    if vl['mean_rel_error'] is not None and not np.isnan(vl['mean_rel_error'])]
    if all_rel_errs:
        mean_rel_err_valid = float(np.mean(all_rel_errs))
        pr5 = mean_rel_err_valid < 0.25
        log(f"\nPR5 mean_rel_err (valid levels) = {mean_rel_err_valid:.4f} "
            f"-> {'PASS' if pr5 else 'FAIL'} (<0.25)")
        analysis['mean_rel_err_valid'] = float(mean_rel_err_valid)
        analysis['PR5'] = bool(pr5)
    else:
        log("\nPR5 SKIP")
        analysis['PR5'] = False

    # Summary table
    log(f"\nFull table (sorted by alpha):")
    log(f"{'alpha':>8}  {'d_eff':>8}  {'A_pred':>8}  {'A_obs':>8}  "
        f"{'B':>8}  {'r_wk':>6}  {'rel_err':>7}  {'n':>3}")
    for r in all_results:
        log(f"  {r['alpha']:>6.3f}  {r['mean_d_eff_1']:>8.3f}  "
            f"{r['mean_A_pred']:>8.4f}  {r['mean_A_local']:>8.4f}  "
            f"{r['mean_B']:>8.4f}  {r['mean_r_wk']:>6.3f}  "
            f"{(r['mean_rel_error'] or 0):>7.3f}  {r['n_valid_seeds']:>3}")

    pass_list = [k for k in ['PR1', 'PR2', 'PR3', 'PR4', 'PR5'] if analysis.get(k, False)]
    verdict = "PASS" if len(pass_list) >= 3 else "FAIL"
    verdict_parts = ['PR1=' + ('PASS' if analysis.get('PR1') else 'FAIL'),
                     'PR2=' + ('PASS' if analysis.get('PR2') else 'FAIL'),
                     'PR3=' + ('PASS' if analysis.get('PR3') else 'FAIL'),
                     'PR4=' + ('PASS' if analysis.get('PR4') else 'FAIL'),
                     'PR5=' + ('PASS' if analysis.get('PR5') else 'FAIL')]
    log(f"\nVERDICT: {' '.join(verdict_parts)}. OVERALL: {verdict}")
    log(f"B_univ_pred={B_PRED:.3f}, B_univ_observed={analysis.get('B_mean_valid', float('nan')):.3f}")

    output = {
        'meta': {
            'kappa': float(KAPPA_NEAREST),
            'C_prior': float(C_PRIOR), 'p_prior': float(P_PRIOR),
            'B_pred': float(B_PRED), 'B_pred_pi': float(B_PRED_PI),
            'timestamp': datetime.datetime.now().isoformat(),
        },
        'analysis': analysis,
        'all_results': all_results,
        'verdict': verdict,
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")
    _log_fh.close()


if __name__ == '__main__':
    main()
