#!/usr/bin/env python -u
"""
MULTI-K SWEEP: B(K) = C_0 * sqrt(K-1) / kappa^p
================================================
PURPOSE:
  Directly test the sqrt(K-1) scaling of the competition law.
  From K=7 and K=14 data:
    B = A_local * sqrt(d_eff_1) = C_0 * sqrt(K-1) / kappa^p
  where C_0 ~ sqrt(2/3) ~ 0.816, p ~ 0.791.

  If this holds across K=4,5,7,10,14,20, the law is confirmed.

EVIDENCE SO FAR:
  K=7:  B(kappa=0.30) = 4.915, C_0=4.915*0.30^0.791/sqrt(6)=0.808
  K=14: B(kappa=0.30) = 7.25,  C_0=7.25*0.30^0.791/sqrt(13)=0.776
  K=14: B(kappa=0.40) = 6.11,  C_0=6.11*0.40^0.791/sqrt(13)=0.820
  Mean C_0 = 0.801, sqrt(2/3) = 0.8165

DESIGN:
  - K values: [4, 5, 7, 10, 14, 20]
  - Fixed kappa = 0.30 (tested in both K=7 sweep and K=14 v3)
  - Alpha levels: [0.75, 1.0, 1.25] (3 levels in valid range)
  - Seeds: 12 per (K, alpha) combination
  - Total: 6K * 3alpha * 12seeds = 216 runs

PRE-REGISTERED (Feb 23, 2026 -- committed BEFORE running):
  Prior: C_0 = 0.816 = sqrt(2/3), p = 0.791 (from K=7 fit, 6 kappa points, r=-0.9925)
  Predicted B(K) at kappa=0.30:
    B(K=4)  = 0.816*sqrt(3)/0.30^0.791  = 3.66
    B(K=5)  = 0.816*sqrt(4)/0.30^0.791  = 4.23
    B(K=7)  = 0.816*sqrt(6)/0.30^0.791  = 5.18  [observed: 4.915]
    B(K=10) = 0.816*sqrt(9)/0.30^0.791  = 6.34
    B(K=14) = 0.816*sqrt(13)/0.30^0.791 = 7.63  [observed: 7.25]
    B(K=20) = 0.816*sqrt(19)/0.30^0.791 = 9.21

  PR1: r(B_K, sqrt(K-1)) > 0.97 (near-linear relationship)
  PR2: fitted slope s = C_0_obs/kappa^p_obs; |s/s_pred - 1| < 0.15 where s_pred=C_0/kappa^p
  PR3: CV of B_K / sqrt(K-1) across 6 K values < 0.10 (universal constant)
  PR4: within-K CV (across alpha levels) < 0.15 for >= 4/6 K values
  PRIMARY: PR1 + PR3 must both pass
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_multi_k_sweep.json"
LOG_PATH = "results/cti_multi_k_sweep_log.txt"

KAPPA = 0.30
D = 768
N_PER_CLASS = 200
N_SEEDS = 12
SEEDS = [42, 137, 271, 919, 2345, 7777, 8888, 9999, 11111, 12345, 31415, 27182]
R_LEVELS = [2.0, 3.0, 5.0, 7.0, 10.0]
N_PAIRS = 15

K_VALUES = [4, 5, 7, 10, 14, 20]
ALPHA_LEVELS = [0.75, 1.0, 1.25]

# Pre-registered predictions
C_0 = np.sqrt(2.0 / 3.0)   # = 0.8165
P_PRIOR = 0.791              # from K=7 sweep

os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')


def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def b_pred(K):
    return C_0 * np.sqrt(K - 1) / (KAPPA ** P_PRIOR)


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


def run_surgery_for_seed(X_tr, y_tr, X_te, y_te, classes, min_valid_pairs=2):
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
    n_pairs_to_test = min(N_PAIRS, len(geo['pair_info']))
    for k_idx, pair in enumerate(geo['pair_info'][:n_pairs_to_test]):
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
    if len(valid) >= min_valid_pairs:
        dk_arr = np.array([v[0] for v in valid])
        lw_arr = np.array([-np.log(v[1]) for v in valid])
        mask = np.isfinite(lw_arr) & np.isfinite(dk_arr)
        if mask.sum() >= min_valid_pairs:
            A_local_fit = float(np.polyfit(dk_arr[mask], lw_arr[mask], 1)[0])
        else:
            A_local_fit = float('nan')
    else:
        A_local_fit = float('nan')

    if not np.isnan(A_local_fit) and len(pair_results) >= 2:
        pred_wks = [float(np.exp(-A_local_fit * pr['delta_kappa_sqrt_deff']))
                    for pr in pair_results]
        emp_wks = [pr['empirical_wk'] for pr in pair_results]
        if len(pred_wks) >= 2 and np.std(pred_wks) > 1e-6:
            r_wk = float(pearsonr(emp_wks, pred_wks)[0])
        else:
            r_wk = 0.0
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
    log("MULTI-K SWEEP: B(K) = C_0 * sqrt(K-1) / kappa^p")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log(f"Fixed kappa={KAPPA}, d={D}, N_PER_CLASS={N_PER_CLASS}")
    log(f"K values: {K_VALUES}")
    log(f"alpha levels: {ALPHA_LEVELS}")
    log(f"N_SEEDS={N_SEEDS}, R_LEVELS={R_LEVELS}")
    log("=" * 70)
    log("PRE-REGISTERED (Feb 23, 2026 -- committed BEFORE running):")
    log(f"  C_0 = sqrt(2/3) = {C_0:.4f}, p = {P_PRIOR:.3f}")
    for K in K_VALUES:
        log(f"  B_pred(K={K:2d}) = {b_pred(K):.3f}  [sqrt(K-1)={np.sqrt(K-1):.3f}]")
    log("  PR1: r(B_K, sqrt(K-1)) > 0.97")
    log("  PR2: |slope_obs/slope_pred - 1| < 0.15 where slope_pred = C_0/kappa^p")
    log("  PR3: CV of B_K/sqrt(K-1) < 0.10 across all K values")
    log("  PR4: within-K CV(B) < 0.15 for >= 4/6 K values")
    log("  PRIMARY: PR1 + PR3 must both pass")
    log("")

    all_K_results = {}

    for K in K_VALUES:
        log(f"\n{'='*60}")
        log(f"K = {K}  [sqrt(K-1)={np.sqrt(K-1):.3f}]")
        log(f"{'='*60}")
        B_p = b_pred(K)
        log(f"  B_pred = {B_p:.3f}")

        # For K < 7, allow fitting with 2 valid pairs
        min_valid_pairs = 2 if K <= 5 else 3

        K_results = {}

        for alpha in ALPHA_LEVELS:
            log(f"\n  --- alpha={alpha} ---")
            Sigma, eigenvalues, _ = make_covariance_alpha(alpha, D)
            lambda_max = float(eigenvalues.max())
            d_eff_theoretical = float(D / lambda_max)
            log(f"  d_eff_theory={d_eff_theoretical:.3f}")

            seed_results = []
            for seed in SEEDS:
                try:
                    X, y, _ = generate_synthetic_embeddings(
                        K, D, N_PER_CLASS * 2, Sigma, KAPPA, seed)
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
                        X_tr, y_tr, X_te, y_te, classes, min_valid_pairs)
                except Exception as e:
                    log(f"    seed={seed}: EXCEPTION {type(e).__name__}: {e}")
                    continue

                if not baseline:
                    log(f"    seed={seed}: SKIP")
                    continue

                d_eff_1 = baseline['d_eff_1']
                B = (A_local_fit * np.sqrt(d_eff_1)
                     if not np.isnan(A_local_fit) else float('nan'))
                log(f"    seed={seed}: A={A_local_fit:.4f}, r_wk={r_wk:.3f}, "
                    f"B={B:.4f}, q_base={baseline['q_base']:.3f}")
                seed_results.append({
                    'seed': int(seed),
                    'A_local_fit': float(A_local_fit),
                    'r_wk': float(r_wk),
                    'B': float(B) if not np.isnan(B) else None,
                    'baseline': baseline,
                })

            if not seed_results:
                log(f"  K={K} alpha={alpha}: ALL SEEDS SKIP")
                continue

            valid_B = [s['B'] for s in seed_results
                       if s['B'] is not None and not np.isnan(s['B'])]
            valid_r = [s['r_wk'] for s in seed_results]
            mean_deff = float(np.mean([s['baseline']['d_eff_1'] for s in seed_results]))
            mean_B = float(np.mean(valid_B)) if valid_B else float('nan')
            std_B = float(np.std(valid_B)) if len(valid_B) > 1 else 0.0
            cv_B = std_B / abs(mean_B) if mean_B != 0 else float('nan')
            mean_r = float(np.mean(valid_r))
            mean_qbase = float(np.mean([s['baseline']['q_base'] for s in seed_results]))

            log(f"  SUMMARY K={K} alpha={alpha}: B={mean_B:.4f}+/-{std_B:.4f}(CV={cv_B:.3f}), "
                f"r_wk={mean_r:.3f}, d_eff={mean_deff:.3f}, q_base={mean_qbase:.3f}")

            K_results[alpha] = {
                'alpha': float(alpha),
                'mean_B': float(mean_B),
                'std_B': float(std_B),
                'cv_B': float(cv_B) if np.isfinite(cv_B) else None,
                'mean_r_wk': float(mean_r),
                'mean_d_eff_1': float(mean_deff),
                'mean_q_base': float(mean_qbase),
                'n_seeds': len(seed_results),
                'n_valid_B': len(valid_B),
            }

        if not K_results:
            log(f"\nK={K}: NO VALID RESULTS")
            continue

        # Aggregate across alpha levels (valid: r_wk >= 0.70)
        valid_alpha_Bs = [K_results[a]['mean_B'] for a in K_results
                          if K_results[a]['mean_r_wk'] >= 0.70
                          and not np.isnan(K_results[a]['mean_B'])]
        B_K_mean = float(np.mean(valid_alpha_Bs)) if valid_alpha_Bs else float('nan')
        B_K_std = float(np.std(valid_alpha_Bs)) if len(valid_alpha_Bs) > 1 else 0.0
        B_K_cv = B_K_std / abs(B_K_mean) if B_K_mean != 0 else float('nan')
        n_valid_alpha = len(valid_alpha_Bs)

        B_normed = B_K_mean / np.sqrt(K - 1) if not np.isnan(B_K_mean) else float('nan')
        C_0_obs = B_K_mean * (KAPPA ** P_PRIOR) / np.sqrt(K - 1) if not np.isnan(B_K_mean) else float('nan')

        log(f"\n  K={K} AGGREGATE: B={B_K_mean:.4f}+/-{B_K_std:.4f}(CV={B_K_cv:.3f}), "
            f"B_pred={B_p:.4f}, rel_err={abs(B_K_mean-B_p)/B_p:.3f}, "
            f"B/sqrt(K-1)={B_normed:.4f}, C_0_obs={C_0_obs:.4f} (valid_alpha={n_valid_alpha}/3)")

        all_K_results[K] = {
            'K': int(K),
            'B_K_mean': float(B_K_mean),
            'B_K_std': float(B_K_std),
            'B_K_cv': float(B_K_cv) if np.isfinite(B_K_cv) else None,
            'B_pred_prior': float(B_p),
            'rel_err_prior': float(abs(B_K_mean - B_p) / B_p) if not np.isnan(B_K_mean) else None,
            'B_normed': float(B_normed) if np.isfinite(B_normed) else None,
            'C_0_obs': float(C_0_obs) if np.isfinite(C_0_obs) else None,
            'n_valid_alpha': int(n_valid_alpha),
            'alpha_results': K_results,
        }

    # ==================== ANALYSIS ====================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    Ks_with_data = [K for K in K_VALUES
                    if K in all_K_results
                    and not np.isnan(all_K_results[K]['B_K_mean'])
                    and all_K_results[K]['n_valid_alpha'] >= 1]

    log(f"\nK values with valid data: {len(Ks_with_data)}/{len(K_VALUES)}")
    log(f"\n{'K':>4}  {'sqrt(K-1)':>9}  {'B_obs':>8}  {'B_pred':>8}  "
        f"{'rel_err':>8}  {'B/sqrt(K-1)':>11}  {'C_0_obs':>7}  {'CV_intra':>8}")
    for K in K_VALUES:
        if K in all_K_results:
            r = all_K_results[K]
            cv_str = f"{r['B_K_cv']:.3f}" if r['B_K_cv'] is not None else "  nan"
            err_str = f"{r['rel_err_prior']:.3f}" if r['rel_err_prior'] is not None else "  nan"
            log(f"  {K:>2}  {np.sqrt(K-1):>9.3f}  {r['B_K_mean']:>8.4f}  "
                f"{r['B_pred_prior']:>8.4f}  {err_str:>8}  "
                f"{(r['B_normed'] or 0):>11.4f}  {(r['C_0_obs'] or 0):>7.4f}  {cv_str:>8}")

    analysis = {}
    if len(Ks_with_data) < 4:
        log("INSUFFICIENT DATA")
        verdict = "INSUFFICIENT"
        analysis['verdict'] = verdict
    else:
        sqrt_Km1 = np.array([np.sqrt(K - 1) for K in Ks_with_data])
        B_obs = np.array([all_K_results[K]['B_K_mean'] for K in Ks_with_data])

        # PR1: r(B_K, sqrt(K-1)) > 0.97
        r_B_sqrtK = float(pearsonr(sqrt_Km1, B_obs)[0])
        pr1 = r_B_sqrtK > 0.97
        log(f"\nPR1 r(B_K, sqrt(K-1)) = {r_B_sqrtK:.4f} -> {'PASS' if pr1 else 'FAIL'} (>0.97)")
        analysis['PR1'] = bool(pr1)
        analysis['r_B_sqrtK'] = float(r_B_sqrtK)

        # Linear regression B = slope * sqrt(K-1) through origin
        slope_obs = float(np.sum(sqrt_Km1 * B_obs) / np.sum(sqrt_Km1 ** 2))
        slope_pred = C_0 / (KAPPA ** P_PRIOR)
        pr2 = abs(slope_obs / slope_pred - 1) < 0.15
        log(f"\nPR2 slope_obs={slope_obs:.4f}, slope_pred={slope_pred:.4f}, "
            f"ratio={slope_obs/slope_pred:.4f} -> {'PASS' if pr2 else 'FAIL'} (ratio within 15%)")
        analysis['PR2'] = bool(pr2)
        analysis['slope_obs'] = float(slope_obs)
        analysis['slope_pred'] = float(slope_pred)
        analysis['C_0_obs'] = float(slope_obs * (KAPPA ** P_PRIOR))

        # PR3: CV of B_K/sqrt(K-1) < 0.10
        B_normed_vals = B_obs / sqrt_Km1
        cv_normed = float(np.std(B_normed_vals) / abs(np.mean(B_normed_vals)))
        pr3 = cv_normed < 0.10
        log(f"\nPR3 CV of B/sqrt(K-1) = {cv_normed:.4f} -> {'PASS' if pr3 else 'FAIL'} (<0.10)")
        log(f"   Mean B/sqrt(K-1) = {np.mean(B_normed_vals):.4f} (C_0/kappa^p = {C_0/KAPPA**P_PRIOR:.4f})")
        log(f"   C_0_obs = {float(np.mean(B_normed_vals) * KAPPA**P_PRIOR):.4f} vs C_0_prior={C_0:.4f}")
        analysis['PR3'] = bool(pr3)
        analysis['cv_normed'] = float(cv_normed)
        analysis['mean_B_normed'] = float(np.mean(B_normed_vals))
        analysis['C_0_obs_mean'] = float(np.mean(B_normed_vals) * KAPPA ** P_PRIOR)

        # PR4: within-K CV < 0.15 for >= 4/6 K values
        n_clean = sum(
            1 for K in Ks_with_data
            if all_K_results[K]['B_K_cv'] is not None
            and all_K_results[K]['B_K_cv'] < 0.15
        )
        pr4 = n_clean >= 4
        log(f"\nPR4 K values with within-K CV < 0.15: {n_clean}/{len(Ks_with_data)}"
            f" -> {'PASS' if pr4 else 'FAIL'} (>= 4)")
        analysis['PR4'] = bool(pr4)
        analysis['n_clean_K'] = int(n_clean)

        primary_pass = pr1 and pr3
        n_secondary = sum([pr2, pr4])
        if primary_pass and n_secondary >= 1:
            verdict = "STRONG_PASS"
        elif primary_pass:
            verdict = "PASS"
        elif pr1 or pr3:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        log(f"\nPRIMARY (PR1+PR3): {'PASS' if primary_pass else 'FAIL'}")
        log(f"SECONDARY (PR2,PR4): {n_secondary}/2 pass")
        log(f"VERDICT: {verdict}")
        log(f"\nFitted universal: B(K, kappa) = {np.mean(B_normed_vals)*KAPPA**P_PRIOR:.3f} * sqrt(K-1) / {KAPPA}^{P_PRIOR}")
        log(f"Prior:           B(K, kappa) = {C_0:.3f} * sqrt(K-1) / kappa^{P_PRIOR}")
        log(f"C_0_obs / sqrt(2/3) = {float(np.mean(B_normed_vals) * KAPPA**P_PRIOR) / C_0:.4f}")

        analysis['verdict'] = verdict

    output = {
        'meta': {
            'kappa': float(KAPPA), 'd': int(D), 'N_PER_CLASS': int(N_PER_CLASS),
            'C_0_prior': float(C_0), 'p_prior': float(P_PRIOR),
            'timestamp': datetime.datetime.now().isoformat(),
        },
        'K_results': {str(K): v for K, v in all_K_results.items()},
        'analysis': analysis,
        'verdict': analysis.get('verdict', 'INCOMPLETE'),
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")
    _log_fh.close()


if __name__ == '__main__':
    main()
