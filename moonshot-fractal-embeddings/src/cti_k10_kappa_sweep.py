#!/usr/bin/env python -u
"""
K=10 MULTI-KAPPA SWEEP: B(kappa) = C_K10 / kappa^p
=====================================================
PURPOSE:
  Test kappa-scaling for K=10, independently of K=7.
  If C_K10 / sqrt(K-1) ≈ C_K7 / sqrt(K-1), confirms C_0 universality.

  From K=7 sweep (STRONG_PASS, r=-0.9925):
    C_K7 = 1.979, p = 0.791
    C_0 = C_K7 / sqrt(K-1) = 1.979 / sqrt(6) = 0.808

  From multi-K sweep (K=10 point):
    B(K=10, kappa=0.30) = 6.23 -> C_K10_obs = 6.23 * 0.30^0.791 = 2.33

  Universal law prediction (C_0 = sqrt(2/3)):
    C_K10 = C_0 * sqrt(9) = 0.816 * 3 = 2.449

  Empirical prediction (from K=7 sweep: C_0 = 0.808):
    C_K10_empirical = 0.808 * 3 = 2.424

DESIGN:
  - K=10, d=768, N_PER_CLASS=200
  - kappa values: [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
  - alpha levels: [0.75, 1.0, 1.25] (3 levels in clean valid range)
  - Seeds: 10 per (kappa, alpha) combination
  - Total: 6 kappa * 3 alpha * 10 seeds = 180 runs

PRE-REGISTERED (Feb 23, 2026 -- committed BEFORE running):
  Prior: C_K10 = 2.449 = sqrt(2/3) * sqrt(9) = 0.816 * 3, p = 0.791
  (from K=7 STRONG_PASS fit C_0=0.808, rounded to sqrt(2/3)=0.816)

  Predicted B(kappa) = 2.449 / kappa^0.791:
    B_pred(0.15) = 2.449 / 0.15^0.791 = 2.449 / 0.2165 = 11.31
    B_pred(0.20) = 2.449 / 0.20^0.791 = 2.449 / 0.273  = 8.97
    B_pred(0.25) = 2.449 / 0.25^0.791 = 2.449 / 0.332  = 7.38
    B_pred(0.30) = 2.449 / 0.30^0.791 = 2.449 / 0.374  = 6.55
    B_pred(0.40) = 2.449 / 0.40^0.791 = 2.449 / 0.460  = 5.32
    B_pred(0.50) = 2.449 / 0.50^0.791 = 2.449 / 0.604  = 4.05

  Cross-check vs multi-K sweep: B(kappa=0.30) = 6.23 obs vs 6.55 pred (4.9% below)

  PR1: For each kappa, |B_obs - B_pred| / B_pred < 0.30 (within 30%)
  PR2: log-log slope of B vs kappa in [-0.90, -0.55] (p in [0.55, 0.90])
  PR3: CV of C-normalized B values < 0.15 (tight law)
  PR4: within-kappa CV of B (across 3 alpha levels) < 0.20 for >= 4/6 kappa values
  PR5: Pearson r(log kappa, log B_obs) < -0.90 (strong negative correlation)

  PRIMARY: PR5 + PR2 must both pass
  SECONDARY: PR3, PR4, PR1

  ADDITIONAL CRITERION:
  C_0_test: |C_K10_fit / sqrt(9) - C_0_prior| / C_0_prior < 0.15
  where C_0_prior = 0.816 (=sqrt(2/3))
  This directly tests the C_0 universality claim.
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_k10_kappa_sweep.json"
LOG_PATH = "results/cti_k10_kappa_sweep_log.txt"

K = 10
N_PER_CLASS = 200
D = 768
N_SEEDS = 10
SEEDS = [42, 137, 271, 919, 2345, 7777, 8888, 9999, 11111, 12345]
R_LEVELS = [2.0, 3.0, 5.0, 7.0, 10.0]
N_PAIRS = 20  # K*(K-1)/2 = 45 total; use 20 for efficiency

# Kappa values to test (same range as K=7 sweep)
KAPPA_VALUES = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Alpha levels: 3 in the "clean" valid range from prior experiments
ALPHA_LEVELS = [0.75, 1.0, 1.25]

# Prior predictions: C_K10 = sqrt(2/3) * sqrt(K-1) = 0.816 * 3 = 2.449, p = 0.791
C_K10_PRIOR = float(np.sqrt(2.0 / 3.0) * np.sqrt(K - 1))   # = 2.449
P_PRIOR = 0.791
C_0_PRIOR = float(np.sqrt(2.0 / 3.0))   # = 0.8165

# Also track vs empirical C_0 from K=7
C_0_K7_empirical = 0.808  # from K=7 STRONG_PASS fit

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
    # K-2 extra classes at linspace(1.5, 4.0) * delta_min
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
        try:
            r_k = float(pearsonr(d_obs, d_pred)[0])
        except Exception:
            r_k = float('nan')
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
        mask = np.isfinite(lw_arr) & np.isfinite(dk_arr)
        if mask.sum() >= 3:
            A_local_fit = float(np.polyfit(dk_arr[mask], lw_arr[mask], 1)[0])
        else:
            A_local_fit = float('nan')
    else:
        A_local_fit = float('nan')

    if not np.isnan(A_local_fit) and len(pair_results) >= 3:
        pred_wks = [float(np.exp(-A_local_fit * pr['delta_kappa_sqrt_deff']))
                    for pr in pair_results]
        emp_wks = [pr['empirical_wk'] for pr in pair_results]
        r_wk = float(pearsonr(emp_wks, pred_wks)[0]) if np.std(pred_wks) > 1e-6 else 0.0
    else:
        r_wk = 0.0

    baseline = {
        'acc_base': float(acc_base), 'q_base': float(q_base),
        'logit_base': float(logit_base),
        'kappa_1': float(kappa_1), 'd_eff_1': float(d_eff_1),
    }
    return A_local_fit, r_wk, baseline, pair_results


def b_pred_from_prior(kappa):
    """Predict B = A_local * sqrt(d_eff_1) for given kappa using C_K10 prior."""
    return C_K10_PRIOR / (kappa ** P_PRIOR)


def main():
    log("=" * 70)
    log("K=10 MULTI-KAPPA SWEEP: B(kappa) = C_K10 / kappa^p")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log(f"d={D}, K={K}, N_PER_CLASS={N_PER_CLASS}")
    log(f"kappa values: {KAPPA_VALUES}")
    log(f"alpha levels: {ALPHA_LEVELS}")
    log(f"N_SEEDS={N_SEEDS}, R_LEVELS={R_LEVELS}")
    log("=" * 70)
    log("PRE-REGISTERED (Feb 23, 2026 -- committed BEFORE running):")
    log(f"  Prior: C_K10 = sqrt(2/3)*sqrt(9) = {C_K10_PRIOR:.4f}, p = {P_PRIOR}")
    log(f"  C_0_prior = sqrt(2/3) = {C_0_PRIOR:.4f}")
    log(f"  C_0_K7_empirical = {C_0_K7_empirical:.4f} (from K=7 STRONG_PASS)")
    log(f"  Cross-check: multi-K sweep gave B(kappa=0.30,K=10)=6.23, pred={b_pred_from_prior(0.30):.2f}")
    for k in KAPPA_VALUES:
        log(f"  B_pred(kappa={k:.2f}) = {b_pred_from_prior(k):.3f}")
    log("  PR1: |B_obs - B_pred| / B_pred < 0.30 for each kappa (within 30%)")
    log("  PR2: log-log slope of B vs kappa in [-0.90, -0.55]")
    log("  PR3: CV of C-normalized B values < 0.15 (tight law)")
    log("  PR4: within-kappa CV(B) < 0.20 for >= 4/6 kappa values")
    log("  PR5: r(log kappa, log B_obs) < -0.90 (strong negative correlation)")
    log("  C_0_test: |C_K10_fit/sqrt(9) - C_0_prior| / C_0_prior < 0.15")
    log("  PRIMARY: PR5 + PR2 must both pass")
    log("")

    all_kappa_results = {}

    for kappa in KAPPA_VALUES:
        log(f"\n{'='*60}")
        log(f"KAPPA = {kappa}")
        log(f"{'='*60}")
        B_pred = b_pred_from_prior(kappa)
        log(f"  B_pred(prior) = {B_pred:.3f}")

        kappa_results = {}

        for alpha in ALPHA_LEVELS:
            log(f"\n  --- alpha={alpha} ---")
            Sigma, eigenvalues, _ = make_covariance_alpha(alpha, D)
            lambda_max = float(eigenvalues.max())
            d_eff_theoretical = float(D / lambda_max)
            log(f"  lambda_max={lambda_max:.4f}, d_eff_theory={d_eff_theoretical:.3f}")

            seed_results = []
            for seed in SEEDS:
                try:
                    X, y, _ = generate_synthetic_embeddings(
                        K, D, N_PER_CLASS * 2, Sigma, kappa, seed)
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
                    log(f"    seed={seed}: EXCEPTION {type(e).__name__}: {e}")
                    continue

                if not baseline:
                    log(f"    seed={seed}: SKIP")
                    continue

                d_eff_1 = baseline['d_eff_1']
                B = (A_local_fit * np.sqrt(d_eff_1)
                     if not np.isnan(A_local_fit) else float('nan'))
                log(f"    seed={seed}: A_obs={A_local_fit:.4f}, r_wk={r_wk:.3f}, "
                    f"B={B:.4f}, q_base={baseline['q_base']:.3f}, "
                    f"d_eff_1={d_eff_1:.3f}")
                seed_results.append({
                    'seed': int(seed),
                    'A_local_fit': float(A_local_fit),
                    'r_wk': float(r_wk),
                    'B': float(B) if not np.isnan(B) else None,
                    'baseline': baseline,
                })

            if not seed_results:
                log(f"  alpha={alpha}: ALL SEEDS SKIP")
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

            log(f"  SUMMARY kappa={kappa} alpha={alpha}: "
                f"B={mean_B:.4f}+/-{std_B:.4f}(CV={cv_B:.3f}), "
                f"r_wk={mean_r:.3f}, d_eff={mean_deff:.3f}, q_base={mean_qbase:.3f}")

            kappa_results[alpha] = {
                'alpha': float(alpha),
                'mean_B': float(mean_B),
                'std_B': float(std_B),
                'cv_B': float(cv_B) if np.isfinite(cv_B) else None,
                'mean_r_wk': float(mean_r),
                'mean_d_eff_1': float(mean_deff),
                'mean_q_base': float(mean_qbase),
                'n_seeds': len(seed_results),
                'n_valid_B': len(valid_B),
                'seed_results': seed_results,
            }

        if not kappa_results:
            log(f"\nkappa={kappa}: NO VALID RESULTS")
            continue

        # Aggregate across alpha levels for this kappa (valid = r_wk >= 0.70)
        valid_alpha_Bs = [kappa_results[a]['mean_B'] for a in kappa_results
                          if kappa_results[a]['mean_r_wk'] >= 0.70
                          and not np.isnan(kappa_results[a]['mean_B'])]
        B_kappa_mean = float(np.mean(valid_alpha_Bs)) if valid_alpha_Bs else float('nan')
        B_kappa_std = float(np.std(valid_alpha_Bs)) if len(valid_alpha_Bs) > 1 else 0.0
        B_kappa_cv = B_kappa_std / abs(B_kappa_mean) if B_kappa_mean != 0 else float('nan')
        n_valid_alpha = len(valid_alpha_Bs)

        # C_0_obs: C_K10 observed / sqrt(K-1)
        C_K10_obs = B_kappa_mean * (kappa ** P_PRIOR) if not np.isnan(B_kappa_mean) else float('nan')
        C_0_obs = C_K10_obs / np.sqrt(K - 1) if not np.isnan(C_K10_obs) else float('nan')

        log(f"\n  KAPPA={kappa} AGGREGATE: B={B_kappa_mean:.4f}+/-{B_kappa_std:.4f}"
            f"(CV={B_kappa_cv:.3f}), B_pred={B_pred:.4f}, "
            f"rel_err={abs(B_kappa_mean-B_pred)/B_pred:.3f}, "
            f"C_K10={C_K10_obs:.4f}, C_0_obs={C_0_obs:.4f} "
            f"(valid_alpha={n_valid_alpha}/3)")

        all_kappa_results[kappa] = {
            'kappa': float(kappa),
            'B_kappa_mean': float(B_kappa_mean),
            'B_kappa_std': float(B_kappa_std),
            'B_kappa_cv': float(B_kappa_cv) if np.isfinite(B_kappa_cv) else None,
            'B_pred_prior': float(B_pred),
            'rel_err_prior': float(abs(B_kappa_mean - B_pred) / B_pred)
                             if not np.isnan(B_kappa_mean) else None,
            'C_K10_obs': float(C_K10_obs) if not np.isnan(C_K10_obs) else None,
            'C_0_obs': float(C_0_obs) if not np.isnan(C_0_obs) else None,
            'n_valid_alpha': int(n_valid_alpha),
            'alpha_results': kappa_results,
        }

    # ==================== ANALYSIS ====================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    kappas_with_data = [k for k in KAPPA_VALUES
                        if k in all_kappa_results
                        and all_kappa_results[k]['B_kappa_mean'] is not None
                        and not np.isnan(all_kappa_results[k]['B_kappa_mean'])
                        and all_kappa_results[k]['n_valid_alpha'] >= 1]

    log(f"\nKappa values with valid data: {len(kappas_with_data)}/{len(KAPPA_VALUES)}")
    log(f"\n{'kappa':>8}  {'B_obs':>8}  {'B_pred':>8}  {'rel_err':>8}  "
        f"{'C_K10':>6}  {'C_0_obs':>7}  {'CV_B':>6}  {'n_valid':>7}")
    for kappa in KAPPA_VALUES:
        if kappa in all_kappa_results:
            r = all_kappa_results[kappa]
            cv_str = f"{r['B_kappa_cv']:.3f}" if r['B_kappa_cv'] is not None else "  nan"
            err_str = f"{r['rel_err_prior']:.3f}" if r['rel_err_prior'] is not None else "  nan"
            c0_str = f"{r['C_0_obs']:.4f}" if r['C_0_obs'] is not None else "   nan"
            ck_str = f"{r['C_K10_obs']:.3f}" if r['C_K10_obs'] is not None else "  nan"
            log(f"  {kappa:>6.2f}  {r['B_kappa_mean']:>8.4f}  "
                f"{r['B_pred_prior']:>8.4f}  {err_str:>8}  "
                f"{ck_str:>6}  {c0_str:>7}  {cv_str:>6}  {r['n_valid_alpha']:>7}")

    analysis = {}
    if len(kappas_with_data) < 3:
        log("INSUFFICIENT DATA for analysis (need >= 3 kappa values)")
        verdict = "INSUFFICIENT"
        analysis['verdict'] = verdict
    else:
        log_kappas = np.array([np.log(k) for k in kappas_with_data])
        log_Bs = np.array([np.log(all_kappa_results[k]['B_kappa_mean'])
                           for k in kappas_with_data])

        # PR5: r(log kappa, log B) < -0.90
        r_kappa_B = float(pearsonr(log_kappas, log_Bs)[0])
        pr5 = r_kappa_B < -0.90
        log(f"\nPR5 r(log kappa, log B) = {r_kappa_B:.4f} -> {'PASS' if pr5 else 'FAIL'} (<-0.90)")
        analysis['PR5'] = bool(pr5)
        analysis['r_kappa_B'] = float(r_kappa_B)

        # PR2: slope in [-0.90, -0.55]
        slope, intercept = np.polyfit(log_kappas, log_Bs, 1)
        pr2 = -0.90 <= slope <= -0.55
        C_fit = np.exp(intercept)
        p_fit = -slope
        log(f"\nPR2 slope = {slope:.4f} (p_fit={p_fit:.3f}), C_K10_fit = {C_fit:.4f} -> "
            f"{'PASS' if pr2 else 'FAIL'} (slope in [-0.90,-0.55])")
        log(f"   C_K10_fit vs C_K10_prior={C_K10_PRIOR:.4f}: "
            f"rel_err={abs(C_fit-C_K10_PRIOR)/C_K10_PRIOR:.3f}")
        C_0_fit = C_fit / np.sqrt(K - 1)
        log(f"   C_0_fit = C_K10_fit/sqrt(K-1) = {C_fit:.4f}/sqrt({K-1}) = {C_0_fit:.4f}")
        log(f"   C_0_prior (sqrt(2/3)) = {C_0_PRIOR:.4f}, "
            f"ratio C_0_fit/C_0_prior = {C_0_fit/C_0_PRIOR:.4f}")
        log(f"   C_0_K7_empirical = {C_0_K7_empirical:.4f}, "
            f"ratio = {C_0_fit/C_0_K7_empirical:.4f}")
        analysis['PR2'] = bool(pr2)
        analysis['slope'] = float(slope)
        analysis['p_fit'] = float(p_fit)
        analysis['C_K10_fit'] = float(C_fit)
        analysis['C_0_fit'] = float(C_0_fit)

        # C_0 universality test
        c0_rel_err = abs(C_0_fit - C_0_PRIOR) / C_0_PRIOR
        c0_pass = c0_rel_err < 0.15
        log(f"\nC_0_test: |C_0_fit - C_0_prior|/C_0_prior = {c0_rel_err:.4f} -> "
            f"{'PASS' if c0_pass else 'FAIL'} (<0.15)")
        analysis['C_0_test'] = bool(c0_pass)
        analysis['C_0_rel_err'] = float(c0_rel_err)

        # PR3: CV of C-normalized B values < 0.15
        C_normalized = [all_kappa_results[k]['B_kappa_mean'] * (k ** p_fit)
                        for k in kappas_with_data]
        cv_C = float(np.std(C_normalized) / abs(np.mean(C_normalized)))
        pr3 = cv_C < 0.15
        log(f"\nPR3 CV of C-normalized B = {cv_C:.4f} -> {'PASS' if pr3 else 'FAIL'} (<0.15)")
        analysis['PR3'] = bool(pr3)
        analysis['cv_C_normalized'] = float(cv_C)

        # PR4: within-kappa CV(B) < 0.20 for >= 4/6 kappa values
        n_clean_kappa = sum(
            1 for k in kappas_with_data
            if all_kappa_results[k]['B_kappa_cv'] is not None
            and all_kappa_results[k]['B_kappa_cv'] < 0.20
        )
        pr4 = n_clean_kappa >= 4
        log(f"\nPR4 kappa values with within-kappa CV < 0.20: "
            f"{n_clean_kappa}/{len(kappas_with_data)} -> {'PASS' if pr4 else 'FAIL'} (>= 4/6)")
        analysis['PR4'] = bool(pr4)
        analysis['n_clean_kappa'] = int(n_clean_kappa)

        # PR1: each kappa within 30% of prior prediction
        n_pr1 = sum(
            1 for k in kappas_with_data
            if all_kappa_results[k]['rel_err_prior'] is not None
            and all_kappa_results[k]['rel_err_prior'] < 0.30
        )
        pr1 = n_pr1 >= len(kappas_with_data) * 0.67
        log(f"\nPR1 kappa values within 30% of prior: "
            f"{n_pr1}/{len(kappas_with_data)} -> {'PASS' if pr1 else 'FAIL'} (need >= 2/3)")
        analysis['PR1'] = bool(pr1)
        analysis['n_pr1_pass'] = int(n_pr1)

        # C_0 values per kappa (universality check)
        c0_values = [all_kappa_results[k]['C_0_obs'] for k in kappas_with_data
                     if all_kappa_results[k]['C_0_obs'] is not None]
        c0_mean = float(np.mean(c0_values))
        c0_cv = float(np.std(c0_values) / c0_mean) if c0_mean > 0 else float('nan')
        log(f"\nC_0_obs per kappa: {[f'{v:.4f}' for v in c0_values]}")
        log(f"  mean C_0_obs = {c0_mean:.4f}, CV = {c0_cv:.4f}")
        log(f"  vs C_0_prior=sqrt(2/3)={C_0_PRIOR:.4f}, "
            f"vs C_0_K7_empirical={C_0_K7_empirical:.4f}")
        analysis['C_0_values_per_kappa'] = c0_values
        analysis['C_0_mean'] = float(c0_mean)
        analysis['C_0_cv'] = float(c0_cv) if np.isfinite(c0_cv) else None

        # Verdict: PRIMARY = PR5 + PR2 both pass
        primary_pass = pr5 and pr2
        n_secondary = sum([pr3, pr4, pr1, c0_pass])
        if primary_pass and n_secondary >= 3:
            verdict = "STRONG_PASS"
        elif primary_pass and n_secondary >= 2:
            verdict = "PASS"
        elif primary_pass:
            verdict = "PARTIAL_PASS"
        elif pr5 or pr2:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        log(f"\nPRIMARY (PR5+PR2): {'PASS' if primary_pass else 'FAIL'}")
        log(f"SECONDARY (PR1,PR3,PR4,C_0_test): {n_secondary}/4 pass")
        log(f"VERDICT: {verdict}")
        log(f"\nFitted law: B(kappa) = {C_fit:.3f} / kappa^{p_fit:.3f}")
        log(f"Prior law:  B(kappa) = {C_K10_PRIOR:.3f} / kappa^{P_PRIOR:.3f}")
        log(f"Universal: B(kappa,K) = {C_0_fit:.4f} * sqrt(K-1) / kappa^{p_fit:.3f}")
        log(f"           (K=7: B_fit = {C_0_fit*np.sqrt(6):.3f} vs K=7 C_fit=1.979)")
        log(f"           (K=14: B_fit = {C_0_fit*np.sqrt(13):.3f} vs K=14 C_fit=2.96)")

        analysis['verdict'] = verdict

    output = {
        'meta': {
            'K': int(K), 'd': int(D), 'N_PER_CLASS': int(N_PER_CLASS),
            'C_K10_prior': float(C_K10_PRIOR),
            'C_0_prior': float(C_0_PRIOR),
            'p_prior': float(P_PRIOR),
            'timestamp': datetime.datetime.now().isoformat(),
        },
        'kappa_results': {str(k): v for k, v in all_kappa_results.items()},
        'analysis': analysis,
        'verdict': analysis.get('verdict', 'INCOMPLETE'),
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")
    _log_fh.close()


if __name__ == "__main__":
    main()
