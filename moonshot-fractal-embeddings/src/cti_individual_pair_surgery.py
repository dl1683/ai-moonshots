#!/usr/bin/env python -u
"""
INDIVIDUAL-PAIR SURGERY: Correct w_k test
===========================================
PURPOSE: Properly test the multi-competitor CTI law by measuring individual
         pair weights w_k through single-pair surgery.

FIX: The previous synthetic_wk.py used slope(m)/slope(1) which is CUMULATIVE.
     The correct empirical w_k requires single-pair surgery for each pair k:
     - Apply surgery to ONLY pair k (not a bundle)
     - slope_k = OLS(delta_logit, delta_pred) where delta_pred from pair 1
     - empirical_wk = slope_k * kappa_1 * sqrt(d_eff_1) / (kappa_k * sqrt(d_eff_k))

THEORY:
  logit(q) = A * sum_k w_k * kappa_k * sqrt(d_eff_k) + C
  Single-pair surgery on pair k: scale kappa_k by sqrt(r), others fixed
  delta_logit = A * w_k * kappa_k * sqrt(d_eff_k) * (sqrt(r) - 1)
  If delta_pred = A * kappa_1 * sqrt(d_eff_1) * (sqrt(r) - 1) [from pair 1 only]:
  slope_k = w_k * kappa_k * sqrt(d_eff_k) / (kappa_1 * sqrt(d_eff_1))
  => empirical_wk = slope_k / (kappa_k * sqrt(d_eff_k)) * (kappa_1 * sqrt(d_eff_1))

PREDICTED:
  predicted_wk = exp(-A_local * (kappa_k - kappa_1) * sqrt(d_eff_1))
  where A_local = 1.75 (empirical from rank-spectrum session 24)

PRE-REGISTERED (Feb 24, 2026):
  W1_ind: Pearson r(predicted_wk, empirical_wk) > 0.85 for pythia-160m
  W2_ind: empirical_wk monotonically decreasing with k (rank correct)
  W3_ind: A_local_fit = -log(wk)/(delta_kappa * sqrt(d_eff)) CV < 0.30
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_individual_pair_surgery.json"
LOG_PATH = "results/cti_individual_pair_surgery_log.txt"

EMBED_PATH = "results/dointerv_multi_pythia-160m_l12.npz"
A_LOCAL = 1.75       # pre-registered
R_LEVELS = [0.3, 0.5, 2.0, 5.0, 10.0]
N_TRAIN_PER_CLASS = 350
N_PAIRS_TO_TEST = 12  # test top-12 pairs (K-2, excluding trivially distant ones)
N_SEEDS = 5
SEEDS = [42, 137, 271, 919, 2345]

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
    """Apply surgery to ONLY pair k (single pair), compensate in null space."""
    mu = geo['mu']
    classes = np.unique(y_tr)
    trW = geo['trW']

    # U_pair: the direction of this single pair
    Delta_hat = pair['Delta_hat']
    U_pair = Delta_hat.reshape(-1, 1)  # (d, 1)

    tr_pair = compute_subspace_trace(X_tr, y_tr, mu, classes, U_pair)

    # Compensation goes into NULL SPACE (not between-class) to keep total trW fixed
    tr_B_total = compute_subspace_trace(X_tr, y_tr, mu, classes, U_B)
    tr_W_null = trW - tr_B_total

    # After scaling pair by 1/sqrt(r), null space gets compensated
    tr_null_new = tr_W_null + tr_pair * (1.0 - 1.0 / r)
    scale_null = float(np.sqrt(tr_null_new / tr_W_null)) if tr_W_null > 1e-12 and tr_null_new > 0 else 1.0
    scale_pair = 1.0 / float(np.sqrt(r))

    def transform(X, y):
        X_new = X.copy()
        for i, c in enumerate(classes):
            mask = (y == c)
            z = X[mask] - mu[i]
            # Decompose: pair direction, rest of between-class, null space
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


def run_individual_pair_surgery(X_tr, y_tr, X_te, y_te, classes, n_pairs):
    """For each of the top-n_pairs, run single-pair surgery, extract slope_k."""
    K = len(classes)
    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)
    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])

    # Reference predictions from pair 1
    kappa_1 = geo['pair_info'][0]['kappa']
    d_eff_1 = geo['pair_info'][0]['d_eff']
    kappa_eff_1 = geo['pair_info'][0]['kappa_eff']

    # Pre-registered prediction based on single-pair 1 formula
    A_keff = 1.477 / float(np.sqrt((K - 1) * d_eff_1))  # A = ALPHA/sqrt((K-1)*d_eff)
    C_keff = logit_base - A_keff * kappa_eff_1
    delta_preds_by_r = {}
    for r in R_LEVELS:
        kappa_eff_r = kappa_eff_1 * float(np.sqrt(r))
        delta_preds_by_r[r] = float(C_keff + A_keff * kappa_eff_r) - logit_base

    pair_results = []
    for k_idx, pair in enumerate(geo['pair_info'][:n_pairs]):
        kappa_k = pair['kappa']
        d_eff_k = pair['d_eff']
        kappa_eff_k = pair['kappa_eff']

        # Single-pair surgery
        single_results = []
        for r in R_LEVELS:
            X_tr_r, X_te_r = apply_single_pair_surgery(
                X_tr, X_te, y_tr, y_te, geo, pair, U_B, r)
            _, _, logit_r = eval_q(X_tr_r, y_tr, X_te_r, y_te, K)
            delta_obs = float(logit_r - logit_base)
            single_results.append({'r': float(r), 'delta_logit': delta_obs})

        # OLS slope: delta_obs = slope_k * delta_pred_1
        nontrivial = [x for x in single_results if abs(x['r'] - 1.0) > 0.01]
        if len(nontrivial) < 3:
            continue
        d_obs = [x['delta_logit'] for x in nontrivial]
        d_pred = [delta_preds_by_r[x['r']] for x in nontrivial]
        if np.std(d_pred) < 1e-8:
            continue

        slope_k = float(np.polyfit(d_pred, d_obs, 1)[0])
        r_k = float(pearsonr(d_obs, d_pred)[0])

        # Empirical w_k: normalize slope by competitive strength
        # slope_k = w_k * kappa_k * sqrt(d_eff_k) / (kappa_1 * sqrt(d_eff_1))
        # => w_k = slope_k * kappa_1 * sqrt(d_eff_1) / (kappa_k * sqrt(d_eff_k))
        norm_factor = float(kappa_1 * np.sqrt(d_eff_1) / (kappa_k * np.sqrt(d_eff_k) + 1e-10))
        empirical_wk = float(slope_k * norm_factor)

        # Predicted w_k
        delta_kappa = kappa_k - kappa_1
        predicted_wk = float(np.exp(-A_LOCAL * delta_kappa * np.sqrt(d_eff_1)))

        pair_results.append({
            'k': k_idx + 1,
            'pair_i': pair['i'], 'pair_j': pair['j'],
            'kappa_k': float(kappa_k), 'd_eff_k': float(d_eff_k),
            'slope_k': float(slope_k), 'r_k': float(r_k),
            'empirical_wk': float(empirical_wk),
            'predicted_wk': float(predicted_wk),
            'delta_kappa': float(delta_kappa),
        })

    return pair_results, {
        'acc_base': float(acc_base), 'q_base': float(q_base), 'logit_base': float(logit_base),
        'kappa_1': float(kappa_1), 'd_eff_1': float(d_eff_1),
    }


def main():
    log("=" * 70)
    log("INDIVIDUAL-PAIR SURGERY: Correct w_k measurement")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log("=" * 70)
    log("PRE-REGISTERED:")
    log("  W1_ind: Pearson r(predicted_wk, empirical_wk) > 0.85")
    log("  W2_ind: empirical_wk monotonically non-increasing")
    log("  W3_ind: A_local_fit CV < 0.30 across seeds")
    log(f"\nArch: pythia-160m, N_seeds={N_SEEDS}, N_pairs={N_PAIRS_TO_TEST}")

    all_seed_results = []

    for seed in SEEDS:
        log(f"\n--- seed={seed} ---")
        X_tr, y_tr, X_te, y_te, classes = load_and_split(EMBED_PATH, N_TRAIN_PER_CLASS, seed)
        K = len(classes)
        log(f"  K={K}, n_train={len(X_tr)}, n_test={len(X_te)}")

        pair_results, baseline_info = run_individual_pair_surgery(
            X_tr, y_tr, X_te, y_te, classes, N_PAIRS_TO_TEST)

        if not pair_results:
            log(f"  SKIP (no valid results)")
            continue

        log(f"  Baseline: acc={baseline_info['acc_base']:.4f}, "
            f"kappa_1={baseline_info['kappa_1']:.4f}, d_eff_1={baseline_info['d_eff_1']:.2f}")
        log(f"  {'k':>3} {'kappa_k':>8} {'slope_k':>8} {'emp_wk':>8} {'pred_wk':>8} {'r_k':>6}")
        for pr in pair_results:
            log(f"  {pr['k']:>3} {pr['kappa_k']:>8.4f} {pr['slope_k']:>8.4f} "
                f"{pr['empirical_wk']:>8.4f} {pr['predicted_wk']:>8.4f} {pr['r_k']:>6.3f}")

        # W1: Pearson r(predicted_wk, empirical_wk)
        emp_wks = [pr['empirical_wk'] for pr in pair_results]
        pred_wks = [pr['predicted_wk'] for pr in pair_results]
        if len(emp_wks) >= 3 and np.std(pred_wks) > 1e-6:
            r_wk = float(pearsonr(emp_wks, pred_wks)[0])
        else:
            r_wk = 0.0

        # W2: monotone decreasing
        is_monotone = all(emp_wks[i] >= emp_wks[i+1] - 0.05 for i in range(len(emp_wks)-1))

        # W3: fit A_local
        log_wks = [-np.log(max(w, 1e-6)) for w in emp_wks]
        delta_kappa_sqrt_d = [pr['delta_kappa'] * np.sqrt(baseline_info['d_eff_1']) for pr in pair_results]
        nonzero = [(lw, dk) for lw, dk in zip(log_wks, delta_kappa_sqrt_d) if dk > 1e-6]
        if len(nonzero) >= 3:
            lw_arr = np.array([x[0] for x in nonzero])
            dk_arr = np.array([x[1] for x in nonzero])
            A_local_fit = float(np.polyfit(dk_arr, lw_arr, 1)[0])
        else:
            A_local_fit = float('nan')

        log(f"  r_wk={r_wk:.3f} ({'PASS' if r_wk > 0.85 else 'FAIL'}), "
            f"monotone={'YES' if is_monotone else 'NO'}, A_local_fit={A_local_fit:.3f}")

        all_seed_results.append({
            'seed': seed,
            'pair_results': pair_results,
            'baseline': baseline_info,
            'r_wk': float(r_wk),
            'is_monotone': bool(is_monotone),
            'A_local_fit': float(A_local_fit),
            'pass_W1': bool(r_wk > 0.85),
        })

    # ==================== POOLED ANALYSIS ====================
    log("\n" + "=" * 70)
    log("POOLED ANALYSIS")
    log("=" * 70)

    if not all_seed_results:
        log("No results!")
        return

    n_W1 = sum(1 for r in all_seed_results if r['pass_W1'])
    n_W2 = sum(1 for r in all_seed_results if r['is_monotone'])
    A_locals = [r['A_local_fit'] for r in all_seed_results if not np.isnan(r['A_local_fit'])]
    mean_A_local = float(np.mean(A_locals)) if A_locals else float('nan')
    cv_A_local = float(np.std(A_locals) / (np.mean(A_locals) + 1e-10)) if len(A_locals) >= 2 else float('nan')

    log(f"W1 (r>0.85): {n_W1}/{len(all_seed_results)} seeds pass")
    log(f"W2 (monotone): {n_W2}/{len(all_seed_results)} seeds pass")
    log(f"W3 (A_local CV<0.30): A_local={mean_A_local:.3f}+/-{np.std(A_locals) if A_locals else 0:.3f}, "
        f"CV={'%.3f' % cv_A_local if not np.isnan(cv_A_local) else 'N/A'}")
    log(f"Pre-registered A_local = {A_LOCAL:.2f}")

    pass_W1 = n_W1 >= (len(all_seed_results) + 1) // 2  # majority
    pass_W2 = n_W2 >= (len(all_seed_results) + 1) // 2
    pass_W3 = not np.isnan(cv_A_local) and cv_A_local < 0.30

    verdict = (f"W1={'PASS' if pass_W1 else 'FAIL'}, "
               f"W2={'PASS' if pass_W2 else 'FAIL'}, "
               f"W3={'PASS' if pass_W3 else 'FAIL'}. "
               f"A_local_fit={mean_A_local:.3f} vs A_LOCAL={A_LOCAL}. "
               f"OVERALL: {'PASS' if pass_W1 and pass_W2 and pass_W3 else 'PARTIAL/FAIL'}")
    log(f"\nVERDICT: {verdict}")

    output = {
        'experiment': 'individual_pair_surgery',
        'timestamp': datetime.datetime.now().isoformat(),
        'seed_results': all_seed_results,
        'analysis': {
            'n_W1_pass': int(n_W1), 'n_W2_pass': int(n_W2),
            'pass_W1': bool(pass_W1), 'pass_W2': bool(pass_W2), 'pass_W3': bool(pass_W3),
            'mean_A_local': float(mean_A_local), 'cv_A_local': float(cv_A_local) if not np.isnan(cv_A_local) else None,
            'A_LOCAL_preregistered': A_LOCAL,
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
