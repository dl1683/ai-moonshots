#!/usr/bin/env python -u
"""
CHECKPOINT PAIR SURGERY: Regime comparison
==========================================
PURPOSE: Run individual pair surgery on pythia-160m at EARLY CHECKPOINT (step=512, q~0.63)
         vs late training (q~0.90) to test whether A_local is regime-dependent.

HYPOTHESIS: Near-ceiling q causes noisy pair surgery. At q~0.63 (linear regime),
            pair surgery should give:
            (a) Higher r_wk (cleaner signal)
            (b) More consistent A_local across seeds

PRE-REGISTERED (Feb 25, 2026):
  C1: r_wk > 0.70 for >= 3/5 seeds at step=512 (vs 0/5 at step=final)
  C2: A_local_fit more consistent: CV(A_local) < 0.50 at step=512
  C3: A_local × sqrt(d_eff_1) ≈ 4-5 at step=512 (B universality check)
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

EMBED_CHECKPOINT = "results/checkpoint_embs_pythia-160m_step512.npz"
EMBED_FINAL = "results/dointerv_multi_pythia-160m_l12.npz"
RESULT_PATH = "results/cti_checkpoint_pair_surgery.json"
LOG_PATH = "results/cti_checkpoint_pair_surgery_log.txt"

A_LOCAL_PREREG = 1.75
R_LEVELS = [0.3, 0.5, 2.0, 5.0, 10.0]
N_TRAIN_PER_CLASS = 350
N_PAIRS = 12
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
        predicted_wk_fixed = float(np.exp(-A_LOCAL_PREREG * delta_kappa_sqrt_d))

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
        return pair_results, {}, float('nan'), float('nan')

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

    # r_wk
    if len(emp_wks) >= 3 and np.std(pred_wks_fixed) > 1e-6:
        r_wk = float(pearsonr(emp_wks, pred_wks_fixed)[0])
    else:
        r_wk = 0.0

    baseline = {
        'acc_base': float(acc_base), 'q_base': float(q_base), 'logit_base': float(logit_base),
        'kappa_1': float(kappa_1), 'd_eff_1': float(d_eff_1),
    }
    return pair_results, baseline, float(A_local_fit), float(r_wk)


def run_one_checkpoint(label, embed_path):
    log(f"\n{'='*60}")
    log(f"CHECKPOINT: {label}")
    log(f"{'='*60}")
    all_seed_results = []
    for seed in SEEDS:
        log(f"  seed={seed} ...")
        X_tr, y_tr, X_te, y_te, classes = load_and_split(embed_path, N_TRAIN_PER_CLASS, seed)
        K = len(classes)
        pair_results, baseline, A_local_fit, r_wk = run_surgery(
            X_tr, y_tr, X_te, y_te, classes, N_PAIRS)
        if not pair_results:
            log(f"    SKIP")
            continue
        B = A_local_fit * np.sqrt(baseline['d_eff_1']) if not np.isnan(A_local_fit) else float('nan')
        log(f"    q_base={baseline['q_base']:.3f}, kappa_1={baseline['kappa_1']:.4f}, "
            f"d_eff_1={baseline['d_eff_1']:.2f}, A_local={A_local_fit:.3f}, r_wk={r_wk:.3f}, B={B:.3f}")
        all_seed_results.append({
            'seed': seed,
            'baseline': baseline,
            'A_local_fit': float(A_local_fit),
            'r_wk': float(r_wk),
            'B': float(B),
        })

    if not all_seed_results:
        return None

    A_locals = [s['A_local_fit'] for s in all_seed_results if not np.isnan(s['A_local_fit'])]
    Bs = [s['B'] for s in all_seed_results if not np.isnan(s['B'])]
    r_wks = [s['r_wk'] for s in all_seed_results]
    mean_A = float(np.mean(A_locals)) if A_locals else float('nan')
    cv_A = float(np.std(A_locals) / (np.mean(A_locals) + 1e-10)) if len(A_locals) >= 2 else float('nan')
    mean_B = float(np.mean(Bs)) if Bs else float('nan')
    cv_B = float(np.std(Bs) / (np.mean(Bs) + 1e-10)) if len(Bs) >= 2 else float('nan')
    mean_r_wk = float(np.mean(r_wks))
    n_pass_C1 = sum(1 for r in r_wks if r > 0.70)

    log(f"\n  SUMMARY {label}:")
    log(f"    A_local = {mean_A:.3f} (CV={cv_A:.3f})")
    log(f"    B = {mean_B:.3f} (CV={cv_B:.3f})")
    log(f"    r_wk = {mean_r_wk:.3f}")
    log(f"    C1 (r_wk>0.70): {n_pass_C1}/{len(all_seed_results)}")

    return {
        'label': label,
        'seed_results': all_seed_results,
        'summary': {
            'mean_A_local': float(mean_A), 'cv_A_local': float(cv_A) if not np.isnan(cv_A) else None,
            'mean_B': float(mean_B), 'cv_B': float(cv_B) if not np.isnan(cv_B) else None,
            'mean_r_wk': float(mean_r_wk),
            'n_pass_C1': int(n_pass_C1), 'n_seeds': len(all_seed_results),
        }
    }


def main():
    log("=" * 70)
    log("CHECKPOINT PAIR SURGERY: Regime comparison")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log("=" * 70)
    log("PRE-REGISTERED:")
    log("  C1: r_wk > 0.70 for >= 3/5 seeds at step=512 (vs 0/5 at step=final)")
    log("  C2: CV(A_local) < 0.50 at step=512 (vs 1.506 at step=final)")
    log("  C3: B = A_local * sqrt(d_eff_1) in [3.0, 7.0] at step=512")

    ckpt_result = run_one_checkpoint("step=512 (q~0.63)", EMBED_CHECKPOINT)
    final_result = run_one_checkpoint("final (q~0.90)", EMBED_FINAL)

    log(f"\n{'='*70}")
    log("COMPARISON")
    log(f"{'='*70}")

    if ckpt_result and final_result:
        ckpt_s = ckpt_result['summary']
        final_s = final_result['summary']
        pass_C1 = ckpt_s['n_pass_C1'] >= 3
        pass_C2 = ckpt_s.get('cv_A_local') is not None and ckpt_s['cv_A_local'] < 0.50
        pass_C3 = ckpt_s['mean_B'] is not None and not np.isnan(ckpt_s['mean_B']) and 3.0 <= ckpt_s['mean_B'] <= 7.0

        log(f"C1 (r_wk>0.70 for >=3/5 seeds): {'PASS' if pass_C1 else 'FAIL'} "
            f"({ckpt_s['n_pass_C1']}/5 at step=512, vs 0/5 final)")
        log(f"C2 (CV < 0.50): {'PASS' if pass_C2 else 'FAIL'} "
            f"(CV={ckpt_s.get('cv_A_local', 'N/A'):.3f} at step=512, vs 1.506 final)")
        log(f"C3 (B in [3,7]): {'PASS' if pass_C3 else 'FAIL'} (B={ckpt_s['mean_B']:.3f})")
        log(f"  step=512: A_local={ckpt_s['mean_A_local']:.3f}, B={ckpt_s['mean_B']:.3f}, r_wk={ckpt_s['mean_r_wk']:.3f}")
        log(f"  final:    A_local={final_s['mean_A_local']:.3f}, B={final_s['mean_B']:.3f}, r_wk={final_s['mean_r_wk']:.3f}")

        verdict = (f"C1={'PASS' if pass_C1 else 'FAIL'}, C2={'PASS' if pass_C2 else 'FAIL'}, "
                   f"C3={'PASS' if pass_C3 else 'FAIL'}. "
                   f"step512 B={ckpt_s['mean_B']:.3f} vs final B={final_s['mean_B']:.3f}")
    else:
        verdict = "INCOMPLETE"

    output = {
        'experiment': 'checkpoint_pair_surgery',
        'timestamp': datetime.datetime.now().isoformat(),
        'checkpoint': ckpt_result,
        'final': final_result,
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
