#!/usr/bin/env python -u
"""
MULTI-SEED CROSS-ARCHITECTURE ABCS
====================================
PURPOSE: Get proper error bars on K_eff universality across architectures.
         Addresses Codex critique: "no error bars on key results."

DESIGN:
  - Same 5 architectures as cross_arch_abcs.py
  - 5 random seeds per architecture (different train/test splits)
  - Report: K_eff_estimated mean +/- std per arch, pooled CV
  - Pre-registered: pooled CV of K_eff_estimated < 0.30

PRE-REGISTRATION (Feb 24, 2026):
  U1_multi: pooled CV(K_eff_estimated) across all archs/seeds < 0.30
  U2_multi: mean K_eff_estimated per arch in [6, 14]
  U3_multi: within-arch CV(K_eff_estimated) < 0.20 for >=4/5 archs
             (low within-arch variance = reproducible signal)
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

EMBED_FILES = [
    ("pythia-160m",       "results/dointerv_multi_pythia-160m_l12.npz"),
    ("bert-base-uncased", "results/dointerv_multi_bert-base-uncased_l10.npz"),
    ("electra-small",     "results/dointerv_multi_electra-small_l3.npz"),
    ("pythia-410m",       "results/dointerv_multi_pythia-410m_l3.npz"),
    ("rwkv-4-169m",       "results/dointerv_multi_rwkv-4-169m_l12.npz"),
]

RESULT_PATH = "results/cti_multiseed_cross_arch.json"
LOG_PATH = "results/cti_multiseed_cross_arch_log.txt"

ALPHA_KAPPA_K14 = 1.477
R_LEVELS = [0.3, 0.5, 2.0, 5.0, 10.0]
M_BUNDLE = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13]
N_TRAIN_PER_CLASS = 350
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


def compute_geometry(X_tr, y_tr, classes):
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
        return None
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


def run_abcs_one_seed(arch_name, path, seed):
    """Run full ABCS for one seed. Returns dict with slopes and K_eff."""
    X_tr, y_tr, X_te, y_te, classes = load_and_split(path, N_TRAIN_PER_CLASS, seed)
    K = len(classes)
    d = X_tr.shape[1]

    geo = compute_geometry(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)

    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])
    tr_bundle_all = compute_subspace_trace(X_tr, y_tr, geo['mu'], classes, U_B)
    tr_W_null = geo['trW'] - tr_bundle_all

    A_keff = ALPHA_KAPPA_K14 / float(np.sqrt(geo['d_eff']))
    C_keff = logit_base - A_keff * geo['kappa_eff']
    delta_preds = {}
    for r in R_LEVELS:
        kappa_eff_r = geo['kappa_eff'] * float(np.sqrt(r))
        delta_preds[r] = float(C_keff + A_keff * kappa_eff_r) - logit_base

    slopes_by_m = {}
    for m in M_BUNDLE:
        U_comp = build_competitive_bundle(geo['pair_info'], m)
        if U_comp is None:
            continue
        actual_m = U_comp.shape[1]
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
        slopes_by_m[m] = {'slope': slope, 'pearson_r': r_val, 'actual_m': actual_m}

    if not slopes_by_m:
        return None

    # K_eff_estimated: 90% saturation
    all_slopes = [(m, slopes_by_m[m]['slope']) for m in sorted(slopes_by_m.keys())]
    max_slope = max(s for _, s in all_slopes)
    k_eff_est = max(m for m, s in all_slopes if s >= 0.9 * max_slope)

    # Pearson r(slope, m) -- A1 criterion
    ms = [m for m, _ in all_slopes]
    ss = [s for _, s in all_slopes]
    r_scale = float(pearsonr(ms, ss)[0]) if len(ms) >= 3 else 0.0

    # slope(m=1) for from_single
    s1 = slopes_by_m.get(1, {}).get('slope', None)
    k_eff_single = float(1.0 / s1) if s1 and s1 > 0 else None

    # slope(m=5) for mid5
    s5 = slopes_by_m.get(5, {}).get('slope', None)
    k_eff_mid5 = float(5.0 / s5) if s5 and s5 > 0 else None

    return {
        'arch': arch_name, 'seed': seed, 'K': int(K), 'd': int(d),
        'acc_base': float(acc_base), 'q_base': float(q_base),
        'slopes_by_m': {str(k): v for k, v in slopes_by_m.items()},
        'r_scale': r_scale,
        'k_eff_estimated': k_eff_est,
        'k_eff_from_single': k_eff_single,
        'k_eff_mid5': k_eff_mid5,
        'pass_A1': bool(r_scale > 0.90),
    }


def main():
    log("=" * 70)
    log("MULTI-SEED CROSS-ARCHITECTURE ABCS")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log(f"N_SEEDS={N_SEEDS}, N_ARCHS={len(EMBED_FILES)}, N_TRAIN_PER_CLASS={N_TRAIN_PER_CLASS}")
    log("=" * 70)
    log("PRE-REGISTERED:")
    log("  U1_multi: pooled CV(K_eff_estimated) < 0.30")
    log("  U2_multi: mean K_eff per arch in [6, 14]")
    log("  U3_multi: within-arch CV < 0.20 for >=4/5 archs")

    all_results = []

    for arch_name, path in EMBED_FILES:
        log(f"\n{'='*50}")
        log(f"ARCH: {arch_name}")
        log(f"{'='*50}")

        arch_seed_results = []
        for seed in SEEDS:
            log(f"  seed={seed} ...", )
            res = run_abcs_one_seed(arch_name, path, seed)
            if res:
                arch_seed_results.append(res)
                log(f"    -> K_eff_est={res['k_eff_estimated']}, r_scale={res['r_scale']:.3f}, A1={'PASS' if res['pass_A1'] else 'FAIL'}")
            else:
                log(f"    -> SKIP (no valid slopes)")

        if arch_seed_results:
            keffs = [r['k_eff_estimated'] for r in arch_seed_results]
            mean_k = float(np.mean(keffs))
            std_k = float(np.std(keffs))
            cv_k = float(std_k / (mean_k + 1e-10))
            n_pass_a1 = sum(1 for r in arch_seed_results if r['pass_A1'])
            log(f"  SUMMARY: K_eff = {mean_k:.1f} +/- {std_k:.1f} (CV={cv_k:.3f}), A1_pass {n_pass_a1}/{len(arch_seed_results)}")
            all_results.append({
                'arch': arch_name,
                'seed_results': arch_seed_results,
                'k_eff_mean': mean_k,
                'k_eff_std': std_k,
                'k_eff_cv': cv_k,
                'k_eff_all': keffs,
                'n_pass_a1': int(n_pass_a1),
            })

    # ==================== POOLED ANALYSIS ====================
    log("\n" + "=" * 70)
    log("POOLED UNIVERSALITY ANALYSIS")
    log("=" * 70)

    all_k_effs = []
    for r in all_results:
        all_k_effs.extend(r['k_eff_all'])

    pooled_mean = float(np.mean(all_k_effs))
    pooled_std = float(np.std(all_k_effs))
    pooled_cv = float(pooled_std / (pooled_mean + 1e-10))

    log(f"Pooled K_eff: {pooled_mean:.1f} +/- {pooled_std:.1f} (CV={pooled_cv:.3f})")

    # U1: pooled CV < 0.30
    pass_U1 = pooled_cv < 0.30
    log(f"U1 (pooled CV < 0.30): {'PASS' if pass_U1 else 'FAIL'} (CV={pooled_cv:.3f})")

    # U2: mean K_eff per arch in [6, 14]
    pass_U2 = all(6 <= r['k_eff_mean'] <= 14 for r in all_results)
    log(f"U2 (per-arch mean in [6,14]): {'PASS' if pass_U2 else 'FAIL'}")
    for r in all_results:
        log(f"  {r['arch']}: mean={r['k_eff_mean']:.1f} +/- {r['k_eff_std']:.1f}")

    # U3: within-arch CV < 0.20 for >=4/5 archs
    n_low_cv = sum(1 for r in all_results if r['k_eff_cv'] < 0.20)
    pass_U3 = n_low_cv >= 4
    log(f"U3 (within-arch CV < 0.20 for >=4/5): {'PASS' if pass_U3 else 'FAIL'} ({n_low_cv}/{len(all_results)})")

    overall = pass_U1 and pass_U2 and pass_U3
    verdict = (f"K_eff = {pooled_mean:.1f} +/- {pooled_std:.1f} "
               f"(pooled CV={pooled_cv:.3f}). "
               f"U1={'PASS' if pass_U1 else 'FAIL'}, "
               f"U2={'PASS' if pass_U2 else 'FAIL'}, "
               f"U3={'PASS' if pass_U3 else 'FAIL'}. "
               f"OVERALL: {'PASS' if overall else 'PARTIAL/FAIL'}")
    log(f"\nVERDICT: {verdict}")

    output = {
        'experiment': 'multiseed_cross_arch_abcs',
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'n_seeds': N_SEEDS, 'seeds': SEEDS,
            'n_train_per_class': N_TRAIN_PER_CLASS,
            'M_bundle': M_BUNDLE,
        },
        'arch_results': all_results,
        'pooled': {
            'all_k_effs': all_k_effs,
            'pooled_mean': pooled_mean, 'pooled_std': pooled_std, 'pooled_cv': pooled_cv,
            'pass_U1': bool(pass_U1), 'pass_U2': bool(pass_U2), 'pass_U3': bool(pass_U3),
            'overall_pass': bool(overall),
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
