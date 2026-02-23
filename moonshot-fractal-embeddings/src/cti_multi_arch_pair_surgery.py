#!/usr/bin/env python -u
"""
MULTI-ARCH INDIVIDUAL-PAIR SURGERY: Fitted A_local
====================================================
PURPOSE: Test whether A_local (w_k decay constant) is universal across
         architectures, using the FITTED A_local instead of pre-registered 1.75.

MOTIVATION: Session 46 found A_local_fit=0.197 on pythia-160m vs pre-registered
            1.75. This experiment tests:
            (1) Is A_local_fit~0.197 consistent across all 5 archs (CV < 0.30)?
            (2) Does A_local_fit correlate with tail_ratio_inv (eta_m finding, r=0.929)?
            (3) Is r_wk higher when we use the FITTED A_local vs fixed 1.75?

THEORY:
  logit(q) = A * sum_k w_k * kappa_k * sqrt(d_eff_k) + C
  w_k = exp(-A_local * (kappa_k - kappa_1) * sqrt(d_eff_1))
  Single-pair surgery on pair k:
    slope_k = empirical_wk * kappa_k * sqrt(d_eff_k) / (kappa_1 * sqrt(d_eff_1))
  Fit A_local from: log(empirical_wk) = -A_local * (kappa_k - kappa_1) * sqrt(d_eff_1)

PRE-REGISTERED (Feb 25, 2026):
  M1: CV(A_local_fit) across architectures < 0.30 (A_local is architecturally universal)
  M2: Pearson r(A_local_fit, tail_ratio_inv) > 0.50 (connects to eta_m)
  M3: mean A_local_fit in [0.10, 0.40] (confirm Session 46 finding ~0.197)
  M4: r_wk_fitted > r_wk_fixed (fitted A_local better predicts empirical w_k)

ARCHITECTURES: pythia-160m, bert-base-uncased, electra-small, pythia-410m, rwkv-4-169m
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_multi_arch_pair_surgery.json"
LOG_PATH = "results/cti_multi_arch_pair_surgery_log.txt"

ARCHS = [
    ('pythia-160m',      'results/dointerv_multi_pythia-160m_l12.npz',       768),
    ('bert-base-uncased','results/dointerv_multi_bert-base-uncased_l10.npz', 768),
    ('electra-small',    'results/dointerv_multi_electra-small_l3.npz',       256),
    ('pythia-410m',      'results/dointerv_multi_pythia-410m_l3.npz',        1024),
    ('rwkv-4-169m',      'results/dointerv_multi_rwkv-4-169m_l12.npz',       768),
]

A_LOCAL_PREREG = 1.75     # from Session 24 rank-spectrum (pre-registered)
R_LEVELS = [0.3, 0.5, 2.0, 5.0, 10.0]
N_TRAIN_PER_CLASS = 350
N_PAIRS_TO_TEST = 12
N_SEEDS = 3
SEEDS = [42, 137, 271]

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


def compute_eigenspectrum(X_tr, y_tr, classes):
    """Compute within-class covariance eigenspectrum (for tail_ratio_inv)."""
    K = len(classes)
    N, d = len(X_tr), X_tr.shape[1]
    S = np.zeros(d)
    for c in classes:
        Xc = X_tr[y_tr == c]
        mu_c = Xc.mean(0)
        Z = Xc - mu_c
        # Per-class scatter (diagonal approx of covariance eigenspectrum via variance)
        S += np.var(Z, axis=0) * (len(Xc) / N)
    S_sorted = np.sort(S)[::-1]  # descending
    n_top = max(1, min(5, len(S_sorted) // 10))
    n_bot = max(1, min(5, len(S_sorted) // 10))
    tail_ratio = float(S_sorted[:n_top].mean() / (S_sorted[-n_bot:].mean() + 1e-12))
    tail_ratio_inv = float(1.0 / (tail_ratio + 1e-12))
    isotropy = float(S_sorted.mean() / (S_sorted.max() + 1e-12))
    return {'tail_ratio': tail_ratio, 'tail_ratio_inv': tail_ratio_inv,
            'isotropy': isotropy}


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
    """Apply surgery to ONLY pair k, compensate in null space."""
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


def run_individual_pair_surgery_for_arch(X_tr, y_tr, X_te, y_te, classes, n_pairs):
    """Run individual pair surgery, return pair_results and fitted A_local."""
    K = len(classes)
    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)
    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])

    kappa_1 = geo['pair_info'][0]['kappa']
    d_eff_1 = geo['pair_info'][0]['d_eff']
    kappa_eff_1 = geo['pair_info'][0]['kappa_eff']

    # Reference predictions from pair 1
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

        # Both predicted w_k versions (fixed and to-be-fitted)
        delta_kappa = kappa_k - kappa_1
        predicted_wk_fixed = float(np.exp(-A_LOCAL_PREREG * delta_kappa * np.sqrt(d_eff_1)))

        pair_results.append({
            'k': k_idx + 1,
            'kappa_k': float(kappa_k), 'd_eff_k': float(d_eff_k),
            'slope_k': float(slope_k), 'r_k': float(r_k),
            'empirical_wk': float(empirical_wk),
            'predicted_wk_fixed': float(predicted_wk_fixed),
            'delta_kappa': float(delta_kappa),
            'delta_kappa_sqrt_d': float(delta_kappa * np.sqrt(d_eff_1)),
        })

    if not pair_results:
        return pair_results, {}, float('nan'), float('nan'), float('nan'), float('nan')

    # Fit A_local from data: log(empirical_wk) = -A_local * delta_kappa * sqrt(d_eff_1)
    emp_wks = [pr['empirical_wk'] for pr in pair_results]
    delta_kappa_sqrt_d = [pr['delta_kappa_sqrt_d'] for pr in pair_results]

    # Only include pairs where delta_kappa > 0 and empirical_wk > 0 for log fit
    valid = [(dk, w) for dk, w in zip(delta_kappa_sqrt_d, emp_wks)
             if dk > 1e-6 and w > 1e-6]
    if len(valid) >= 3:
        dk_arr = np.array([v[0] for v in valid])
        lw_arr = np.array([-np.log(v[1]) for v in valid])
        A_local_fit = float(np.polyfit(dk_arr, lw_arr, 1)[0])
    else:
        A_local_fit = float('nan')

    # r_wk with fixed A_local
    pred_wks_fixed = [pr['predicted_wk_fixed'] for pr in pair_results]
    if len(emp_wks) >= 3 and np.std(pred_wks_fixed) > 1e-6:
        r_wk_fixed = float(pearsonr(emp_wks, pred_wks_fixed)[0])
    else:
        r_wk_fixed = 0.0

    # r_wk with FITTED A_local
    if not np.isnan(A_local_fit):
        pred_wks_fitted = [float(np.exp(-A_local_fit * pr['delta_kappa_sqrt_d']))
                           for pr in pair_results]
        if len(emp_wks) >= 3 and np.std(pred_wks_fitted) > 1e-6:
            r_wk_fitted = float(pearsonr(emp_wks, pred_wks_fitted)[0])
        else:
            r_wk_fitted = 0.0
    else:
        r_wk_fitted = 0.0

    baseline_info = {
        'acc_base': float(acc_base), 'q_base': float(q_base),
        'kappa_1': float(kappa_1), 'd_eff_1': float(d_eff_1),
    }
    return pair_results, baseline_info, float(A_local_fit), float(r_wk_fixed), float(r_wk_fitted)


def main():
    log("=" * 70)
    log("MULTI-ARCH INDIVIDUAL-PAIR SURGERY: Fitted A_local")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log(f"N_ARCHS={len(ARCHS)}, N_SEEDS={N_SEEDS}, N_PAIRS={N_PAIRS_TO_TEST}")
    log("=" * 70)
    log("PRE-REGISTERED (Feb 25, 2026):")
    log("  M1: CV(A_local_fit) across archs < 0.30")
    log("  M2: Pearson r(A_local_fit, tail_ratio_inv) > 0.50")
    log("  M3: mean A_local_fit in [0.10, 0.40]")
    log("  M4: r_wk_fitted > r_wk_fixed (fitted A_local better)")

    all_arch_results = []

    for arch_name, embed_path, d_arch in ARCHS:
        log(f"\n{'='*50}")
        log(f"ARCH: {arch_name} (d={d_arch})")
        log(f"{'='*50}")

        arch_seed_results = []
        for seed in SEEDS:
            log(f"  seed={seed} ...")
            X_tr, y_tr, X_te, y_te, classes = load_and_split(embed_path, N_TRAIN_PER_CLASS, seed)
            K = len(classes)

            pair_results, baseline_info, A_local_fit, r_wk_fixed, r_wk_fitted = \
                run_individual_pair_surgery_for_arch(X_tr, y_tr, X_te, y_te, classes, N_PAIRS_TO_TEST)

            if not pair_results:
                log(f"    SKIP (no valid results)")
                continue

            log(f"    A_local_fit={A_local_fit:.3f}, r_wk_fixed={r_wk_fixed:.3f}, r_wk_fitted={r_wk_fitted:.3f}")

            arch_seed_results.append({
                'seed': seed,
                'A_local_fit': float(A_local_fit),
                'r_wk_fixed': float(r_wk_fixed),
                'r_wk_fitted': float(r_wk_fitted),
                'baseline': baseline_info,
                'n_pairs': len(pair_results),
            })

        if not arch_seed_results:
            log(f"  No valid seeds for {arch_name}")
            continue

        # Per-arch summary
        A_locals = [s['A_local_fit'] for s in arch_seed_results if not np.isnan(s['A_local_fit'])]
        mean_A = float(np.mean(A_locals)) if A_locals else float('nan')
        std_A = float(np.std(A_locals)) if len(A_locals) > 1 else 0.0
        mean_r_fixed = float(np.mean([s['r_wk_fixed'] for s in arch_seed_results]))
        mean_r_fitted = float(np.mean([s['r_wk_fitted'] for s in arch_seed_results]))

        # Compute tail_ratio_inv for this arch (use seed=42 for representative geometry)
        X_tr, y_tr, _, _, classes = load_and_split(embed_path, N_TRAIN_PER_CLASS, 42)
        spec = compute_eigenspectrum(X_tr, y_tr, classes)

        log(f"  SUMMARY: A_local = {mean_A:.3f} +/- {std_A:.3f}, "
            f"r_wk_fixed={mean_r_fixed:.3f}, r_wk_fitted={mean_r_fitted:.3f}, "
            f"tail_ratio_inv={spec['tail_ratio_inv']:.4f}")

        all_arch_results.append({
            'arch': arch_name,
            'd': d_arch,
            'seed_results': arch_seed_results,
            'mean_A_local': float(mean_A),
            'std_A_local': float(std_A),
            'mean_r_wk_fixed': float(mean_r_fixed),
            'mean_r_wk_fitted': float(mean_r_fitted),
            'tail_ratio_inv': float(spec['tail_ratio_inv']),
            'isotropy': float(spec['isotropy']),
        })

    # ==================== POOLED ANALYSIS ====================
    log("\n" + "=" * 70)
    log("POOLED UNIVERSALITY ANALYSIS")
    log("=" * 70)

    if not all_arch_results:
        log("No results!")
        return

    A_locals_per_arch = [r['mean_A_local'] for r in all_arch_results if not np.isnan(r['mean_A_local'])]
    tail_ratios = [r['tail_ratio_inv'] for r in all_arch_results]
    mean_r_fixed_all = [r['mean_r_wk_fixed'] for r in all_arch_results]
    mean_r_fitted_all = [r['mean_r_wk_fitted'] for r in all_arch_results]

    if len(A_locals_per_arch) < 2:
        log("Not enough arch results for pooled analysis")
        return

    # M1: CV(A_local) across archs
    pooled_A_mean = float(np.mean(A_locals_per_arch))
    pooled_A_std = float(np.std(A_locals_per_arch))
    pooled_CV = float(pooled_A_std / (pooled_A_mean + 1e-10))
    pass_M1 = pooled_CV < 0.30

    log(f"Pooled A_local: {pooled_A_mean:.3f} +/- {pooled_A_std:.3f} (CV={pooled_CV:.3f})")
    log(f"M1 (CV < 0.30): {'PASS' if pass_M1 else 'FAIL'} (CV={pooled_CV:.3f})")

    # M2: r(A_local_fit, tail_ratio_inv)
    if len(A_locals_per_arch) >= 3 and np.std(tail_ratios[:len(A_locals_per_arch)]) > 1e-6:
        r_M2, p_M2 = pearsonr(A_locals_per_arch, tail_ratios[:len(A_locals_per_arch)])
        r_M2, p_M2 = float(r_M2), float(p_M2)
    else:
        r_M2, p_M2 = 0.0, 1.0
    pass_M2 = r_M2 > 0.50
    log(f"M2 r(A_local, tail_ratio_inv): r={r_M2:.3f} (p={p_M2:.3f}) {'PASS' if pass_M2 else 'FAIL'}")

    # M3: mean A_local in [0.10, 0.40]
    pass_M3 = 0.10 <= pooled_A_mean <= 0.40
    log(f"M3 mean A_local in [0.10, 0.40]: {'PASS' if pass_M3 else 'FAIL'} (mean={pooled_A_mean:.3f})")

    # M4: r_wk_fitted > r_wk_fixed
    mean_fixed = float(np.mean(mean_r_fixed_all))
    mean_fitted = float(np.mean(mean_r_fitted_all))
    pass_M4 = mean_fitted > mean_fixed
    log(f"M4 r_wk_fitted > r_wk_fixed: {'PASS' if pass_M4 else 'FAIL'} "
        f"(fitted={mean_fitted:.3f} vs fixed={mean_fixed:.3f})")

    log(f"\nPer-arch A_local:")
    for r in all_arch_results:
        log(f"  {r['arch']:25s}: A_local={r['mean_A_local']:.3f}+/-{r['std_A_local']:.3f}, "
            f"tail_inv={r['tail_ratio_inv']:.4f}")

    overall_pass = all([pass_M1, pass_M3])  # M1 and M3 are primary
    verdict = (f"M1={'PASS' if pass_M1 else 'FAIL'}, M2={'PASS' if pass_M2 else 'FAIL'}, "
               f"M3={'PASS' if pass_M3 else 'FAIL'}, M4={'PASS' if pass_M4 else 'FAIL'}. "
               f"A_local={pooled_A_mean:.3f}+/-{pooled_A_std:.3f} (CV={pooled_CV:.3f}). "
               f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    log(f"\nVERDICT: {verdict}")

    output = {
        'experiment': 'multi_arch_pair_surgery',
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'archs': [a[0] for a in ARCHS],
            'seeds': SEEDS,
            'n_pairs': N_PAIRS_TO_TEST,
            'A_LOCAL_prereg': A_LOCAL_PREREG,
        },
        'arch_results': all_arch_results,
        'pooled': {
            'mean_A_local': float(pooled_A_mean),
            'std_A_local': float(pooled_A_std),
            'cv_A_local': float(pooled_CV),
            'r_A_vs_tailinv': float(r_M2), 'p_A_vs_tailinv': float(p_M2),
            'mean_r_wk_fixed': float(mean_fixed),
            'mean_r_wk_fitted': float(mean_fitted),
        },
        'criteria': {
            'pass_M1': bool(pass_M1), 'pass_M2': bool(pass_M2),
            'pass_M3': bool(pass_M3), 'pass_M4': bool(pass_M4),
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
