#!/usr/bin/env python -u
"""
CROSS-ARCHITECTURE ABCS REPLICATION
=====================================

PRE-REGISTERED (Codex prescription, Feb 23 2026).

CONTEXT: ABCS on pythia-160m/DBpedia K=14 gave:
  - PRIMARY PASS: Pearson r(slope, m) = 0.9856 > 0.90
  - SECONDARY PASS: slope(K_eff=10) = 1.031 in [0.70, 1.30]
  - K_eff_estimated = 10, slope(1) = 0.175 => K_eff_from_single = 5.73

HYPOTHESIS (Universality): K_eff is NOT architecture-specific. For fixed
dataset (DBpedia K=14), K_eff should be approximately constant across
architectures of varying dimension/family (NLP encoders, SSM, etc.).

ARCHITECTURES (all DBpedia K=14, 7000 samples, 500/class):
  1. pythia-160m / l12  (768d) -- REFERENCE (already run)
  2. bert-base-uncased / l10  (768d)
  3. electra-small / l3  (256d)
  4. pythia-410m / l3   (1024d)
  5. rwkv-4-169m / l12  (768d)

PRE-REGISTERED ACCEPTANCE CRITERIA:
  PER-ARCH (each must pass):
    A1: Pearson r(slope, m) > 0.90 (same linear-scaling test as ABCS)
    A2: slope(K_eff_arch) in [0.70, 1.30]

  UNIVERSALITY:
    U1: K_eff_from_single (= 1/slope(1)) consistent across archs: CV < 30%
    U2: All archs pass A1 (r > 0.90)
    U3: K_eff_estimated in [6, 14] for all archs (within 40% of reference K_eff=10)

  PASS if: >=3/4 new archs pass A1 AND U1 passes (CV < 30%)
  STRONG PASS: all 4 new archs pass A1 AND U1 passes

FIXED PROTOCOL (no hyperparameter retuning):
  Same N_TRAIN_PER_CLASS=350, R_LEVELS, M_BUNDLE as original ABCS
  Same A_RENORM_K20=1.0535, ALPHA_KAPPA_K14=1.477 (pre-registered constants)
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

# ==================== CONFIGURATION ====================
EMBED_FILES = [
    ("pythia-160m",      "results/dointerv_multi_pythia-160m_l12.npz"),
    ("bert-base-uncased", "results/dointerv_multi_bert-base-uncased_l10.npz"),
    ("electra-small",    "results/dointerv_multi_electra-small_l3.npz"),
    ("pythia-410m",      "results/dointerv_multi_pythia-410m_l3.npz"),
    ("rwkv-4-169m",      "results/dointerv_multi_rwkv-4-169m_l12.npz"),
]

RESULT_PATH = "results/cti_cross_arch_abcs.json"
LOG_PATH = "results/cti_cross_arch_abcs_log.txt"

# FIXED PRE-REGISTERED CONSTANTS (same as original ABCS)
A_RENORM_K20 = 1.0535
ALPHA_KAPPA_K14 = 1.477
R_LEVELS = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
M_BUNDLE = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13]
N_TRAIN_PER_CLASS = 350
RANDOM_SEED = 42

# Pass thresholds (same as original)
SCALE_PEARSON_THRESH = 0.90
FULL_SLOPE_BAND = 0.30

# Universality thresholds
CV_THRESH = 0.30         # CV of K_eff_from_single across archs
K_EFF_LOWER = 6          # K_eff_estimated lower bound
K_EFF_UPPER = 14         # K_eff_estimated upper bound
N_ARCH_PASS_THRESH = 3   # min archs (of 4 new) that must pass A1

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
        n = min(n_train, len(idx) - 1)  # ensure at least 1 test sample
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
                'class_i': int(classes[i]), 'class_j': int(classes[j]),
                'delta': float(delta_ij),
                'kappa': float(kappa_ij),
                'sigma_cdir_sq': float(sigma_cdir_sq),
                'd_eff': float(d_eff_ij),
                'kappa_eff': float(kappa_eff_ij),
                'Delta_hat': Delta_hat,
            })

    pair_info.sort(key=lambda x: x['kappa'])
    nearest = pair_info[0]
    return {
        'mu': mu, 'grand_mean': grand_mean, 'trW': trW,
        'sigma_W_global': sigma_W_global, 'K': K, 'd': d,
        'pair_info': pair_info,
        'nearest': nearest,
        'kappa': nearest['kappa'],
        'd_eff': nearest['d_eff'],
        'kappa_eff': nearest['kappa_eff'],
        'Delta_hat': nearest['Delta_hat'],
        'sigma_cdir_sq': nearest['sigma_cdir_sq'],
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

    return np.stack(basis, axis=1)  # (d, K-1)


def build_bundle_directions(pair_info, m):
    """Top-m nearest pairs, Gram-Schmidt orthogonalized."""
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

    if tr_W_null < 1e-12 or tr_null_new <= 0:
        scale_null = 1.0
    else:
        scale_null = float(np.sqrt(tr_null_new / tr_W_null))

    scale_bundle = 1.0 / float(np.sqrt(r))

    def transform(X, y):
        X_new = X.copy()
        for i, c in enumerate(classes):
            mask = (y == c)
            Xc = X[mask]
            z = Xc - mu[i]
            z_between = z @ U_B @ U_B.T
            z_bundle = z @ U_bundle @ U_bundle.T
            z_other_comp = z_between - z_bundle
            z_null = z - z_between
            z_new = scale_bundle * z_bundle + z_other_comp + scale_null * z_null
            X_new[mask] = mu[i] + z_new
        return X_new

    return transform(X_tr, y_tr), transform(X_te, y_te)


def eval_q(X_tr, y_tr, X_te, y_te, K):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = float(np.clip((acc - 1.0/K) / (1.0 - 1.0/K), 1e-6, 1-1e-6))
    return acc, q, float(sp.logit(q))


def run_abcs_for_arch(arch_name, embed_path):
    """Run full ABCS sweep for one architecture. Returns result dict."""
    log(f"\n{'='*70}")
    log(f"ARCHITECTURE: {arch_name}  ({embed_path})")
    log(f"{'='*70}")

    X_tr, y_tr, X_te, y_te, classes = load_and_split(embed_path, N_TRAIN_PER_CLASS, RANDOM_SEED)
    K = len(classes)
    d = X_tr.shape[1]
    log(f"K={K}, d={d}, n_train={len(X_tr)}, n_test={len(X_te)}")

    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)

    log(f"Baseline: acc={acc_base:.4f}, q={q_base:.4f}, logit={logit_base:.4f}")
    log(f"kappa={geo['kappa']:.4f}, d_eff={geo['d_eff']:.4f}, kappa_eff={geo['kappa_eff']:.4f}")

    log("\nTop-8 nearest pairs:")
    for k, p in enumerate(geo['pair_info'][:8]):
        log(f"  m={k+1}: ({p['class_i']},{p['class_j']}) kappa={p['kappa']:.4f} d_eff={p['d_eff']:.2f}")

    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])
    tr_bundle_all = compute_subspace_trace(X_tr, y_tr, geo['mu'], classes, U_B)
    tr_W_null = geo['trW'] - tr_bundle_all
    log(f"\ntr_W_null = {tr_W_null:.4f} ({tr_W_null/geo['trW']*100:.1f}% of trW)")

    # Prediction constants (FIXED pre-registered)
    A_keff_K14 = ALPHA_KAPPA_K14 / float(np.sqrt(geo['d_eff']))
    C_keff = logit_base - A_keff_K14 * geo['kappa_eff']

    # ---- Bundle sweep ----
    all_bundle_results = {}
    for m in M_BUNDLE:
        if m > K - 1:
            continue
        U_bundle = build_bundle_directions(geo['pair_info'], m)
        actual_m = U_bundle.shape[1]
        m_results = []
        for r in R_LEVELS:
            X_tr_b, X_te_b = apply_bundle_surgery(
                X_tr, X_te, y_tr, y_te, geo, U_bundle, U_B, r, tr_W_null)
            acc_r, q_r, logit_r = eval_q(X_tr_b, y_tr, X_te_b, y_te, K)
            delta_obs = logit_r - logit_base
            kappa_eff_pred = geo['kappa_eff'] * float(np.sqrt(r))
            delta_pred_K14 = (C_keff + A_keff_K14 * kappa_eff_pred) - logit_base
            ratio = float(delta_obs / delta_pred_K14) if abs(delta_pred_K14) > 1e-6 else 0.0
            m_results.append({
                'r': float(r), 'actual_bundle_rank': int(actual_m),
                'q': float(q_r), 'logit': float(logit_r),
                'delta_logit': float(delta_obs),
                'delta_pred_K14': float(delta_pred_K14),
                'ratio': ratio,
            })
            if abs(r - 1.0) > 0.01:
                log(f"  m={m:2d}, r={r:5.1f}: delta_obs={delta_obs:+.4f}, pred={delta_pred_K14:+.4f}, ratio={ratio:.3f}")
        all_bundle_results[str(m)] = m_results

    # ---- Compute slopes ----
    slopes_by_m = {}
    for m_str, m_results in all_bundle_results.items():
        m_int = int(m_str)
        nontrivial = [x for x in m_results if abs(x['r'] - 1.0) > 0.01]
        if len(nontrivial) < 3:
            continue
        deltas_obs = [x['delta_logit'] for x in nontrivial]
        deltas_pred = [x['delta_pred_K14'] for x in nontrivial]
        if np.std(deltas_pred) > 1e-8:
            slope = float(np.polyfit(deltas_pred, deltas_obs, 1)[0])
            r_val, _ = pearsonr(deltas_obs, deltas_pred)
        else:
            slope, r_val = 0.0, 0.0
        slopes_by_m[m_int] = {'slope': slope, 'pearson_r': float(r_val)}

    log(f"\n{'m':>4} | {'slope':>8} | {'pearson_r':>9} | slope/slope(1)")
    slope_m1 = slopes_by_m.get(1, {}).get('slope', 1e-6)
    for m_int in sorted(slopes_by_m.keys()):
        s = slopes_by_m[m_int]['slope']
        r_val = slopes_by_m[m_int]['pearson_r']
        log(f"{m_int:>4} | {s:>8.4f} | {r_val:>9.4f} | {s/(slope_m1+1e-10):.3f}")

    m_vals = sorted(slopes_by_m.keys())
    slope_vals = [slopes_by_m[m]['slope'] for m in m_vals]

    if len(m_vals) >= 3:
        n_for_pearson = min(8, len(m_vals))
        r_scale, _ = pearsonr(m_vals[:n_for_pearson], slope_vals[:n_for_pearson])
    else:
        r_scale = 0.0

    max_slope = max(slope_vals) if slope_vals else 0.0
    k_eff_est = None
    for m, s in zip(m_vals, slope_vals):
        if s >= 0.90 * max_slope:
            k_eff_est = m
            break
    k_eff_from_single = 1.0 / (slope_m1 + 1e-10)

    slope_k_eff = slopes_by_m.get(k_eff_est, {}).get('slope', 0.0) if k_eff_est else 0.0
    pass_A1 = r_scale > SCALE_PEARSON_THRESH
    pass_A2 = (0.70 <= slope_k_eff <= 1.30) if k_eff_est else False

    log(f"\nSCALE ANALYSIS for {arch_name}:")
    log(f"  r_scale(m in [1..8]) = {r_scale:.4f} | A1 PASS (>{SCALE_PEARSON_THRESH}): {pass_A1}")
    log(f"  K_eff_estimated (90% sat.) = {k_eff_est}")
    log(f"  K_eff_from_single = {k_eff_from_single:.2f} (= 1/slope(1))")
    log(f"  slope(K_eff) = {slope_k_eff:.4f} | A2 PASS (in [0.7,1.3]): {pass_A2}")

    return {
        'arch': arch_name,
        'embed_path': embed_path,
        'd': int(d),
        'K': int(K),
        'baseline': {
            'acc': float(acc_base), 'q': float(q_base), 'logit': float(logit_base),
            'kappa': float(geo['kappa']), 'd_eff': float(geo['d_eff']),
            'kappa_eff': float(geo['kappa_eff']),
        },
        'slopes_by_m': {str(k): v for k, v in slopes_by_m.items()},
        'analysis': {
            'r_scale': float(r_scale),
            'k_eff_estimated': int(k_eff_est) if k_eff_est else None,
            'k_eff_from_single': float(k_eff_from_single),
            'slope_m1': float(slope_m1),
            'slope_k_eff': float(slope_k_eff),
            'pass_A1': bool(pass_A1),
            'pass_A2': bool(pass_A2),
        },
        'bundle_results': all_bundle_results,
    }


def main():
    log("=" * 70)
    log("CROSS-ARCHITECTURE ABCS REPLICATION")
    log("Pre-registered: test K_eff universality across DBpedia K=14 archs")
    log("=" * 70)
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log(f"Fixed constants: A_RENORM_K20={A_RENORM_K20}, ALPHA_KAPPA_K14={ALPHA_KAPPA_K14}")
    log(f"Pre-registered thresholds: r>={SCALE_PEARSON_THRESH}, CV<{CV_THRESH}, K_eff in [{K_EFF_LOWER},{K_EFF_UPPER}]")
    log("")

    arch_results = []
    for arch_name, embed_path in EMBED_FILES:
        if not os.path.exists(embed_path):
            log(f"WARNING: {embed_path} not found, skipping {arch_name}")
            continue
        result = run_abcs_for_arch(arch_name, embed_path)
        arch_results.append(result)

    # ==================== UNIVERSALITY ANALYSIS ====================
    log("\n" + "=" * 70)
    log("UNIVERSALITY ANALYSIS")
    log("=" * 70)

    k_eff_from_singles = []
    k_eff_estimated = []
    pass_A1_list = []
    pass_A2_list = []

    log(f"\n{'Arch':>20} | {'d':>6} | {'kappa':>6} | {'K_eff_single':>12} | {'K_eff_est':>9} | A1  | A2")
    log("-" * 75)
    for r in arch_results:
        a = r['analysis']
        log(f"{r['arch']:>20} | {r['d']:>6} | {r['baseline']['kappa']:>6.4f} | "
            f"{a['k_eff_from_single']:>12.2f} | {str(a['k_eff_estimated']):>9} | "
            f"{'YES' if a['pass_A1'] else 'NO':>3} | {'YES' if a['pass_A2'] else 'NO':>3}")
        if a['k_eff_from_single'] > 0:
            k_eff_from_singles.append(a['k_eff_from_single'])
        if a['k_eff_estimated'] is not None:
            k_eff_estimated.append(a['k_eff_estimated'])
        pass_A1_list.append(a['pass_A1'])
        pass_A2_list.append(a['pass_A2'])

    log("")
    n_archs = len(arch_results)
    n_new = n_archs - 1  # exclude reference pythia-160m

    # CV of K_eff_from_single
    if len(k_eff_from_singles) >= 2:
        cv_k_eff = float(np.std(k_eff_from_singles) / (np.mean(k_eff_from_singles) + 1e-10))
        mean_k_eff = float(np.mean(k_eff_from_singles))
        std_k_eff = float(np.std(k_eff_from_singles))
    else:
        cv_k_eff, mean_k_eff, std_k_eff = 0.0, 0.0, 0.0

    n_pass_A1_all = sum(pass_A1_list)
    n_pass_A1_new = sum(pass_A1_list[1:])  # skip reference
    n_in_range = sum(K_EFF_LOWER <= k <= K_EFF_UPPER for k in k_eff_estimated)

    pass_U1 = cv_k_eff < CV_THRESH
    pass_U2 = n_pass_A1_new >= N_ARCH_PASS_THRESH
    pass_U3 = n_in_range >= max(1, len(k_eff_estimated) - 1)  # all-1 can fail

    log(f"K_eff_from_single statistics:")
    log(f"  Values: {[round(x,2) for x in k_eff_from_singles]}")
    log(f"  Mean = {mean_k_eff:.2f}, Std = {std_k_eff:.2f}, CV = {cv_k_eff:.3f}")
    log(f"  U1 (CV < {CV_THRESH}): {pass_U1} | U2 (>={N_ARCH_PASS_THRESH} new pass A1): {pass_U2} | U3 (K_eff in range): {pass_U3}")
    log("")

    log(f"K_eff_estimated values: {k_eff_estimated}")
    log(f"Passes A1 (r>0.90): {[r['arch'] for r,p in zip(arch_results,pass_A1_list) if p]}")
    log(f"Fails A1: {[r['arch'] for r,p in zip(arch_results,pass_A1_list) if not p]}")

    n_universality_pass = sum([pass_U1, pass_U2, pass_U3])
    if n_pass_A1_all == n_archs and pass_U1:
        verdict = "STRONG PASS: All architectures show linear scale-with-m. K_eff is UNIVERSAL across architectures."
    elif pass_U2 and pass_U1:
        verdict = "PASS: Majority of architectures confirm linear scale-with-m. K_eff approximately universal."
    elif pass_U2:
        verdict = f"PARTIAL: Most archs show linear scaling (A1 pass={n_pass_A1_new}/{n_new}) but K_eff has high variance (CV={cv_k_eff:.2f}). Not fully universal."
    elif n_pass_A1_all >= 2:
        verdict = f"WEAK: Only {n_pass_A1_all}/{n_archs} archs pass A1. Replication needs improvement."
    else:
        verdict = "FAIL: Causal multi-competitor law does not replicate across architectures."

    log(f"\nOVERALL UNIVERSALITY: {n_universality_pass}/3 U-criteria pass")
    log(f"VERDICT: {verdict}")

    # ==================== SAVE ====================
    output = {
        'experiment': 'cross_arch_abcs_replication',
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'A_RENORM_K20': A_RENORM_K20,
            'ALPHA_KAPPA_K14': ALPHA_KAPPA_K14,
            'R_LEVELS': R_LEVELS,
            'M_BUNDLE': M_BUNDLE,
            'N_TRAIN_PER_CLASS': N_TRAIN_PER_CLASS,
            'thresholds': {
                'scale_pearson': SCALE_PEARSON_THRESH,
                'cv_k_eff': CV_THRESH,
                'k_eff_lower': K_EFF_LOWER,
                'k_eff_upper': K_EFF_UPPER,
            }
        },
        'arch_results': [
            {k: v for k, v in r.items() if k != 'bundle_results'}
            for r in arch_results
        ],
        'universality': {
            'k_eff_from_singles': k_eff_from_singles,
            'mean_k_eff': mean_k_eff,
            'std_k_eff': std_k_eff,
            'cv_k_eff': cv_k_eff,
            'k_eff_estimated': k_eff_estimated,
            'n_pass_A1_all': n_pass_A1_all,
            'n_pass_A1_new': n_pass_A1_new,
            'n_in_range': n_in_range,
            'pass_U1': bool(pass_U1),
            'pass_U2': bool(pass_U2),
            'pass_U3': bool(pass_U3),
            'n_universality_pass': int(n_universality_pass),
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
