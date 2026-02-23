#!/usr/bin/env python -u
"""
BGE-SMALL ON DBPEDIA K=14 ABCS
================================

PRE-REGISTERED (Feb 23 2026). Critical experiment.

QUESTION: Does BGE-small (384d) on DBpedia K=14 show K_eff~10 (dataset effect)
          or K_eff~13=K-1 (architecture effect)?

If K_eff~10: DBpedia geometry is sparse (same 10 pairs compete across ALL architectures)
If K_eff~13: BGE-small sees more competition (architecture-specific regime)

Context:
- 5 dointerv archs (pythia-160m, bert, electra, pythia-410m, rwkv) all on DBpedia K=14 -> K_eff_est=[10,10,8,10,10]
- BGE-small on balanced datasets (newsgroups-l1 K=20) -> K_eff=19=K-1 (dense regime)
- DOES BGE-small replicate K_eff=10 for DBpedia K=14?

Data: data/beir/dbpedia14_train_50000_embeddings.npy (50000, 384), l1 labels (K=14)
Same N_TRAIN_PER_CLASS=350 as dointerv protocol.
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

EMBED_PATH = "data/beir/dbpedia14_train_50000_embeddings.npy"
LABELS_PATH = "data/beir/dbpedia14_train_50000_labels.npz"
RESULT_PATH = "results/cti_bge_dbpedia14_abcs.json"
LOG_PATH = "results/cti_bge_dbpedia14_abcs_log.txt"

A_RENORM_K20 = 1.0535
ALPHA_KAPPA_K14 = 1.477
R_LEVELS = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
M_BUNDLE = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13]
N_TRAIN_PER_CLASS = 350
RANDOM_SEED = 42

os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')

def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def load_and_split():
    X = np.load(EMBED_PATH, mmap_mode='r').astype(np.float64)
    lab = np.load(LABELS_PATH)
    y = lab['l1']
    classes = np.unique(y)
    K = len(classes)

    rng = np.random.default_rng(RANDOM_SEED)
    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = min(N_TRAIN_PER_CLASS, len(idx) - 1)
        X_tr_list.append(X[idx[:n]]); y_tr_list.append(y[idx[:n]])
        X_te_list.append(X[idx[n:]]); y_te_list.append(y[idx[n:]])

    return (np.concatenate(X_tr_list).copy(), np.concatenate(y_tr_list),
            np.concatenate(X_te_list).copy(), np.concatenate(y_te_list), classes)


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


def build_bundle_directions(pair_info, m):
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


def eval_q(X_tr, y_tr, X_te, y_te, K):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = float(np.clip((acc - 1.0/K) / (1.0 - 1.0/K), 1e-6, 1-1e-6))
    return acc, q, float(sp.logit(q))


def main():
    log("=" * 70)
    log("BGE-SMALL ON DBPEDIA K=14 ABCS")
    log("QUESTION: K_eff~10 (dataset effect) or K_eff~13 (architecture)?")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log("=" * 70)

    X_tr, y_tr, X_te, y_te, classes = load_and_split()
    K = len(classes)
    d = X_tr.shape[1]
    log(f"K={K}, d={d}, n_train={len(X_tr)}, n_test={len(X_te)}")

    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)
    log(f"Baseline: acc={acc_base:.4f}, q={q_base:.4f}, logit={logit_base:.4f}")
    log(f"kappa={geo['kappa']:.4f}, d_eff={geo['d_eff']:.4f}, kappa_eff={geo['kappa_eff']:.4f}")

    log("\nTop-8 nearest pairs:")
    for k, p in enumerate(geo['pair_info'][:8]):
        log(f"  m={k+1}: ({p['i']},{p['j']}) kappa={p['kappa']:.4f} d_eff={p['d_eff']:.2f}")

    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])
    tr_bundle_all = compute_subspace_trace(X_tr, y_tr, geo['mu'], classes, U_B)
    tr_W_null = geo['trW'] - tr_bundle_all
    log(f"\ntr_W_null = {tr_W_null:.4f} ({tr_W_null/geo['trW']*100:.1f}% of trW)")

    A_keff_K14 = ALPHA_KAPPA_K14 / float(np.sqrt(geo['d_eff']))
    C_keff = logit_base - A_keff_K14 * geo['kappa_eff']

    all_slopes = {}
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
            delta_pred = (C_keff + A_keff_K14 * kappa_eff_pred) - logit_base
            ratio = float(delta_obs / delta_pred) if abs(delta_pred) > 1e-6 else 0.0
            m_results.append({'r': float(r), 'delta_logit': float(delta_obs),
                               'delta_pred': float(delta_pred), 'ratio': ratio})
            if abs(r - 1.0) > 0.01:
                log(f"  m={m:2d}, r={r:5.1f}: delta_obs={delta_obs:+.4f}, ratio={ratio:.3f}")

        nontrivial = [x for x in m_results if abs(x['r'] - 1.0) > 0.01]
        if len(nontrivial) >= 3:
            deltas_obs = [x['delta_logit'] for x in nontrivial]
            deltas_pred = [x['delta_pred'] for x in nontrivial]
            slope = float(np.polyfit(deltas_pred, deltas_obs, 1)[0]) if np.std(deltas_pred) > 1e-8 else 0.0
            r_val = float(pearsonr(deltas_obs, deltas_pred)[0]) if np.std(deltas_pred) > 1e-8 else 0.0
        else:
            slope, r_val = 0.0, 0.0
        all_slopes[m] = {'slope': slope, 'pearson_r': r_val, 'actual_m': int(actual_m)}

    log(f"\n{'m':>4} | {'slope':>8} | {'pearson_r':>9} | slope/slope(1)")
    slope_m1 = all_slopes.get(1, {}).get('slope', 1e-6)
    for m, info in sorted(all_slopes.items()):
        s = info['slope']
        log(f"{m:>4} | {s:>8.4f} | {info['pearson_r']:>9.4f} | {s/(slope_m1+1e-10):.3f}")

    m_vals = sorted(all_slopes.keys())
    slope_vals = [all_slopes[m]['slope'] for m in m_vals]

    n_for_pearson = min(8, len(m_vals))
    r_scale = float(pearsonr(m_vals[:n_for_pearson], slope_vals[:n_for_pearson])[0]) if n_for_pearson >= 3 else 0.0

    max_slope = max(slope_vals) if slope_vals else 0.0
    k_eff_est = None
    for m, s in zip(m_vals, slope_vals):
        if s >= 0.90 * max_slope:
            k_eff_est = m
            break

    k_eff_from_single = 1.0 / (slope_m1 + 1e-10)
    slope_m5 = all_slopes.get(5, {}).get('slope', 0.0)
    k_eff_mid5 = 5.0 / (slope_m5 + 1e-10)

    log(f"\nr_scale = {r_scale:.4f} | K_eff_est = {k_eff_est}")
    log(f"K_eff_from_single = {k_eff_from_single:.2f} | K_eff_mid5 = {k_eff_mid5:.2f}")

    # THE KEY COMPARISON
    log("\n" + "=" * 70)
    log("KEY COMPARISON: BGE-small vs dointerv architectures on DBpedia K=14")
    log("=" * 70)
    log(f"BGE-small K_eff_est = {k_eff_est}")
    log(f"Dointerv K_eff_est  = [10, 10, 8, 10, 10] (mean=9.6)")
    if k_eff_est is not None:
        if k_eff_est <= 10:
            conclusion = f"K_eff~{k_eff_est} <= 10: DBpedia geometry is SPARSE for ALL architectures. K_eff is DATASET-SPECIFIC."
        else:
            conclusion = f"K_eff~{k_eff_est} > 10: BGE-small sees more competition. K_eff is ARCHITECTURE-SPECIFIC too."
    else:
        conclusion = "Cannot determine K_eff."
    log(f"\nCONCLUSION: {conclusion}")

    result = {
        'experiment': 'bge_dbpedia14_abcs',
        'timestamp': datetime.datetime.now().isoformat(),
        'arch': 'bge-small',
        'd': int(d), 'K': int(K),
        'baseline': {'acc': float(acc_base), 'q': float(q_base), 'logit': float(logit_base),
                     'kappa': float(geo['kappa']), 'd_eff': float(geo['d_eff']),
                     'kappa_eff': float(geo['kappa_eff'])},
        'slopes_by_m': {str(k): v for k, v in all_slopes.items()},
        'analysis': {
            'r_scale': float(r_scale),
            'k_eff_estimated': int(k_eff_est) if k_eff_est else None,
            'k_eff_from_single': float(k_eff_from_single),
            'k_eff_mid5': float(k_eff_mid5),
            'pass_A1': bool(r_scale > 0.90),
        },
        'conclusion': conclusion,
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
