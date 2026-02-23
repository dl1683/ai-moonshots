#!/usr/bin/env python -u
"""
K-SCALING ABCS: Does K_eff scale with K?
==========================================

PRE-REGISTERED (Feb 23 2026).

KEY QUESTION: Is K_eff ≈ 10 universally (fixed competition)
              OR K_eff ≈ K (all competitors matter)?

From DBpedia K=14: K_eff_estimated ≈ 10 (across 5 archs).
Prediction:
  H_Kscale: K_eff ∝ K  => for K=6: K_eff≈4, K=9: K_eff≈6, K=20: K_eff≈14
  H_Kfix:   K_eff ≈ 10 => for K=6: K_eff≈5 (capped), K=9: K_eff≈9, K=20: K_eff≈10

DATASETS (all BGE-small embeddings, d=384):
  - newsgroups l0: K=6
  - dbpedia l0:    K=9
  - clinc l0:      K=10
  - newsgroups l1: K=20

The discriminating test is K=20: K_eff=10 (H_Kfix) vs K_eff>=14 (H_Kscale)?

FIXED PROTOCOL: same R_LEVELS, same A_RENORM_K20, focus on K_eff_estimated.
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp
import datetime

DATASETS = [
    {
        'name': 'newsgroups-l0',
        'train_path': 'data/cache_embeddings/bge-small/newsgroups/train.npz',
        'test_path': 'data/cache_embeddings/bge-small/newsgroups/test.npz',
        'label_key': 'l0_labels',
        'K_expected': 6,
    },
    {
        'name': 'dbpedia-l0',
        'train_path': 'data/cache_embeddings/bge-small/dbpedia_classes/train.npz',
        'test_path': 'data/cache_embeddings/bge-small/dbpedia_classes/test.npz',
        'label_key': 'l0_labels',
        'K_expected': 9,
    },
    {
        'name': 'clinc-l0',
        'train_path': 'data/cache_embeddings/bge-small/clinc/train.npz',
        'test_path': 'data/cache_embeddings/bge-small/clinc/test.npz',
        'label_key': 'l0_labels',
        'K_expected': 10,
    },
    {
        'name': 'newsgroups-l1',
        'train_path': 'data/cache_embeddings/bge-small/newsgroups/train.npz',
        'test_path': 'data/cache_embeddings/bge-small/newsgroups/test.npz',
        'label_key': 'l1_labels',
        'K_expected': 20,
    },
]

RESULT_PATH = "results/cti_k_scaling_abcs.json"
LOG_PATH = "results/cti_k_scaling_abcs_log.txt"

A_RENORM_K20 = 1.0535   # pre-registered
R_LEVELS = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

# N_TRAIN by K (balanced across K values)
N_TRAIN_BY_K = {6: 400, 9: 400, 10: 400, 20: 200}
RANDOM_SEED = 42

os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')

def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def load_bge_data(train_path, test_path, label_key, n_train_per_class, seed):
    tr = np.load(train_path)
    te = np.load(test_path)
    X_all = np.concatenate([tr['embeddings'].astype(np.float64), te['embeddings'].astype(np.float64)])
    y_all = np.concatenate([tr[label_key], te[label_key]])
    classes = np.unique(y_all)

    rng = np.random.default_rng(seed)
    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    for c in classes:
        idx = np.where(y_all == c)[0]
        rng.shuffle(idx)
        n = min(n_train_per_class, len(idx) - 1)
        X_tr_list.append(X_all[idx[:n]]); y_tr_list.append(y_all[idx[:n]])
        X_te_list.append(X_all[idx[n:]]); y_te_list.append(y_all[idx[n:]])
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
                'delta': float(delta_ij),
                'kappa': float(kappa_ij),
                'd_eff': float(d_eff_ij),
                'kappa_eff': float(kappa_eff_ij),
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


def run_dataset(ds_cfg):
    name = ds_cfg['name']
    K_exp = ds_cfg['K_expected']
    n_train = N_TRAIN_BY_K.get(K_exp, 200)

    log(f"\n{'='*70}")
    log(f"DATASET: {name} (K_expected={K_exp}, n_train={n_train})")
    log(f"{'='*70}")

    X_tr, y_tr, X_te, y_te, classes = load_bge_data(
        ds_cfg['train_path'], ds_cfg['test_path'], ds_cfg['label_key'], n_train, RANDOM_SEED)
    K = len(classes)
    d = X_tr.shape[1]
    log(f"Actual K={K}, d={d}, n_train={len(X_tr)}, n_test={len(X_te)}")

    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)
    log(f"Baseline: acc={acc_base:.4f}, q={q_base:.4f}, logit={logit_base:.4f}")
    log(f"kappa={geo['kappa']:.4f}, d_eff={geo['d_eff']:.4f}, kappa_eff={geo['kappa_eff']:.4f}")

    # Dynamic M_BUNDLE based on K
    if K <= 6:
        m_bundle = list(range(1, K))
    elif K <= 10:
        m_bundle = list(range(1, K))
    elif K <= 20:
        m_bundle = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, K-1]
    else:
        m_bundle = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, K-1]
    m_bundle = sorted(set(m for m in m_bundle if 1 <= m < K))

    log(f"M_BUNDLE: {m_bundle}")

    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])
    tr_bundle_all = compute_subspace_trace(X_tr, y_tr, geo['mu'], classes, U_B)
    tr_W_null = geo['trW'] - tr_bundle_all
    log(f"tr_W_null = {tr_W_null:.4f} ({tr_W_null/geo['trW']*100:.1f}% of trW)")

    # Use A_RENORM_K20 as approximate pre-registered constant for delta_pred
    A_keff = A_RENORM_K20 / float(np.sqrt(geo['d_eff']))
    C_const = logit_base - A_keff * geo['kappa_eff']

    slopes_by_m = {}
    for m in m_bundle:
        U_bundle = build_bundle_directions(geo['pair_info'], m)
        actual_m = U_bundle.shape[1]

        m_results = []
        for r in R_LEVELS:
            X_tr_b, X_te_b = apply_bundle_surgery(
                X_tr, X_te, y_tr, y_te, geo, U_bundle, U_B, r, tr_W_null)
            acc_r, q_r, logit_r = eval_q(X_tr_b, y_tr, X_te_b, y_te, K)
            delta_obs = logit_r - logit_base
            kappa_eff_pred = geo['kappa_eff'] * float(np.sqrt(r))
            delta_pred = (C_const + A_keff * kappa_eff_pred) - logit_base
            ratio = float(delta_obs / delta_pred) if abs(delta_pred) > 1e-6 else 0.0
            m_results.append({
                'r': float(r), 'actual_bundle_rank': int(actual_m),
                'delta_logit': float(delta_obs), 'delta_pred': float(delta_pred),
                'ratio': ratio,
            })

        nontrivial = [x for x in m_results if abs(x['r'] - 1.0) > 0.01]
        if len(nontrivial) >= 3:
            deltas_obs = [x['delta_logit'] for x in nontrivial]
            deltas_pred = [x['delta_pred'] for x in nontrivial]
            if np.std(deltas_pred) > 1e-8:
                slope = float(np.polyfit(deltas_pred, deltas_obs, 1)[0])
                r_val, _ = pearsonr(deltas_obs, deltas_pred)
            else:
                slope, r_val = 0.0, 0.0
        else:
            slope, r_val = 0.0, 0.0
        slopes_by_m[m] = {'slope': slope, 'pearson_r': float(r_val)}

    log(f"\n{'m':>4} | {'slope':>8} | {'pearson_r':>9} | ratio_to_m1")
    slope_m1 = slopes_by_m.get(1, {}).get('slope', 1e-6)
    for m, info in sorted(slopes_by_m.items()):
        s = info['slope']
        r_val = info['pearson_r']
        log(f"{m:>4} | {s:>8.4f} | {r_val:>9.4f} | {s/(slope_m1+1e-10):.3f}")

    m_vals = sorted(slopes_by_m.keys())
    slope_vals = [slopes_by_m[m]['slope'] for m in m_vals]

    n_for_pearson = min(min(K-1, 8), len(m_vals))
    if n_for_pearson >= 3:
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
    pass_A1 = r_scale > 0.90

    log(f"\nSCALE ANALYSIS:")
    log(f"  r_scale = {r_scale:.4f} | A1 PASS: {pass_A1}")
    log(f"  K_eff_estimated (90% sat.) = {k_eff_est} | K_eff_from_single = {k_eff_from_single:.2f}")
    log(f"  K={K}, K_eff_est/K ratio = {(k_eff_est/K):.2f}" if k_eff_est else "")

    return {
        'name': name,
        'K': int(K),
        'd': int(d),
        'baseline': {'acc': float(acc_base), 'q': float(q_base), 'logit': float(logit_base),
                     'kappa': float(geo['kappa']), 'd_eff': float(geo['d_eff']),
                     'kappa_eff': float(geo['kappa_eff'])},
        'slopes_by_m': {str(k): v for k, v in slopes_by_m.items()},
        'analysis': {
            'r_scale': float(r_scale),
            'k_eff_estimated': int(k_eff_est) if k_eff_est else None,
            'k_eff_from_single': float(k_eff_from_single),
            'k_eff_over_K': float(k_eff_est / K) if k_eff_est else None,
            'pass_A1': bool(pass_A1),
        },
    }


def main():
    log("=" * 70)
    log("K-SCALING ABCS: Does K_eff scale with K or is it fixed?")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log("=" * 70)
    log("H_Kscale: K_eff prop K | H_Kfix: K_eff ~ 10 always")
    log("")

    all_results = []
    for ds_cfg in DATASETS:
        result = run_dataset(ds_cfg)
        all_results.append(result)

    # Universality analysis
    log("\n" + "=" * 70)
    log("K-SCALING SUMMARY")
    log("=" * 70)
    log(f"\n{'Dataset':>20} | {'K':>4} | {'K_eff_est':>9} | {'K_eff/K':>7} | {'K_eff_single':>12} | A1")
    log("-" * 70)
    for r in all_results:
        a = r['analysis']
        k_ratio = f"{a['k_eff_over_K']:.2f}" if a['k_eff_over_K'] else "?"
        k_est = str(a['k_eff_estimated']) if a['k_eff_estimated'] else "?"
        log(f"{r['name']:>20} | {r['K']:>4} | {k_est:>9} | {k_ratio:>7} | {a['k_eff_from_single']:>12.2f} | {'YES' if a['pass_A1'] else 'NO'}")

    # Test H_Kscale vs H_Kfix
    k_values = [r['K'] for r in all_results if r['analysis']['k_eff_estimated']]
    k_eff_est_vals = [r['analysis']['k_eff_estimated'] for r in all_results if r['analysis']['k_eff_estimated']]

    if len(k_values) >= 3:
        r_k_scale, p_k_scale = pearsonr(k_values, k_eff_est_vals)
        log(f"\nPearson r(K, K_eff_estimated) = {r_k_scale:.4f} (p={p_k_scale:.4f})")
        if r_k_scale > 0.80:
            verdict = "H_KSCALE: K_eff SCALES with K. Multi-competitor law is density-adaptive."
        elif r_k_scale < 0.40:
            verdict = "H_KFIX: K_eff is approximately constant. Fixed ~10 competitors active regardless of K."
        else:
            verdict = f"AMBIGUOUS: r={r_k_scale:.3f}. Neither pure scale nor pure fixed."
    else:
        r_k_scale = 0.0
        verdict = "INSUFFICIENT DATA for scaling test"

    log(f"\nVERDICT: {verdict}")

    output = {
        'experiment': 'k_scaling_abcs',
        'timestamp': datetime.datetime.now().isoformat(),
        'results': all_results,
        'scaling': {
            'k_values': k_values,
            'k_eff_est_vals': k_eff_est_vals,
            'pearson_r_k_scale': float(r_k_scale) if len(k_values) >= 3 else None,
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
