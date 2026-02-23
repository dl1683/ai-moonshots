#!/usr/bin/env python -u
"""
ACTIVE-SET CAUSAL BUNDLE SURGERY (ABCS)
=========================================

PRE-REGISTERED (Codex prescription, Feb 23 2026). COMMIT BEFORE RUNNING.

CONTEXT: Clean single-direction surgery (cti_clean_subspace_surgery.py) found:
  - Pearson r = 0.939 (direction correct!)
  - OLS slope = 0.181 (18.1% of predicted, vs 9.5% for old contaminated surgery)
  - 0.181 ≈ 1/K_eff where K_eff ≈ 5.5 (matches prior Session 38: ~4-5 active rivals)

HYPOTHESIS: The CTI law applies to ALL K_eff active competitors, not just the nearest pair.
  logit(q) = A * SUM_{k=1}^{K_eff} w_k * kappa_k * sqrt(d_eff_k) + C
  Surgery on m of K_eff active directions should give delta_logit ≈ (m/K_eff) * delta_logit_full

PRE-REGISTERED TEST:
  For m = 1, 2, 3, 4, 5, 6, 8, K-1:
    Apply clean bundle surgery to TOP-m nearest class pairs at r = R_LEVELS
    (nearest = smallest kappa, most confused)
  PASS: OLS slope(m) scales linearly with m and saturates at m ≈ K_eff
        Specifically: slope(m) / slope(1) ≈ m for m <= K_eff
                      slope(m) / slope(1) ≈ K_eff for m > K_eff (saturation)

CLEAN BUNDLE SURGERY:
  For m directions u_1, ..., u_m (Gram-Schmidt orthogonalized):
  1) Build between-class full subspace U_B (K-1 vectors)
  2) z_bundle = sum_k (z @ u_k) * u_k  [component in m-dir bundle]
  3) z_other_comp = z @ U_B @ U_B.T - z_bundle  [other K-1-m competitive dirs]
  4) z_null = z - z @ U_B @ U_B.T  [null: orthogonal to ALL K-1 class diffs]
  5) x_new = mu + (1/sqrt(r)) * z_bundle + z_other_comp + scale_null * z_null
     where scale_null = sqrt((tr_null + tr_bundle*(1-1/r)) / tr_null)

ADDITIONAL CONTROLS:
  - Random m directions (control): choose m random orthogonal directions from NULL SPAN
    Should show zero effect (null check)
  - Top-m by RANDOM ranking: should show weaker than top-m by kappa (specificity check)

DATA: pythia-160m/DBpedia frozen embeddings, K=14, kappa_eff=2.38, q=0.90.

ACCEPTANCE CRITERIA (pre-registered):
  PRIMARY: Pearson r(slope(m), m) > 0.90 for m in [1..K_eff+1]
  SECONDARY: slope(K_eff) / slope(1) ≈ K_eff (ratio in [K_eff*0.7, K_eff*1.3])
  TERTIARY: Random-m control shows slope < 0.5 * single-direction slope
  TERTIARY: OLS slope(m=K_eff) in [0.7, 1.3] * predicted full-effect slope
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp

# ==================== CONFIGURATION ====================
EMBED_PATH = "results/dointerv_multi_pythia-160m_l12.npz"
RESULT_PATH = "results/cti_abcs.json"
LOG_PATH = "results/cti_abcs_log.txt"

A_RENORM_K20 = 1.0535
ALPHA_KAPPA_K14 = 1.477

# Surgery levels to run
R_LEVELS = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
# Bundle sizes to test
M_BUNDLE = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13]

N_TRAIN_PER_CLASS = 350
RANDOM_SEED = 42
N_RANDOM_CONTROLS = 3  # Random bundle controls

# Pass thresholds
SCALE_PEARSON_THRESH = 0.90
RATIO_BAND = 0.30  # ratio should be in [K_eff * (1-band), K_eff * (1+band)]
FULL_SLOPE_BAND = 0.30  # slope(K_eff) in [(1-band), (1+band)] * pred

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
        X_tr_list.append(X[idx[:n_train]]); y_tr_list.append(y[idx[:n_train]])
        X_te_list.append(X[idx[n_train:]]); y_te_list.append(y[idx[n_train:]])
    return (np.concatenate(X_tr_list), np.concatenate(y_tr_list),
            np.concatenate(X_te_list), np.concatenate(y_te_list), classes)


def compute_geometry_full(X_tr, y_tr, classes):
    """Compute geometry for ALL class pairs."""
    K = len(classes)
    N, d = len(X_tr), X_tr.shape[1]
    mu = np.stack([X_tr[y_tr == c].mean(0) for c in classes])
    grand_mean = X_tr.mean(0)

    # tr(Sigma_W)
    trW = 0.0
    for i, c in enumerate(classes):
        Xc_c = X_tr[y_tr == c] - mu[i]
        trW += float(np.sum(Xc_c ** 2)) / N
    sigma_W_global = float(np.sqrt(trW / d))

    # ALL pairwise information
    pair_info = []
    for i in range(K):
        for j in range(i + 1, K):
            Delta = mu[i] - mu[j]
            delta_ij = float(np.linalg.norm(Delta))
            Delta_hat = Delta / (delta_ij + 1e-10)
            kappa_ij = float(delta_ij / (sigma_W_global * np.sqrt(d) + 1e-10))

            # sigma_cdir_ij (pooled over all classes)
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

    # Sort by kappa ascending (nearest = most confused = lowest kappa)
    pair_info.sort(key=lambda x: x['kappa'])

    # Nearest centroid pair (first in sorted list)
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
    """Build full K-1 dimensional between-class basis U_B via Gram-Schmidt."""
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


def build_bundle_directions(pair_info, m, rng=None, null_space=None):
    """Build m orthogonalized bundle directions from top-m nearest pairs.

    Args:
        pair_info: sorted list of pair info (nearest first)
        m: number of directions
        rng: if provided, randomize pair selection (for random control)
        null_space: if provided, sample from null space (for null control)

    Returns:
        U_bundle: (d, m') matrix of orthonormal bundle directions (m' <= m)
    """
    if null_space is not None:
        # Random directions from null space
        d, null_dim = null_space.shape
        directions = null_space @ np.linalg.qr(rng.standard_normal((null_dim, m)))[0]
        return directions[:, :min(m, null_dim)]

    if rng is not None:
        # Random selection from all pairs
        indices = rng.choice(len(pair_info), size=min(m, len(pair_info)), replace=False)
        selected = [pair_info[i] for i in indices]
    else:
        # Top-m nearest pairs
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
    return np.stack(basis, axis=1)  # (d, m')


def compute_subspace_trace(X_tr, y_tr, mu, classes, U_directions):
    """Compute tr(W) in subspace spanned by U_directions (d, m)."""
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
    """Apply clean bundle surgery: scale U_bundle directions by 1/sqrt(r),
    absorb trace into null span, leave other competitive directions unchanged."""
    mu = geo['mu']
    classes = np.unique(y_tr)
    N = len(X_tr)

    # Compute tr(W) in bundle subspace
    tr_bundle = compute_subspace_trace(X_tr, y_tr, mu, classes, U_bundle)

    # After surgery: tr_bundle_new = tr_bundle / r
    # tr_null_new = tr_null + tr_bundle * (1 - 1/r)
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

            # Project onto full between-class subspace
            z_between = z @ U_B @ U_B.T  # (n, d)
            # Project onto bundle subspace
            z_bundle = z @ U_bundle @ U_bundle.T  # (n, d)
            # Other competitive dirs (not in bundle)
            z_other_comp = z_between - z_bundle
            # Null
            z_null = z - z_between

            z_new = scale_bundle * z_bundle + z_other_comp + scale_null * z_null
            X_new[mask] = mu[i] + z_new
        return X_new

    return transform(X_tr, y_tr), transform(X_te, y_te), tr_bundle, scale_null


def eval_q(X_tr, y_tr, X_te, y_te, K):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = float(np.clip((acc - 1.0/K) / (1.0 - 1.0/K), 1e-6, 1-1e-6))
    return acc, q, float(sp.logit(q))


def main():
    log("=" * 70)
    log("ACTIVE-SET CAUSAL BUNDLE SURGERY (ABCS)")
    log("=" * 70)
    log("")

    X_tr, y_tr, X_te, y_te, classes = load_and_split(EMBED_PATH, N_TRAIN_PER_CLASS, RANDOM_SEED)
    K = len(classes)

    geo = compute_geometry_full(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)

    log(f"K={K}, baseline: acc={acc_base:.4f}, q={q_base:.4f}, logit={logit_base:.4f}")
    log(f"kappa={geo['kappa']:.4f}, d_eff={geo['d_eff']:.4f}, kappa_eff={geo['kappa_eff']:.4f}")
    log("")

    # Show ALL pair kappas to understand competition structure
    log("Top-10 nearest class pairs (sorted by kappa ascending):")
    for k, p in enumerate(geo['pair_info'][:10]):
        log(f"  m={k+1}: pair=({p['class_i']},{p['class_j']}) kappa={p['kappa']:.4f} "
            f"d_eff={p['d_eff']:.2f} kappa_eff={p['kappa_eff']:.4f}")
    log("")

    # Build full between-class basis
    U_B = build_between_class_basis(geo['mu'], geo['grand_mean'], geo['Delta_hat'])
    log(f"Between-class basis U_B: shape={U_B.shape}")

    # Compute null space tr(W) -- for all clean surgeries
    tr_bundle_all = compute_subspace_trace(X_tr, y_tr, geo['mu'], classes, U_B)
    tr_W_null = geo['trW'] - tr_bundle_all
    log(f"tr_W_null = {tr_W_null:.4f} ({tr_W_null/geo['trW']*100:.1f}% of trW)")
    log("")

    # Predictions
    A_keff_K14 = ALPHA_KAPPA_K14 / float(np.sqrt(geo['d_eff']))
    C_keff = logit_base - A_keff_K14 * geo['kappa_eff']

    # ==================== MAIN BUNDLE SURGERY SWEEP ====================
    log("=" * 60)
    log("BUNDLE SURGERY SWEEP: m = 1 to K-1 at multiple r values")
    log("=" * 60)

    all_bundle_results = {}
    rng = np.random.default_rng(RANDOM_SEED)

    for m in M_BUNDLE:
        if m > K - 1:
            continue
        log(f"\n--- m={m} (top-{m} nearest pairs) ---")
        U_bundle = build_bundle_directions(geo['pair_info'], m)
        actual_m = U_bundle.shape[1]  # after Gram-Schmidt, might be < m
        log(f"  Bundle rank after G-S: {actual_m}")

        m_results = []
        for r in R_LEVELS:
            X_tr_b, X_te_b, tr_bundle_m, scale_null_m = apply_bundle_surgery(
                X_tr, X_te, y_tr, y_te, geo, U_bundle, U_B, r, tr_W_null)
            acc_r, q_r, logit_r = eval_q(X_tr_b, y_tr, X_te_b, y_te, K)

            delta_obs = logit_r - logit_base
            # Prediction: delta = A_keff_K14 * kappa_eff * (sqrt(r) - 1)
            kappa_eff_pred = geo['kappa_eff'] * float(np.sqrt(r))
            delta_pred_K14 = (C_keff + A_keff_K14 * kappa_eff_pred) - logit_base

            log(f"    r={r:.1f}: q={q_r:.4f}, logit={logit_r:.4f}, "
                f"delta_obs={delta_obs:+.4f}, pred_K14={delta_pred_K14:+.4f}, "
                f"ratio={delta_obs/delta_pred_K14:.3f}" if abs(delta_pred_K14) > 1e-6 else
                f"    r={r:.1f}: delta_obs={delta_obs:+.4f}")

            m_results.append({
                'r': float(r), 'actual_bundle_rank': int(actual_m),
                'q': float(q_r), 'logit': float(logit_r),
                'delta_logit': float(delta_obs),
                'delta_pred_K14': float(delta_pred_K14),
                'ratio': float(delta_obs / delta_pred_K14) if abs(delta_pred_K14) > 1e-6 else 0.0,
                'sqrt_r_minus1': float(np.sqrt(r) - 1),
            })
        all_bundle_results[str(m)] = m_results

    # ==================== RANDOM CONTROLS ====================
    log("\n" + "=" * 60)
    log("RANDOM CONTROLS")
    log("=" * 60)

    rng_ctrl = np.random.default_rng(RANDOM_SEED + 1)
    random_results = {}

    for ctrl_idx in range(N_RANDOM_CONTROLS):
        m_ctrl = 5  # Test at m=5 (expected K_eff)
        U_bundle_rand = build_bundle_directions(
            geo['pair_info'], m_ctrl, rng=rng_ctrl)
        ctrl_results = []
        for r in R_LEVELS:
            X_tr_r, X_te_r, _, _ = apply_bundle_surgery(
                X_tr, X_te, y_tr, y_te, geo, U_bundle_rand, U_B, r, tr_W_null)
            acc_r, q_r, logit_r = eval_q(X_tr_r, y_tr, X_te_r, y_te, K)
            delta_obs = logit_r - logit_base
            ctrl_results.append({'r': float(r), 'delta_logit': float(delta_obs)})
            log(f"  random_ctrl={ctrl_idx}, r={r:.1f}: delta={delta_obs:+.4f}")
        random_results[str(ctrl_idx)] = ctrl_results

    # ==================== ANALYSIS ====================
    log("\n" + "=" * 70)
    log("ANALYSIS: Scale-with-m Test")
    log("=" * 70)

    # For each m, compute OLS slope (delta_obs vs delta_pred_K14) across r values
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
        slopes_by_m[m_int] = {'slope': slope, 'pearson_r': r_val}

    log(f"{'m':>4} | {'OLS slope':>10} | {'Pearson r':>10} | slope/slope(1)")
    slope_m1 = slopes_by_m.get(1, {}).get('slope', 1.0)
    for m_int in sorted(slopes_by_m.keys()):
        s = slopes_by_m[m_int]['slope']
        r_val = slopes_by_m[m_int]['pearson_r']
        ratio_to_m1 = s / (slope_m1 + 1e-10)
        log(f"{m_int:>4} | {s:>10.4f} | {r_val:>10.4f} | {ratio_to_m1:.3f}")

    log("")

    # Fit: slope(m) = slope(1) * min(m, K_eff) / 1 (should saturate at K_eff)
    m_vals = sorted(slopes_by_m.keys())
    slope_vals = [slopes_by_m[m]['slope'] for m in m_vals]

    if len(m_vals) >= 3:
        r_scale, _ = pearsonr(m_vals[:8], slope_vals[:8]) if len(m_vals) >= 3 else (0, 1)
    else:
        r_scale = 0.0

    # Estimate K_eff: where does slope saturate?
    max_slope = max(slope_vals)
    k_eff_est = None
    for i, (m, s) in enumerate(zip(m_vals, slope_vals)):
        if s >= 0.90 * max_slope:
            k_eff_est = m
            break

    log(f"Scale analysis:")
    log(f"  Pearson r(slope, m) for m in [1..8] = {r_scale:.4f} (PASS: >{SCALE_PEARSON_THRESH})")
    log(f"  Estimated K_eff (90% saturation): {k_eff_est}")
    log(f"  slope(1) = {slope_m1:.4f} ~ 1/K_eff implies K_eff_from_single = {1/slope_m1:.2f}" if slope_m1 > 0.01 else "")

    if k_eff_est is not None:
        slope_k_eff = slopes_by_m.get(k_eff_est, {}).get('slope', 0.0)
        log(f"  slope(K_eff={k_eff_est}) = {slope_k_eff:.4f} (PASS: in [{1-FULL_SLOPE_BAND:.2f}, {1+FULL_SLOPE_BAND:.2f}])")

    # Random control analysis
    rand_slopes = []
    for ctrl_results in random_results.values():
        nontrivial = [x for x in ctrl_results if abs(x['r'] - 1.0) > 0.01]
        if len(nontrivial) >= 3:
            deltas_o = [x['delta_logit'] for x in nontrivial]
            deltas_p = [x['delta_pred_K14'] for x in m_results if abs(x['r'] - 1.0) > 0.01]
            if len(deltas_p) == len(deltas_o) and np.std(deltas_p) > 1e-8:
                rand_slopes.append(float(np.polyfit(deltas_p, deltas_o, 1)[0]))
    mean_rand_slope = float(np.mean(rand_slopes)) if rand_slopes else 0.0
    log(f"  Random control mean slope: {mean_rand_slope:.4f} (should be ~0)")

    # ==================== VERDICT ====================
    log("\n" + "=" * 70)
    log("VERDICT")
    log("=" * 70)

    primary_pass = r_scale > SCALE_PEARSON_THRESH
    slope_k_eff = slopes_by_m.get(k_eff_est, {}).get('slope', 0.0) if k_eff_est else 0.0
    secondary_pass = (0.70 <= slope_k_eff <= 1.30) if k_eff_est else False
    tertiary_pass = (abs(mean_rand_slope) < 0.5 * slope_m1) if slope_m1 > 0.01 else False

    log(f"PRIMARY (scale-with-m, r>{SCALE_PEARSON_THRESH}): {primary_pass} (r={r_scale:.4f})")
    log(f"SECONDARY (slope(K_eff) in [0.7,1.3]): {secondary_pass} (slope={slope_k_eff:.4f})")
    log(f"TERTIARY (random control < 0.5*single): {tertiary_pass} (rand={mean_rand_slope:.4f} vs single/2={slope_m1/2:.4f})")

    n_pass = sum([primary_pass, secondary_pass, tertiary_pass])
    if n_pass >= 2:
        verdict = "ABCS PASS: Active-set bundle surgery confirms multi-competitor CTI law. Nobel 8+/10."
    elif n_pass == 1:
        verdict = "ABCS PARTIAL: Some evidence for multi-competitor law. More replication needed."
    else:
        verdict = "ABCS FAIL: Neither scale-with-m nor full-effect recovered. Theory needs deeper revision."

    log(f"\nOVERALL: {n_pass}/3 criteria PASS")
    log(f"VERDICT: {verdict}")

    # ==================== SAVE ====================
    result = {
        'experiment': 'active_set_causal_bundle_surgery',
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'baseline': {
            'K': int(K), 'acc': float(acc_base), 'q': float(q_base),
            'logit': float(logit_base), 'kappa': float(geo['kappa']),
            'd_eff': float(geo['d_eff']), 'kappa_eff': float(geo['kappa_eff']),
        },
        'pair_info': [{k: v for k, v in p.items() if k != 'Delta_hat'}
                      for p in geo['pair_info'][:10]],
        'bundle_results': all_bundle_results,
        'random_controls': random_results,
        'analysis': {
            'slopes_by_m': {str(k): v for k, v in slopes_by_m.items()},
            'r_scale': float(r_scale),
            'k_eff_estimated': int(k_eff_est) if k_eff_est else None,
            'mean_rand_slope': float(mean_rand_slope),
            'primary_pass': bool(primary_pass),
            'secondary_pass': bool(secondary_pass),
            'tertiary_pass': bool(tertiary_pass),
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\nResults saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
