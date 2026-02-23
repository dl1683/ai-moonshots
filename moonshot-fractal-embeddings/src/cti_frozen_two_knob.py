#!/usr/bin/env python -u
"""
FROZEN EMBEDDING TWO-KNOB CAUSAL TEST
======================================

PRE-REGISTERED (Codex prescription, Feb 23 2026). COMMIT BEFORE RUNNING.

MOTIVATION: Linear regime surgery (epoch 4, kappa_eff=1.0, K=20) showed d_eff surgery
barely changes q (A_empirical = 0.050 vs A_renorm = 1.054, ratio=5%). This could be:
  (a) Wrong regime: kappa_eff too low, K=20 dense competition invalidates Gumbel race
  (b) Theory revision: kappa alone is causal, d_eff is observational confound

FIX: Use FROZEN pretrained embeddings at kappa_eff~1.9 (high regime, sparse competition).
  - Data: pythia-160m/DBpedia layer 12 (K=14, kappa_eff~1.9, q~0.90)
  - K=14 (sparser than K=20: Gumbel race approximation holds better)
  - kappa_eff=1.9 (vs 1.0 in linear regime: much further from random)

TWO INDEPENDENT KNOBS:
  Knob 1 (kappa):  Scale class centroids around grand mean by factor s.
    Effect: kappa -> s*kappa, d_eff UNCHANGED, kappa_eff -> s*kappa_eff.
  Knob 2 (d_eff):  Redistribute within-class variance along nearest centroid direction.
    scale_along=1/sqrt(r), scale_perp=sqrt((trW-sigma_cdir^2/r)/(trW-sigma_cdir^2))
    Effect: d_eff -> r*d_eff, kappa UNCHANGED, kappa_eff -> kappa*sqrt(r*d_eff).

PRE-REGISTERED HYPOTHESES (mutually exclusive):
  H_kappa (kappa alone is causal):
    - d_eff surgery arm: max|delta_logit| < 0.05 for r in [0.25, 4.0]  [NULL ON d_eff]
    - kappa surgery arm: Pearson r > 0.98 with logit ~ s*kappa (linear in s)
    - Matched pairs test: |Delta_logit| > 0.15 for same-kappa-diff-d_eff pairs
      (because kappa_eff formula would predict same q, but kappa formula predicts DIFFERENT q)
    PASS if: max_deff_delta < 0.05 AND kappa_pearson > 0.98

  H_keff (kappa_eff = kappa*sqrt(d_eff) is sufficient):
    - d_eff surgery arm: Pearson r(delta_logit_obs, sqrt(r)-1) > 0.95
      AND calibration ratio A_emp/A_renorm in [0.80, 1.20]
    - Matched pairs test: |Delta_logit| < 0.10 for same-kappa_eff-diff-decomp pairs
    PASS if: deff_pearson > 0.95 AND |calib_ratio - 1.0| < 0.20 AND mean_matched_delta < 0.10

Pre-registered constants:
  alpha_kappa_K14 = 1.477  (from multi-arch do-intervention, DBpedia K=14, pre-registered)
  A_renorm_K20 = 1.0535    (Theorem 15, used for kappa_eff formula prediction)
  A_keff_K14 = alpha_kappa_K14 / sqrt(d_eff_base)  [derived, not free parameter]
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
import scipy.special as sp

# ==================== CONFIGURATION ====================
EMBED_PATH = "results/dointerv_multi_pythia-160m_l12.npz"
RESULT_PATH = "results/cti_frozen_two_knob.json"
LOG_PATH = "results/cti_frozen_two_knob_log.txt"

# Pre-registered constants
ALPHA_KAPPA_K14 = 1.477    # kappa-only formula slope for K=14 DBpedia
A_RENORM_K20 = 1.0535      # Theorem 15, K=20

# Surgery grids (pre-registered)
DEFF_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]  # r values
KAPPA_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5]  # s values

# Train/test split per class
N_TRAIN_PER_CLASS = 350   # 4900 total train
# Rest used for test: 500 - 350 = 150 per class = 2100 total test

# Pre-registered pass thresholds
NULL_DEFF_MAX_DELTA = 0.05   # |delta_logit| < 0.05 => d_eff causally inactive (H_kappa wins)
KAPPA_PEARSON_THRESH = 0.98  # r > 0.98 for kappa surgery (both hypotheses predict this)
DEFF_PEARSON_THRESH = 0.95   # r > 0.95 for d_eff surgery (only H_keff predicts this)
CALIB_RATIO_BAND = 0.20      # |A_emp/A_keff - 1| < 0.20 (H_keff calibration)
PAIR_MATCH_THRESH = 0.10     # |Delta_logit| < 0.10 (H_keff matched pairs)
PAIR_KAPPA_MIN_DELTA = 0.15  # matched pairs: kappa differs by >=15% (H_kappa prediction)

RANDOM_SEED = 42

# ==================== LOGGING ====================
os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')

def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


# ==================== DATA LOADING ====================
def load_and_split(path, n_train_per_class, seed):
    data = np.load(path)
    X, y = data['X'], data['y']
    classes = np.unique(y)
    rng = np.random.default_rng(seed)

    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        X_tr_list.append(X[idx[:n_train_per_class]])
        y_tr_list.append(y[idx[:n_train_per_class]])
        X_te_list.append(X[idx[n_train_per_class:]])
        y_te_list.append(y[idx[n_train_per_class:]])

    X_tr = np.concatenate(X_tr_list).astype(np.float64)
    y_tr = np.concatenate(y_tr_list)
    X_te = np.concatenate(X_te_list).astype(np.float64)
    y_te = np.concatenate(y_te_list)
    return X_tr, y_tr, X_te, y_te, classes


# ==================== GEOMETRY ====================
def compute_geometry(X_tr, y_tr, classes):
    K = len(classes)
    N = len(X_tr)
    d = X_tr.shape[1]

    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])
    grand_mean = X_tr.mean(0)

    # tr(Sigma_W): total within-class variance / N
    trW = 0.0
    for i, c in enumerate(classes):
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[i]
        trW += float(np.sum(Xc_c ** 2)) / N

    sigma_W_global = float(np.sqrt(trW / d))

    # Nearest centroid pair
    min_dist = np.inf
    j1, j2 = 0, 1
    for i in range(K):
        for j in range(i + 1, K):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist = dist
                j1, j2 = i, j

    delta_min = float(min_dist)
    Delta = centroids[j1] - centroids[j2]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)

    # sigma_centroid_dir^2 = Delta_hat^T Sigma_W Delta_hat
    sigma_cdir_sq = 0.0
    for i, c in enumerate(classes):
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[i]
        proj = Xc_c @ Delta_hat
        sigma_cdir_sq += (n_c / N) * float(np.mean(proj ** 2))

    kappa = float(delta_min / (sigma_W_global * np.sqrt(d) + 1e-10))
    d_eff = float(trW / (sigma_cdir_sq + 1e-10))
    kappa_eff = kappa * float(np.sqrt(d_eff))
    sigma_cdir = float(np.sqrt(sigma_cdir_sq))

    return {
        'centroids': centroids,
        'grand_mean': grand_mean,
        'Delta_hat': Delta_hat,
        'trW': trW,
        'sigma_W_global': sigma_W_global,
        'sigma_cdir_sq': sigma_cdir_sq,
        'sigma_cdir': sigma_cdir,
        'd_eff': d_eff,
        'kappa': kappa,
        'kappa_eff': kappa_eff,
        'delta_min': delta_min,
        'nearest_pair': (int(j1), int(j2)),
        'K': K, 'd': d,
    }


# ==================== SURGERY OPERATIONS ====================
def apply_deff_surgery(X_tr, X_te, y_tr, y_te, geo, r, recompute_centroids=False):
    """d_eff surgery: scale_along=1/sqrt(r), preserve tr(W). kappa unchanged.

    If recompute_centroids=True, recomputes centroids from X_tr before surgery
    (needed when applying d_eff surgery after kappa surgery on same embeddings).
    Delta_hat and sigma_cdir_sq are NOT recomputed (kappa surgery preserves them).
    """
    Delta_hat = geo['Delta_hat']
    trW = geo['trW']
    sigma_cdir_sq = geo['sigma_cdir_sq']
    classes = np.unique(y_tr)

    if recompute_centroids:
        centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])
    else:
        centroids = geo['centroids']

    # Validity check
    min_r = sigma_cdir_sq / (trW + 1e-10) * 1.001
    if r < min_r:
        r = min_r  # clamp

    scale_along = 1.0 / float(np.sqrt(r))
    denom = trW - sigma_cdir_sq
    num = trW - sigma_cdir_sq / r
    if denom < 1e-12:
        scale_perp = 1.0
    else:
        scale_perp = float(np.sqrt(max(0.0, num / denom)))

    def transform(X, y):
        X_new = X.copy()
        for i, c in enumerate(classes):
            mask = (y == c)
            Xc = X[mask]
            z = Xc - centroids[i]
            proj = z @ Delta_hat
            z_along = proj[:, None] * Delta_hat[None, :]
            z_perp = z - z_along
            X_new[mask] = centroids[i] + scale_along * z_along + scale_perp * z_perp
        return X_new

    X_tr_new = transform(X_tr, y_tr)
    X_te_new = transform(X_te, y_te)
    return X_tr_new, X_te_new


def apply_kappa_surgery(X_tr, X_te, y_tr, y_te, geo, s):
    """kappa surgery: scale centroids around grand mean by s. d_eff unchanged."""
    centroids = geo['centroids']
    grand_mean = geo['grand_mean']
    classes = np.unique(y_tr)

    def transform(X, y):
        X_new = X.copy()
        for i, c in enumerate(classes):
            mu_new = grand_mean + s * (centroids[i] - grand_mean)
            mask = (y == c)
            # shift: X_new = mu_new + (X - mu_old) = X + (mu_new - mu_old)
            X_new[mask] = X[mask] + (mu_new - centroids[i])
        return X_new

    X_tr_new = transform(X_tr, y_tr)
    X_te_new = transform(X_te, y_te)
    return X_tr_new, X_te_new


# ==================== EVALUATION ====================
def eval_q(X_tr, y_tr, X_te, y_te, K):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    q = float(np.clip(q, 1e-6, 1 - 1e-6))
    logit_q = float(sp.logit(q))
    return acc, q, logit_q


# ==================== MAIN ====================
def main():
    log("=" * 70)
    log("FROZEN EMBEDDING TWO-KNOB CAUSAL TEST")
    log("=" * 70)
    log(f"Data: {EMBED_PATH}")
    log(f"Pre-registered alpha_kappa_K14 = {ALPHA_KAPPA_K14}")
    log(f"Pre-registered A_renorm_K20    = {A_RENORM_K20}")
    log(f"d_eff levels: {DEFF_LEVELS}")
    log(f"kappa levels: {KAPPA_LEVELS}")
    log("")

    # Load data
    X_tr, y_tr, X_te, y_te, classes = load_and_split(
        EMBED_PATH, N_TRAIN_PER_CLASS, RANDOM_SEED)
    K = len(classes)
    log(f"K={K}, X_tr={X_tr.shape}, X_te={X_te.shape}")

    # Baseline geometry + accuracy
    geo = compute_geometry(X_tr, y_tr, classes)
    acc_base, q_base, logit_base = eval_q(X_tr, y_tr, X_te, y_te, K)

    log(f"Baseline: acc={acc_base:.4f}, q={q_base:.4f}, logit={logit_base:.4f}")
    log(f"  kappa={geo['kappa']:.4f}, d_eff={geo['d_eff']:.4f}, kappa_eff={geo['kappa_eff']:.4f}")
    log(f"  sigma_cdir={geo['sigma_cdir']:.4f}, trW={geo['trW']:.4f}")
    log(f"  Nearest pair: classes {geo['nearest_pair']}")
    log("")

    # Pre-computed formula constants
    A_keff_K14 = ALPHA_KAPPA_K14 / float(np.sqrt(geo['d_eff']))
    C_kappa = logit_base - ALPHA_KAPPA_K14 * geo['kappa']
    C_keff = logit_base - A_keff_K14 * geo['kappa_eff']
    C_keff_K20 = logit_base - A_RENORM_K20 * geo['kappa_eff']

    log(f"Derived A_keff_K14 = alpha_kappa / sqrt(d_eff) = {ALPHA_KAPPA_K14:.3f} / sqrt({geo['d_eff']:.2f}) = {A_keff_K14:.4f}")
    log(f"C_kappa = {C_kappa:.4f} (kappa-only offset)")
    log(f"C_keff  = {C_keff:.4f} (kappa_eff K14 offset)")
    log(f"C_keff_K20 = {C_keff_K20:.4f} (kappa_eff K20 offset)")
    log("")

    # ==================== ARM 1: d_eff SURGERY (kappa fixed) ====================
    log("=" * 50)
    log("ARM 1: d_eff SURGERY (kappa fixed)")
    log("  kappa_eff theory: delta_logit = A_keff * kappa * sqrt(d_eff) * (sqrt(r) - 1)")
    log("  kappa-only theory: delta_logit = 0 for all r")
    log("=" * 50)

    deff_results = []
    for r in DEFF_LEVELS:
        X_tr_s, X_te_s = apply_deff_surgery(X_tr, X_te, y_tr, y_te, geo, r)
        acc_r, q_r, logit_r = eval_q(X_tr_s, y_tr, X_te_s, y_te, K)

        # Verify surgery
        geo_new = compute_geometry(X_tr_s, y_tr, classes)
        kappa_chg = abs(geo_new['kappa'] - geo['kappa']) / (geo['kappa'] + 1e-10) * 100
        d_eff_actual = geo_new['d_eff']

        # Predictions
        kappa_eff_pred = geo['kappa'] * np.sqrt(r * geo['d_eff'])
        logit_pred_keff_K14 = C_keff + A_keff_K14 * kappa_eff_pred
        logit_pred_keff_K20 = C_keff_K20 + A_RENORM_K20 * kappa_eff_pred
        logit_pred_kappa = logit_base  # kappa unchanged, so no change predicted

        delta_obs = logit_r - logit_base
        delta_pred_keff_K14 = logit_pred_keff_K14 - logit_base
        delta_pred_keff_K20 = logit_pred_keff_K20 - logit_base

        log(f"  [r={r:5.2f}] q={q_r:.4f}, logit={logit_r:.4f} | "
            f"delta_obs={delta_obs:+.4f} | "
            f"pred_keff_K14={delta_pred_keff_K14:+.4f} | "
            f"pred_keff_K20={delta_pred_keff_K20:+.4f} | "
            f"kappa_chg={kappa_chg:.3f}%")

        deff_results.append({
            'r': float(r),
            'd_eff_actual': float(d_eff_actual),
            'kappa_chg_pct': float(kappa_chg),
            'acc': float(acc_r),
            'q': float(q_r),
            'logit': float(logit_r),
            'delta_logit_obs': float(delta_obs),
            'delta_logit_pred_keff_K14': float(delta_pred_keff_K14),
            'delta_logit_pred_keff_K20': float(delta_pred_keff_K20),
            'sqrt_r_minus1': float(np.sqrt(r) - 1),
        })

    # Arm 1 analysis
    log("")
    # Filter out r=1 (no change, trivial)
    nontrivial_deff = [x for x in deff_results if abs(x['r'] - 1.0) > 0.01]
    deltas_obs = [x['delta_logit_obs'] for x in nontrivial_deff]
    sqrt_r_minus1 = [x['sqrt_r_minus1'] for x in nontrivial_deff]
    deltas_pred_K14 = [x['delta_logit_pred_keff_K14'] for x in nontrivial_deff]

    max_abs_delta = float(np.max(np.abs(deltas_obs)))
    r_arm1_keff, _ = pearsonr(deltas_obs, sqrt_r_minus1) if len(deltas_obs) > 2 else (0, 1)
    r_arm1_pred_K14, _ = pearsonr(deltas_obs, deltas_pred_K14) if len(deltas_obs) > 2 else (0, 1)

    # Calibration: empirical A / predicted A
    # delta_obs = A_emp * kappa * sqrt(d_eff) * (sqrt(r)-1) = A_emp/A_keff * delta_pred
    # Use OLS slope of delta_obs on delta_pred_K14
    if len(deltas_pred_K14) > 2 and np.std(deltas_pred_K14) > 1e-8:
        slope_K14 = float(np.polyfit(deltas_pred_K14, deltas_obs, 1)[0])
    else:
        slope_K14 = 0.0

    log(f"Arm 1 (d_eff surgery) summary:")
    log(f"  max |delta_logit_obs| = {max_abs_delta:.4f} (H_kappa null: <{NULL_DEFF_MAX_DELTA})")
    log(f"  Pearson r(delta_obs, sqrt(r)-1) = {r_arm1_keff:.4f} (H_keff: >{DEFF_PEARSON_THRESH})")
    log(f"  Pearson r(delta_obs, delta_pred_K14) = {r_arm1_pred_K14:.4f}")
    log(f"  OLS slope delta_obs/delta_pred_K14 = {slope_K14:.4f} (H_keff: ~1.0)")
    null_deff_pass = max_abs_delta < NULL_DEFF_MAX_DELTA
    keff_deff_pass = (r_arm1_keff > DEFF_PEARSON_THRESH and
                      abs(slope_K14 - 1.0) < CALIB_RATIO_BAND)
    log(f"  H_kappa NULL PASS (d_eff inactive): {null_deff_pass}")
    log(f"  H_keff  CAUSAL PASS (d_eff active): {keff_deff_pass}")
    log("")

    # ==================== ARM 2: kappa SURGERY (d_eff fixed) ====================
    log("=" * 50)
    log("ARM 2: kappa SURGERY (d_eff fixed)")
    log("  Both theories predict: logit changes proportionally to s*kappa_eff")
    log("=" * 50)

    kappa_results = []
    for s in KAPPA_LEVELS:
        X_tr_s, X_te_s = apply_kappa_surgery(X_tr, X_te, y_tr, y_te, geo, s)
        acc_s, q_s, logit_s = eval_q(X_tr_s, y_tr, X_te_s, y_te, K)

        # Verify surgery
        geo_new = compute_geometry(X_tr_s, y_tr, classes)
        kappa_new = geo_new['kappa']
        d_eff_new = geo_new['d_eff']
        d_eff_chg = abs(d_eff_new - geo['d_eff']) / (geo['d_eff'] + 1e-10) * 100

        kappa_eff_new = s * geo['kappa_eff']
        logit_pred_keff = C_keff + A_keff_K14 * kappa_eff_new
        logit_pred_kappa = C_kappa + ALPHA_KAPPA_K14 * kappa_new

        delta_obs = logit_s - logit_base

        log(f"  [s={s:5.3f}] q={q_s:.4f}, logit={logit_s:.4f} | "
            f"kappa_new={kappa_new:.4f} (expected {s*geo['kappa']:.4f}) | "
            f"d_eff_chg={d_eff_chg:.2f}% | "
            f"pred_keff={logit_pred_keff:.4f} | pred_kappa={logit_pred_kappa:.4f}")

        kappa_results.append({
            's': float(s),
            'kappa_new': float(kappa_new),
            'kappa_expected': float(s * geo['kappa']),
            'd_eff_new': float(d_eff_new),
            'd_eff_chg_pct': float(d_eff_chg),
            'acc': float(acc_s),
            'q': float(q_s),
            'logit': float(logit_s),
            'delta_logit_obs': float(delta_obs),
            'logit_pred_keff': float(logit_pred_keff),
            'logit_pred_kappa': float(logit_pred_kappa),
            'kappa_eff_new': float(kappa_eff_new),
        })

    # Arm 2 analysis
    log("")
    logit_obs_kappa = [x['logit'] for x in kappa_results]
    logit_pred_kappa_arm = [x['logit_pred_kappa'] for x in kappa_results]
    logit_pred_keff_arm = [x['logit_pred_keff'] for x in kappa_results]
    kappa_new_vals = [x['kappa_new'] for x in kappa_results]

    r_arm2_kappa, _ = pearsonr(logit_obs_kappa, kappa_new_vals) if len(logit_obs_kappa) > 2 else (0, 1)
    r_arm2_keff_pred, _ = pearsonr(logit_obs_kappa, logit_pred_keff_arm) if len(logit_obs_kappa) > 2 else (0, 1)
    r_arm2_kappa_pred, _ = pearsonr(logit_obs_kappa, logit_pred_kappa_arm) if len(logit_obs_kappa) > 2 else (0, 1)

    if len(logit_pred_kappa_arm) > 2 and np.std(logit_pred_kappa_arm) > 1e-8:
        slope_kappa_arm = float(np.polyfit(logit_pred_kappa_arm, logit_obs_kappa, 1)[0])
    else:
        slope_kappa_arm = 0.0

    log(f"Arm 2 (kappa surgery) summary:")
    log(f"  Pearson r(logit_obs, kappa_new) = {r_arm2_kappa:.4f}")
    log(f"  Pearson r(logit_obs, pred_kappa) = {r_arm2_kappa_pred:.4f}")
    log(f"  Pearson r(logit_obs, pred_keff)  = {r_arm2_keff_pred:.4f}")
    log(f"  OLS slope obs/pred_kappa = {slope_kappa_arm:.4f}")
    log("")

    # ==================== ARM 3: MATCHED PAIRS TEST ====================
    log("=" * 50)
    log("ARM 3: MATCHED PAIRS (same kappa_eff, different decomposition)")
    log("=" * 50)

    # Build grid of (s, r) pairs and collect kappa_eff and logit for each
    # kappa_eff(s,r) = s * kappa_base * sqrt(r * d_eff_base) = s*sqrt(r) * kappa_eff_base
    # We want pairs (s1,r1) and (s2,r2) s.t.:
    #   |s1*sqrt(r1) - s2*sqrt(r2)| / mean <= 5%  (same kappa_eff)
    #   |s1 - s2| / mean >= 20%                    (different kappa)
    #   |r1 - r2| / mean >= 20%                    (different d_eff)

    # Collect all grid points
    grid_results = []
    log("Computing grid of (s,r) combinations for matched pairs...")
    # Use a subset for speed
    s_grid = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    r_grid = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 4.0]

    for s in s_grid:
        for r in r_grid:
            # Apply both surgeries
            X_tr_s, X_te_s = apply_kappa_surgery(X_tr, X_te, y_tr, y_te, geo, s)
            # Use recompute_centroids=True: kappa surgery moved centroids,
            # but Delta_hat and sigma_cdir_sq are unchanged (within-class geometry intact)
            X_tr_sr, X_te_sr = apply_deff_surgery(
                X_tr_s, X_te_s, y_tr, y_te, geo, r, recompute_centroids=True)
            acc_sr, q_sr, logit_sr = eval_q(X_tr_sr, y_tr, X_te_sr, y_te, K)
            kappa_eff_sr = s * np.sqrt(r) * geo['kappa_eff']

            grid_results.append({
                's': float(s), 'r': float(r),
                'kappa_eff_target': float(kappa_eff_sr),
                'kappa_expected': float(s * geo['kappa']),
                'd_eff_expected': float(r * geo['d_eff']),
                'logit': float(logit_sr), 'q': float(q_sr),
            })

    log(f"Grid computed: {len(grid_results)} conditions")

    # Find matched pairs
    matched_pairs = []
    n = len(grid_results)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = grid_results[i], grid_results[j]
            keff_a = a['kappa_eff_target']
            keff_b = b['kappa_eff_target']
            keff_mean = (keff_a + keff_b) / 2 + 1e-10
            if abs(keff_a - keff_b) / keff_mean > 0.05:  # not same kappa_eff
                continue

            s_a, s_b = a['s'], b['s']
            r_a, r_b = a['r'], b['r']
            s_sep = abs(s_a - s_b) / ((s_a + s_b) / 2 + 1e-10)
            r_sep = abs(r_a - r_b) / ((r_a + r_b) / 2 + 1e-10)

            if s_sep < 0.15 or r_sep < 0.15:  # knobs not sufficiently different
                continue

            delta_logit = abs(a['logit'] - b['logit'])
            matched_pairs.append({
                's1': s_a, 'r1': r_a, 'kappa1': a['kappa_expected'], 'd_eff1': a['d_eff_expected'],
                's2': s_b, 'r2': r_b, 'kappa2': b['kappa_expected'], 'd_eff2': b['d_eff_expected'],
                'kappa_eff1': keff_a, 'kappa_eff2': keff_b,
                'logit1': a['logit'], 'logit2': b['logit'],
                'delta_logit': float(delta_logit),
            })

    log(f"Found {len(matched_pairs)} matched pairs")
    if matched_pairs:
        deltas = [p['delta_logit'] for p in matched_pairs]
        mean_delta = float(np.mean(deltas))
        frac_pass = float(np.mean([d < PAIR_MATCH_THRESH for d in deltas]))
        log(f"  Mean |Delta_logit| = {mean_delta:.4f} (H_keff: <{PAIR_MATCH_THRESH})")
        log(f"  Fraction with |Delta_logit| < {PAIR_MATCH_THRESH}: {frac_pass:.3f}")
        log(f"  H_keff matched pairs PASS: {mean_delta < PAIR_MATCH_THRESH}")
        for p in matched_pairs[:5]:
            log(f"    (s={p['s1']:.2f},r={p['r1']:.2f}) vs (s={p['s2']:.2f},r={p['r2']:.2f}): "
                f"kappa_eff=({p['kappa_eff1']:.3f},{p['kappa_eff2']:.3f}), "
                f"|Delta_logit|={p['delta_logit']:.4f}")
    else:
        mean_delta = 0.0
        frac_pass = 0.0
        log("  No matched pairs found")
    log("")

    # ==================== FINAL VERDICT ====================
    log("=" * 70)
    log("FINAL VERDICT")
    log("=" * 70)

    h_kappa_pass = null_deff_pass and r_arm2_kappa > KAPPA_PEARSON_THRESH
    h_keff_pass = keff_deff_pass and mean_delta < PAIR_MATCH_THRESH

    log(f"H_kappa (kappa alone is causal):")
    log(f"  d_eff null (max_delta<{NULL_DEFF_MAX_DELTA}): {null_deff_pass} ({max_abs_delta:.4f})")
    log(f"  kappa linear (r>{KAPPA_PEARSON_THRESH}): {r_arm2_kappa > KAPPA_PEARSON_THRESH} ({r_arm2_kappa:.4f})")
    log(f"  OVERALL H_kappa PASS: {h_kappa_pass}")
    log("")
    log(f"H_keff (kappa_eff = kappa*sqrt(d_eff) is sufficient):")
    log(f"  d_eff active (r>{DEFF_PEARSON_THRESH}, calib in band): {keff_deff_pass}")
    log(f"    Pearson(delta_obs, sqrt(r)-1) = {r_arm1_keff:.4f}")
    log(f"    OLS slope (should be 1.0) = {slope_K14:.4f}")
    log(f"  matched pairs null (mean<{PAIR_MATCH_THRESH}): {mean_delta < PAIR_MATCH_THRESH} ({mean_delta:.4f})")
    log(f"  OVERALL H_keff PASS: {h_keff_pass}")
    log("")

    if h_kappa_pass and not h_keff_pass:
        verdict = "H_kappa WINS: kappa alone is causal, d_eff is a confound"
    elif h_keff_pass and not h_kappa_pass:
        verdict = "H_keff WINS: kappa_eff = kappa*sqrt(d_eff) is sufficient"
    elif h_kappa_pass and h_keff_pass:
        verdict = "BOTH PASS: degenerate case (small surgery range?)"
    else:
        verdict = "NEITHER PASS: complex regime, needs investigation"
    log(f"VERDICT: {verdict}")

    # ==================== SAVE RESULTS ====================
    result = {
        'experiment': 'frozen_two_knob_causal_test',
        'data': EMBED_PATH,
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'baseline': {
            'K': int(K),
            'acc': float(acc_base),
            'q': float(q_base),
            'logit': float(logit_base),
            'kappa': float(geo['kappa']),
            'd_eff': float(geo['d_eff']),
            'kappa_eff': float(geo['kappa_eff']),
            'sigma_cdir': float(geo['sigma_cdir']),
            'trW': float(geo['trW']),
        },
        'derived_constants': {
            'A_keff_K14': float(A_keff_K14),
            'C_kappa': float(C_kappa),
            'C_keff': float(C_keff),
        },
        'arm1_deff_surgery': deff_results,
        'arm1_summary': {
            'max_abs_delta': float(max_abs_delta),
            'pearson_sqrt_r': float(r_arm1_keff),
            'pearson_pred_K14': float(r_arm1_pred_K14),
            'ols_slope_K14': float(slope_K14),
            'h_kappa_null_pass': bool(null_deff_pass),
            'h_keff_causal_pass': bool(keff_deff_pass),
        },
        'arm2_kappa_surgery': kappa_results,
        'arm2_summary': {
            'pearson_kappa': float(r_arm2_kappa),
            'pearson_pred_kappa': float(r_arm2_kappa_pred),
            'pearson_pred_keff': float(r_arm2_keff_pred),
            'ols_slope_kappa': float(slope_kappa_arm),
        },
        'arm3_matched_pairs': matched_pairs[:20],  # save first 20
        'arm3_summary': {
            'n_pairs': len(matched_pairs),
            'mean_delta_logit': float(mean_delta),
            'frac_pass_thresh': float(frac_pass),
        },
        'verdict': {
            'h_kappa_pass': bool(h_kappa_pass),
            'h_keff_pass': bool(h_keff_pass),
            'conclusion': verdict,
        }
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\nResults saved to {RESULT_PATH}")
    log(f"Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
