#!/usr/bin/env python -u
"""
TWO-KNOB IDENTIFIABILITY: kappa_eff = kappa * sqrt(d_eff) is the sufficient statistic
======================================================================================

PRE-REGISTERED (Codex design, Feb 23 2026). COMMIT BEFORE RUNNING.

HYPOTHESIS: kappa_eff = kappa_nearest * sqrt(d_eff_formula) is the UNIQUE sufficient
statistic for q. Two different (kappa, d_eff) combinations with the SAME kappa_eff
should give the SAME logit(q).

TWO INDEPENDENT KNOBS:
  Knob 1 (kappa):  Scale class centroids around grand mean by factor s.
    Effect: kappa -> s * kappa_base, d_eff UNCHANGED, kappa_eff -> s * kappa_eff_base.
  Knob 2 (d_eff):  Redistribue within-class variance (existing surgery, factor r).
    Effect: d_eff -> r * d_eff_base, kappa UNCHANGED, kappa_eff -> kappa * sqrt(r*d_eff).
  Combined: kappa_eff = (s * kappa_base) * sqrt(r * d_eff_base) = s*sqrt(r) * kappa_eff_base

  A matched pair (s1,r1) and (s2,r2) satisfies:
    s1 * sqrt(r1) = s2 * sqrt(r2) -- same kappa_eff
    |s1 - s2| / mean(s1,s2) >= 0.15 -- different kappa decomposition
    |r1 - r2| / mean(r1,r2) >= 0.15 -- different d_eff decomposition

IDENTIFIABILITY TEST: If kappa_eff is sufficient, matched pairs should have
  |logit(q1) - logit(q2)| <= 0.10 logit units (small residual).

DATA: Reuses linear-regime checkpoint embeddings from cti_linear_regime_surgery.py
  (must run AFTER linear regime surgery completes and saves embeddings).
  Uses train_eval_ds embeddings (eval-mode transforms, no augmentation confound).

PRE-REGISTERED:
  A_renorm(K=20) = 1.0535 (Theorem 15)
  Linear regime: kappa_eff in [0.5, 2.0]
  PASS criteria (ALL required):
    P1: Pearson r(logit_obs, logit_pred) >= 0.99
    P2: Mean calibration error <= 0.10 (10%)
    P3: Mean |Delta_logit| over matched pairs <= 0.08
    P4: >=80% of matched pairs: |Delta_logit| <= 0.10
    P5: Residual correlation with kappa: |corr| < 0.10 (sufficiency)
    P6: Residual correlation with d_eff: |corr| < 0.10 (sufficiency)
    P7: >= 30 total matched pairs (>= 8 per seed)

Nobel/Turing prediction for full PASS: Nobel 8.0/10, Turing 8.6/10 (from 6.6/8.0).
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr

# ==================== CONFIGURATION ====================
K = 20
N_SEEDS = 3
DEVICE_STR = "numpy"  # Pure numpy surgery, no GPU needed

A_RENORM_K20 = 1.0535   # Pre-registered constant (Theorem 15)
KAPPA_EFF_MIN = 0.5
KAPPA_EFF_MAX = 2.0

# Grid for two-knob sweep
KAPPA_SCALES = [0.75, 0.85, 0.95, 1.00, 1.10, 1.20, 1.35]   # s values (kappa knob)
DEFF_SCALES  = [0.70, 0.85, 1.00, 1.20, 1.50, 1.80]          # r values (d_eff knob)

# Pre-registered thresholds
MATCH_TOLERANCE = 0.05   # |kappa_eff1 - kappa_eff2| / mean <= 5%
MIN_KNOB_SEP    = 0.15   # each knob must differ by >=15%
MIN_PAIRS_TOTAL = 30
MIN_PAIRS_SEED  = 8
PAIR_PASS_DELTA = 0.10   # |Delta_logit| <= 0.10 counts as "pass" per pair
MEAN_DELTA_THRESH = 0.08 # mean |Delta_logit| <= 0.08
CORR_THRESH     = 0.10   # |residual correlation| < 0.10 for sufficiency
PEARSON_THRESH  = 0.99
CALIB_THRESH    = 0.10

EMBED_DIR   = "results/linear_regime_surgery_embeddings"
RESULT_PATH = "results/cti_two_knob_identifiability_keff.json"
LOG_PATH    = "results/cti_two_knob_identifiability_keff_log.txt"

# Clear old log
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


# ==================== GEOMETRY ====================
def compute_geometry(X_tr, y_tr):
    """CTI geometry: kappa_nearest, d_eff_formula, kappa_eff."""
    classes = np.unique(y_tr)
    K_actual = len(classes)
    N = len(X_tr)
    d = X_tr.shape[1]

    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])
    grand_mean = X_tr.mean(0)

    trW = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c] - centroids[int(c)]
        trW += float(np.sum(Xc ** 2)) / N
    sigma_W_global = float(np.sqrt(trW / d))

    min_dist = float('inf')
    min_i, min_j = 0, 1
    for i in range(K_actual):
        for j in range(i + 1, K_actual):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j

    delta_min = float(min_dist)
    kappa_nearest = float(delta_min / (sigma_W_global * np.sqrt(d) + 1e-10))

    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)

    sigma_centroid_sq = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[int(c)]
        proj = Xc_c @ Delta_hat
        sigma_centroid_sq += (n_c / N) * float(np.mean(proj ** 2))

    d_eff_formula = float(trW / (sigma_centroid_sq + 1e-10))
    kappa_eff = kappa_nearest * float(np.sqrt(d_eff_formula))

    return {
        'centroids': centroids,
        'grand_mean': grand_mean,
        'Delta_hat': Delta_hat,
        'trW': trW,
        'sigma_W_global': sigma_W_global,
        'sigma_centroid_sq': sigma_centroid_sq,
        'd_eff_formula': d_eff_formula,
        'kappa_nearest': kappa_nearest,
        'kappa_eff': kappa_eff,
        'delta_min': delta_min,
        'nearest_pair': (int(min_i), int(min_j)),
        'K_actual': K_actual,
    }


def compute_q(X_tr, y_tr, X_te, y_te):
    """1-NN normalized accuracy (n_jobs=1 for Windows CUDA compatibility)."""
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = float(knn.score(X_te, y_te))
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    return float(q), float(acc)


# ==================== SURGERY OPERATORS ====================
def apply_kappa_surgery(X, y, s):
    """Scale class centroids around grand mean by factor s.
    Effect: kappa -> s * kappa_base (delta_min scales by s).
    d_eff unchanged (within-class variance unchanged, Delta_hat direction unchanged).
    """
    classes = np.unique(y)
    grand_mean = X.mean(0)
    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        mu_c = X[mask].mean(0)
        # Shift all samples: new centroid = grand_mean + s*(mu_c - grand_mean)
        # Individual samples: x_new = x + (s-1)*(mu_c - grand_mean)
        X_new[mask] = X[mask] + (s - 1.0) * (mu_c - grand_mean)
    return X_new


def apply_deff_surgery(X, y, geometry, r):
    """Redistribute within-class variance: d_eff_new = r * d_eff_base, kappa unchanged.
    Uses existing surgery from cti_linear_regime_surgery.py.
    """
    centroids = geometry['centroids']
    Delta_hat = geometry['Delta_hat']
    trW = geometry['trW']
    sigma_centroid_sq = geometry['sigma_centroid_sq']
    classes = np.unique(y)

    min_r = float(sigma_centroid_sq / (trW + 1e-10)) * 1.001
    if r < min_r:
        r = min_r

    scale_along = 1.0 / float(np.sqrt(r))
    denom = trW - sigma_centroid_sq
    num = trW - sigma_centroid_sq / r
    scale_perp = float(np.sqrt(max(0.0, num / denom))) if denom > 1e-12 else 1.0

    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        z = X[mask] - centroids[int(c)]
        proj = z @ Delta_hat
        z_along = proj[:, None] * Delta_hat[None, :]
        z_perp = z - z_along
        X_new[mask] = centroids[int(c)] + scale_along * z_along + scale_perp * z_perp
    return X_new


# ==================== MATCHED PAIR CONSTRUCTION ====================
def find_matched_pairs(conditions):
    """Greedy one-to-one matched pairs: same kappa_eff, different (kappa, d_eff).

    conditions: list of dicts with keys: s, r, kappa_eff, kappa, d_eff, logit_q, q
    Returns list of (i, j) pairs by indices.
    """
    n = len(conditions)
    used = [False] * n

    # Build all candidate pairs sorted by knob separation
    candidates = []
    for i in range(n):
        ci = conditions[i]
        if ci['valid'] is False:
            continue
        for j in range(i + 1, n):
            cj = conditions[j]
            if cj['valid'] is False:
                continue
            # kappa_eff match within tolerance
            keff_mean = (ci['kappa_eff'] + cj['kappa_eff']) / 2.0
            keff_gap = abs(ci['kappa_eff'] - cj['kappa_eff']) / (keff_mean + 1e-10)
            if keff_gap > MATCH_TOLERANCE:
                continue
            # kappa must differ by >= MIN_KNOB_SEP
            kappa_mean = (ci['kappa'] + cj['kappa']) / 2.0
            kappa_gap = abs(ci['kappa'] - cj['kappa']) / (kappa_mean + 1e-10)
            if kappa_gap < MIN_KNOB_SEP:
                continue
            # d_eff must differ by >= MIN_KNOB_SEP
            deff_mean = (ci['d_eff'] + cj['d_eff']) / 2.0
            deff_gap = abs(ci['d_eff'] - cj['d_eff']) / (deff_mean + 1e-10)
            if deff_gap < MIN_KNOB_SEP:
                continue
            # Score: larger separation = better match
            score = kappa_gap + deff_gap
            candidates.append((score, i, j, keff_gap))

    candidates.sort(reverse=True)
    pairs = []
    for _, i, j, keff_gap in candidates:
        if not used[i] and not used[j]:
            used[i] = True
            used[j] = True
            pairs.append((i, j, keff_gap))

    return pairs


# ==================== MAIN ====================
def main():
    os.makedirs('results', exist_ok=True)

    log("=" * 70)
    log("TWO-KNOB IDENTIFIABILITY: kappa_eff sufficient statistic test")
    log("=" * 70)
    log(f"A_renorm(K=20) = {A_RENORM_K20} (pre-registered)")
    log(f"Linear regime: kappa_eff in [{KAPPA_EFF_MIN}, {KAPPA_EFF_MAX}]")
    log(f"Kappa scales: {KAPPA_SCALES}")
    log(f"d_eff scales: {DEFF_SCALES}")
    log(f"Grid size: {len(KAPPA_SCALES)} x {len(DEFF_SCALES)} = "
        f"{len(KAPPA_SCALES) * len(DEFF_SCALES)} conditions per seed")
    log(f"Match tolerance: kappa_eff within {MATCH_TOLERANCE*100:.0f}%, "
        f"knob separation >= {MIN_KNOB_SEP*100:.0f}%")
    log(f"\nPRE-REGISTERED PASS CRITERIA:")
    log(f"  P1: Pearson r >= {PEARSON_THRESH}")
    log(f"  P2: Mean calibration error <= {CALIB_THRESH}")
    log(f"  P3: Mean |Delta_logit| (matched pairs) <= {MEAN_DELTA_THRESH}")
    log(f"  P4: >=80% matched pairs: |Delta_logit| <= {PAIR_PASS_DELTA}")
    log(f"  P5: |corr(residual, kappa)| < {CORR_THRESH}")
    log(f"  P6: |corr(residual, d_eff)| < {CORR_THRESH}")
    log(f"  P7: >= {MIN_PAIRS_TOTAL} total pairs (>= {MIN_PAIRS_SEED} per seed)\n")

    # Check if embedding files exist
    missing_seeds = []
    for seed in range(N_SEEDS):
        if not any(
            os.path.exists(f"{EMBED_DIR}/X_tr_seed{seed}_ep{ep}.npy")
            for ep in range(1, 61)
        ):
            missing_seeds.append(seed)

    if missing_seeds:
        log(f"ERROR: Missing linear-regime embeddings for seeds {missing_seeds}.")
        log(f"Run src/cti_linear_regime_surgery.py first to generate embeddings.")
        return

    all_conditions = []
    all_pairs = []

    for seed in range(N_SEEDS):
        log(f"\n{'='*60}")
        log(f"SEED {seed}")
        log(f"{'='*60}")

        # Find saved checkpoint for this seed
        best_ep = None
        best_keff_dist = float('inf')
        for ep in range(1, 61):
            fpath = f"{EMBED_DIR}/X_tr_seed{seed}_ep{ep}.npy"
            if os.path.exists(fpath):
                X_tmp = np.load(fpath)
                y_tmp = np.load(f"{EMBED_DIR}/y_tr_seed{seed}_ep{ep}.npy")
                geo_tmp = compute_geometry(X_tmp, y_tmp)
                keff = geo_tmp['kappa_eff']
                dist = abs(keff - 1.0)
                if KAPPA_EFF_MIN <= keff <= KAPPA_EFF_MAX and dist < best_keff_dist:
                    best_keff_dist = dist
                    best_ep = ep

        if best_ep is None:
            log(f"  No linear-regime checkpoint found for seed {seed}. Skipping.")
            continue

        log(f"  Using checkpoint epoch={best_ep}")
        X_tr_full = np.load(f"{EMBED_DIR}/X_tr_seed{seed}_ep{best_ep}.npy")
        y_tr_full = np.load(f"{EMBED_DIR}/y_tr_seed{seed}_ep{best_ep}.npy")
        # Load test embeddings — check for ep-specific or non-ep test files
        X_te_path = f"{EMBED_DIR}/X_te_seed{seed}_ep{best_ep}.npy"
        if not os.path.exists(X_te_path):
            # Try without epoch suffix
            X_te_path_base = f"{EMBED_DIR}/X_te_seed{seed}.npy"
            if os.path.exists(X_te_path_base):
                X_te_path = X_te_path_base
            else:
                X_te_path = None

        if X_te_path is not None:
            X_te_base = np.load(X_te_path)
            y_te_path = X_te_path.replace("X_te", "y_te")
            y_te = np.load(y_te_path)
            log(f"  Loaded test embeddings from {X_te_path}")
        else:
            # Fallback: stratified 80/20 split of X_tr_full into train/val
            log(f"  WARNING: No test embeddings found for seed {seed}.")
            log(f"  Fallback: stratified 80/20 split of X_tr as train/val.")
            rng = np.random.default_rng(seed + 42)
            train_idx, val_idx = [], []
            for c in np.unique(y_tr_full):
                c_idx = np.where(y_tr_full == c)[0]
                rng.shuffle(c_idx)
                n_val = max(1, len(c_idx) // 5)  # 20% val
                val_idx.extend(c_idx[:n_val])
                train_idx.extend(c_idx[n_val:])
            train_idx = np.array(train_idx)
            val_idx = np.array(val_idx)
            X_tr_full, y_tr_full, X_te_base, y_te = (
                X_tr_full[train_idx], y_tr_full[train_idx],
                X_tr_full[val_idx], y_tr_full[val_idx],
            )
            log(f"  Split: {len(train_idx)} train, {len(val_idx)} val")

        X_tr_base = X_tr_full
        y_tr = y_tr_full

        geo_base = compute_geometry(X_tr_base, y_tr)
        q_base, acc_base = compute_q(X_tr_base, y_tr, X_te_base, y_te)
        kappa_eff_base = geo_base['kappa_eff']
        logit_q_base = float(np.log(q_base / (1 - q_base + 1e-10) + 1e-10))
        C_seed = logit_q_base - A_RENORM_K20 * kappa_eff_base

        log(f"  Base: kappa={geo_base['kappa_nearest']:.4f}, d_eff={geo_base['d_eff_formula']:.3f}, "
            f"kappa_eff={kappa_eff_base:.4f}, q={q_base:.4f}")
        log(f"  C_seed = {C_seed:.4f}")

        # Baseline condition (s=1.0, r=1.0)
        seed_conditions = [{
            'seed': seed, 's': 1.0, 'r': 1.0,
            'kappa': float(geo_base['kappa_nearest']),
            'd_eff': float(geo_base['d_eff_formula']),
            'kappa_eff': float(kappa_eff_base),
            'q': float(q_base),
            'logit_q': float(logit_q_base),
            'logit_pred': float(C_seed + A_RENORM_K20 * kappa_eff_base),
            'valid': True,
        }]

        # Build grid of (s, r) conditions
        for s in KAPPA_SCALES:
            # Apply kappa surgery first
            X_scaled = apply_kappa_surgery(X_tr_base, y_tr, s)
            X_te_scaled = apply_kappa_surgery(X_te_base, y_te, s)
            geo_scaled = compute_geometry(X_scaled, y_tr)

            # Verify kappa scaling
            kappa_actual_s = geo_scaled['kappa_nearest']
            d_eff_after_kappa = geo_scaled['d_eff_formula']
            kappa_d_change_pct = abs(d_eff_after_kappa - geo_base['d_eff_formula']) / (
                geo_base['d_eff_formula'] + 1e-10) * 100

            for r in DEFF_SCALES:
                # Apply d_eff surgery on top of kappa-scaled embeddings
                X_combined = apply_deff_surgery(X_scaled, y_tr, geo_scaled, r)
                X_te_combined = apply_deff_surgery(X_te_scaled, y_te, geo_scaled, r)
                geo_combined = compute_geometry(X_combined, y_tr)

                # Verify surgery: kappa should equal s*kappa_base, d_eff should equal r*d_eff_after_kappa
                kappa_combined = geo_combined['kappa_nearest']
                d_eff_combined = geo_combined['d_eff_formula']
                kappa_eff_combined = geo_combined['kappa_eff']

                # Quality filter: keep only linear regime conditions
                valid = (KAPPA_EFF_MIN <= kappa_eff_combined <= KAPPA_EFF_MAX)

                q_cond, acc_cond = compute_q(X_combined, y_tr, X_te_combined, y_te)
                logit_q_cond = float(np.log(q_cond / (1 - q_cond + 1e-10) + 1e-10))
                logit_pred = float(C_seed + A_RENORM_K20 * kappa_eff_combined)
                calib_error = float(abs(logit_q_cond - logit_pred) / (
                    abs(logit_pred - logit_q_base) + 1e-3))

                cond = {
                    'seed': seed, 's': float(s), 'r': float(r),
                    'kappa': float(kappa_combined),
                    'd_eff': float(d_eff_combined),
                    'kappa_eff': float(kappa_eff_combined),
                    'q': float(q_cond),
                    'logit_q': float(logit_q_cond),
                    'logit_pred': float(logit_pred),
                    'calib_error': float(calib_error),
                    'kappa_d_change_pct': float(kappa_d_change_pct),
                    'valid': valid,
                }
                seed_conditions.append(cond)

                status = "OK" if valid else "OUT-OF-REGIME"
                log(f"  s={s:.2f} r={r:.2f}: kappa={kappa_combined:.4f}, "
                    f"d_eff={d_eff_combined:.3f}, kappa_eff={kappa_eff_combined:.4f}, "
                    f"q={q_cond:.4f} [{status}]")

        all_conditions.extend(seed_conditions)

        # Find matched pairs for this seed
        valid_conditions = [c for c in seed_conditions if c['valid']]
        log(f"\n  Valid (linear-regime) conditions: {len(valid_conditions)}")
        seed_pairs = find_matched_pairs(valid_conditions)
        log(f"  Matched pairs found: {len(seed_pairs)}")
        for pi, pj, keff_gap in seed_pairs:
            ci, cj = valid_conditions[pi], valid_conditions[pj]
            delta_logit = abs(ci['logit_q'] - cj['logit_q'])
            log(f"    Pair: (s={ci['s']:.2f},r={ci['r']:.2f}) vs (s={cj['s']:.2f},r={cj['r']:.2f}): "
                f"kappa_eff_gap={keff_gap*100:.1f}%, "
                f"kappa_diff={abs(ci['kappa']-cj['kappa'])/max(ci['kappa'],cj['kappa'])*100:.1f}%, "
                f"|Delta_logit|={delta_logit:.4f}")
        all_pairs.extend([
            {
                'seed': seed,
                'i_s': valid_conditions[pi]['s'], 'i_r': valid_conditions[pi]['r'],
                'j_s': valid_conditions[pj]['s'], 'j_r': valid_conditions[pj]['r'],
                'kappa_eff_i': valid_conditions[pi]['kappa_eff'],
                'kappa_eff_j': valid_conditions[pj]['kappa_eff'],
                'kappa_i': valid_conditions[pi]['kappa'],
                'kappa_j': valid_conditions[pj]['kappa'],
                'd_eff_i': valid_conditions[pi]['d_eff'],
                'd_eff_j': valid_conditions[pj]['d_eff'],
                'logit_i': valid_conditions[pi]['logit_q'],
                'logit_j': valid_conditions[pj]['logit_q'],
                'delta_logit': abs(valid_conditions[pi]['logit_q'] - valid_conditions[pj]['logit_q']),
                'keff_gap': keff_gap,
            }
            for pi, pj, keff_gap in seed_pairs
        ])

    # ==================== GLOBAL ANALYSIS ====================
    log(f"\n{'='*70}")
    log("GLOBAL ANALYSIS")
    log(f"{'='*70}")

    valid_all = [c for c in all_conditions if c['valid']]
    log(f"Total conditions (linear regime): {len(valid_all)}")
    log(f"Total matched pairs: {len(all_pairs)}")

    if len(valid_all) < 3:
        log("ERROR: Too few valid conditions for analysis.")
        return

    # P1: Pearson r (global law)
    logit_obs = np.array([c['logit_q'] for c in valid_all])
    logit_pred = np.array([c['logit_pred'] for c in valid_all])
    r_pearson, _ = pearsonr(logit_obs, logit_pred)

    # P2: Mean calibration error
    calib_errors = [c['calib_error'] for c in valid_all
                    if abs(c['logit_pred'] - c.get('logit_q', 0)) > 0.05]
    mean_calib = float(np.mean(calib_errors)) if calib_errors else float('nan')

    # Residual correlations (P5, P6)
    residuals = logit_obs - logit_pred
    kappas_arr = np.array([c['kappa'] for c in valid_all])
    deffs_arr = np.array([c['d_eff'] for c in valid_all])
    if np.std(kappas_arr) > 0:
        corr_kappa, _ = pearsonr(residuals, kappas_arr)
    else:
        corr_kappa = 0.0
    if np.std(deffs_arr) > 0:
        corr_deff, _ = pearsonr(residuals, deffs_arr)
    else:
        corr_deff = 0.0

    # P3, P4: Matched pair analysis
    if all_pairs:
        delta_logits = np.array([p['delta_logit'] for p in all_pairs])
        mean_delta = float(np.mean(delta_logits))
        pct_pass_pairs = float(np.mean(delta_logits <= PAIR_PASS_DELTA) * 100)
    else:
        mean_delta = float('nan')
        pct_pass_pairs = 0.0

    # P7: Pair count
    n_pairs_per_seed = {}
    for p in all_pairs:
        n_pairs_per_seed[p['seed']] = n_pairs_per_seed.get(p['seed'], 0) + 1
    min_pairs_per_seed = min(n_pairs_per_seed.values()) if n_pairs_per_seed else 0
    n_pairs_total = len(all_pairs)

    # Results
    log(f"\nPRE-REGISTERED RESULTS:")
    log(f"  P1: Pearson r = {r_pearson:.4f} (threshold >= {PEARSON_THRESH}): "
        f"{'PASS' if r_pearson >= PEARSON_THRESH else 'FAIL'}")
    log(f"  P2: Mean calibration error = {mean_calib:.4f} (threshold <= {CALIB_THRESH}): "
        f"{'PASS' if mean_calib <= CALIB_THRESH else 'FAIL'}")
    log(f"  P3: Mean |Delta_logit| = {mean_delta:.4f} (threshold <= {MEAN_DELTA_THRESH}): "
        f"{'PASS' if mean_delta <= MEAN_DELTA_THRESH else 'FAIL'}")
    log(f"  P4: {pct_pass_pairs:.1f}% pairs within {PAIR_PASS_DELTA} (threshold >= 80%): "
        f"{'PASS' if pct_pass_pairs >= 80 else 'FAIL'}")
    log(f"  P5: |corr(residual, kappa)| = {abs(corr_kappa):.4f} (threshold < {CORR_THRESH}): "
        f"{'PASS' if abs(corr_kappa) < CORR_THRESH else 'FAIL'}")
    log(f"  P6: |corr(residual, d_eff)| = {abs(corr_deff):.4f} (threshold < {CORR_THRESH}): "
        f"{'PASS' if abs(corr_deff) < CORR_THRESH else 'FAIL'}")
    log(f"  P7: n_pairs = {n_pairs_total} (total), {min_pairs_per_seed} (min per seed): "
        f"{'PASS' if n_pairs_total >= MIN_PAIRS_TOTAL and min_pairs_per_seed >= MIN_PAIRS_SEED else 'FAIL'}")

    criteria = {
        'P1_pearson_pass': bool(r_pearson >= PEARSON_THRESH),
        'P2_calib_pass': bool(mean_calib <= CALIB_THRESH),
        'P3_mean_delta_pass': bool(mean_delta <= MEAN_DELTA_THRESH),
        'P4_pct_pairs_pass': bool(pct_pass_pairs >= 80),
        'P5_corr_kappa_pass': bool(abs(corr_kappa) < CORR_THRESH),
        'P6_corr_deff_pass': bool(abs(corr_deff) < CORR_THRESH),
        'P7_pairs_count_pass': bool(
            n_pairs_total >= MIN_PAIRS_TOTAL and min_pairs_per_seed >= MIN_PAIRS_SEED),
    }
    n_pass = sum(criteria.values())
    overall_pass = all(criteria.values())
    log(f"\nOVERALL: {n_pass}/7 criteria PASS: {'FULL PASS' if overall_pass else 'PARTIAL PASS' if n_pass >= 4 else 'FAIL'}")

    if overall_pass:
        log("\nNOBEL/TURING PREDICTION: Nobel ~8.0/10, Turing ~8.6/10")
    elif n_pass >= 5:
        log("\nPARTIAL PASS (5-6/7): Nobel ~7.2-7.5/10 expected")

    result = {
        'experiment': 'two_knob_identifiability_keff',
        'preregistered': {
            'A_renorm': A_RENORM_K20,
            'thresholds': {
                'pearson': PEARSON_THRESH,
                'calib': CALIB_THRESH,
                'mean_delta': MEAN_DELTA_THRESH,
                'pair_pass_delta': PAIR_PASS_DELTA,
                'corr_thresh': CORR_THRESH,
                'min_pairs_total': MIN_PAIRS_TOTAL,
                'min_pairs_seed': MIN_PAIRS_SEED,
            },
        },
        'results': {
            'r_pearson': float(r_pearson),
            'mean_calib': float(mean_calib),
            'mean_delta_logit': float(mean_delta),
            'pct_pairs_pass': float(pct_pass_pairs),
            'corr_residual_kappa': float(corr_kappa),
            'corr_residual_deff': float(corr_deff),
            'n_pairs_total': int(n_pairs_total),
            'min_pairs_per_seed': int(min_pairs_per_seed),
            'n_valid_conditions': int(len(valid_all)),
        },
        'criteria': criteria,
        'overall_pass': overall_pass,
        'n_criteria_pass': int(n_pass),
        'conditions': all_conditions,
        'matched_pairs': all_pairs,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else x)
    log(f"\nResults saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
