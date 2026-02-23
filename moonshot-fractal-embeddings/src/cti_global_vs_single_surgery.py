"""
Global vs Single Surgery Ratio Test (Pre-registered: src/PREREGISTRATION_global_surgery_ratio.md)

Tests the 1/d_eff attenuation hypothesis:
  delta_logit_global / delta_logit_single ~= d_eff_base

Uses saved linear-regime embeddings from cti_linear_regime_surgery.py
(seeds 0,1 at epoch 4; seed 2 at epoch 5 -- kappa_eff ~= 1.0)

Pre-registered success criteria:
  H1-PASS: median ratio in [d_eff_base/3, d_eff_base*3]
  H2-PASS: delta_logit_global positive for r>1, negative for r<1 (3/4 pairs)
  H3-PASS: kappa invariance < 0.5% for both surgeries
"""
import os
import json
import numpy as np
from scipy.special import logit as scipy_logit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
EMBED_DIR = os.path.join(RESULTS_DIR, "linear_regime_surgery_embeddings")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_global_vs_single_surgery.json")
LOG_PATH = os.path.join(RESULTS_DIR, "cti_global_vs_single_surgery_log.txt")

K = 20
A_RENORM_K20 = 1.0535
SURGERY_LEVELS = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]  # r=1.0 is baseline
PREREGISTERED_COMMIT = "1abdef6"

log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)


def compute_geometry(X_tr, y_tr):
    """Compute all CTI geometry metrics from training embeddings."""
    classes = np.unique(y_tr)
    K_actual = len(classes)
    N = len(X_tr)
    d = X_tr.shape[1]

    # Class centroids
    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])  # (K, d)

    # tr(Sigma_W): total within-class variance
    trW = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        Xc_c = Xc - centroids[c]
        trW += float(np.sum(Xc_c ** 2)) / N

    sigma_W_global = float(np.sqrt(trW / d))

    # Nearest centroid pair
    min_dist = float('inf')
    min_i, min_j = 0, 1
    for i in range(K_actual):
        for j in range(i + 1, K_actual):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j

    delta_min = float(min_dist)
    kappa_nearest = float(delta_min / (sigma_W_global * np.sqrt(d) + 1e-10))

    # Delta_hat: unit vector of nearest centroid pair direction
    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)

    # sigma_centroid_dir^2 = Delta_hat^T Sigma_W Delta_hat
    sigma_centroid_sq = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        proj = Xc_c @ Delta_hat
        sigma_centroid_sq += (n_c / N) * float(np.mean(proj ** 2))

    sigma_centroid_dir = float(np.sqrt(sigma_centroid_sq + 1e-10))
    d_eff_formula = float(trW / (sigma_centroid_sq + 1e-10))

    # Signal subspace: top K-1 singular vectors of centered centroids
    grand_mean = centroids.mean(0)
    centroids_centered = centroids - grand_mean
    _, _, Vt = np.linalg.svd(centroids_centered, full_matrices=False)
    n_sig = min(K_actual - 1, d, Vt.shape[0])
    P_B = Vt[:n_sig, :]  # (K-1, d) orthonormal signal subspace basis

    # tr(Sigma_W) in signal subspace
    trW_sig = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        Xc_proj = Xc_c @ P_B.T  # (n_c, K-1)
        trW_sig += (n_c / N) * float(np.sum(Xc_proj ** 2)) / n_c

    trW_null = trW - trW_sig

    return {
        'centroids': centroids,
        'Delta_hat': Delta_hat,
        'trW': trW,
        'trW_sig': trW_sig,
        'trW_null': trW_null,
        'sigma_W_global': sigma_W_global,
        'sigma_centroid_sq': sigma_centroid_sq,
        'sigma_centroid_dir': sigma_centroid_dir,
        'd_eff_formula': d_eff_formula,
        'kappa_nearest': kappa_nearest,
        'delta_min': delta_min,
        'nearest_pair': (int(min_i), int(min_j)),
        'K_actual': K_actual,
        'd': d,
        'P_B': P_B,
        'n_sig': n_sig,
    }


def apply_single_surgery(X, y, geometry, r):
    """Single-direction surgery (existing method): scale only Delta_hat direction."""
    centroids = geometry['centroids']
    Delta_hat = geometry['Delta_hat']
    trW = geometry['trW']
    sigma_centroid_sq = geometry['sigma_centroid_sq']
    classes = np.unique(y)

    # Clamp r to valid range
    min_r = float(sigma_centroid_sq / (trW + 1e-10)) * 1.001
    r_eff = max(r, min_r)

    scale_along = 1.0 / float(np.sqrt(r_eff))
    denom = trW - sigma_centroid_sq
    num = trW - sigma_centroid_sq / r_eff
    scale_perp = float(np.sqrt(max(0.0, num / (denom + 1e-12))))

    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        z = X[mask] - centroids[c]
        proj = z @ Delta_hat
        z_along = proj[:, None] * Delta_hat[None, :]
        z_perp = z - z_along
        X_new[mask] = centroids[c] + scale_along * z_along + scale_perp * z_perp

    return X_new, float(scale_along), float(scale_perp)


def apply_global_surgery(X, y, geometry, r):
    """Global surgery: scale ALL K-1 signal subspace directions by 1/sqrt(r).

    Valid range: r >= trW_sig / trW  (otherwise scale_null^2 would be negative)
    For r >= 1, always valid. For r < 1, only valid if trW_sig/trW <= r.
    Returns (X_new, scale_sig, scale_null, valid) where valid=False means r outside range.
    """
    centroids = geometry['centroids']
    P_B = geometry['P_B']
    trW = geometry['trW']
    trW_sig = geometry['trW_sig']
    trW_null = geometry['trW_null']
    classes = np.unique(y)

    # Minimum valid r: trW_sig/trW (below this, scale_null^2 < 0)
    r_min_valid = trW_sig / (trW + 1e-10)

    # scale_sig = 1/sqrt(r)
    # scale_null: chosen to preserve tr(Sigma_W)
    # tr(W_new) = tr(W_sig)/r + scale_null^2 * tr(W_null) = tr(W)
    # scale_null^2 = (tr(W) - tr(W_sig)/r) / tr(W_null)
    scale_sig = 1.0 / float(np.sqrt(r))
    valid = (r >= r_min_valid)
    if trW_null > 1e-10:
        scale_null_sq = (trW - trW_sig / r) / trW_null
        if scale_null_sq < 0:
            # r outside valid range: preserve kappa approx by clamping (but mark invalid)
            scale_null_sq = 0.0
        scale_null = float(np.sqrt(scale_null_sq))
    else:
        scale_null = 1.0

    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        z = X[mask] - centroids[c]  # (n_c, d)
        # Project onto signal subspace
        z_sig_coords = z @ P_B.T  # (n_c, K-1)
        z_sig = z_sig_coords @ P_B  # (n_c, d) back-projected
        z_null = z - z_sig  # (n_c, d) null component
        X_new[mask] = centroids[c] + scale_sig * z_sig + scale_null * z_null

    return X_new, float(scale_sig), float(scale_null), valid, float(r_min_valid)


def compute_q_knn(X_tr, y_tr, X_te, y_te):
    """Compute normalized 1-NN accuracy."""
    classes = np.unique(y_tr)
    K_actual = len(classes)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = float(np.mean(knn.predict(X_te) == y_te))
    q_norm = float(np.clip((acc - 1.0 / K_actual) / (1.0 - 1.0 / K_actual), 0.001, 0.999))
    return q_norm, acc


def safe_logit(q):
    q = float(np.clip(q, 0.001, 0.999))
    return float(np.log(q / (1 - q)))


def run_seed(seed, X_tr_full, y_tr_full):
    """Run both surgeries for one seed."""
    log(f"\n{'='*60}")
    log(f"SEED {seed}")
    log(f"{'='*60}")

    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(sss.split(X_tr_full, y_tr_full))
    X_tr, y_tr = X_tr_full[tr_idx], y_tr_full[tr_idx]
    X_te, y_te = X_tr_full[te_idx], y_tr_full[te_idx]

    log(f"  Split: {len(X_tr)} train, {len(X_te)} test, K={len(np.unique(y_tr))}")

    # Compute geometry
    geo = compute_geometry(X_tr, y_tr)
    d_eff_base = geo['d_eff_formula']
    kappa_base = geo['kappa_nearest']
    kappa_eff_base = kappa_base * float(np.sqrt(d_eff_base))
    log(f"  Geometry: d_eff={d_eff_base:.3f}, kappa={kappa_base:.4f}, "
        f"kappa_eff={kappa_eff_base:.3f}")
    log(f"  Signal dims: n_sig={geo['n_sig']}, "
        f"trW_sig={geo['trW_sig']:.3f}, trW_null={geo['trW_null']:.3f}")

    # Baseline q
    q_base, acc_base = compute_q_knn(X_tr, y_tr, X_te, y_te)
    logit_base = safe_logit(q_base)
    C_single = logit_base - A_RENORM_K20 * kappa_base * float(np.sqrt(d_eff_base))
    log(f"  Baseline: q={q_base:.4f}, logit={logit_base:.4f}, C_fit={C_single:.4f}")

    records = []
    for r in SURGERY_LEVELS:
        log(f"\n  [r={r:.2f}] ...")

        # ARM 1: Single-direction surgery
        X_tr_s, sa, sp = apply_single_surgery(X_tr, y_tr, geo, r)
        X_te_s, _, _ = apply_single_surgery(X_te, y_te, geo, r)
        q_single, _ = compute_q_knn(X_tr_s, y_tr, X_te_s, y_te)
        logit_single = safe_logit(q_single)
        delta_single = logit_single - logit_base

        # Verify kappa invariance for single surgery
        geo_s = compute_geometry(X_tr_s, y_tr)
        kappa_chg_single = abs(geo_s['kappa_nearest'] - kappa_base) / (kappa_base + 1e-10) * 100

        # ARM 2: Global multi-direction surgery
        X_tr_g, sg, sn, global_valid, r_min_valid = apply_global_surgery(X_tr, y_tr, geo, r)
        X_te_g, _, _, _, _ = apply_global_surgery(X_te, y_te, geo, r)
        q_global, _ = compute_q_knn(X_tr_g, y_tr, X_te_g, y_te)
        logit_global = safe_logit(q_global)
        delta_global = logit_global - logit_base

        # Verify kappa invariance for global surgery
        geo_g = compute_geometry(X_tr_g, y_tr)
        kappa_chg_global = abs(geo_g['kappa_nearest'] - kappa_base) / (kappa_base + 1e-10) * 100

        if not global_valid:
            log(f"    WARNING: r={r} < r_min_valid={r_min_valid:.3f}: "
                f"global surgery invalid (scale_null^2<0). Excluding from H1/H3.")

        # CTI formula prediction
        kappa_eff_new = kappa_base * float(np.sqrt(r * d_eff_base))
        delta_pred_full = A_RENORM_K20 * kappa_base * float(np.sqrt(d_eff_base)) * (float(np.sqrt(r)) - 1)
        delta_pred_attenuated = delta_pred_full / d_eff_base  # 1/d_eff prediction

        # Ratio
        if abs(delta_single) > 1e-4:
            ratio = delta_global / delta_single
        else:
            ratio = float('nan')

        log(f"    Single: q={q_single:.4f}, delta_logit={delta_single:.4f}, "
            f"kappa_chg={kappa_chg_single:.3f}%")
        log(f"    Global: q={q_global:.4f}, delta_logit={delta_global:.4f}, "
            f"kappa_chg={kappa_chg_global:.3f}%")
        log(f"    Pred(full)={delta_pred_full:.4f}, Pred(attenuated)={delta_pred_attenuated:.4f}")
        log(f"    Ratio global/single={ratio:.2f} (predicted d_eff={d_eff_base:.2f})")

        records.append({
            'seed': seed,
            'r': r,
            'd_eff_base': d_eff_base,
            'kappa_base': kappa_base,
            'kappa_eff_base': kappa_eff_base,
            'q_base': q_base,
            'logit_base': logit_base,
            # Single surgery
            'q_single': q_single,
            'logit_single': logit_single,
            'delta_logit_single': delta_single,
            'kappa_chg_single_pct': kappa_chg_single,
            'scale_along': sa,
            'scale_perp': sp,
            # Global surgery
            'q_global': q_global,
            'logit_global': logit_global,
            'delta_logit_global': delta_global,
            'kappa_chg_global_pct': kappa_chg_global,
            'scale_sig': sg,
            'scale_null': sn,
            'global_valid': bool(global_valid),
            'r_min_valid_global': float(r_min_valid),
            # Predictions
            'delta_pred_full': delta_pred_full,
            'delta_pred_attenuated': delta_pred_attenuated,
            # Ratio test
            'ratio_global_single': ratio,
            'ratio_predicted': d_eff_base,
        })

    return records


def analyze_results(all_records):
    """Evaluate pre-registered criteria."""
    log("\n" + "=" * 70)
    log("PRE-REGISTERED EVALUATION")
    log("=" * 70)

    # Restrict to: r != 1.0, global surgery valid, ratio not nan
    ratio_records = [r for r in all_records
                     if r['r'] != 1.0
                     and r.get('global_valid', True)
                     and not np.isnan(r['ratio_global_single'])]
    invalid_records = [r for r in all_records if r['r'] != 1.0 and not r.get('global_valid', True)]
    if invalid_records:
        log(f"\nNOTE: {len(invalid_records)} records excluded (global surgery invalid for r < r_min_valid):")
        log(f"  Excluded (seed, r, r_min_valid): "
            f"{[(r['seed'], r['r'], round(r['r_min_valid_global'], 3)) for r in invalid_records]}")
    ratios = [r['ratio_global_single'] for r in ratio_records]
    d_effs = [r['d_eff_base'] for r in ratio_records]
    median_ratio = float(np.median(ratios)) if ratios else float('nan')
    median_d_eff = float(np.median(d_effs)) if d_effs else float('nan')

    log(f"\nH1 (ratio in [d_eff/3, d_eff*3]):")
    log(f"  d_eff_base values: {[round(d, 2) for d in d_effs]}")
    log(f"  Ratios (global/single): {[round(x, 2) for x in ratios]}")
    log(f"  Median ratio = {median_ratio:.2f}, Median d_eff = {median_d_eff:.2f}")
    h1_pass = (median_d_eff / 3 <= median_ratio <= median_d_eff * 3) if not np.isnan(median_ratio) else False
    log(f"  H1: {'PASS' if h1_pass else 'FAIL'} "
        f"(interval [{median_d_eff/3:.2f}, {median_d_eff*3:.2f}])")

    log(f"\nH2 (global direction consistent):")
    direction_correct = 0
    for r in ratio_records:
        if r['r'] > 1.0 and r['delta_logit_global'] > 0:
            direction_correct += 1
        elif r['r'] < 1.0 and r['delta_logit_global'] < 0:
            direction_correct += 1
    h2_pass = direction_correct >= 3 * len([r for r in ratio_records if r['r'] != 1.0]) / 4
    log(f"  Direction correct: {direction_correct}/{len(ratio_records)} pairs")
    log(f"  H2: {'PASS' if h2_pass else 'FAIL'}")

    log(f"\nH3 (kappa invariance < 0.5%, valid records only):")
    valid_records_h3 = [r for r in all_records if r.get('global_valid', True)]
    max_kappa_single = max((r['kappa_chg_single_pct'] for r in valid_records_h3), default=float('nan'))
    max_kappa_global = max((r['kappa_chg_global_pct'] for r in valid_records_h3), default=float('nan'))
    h3_pass = max_kappa_single < 0.5 and max_kappa_global < 0.5
    log(f"  Max kappa change single (valid): {max_kappa_single:.4f}%")
    log(f"  Max kappa change global (valid): {max_kappa_global:.4f}%")
    log(f"  H3: {'PASS' if h3_pass else 'FAIL'}")

    overall_pass = h1_pass and h2_pass and h3_pass
    log(f"\n{'='*70}")
    log(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    if overall_pass:
        log("  INTERPRETATION: 1/d_eff attenuation hypothesis CONFIRMED.")
        log("  The 16x prior failure is a predicted partial-intervention effect.")
        log("  CTI law is causally valid; single-direction surgery is the limiting factor.")
    else:
        log("  INTERPRETATION: 1/d_eff hypothesis not confirmed at this tolerance.")
        if not h1_pass:
            log(f"  H1 FAIL: ratio={median_ratio:.2f} outside [{median_d_eff/3:.2f},{median_d_eff*3:.2f}]")

    return {
        'h1_pass': bool(h1_pass),
        'h2_pass': bool(h2_pass),
        'h3_pass': bool(h3_pass),
        'overall_pass': bool(overall_pass),
        'median_ratio': float(median_ratio),
        'median_d_eff': float(median_d_eff),
        'all_ratios': [float(x) for x in ratios],
        'all_d_effs': [float(x) for x in d_effs],
        'max_kappa_chg_single': float(max_kappa_single),
        'max_kappa_chg_global': float(max_kappa_global),
    }


def main():
    log("=" * 70)
    log("GLOBAL vs SINGLE SURGERY RATIO TEST (PRE-REGISTERED)")
    log(f"Pre-reg: src/PREREGISTRATION_global_surgery_ratio.md")
    log(f"Pre-reg commit: {PREREGISTERED_COMMIT}")
    log(f"A_renorm(K=20) = {A_RENORM_K20}")
    log(f"Surgery levels r: {SURGERY_LEVELS}")
    log(f"H1: median ratio in [d_eff/3, d_eff*3]")
    log("=" * 70)

    # Load saved linear regime embeddings
    seed_files = []
    for s in [0, 1, 2]:
        for ep in [4, 5]:
            tr_path = os.path.join(EMBED_DIR, f"X_tr_seed{s}_ep{ep}.npy")
            yl_path = os.path.join(EMBED_DIR, f"y_tr_seed{s}_ep{ep}.npy")
            if os.path.exists(tr_path) and os.path.exists(yl_path):
                seed_files.append((s, ep, tr_path, yl_path))
                break
        else:
            log(f"  WARNING: No saved embeddings for seed={s}. Skipping.")

    log(f"\nFound {len(seed_files)} seed(s): {[(s, ep) for s, ep, _, _ in seed_files]}")

    all_records = []
    for seed, epoch, tr_path, yl_path in seed_files:
        log(f"\nLoading seed={seed} epoch={epoch}: {tr_path}")
        X_tr_full = np.load(tr_path)
        y_tr_full = np.load(yl_path)
        log(f"  Shape: {X_tr_full.shape}, K={len(np.unique(y_tr_full))}")
        records = run_seed(seed, X_tr_full, y_tr_full)
        all_records.extend(records)

    if not all_records:
        log("ERROR: No records generated. Check embedding files.")
        return

    # Analyze
    analysis = analyze_results(all_records)

    # Save results
    output = {
        'experiment': 'global_vs_single_surgery_ratio_test',
        'preregistration': 'src/PREREGISTRATION_global_surgery_ratio.md',
        'preregistered_commit': PREREGISTERED_COMMIT,
        'A_renorm': A_RENORM_K20,
        'surgery_levels': SURGERY_LEVELS,
        'n_seeds': len(seed_files),
        'n_records': len(all_records),
        'analysis': analysis,
        'records': all_records,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {OUTPUT_PATH}")

    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))
    log(f"Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
