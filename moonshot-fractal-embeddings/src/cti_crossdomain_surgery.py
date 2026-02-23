"""
Cross-Domain Global Surgery Ratio Test (Pre-registered: src/PREREGISTRATION_crossdomain_surgery.md)
Pre-reg commit: 09ba558

Tests whether the 1/d_eff attenuation mechanism generalises from vision (CIFAR-100)
to NLP text LM embeddings.

Same H1/H2/H3 criteria as CIFAR surgery (commit 59faa5d).
Uses pre-saved causal_v2 embeddings (last-layer, eval mode, no augmentation).
"""
import os
import json
import numpy as np
from scipy.special import logit as scipy_logit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_crossdomain_surgery.json")
LOG_PATH = os.path.join(RESULTS_DIR, "cti_crossdomain_surgery_log.txt")

K = 20
A_RENORM_K20 = 1.0535  # Pre-registered constant (Theorem 15, K=20)
SURGERY_LEVELS = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
PREREGISTERED_COMMIT = "09ba558"

# Pre-saved NLP embeddings (last layer, eval mode, no augmentation)
ARCHITECTURES = [
    {"name": "deberta-base",  "file": "causal_v2_embs_deberta-base_20newsgroups.npz"},
    {"name": "olmo-1b",       "file": "causal_v2_embs_olmo-1b_20newsgroups.npz"},
    {"name": "qwen3-0.6b",    "file": "causal_v2_embs_qwen3-0.6b_20newsgroups.npz"},
]

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

    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])

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
    P_B = Vt[:n_sig, :]  # (K-1, d)

    trW_sig = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        Xc_proj = Xc_c @ P_B.T
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
        'd_eff_formula': d_eff_formula,
        'kappa_nearest': kappa_nearest,
        'K_actual': K_actual,
        'd': d,
        'P_B': P_B,
        'n_sig': n_sig,
    }


def apply_single_surgery(X, y, geometry, r):
    centroids = geometry['centroids']
    Delta_hat = geometry['Delta_hat']
    trW = geometry['trW']
    sigma_centroid_sq = geometry['sigma_centroid_sq']
    classes = np.unique(y)

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
    centroids = geometry['centroids']
    P_B = geometry['P_B']
    trW = geometry['trW']
    trW_sig = geometry['trW_sig']
    trW_null = geometry['trW_null']
    classes = np.unique(y)

    r_min_valid = trW_sig / (trW + 1e-10)
    scale_sig = 1.0 / float(np.sqrt(r))
    valid = (r >= r_min_valid)
    if trW_null > 1e-10:
        scale_null_sq = (trW - trW_sig / r) / trW_null
        if scale_null_sq < 0:
            scale_null_sq = 0.0
        scale_null = float(np.sqrt(scale_null_sq))
    else:
        scale_null = 1.0

    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        z = X[mask] - centroids[c]
        z_sig_coords = z @ P_B.T
        z_sig = z_sig_coords @ P_B
        z_null = z - z_sig
        X_new[mask] = centroids[c] + scale_sig * z_sig + scale_null * z_null
    return X_new, float(scale_sig), float(scale_null), valid, float(r_min_valid)


def compute_q_knn(X_tr, y_tr, X_te, y_te):
    classes = np.unique(y_tr)
    K_actual = len(classes)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn.fit(X_tr, y_tr)
    acc = float(np.mean(knn.predict(X_te) == y_te))
    q_norm = float(np.clip((acc - 1.0 / K_actual) / (1.0 - 1.0 / K_actual), 0.001, 0.999))
    return q_norm, acc


def safe_logit(q):
    q = float(np.clip(q, 0.001, 0.999))
    return float(np.log(q / (1 - q)))


def run_arch(arch_idx, arch_name, X_full, y_full):
    log(f"\n{'='*60}")
    log(f"ARCH {arch_idx}: {arch_name}")
    log(f"{'='*60}")

    # Stratified 80/20 split (use arch_idx as random_state for reproducibility)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=arch_idx)
    tr_idx, te_idx = next(sss.split(X_full, y_full))
    X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
    X_te, y_te = X_full[te_idx], y_full[te_idx]

    log(f"  Split: {len(X_tr)} train, {len(X_te)} test, K={len(np.unique(y_tr))}, d={X_full.shape[1]}")

    geo = compute_geometry(X_tr, y_tr)
    d_eff_base = geo['d_eff_formula']
    kappa_base = geo['kappa_nearest']
    kappa_eff_base = kappa_base * float(np.sqrt(d_eff_base))
    log(f"  Geometry: d_eff={d_eff_base:.3f}, kappa={kappa_base:.4f}, kappa_eff={kappa_eff_base:.3f}")
    log(f"  Signal dims: n_sig={geo['n_sig']}, trW_sig={geo['trW_sig']:.3f}, trW_null={geo['trW_null']:.3f}")

    q_base, acc_base = compute_q_knn(X_tr, y_tr, X_te, y_te)
    logit_base = safe_logit(q_base)
    C_fit = logit_base - A_RENORM_K20 * kappa_base * float(np.sqrt(d_eff_base))
    log(f"  Baseline: q={q_base:.4f}, logit={logit_base:.4f}, C_fit={C_fit:.4f}")

    records = []
    for r in SURGERY_LEVELS:
        log(f"\n  [r={r:.2f}] ...")

        # ARM 1: Single-direction surgery
        X_tr_s, sa, sp = apply_single_surgery(X_tr, y_tr, geo, r)
        X_te_s, _, _ = apply_single_surgery(X_te, y_te, geo, r)
        q_single, _ = compute_q_knn(X_tr_s, y_tr, X_te_s, y_te)
        logit_single = safe_logit(q_single)
        delta_single = logit_single - logit_base

        geo_s = compute_geometry(X_tr_s, y_tr)
        kappa_chg_single = abs(geo_s['kappa_nearest'] - kappa_base) / (kappa_base + 1e-10) * 100

        # ARM 2: Global surgery
        X_tr_g, sg, sn, global_valid, r_min_valid = apply_global_surgery(X_tr, y_tr, geo, r)
        X_te_g, _, _, _, _ = apply_global_surgery(X_te, y_te, geo, r)
        q_global, _ = compute_q_knn(X_tr_g, y_tr, X_te_g, y_te)
        logit_global = safe_logit(q_global)
        delta_global = logit_global - logit_base

        geo_g = compute_geometry(X_tr_g, y_tr)
        kappa_chg_global = abs(geo_g['kappa_nearest'] - kappa_base) / (kappa_base + 1e-10) * 100

        if not global_valid:
            log(f"    WARNING: r={r} < r_min_valid={r_min_valid:.3f}: global surgery invalid. Excluding from H1/H3.")

        # CTI formula predictions
        delta_pred_full = A_RENORM_K20 * kappa_base * float(np.sqrt(d_eff_base)) * (float(np.sqrt(r)) - 1)
        delta_pred_attenuated = delta_pred_full / d_eff_base  # 1/d_eff prediction

        ratio = float('nan')
        if abs(delta_single) > 1e-6:
            ratio = delta_global / delta_single

        log(f"    Single: q={q_single:.4f}, delta_logit={delta_single:.4f}, kappa_chg={kappa_chg_single:.3f}%")
        log(f"    Global: q={q_global:.4f}, delta_logit={delta_global:.4f}, kappa_chg={kappa_chg_global:.3f}%")
        log(f"    Pred(full)={delta_pred_full:.4f}, Pred(attenuated)={delta_pred_attenuated:.4f}")
        if not np.isnan(ratio):
            log(f"    Ratio global/single={ratio:.2f} (predicted d_eff={d_eff_base:.2f})")

        records.append({
            'arch': arch_name,
            'arch_idx': arch_idx,
            'r': r,
            'q_base': q_base,
            'logit_base': logit_base,
            'q_single': q_single,
            'delta_single': delta_single,
            'kappa_chg_single': kappa_chg_single,
            'q_global': q_global,
            'delta_global': delta_global,
            'kappa_chg_global': kappa_chg_global,
            'global_valid': global_valid,
            'r_min_valid': r_min_valid,
            'ratio_global_over_single': ratio,
            'd_eff_base': d_eff_base,
            'kappa_base': kappa_base,
            'kappa_eff_base': kappa_eff_base,
            'delta_pred_full': delta_pred_full,
            'delta_pred_attenuated': delta_pred_attenuated,
        })

    return records, d_eff_base


def main():
    log("=" * 70)
    log("CROSS-DOMAIN GLOBAL VS SINGLE SURGERY RATIO TEST (NLP)")
    log("=" * 70)
    log(f"Pre-reg commit: {PREREGISTERED_COMMIT}")
    log(f"A_RENORM_K20 = {A_RENORM_K20}, K = {K}")
    log(f"Surgery levels: {SURGERY_LEVELS}")
    log(f"Architectures: {[a['name'] for a in ARCHITECTURES]}")
    log("")

    all_records = []
    arch_d_effs = {}

    for arch_idx, arch_info in enumerate(ARCHITECTURES):
        arch_name = arch_info['name']
        embed_path = os.path.join(RESULTS_DIR, arch_info['file'])

        if not os.path.exists(embed_path):
            log(f"\nERROR: Embedding file not found: {embed_path}")
            continue

        log(f"\nLoading {arch_name}: {embed_path}")
        data = np.load(embed_path)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.int64)
        log(f"  Shape: {X.shape}, K={len(np.unique(y))}")

        records, d_eff_base = run_arch(arch_idx, arch_name, X, y)
        all_records.extend(records)
        arch_d_effs[arch_name] = d_eff_base

    # ================================================================
    # PRE-REGISTERED EVALUATION
    # ================================================================
    log(f"\n{'='*70}")
    log("PRE-REGISTERED EVALUATION")
    log(f"{'='*70}")

    # Separate valid records (global surgery valid, r != 1.0)
    excluded = [(r['arch'], r['r'], r['r_min_valid'])
                for r in all_records if not r['global_valid'] and r['r'] != 1.0]
    valid_nonbaseline = [r for r in all_records
                         if r['global_valid'] and r['r'] != 1.0 and not np.isnan(r['ratio_global_over_single'])]

    if excluded:
        log(f"\nNOTE: {len(excluded)} records excluded (global surgery invalid):")
        log(f"  Excluded (arch, r, r_min_valid): {excluded}")

    # H1: median ratio in [d_eff/3, d_eff*3]
    log(f"\nH1 (ratio in [d_eff/3, d_eff*3]):")
    d_effs = [r['d_eff_base'] for r in valid_nonbaseline]
    ratios = [r['ratio_global_over_single'] for r in valid_nonbaseline]
    log(f"  d_eff_base values: {[round(x, 2) for x in d_effs]}")
    log(f"  Ratios (global/single): {[round(x, 2) for x in ratios]}")

    h1_pass = False
    if ratios:
        median_ratio = float(np.median(ratios))
        median_d_eff = float(np.median(d_effs))
        lo = median_d_eff / 3
        hi = median_d_eff * 3
        h1_pass = lo <= median_ratio <= hi
        log(f"  Median ratio = {median_ratio:.2f}, Median d_eff = {median_d_eff:.2f}")
        log(f"  H1: {'PASS' if h1_pass else 'FAIL'} (interval [{lo:.2f}, {hi:.2f}])")
    else:
        log("  H1: INCONCLUSIVE (no valid records)")

    # H2: global direction consistent
    log(f"\nH2 (global direction consistent):")
    h2_pairs = [(r['r'], r['delta_global']) for r in valid_nonbaseline]
    h2_correct = sum(1 for r, dg in h2_pairs if (r > 1 and dg > 0) or (r < 1 and dg < 0))
    h2_pass = h2_correct >= max(1, int(0.75 * len(h2_pairs)))
    log(f"  Direction correct: {h2_correct}/{len(h2_pairs)} pairs")
    log(f"  H2: {'PASS' if h2_pass else 'FAIL'}")

    # H3: kappa invariance
    log(f"\nH3 (kappa invariance < 0.5%, valid records only):")
    single_chgs = [r['kappa_chg_single'] for r in valid_nonbaseline]
    global_chgs = [r['kappa_chg_global'] for r in valid_nonbaseline]
    max_single = max(single_chgs) if single_chgs else 0
    max_global = max(global_chgs) if global_chgs else 0
    h3_pass = max_single < 0.5 and max_global < 0.5
    log(f"  Max kappa change single (valid): {max_single:.4f}%")
    log(f"  Max kappa change global (valid): {max_global:.4f}%")
    log(f"  H3: {'PASS' if h3_pass else 'FAIL'}")

    overall_pass = h1_pass and h2_pass and h3_pass
    log(f"\n{'='*70}")
    log(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    if overall_pass:
        log("  INTERPRETATION: 1/d_eff attenuation hypothesis CONFIRMED in NLP domain.")
        log("  Combined with CIFAR-100 PASS (commit 59faa5d), the mechanism is domain-agnostic.")
    else:
        log("  INTERPRETATION: Cross-domain generalisation of 1/d_eff attenuation NOT confirmed.")

    # Save results
    result = {
        'experiment': 'crossdomain_global_vs_single_surgery_nlp',
        'preregistered_commit': PREREGISTERED_COMMIT,
        'cifar_reference_commit': '59faa5d',
        'A_RENORM_K20': A_RENORM_K20,
        'K': K,
        'surgery_levels': SURGERY_LEVELS,
        'architectures': [a['name'] for a in ARCHITECTURES],
        'all_records': all_records,
        'evaluation': {
            'h1_pass': h1_pass,
            'h2_pass': h2_pass,
            'h3_pass': h3_pass,
            'overall_pass': overall_pass,
            'n_valid': len(valid_nonbaseline),
            'n_excluded': len(excluded),
            'excluded': excluded,
            'ratios': ratios,
            'd_effs': d_effs,
            'median_ratio': float(np.median(ratios)) if ratios else None,
            'median_d_eff': float(np.median(d_effs)) if d_effs else None,
            'h2_correct': h2_correct,
            'h2_total': len(h2_pairs),
            'max_kappa_chg_single': max_single,
            'max_kappa_chg_global': max_global,
        }
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\nSaved to {OUTPUT_PATH}")

    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))

    return overall_pass


if __name__ == '__main__':
    main()
