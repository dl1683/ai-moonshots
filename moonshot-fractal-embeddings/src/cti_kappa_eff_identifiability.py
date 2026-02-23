"""
kappa_eff Identifiability via Null-Space Scaling
Pre-registration: src/PREREGISTRATION_kappa_eff_identifiability.md
Pre-reg commit: 6938940

KEY PREDICTION: kappa_eff = kappa * sqrt(d_eff) = delta_min / sigma_centroid_dir
is invariant to null-space scaling of within-class residuals. If q is ALSO invariant
(H4 PRIMARY), then kappa_eff is the sufficient statistic for 1-NN accuracy.

Method: scale within-class null-space components by s, leaving signal-subspace intact.
- kappa CHANGES (decreases for s>1)
- d_eff CHANGES (increases for s>1)
- kappa_eff INVARIANT (exactly, by construction)
- q PREDICTED INVARIANT (CTI law predicts q = f(kappa_eff))
"""
import os
import json
import sys
import numpy as np
from scipy.special import logit as scipy_logit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_kappa_eff_identifiability.json")
PREREGISTERED_COMMIT = "6938940"

K = 20
NULL_SCALES = [0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

# Pre-saved NLP embeddings (last layer, eval mode)
ARCHITECTURES = [
    {"name": "deberta-base", "file": "causal_v2_embs_deberta-base_20newsgroups.npz"},
    {"name": "olmo-1b",      "file": "causal_v2_embs_olmo-1b_20newsgroups.npz"},
    {"name": "qwen3-0.6b",   "file": "causal_v2_embs_qwen3-0.6b_20newsgroups.npz"},
]

log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)


def compute_geometry_and_subspace(X_tr, y_tr):
    """Compute CTI geometry + signal/null subspace decomposition."""
    classes = np.unique(y_tr)
    K_actual = len(classes)
    N, d = X_tr.shape

    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])

    # tr(Sigma_W): total within-class variance
    trW = 0.0
    for i, c in enumerate(classes):
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[i]
        trW += float(np.sum(Xc_c ** 2)) / N

    sigma_W_global = float(np.sqrt(trW / d + 1e-10))

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
    for i, c in enumerate(classes):
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[i]
        proj = Xc_c @ Delta_hat
        sigma_centroid_sq += float(np.sum(proj ** 2)) / N

    sigma_centroid_dir = float(np.sqrt(sigma_centroid_sq + 1e-10))
    d_eff_formula = float(trW / (sigma_centroid_sq + 1e-10))
    kappa_eff = float(kappa_nearest * np.sqrt(d_eff_formula))

    # Signal subspace: top K-1 singular vectors of centered centroids
    grand_mean = centroids.mean(0)
    centroids_centered = centroids - grand_mean
    try:
        _, _, Vt = np.linalg.svd(centroids_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        from sklearn.utils.extmath import randomized_svd
        n_comp = min(K_actual - 1, d - 1, 50)
        _, _, Vt = randomized_svd(centroids_centered, n_components=n_comp, random_state=42)

    n_sig = min(K_actual - 1, d, Vt.shape[0])
    P_B = Vt[:n_sig, :]  # (K-1, d) -- signal subspace basis

    # Decompose trW into signal and null
    trW_sig = 0.0
    for i, c in enumerate(classes):
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[i]
        Xc_proj = Xc_c @ P_B.T  # (n_c, K-1)
        trW_sig += float(np.sum(Xc_proj ** 2)) / N

    trW_null = trW - trW_sig

    return {
        'classes': classes,
        'centroids': centroids,
        'Delta_hat': Delta_hat,
        'delta_min': delta_min,
        'trW': trW,
        'trW_sig': trW_sig,
        'trW_null': trW_null,
        'sigma_W_global': sigma_W_global,
        'sigma_centroid_sq': sigma_centroid_sq,
        'sigma_centroid_dir': sigma_centroid_dir,
        'd_eff_formula': d_eff_formula,
        'kappa_nearest': kappa_nearest,
        'kappa_eff': kappa_eff,
        'K_actual': K_actual,
        'd': d,
        'P_B': P_B,
        'n_sig': n_sig,
    }


def apply_null_scaling(X, y, geometry, s):
    """Scale within-class null-space components by s.

    Preserves: class centroids, signal-subspace variance
    Changes: null-space variance (scales by s^2)
    Invariant: kappa_eff = delta_min / sigma_centroid_dir (mathematical guarantee)
    """
    centroids = geometry['centroids']
    classes = geometry['classes']
    P_B = geometry['P_B']  # (K-1, d) signal subspace

    X_new = X.copy()
    for i, c in enumerate(classes):
        mask = (y == c)
        z = X[mask] - centroids[i]        # within-class residuals (n_c, d)
        z_sig_coords = z @ P_B.T          # (n_c, K-1) signal coords
        z_sig = z_sig_coords @ P_B        # (n_c, d) signal component
        z_null = z - z_sig                # (n_c, d) null component
        X_new[mask] = centroids[i] + z_sig + s * z_null
    return X_new


def recompute_geometry_from_scaled(X_tr, y_tr, geometry_base):
    """Recompute kappa, d_eff, kappa_eff from scaled embeddings.

    Uses same Delta_hat and P_B as baseline (computed from unscaled train set),
    since centroids don't change under null-space scaling applied to residuals.
    """
    classes = geometry_base['classes']
    centroids = geometry_base['centroids']  # unchanged by null scaling
    Delta_hat = geometry_base['Delta_hat']  # unchanged (centroids unchanged)
    P_B = geometry_base['P_B']             # unchanged (centroids unchanged)
    K_actual = geometry_base['K_actual']
    N, d = X_tr.shape

    # Recompute tr(Sigma_W) from scaled embeddings
    trW_new = 0.0
    for i, c in enumerate(classes):
        Xc = X_tr[y_tr == c]
        Xc_c = Xc - centroids[i]
        trW_new += float(np.sum(Xc_c ** 2)) / N

    sigma_W_new = float(np.sqrt(trW_new / d + 1e-10))
    kappa_new = float(geometry_base['delta_min'] / (sigma_W_new * np.sqrt(d) + 1e-10))

    # sigma_centroid_dir is theoretically invariant; compute empirically to verify
    sigma_centroid_sq_new = 0.0
    for i, c in enumerate(classes):
        Xc = X_tr[y_tr == c]
        Xc_c = Xc - centroids[i]
        proj = Xc_c @ Delta_hat
        sigma_centroid_sq_new += float(np.sum(proj ** 2)) / N

    sigma_centroid_dir_new = float(np.sqrt(sigma_centroid_sq_new + 1e-10))
    d_eff_new = float(trW_new / (sigma_centroid_sq_new + 1e-10))
    kappa_eff_new = float(kappa_new * np.sqrt(d_eff_new))
    # Alternative: kappa_eff = delta_min / sigma_centroid_dir (direct formula)
    kappa_eff_direct = float(geometry_base['delta_min'] / (sigma_centroid_dir_new + 1e-10))

    return {
        'trW': trW_new,
        'sigma_W': sigma_W_new,
        'kappa': kappa_new,
        'sigma_centroid_dir': sigma_centroid_dir_new,
        'd_eff': d_eff_new,
        'kappa_eff': kappa_eff_new,
        'kappa_eff_direct': kappa_eff_direct,
    }


def compute_q_knn(X_tr, y_tr, X_te, y_te):
    K_actual = len(np.unique(y_tr))
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = float(np.mean(knn.predict(X_te) == y_te))
    q_norm = float(np.clip((acc - 1.0 / K_actual) / (1.0 - 1.0 / K_actual), 0.001, 0.999))
    return q_norm, acc


def run_arch(arch_idx, arch_name, X_full, y_full):
    log(f"\n{'='*60}")
    log(f"ARCH {arch_idx}: {arch_name}")
    log(f"{'='*60}")

    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(X_full, y_full))
    X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
    X_te, y_te = X_full[te_idx], y_full[te_idx]

    log(f"  Split: {len(X_tr)} train, {len(X_te)} test, K={len(np.unique(y_tr))}, d={X_full.shape[1]}")

    # Compute baseline geometry (s=1.0, no scaling)
    log("  Computing baseline geometry...")
    geo = compute_geometry_and_subspace(X_tr, y_tr)
    log(f"  Baseline: kappa={geo['kappa_nearest']:.4f}, d_eff={geo['d_eff_formula']:.2f}, kappa_eff={geo['kappa_eff']:.4f}")
    log(f"  trW={geo['trW']:.2f}, trW_sig={geo['trW_sig']:.2f}, trW_null={geo['trW_null']:.2f}")
    log(f"  sigma_centroid_dir={geo['sigma_centroid_dir']:.4f}, delta_min={geo['delta_min']:.4f}")

    records = []

    for s in NULL_SCALES:
        log(f"\n  [s={s:.2f}] ...")

        # Apply null-space scaling to both train and test
        X_tr_scaled = apply_null_scaling(X_tr, y_tr, geo, s)
        X_te_scaled = apply_null_scaling(X_te, y_te, geo, s)

        # Recompute geometry from scaled train embeddings
        geo_s = recompute_geometry_from_scaled(X_tr_scaled, y_tr, geo)

        # Compute 1-NN accuracy
        q_norm, acc = compute_q_knn(X_tr_scaled, y_tr, X_te_scaled, y_te)

        kappa_ratio = geo_s['kappa'] / (geo['kappa_nearest'] + 1e-10)
        d_eff_ratio = geo_s['d_eff'] / (geo['d_eff_formula'] + 1e-10)
        kappa_eff_ratio = geo_s['kappa_eff'] / (geo['kappa_eff'] + 1e-10)
        q_delta = q_norm - records[NULL_SCALES.index(1.0)]['q_norm'] if 1.0 in NULL_SCALES[:NULL_SCALES.index(s)] else None

        log(f"    kappa={geo_s['kappa']:.4f} (ratio={kappa_ratio:.3f})")
        log(f"    d_eff={geo_s['d_eff']:.2f} (ratio={d_eff_ratio:.3f})")
        log(f"    kappa_eff={geo_s['kappa_eff']:.4f} (ratio={kappa_eff_ratio:.3f})")
        log(f"    kappa_eff_direct={geo_s['kappa_eff_direct']:.4f}")
        log(f"    q={q_norm:.4f}, acc={acc:.4f}")

        records.append({
            's': s,
            'kappa': float(geo_s['kappa']),
            'd_eff': float(geo_s['d_eff']),
            'kappa_eff': float(geo_s['kappa_eff']),
            'kappa_eff_direct': float(geo_s['kappa_eff_direct']),
            'q_norm': float(q_norm),
            'acc': float(acc),
            'kappa_ratio': float(kappa_ratio),
            'd_eff_ratio': float(d_eff_ratio),
            'kappa_eff_ratio': float(kappa_eff_ratio),
        })

    # Pre-registered evaluation
    kappas = [r['kappa'] for r in records]
    d_effs = [r['d_eff'] for r in records]
    kappa_effs = [r['kappa_eff'] for r in records]
    qs = [r['q_norm'] for r in records]

    # H1: kappa_eff varies < 1%
    kappa_eff_cv = float(np.std(kappa_effs) / (np.mean(kappa_effs) + 1e-10))
    h1_pass = kappa_eff_cv < 0.01
    log(f"\n  H1 (kappa_eff invariance): CV={kappa_eff_cv*100:.3f}% {'PASS' if h1_pass else 'FAIL'} (threshold 1%)")

    # H2: kappa at s=3 < 0.7 * kappa at s=0.5
    kappa_s0p5 = records[NULL_SCALES.index(0.5)]['kappa']
    kappa_s3p0 = records[NULL_SCALES.index(3.0)]['kappa']
    h2_ratio = kappa_s3p0 / (kappa_s0p5 + 1e-10)
    h2_pass = h2_ratio < 0.7
    log(f"  H2 (kappa sensitivity): kappa(s=3)/kappa(s=0.5)={h2_ratio:.3f} {'PASS' if h2_pass else 'FAIL'} (threshold <0.7)")

    # H3: d_eff at s=3 > 1.5 * d_eff at s=0.5
    d_eff_s0p5 = records[NULL_SCALES.index(0.5)]['d_eff']
    d_eff_s3p0 = records[NULL_SCALES.index(3.0)]['d_eff']
    h3_ratio = d_eff_s3p0 / (d_eff_s0p5 + 1e-10)
    h3_pass = h3_ratio > 1.5
    log(f"  H3 (d_eff sensitivity): d_eff(s=3)/d_eff(s=0.5)={h3_ratio:.3f} {'PASS' if h3_pass else 'FAIL'} (threshold >1.5)")

    # H4 (PRIMARY): q varies < 5% across all s
    q_range = float(max(qs) - min(qs))
    h4_pass = q_range < 0.05
    log(f"  H4 (q invariance PRIMARY): range={q_range:.4f} {'PASS' if h4_pass else 'FAIL'} (threshold <0.05)")

    # H5: correlation analysis
    from scipy.stats import spearmanr
    rho_kappa_q, _ = spearmanr(kappas, qs)
    rho_kappa_eff_dq, _ = spearmanr(kappa_effs, [abs(q - records[3]['q_norm']) for q in qs])
    log(f"  H5: r(kappa, q)={rho_kappa_q:.3f}, r(kappa_eff_deviation, q_deviation)={rho_kappa_eff_dq:.3f}")

    return {
        'arch': arch_name,
        'baseline_kappa': float(geo['kappa_nearest']),
        'baseline_d_eff': float(geo['d_eff_formula']),
        'baseline_kappa_eff': float(geo['kappa_eff']),
        'baseline_sigma_centroid_dir': float(geo['sigma_centroid_dir']),
        'baseline_delta_min': float(geo['delta_min']),
        'records': records,
        'h1_pass': h1_pass,
        'h1_kappa_eff_cv': kappa_eff_cv,
        'h2_pass': h2_pass,
        'h2_kappa_ratio': float(h2_ratio),
        'h3_pass': h3_pass,
        'h3_d_eff_ratio': float(h3_ratio),
        'h4_pass': h4_pass,
        'h4_q_range': q_range,
        'rho_kappa_q': float(rho_kappa_q),
        'rho_kappa_eff_dq': float(rho_kappa_eff_dq),
    }


def main():
    log("=" * 70)
    log("kappa_eff IDENTIFIABILITY TEST (NULL-SPACE SCALING)")
    log("=" * 70)
    log(f"Pre-reg commit: {PREREGISTERED_COMMIT}")
    log(f"Null-space scales: {NULL_SCALES}")
    log(f"H4 (PRIMARY): q varies < 5% when kappa_eff is invariant")
    log("")

    all_arch_results = []

    for arch_idx, arch_info in enumerate(ARCHITECTURES):
        emb_path = os.path.join(RESULTS_DIR, arch_info["file"])
        if not os.path.exists(emb_path):
            log(f"\nSKIP {arch_info['name']}: file not found at {emb_path}")
            continue

        log(f"\nLoading {arch_info['name']} from {emb_path}")
        data = np.load(emb_path)
        X_full = data['X'].astype(np.float32)
        y_full = data['y'].astype(np.int32)

        # Sanitize NaN/Inf
        X_full = np.where(np.isfinite(X_full), X_full, 0.0)

        log(f"  Shape: {X_full.shape}, K={len(np.unique(y_full))}")

        result = run_arch(arch_idx, arch_info["name"], X_full, y_full)
        all_arch_results.append(result)

    # Overall evaluation
    log("\n" + "=" * 70)
    log("PRE-REGISTERED EVALUATION")
    log("=" * 70)

    valid = [r for r in all_arch_results]
    h1_archs = sum(r['h1_pass'] for r in valid)
    h2_archs = sum(r['h2_pass'] for r in valid)
    h3_archs = sum(r['h3_pass'] for r in valid)
    h4_archs = sum(r['h4_pass'] for r in valid)

    log(f"\nH1 (kappa_eff invariant): {h1_archs}/{len(valid)} archs PASS")
    log(f"H2 (kappa sensitive):    {h2_archs}/{len(valid)} archs PASS")
    log(f"H3 (d_eff sensitive):    {h3_archs}/{len(valid)} archs PASS")
    log(f"H4 (q invariant PRIMARY): {h4_archs}/{len(valid)} archs PASS")

    overall_pass = h1_archs >= 2 and h4_archs >= 2
    log(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    if overall_pass:
        log("  kappa_eff = delta_min/sigma_centroid_dir is the sufficient statistic for q.")
        log("  Changing d_eff and kappa inversely while holding kappa_eff constant")
        log("  leaves 1-NN accuracy unchanged.")
    else:
        log("  q changes even when kappa_eff is constant.")
        log("  kappa_eff alone is not sufficient; higher-order geometry matters.")

    # Save
    output = {
        'experiment': 'kappa_eff_identifiability_null_scaling',
        'preregistered_commit': PREREGISTERED_COMMIT,
        'null_scales': NULL_SCALES,
        'arch_results': all_arch_results,
        'evaluation': {
            'h1_archs': h1_archs,
            'h2_archs': h2_archs,
            'h3_archs': h3_archs,
            'h4_archs': h4_archs,
            'overall_pass': overall_pass,
        }
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
