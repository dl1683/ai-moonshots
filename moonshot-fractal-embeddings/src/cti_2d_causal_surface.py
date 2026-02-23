"""
2D Causal Surface: Bivariate kappa x kappa_eff Law
Pre-registration: src/PREREGISTRATION_2d_causal_surface.md
Pre-reg commit: 2bc7da6

Tests: logit(q) = alpha1*kappa + alpha2*kappa_eff + C
via orthogonal 2D factorial manipulation of kappa and kappa_eff.

Factor A (null-space scaling, s_null): changes kappa, invariant kappa_eff
Factor B (compensated signal scaling, s_signal): changes kappa_eff, invariant kappa

Grid: (kappa_fraction, kappa_eff_fraction) in {0.5,1.0,2.0} x {0.5,1.0,2.0}
"""
import os
import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_2d_causal_surface.json")
PREREGISTERED_COMMIT = "2bc7da6"

# Grid targets (a=kappa_frac, b=kappa_eff_frac)
KAPPA_FRACS = [0.5, 1.0, 2.0]
KAPPA_EFF_FRACS = [0.5, 1.0, 2.0]

# Architectures (excluding deberta: trW_null~0)
ARCHITECTURES = [
    {"name": "olmo-1b",    "file": "causal_v2_embs_olmo-1b_20newsgroups.npz"},
    {"name": "qwen3-0.6b", "file": "causal_v2_embs_qwen3-0.6b_20newsgroups.npz"},
]

log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)


def safe_logit(q, eps=0.001):
    q = float(np.clip(q, eps, 1 - eps))
    return float(np.log(q / (1 - q)))


def compute_base_geometry(X_tr, y_tr):
    """Compute baseline geometry and subspace decomposition."""
    classes = np.unique(y_tr)
    K_actual = len(classes)
    N, d = X_tr.shape

    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])

    trW = 0.0
    for i, c in enumerate(classes):
        Xc_c = X_tr[y_tr == c] - centroids[i]
        trW += float(np.sum(Xc_c ** 2)) / N

    # Signal subspace: top K-1 SVD of centered centroids
    grand_mean = centroids.mean(0)
    centroids_c = centroids - grand_mean
    try:
        _, _, Vt = np.linalg.svd(centroids_c, full_matrices=False)
    except np.linalg.LinAlgError:
        from sklearn.utils.extmath import randomized_svd
        _, _, Vt = randomized_svd(centroids_c, n_components=min(K_actual - 1, 50), random_state=42)
    n_sig = min(K_actual - 1, d, Vt.shape[0])
    P_B = Vt[:n_sig, :]  # (K-1, d)

    # trW_sig and trW_null
    trW_sig = 0.0
    for i, c in enumerate(classes):
        Xc_c = X_tr[y_tr == c] - centroids[i]
        Xc_proj = Xc_c @ P_B.T
        trW_sig += float(np.sum(Xc_proj ** 2)) / N
    trW_null = trW - trW_sig

    # Nearest centroid pair
    min_dist = float('inf')
    min_i, min_j = 0, 1
    for i in range(K_actual):
        for j in range(i + 1, K_actual):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j

    delta_min = float(min_dist)
    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)

    sigma_centroid_sq = 0.0
    for i, c in enumerate(classes):
        Xc_c = X_tr[y_tr == c] - centroids[i]
        sigma_centroid_sq += float(np.sum((Xc_c @ Delta_hat) ** 2)) / N

    sigma_cd = float(np.sqrt(sigma_centroid_sq + 1e-10))
    d_eff = float(trW / (sigma_centroid_sq + 1e-10))
    kappa = float(delta_min / np.sqrt(trW + 1e-10))   # = delta_min / (sigma_W * sqrt(d)) * sqrt(d)
    # Note: kappa_nearest = delta_min / (sigma_W * sqrt(d)), but here we use
    # kappa = delta_min / sqrt(trW) which equals kappa_nearest * sqrt(d/d) = kappa_nearest
    # WAIT: sigma_W = sqrt(trW/d), so sigma_W * sqrt(d) = sqrt(trW). Yes, kappa = kappa_nearest.
    kappa_eff = float(delta_min / (sigma_cd + 1e-10))

    return {
        'classes': classes, 'centroids': centroids, 'P_B': P_B, 'Delta_hat': Delta_hat,
        'delta_min': delta_min, 'trW': trW, 'trW_sig': trW_sig, 'trW_null': trW_null,
        'sigma_cd': sigma_cd, 'd_eff': d_eff, 'kappa': kappa, 'kappa_eff': kappa_eff,
        'K_actual': K_actual, 'd': d, 'n_sig': n_sig,
    }


def apply_2d_scaling(X, y, geo, s_signal, s_null):
    """Apply 2D orthogonal scaling to embeddings.

    s_signal: scale factor for signal-subspace residuals
    s_null: scale factor for null-space residuals

    Combined effect:
    - kappa_eff changes by 1/s_signal (kappa_eff_new = kappa_eff_base / s_signal)
    - kappa changes according to new tr(Sigma_W)
    """
    centroids = geo['centroids']
    classes = geo['classes']
    P_B = geo['P_B']

    X_new = X.copy()
    for i, c in enumerate(classes):
        mask = (y == c)
        z = X[mask] - centroids[i]
        z_sig_coords = z @ P_B.T
        z_sig = z_sig_coords @ P_B
        z_null = z - z_sig
        X_new[mask] = centroids[i] + s_signal * z_sig + s_null * z_null
    return X_new


def recompute_kappa_kappa_eff(X_tr, y_tr, geo):
    """Recompute kappa and kappa_eff from modified embeddings."""
    classes = geo['classes']
    centroids = geo['centroids']   # centroids are unchanged
    Delta_hat = geo['Delta_hat']   # centroid direction unchanged
    P_B = geo['P_B']
    N, d = X_tr.shape

    trW_new = 0.0
    for i, c in enumerate(classes):
        Xc_c = X_tr[y_tr == c] - centroids[i]
        trW_new += float(np.sum(Xc_c ** 2)) / N

    # sigma_cd_new (empirical, should equal s_signal * sigma_cd_base)
    sigma_cd_sq_new = 0.0
    for i, c in enumerate(classes):
        Xc_c = X_tr[y_tr == c] - centroids[i]
        sigma_cd_sq_new += float(np.sum((Xc_c @ Delta_hat) ** 2)) / N

    sigma_cd_new = float(np.sqrt(sigma_cd_sq_new + 1e-10))
    kappa_new = float(geo['delta_min'] / np.sqrt(trW_new + 1e-10))
    kappa_eff_new = float(geo['delta_min'] / (sigma_cd_new + 1e-10))
    d_eff_new = float(trW_new / (sigma_cd_sq_new + 1e-10))

    return {
        'trW': trW_new, 'sigma_cd': sigma_cd_new,
        'kappa': kappa_new, 'kappa_eff': kappa_eff_new, 'd_eff': d_eff_new
    }


def compute_q(X_tr, y_tr, X_te, y_te):
    K = len(np.unique(y_tr))
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = float(np.mean(knn.predict(X_te) == y_te))
    q = float(np.clip((acc - 1.0 / K) / (1.0 - 1.0 / K), 0.001, 0.999))
    return q, acc


def compute_s_null_for_compensation(geo, a, b):
    """Compute s_null that achieves kappa = a * kappa_base with s_signal = 1/b.

    Solve: a^2 * trW = s_signal^2 * trW_sig + s_null^2 * trW_null
    with s_signal = 1/b.
    """
    s_signal = 1.0 / b
    s_null_sq = (a ** 2 * geo['trW'] - s_signal ** 2 * geo['trW_sig']) / (geo['trW_null'] + 1e-10)
    return s_signal, float(s_null_sq)


def run_arch(arch_name, X_full, y_full):
    log(f"\n{'='*60}")
    log(f"ARCH: {arch_name}")
    log(f"{'='*60}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(X_full, y_full))
    X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
    X_te, y_te = X_full[te_idx], y_full[te_idx]

    log(f"  Split: {len(X_tr)} train, {len(X_te)} test")

    geo = compute_base_geometry(X_tr, y_tr)
    log(f"  Baseline: kappa={geo['kappa']:.4f}, kappa_eff={geo['kappa_eff']:.4f}")
    log(f"  trW={geo['trW']:.3f}, trW_sig={geo['trW_sig']:.3f}, trW_null={geo['trW_null']:.3f}")
    log(f"  d_eff={geo['d_eff']:.1f}")

    if geo['trW_null'] < 0.001:
        log(f"  SKIP: trW_null~0, orthogonal manipulation infeasible")
        return None

    records = []
    kappa_base = geo['kappa']
    kappa_eff_base = geo['kappa_eff']

    for a in KAPPA_FRACS:
        for b in KAPPA_EFF_FRACS:
            s_signal, s_null_sq = compute_s_null_for_compensation(geo, a, b)

            if s_null_sq < 0.01:
                log(f"\n  (a={a}, b={b}): INFEASIBLE s_null^2={s_null_sq:.3f} < 0.01, skipping")
                continue

            s_null = float(np.sqrt(s_null_sq))
            log(f"\n  (a={a}, b={b}): s_signal={s_signal:.3f}, s_null={s_null:.3f}")

            X_tr_mod = apply_2d_scaling(X_tr, y_tr, geo, s_signal, s_null)
            X_te_mod = apply_2d_scaling(X_te, y_te, geo, s_signal, s_null)

            geom_new = recompute_kappa_kappa_eff(X_tr_mod, y_tr, geo)
            q, acc = compute_q(X_tr_mod, y_tr, X_te_mod, y_te)

            kappa_actual_frac = geom_new['kappa'] / (kappa_base + 1e-10)
            kappa_eff_actual_frac = geom_new['kappa_eff'] / (kappa_eff_base + 1e-10)

            log(f"    kappa={geom_new['kappa']:.4f} (target={a:.1f}x, actual={kappa_actual_frac:.3f}x)")
            log(f"    kappa_eff={geom_new['kappa_eff']:.4f} (target={b:.1f}x, actual={kappa_eff_actual_frac:.3f}x)")
            log(f"    q={q:.4f}, acc={acc:.4f}")

            records.append({
                'a_target': a, 'b_target': b,
                's_signal': float(s_signal), 's_null': float(s_null),
                'kappa': float(geom_new['kappa']),
                'kappa_eff': float(geom_new['kappa_eff']),
                'd_eff': float(geom_new['d_eff']),
                'q_norm': float(q),
                'acc': float(acc),
                'logit_q': float(safe_logit(q)),
                'kappa_actual_frac': float(kappa_actual_frac),
                'kappa_eff_actual_frac': float(kappa_eff_actual_frac),
            })

    if len(records) < 4:
        log(f"  SKIP: too few feasible points ({len(records)})")
        return None

    log(f"\n  N feasible points: {len(records)}")

    kappas = np.array([r['kappa'] for r in records])
    kappa_effs = np.array([r['kappa_eff'] for r in records])
    logit_qs = np.array([r['logit_q'] for r in records])
    qs = np.array([r['q_norm'] for r in records])

    # Fit univariate and bivariate models
    def ols_r2(X_design, y):
        reg = LinearRegression().fit(X_design, y)
        resid = y - reg.predict(X_design)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        return r2, reg.coef_, float(reg.intercept_)

    r2_kappa, coef_k, c_k = ols_r2(kappas.reshape(-1, 1), logit_qs)
    r2_kappa_eff, coef_ke, c_ke = ols_r2(kappa_effs.reshape(-1, 1), logit_qs)
    r2_bivariate, coef_biv, c_biv = ols_r2(np.column_stack([kappas, kappa_effs]), logit_qs)

    log(f"\n  Univariate kappa:     R2={r2_kappa:.4f}, alpha={coef_k[0]:.3f}")
    log(f"  Univariate kappa_eff: R2={r2_kappa_eff:.4f}, alpha={coef_ke[0]:.3f}")
    log(f"  Bivariate:            R2={r2_bivariate:.4f}, alpha1={coef_biv[0]:.3f}, alpha2={coef_biv[1]:.3f}")
    log(f"  R2 improvement: {r2_bivariate - max(r2_kappa, r2_kappa_eff):.4f}")

    # Pre-registered evaluations
    # H1: baseline check (a=1, b=1)
    baseline_rec = next((r for r in records if r['a_target'] == 1.0 and r['b_target'] == 1.0), None)
    h1_pass = False
    if baseline_rec:
        h1_kappa_err = abs(baseline_rec['kappa_actual_frac'] - 1.0)
        h1_kappa_eff_err = abs(baseline_rec['kappa_eff_actual_frac'] - 1.0)
        h1_pass = h1_kappa_err < 0.01 and h1_kappa_eff_err < 0.01
        log(f"\n  H1 (baseline check): kappa_err={h1_kappa_err:.4f}, kappa_eff_err={h1_kappa_eff_err:.4f}: {'PASS' if h1_pass else 'FAIL'}")

    # H2: kappa_eff invariance in null-only row (b=1.0, vary a)
    null_row = [r for r in records if r['b_target'] == 1.0]
    h2_pass = False
    if len(null_row) >= 2:
        ke_fracs = [abs(r['kappa_eff_actual_frac'] - 1.0) for r in null_row]
        h2_pass = max(ke_fracs) < 0.02
        log(f"  H2 (kappa_eff inv, b=1 row): max_err={max(ke_fracs):.4f}: {'PASS' if h2_pass else 'FAIL'}")

    # H3: kappa invariance in signal-only column (a=1.0, vary b)
    sig_col = [r for r in records if r['a_target'] == 1.0]
    h3_pass = False
    if len(sig_col) >= 2:
        k_fracs = [abs(r['kappa_actual_frac'] - 1.0) for r in sig_col]
        h3_pass = max(k_fracs) < 0.02
        log(f"  H3 (kappa inv, a=1 col): max_err={max(k_fracs):.4f}: {'PASS' if h3_pass else 'FAIL'}")

    # H4: bivariate R2 improvement
    r2_best_uni = max(r2_kappa, r2_kappa_eff)
    r2_delta = r2_bivariate - r2_best_uni
    h4_pass = r2_delta > 0.05
    log(f"  H4 (bivariate improvement): delta_R2={r2_delta:.4f}: {'PASS' if h4_pass else 'FAIL'} (threshold 0.05)")

    # H5: sign check
    h5_pass = float(coef_biv[0]) > 0 and float(coef_biv[1]) > 0
    log(f"  H5 (sign check): alpha1={coef_biv[0]:.3f}, alpha2={coef_biv[1]:.3f}: {'PASS' if h5_pass else 'FAIL'}")

    return {
        'arch': arch_name,
        'n_points': len(records),
        'baseline_kappa': float(kappa_base),
        'baseline_kappa_eff': float(kappa_eff_base),
        'baseline_trW_sig': float(geo['trW_sig']),
        'baseline_trW_null': float(geo['trW_null']),
        'records': records,
        'r2_kappa': float(r2_kappa),
        'r2_kappa_eff': float(r2_kappa_eff),
        'r2_bivariate': float(r2_bivariate),
        'r2_delta': float(r2_delta),
        'alpha1_bivariate': float(coef_biv[0]),
        'alpha2_bivariate': float(coef_biv[1]),
        'alpha_kappa_univariate': float(coef_k[0]),
        'alpha_kappa_eff_univariate': float(coef_ke[0]),
        'h1_pass': h1_pass,
        'h2_pass': h2_pass,
        'h3_pass': h3_pass,
        'h4_pass': h4_pass,
        'h5_pass': h5_pass,
    }


def main():
    log("=" * 70)
    log("2D CAUSAL SURFACE: BIVARIATE kappa x kappa_eff LAW")
    log("=" * 70)
    log(f"Pre-reg commit: {PREREGISTERED_COMMIT}")
    log(f"Grid: kappa_fracs={KAPPA_FRACS}, kappa_eff_fracs={KAPPA_EFF_FRACS}")
    log("")

    arch_results = []

    for arch_info in ARCHITECTURES:
        emb_path = os.path.join(RESULTS_DIR, arch_info["file"])
        if not os.path.exists(emb_path):
            log(f"\nSKIP {arch_info['name']}: file not found")
            continue

        log(f"\nLoading {arch_info['name']} ...")
        data = np.load(emb_path)
        X_full = np.where(np.isfinite(data['X']), data['X'], 0.0).astype(np.float32)
        y_full = data['y'].astype(np.int32)
        log(f"  Shape: {X_full.shape}, K={len(np.unique(y_full))}")

        result = run_arch(arch_info["name"], X_full, y_full)
        if result is not None:
            arch_results.append(result)

    # Cross-arch transfer (H6): fit on one, predict on other
    log(f"\n{'='*70}")
    log("CROSS-ARCH TRANSFER (H6)")
    log("=" * 70)

    h6_results = []
    for i, r_fit in enumerate(arch_results):
        for j, r_pred in enumerate(arch_results):
            if i == j:
                continue
            kappas_fit = np.array([rec['kappa'] for rec in r_fit['records']])
            ke_fit = np.array([rec['kappa_eff'] for rec in r_fit['records']])
            lq_fit = np.array([rec['logit_q'] for rec in r_fit['records']])
            kappas_pred = np.array([rec['kappa'] for rec in r_pred['records']])
            ke_pred = np.array([rec['kappa_eff'] for rec in r_pred['records']])
            lq_pred = np.array([rec['logit_q'] for rec in r_pred['records']])

            reg = LinearRegression().fit(np.column_stack([kappas_fit, ke_fit]), lq_fit)
            pred = reg.predict(np.column_stack([kappas_pred, ke_pred]))
            if np.var(lq_pred) > 1e-6:
                r_pearson = float(np.corrcoef(pred, lq_pred)[0, 1])
            else:
                r_pearson = float('nan')

            h6_pass = r_pearson >= 0.90
            log(f"  Fit={r_fit['arch']} -> Pred={r_pred['arch']}: r={r_pearson:.3f} {'PASS' if h6_pass else 'FAIL'} (threshold 0.90)")
            h6_results.append({'fit_arch': r_fit['arch'], 'pred_arch': r_pred['arch'],
                                'r_transfer': r_pearson, 'h6_pass': h6_pass})

    # Overall evaluation
    log(f"\n{'='*70}")
    log("PRE-REGISTERED EVALUATION")
    log("=" * 70)

    h4_archs = sum(r['h4_pass'] for r in arch_results)
    h5_archs = sum(r['h5_pass'] for r in arch_results)
    h6_pass_any = any(r['h6_pass'] for r in h6_results) if h6_results else False

    log(f"\nH4 (bivariate improvement): {h4_archs}/{len(arch_results)}")
    log(f"H5 (sign correct):          {h5_archs}/{len(arch_results)}")
    log(f"H6 (cross-arch transfer):   {'PASS' if h6_pass_any else 'FAIL'}")

    overall_pass = h4_archs >= 1 and h5_archs >= 1 and h6_pass_any
    log(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    if overall_pass:
        log("  The bivariate law logit(q) = alpha1*kappa + alpha2*kappa_eff + C")
        log("  is validated: both components independently predict q,")
        log("  and the bivariate coefficients transfer across architectures.")
    else:
        log("  Bivariate model does not significantly outperform univariate,")
        log("  OR cross-architecture transfer fails.")

    output = {
        'experiment': '2d_causal_surface_bivariate_law',
        'preregistered_commit': PREREGISTERED_COMMIT,
        'kappa_fracs': KAPPA_FRACS,
        'kappa_eff_fracs': KAPPA_EFF_FRACS,
        'arch_results': arch_results,
        'h6_transfer': h6_results,
        'evaluation': {
            'h4_archs': h4_archs,
            'h5_archs': h5_archs,
            'h6_pass_any': h6_pass_any,
            'overall_pass': overall_pass,
        }
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
