"""
Gaussian Synthetic d_eff Surgery Test
======================================
DEFINITIVE test: Does d_eff surgery work on PURELY GAUSSIAN data?

If YES: law is mathematically valid. Neural network surgery fails because
        real embeddings deviate from Gaussian (non-Gaussian structure).
If NO:  law is fundamentally wrong. d_eff is NOT causal even in theory.

DESIGN:
  1. Sample K Gaussian classes with known centroids, covariance
  2. Set kappa_nearest, d_eff to specific values (control)
  3. Apply d_eff surgery (change sigma_centroid_dir, keep kappa fixed)
  4. Measure actual q_new vs predicted q_new
  5. Compare Pearson r and calibration error

PRE-REGISTERED:
  - Pearson r(actual, predicted) > 0.99 ACROSS r levels
  - Mean calibration error < 10%
  - kappa_nearest change < 0.1%

If PASS: law is valid in Gaussian regime (neural net failure is non-Gaussianity)
If FAIL: law is fundamentally wrong (d_eff not causal even in theory)
"""

import numpy as np
import json
from scipy import stats

np.random.seed(42)

# ==================== CONFIGURATION ====================
K = 20
D_VALUES = [64, 128]         # Test two dimensions (fast)
KAPPA_VALUES = [0.5, 1.0]    # Two key kappa regimes
N_SAMPLES_PER_CLASS = 200    # N per class for q estimation
N_BOOTSTRAP = 5              # Repetitions for stability
A_RENORM = 1.0535

SURGERY_LEVELS = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

OUT_JSON = "results/cti_gaussian_surgery_test.json"


def compute_kappa(centroids, sigma_W_global, d):
    K = len(centroids)
    min_dist = float('inf')
    min_i, min_j = 0, 1
    for i in range(K):
        for j in range(i+1, K):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j
    return min_dist / (sigma_W_global * np.sqrt(d)), min_i, min_j


def compute_q(X, y, K_val):
    """Compute normalized 1-NN accuracy q."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    acc = knn.score(X, y)
    return (acc - 1.0/K_val) / (1.0 - 1.0/K_val)


def apply_deff_surgery(X, y, centroids, ci, cj, r, sigma_centroid_sq, trW):
    """
    Apply d_eff surgery: change d_eff by factor r while keeping kappa and trW fixed.
    Works on the nearest centroid pair (ci, cj).
    """
    Delta = centroids[ci] - centroids[cj]
    Delta_hat = Delta / np.linalg.norm(Delta)

    # Target: sigma_cdir_new^2 = sigma_centroid_sq / r (so d_eff_new = r * d_eff_base)
    # scale_along^2 = sigma_centroid_sq / (r * sigma_centroid_sq) = 1/r
    # => scale_along = 1/sqrt(r)
    # trW_new = scale_along^2 * sigma_cdir_sq + scale_perp^2 * (trW - sigma_cdir_sq) = trW
    # => scale_perp^2 = (trW - sigma_cdir_sq/r) / (trW - sigma_cdir_sq)
    scale_along = 1.0 / np.sqrt(r)
    denom = trW - sigma_centroid_sq
    if denom < 1e-10:
        return X.copy(), False  # Degenerate
    scale_perp_sq = (trW - sigma_centroid_sq / r) / denom
    if scale_perp_sq < 0:
        return X.copy(), False  # Invalid r
    scale_perp = np.sqrt(scale_perp_sq)

    X_new = np.zeros_like(X)
    for c in np.unique(y):
        mask = y == c
        Xc = X[mask]
        mu_c = centroids[c]
        z = Xc - mu_c
        z_along = (z @ Delta_hat).reshape(-1, 1) * Delta_hat.reshape(1, -1)
        z_perp = z - z_along
        X_new[mask] = mu_c + scale_along * z_along + scale_perp * z_perp

    return X_new, True


def run_one_config(d, kappa_target, rep):
    """Run surgery on synthetic Gaussian data for one (d, kappa, rep) config."""
    # Generate K Gaussian classes
    # Set centroids on a regular simplex (approximate ETF) scaled to get kappa_target
    # sigma_W_global^2 = 1/d (unit variance per dim)
    sigma_W_global = 1.0 / np.sqrt(d)  # scalar
    trW_per_class = 1.0  # tr(Sigma_W) = d * sigma_W_global^2 = 1

    # Simple centroid arrangement: place 2 classes at +/- delta/2 * e1, rest randomly
    # For simplicity: all centroids random on unit sphere scaled to kappa_target
    centroids_np = np.random.randn(K, d)
    # Normalize so nearest pair has kappa = kappa_target
    # Find nearest pair after normalization
    scale = 0.1  # initial scale
    centroids_np = centroids_np * scale

    # Iteratively adjust to get desired kappa
    for _ in range(50):
        kappa_val, mi, mj = compute_kappa(centroids_np, sigma_W_global, d)
        if abs(kappa_val - kappa_target) < 0.01:
            break
        centroids_np *= kappa_target / max(kappa_val, 1e-6)

    kappa_actual, mi, mj = compute_kappa(centroids_np, sigma_W_global, d)

    # Sample from Gaussian classes
    X_list, y_list = [], []
    for c in range(K):
        Xc = np.random.randn(N_SAMPLES_PER_CLASS, d) * sigma_W_global + centroids_np[c]
        X_list.append(Xc)
        y_list.extend([c] * N_SAMPLES_PER_CLASS)
    X = np.vstack(X_list)
    y = np.array(y_list)

    # Compute geometry
    sigma_centroid_sq = 0.0
    Delta = centroids_np[mi] - centroids_np[mj]
    Delta_hat = Delta / np.linalg.norm(Delta)
    for c in range(K):
        Xc = X[y == c]
        z = Xc - centroids_np[c]
        proj = z @ Delta_hat
        sigma_centroid_sq += np.mean(proj**2) / K

    trW = trW_per_class  # Known by construction (isotropic)
    d_eff_base = trW / (sigma_centroid_sq + 1e-10)

    # Baseline q
    q_base = compute_q(X, y, K)
    q_base_safe = float(np.clip(q_base, 1e-6, 1 - 1e-6))
    logit_q_base = float(np.log(q_base_safe / (1 - q_base_safe)))

    kappa_eff_base = kappa_actual * np.sqrt(d_eff_base)
    C_fitted = logit_q_base - A_RENORM * kappa_eff_base

    results = []
    for r in SURGERY_LEVELS:
        X_new, ok = apply_deff_surgery(X, y, centroids_np, mi, mj, r, sigma_centroid_sq, trW)
        if not ok:
            continue

        # Verify kappa unchanged
        kappa_new, _, _ = compute_kappa(X_new, sigma_W_global, d)
        kappa_change_pct = abs(kappa_new - kappa_actual) / (kappa_actual + 1e-10) * 100

        q_new = compute_q(X_new, y, K)
        q_new_safe = float(np.clip(q_new, 1e-6, 1 - 1e-6))
        logit_q_new = float(np.log(q_new_safe / (1 - q_new_safe)))

        # Predicted via law
        logit_q_pred = C_fitted + A_RENORM * kappa_actual * np.sqrt(r * d_eff_base)

        delta_actual = float(logit_q_new - logit_q_base)
        delta_pred = float(logit_q_pred - logit_q_base)
        if abs(delta_pred) > 1e-6:
            calib = abs(delta_actual - delta_pred) / abs(delta_pred)
        else:
            calib = 0.0

        results.append({
            'd': d, 'kappa_target': kappa_target, 'kappa_actual': float(kappa_actual),
            'rep': rep, 'r': float(r), 'd_eff_base': float(d_eff_base),
            'd_eff_new': float(r * d_eff_base), 'kappa_eff_base': float(kappa_eff_base),
            'q_base': float(q_base), 'logit_q_base': float(logit_q_base),
            'q_new': float(q_new), 'logit_q_new': float(logit_q_new),
            'logit_q_pred': float(logit_q_pred),
            'delta_actual': float(delta_actual), 'delta_pred': float(delta_pred),
            'calib': float(calib), 'kappa_change_pct': float(kappa_change_pct),
            'C_fitted': float(C_fitted),
        })

    return results


def main():
    print("=" * 70)
    print("GAUSSIAN SYNTHETIC d_eff SURGERY TEST")
    print("=" * 70)
    print("Definitive test: Does d_eff causally affect q in Gaussian regime?")
    print(f"A_RENORM = {A_RENORM}")
    print(f"K = {K}, D_VALUES = {D_VALUES}, KAPPA_VALUES = {KAPPA_VALUES}")
    print()

    all_records = []
    total = len(D_VALUES) * len(KAPPA_VALUES) * N_BOOTSTRAP
    done = 0
    for d in D_VALUES:
        for kappa_target in KAPPA_VALUES:
            for rep in range(N_BOOTSTRAP):
                records = run_one_config(d, kappa_target, rep)
                all_records.extend(records)
                done += 1
                if done % 5 == 0:
                    print(f"  [{done}/{total}] d={d} kappa={kappa_target} rep={rep}")

    print(f"\nTotal records: {len(all_records)}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Pearson r between actual and predicted logit_q
    actual_logits = [r['logit_q_new'] for r in all_records]
    pred_logits = [r['logit_q_pred'] for r in all_records]
    delta_actuals = [r['delta_actual'] for r in all_records if abs(r['delta_pred']) > 0.01]
    delta_preds = [r['delta_pred'] for r in all_records if abs(r['delta_pred']) > 0.01]
    calibs = [r['calib'] for r in all_records if abs(r['delta_pred']) > 0.01]
    kappa_changes = [r['kappa_change_pct'] for r in all_records]

    r_pearson = float(np.corrcoef(actual_logits, pred_logits)[0, 1]) if len(actual_logits) > 1 else float('nan')
    mean_calib = float(np.mean(calibs)) if calibs else float('nan')
    max_kappa_chg = float(np.max(kappa_changes))

    print(f"\nOverall (all d, kappa, r, rep):")
    print(f"  Pearson r(actual, predicted logit_q): {r_pearson:.4f}  [PASS>0.99: {'PASS' if r_pearson > 0.99 else 'FAIL'}]")
    print(f"  Mean calibration error: {mean_calib:.4f}  [PASS<0.10: {'PASS' if mean_calib < 0.10 else 'FAIL'}]")
    print(f"  Max kappa change: {max_kappa_chg:.6f}%  [PASS<0.1: {'PASS' if max_kappa_chg < 0.1 else 'FAIL'}]")

    # By kappa regime
    print(f"\nBy kappa_target:")
    print(f"  {'kappa':>8} {'n':>6} {'r_pearson':>10} {'calib':>10} {'delta_ratio':>12}")
    for kappa_target in KAPPA_VALUES:
        recs_k = [r for r in all_records if abs(r['kappa_target'] - kappa_target) < 0.01]
        if len(recs_k) < 2:
            continue
        al = [r['logit_q_new'] for r in recs_k]
        pl = [r['logit_q_pred'] for r in recs_k]
        rp = float(np.corrcoef(al, pl)[0, 1]) if len(al) > 1 else float('nan')
        da = [r['delta_actual'] for r in recs_k if abs(r['delta_pred']) > 0.01]
        dp = [r['delta_pred'] for r in recs_k if abs(r['delta_pred']) > 0.01]
        if da:
            cb = float(np.mean([abs(a-p)/abs(p) for a, p in zip(da, dp)]))
            dr = float(np.mean([abs(a)/abs(p) for a, p in zip(da, dp) if abs(p) > 0.01]))
        else:
            cb, dr = float('nan'), float('nan')
        print(f"  {kappa_target:>8.2f} {len(recs_k):>6} {rp:>10.4f} {cb:>10.4f} {dr:>12.4f}")

    # Sample predictions
    print(f"\nSample predictions (d=128, kappa=0.8, rep=0):")
    sample = [r for r in all_records if r['d'] == 128 and abs(r['kappa_target'] - 0.8) < 0.01 and r['rep'] == 0]
    for r in sample:
        print(f"  r={r['r']:.2f}: logit_actual={r['logit_q_new']:.4f}, "
              f"logit_pred={r['logit_q_pred']:.4f}, "
              f"delta_actual={r['delta_actual']:+.4f}, delta_pred={r['delta_pred']:+.4f}, "
              f"calib={r['calib']:.4f}")

    # VERDICT
    primary_pass = r_pearson > 0.99
    secondary_pass = mean_calib < 0.10
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"PRIMARY (r>0.99): {'PASS' if primary_pass else 'FAIL'} (r={r_pearson:.4f})")
    print(f"SECONDARY (calib<10%): {'PASS' if secondary_pass else 'FAIL'} (calib={mean_calib:.4f})")
    if primary_pass and secondary_pass:
        print("\n>>> PASS: Law valid in Gaussian regime. NN surgery failure = non-Gaussianity.")
    else:
        print("\n>>> FAIL: Law invalid even in Gaussian regime. Theory needs fundamental revision.")

    out = {
        "experiment": "gaussian_synthetic_deff_surgery",
        "description": "d_eff surgery on purely Gaussian data (definitive theory test)",
        "A_RENORM": A_RENORM, "K": K, "D_VALUES": D_VALUES, "KAPPA_VALUES": KAPPA_VALUES,
        "N_BOOTSTRAP": N_BOOTSTRAP, "SURGERY_LEVELS": SURGERY_LEVELS,
        "summary": {
            "pearson_r": r_pearson, "mean_calib": mean_calib, "max_kappa_change": max_kappa_chg,
            "primary_pass": bool(primary_pass), "secondary_pass": bool(secondary_pass),
            "n_records": len(all_records),
        },
        "by_kappa": {},
        "records": all_records,
    }

    for kappa_target in KAPPA_VALUES:
        recs_k = [r for r in all_records if abs(r['kappa_target'] - kappa_target) < 0.01]
        if len(recs_k) < 2:
            continue
        al = [r['logit_q_new'] for r in recs_k]
        pl = [r['logit_q_pred'] for r in recs_k]
        rp = float(np.corrcoef(al, pl)[0, 1]) if len(al) > 1 else float('nan')
        da = [r['delta_actual'] for r in recs_k if abs(r['delta_pred']) > 0.01]
        dp = [r['delta_pred'] for r in recs_k if abs(r['delta_pred']) > 0.01]
        cb = float(np.mean([abs(a-p)/abs(p) for a, p in zip(da, dp)])) if da else float('nan')
        out["by_kappa"][str(kappa_target)] = {"pearson_r": rp, "mean_calib": cb, "n": len(recs_k)}

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
