"""
Fast Gaussian Synthetic d_eff Surgery Test
==========================================
DEFINITIVE test: Does d_eff surgery work on Gaussian data?

Uses FAST analytical q estimation via Monte Carlo on small samples.
Also uses a clever centroid setup to guarantee exact kappa.

KEY INSIGHT: If this PASSES, neural network surgery fails due to non-Gaussianity.
If FAILS, d_eff is genuinely not causal.
"""

import numpy as np
import json
from scipy.stats import norm

np.random.seed(42)

K = 20
D = 128                   # Fixed dimension
N_PER_CLASS = 100         # Small N for speed
N_CONFIGS = 20            # Random configs
A_RENORM = 1.0535

SURGERY_LEVELS = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
KAPPA_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5]

OUT_JSON = "results/cti_gaussian_surgery_test.json"


def compute_q_fast(X, y, K_val):
    """Fast 1-NN accuracy using centroid distances (approximate but fast)."""
    classes = sorted(np.unique(y).tolist())
    centroids = np.stack([X[y == c].mean(0) for c in classes])
    correct = 0
    total = 0
    for i, c in enumerate(classes):
        Xc = X[y == c]
        # For each sample in class c, find nearest centroid
        dists = np.linalg.norm(Xc[:, None, :] - centroids[None, :, :], axis=2)  # (N_c, K)
        pred = np.argmin(dists, axis=1)
        correct += np.sum(pred == i)
        total += len(Xc)
    acc = correct / total
    return (acc - 1.0/K_val) / (1.0 - 1.0/K_val)


def run_one(kappa_target, config_idx):
    """Run for one kappa target."""
    sigma_W_global = 1.0 / np.sqrt(D)  # Per-dim std

    # Place class 0 at 0, class 1 at delta*e1 (so we control kappa exactly)
    delta = kappa_target * sigma_W_global * np.sqrt(D)
    centroids = np.zeros((K, D))
    centroids[1, 0] = delta
    # Classes 2..K-1: random far-away (factor 5x farther)
    np.random.seed(config_idx * 100)
    centroids[2:, :] = np.random.randn(K-2, D) * sigma_W_global * np.sqrt(D) * 5

    # Sample data
    X = np.vstack([np.random.randn(N_PER_CLASS, D) * sigma_W_global + centroids[c]
                   for c in range(K)])
    y = np.repeat(np.arange(K), N_PER_CLASS)

    # Compute geometry
    trW = D * sigma_W_global**2  # = 1.0 for our setup
    Delta_hat = np.zeros(D)
    Delta_hat[0] = 1.0  # e1 direction

    sigma_centroid_sq = 0.0
    for c in range(K):
        Xc = X[y == c]
        z = Xc - centroids[c]
        proj = z @ Delta_hat
        sigma_centroid_sq += np.mean(proj**2) / K

    d_eff_base = trW / (sigma_centroid_sq + 1e-10)

    # Verify kappa
    kappa_actual = delta / (sigma_W_global * np.sqrt(D))

    q_base = compute_q_fast(X, y, K)
    q_base_safe = float(np.clip(q_base, 1e-6, 1 - 1e-6))
    logit_q_base = float(np.log(q_base_safe / (1 - q_base_safe)))

    kappa_eff_base = kappa_actual * np.sqrt(d_eff_base)
    C_fitted = logit_q_base - A_RENORM * kappa_eff_base

    results = []
    for r in SURGERY_LEVELS:
        # Surgery: scale_along = 1/sqrt(r), scale_perp ensures trW preserved
        denom = trW - sigma_centroid_sq
        if denom < 1e-10:
            continue
        scale_perp_sq = (trW - sigma_centroid_sq / r) / denom
        if scale_perp_sq < 0:
            continue
        scale_along = 1.0 / np.sqrt(r)
        scale_perp = np.sqrt(scale_perp_sq)

        X_new = np.zeros_like(X)
        for c in range(K):
            mask = y == c
            z = X[mask] - centroids[c]
            z_along = (z @ Delta_hat).reshape(-1, 1) * Delta_hat.reshape(1, -1)
            z_perp = z - z_along
            X_new[mask] = centroids[c] + scale_along * z_along + scale_perp * z_perp

        # Verify trW preserved
        trW_new = sum(np.sum((X_new[y==c] - centroids[c])**2) for c in range(K)) / (K * N_PER_CLASS)
        # Check kappa preserved
        kappa_new = delta / (np.sqrt(trW_new / D) * np.sqrt(D))

        q_new = compute_q_fast(X_new, y, K)
        q_new_safe = float(np.clip(q_new, 1e-6, 1 - 1e-6))
        logit_q_new = float(np.log(q_new_safe / (1 - q_new_safe)))

        # Predicted
        logit_q_pred = C_fitted + A_RENORM * kappa_actual * np.sqrt(r * d_eff_base)

        delta_actual = logit_q_new - logit_q_base
        delta_pred = logit_q_pred - logit_q_base
        if abs(delta_pred) > 1e-6:
            calib = abs(delta_actual - delta_pred) / abs(delta_pred)
        else:
            calib = 0.0

        results.append({
            'kappa_target': kappa_target,
            'kappa_actual': float(kappa_actual),
            'config_idx': config_idx,
            'r': float(r),
            'd_eff_base': float(d_eff_base),
            'd_eff_new': float(r * d_eff_base),
            'kappa_eff_base': float(kappa_eff_base),
            'q_base': float(q_base),
            'logit_q_base': float(logit_q_base),
            'q_new': float(q_new),
            'logit_q_new': float(logit_q_new),
            'logit_q_pred': float(logit_q_pred),
            'delta_actual': float(delta_actual),
            'delta_pred': float(delta_pred),
            'calib': float(calib),
            'kappa_change_pct': abs(kappa_new - kappa_actual) / kappa_actual * 100,
            'C_fitted': float(C_fitted),
        })

    return results


def main():
    print("=" * 70)
    print("GAUSSIAN SYNTHETIC d_eff SURGERY TEST (FAST)")
    print("=" * 70)
    print(f"K={K}, D={D}, N_PER_CLASS={N_PER_CLASS}, N_CONFIGS={N_CONFIGS}")
    print(f"A_RENORM={A_RENORM}, KAPPA_VALUES={KAPPA_VALUES}")
    print()

    all_records = []
    total = len(KAPPA_VALUES) * N_CONFIGS
    done = 0
    for kappa_target in KAPPA_VALUES:
        for config_idx in range(N_CONFIGS):
            recs = run_one(kappa_target, config_idx)
            all_records.extend(recs)
            done += 1
            if done % 5 == 0:
                print(f"  [{done}/{total}] kappa={kappa_target} config={config_idx}")

    print(f"\nTotal records: {len(all_records)}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    actual_logits = [r['logit_q_new'] for r in all_records]
    pred_logits = [r['logit_q_pred'] for r in all_records]
    delta_actuals = [r['delta_actual'] for r in all_records if abs(r['delta_pred']) > 0.01]
    delta_preds = [r['delta_pred'] for r in all_records if abs(r['delta_pred']) > 0.01]

    r_pearson = float(np.corrcoef(actual_logits, pred_logits)[0, 1]) if len(actual_logits) > 1 else float('nan')
    calibs = [r['calib'] for r in all_records if abs(r['delta_pred']) > 0.01]
    mean_calib = float(np.mean(calibs)) if calibs else float('nan')
    max_kappa_chg = float(np.max([r['kappa_change_pct'] for r in all_records]))

    print(f"\nOverall Pearson r(actual, pred logit_q): {r_pearson:.4f}  "
          f"[PASS>0.99: {'PASS' if r_pearson > 0.99 else 'FAIL'}]")
    print(f"Mean calib error: {mean_calib:.4f}  "
          f"[PASS<0.10: {'PASS' if mean_calib < 0.10 else 'FAIL'}]")
    print(f"Max kappa change: {max_kappa_chg:.6f}%  "
          f"[PASS<0.1: {'PASS' if max_kappa_chg < 0.1 else 'FAIL'}]")

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
            dr = float(np.mean([abs(a)/(abs(p)+1e-10) for a, p in zip(da, dp)]))
        else:
            cb, dr = float('nan'), float('nan')
        print(f"  {kappa_target:>8.2f} {len(recs_k):>6} {rp:>10.4f} {cb:>10.4f} {dr:>12.4f}")

    print(f"\nSample predictions (kappa=1.0, config=0):")
    sample = [r for r in all_records if abs(r['kappa_target'] - 1.0) < 0.01 and r['config_idx'] == 0]
    for r in sample:
        print(f"  r={r['r']:.2f}: logit_act={r['logit_q_new']:.4f}, logit_pred={r['logit_q_pred']:.4f}, "
              f"delta_act={r['delta_actual']:+.4f}, delta_pred={r['delta_pred']:+.4f}, calib={r['calib']:.4f}")

    primary_pass = r_pearson > 0.99
    secondary_pass = mean_calib < 0.10

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"PRIMARY (r>0.99):   {'PASS' if primary_pass else 'FAIL'} (r={r_pearson:.4f})")
    print(f"SECONDARY (calib<10%): {'PASS' if secondary_pass else 'FAIL'} (calib={mean_calib:.4f})")
    if primary_pass and secondary_pass:
        print("\n>>> PASS: Law valid in Gaussian regime.")
        print(">>> Neural network surgery failure = non-Gaussian embeddings.")
    elif r_pearson > 0.90:
        print("\n>>> PARTIAL: Shape correct but calibration off.")
        print(">>> Law IS directionally correct but A_RENORM needs recalibration.")
    else:
        print("\n>>> FAIL: Law fundamentally wrong even in Gaussian regime.")
        print(">>> d_eff NOT causal. Theory needs complete revision.")

    out = {
        "experiment": "gaussian_synthetic_deff_surgery_fast",
        "description": "d_eff surgery on purely Gaussian data (definitive theory test)",
        "A_RENORM": A_RENORM, "K": K, "D": D, "N_PER_CLASS": N_PER_CLASS,
        "N_CONFIGS": N_CONFIGS, "KAPPA_VALUES": KAPPA_VALUES, "SURGERY_LEVELS": SURGERY_LEVELS,
        "summary": {
            "pearson_r": r_pearson, "mean_calib": mean_calib,
            "max_kappa_change": max_kappa_chg,
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
