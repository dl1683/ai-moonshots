"""
Direct measurement of N_eff (effective competitor count) from existing cache data.

Theory: logit(q) = alpha*kappa - beta*log(K-1) + C
=> N_eff_obs = exp(beta*log(K-1)) = (K-1)^beta
=> if beta=0.5 (sparse), N_eff = sqrt(K-1)

This script:
1. Loads all 444 valid points (same filtering as comprehensive universality)
2. Fits alpha and C using kappa only (collapsing K-variation)
3. Computes residuals = logit(q) - alpha*kappa - C
4. Estimates N_eff_obs = exp(residuals) per point
5. Tests: log(N_eff_obs) ~ beta * log(K-1) -> expect beta ~ 0.5

Pre-registration: beta_neff in [0.35, 0.65] counts as confirming sparse competition.

Output: results/cti_neff_direct_measurement.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, linregress

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CACHE_DIR = RESULTS_DIR
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_neff_direct_measurement.json")

# Pre-registered bounds
BETA_NEFF_LOW  = 0.35
BETA_NEFF_HIGH = 0.65
PEARSON_P_THRESH = 0.001  # stricter since n=444


def load_all_points():
    """Load all valid cache points (same filter as comprehensive universality)."""
    points = []
    for fname in os.listdir(CACHE_DIR):
        if not (fname.startswith("kappa_near_cache_") and fname.endswith(".json")):
            continue
        fpath = os.path.join(CACHE_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for entry in data:
            q = entry.get("q")
            kappa = entry.get("kappa_nearest")
            K = entry.get("K")
            if q is None or kappa is None or K is None:
                continue
            if q <= 0 or q >= 1.0:
                continue
            if kappa <= 0:
                continue
            q_norm = (q - 1.0 / K) / (1.0 - 1.0 / K)
            if q_norm <= 0 or q_norm >= 1.0:
                continue
            logit_q = float(np.log(q_norm / (1 - q_norm)))
            points.append({
                "model": entry.get("model", ""),
                "dataset": entry.get("dataset", ""),
                "K": int(K),
                "q": float(q),
                "q_norm": float(q_norm),
                "kappa": float(kappa),
                "logit_q": logit_q,
                "log_km1": float(np.log(K - 1)),
            })
    return points


def main():
    print("Loading cache points...")
    pts = load_all_points()
    n = len(pts)
    print(f"Loaded {n} valid points")

    kappas  = np.array([p["kappa"] for p in pts])
    logit_q = np.array([p["logit_q"] for p in pts])
    log_km1 = np.array([p["log_km1"] for p in pts])
    Ks      = np.array([p["K"] for p in pts])

    # Step 1: fit kappa-only model (alpha, C, no K term)
    # logit(q) = alpha*kappa + C
    X_kappa = np.column_stack([kappas, np.ones(n)])
    coeffs, _, _, _ = np.linalg.lstsq(X_kappa, logit_q, rcond=None)
    alpha_kappa, C_kappa = coeffs[0], coeffs[1]
    print(f"Kappa-only fit: alpha={alpha_kappa:.4f}, C={C_kappa:.4f}")

    # Step 2: compute residuals
    residuals = logit_q - alpha_kappa * kappas - C_kappa
    # N_eff_obs = exp(-(residuals + beta_theory*log(K-1)))...
    # Actually residuals = -beta*log(K-1) + noise
    # So log(N_eff) = -residuals = beta*log(K-1)
    # N_eff = exp(-residuals)
    # But we want to be careful: residuals ~ -beta*log(K-1) + noise
    # => -residuals ~ beta*log(K-1)

    # Step 3: regress -residuals vs log(K-1)
    neg_resid = -residuals

    slope, intercept, r_val, p_val, se = linregress(log_km1, neg_resid)
    r_pearson, p_pearson = pearsonr(log_km1, neg_resid)
    print(f"\nN_eff measurement:")
    print(f"  log(N_eff) ~ beta * log(K-1)")
    print(f"  slope (beta_neff) = {slope:.4f} +/- {se:.4f}")
    print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.2e}")
    print(f"  Pre-reg range: [{BETA_NEFF_LOW}, {BETA_NEFF_HIGH}]")
    pr_beta = BETA_NEFF_LOW <= slope <= BETA_NEFF_HIGH
    pr_p = p_pearson < PEARSON_P_THRESH
    print(f"  PR_BETA: {'PASS' if pr_beta else 'FAIL'}")
    print(f"  PR_P:    {'PASS' if pr_p else 'FAIL'}")

    # Step 4: per-K analysis (group by K)
    K_unique = sorted(set(Ks))
    print(f"\nPer-K mean N_eff and sqrt(K-1) comparison:")
    print(f"{'K':>6} {'sqrt(K-1)':>10} {'N_eff_obs':>10} {'ratio':>7} {'n_pts':>6}")
    per_k = []
    for K in K_unique:
        mask = Ks == K
        resid_k = residuals[mask]
        n_k = mask.sum()
        neff_obs = float(np.exp(np.mean(-resid_k)))
        neff_theory = float(np.sqrt(K - 1))
        ratio = neff_obs / neff_theory
        print(f"{K:>6} {neff_theory:>10.3f} {neff_obs:>10.3f} {ratio:>7.3f} {n_k:>6}")
        per_k.append({"K": int(K), "n_pts": int(n_k),
                      "neff_theory_sqrt": neff_theory,
                      "neff_obs": neff_obs,
                      "ratio_obs_theory": ratio})

    # Step 5: full model check (should recover alpha ~ 3.6, beta ~ 0.478)
    X_full = np.column_stack([kappas, log_km1, np.ones(n)])
    coeffs_full, _, _, _ = np.linalg.lstsq(X_full, logit_q, rcond=None)
    alpha_full, beta_full, C_full = coeffs_full
    pred_full = X_full @ coeffs_full
    ss_res = np.sum((logit_q - pred_full)**2)
    ss_tot = np.sum((logit_q - logit_q.mean())**2)
    R2_full = 1 - ss_res / ss_tot
    print(f"\nFull model check: alpha={alpha_full:.4f}, beta_neg={-beta_full:.4f}, "
          f"C={C_full:.4f}, R2={R2_full:.4f}")
    print(f"(beta_neg in paper notation = |beta| = {abs(beta_full):.4f}; expected ~0.478)")

    # Save
    result = {
        "experiment": "neff_direct_measurement",
        "n_points": n,
        "kappa_only_alpha": float(alpha_kappa),
        "kappa_only_C": float(C_kappa),
        "beta_neff_slope": float(slope),
        "beta_neff_se": float(se),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "pre_reg": {
            "PR_BETA": f"beta_neff in [{BETA_NEFF_LOW}, {BETA_NEFF_HIGH}]",
            "PR_P": f"p < {PEARSON_P_THRESH}",
        },
        "verdict": {
            "PR_BETA": bool(pr_beta),
            "PR_P": bool(pr_p),
        },
        "full_model_alpha": float(alpha_full),
        "full_model_beta_neg": float(-beta_full),
        "full_model_R2": float(R2_full),
        "per_K": per_k,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
