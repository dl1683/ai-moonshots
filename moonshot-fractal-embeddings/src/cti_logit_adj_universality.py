#!/usr/bin/env python -u
"""
UNIVERSALITY VIA ADJUSTED LOGIT: logit_adj = logit(q) + log(K-1)

Theorem 1 predicts: logit(q) = A * kappa - log(K-1) + C
=> logit_adj = logit(q) + log(K-1) = A * kappa + C

This "adjusted logit" should be UNIVERSALLY linear in kappa,
with SAME A and C regardless of K (number of classes).

This is theoretically better than kappa/sqrt(K) because:
- log(K-1) term comes directly from the Gumbel Race mechanism
- sqrt(K) is empirical phenomenology, not theoretically derived

TEST: Using existing cross-dataset data (observable_order_parameter.json
      plus spectral_collapse data), verify that:
      1. logit_adj ~ kappa is universal (same slope across K values)
      2. Residuals of logit_adj on kappa are uncorrelated with K
      3. kappa/sqrt(K) still has K-residual correlation

ALSO: Use the within-dataset K-variation data to directly compare.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def safe_logit(q, eps=1e-4):
    q = float(np.clip(q, eps, 1 - eps))
    return float(np.log(q / (1 - q)))


def main():
    print("=" * 70)
    print("UNIVERSALITY: logit_adj = logit(q) + log(K-1) vs kappa/sqrt(K)")
    print("=" * 70)

    # ============================================================
    # TEST 1: Observable Order-Parameter synthetic data
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 1: Observable Order-Parameter synthetic data (K = 5..100)")
    print(f"{'='*70}")

    oop_path = RESULTS_DIR / "cti_observable_order_parameter.json"
    with open(oop_path) as f:
        oop = json.load(f)

    data_rows = oop.get("data", [])
    print(f"  Loaded {len(data_rows)} rows")

    if data_rows:
        q_vals = np.array([r["q"] for r in data_rows])
        kappa_vals = np.array([r["kappa_spec"] for r in data_rows])
        K_vals = np.array([r["K"] for r in data_rows])
        dr_vals = np.array([r["dist_ratio"] for r in data_rows])

        # Clip q to valid range
        q_c = np.clip(q_vals, 1e-4, 1 - 1e-4)
        logit_q = np.log(q_c / (1 - q_c))
        logit_adj = logit_q + np.log(np.maximum(K_vals - 1, 1))
        kappa_sqrt_K = kappa_vals / np.sqrt(K_vals)

        # Fit 1: logit_adj ~ kappa (should be linear, universal)
        reg_adj = LinearRegression().fit(kappa_vals.reshape(-1, 1), logit_adj)
        logit_adj_pred = reg_adj.predict(kappa_vals.reshape(-1, 1))
        ss_res = np.sum((logit_adj - logit_adj_pred) ** 2)
        ss_tot = np.sum((logit_adj - logit_adj.mean()) ** 2)
        r2_adj = 1 - ss_res / ss_tot
        residuals_adj = logit_adj - logit_adj_pred

        # Fit 2: logit_q ~ kappa/sqrt(K) (phenomenological)
        reg_sqk = LinearRegression().fit(kappa_sqrt_K.reshape(-1, 1), logit_q)
        logit_q_pred = reg_sqk.predict(kappa_sqrt_K.reshape(-1, 1))
        ss_res2 = np.sum((logit_q - logit_q_pred) ** 2)
        ss_tot2 = np.sum((logit_q - logit_q.mean()) ** 2)
        r2_sqk = 1 - ss_res2 / ss_tot2
        residuals_sqk = logit_q - logit_q_pred

        # Check: does K correlate with residuals?
        rho_resid_adj = float(spearmanr(K_vals, residuals_adj).statistic)
        rho_resid_sqk = float(spearmanr(K_vals, residuals_sqk).statistic)

        print(f"\n  logit_adj = logit(q) + log(K-1)  ~  kappa:")
        print(f"    R2 = {r2_adj:.4f}")
        print(f"    Slope = {reg_adj.coef_[0]:.4f}, Intercept = {reg_adj.intercept_:.4f}")
        print(f"    rho(residual, K) = {rho_resid_adj:.4f}  [lower = better universality]")

        print(f"\n  logit(q)  ~  kappa/sqrt(K):")
        print(f"    R2 = {r2_sqk:.4f}")
        print(f"    Slope = {reg_sqk.coef_[0]:.4f}, Intercept = {reg_sqk.intercept_:.4f}")
        print(f"    rho(residual, K) = {rho_resid_sqk:.4f}  [lower = better universality]")

        # Verdict
        print(f"\n  VERDICT:")
        if abs(rho_resid_adj) < abs(rho_resid_sqk):
            print(f"    logit_adj has LESS K-residual correlation ({rho_resid_adj:.4f} < {rho_resid_sqk:.4f})")
            print(f"    -> logit_adj normalization is MORE universal (theory wins)")
        else:
            print(f"    kappa/sqrt(K) has LESS K-residual correlation ({rho_resid_sqk:.4f} < {rho_resid_adj:.4f})")
            print(f"    -> kappa/sqrt(K) is MORE universal (empirical beats theory)")

    # ============================================================
    # TEST 2: Within-dataset K variation
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 2: Within-dataset K variation (cti_within_dataset_K.json)")
    print(f"{'='*70}")

    k_path = RESULTS_DIR / "cti_within_dataset_K.json"
    with open(k_path) as f:
        k_data = json.load(f)

    # The existing M_log model: logit(q) = a*logK + b*kappa + c
    m_log = k_data.get("global_models", {}).get("M_log", {})
    m_sqrt = k_data.get("global_models", {}).get("M_sqrt", {})
    print(f"\n  From cti_within_dataset_K.json:")
    print(f"    M_log (a*log(K) + b*kappa + c): R2 = {m_log.get('r2', 'N/A'):.4f}")
    print(f"    M_sqrt (a*kappa/sqrt(K) + b):   R2 = {m_sqrt.get('r2', 'N/A'):.4f}")
    print(f"\n  NOTE: If M_log params are [logK_coef, kappa_coef, intercept]:")
    if "params" in m_log:
        p = m_log["params"]
        print(f"    a={p[0]:.4f} (logK), b={p[1]:.4f} (kappa), c={p[2]:.4f} (intercept)")
        # Compare to Theorem 1: logit(q) = A*kappa - log(K-1) + C
        # If a ~ -1.0, this is consistent with Theorem 1
        print(f"    logK coefficient = {p[0]:.4f} (Theorem 1 predicts ~ -1.0)")

    # ============================================================
    # TEST 3: Multi-dataset spectral collapse data
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 3: Multi-dataset spectral collapse (cti_spectral_collapse.json)")
    print(f"{'='*70}")

    sc_path = RESULTS_DIR / "cti_spectral_collapse.json"
    if sc_path.exists():
        with open(sc_path) as f:
            sc = json.load(f)
        print(f"  Keys: {list(sc.keys())[:10]}")
        # Try to find dataset-level data
        if "per_dataset" in sc:
            for ds_name, ds_data in sc["per_dataset"].items():
                print(f"  Dataset {ds_name}: {ds_data}")
        elif "data" in sc:
            rows = sc["data"]
            if rows and isinstance(rows[0], dict) and "K" in rows[0]:
                q_vals = np.array([r.get("q", float("nan")) for r in rows])
                kappa_vals = np.array([r.get("kappa", float("nan")) for r in rows])
                K_vals = np.array([r.get("K", float("nan")) for r in rows])
                valid = np.isfinite(q_vals) & np.isfinite(kappa_vals) & np.isfinite(K_vals) & (q_vals > 0) & (q_vals < 1)
                print(f"  {valid.sum()} valid rows with K in {set(K_vals[valid].astype(int))}")

    # ============================================================
    # TEST 4: Use per_dataset_logK data
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 4: Per-dataset log(K) analysis")
    print(f"{'='*70}")

    logk_path = RESULTS_DIR / "cti_per_dataset_logK.json"
    if logk_path.exists():
        with open(logk_path) as f:
            logk = json.load(f)
        print(f"  Keys: {list(logk.keys())[:20]}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY: Universality Comparison")
    print(f"{'='*70}")
    print("""
  Theorem 1 prediction: logit(q) = A*kappa - log(K-1) + C
  => adjusted logit: logit_adj = logit(q) + log(K-1) = A*kappa + C

  If this is correct, then:
  - Slope A should be CONSTANT across K values
  - R2(logit_adj ~ kappa) should be HIGH (>0.9)
  - rho(residual(logit_adj, kappa), K) should be near 0

  vs. phenomenological: kappa/sqrt(K) (no theoretical derivation for sqrt(K))
  - sqrt(K) motivated by practical fitting, not theory
  - Theory says log(K-1) not sqrt(K)

  IMPLICATION: If logit_adj works better, the Gumbel Race Law is
  THEORETICALLY motivated, not just phenomenological. This is critical
  for the Nobel claim — the law is DERIVED, not curve-fitted.
""")

    # Save results
    results = {
        "test1_oop_synthetic": {
            "r2_logit_adj": float(r2_adj) if data_rows else None,
            "r2_kappa_sqrtK": float(r2_sqk) if data_rows else None,
            "rho_resid_K_adj": float(rho_resid_adj) if data_rows else None,
            "rho_resid_K_sqk": float(rho_resid_sqk) if data_rows else None,
        },
        "test2_within_dataset_K": {
            "M_log_r2": m_log.get("r2"),
            "M_sqrt_r2": m_sqrt.get("r2"),
            "logK_coef_theorem1_expects_-1": m_log.get("params", [None])[0],
        },
    }
    out_path = RESULTS_DIR / "cti_logit_adj_universality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
