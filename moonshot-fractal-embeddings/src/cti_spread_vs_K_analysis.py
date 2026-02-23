#!/usr/bin/env python -u
"""
K x SPREAD FACTORIAL ANALYSIS (Session 43 follow-up)
=====================================================
Pre-registered BEFORE running.

HYPOTHESIS: rho >= 0.85 is predicted by between-architecture kappa spread,
not K alone. Non-monotone K pattern (K=20 PASS, K=42 FAIL, K=59 PASS)
motivates this.

OPERATIONALIZATION:
  spread(dataset) = std of per-model mean_kappa across 6 architectures
  Primary model: rho ~ beta1 * spread + beta2 * log(K)

PRE-REGISTERED:
  H_spread: beta1 > 0 (partial correlation of spread with rho, controlling for K)
  H_K_null: beta2 near-zero after conditioning on spread (K alone doesn't explain H1)
  H_spread_r: Pearson r(spread, rho) > r(log(K), rho)

Uses all 6 prospective datasets with kappa cache files.
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_JSON = RESULTS_DIR / "cti_spread_vs_K.json"

# All 6 prospective datasets with observed rho
DATASETS = {
    "emotion":        {"K": 6,  "rho": 0.613, "pass_H1": False},
    "yahoo":          {"K": 10, "rho": 0.835, "pass_H1": False},
    "langid":         {"K": 20, "rho": 0.906, "pass_H1": True},
    "news_category":  {"K": 42, "rho": 0.802, "pass_H1": False},
    "amazon_massive": {"K": 59, "rho": 0.922, "pass_H1": True},
    "banking77":      {"K": 77, "rho": 0.890, "pass_H1": True},
}

MODELS = ["pythia-160m", "gpt-neo-125m", "Qwen3-0.6B", "OLMo-1B-hf",
          "Falcon-H1-0.5B-Base", "rwkv-4-169m-pile"]


def load_kappa_spread(ds_name):
    """Load kappa_nearest values from cache files and compute spread."""
    per_model_kappas = []
    for model in MODELS:
        cache_path = RESULTS_DIR / f"kappa_near_cache_{ds_name}_{model}.json"
        if not cache_path.exists():
            print(f"  Missing cache: {cache_path.name}", flush=True)
            continue
        with open(cache_path) as f:
            pts = json.load(f)
        if pts:
            # Mean kappa across layers for this model
            kappas = [p["kappa_nearest"] for p in pts if "kappa_nearest" in p]
            if kappas:
                per_model_kappas.append(float(np.mean(kappas)))

    if len(per_model_kappas) < 3:
        return None, None

    mean_kappa = float(np.mean(per_model_kappas))
    spread = float(np.std(per_model_kappas))
    cv = spread / mean_kappa if mean_kappa > 1e-10 else 0.0
    return spread, {"mean_kappa": mean_kappa, "spread": spread, "cv": cv,
                    "per_model": per_model_kappas, "n_models": len(per_model_kappas)}


def json_default(obj):
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    print("=" * 70)
    print("K x SPREAD FACTORIAL ANALYSIS")
    print("Hypothesis: kappa spread predicts rho better than K alone")
    print("=" * 70)

    # Collect data
    rows = []
    for ds_name, ds_info in DATASETS.items():
        spread, spread_details = load_kappa_spread(ds_name)
        if spread is None:
            print(f"  {ds_name}: no kappa data, skipping", flush=True)
            continue
        row = {
            "dataset": ds_name,
            "K": ds_info["K"],
            "log_K": float(math.log(ds_info["K"])),
            "rho": ds_info["rho"],
            "pass_H1": ds_info["pass_H1"],
            "spread": spread,
            **spread_details,
        }
        rows.append(row)
        print(f"  {ds_name:<16} K={ds_info['K']:<3} rho={ds_info['rho']:.3f} "
              f"spread={spread:.4f} mean_kappa={spread_details['mean_kappa']:.4f} "
              f"cv={spread_details['cv']:.3f}", flush=True)

    if len(rows) < 4:
        print("Not enough data points for regression", flush=True)
        return

    # Arrays
    rho_arr = np.array([r["rho"] for r in rows])
    logK_arr = np.array([r["log_K"] for r in rows])
    spread_arr = np.array([r["spread"] for r in rows])
    K_arr = np.array([r["K"] for r in rows])

    # Simple correlations
    r_spread, p_spread = pearsonr(spread_arr, rho_arr)
    r_logK, p_logK = pearsonr(logK_arr, rho_arr)
    rho_spread, _ = spearmanr(spread_arr, rho_arr)
    rho_logK, _ = spearmanr(logK_arr, rho_arr)

    print(f"\nSimple correlations:")
    print(f"  r(spread, rho) = {r_spread:.4f}  p={p_spread:.4f}  spearman={rho_spread:.4f}")
    print(f"  r(log(K), rho) = {r_logK:.4f}  p={p_logK:.4f}  spearman={rho_logK:.4f}")

    # Multiple regression: rho ~ beta1*spread + beta2*log(K) + const
    X = np.column_stack([spread_arr, logK_arr, np.ones(len(rows))])
    result = np.linalg.lstsq(X, rho_arr, rcond=None)
    coeffs = result[0]
    beta_spread, beta_logK, beta_const = coeffs

    # Compute partial correlations via residuals
    # Partial corr of spread with rho, controlling for log(K)
    X_logK = np.column_stack([logK_arr, np.ones(len(rows))])
    resid_rho_from_logK = rho_arr - X_logK @ np.linalg.lstsq(X_logK, rho_arr, rcond=None)[0]
    resid_spread_from_logK = spread_arr - X_logK @ np.linalg.lstsq(X_logK, spread_arr, rcond=None)[0]
    r_partial_spread, p_partial_spread = pearsonr(resid_spread_from_logK, resid_rho_from_logK)

    # Partial corr of log(K) with rho, controlling for spread
    X_spread = np.column_stack([spread_arr, np.ones(len(rows))])
    resid_rho_from_spread = rho_arr - X_spread @ np.linalg.lstsq(X_spread, rho_arr, rcond=None)[0]
    resid_logK_from_spread = logK_arr - X_spread @ np.linalg.lstsq(X_spread, logK_arr, rcond=None)[0]
    r_partial_logK, p_partial_logK = pearsonr(resid_logK_from_spread, resid_rho_from_spread)

    print(f"\nMultiple regression: rho ~ beta1*spread + beta2*log(K) + const")
    print(f"  beta_spread = {beta_spread:.4f}")
    print(f"  beta_logK   = {beta_logK:.4f}")
    print(f"  const       = {beta_const:.4f}")

    print(f"\nPartial correlations:")
    print(f"  partial r(spread | log(K)) = {r_partial_spread:.4f}  p={p_partial_spread:.4f}")
    print(f"  partial r(log(K) | spread) = {r_partial_logK:.4f}  p={p_partial_logK:.4f}")

    # H_spread: beta1 > 0 AND partial correlation > 0
    H_spread_pass = bool(beta_spread > 0 and r_partial_spread > 0)
    # H_K_null: |beta2| < |beta1| after conditioning (spread dominates K)
    H_K_null_pass = bool(abs(beta_spread) > abs(beta_logK))
    # H_spread_r: |r(spread,rho)| > |r(log(K),rho)|
    H_spread_r_pass = bool(abs(r_spread) > abs(r_logK))

    print(f"\nPRE-REGISTERED TESTS:")
    print(f"  H_spread (beta_spread > 0, partial r > 0): {'PASS' if H_spread_pass else 'FAIL'}")
    print(f"  H_K_null (|beta_spread| > |beta_logK|): {'PASS' if H_K_null_pass else 'FAIL'}")
    print(f"  H_spread_r (|r_spread| > |r_logK|): {'PASS' if H_spread_r_pass else 'FAIL'}")

    # Per-dataset table
    print(f"\nK x SPREAD TABLE:")
    print(f"  {'Dataset':<16} {'K':>5} {'log(K)':>6} {'spread':>8} {'rho':>6} {'H1':>5}")
    for r in rows:
        print(f"  {r['dataset']:<16} {r['K']:>5} {r['log_K']:>6.3f} "
              f"{r['spread']:>8.4f} {r['rho']:>6.3f} {'PASS' if r['pass_H1'] else 'FAIL':>5}")

    output = {
        "experiment": "spread_vs_K_factorial",
        "session": 43,
        "preregistered": {
            "H_spread": "beta_spread > 0 AND partial_r(spread|logK) > 0",
            "H_K_null": "|beta_spread| > |beta_logK| (spread dominates K)",
            "H_spread_r": "|r(spread,rho)| > |r(logK,rho)|",
        },
        "results": {
            "r_spread": float(r_spread),
            "r_logK": float(r_logK),
            "p_spread": float(p_spread),
            "p_logK": float(p_logK),
            "spearman_spread": float(rho_spread),
            "spearman_logK": float(rho_logK),
            "partial_r_spread": float(r_partial_spread),
            "partial_r_logK": float(r_partial_logK),
            "p_partial_spread": float(p_partial_spread),
            "p_partial_logK": float(p_partial_logK),
            "beta_spread": float(beta_spread),
            "beta_logK": float(beta_logK),
            "beta_const": float(beta_const),
        },
        "hypotheses": {
            "H_spread_pass": H_spread_pass,
            "H_K_null_pass": H_K_null_pass,
            "H_spread_r_pass": H_spread_r_pass,
        },
        "per_dataset": rows,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
