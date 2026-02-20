#!/usr/bin/env python
"""
DERIVE kappa_c(K, n, d) FROM GAUSSIAN-CLUSTER THEORY

From the Gaussian mixture model:
  x = mu_y + epsilon, epsilon ~ N(0, Sigma_W), mu_y ~ N(0, Sigma_B)
  kappa = tr(Sigma_B) / tr(Sigma_W)

Same-class squared distance: E[D+] = 2*tr(Sigma_W) (between two points from same class)
Different-class squared distance: E[D-] = 2*tr(Sigma_W + Sigma_B) = 2*(1+kappa)*tr(Sigma_W)

For 1-NN with n_same samples from same class and n_diff from different classes:
  min D+ ~ 2*tr(Sigma_W) * (1 - c_+/sqrt(d))  where c_+ depends on n_same
  min D- ~ 2*(1+kappa)*tr(Sigma_W) * (1 - c_-/sqrt(d))  where c_- depends on n_diff

The critical point kappa_c is where E[min D+] = E[min D-]:
  2*tr(Sigma_W)*(1 - c_+/sqrt(d)) = 2*(1+kappa_c)*tr(Sigma_W)*(1 - c_-/sqrt(d))

Solving:
  1 - c_+/sqrt(d) = (1+kappa_c)*(1 - c_-/sqrt(d))
  kappa_c = (c_- - c_+) / (sqrt(d) - c_-)

For high d, the minimum of n iid samples from a distribution concentrates at:
  min ~ mu - sigma * sqrt(2*log(n))
In our case, D+ has n_same = n/K - 1 neighbors, D- has n_diff = n*(K-1)/K neighbors.

Using the chi-squared approximation for squared distances in d dimensions:
  D ~ 2*tr(Sigma)*(1 + sqrt(2/d)*z) where z ~ N(0,1)

The minimum of n such distances:
  min D ~ 2*tr(Sigma)*(1 - sqrt(2/d)*sqrt(2*log(n)))

So:
  c_+ = sqrt(2)*sqrt(2*log(n/K))  [same class neighbors]
  c_- = sqrt(2)*sqrt(2*log(n*(K-1)/K))  [different class neighbors]

kappa_c = (c_- - c_+) / (sqrt(d) - c_-)

Since n*(K-1)/K >> n/K for large K:
  c_- - c_+ = sqrt(2) * [sqrt(2*log(n*(K-1)/K)) - sqrt(2*log(n/K))]
            = sqrt(2) * sqrt(2) * [sqrt(log(n*(K-1)/K)) - sqrt(log(n/K))]
            = 2 * [sqrt(log(n) + log(1-1/K)) - sqrt(log(n) - log(K))]

For large K, this simplifies to:
  ~ 2 * [sqrt(log(n)) - sqrt(log(n) - log(K))]
  ~ 2 * log(K) / (2*sqrt(log(n))) = log(K)/sqrt(log(n))

So approximately:
  kappa_c ~ log(K) / (sqrt(d)*sqrt(log(n))) * [correction terms]

This predicts kappa_c grows with K and shrinks with d and n.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))


def sigmoid(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def kappac_theory(K, n, d):
    """Theoretical kappa_c from Gaussian-cluster model.

    kappa_c = (c_- - c_+) / (sqrt(d) - c_-)
    where c_pm = 2*sqrt(log(n_pm))
    n_+ = n/K, n_- = n*(K-1)/K
    """
    n_same = max(n / K, 2)
    n_diff = max(n * (K - 1) / K, 2)

    c_plus = 2 * np.sqrt(np.log(n_same))
    c_minus = 2 * np.sqrt(np.log(n_diff))

    denominator = np.sqrt(d) - c_minus
    if denominator <= 0:
        return 10.0  # very high kappa_c (degenerate case)

    return (c_minus - c_plus) / denominator


def main():
    print("=" * 70)
    print("DERIVING kappa_c(K, n, d) FROM GAUSSIAN-CLUSTER THEORY")
    print("=" * 70)

    # Load empirical kappa_c values from theory predictions
    with open(RESULTS_DIR / "cti_theory_predictions.json") as f:
        theory_data = json.load(f)

    pred2 = theory_data["predictions"]["2_critical_point"]

    # Get dataset info
    from hierarchical_datasets import load_hierarchical_dataset

    ds_info = {}
    for ds_name in pred2:
        ds = load_hierarchical_dataset(ds_name, split="test", max_samples=2000)
        labels = np.array([s.level1_label for s in ds.samples])
        K = len(np.unique(labels))
        n = len(labels)

        # Get embedding dimension from model (use cached data)
        # Approximate d from the first model used
        ds_info[ds_name] = {
            "K": K,
            "n": n,
            "kappa_c_empirical": pred2[ds_name]["kappa_c_logistic"],
        }

    # Get embedding dimensions (from model configs)
    # Averaged across models — use most common model Qwen2-0.5B (d=896)
    # Actually, we're averaging over layers so effective d varies
    # Let's estimate from the data. The key models:
    # SmolLM2-360M: d=960, Pythia-410m: d=1024, Qwen2-0.5B: d=896, Qwen3-0.6B: d=1024
    # Mamba-130m: d=768, Mamba-370m: d=1024, Mamba-790m: d=1536
    # Layer-averaged effective d varies, but let's use a representative value

    # For theory test, use d as a parameter to fit
    print(f"\n--- Empirical kappa_c values ---")
    for ds_name, info in sorted(ds_info.items(), key=lambda x: x[1]["K"]):
        print(f"  {ds_name:>20}: K={info['K']:>3}, n={info['n']:>4}, "
              f"kappa_c={info['kappa_c_empirical']:.4f}")

    # Test theory with different d values
    print(f"\n--- Theory predictions for different d ---")
    Ks = np.array([ds_info[ds]["K"] for ds in sorted(ds_info)])
    ns = np.array([ds_info[ds]["n"] for ds in sorted(ds_info)])
    kcs_empirical = np.array([ds_info[ds]["kappa_c_empirical"] for ds in sorted(ds_info)])
    ds_names = sorted(ds_info)

    for d_test in [100, 200, 500, 1000, 2000]:
        kcs_theory = np.array([kappac_theory(K, n, d_test) for K, n in zip(Ks, ns)])
        rho, p = spearmanr(kcs_theory, kcs_empirical)
        r, pr = pearsonr(kcs_theory, kcs_empirical)
        print(f"  d={d_test:>5}: theory_kc={[f'{x:.4f}' for x in kcs_theory]}, "
              f"rho={rho:.4f}, r={r:.4f}")

    # Fit d as free parameter
    print(f"\n--- Fitting d to match empirical kappa_c ---")

    def loss(params):
        d_fit = params[0]
        scale = params[1]
        offset = params[2]
        if d_fit < 10:
            return 1e10
        kcs_pred = np.array([scale * kappac_theory(K, n, d_fit) + offset
                             for K, n in zip(Ks, ns)])
        return np.sum((kcs_pred - kcs_empirical) ** 2)

    best = minimize(loss, [500, 1.0, 0.0], method="Nelder-Mead")
    d_best, scale_best, offset_best = best.x

    kcs_best = np.array([scale_best * kappac_theory(K, n, d_best) + offset_best
                         for K, n in zip(Ks, ns)])
    rho_best, _ = spearmanr(kcs_best, kcs_empirical)
    r_best, _ = pearsonr(kcs_best, kcs_empirical)
    mae_best = float(np.mean(np.abs(kcs_best - kcs_empirical)))

    print(f"  d_fit={d_best:.0f}, scale={scale_best:.4f}, offset={offset_best:.4f}")
    print(f"  rho={rho_best:.4f}, r={r_best:.4f}, MAE={mae_best:.4f}")

    for ds_name, kc_emp, kc_pred in zip(ds_names, kcs_empirical, kcs_best):
        K = ds_info[ds_name]["K"]
        print(f"    {ds_name:>20} (K={K:>3}): emp={kc_emp:.4f}, pred={kc_pred:.4f}, "
              f"err={abs(kc_emp-kc_pred):.4f}")

    # Also test simpler formula: kappa_c ~ a * log(K) + b
    print(f"\n--- Simplified models for kappa_c ---")

    # Model 1: kappa_c = a * log(K) + b
    log_Ks = np.log(Ks)
    a1, b1 = np.polyfit(log_Ks, kcs_empirical, 1)
    pred1 = a1 * log_Ks + b1
    r2_1 = 1 - np.sum((kcs_empirical - pred1) ** 2) / np.sum((kcs_empirical - kcs_empirical.mean()) ** 2)
    print(f"  kappa_c = {a1:.4f}*log(K) + {b1:.4f}: R^2={r2_1:.4f}")

    # Model 2: kappa_c = a * sqrt(K) + b
    sqrt_Ks = np.sqrt(Ks)
    a2, b2 = np.polyfit(sqrt_Ks, kcs_empirical, 1)
    pred2_vals = a2 * sqrt_Ks + b2
    r2_2 = 1 - np.sum((kcs_empirical - pred2_vals) ** 2) / np.sum((kcs_empirical - kcs_empirical.mean()) ** 2)
    print(f"  kappa_c = {a2:.4f}*sqrt(K) + {b2:.4f}: R^2={r2_2:.4f}")

    # Model 3: kappa_c = a * K + b
    a3, b3 = np.polyfit(Ks.astype(float), kcs_empirical, 1)
    pred3 = a3 * Ks + b3
    r2_3 = 1 - np.sum((kcs_empirical - pred3) ** 2) / np.sum((kcs_empirical - kcs_empirical.mean()) ** 2)
    print(f"  kappa_c = {a3:.6f}*K + {b3:.4f}: R^2={r2_3:.4f}")

    # Model 4: kappa_c = a * log(K)/sqrt(log(n)) + b (from theory)
    theory_term = np.log(Ks) / np.sqrt(np.log(ns))
    a4, b4 = np.polyfit(theory_term, kcs_empirical, 1)
    pred4 = a4 * theory_term + b4
    r2_4 = 1 - np.sum((kcs_empirical - pred4) ** 2) / np.sum((kcs_empirical - kcs_empirical.mean()) ** 2)
    print(f"  kappa_c = {a4:.4f}*log(K)/sqrt(log(n)) + {b4:.4f}: R^2={r2_4:.4f}")

    # Best model
    models = {
        "a*log(K)+b": r2_1,
        "a*sqrt(K)+b": r2_2,
        "a*K+b": r2_3,
        "a*log(K)/sqrt(log(n))+b": r2_4,
    }
    best_model = max(models, key=models.get)
    print(f"\n  Best simple model: {best_model} (R^2={models[best_model]:.4f})")

    # ============================================================
    # LEAVE-ONE-OUT PREDICTION OF kappa_c
    # ============================================================
    print(f"\n{'='*70}")
    print("LEAVE-ONE-OUT PREDICTION OF kappa_c")
    print(f"{'='*70}")

    # Use the best model
    loo_errors = []
    for i in range(len(Ks)):
        mask = np.ones(len(Ks), dtype=bool)
        mask[i] = False

        # Fit on training data
        a_loo, b_loo = np.polyfit(log_Ks[mask], kcs_empirical[mask], 1)
        kc_pred = a_loo * log_Ks[i] + b_loo
        err = abs(kc_pred - kcs_empirical[i])
        loo_errors.append(err)
        print(f"  Hold out {ds_names[i]:>20} (K={Ks[i]:>3}): "
              f"pred={kc_pred:.4f}, emp={kcs_empirical[i]:.4f}, err={err:.4f}")

    mean_loo_mae = np.mean(loo_errors)
    print(f"\n  Mean LOO MAE: {mean_loo_mae:.4f}")

    # ============================================================
    # SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("Theory kappa_c rank-correlates with K (rho>0.9)",
         bool(rho_best > 0.9), f"rho={rho_best:.4f}"),
        ("Theory + fit MAE < 0.05",
         mae_best < 0.05, f"MAE={mae_best:.4f}"),
        ("Simple log(K) model R^2 > 0.90",
         r2_1 > 0.90, f"R^2={r2_1:.4f}"),
        ("LOO prediction MAE < 0.05",
         mean_loo_mae < 0.05, f"MAE={mean_loo_mae:.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "derive_kappac",
        "theory": "kappa_c = (c_- - c_+) / (sqrt(d) - c_-) where c_pm = 2*sqrt(log(n_pm))",
        "empirical_kappac": {ds: ds_info[ds] for ds in ds_names},
        "fitted_d": float(d_best),
        "fitted_scale": float(scale_best),
        "fitted_offset": float(offset_best),
        "theory_fit": {
            "rho": float(rho_best), "r": float(r_best), "mae": float(mae_best),
        },
        "simple_models": {
            "log_K": {"a": float(a1), "b": float(b1), "r2": float(r2_1)},
            "sqrt_K": {"a": float(a2), "b": float(b2), "r2": float(r2_2)},
            "linear_K": {"a": float(a3), "b": float(b3), "r2": float(r2_3)},
            "theory_term": {"a": float(a4), "b": float(b4), "r2": float(r2_4)},
        },
        "best_simple_model": best_model,
        "loo_mae": float(mean_loo_mae),
        "scorecard": {
            "passes": passes, "total": len(checks),
            "details": [{"criterion": c, "passed": bool(p), "value": v}
                         for c, p, v in checks],
        },
    }

    out_path = RESULTS_DIR / "cti_derive_kappac.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
