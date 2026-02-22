#!/usr/bin/env python -u
"""
TEST: Does A_surgery scale as A_renorm / d_eff?
(the 1/d_eff scaling hypothesis)

If confirmed: each active dimension contributes independently, 
and the formula is an additive sum over d_eff dimensions.
Surgery tests one dimension -> gets 1/d_eff of the global prediction.
"""

import json
import sys
import numpy as np

RESULT_PATH = "results/cti_linear_regime_surgery.json"
A_RENORM = 1.0535

def fit_A_emp(r_vals, delta_logit_obs, kappa_base, d_eff_base):
    """Fit A_emp from surgery data using OLS on the sqrt form."""
    r_arr = np.array(r_vals)
    delta_arr = np.array(delta_logit_obs)
    mask = (r_arr != 1.0)
    r_m = r_arr[mask]
    delta_m = delta_arr[mask]
    X = kappa_base * np.sqrt(d_eff_base) * (np.sqrt(r_m) - 1)
    A_emp = np.dot(X, delta_m) / np.dot(X, X)
    pred = A_emp * X
    SS = np.sum((delta_m - np.mean(delta_m))**2)
    R2 = 1 - np.sum((delta_m - pred)**2) / SS if SS > 0 else 0.0
    return A_emp, R2

def main():
    try:
        with open(RESULT_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {RESULT_PATH} not found. Run cti_linear_regime_surgery.py first.")
        sys.exit(1)

    # Collect per-seed results
    seed_results = {}
    for rec in data:
        seed = rec.get("seed")
        if seed is None:
            continue
        if seed not in seed_results:
            seed_results[seed] = {
                "kappa_base": rec.get("kappa_base"),
                "d_eff_base": rec.get("d_eff_base"),
                "kappa_eff_base": rec.get("kappa_eff_base"),
                "q_base": rec.get("q_base"),
                "logit_base": rec.get("logit_base"),
                "r_vals": [],
                "logit_act": []
            }
        seed_results[seed]["r_vals"].append(rec["r"])
        seed_results[seed]["logit_act"].append(rec["logit_actual"])

    print("="*70)
    print("TEST: A_surgery = A_renorm / d_eff  (1/d_eff Scaling Hypothesis)")
    print("="*70)
    print()
    print(f"{'Seed':>6} | {'d_eff':>7} | {'kappa':>7} | {'kappa_eff':>9} | {'A_emp':>7} | {'A*d_eff':>9} | {'A_renorm':>8} | {'R2':>6}")
    print("-"*80)

    A_emp_vals = []
    d_eff_vals = []
    products = []

    for seed, sdata in sorted(seed_results.items()):
        kappa = sdata["kappa_base"]
        d_eff = sdata["d_eff_base"]
        kappa_eff = sdata["kappa_eff_base"]
        logit_base = sdata["logit_base"]

        delta_obs = [la - logit_base for la in sdata["logit_act"]]
        A_emp, R2 = fit_A_emp(sdata["r_vals"], delta_obs, kappa, d_eff)
        product = A_emp * d_eff
        products.append(product)
        A_emp_vals.append(A_emp)
        d_eff_vals.append(d_eff)

        print(f"{seed:>6} | {d_eff:>7.2f} | {kappa:>7.4f} | {kappa_eff:>9.4f} | {A_emp:>7.4f} | {product:>9.4f} | {A_RENORM:>8.4f} | {R2:>6.4f}")

    print("-"*80)
    if products:
        print(f"{'MEAN':>6} | {'':>7} | {'':>7} | {'':>9} | {np.mean(A_emp_vals):>7.4f} | {np.mean(products):>9.4f} | {A_RENORM:>8.4f} | {'':>6}")
        print(f"{'STD':>6} | {'':>7} | {'':>7} | {'':>9} | {np.std(A_emp_vals):>7.4f} | {np.std(products):>9.4f} | {'':>8} | {'':>6}")
        mean_product = np.mean(products)
        ratio = mean_product / A_RENORM
        print()
        print(f"Mean A_emp * d_eff = {mean_product:.4f}  (A_renorm = {A_RENORM:.4f})")
        print(f"Ratio = {ratio:.3f}  (ideal = 1.000 if hypothesis A_emp = A_renorm/d_eff is true)")
        print()

        # Correlation test: does A_emp decrease with d_eff?
        if len(d_eff_vals) > 1:
            r_corr = np.corrcoef(d_eff_vals, A_emp_vals)[0, 1]
            print(f"Pearson r(d_eff, A_emp) = {r_corr:.4f}")
            print(f"  Hypothesis predicts r = -1.0 (perfect inverse)")
            print(f"  Null (A_emp = constant) predicts r = 0.0")
            print()

        # Verdict
        if abs(ratio - 1.0) < 0.25 and len(products) >= 2:
            print("VERDICT: CONSISTENT with 1/d_eff hypothesis (ratio within 25% of 1.0)")
            print("  -> Surgery confirms formula as global law over d_eff dimensions")
        elif ratio > 1.2:
            print(f"VERDICT: A_emp * d_eff > A_renorm by {(ratio-1)*100:.1f}%")
            print("  -> d_eff causal effect weaker than 1/d_eff prediction")
            print("  -> Alternative: A_emp approximately constant, formula is partially correlational")
        else:
            print(f"VERDICT: INCONCLUSIVE ({len(products)} seeds only)")

if __name__ == "__main__":
    main()
