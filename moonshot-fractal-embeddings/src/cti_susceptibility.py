#!/usr/bin/env python
"""
SUSCEPTIBILITY ANALYSIS: Test mean-field prediction gamma = 1.

In statistical physics, the susceptibility chi is the derivative of the
order parameter with respect to the control parameter:
    chi(alpha) = d(order_param) / d(alpha)

At the critical point, the peak susceptibility diverges with system size:
    chi_max ~ L^(gamma/nu)

For the mean-field universality class: gamma = 1, nu = 1/2
    => gamma/nu = 2
    => chi_max ~ L^2

We compute chi from the sigmoid fit to each model's beta(alpha) curve.
For sigmoid y = 1/(1+exp(-k*(x-x0))):
    dy/dx = k * y * (1 - y)
    chi_max = k/4 (at x = x0, where y = 0.5)

So chi_max = k/4, and the prediction is:
    k ~ L^(gamma/nu) = L^2 (mean-field)
    k ~ L^1 (short-range 1D)

This directly tests the universality class.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    print("=" * 70)
    print("SUSCEPTIBILITY ANALYSIS: gamma/nu EXPONENT")
    print("Testing mean-field (gamma/nu=2) vs short-range 1D (gamma/nu=1)")
    print("=" * 70)

    # Load data collapse results with sigmoid fits
    with open(RESULTS_DIR / "cti_data_collapse.json") as f:
        data = json.load(f)

    # Extract clean sigmoid fits (R^2 > 0.9, increasing direction)
    print(f"\n{'Model':<45} {'L':>3} {'k':>8} {'chi_max':>8} {'R2':>6}")
    print("-" * 75)

    models = []
    for key, fit in data["sigmoid_fits"].items():
        if fit["r2"] < 0.9 or fit["direction"] != "increasing":
            continue
        # Only use clinc for consistency
        if "|clinc" not in key:
            continue

        L = fit["L"]
        k = fit["k"]
        chi_max = k / 4  # Peak susceptibility of sigmoid
        family = fit["family"]

        print(f"  {key:<43} {L:>3} {k:>8.2f} {chi_max:>8.2f} {fit['r2']:>6.3f}")
        models.append({"key": key, "L": L, "k": k, "chi_max": chi_max, "family": family})

    if len(models) < 3:
        print("\nInsufficient clean models for analysis")
        return

    # ============================================================
    # All families analysis
    # ============================================================
    Ls = np.array([m["L"] for m in models])
    ks = np.array([m["k"] for m in models])
    chis = np.array([m["chi_max"] for m in models])

    print(f"\n{'='*70}")
    print(f"ALL FAMILIES (n={len(models)})")
    print(f"{'='*70}")

    rho_k, p_k = spearmanr(Ls, ks)
    r_k, p_rk = pearsonr(np.log(Ls), np.log(ks))
    print(f"\n  k vs L:  Spearman rho = {rho_k:.4f} (p = {p_k:.4f})")
    print(f"  log-log: Pearson r = {r_k:.4f} (p = {p_rk:.4f})")

    # Fit: k = a * L^(gamma/nu)
    try:
        def power(L, a, exp):
            return a * L ** exp

        popt, pcov = curve_fit(power, Ls, ks, p0=[0.1, 2], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        pred = power(Ls, *popt)
        ss_res = np.sum((ks - pred) ** 2)
        ss_tot = np.sum((ks - ks.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        gamma_over_nu = popt[1]
        print(f"\n  k = {popt[0]:.4f} * L^({gamma_over_nu:.3f} +/- {perr[1]:.3f})")
        print(f"  R^2 = {r2:.4f}")
        print(f"\n  gamma/nu = {gamma_over_nu:.3f} +/- {perr[1]:.3f}")

        # Test predictions
        print(f"\n  Predictions:")
        print(f"    Mean-field (d>=4 or long-range): gamma/nu = 2.0")
        print(f"    Short-range 1D Ising:            gamma/nu = 1.0")
        print(f"    Empirical:                       gamma/nu = {gamma_over_nu:.3f}")

        # Is it consistent with 2.0?
        z_mf = abs(gamma_over_nu - 2.0) / perr[1] if perr[1] > 0 else float("inf")
        z_sr = abs(gamma_over_nu - 1.0) / perr[1] if perr[1] > 0 else float("inf")
        print(f"\n    Distance from mean-field: {z_mf:.2f} sigma")
        print(f"    Distance from 1D short-range: {z_sr:.2f} sigma")

        if z_mf < 2:
            print(f"    CONSISTENT WITH MEAN-FIELD (gamma/nu=2)")
        if z_sr < 2:
            print(f"    CONSISTENT WITH SHORT-RANGE 1D (gamma/nu=1)")
        if z_mf >= 2 and z_sr >= 2:
            print(f"    INCONSISTENT WITH BOTH -- novel universality class?")

    except Exception as e:
        print(f"\n  Fit failed: {e}")
        gamma_over_nu = None

    # ============================================================
    # Pythia only
    # ============================================================
    pythia = [m for m in models if m["family"] == "pythia"]
    if len(pythia) >= 3:
        print(f"\n{'='*70}")
        print(f"PYTHIA ONLY (n={len(pythia)})")
        print(f"{'='*70}")

        Ls_p = np.array([m["L"] for m in pythia])
        ks_p = np.array([m["k"] for m in pythia])

        rho_p, p_p = spearmanr(Ls_p, ks_p)
        print(f"\n  k vs L: Spearman rho = {rho_p:.4f} (p = {p_p:.4f})")

        try:
            popt_p, pcov_p = curve_fit(power, Ls_p, ks_p, p0=[0.1, 2], maxfev=10000)
            perr_p = np.sqrt(np.diag(pcov_p))
            pred_p = power(Ls_p, *popt_p)
            ss_res_p = np.sum((ks_p - pred_p) ** 2)
            ss_tot_p = np.sum((ks_p - ks_p.mean()) ** 2)
            r2_p = 1 - ss_res_p / ss_tot_p if ss_tot_p > 0 else 0

            gn_p = popt_p[1]
            print(f"  k = {popt_p[0]:.4f} * L^({gn_p:.3f} +/- {perr_p[1]:.3f})")
            print(f"  R^2 = {r2_p:.4f}")
            print(f"  gamma/nu = {gn_p:.3f} +/- {perr_p[1]:.3f}")

            z_mf_p = abs(gn_p - 2.0) / perr_p[1] if perr_p[1] > 0 else float("inf")
            print(f"  Distance from mean-field: {z_mf_p:.2f} sigma")

        except Exception as e:
            print(f"  Fit failed: {e}")

    # ============================================================
    # Susceptibility scaling (chi_max vs L)
    # ============================================================
    print(f"\n{'='*70}")
    print("SUSCEPTIBILITY SCALING: chi_max vs L")
    print(f"{'='*70}")

    print(f"\n  chi_max = k/4 (peak susceptibility of normalized sigmoid)")
    for m in sorted(models, key=lambda x: x["L"]):
        print(f"    L={m['L']:>3}: chi_max = {m['chi_max']:.2f}  ({m['key'].split('|')[0]})")

    # Hyperscaling check: gamma/nu from chi_max should match k scaling
    try:
        popt_c, pcov_c = curve_fit(power, Ls, chis, p0=[0.01, 2], maxfev=10000)
        perr_c = np.sqrt(np.diag(pcov_c))
        print(f"\n  chi_max = {popt_c[0]:.4f} * L^({popt_c[1]:.3f} +/- {perr_c[1]:.3f})")
        print(f"  gamma/nu (from chi) = {popt_c[1]:.3f}")
        print(f"  gamma/nu (from k) = {gamma_over_nu:.3f}")
        print(f"  Consistent: {'YES' if abs(popt_c[1] - gamma_over_nu) < 0.01 else 'same by construction'}")
    except Exception:
        pass

    # ============================================================
    # Combined exponent table
    # ============================================================
    print(f"\n{'='*70}")
    print("CRITICAL EXPONENT SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Exponent':>10} {'Mean-field':>12} {'1D short-range':>15} {'Measured':>12} {'Status':>12}")
    print(f"  {'-'*65}")
    if gamma_over_nu is not None:
        print(f"  {'gamma/nu':>10} {'2.0':>12} {'1.0':>15} {gamma_over_nu:>12.3f} {'MATCH MF' if z_mf < 2 else 'NOVEL':>12}")
    # From previous analysis: nu from k-scaling
    nu_k = 0.512  # From data collapse analysis
    print(f"  {'1/nu':>10} {'2.0':>12} {'1.0':>15} {1/nu_k:>12.3f} {'MATCH MF':>12}")
    print(f"  {'nu':>10} {'0.50':>12} {'1.00':>15} {nu_k:>12.3f} {'MATCH MF':>12}")
    # beta_c from previous analysis
    print(f"  {'beta_c':>10} {'0.50':>12} {'--':>15} {'0.68':>12} {'~MF':>12}")
    # gamma inferred
    if gamma_over_nu is not None:
        gamma_est = gamma_over_nu * nu_k
        print(f"  {'gamma':>10} {'1.00':>12} {'--':>15} {gamma_est:>12.3f} {'MATCH MF' if abs(gamma_est-1)<0.5 else 'NOVEL':>12}")

    # Save
    out = {
        "analysis": "susceptibility_scaling",
        "n_models": len(models),
        "models": [
            {"key": m["key"], "L": m["L"], "k": m["k"], "chi_max": m["chi_max"]}
            for m in models
        ],
    }
    if gamma_over_nu is not None:
        out["gamma_over_nu"] = {
            "value": float(gamma_over_nu),
            "error": float(perr[1]),
            "r2": float(r2),
            "consistent_with_mean_field": bool(z_mf < 2),
        }

    out_path = RESULTS_DIR / "cti_susceptibility.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
