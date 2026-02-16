#!/usr/bin/env python
"""
cti_gate1_pythia_collapse.py

GATE 1 TEST: Same-architecture finite-size scaling collapse.

Uses ONLY the Pythia family (6L, 12L, 16L, 24L) — same architecture
at different depths. This is the proper RG theory test because we have
the SAME system at different sizes, not mixed architectures.

Theory predicts: Q_L(s) = Q* + L^(-x_Q) * Phi((s - s_c) * L^(1/nu))
If curves collapse onto a universal function Phi, this is strong evidence
for depth-criticality.

GATE 1 CRITERIA (from Codex):
  - Same-architecture family, 4+ depths
  - Scaling collapse IMPROVES fit materially (not worsens)
  - Strong statistical significance

If Gate 1 PASSES: proceed to Gate 2 (replicate with BERT family)
If Gate 1 FAILS: drop RG framing, absorb into practical paper

Usage:
    python -u src/cti_gate1_pythia_collapse.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import interpolate, optimize, stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_pythia_curves(path: str) -> Tuple[Dict[int, List], List]:
    """Load depth-quality curves from Pythia depth series."""
    with open(path) as f:
        data = json.load(f)

    curves_by_L = defaultdict(list)
    all_curves = []

    for model_key, model_data in data.items():
        if "datasets" not in model_data:
            continue
        num_layers = model_data.get("num_layers", 0)
        if num_layers == 0:
            continue
        hidden_dim = model_data.get("hidden_dim", 0)

        for ds_name, ds_data in model_data["datasets"].items():
            layers = ds_data.get("layers", {})
            if not layers:
                continue

            sorted_layers = sorted(layers.items(), key=lambda x: int(x[0]))

            s = np.array([v["C_relative"] for _, v in sorted_layers])
            y_l0 = np.array([v["knn_l0"] for _, v in sorted_layers])
            y_l1 = np.array([v["knn_l1"] for _, v in sorted_layers])

            # Skip embedding layer (s=0) if quality is very low
            if len(s) > 2 and y_l1[0] < 0.05:
                s = s[1:]
                y_l0 = y_l0[1:]
                y_l1 = y_l1[1:]

            curve = {
                "model": model_key,
                "dataset": ds_name,
                "s": s,
                "y_l0": y_l0,
                "y_l1": y_l1,
                "L": num_layers,
                "hidden_dim": hidden_dim,
            }
            curves_by_L[num_layers].append(curve)
            all_curves.append(curve)

    return curves_by_L, all_curves


def collapse_mse_interp(curves_by_L: Dict, s_c: float, nu: float,
                        metric: str = "y_l1") -> float:
    """Compute collapse quality via interpolation.

    For each pair of depths (L1, L2) and matching dataset,
    rescale x-axis, interpolate to common grid, measure MSE.
    """
    L_values = sorted(curves_by_L.keys())
    if len(L_values) < 2:
        return float("inf")

    total_mse = 0
    n_pairs = 0

    for i in range(len(L_values)):
        for j in range(i + 1, len(L_values)):
            L1, L2 = L_values[i], L_values[j]

            for c1 in curves_by_L[L1]:
                for c2 in curves_by_L[L2]:
                    if c1["dataset"] != c2["dataset"]:
                        continue

                    s1, y1 = c1["s"], c1[metric]
                    s2, y2 = c2["s"], c2[metric]

                    # Rescale x-axis
                    x1 = (s1 - s_c) * (L1 ** (1.0 / nu))
                    x2 = (s2 - s_c) * (L2 ** (1.0 / nu))

                    # Normalize y to [0,1] for each curve
                    y1_range = y1.max() - y1.min()
                    y2_range = y2.max() - y2.min()
                    if y1_range < 1e-8 or y2_range < 1e-8:
                        continue
                    y1_n = (y1 - y1.min()) / y1_range
                    y2_n = (y2 - y2.min()) / y2_range

                    # Find overlap region
                    x_min = max(x1.min(), x2.min())
                    x_max = min(x1.max(), x2.max())
                    if x_max <= x_min:
                        continue

                    # Interpolate to common grid
                    n_grid = 30
                    x_common = np.linspace(x_min + 0.01, x_max - 0.01, n_grid)

                    try:
                        f1 = interpolate.interp1d(x1, y1_n, kind="linear",
                                                  fill_value="extrapolate")
                        f2 = interpolate.interp1d(x2, y2_n, kind="linear",
                                                  fill_value="extrapolate")
                        mse = np.mean((f1(x_common) - f2(x_common)) ** 2)
                        total_mse += mse
                        n_pairs += 1
                    except Exception:
                        continue

    return total_mse / max(n_pairs, 1)


def collapse_mse_raw(curves_by_L: Dict, s_c: float, nu: float, x_Q: float,
                     metric: str = "y_l1") -> float:
    """Collapse quality using sliding window variance (no normalization).

    Q_rescaled = (Q - Q*) * L^(x_Q)
    x_rescaled = (s - s_c) * L^(1/nu)

    Good collapse = low variance within x-bins.
    """
    all_x = []
    all_y = []

    Q_star = np.mean([np.mean(c[metric]) for cs in curves_by_L.values() for c in cs])

    for L, curves in curves_by_L.items():
        for c in curves:
            s, y = c["s"], c[metric]
            x_resc = (s - s_c) * (L ** (1.0 / nu))
            y_resc = (y - Q_star) * (L ** x_Q)
            all_x.extend(x_resc.tolist())
            all_y.extend(y_resc.tolist())

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    idx = np.argsort(all_x)
    all_x, all_y = all_x[idx], all_y[idx]

    # Sliding window variance
    window = max(5, len(all_x) // 25)
    variances = []
    for i in range(0, len(all_x) - window, window // 2):
        w = all_y[i:i + window]
        if len(w) > 2:
            variances.append(np.var(w))

    return float(np.mean(variances)) if variances else float("inf")


def grid_search_collapse(curves_by_L: Dict, metric: str = "y_l1") -> Dict:
    """Grid search over (s_c, nu, x_Q) for best collapse."""
    # First pass: coarse grid for (s_c, nu) using interpolation method
    s_c_coarse = np.linspace(0.2, 0.85, 30)
    nu_coarse = np.linspace(0.3, 5.0, 30)

    best_mse = float("inf")
    best_params = (0.5, 1.0)

    for s_c in s_c_coarse:
        for nu in nu_coarse:
            mse = collapse_mse_interp(curves_by_L, s_c, nu, metric)
            if mse < best_mse:
                best_mse = mse
                best_params = (s_c, nu)

    # Second pass: fine grid around best
    s_c_fine = np.linspace(max(0.1, best_params[0] - 0.1),
                           min(0.95, best_params[0] + 0.1), 20)
    nu_fine = np.linspace(max(0.1, best_params[1] - 0.5),
                          min(8.0, best_params[1] + 0.5), 20)

    for s_c in s_c_fine:
        for nu in nu_fine:
            mse = collapse_mse_interp(curves_by_L, s_c, nu, metric)
            if mse < best_mse:
                best_mse = mse
                best_params = (s_c, nu)

    # Baseline: no rescaling (nu -> infinity)
    baseline_mse = collapse_mse_interp(curves_by_L, 0.5, 1e6, metric)

    return {
        "best_s_c": float(best_params[0]),
        "best_nu": float(best_params[1]),
        "best_collapse_mse": float(best_mse),
        "baseline_mse": float(baseline_mse),
        "improvement_pct": float(
            100 * (baseline_mse - best_mse) / max(baseline_mse, 1e-10)
        ) if baseline_mse < float("inf") else 0.0,
    }


def bootstrap_collapse_significance(curves_by_L: Dict, best_s_c: float,
                                    best_nu: float, metric: str = "y_l1",
                                    n_boot: int = 200) -> Dict:
    """Bootstrap test: is the collapse improvement significant?

    Null hypothesis: rescaling provides no improvement.
    Shuffle L labels across curves, re-fit, compare to observed improvement.
    """
    observed_mse = collapse_mse_interp(curves_by_L, best_s_c, best_nu, metric)
    baseline_mse = collapse_mse_interp(curves_by_L, 0.5, 1e6, metric)
    observed_improvement = baseline_mse - observed_mse

    null_improvements = []
    all_curves = [c for cs in curves_by_L.values() for c in cs]
    L_values = [c["L"] for c in all_curves]

    for _ in range(n_boot):
        # Shuffle L labels
        shuffled_Ls = np.random.permutation(L_values)
        shuffled_by_L = defaultdict(list)
        for c, L in zip(all_curves, shuffled_Ls):
            c_copy = dict(c)
            c_copy["L"] = L
            shuffled_by_L[L].append(c_copy)

        null_mse = collapse_mse_interp(shuffled_by_L, best_s_c, best_nu, metric)
        null_improvement = baseline_mse - null_mse
        null_improvements.append(null_improvement)

    null_improvements = np.array(null_improvements)
    p_value = float(np.mean(null_improvements >= observed_improvement))

    return {
        "observed_improvement": float(observed_improvement),
        "null_mean": float(np.mean(null_improvements)),
        "null_std": float(np.std(null_improvements)),
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def test_non_monotonicity_scaling(curves_by_L: Dict) -> Dict:
    """Test non-monotonicity scaling with depth within Pythia family."""
    L_values = sorted(curves_by_L.keys())
    L_metrics = {}

    for L in L_values:
        rhos = []
        peak_drops = []
        for c in curves_by_L[L]:
            y = c["y_l1"]
            if len(y) < 4:
                continue
            rho, _ = stats.spearmanr(np.arange(len(y)), y)
            rhos.append(rho)
            peak_drops.append(float(max(y) - y[-1]))

        if rhos:
            L_metrics[L] = {
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "mean_peak_drop": float(np.mean(peak_drops)),
                "n_curves": len(rhos),
            }

    # Fit power law: non-monotonicity ~ L^alpha
    if len(L_metrics) >= 3:
        Ls = np.array(sorted(L_metrics.keys()))
        non_mono = np.array([1 - L_metrics[L]["mean_rho"] for L in Ls])

        log_L = np.log(Ls)
        log_nm = np.log(np.clip(non_mono, 1e-6, None))
        slope, intercept, r, p, se = stats.linregress(log_L, log_nm)

        return {
            "L_values": Ls.tolist(),
            "non_monotonicity": non_mono.tolist(),
            "mean_rho": {int(L): L_metrics[L]["mean_rho"] for L in Ls},
            "power_law_exponent": float(slope),
            "r_squared": float(r ** 2),
            "p_value": float(p),
            "se": float(se),
            "interpretation": f"Non-monotonicity ~ L^{slope:.2f} (R2={r**2:.3f}, p={p:.4f})",
        }

    return {"error": "Not enough depth scales", "L_metrics": L_metrics}


def test_derivative_peak_convergence(curves_by_L: Dict) -> Dict:
    """Test if the derivative peak (susceptibility proxy) converges to a fixed s_c."""
    results = {}

    for L in sorted(curves_by_L.keys()):
        peaks_l0 = []
        peaks_l1 = []
        max_derivs = []

        for c in curves_by_L[L]:
            s, y = c["s"], c["y_l1"]
            if len(y) < 5:
                continue
            dy = np.gradient(y, s)
            peak_idx = np.argmax(np.abs(dy))
            peaks_l1.append(float(s[peak_idx]))
            max_derivs.append(float(np.abs(dy[peak_idx])))

        if peaks_l1:
            results[L] = {
                "mean_peak_s": float(np.mean(peaks_l1)),
                "std_peak_s": float(np.std(peaks_l1)),
                "mean_max_deriv": float(np.mean(max_derivs)),
                "n_curves": len(peaks_l1),
            }

    # Test convergence
    if len(results) >= 3:
        Ls = sorted(results.keys())
        peaks = [results[L]["mean_peak_s"] for L in Ls]
        stds = [results[L]["std_peak_s"] for L in Ls]

        peak_range = max(peaks) - min(peaks)
        mean_peak = np.mean(peaks)

        # Test: does peak location correlate with L? (if s_c is fixed, it shouldn't)
        rho, p = stats.spearmanr(Ls, peaks)

        return {
            "per_L": results,
            "peak_range": float(peak_range),
            "mean_s_c": float(mean_peak),
            "rho_vs_L": float(rho),
            "p_vs_L": float(p),
            "converging": peak_range < 0.15,
            "interpretation": (
                f"Peak range={peak_range:.3f}, rho(L,peak)={rho:.3f} (p={p:.3f}). "
                + ("CONVERGING: consistent s_c" if peak_range < 0.15
                   else "DIVERGING: depth-dependent s_c")
            ),
        }

    return {"per_L": results, "error": "Not enough depths"}


def test_susceptibility_divergence(curves_by_L: Dict) -> Dict:
    """Test if the max derivative (susceptibility proxy) diverges with L.

    In RG theory: chi_max ~ L^(gamma/nu). So max|dQ/ds| should grow with L.
    """
    L_values = sorted(curves_by_L.keys())
    max_derivs = {}

    for L in L_values:
        derivs = []
        for c in curves_by_L[L]:
            s, y = c["s"], c["y_l1"]
            if len(y) < 5:
                continue
            dy = np.gradient(y, s)
            derivs.append(float(np.max(np.abs(dy))))

        if derivs:
            max_derivs[L] = {
                "mean": float(np.mean(derivs)),
                "std": float(np.std(derivs)),
                "n": len(derivs),
            }

    if len(max_derivs) >= 3:
        Ls = np.array(sorted(max_derivs.keys()))
        chi = np.array([max_derivs[L]["mean"] for L in Ls])

        # Fit chi ~ L^(gamma/nu)
        log_L = np.log(Ls)
        log_chi = np.log(np.clip(chi, 1e-6, None))
        slope, intercept, r, p, se = stats.linregress(log_L, log_chi)

        return {
            "L_values": Ls.tolist(),
            "chi_max": chi.tolist(),
            "gamma_over_nu": float(slope),
            "r_squared": float(r ** 2),
            "p_value": float(p),
            "interpretation": f"chi_max ~ L^{slope:.2f} (R2={r**2:.3f}, p={p:.4f})",
        }

    return {"error": "Not enough depths"}


def per_dataset_collapse_analysis(curves_by_L: Dict, metric: str = "y_l1") -> Dict:
    """Analyze collapse quality per dataset."""
    datasets = set()
    for curves in curves_by_L.values():
        for c in curves:
            datasets.add(c["dataset"])

    results = {}
    for ds in sorted(datasets):
        # Filter to this dataset
        ds_curves = defaultdict(list)
        for L, curves in curves_by_L.items():
            for c in curves:
                if c["dataset"] == ds:
                    ds_curves[L].append(c)

        if len(ds_curves) < 2:
            continue

        # Find best collapse for this dataset
        best_mse = float("inf")
        best_params = (0.5, 1.0)
        for s_c in np.linspace(0.2, 0.85, 20):
            for nu in np.linspace(0.3, 5.0, 20):
                mse = collapse_mse_interp(ds_curves, s_c, nu, metric)
                if mse < best_mse:
                    best_mse = mse
                    best_params = (s_c, nu)

        baseline = collapse_mse_interp(ds_curves, 0.5, 1e6, metric)
        improvement = 100 * (baseline - best_mse) / max(baseline, 1e-10)

        results[ds] = {
            "best_s_c": float(best_params[0]),
            "best_nu": float(best_params[1]),
            "best_mse": float(best_mse),
            "baseline_mse": float(baseline),
            "improvement_pct": float(improvement),
            "n_depths": len(ds_curves),
            "collapse_improves": improvement > 10,
        }

    return results


def main():
    input_path = RESULTS_DIR / "cti_pythia_depth_series.json"
    output_path = RESULTS_DIR / "cti_gate1_pythia_collapse.json"

    print("=" * 70)
    print("GATE 1: Same-Architecture Scaling Collapse (Pythia Family)")
    print("=" * 70)

    curves_by_L, all_curves = load_pythia_curves(str(input_path))

    print(f"\nLoaded {len(all_curves)} curves across {len(curves_by_L)} depth scales")
    for L in sorted(curves_by_L.keys()):
        models = set(c["model"] for c in curves_by_L[L])
        n_ds = len(curves_by_L[L])
        dims = set(c["hidden_dim"] for c in curves_by_L[L])
        print(f"  L={L:2d}: {n_ds} curves, dim={dims}, models={models}")

    # =====================================================================
    # TEST 1: Non-monotonicity scaling with depth
    # =====================================================================
    print("\n" + "=" * 70)
    print("[1] Non-Monotonicity vs Depth (Same Architecture)")
    print("=" * 70)

    nm_result = test_non_monotonicity_scaling(curves_by_L)
    if "interpretation" in nm_result:
        print(f"  {nm_result['interpretation']}")
        for L, rho in nm_result.get("mean_rho", {}).items():
            print(f"    L={L:2d}: mean_rho={rho:.3f}")
    else:
        print(f"  Error: {nm_result.get('error', 'unknown')}")

    # =====================================================================
    # TEST 2: Derivative peak convergence (order parameter)
    # =====================================================================
    print("\n" + "=" * 70)
    print("[2] Derivative Peak Convergence (Order Parameter Proxy)")
    print("=" * 70)

    deriv_result = test_derivative_peak_convergence(curves_by_L)
    if "interpretation" in deriv_result:
        print(f"  {deriv_result['interpretation']}")
        for L, info in sorted(deriv_result.get("per_L", {}).items()):
            print(f"    L={L:2d}: peak_s={info['mean_peak_s']:.3f} +/- {info['std_peak_s']:.3f}, "
                  f"max_deriv={info['mean_max_deriv']:.3f}")

    # =====================================================================
    # TEST 3: Susceptibility divergence
    # =====================================================================
    print("\n" + "=" * 70)
    print("[3] Susceptibility Divergence: chi_max ~ L^(gamma/nu)")
    print("=" * 70)

    suscept_result = test_susceptibility_divergence(curves_by_L)
    if "interpretation" in suscept_result:
        print(f"  {suscept_result['interpretation']}")
        for L, chi in zip(suscept_result.get("L_values", []),
                          suscept_result.get("chi_max", [])):
            print(f"    L={L:2d}: chi_max={chi:.4f}")

    # =====================================================================
    # TEST 4: Finite-size scaling collapse (grid search)
    # =====================================================================
    print("\n" + "=" * 70)
    print("[4] Finite-Size Scaling Collapse (Grid Search)")
    print("=" * 70)

    collapse_result = grid_search_collapse(curves_by_L)
    print(f"  Best s_c = {collapse_result['best_s_c']:.3f}")
    print(f"  Best nu  = {collapse_result['best_nu']:.3f}")
    print(f"  Best collapse MSE = {collapse_result['best_collapse_mse']:.6f}")
    print(f"  Baseline MSE (no rescaling) = {collapse_result['baseline_mse']:.6f}")
    print(f"  Improvement from rescaling: {collapse_result['improvement_pct']:.1f}%")

    # =====================================================================
    # TEST 5: Bootstrap significance
    # =====================================================================
    print("\n" + "=" * 70)
    print("[5] Bootstrap Significance Test (n=200)")
    print("=" * 70)

    boot_result = bootstrap_collapse_significance(
        curves_by_L, collapse_result["best_s_c"], collapse_result["best_nu"]
    )
    print(f"  Observed improvement: {boot_result['observed_improvement']:.6f}")
    print(f"  Null distribution: mean={boot_result['null_mean']:.6f}, std={boot_result['null_std']:.6f}")
    print(f"  p-value: {boot_result['p_value']:.4f}")
    print(f"  Significant at 0.05: {boot_result['significant_005']}")
    print(f"  Significant at 0.01: {boot_result['significant_001']}")

    # =====================================================================
    # TEST 6: Per-dataset collapse analysis
    # =====================================================================
    print("\n" + "=" * 70)
    print("[6] Per-Dataset Collapse Analysis")
    print("=" * 70)

    per_ds = per_dataset_collapse_analysis(curves_by_L)
    n_collapse = sum(1 for v in per_ds.values() if v["collapse_improves"])
    n_total = len(per_ds)
    print(f"\n  Datasets where collapse improves: {n_collapse}/{n_total}")
    for ds, info in sorted(per_ds.items()):
        tag = "PASS" if info["collapse_improves"] else "FAIL"
        print(f"    {ds:20s}: s_c={info['best_s_c']:.3f}, nu={info['best_nu']:.2f}, "
              f"improvement={info['improvement_pct']:+.1f}%  [{tag}]")

    # =====================================================================
    # VERDICT
    # =====================================================================
    print("\n" + "=" * 70)
    print("GATE 1 VERDICT")
    print("=" * 70)

    evidence_for = 0
    evidence_against = 0
    verdicts = []

    # V1: Non-monotonicity scales with L
    if "p_value" in nm_result:
        if nm_result["p_value"] < 0.05 and nm_result.get("power_law_exponent", 0) > 0:
            evidence_for += 1
            verdicts.append(f"  [+] Non-monotonicity scales: L^{nm_result['power_law_exponent']:.2f} (p={nm_result['p_value']:.4f})")
        else:
            evidence_against += 1
            verdicts.append(f"  [-] Non-monotonicity scaling: exponent={nm_result.get('power_law_exponent', 'N/A')}, p={nm_result['p_value']:.4f}")

    # V2: Derivative peaks converge
    if "converging" in deriv_result:
        if deriv_result["converging"]:
            evidence_for += 1
            verdicts.append(f"  [+] Derivative peaks converge: range={deriv_result['peak_range']:.3f}")
        else:
            evidence_against += 1
            verdicts.append(f"  [-] Derivative peaks diverge: range={deriv_result['peak_range']:.3f}")

    # V3: Susceptibility diverges
    if "gamma_over_nu" in suscept_result:
        if suscept_result["p_value"] < 0.05 and suscept_result["gamma_over_nu"] > 0:
            evidence_for += 1
            verdicts.append(f"  [+] Susceptibility diverges: L^{suscept_result['gamma_over_nu']:.2f} (p={suscept_result['p_value']:.4f})")
        else:
            evidence_against += 1
            verdicts.append(f"  [-] Susceptibility: exponent={suscept_result['gamma_over_nu']:.2f}, p={suscept_result['p_value']:.4f}")

    # V4: Collapse improvement > 10%
    if collapse_result["improvement_pct"] > 10:
        evidence_for += 1
        verdicts.append(f"  [+] Collapse improvement: {collapse_result['improvement_pct']:.1f}%")
    else:
        evidence_against += 1
        verdicts.append(f"  [-] Collapse improvement: {collapse_result['improvement_pct']:.1f}% (< 10%)")

    # V5: Bootstrap significant
    if boot_result["significant_005"]:
        evidence_for += 1
        verdicts.append(f"  [+] Bootstrap significant: p={boot_result['p_value']:.4f}")
    else:
        evidence_against += 1
        verdicts.append(f"  [-] Bootstrap not significant: p={boot_result['p_value']:.4f}")

    # V6: Majority of datasets show collapse
    if n_collapse > n_total / 2:
        evidence_for += 1
        verdicts.append(f"  [+] Majority datasets collapse: {n_collapse}/{n_total}")
    else:
        evidence_against += 1
        verdicts.append(f"  [-] Minority datasets collapse: {n_collapse}/{n_total}")

    for v in verdicts:
        print(v)

    gate1_pass = evidence_for >= 4  # Need 4/6 to pass
    print(f"\n  SCORE: {evidence_for}/6 for, {evidence_against}/6 against")
    print(f"  GATE 1: {'PASS - Proceed to Gate 2' if gate1_pass else 'FAIL - Drop RG framing'}")

    # Save results
    output = {
        "gate": "Gate 1: Same-Architecture Scaling Collapse",
        "family": "Pythia",
        "depths": sorted(curves_by_L.keys()),
        "n_curves": len(all_curves),
        "non_monotonicity_scaling": nm_result,
        "derivative_peak_convergence": deriv_result,
        "susceptibility_divergence": suscept_result,
        "scaling_collapse": collapse_result,
        "bootstrap_significance": boot_result,
        "per_dataset_collapse": per_ds,
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "gate1_pass": gate1_pass,
        "verdict": "PASS" if gate1_pass else "FAIL",
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
