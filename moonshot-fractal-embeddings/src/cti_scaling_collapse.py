#!/usr/bin/env python
"""
cti_scaling_collapse.py

Test finite-size scaling collapse prediction from RG theory.

Theory predicts: Q_L(s) = Q* + L^(-x_Q) * Phi((s - s_c) * L^(1/nu))
where Phi is UNIVERSAL across architectures.

If curves at different L collapse onto one function when rescaled,
this is strong evidence for the depth-criticality universality theorem.

Usage:
    python -u src/cti_scaling_collapse.py --input results/cti_atlas_fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import interpolate, optimize, stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def extract_curves(atlas: Dict) -> Dict[str, Dict]:
    """Extract normalized depth-quality curves grouped by L."""
    curves_by_L = defaultdict(list)
    all_curves = []

    for model_key, model_data in atlas.items():
        if "datasets" not in model_data:
            continue
        num_layers = model_data.get("num_layers", 0)
        if num_layers == 0:
            continue

        for ds_name, ds_data in model_data["datasets"].items():
            layers = ds_data.get("layers", {})
            if not layers:
                continue

            sorted_layers = sorted(layers.items(), key=lambda x: int(x[0]))

            # Normalized depth s = layer / L
            s = np.array([int(k) / num_layers for k, v in sorted_layers])
            y = np.array([v["knn_l1"] for _, v in sorted_layers])

            # Skip degenerate embedding layer (s=0)
            if len(s) > 2 and y[0] < 0.05:
                s = s[1:]
                y = y[1:]

            curves_by_L[num_layers].append({
                "model": model_key,
                "dataset": ds_name,
                "s": s,
                "y": y,
                "L": num_layers,
            })
            all_curves.append(curves_by_L[num_layers][-1])

    return curves_by_L, all_curves


def collapse_quality(curves: List[Dict], s_c: float, nu: float, x_Q: float) -> float:
    """Compute collapse quality for given parameters.

    Rescale each curve: x = (s - s_c) * L^(1/nu)
    Then measure how well all curves overlap.
    Lower = better collapse.
    """
    # Collect all rescaled points
    all_x = []
    all_y = []
    all_L = []

    for curve in curves:
        s, y, L = curve["s"], curve["y"], curve["L"]
        x_rescaled = (s - s_c) * (L ** (1.0 / nu))

        # Normalize y: subtract Q* approximation and scale by L^(-x_Q)
        y_rescaled = (y - np.mean(y)) * (L ** x_Q)

        all_x.extend(x_rescaled.tolist())
        all_y.extend(y_rescaled.tolist())
        all_L.extend([L] * len(x_rescaled))

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    if len(all_x) < 10:
        return float("inf")

    # Sort by x
    idx = np.argsort(all_x)
    all_x = all_x[idx]
    all_y = all_y[idx]

    # Compute collapse metric: variance of y in sliding windows
    window_size = max(5, len(all_x) // 30)
    variances = []
    for i in range(0, len(all_x) - window_size, window_size // 2):
        window = all_y[i:i + window_size]
        if len(window) > 2:
            variances.append(np.var(window))

    if not variances:
        return float("inf")

    return float(np.mean(variances))


def collapse_quality_interp(curves_by_L: Dict[int, List], s_c: float, nu: float) -> float:
    """Alternative collapse metric using interpolation.

    For each pair of depth scales (L1, L2), interpolate curves to
    common rescaled grid and compute MSE.
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

                    s1, y1 = c1["s"], c1["y"]
                    s2, y2 = c2["s"], c2["y"]

                    # Rescale
                    x1 = (s1 - s_c) * (L1 ** (1.0 / nu))
                    x2 = (s2 - s_c) * (L2 ** (1.0 / nu))

                    # Normalize y to [0,1]
                    y1_n = (y1 - y1.min()) / max(y1.max() - y1.min(), 1e-8)
                    y2_n = (y2 - y2.min()) / max(y2.max() - y2.min(), 1e-8)

                    # Find overlap region
                    x_min = max(x1.min(), x2.min())
                    x_max = min(x1.max(), x2.max())

                    if x_max <= x_min:
                        continue

                    # Interpolate to common grid
                    n_grid = 20
                    x_common = np.linspace(x_min, x_max, n_grid)

                    try:
                        f1 = interpolate.interp1d(x1, y1_n, kind="linear", fill_value="extrapolate")
                        f2 = interpolate.interp1d(x2, y2_n, kind="linear", fill_value="extrapolate")
                        y1_interp = f1(x_common)
                        y2_interp = f2(x_common)
                        mse = np.mean((y1_interp - y2_interp) ** 2)
                        total_mse += mse
                        n_pairs += 1
                    except Exception:
                        continue

    if n_pairs == 0:
        return float("inf")

    return total_mse / n_pairs


def grid_search_collapse(curves_by_L: Dict[int, List], all_curves: List[Dict]) -> Dict:
    """Grid search over (s_c, nu) to find best collapse."""
    s_c_range = np.linspace(0.3, 0.9, 25)
    nu_range = np.linspace(0.3, 3.0, 25)

    best_quality = float("inf")
    best_params = (0.5, 1.0)
    results_grid = []

    print("  Grid search: 625 parameter combinations...")
    for s_c in s_c_range:
        for nu in nu_range:
            q = collapse_quality_interp(curves_by_L, s_c, nu)
            results_grid.append((s_c, nu, q))
            if q < best_quality:
                best_quality = q
                best_params = (s_c, nu)

    return {
        "best_s_c": float(best_params[0]),
        "best_nu": float(best_params[1]),
        "best_collapse_mse": float(best_quality),
        "grid": [(float(s), float(n), float(q)) for s, n, q in results_grid],
    }


def test_non_monotonicity_scaling(curves_by_L: Dict[int, List]) -> Dict:
    """Test: does non-monotonicity scale with L?

    Theory predicts: fluctuation amplitude ~ L^(gamma/nu)
    Test: compute non-monotonicity metric for each L, fit power law.
    """
    L_values = sorted(curves_by_L.keys())
    L_metrics = {}

    for L in L_values:
        rhos = []
        peak_drops = []
        for curve in curves_by_L[L]:
            s, y = curve["s"], curve["y"]
            if len(y) < 4:
                continue
            # Monotonicity
            rho, _ = stats.spearmanr(np.arange(len(y)), y)
            rhos.append(rho)
            # Peak vs final drop
            peak_vs_final = max(y) - y[-1]
            peak_drops.append(peak_vs_final)

        if rhos:
            L_metrics[L] = {
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "mean_peak_drop": float(np.mean(peak_drops)),
                "n_curves": len(rhos),
            }

    # Fit: non-monotonicity ~ L^alpha
    if len(L_metrics) >= 3:
        Ls = np.array(sorted(L_metrics.keys()))
        non_mono = np.array([1 - L_metrics[L]["mean_rho"] for L in Ls])
        peak_drops = np.array([L_metrics[L]["mean_peak_drop"] for L in Ls])

        # Log-log fit
        log_L = np.log(Ls)
        log_nm = np.log(np.clip(non_mono, 1e-6, None))
        slope, intercept, r, p, se = stats.linregress(log_L, log_nm)

        return {
            "L_values": Ls.tolist(),
            "non_monotonicity": non_mono.tolist(),
            "peak_drops": peak_drops.tolist(),
            "power_law_exponent": float(slope),
            "r_squared": float(r ** 2),
            "p_value": float(p),
            "interpretation": f"Non-monotonicity ~ L^{slope:.2f} (R2={r**2:.3f}, p={p:.4f})",
            "per_L": L_metrics,
        }

    return {"error": "Not enough depth scales", "per_L": L_metrics}


def test_family_scaling(curves_by_L: Dict[int, List]) -> Dict:
    """Test non-monotonicity scaling within model families."""
    # Group curves by family
    family_map = {
        "bge-small": ("bge", 12), "bge-base": ("bge", 12), "bge-large": ("bge", 24),
        "e5-small": ("e5", 12), "e5-base": ("e5", 12), "e5-large": ("e5", 24),
    }

    family_data = defaultdict(list)
    for L, curves in curves_by_L.items():
        for curve in curves:
            model = curve["model"]
            if model in family_map:
                family, _ = family_map[model]
                rho, _ = stats.spearmanr(np.arange(len(curve["y"])), curve["y"])
                # Use hidden dim as "size" proxy
                size_map = {"bge-small": 384, "bge-base": 768, "bge-large": 1024,
                           "e5-small": 384, "e5-base": 768, "e5-large": 1024}
                family_data[family].append({
                    "model": model,
                    "L": L,
                    "rho": rho,
                    "size": size_map.get(model, 0),
                    "dataset": curve["dataset"],
                })

    results = {}
    for family, data in family_data.items():
        sizes = np.array([d["size"] for d in data])
        rhos = np.array([d["rho"] for d in data])
        unique_sizes = sorted(set(sizes))

        if len(unique_sizes) >= 3:
            mean_rhos = [np.mean(rhos[sizes == s]) for s in unique_sizes]
            slope, intercept, r, p, se = stats.linregress(
                np.log(unique_sizes), np.log(np.clip(1 - np.array(mean_rhos), 1e-6, None))
            )
            results[family] = {
                "sizes": unique_sizes,
                "mean_rho": [float(r) for r in mean_rhos],
                "non_monotonicity_exponent": float(slope),
                "r_squared": float(r ** 2),
                "p_value": float(p),
            }

    return results


def compute_order_parameter_proxy(curves_by_L: Dict[int, List]) -> Dict:
    """Compute proxy order parameter: derivative of quality curve.

    Near a phase transition, the derivative should peak (susceptibility divergence).
    Check if peak location is consistent across depths.
    """
    results = {}

    for L, curves in sorted(curves_by_L.items()):
        deriv_peaks = []
        deriv_maxvals = []
        for curve in curves:
            s, y = curve["s"], curve["y"]
            if len(y) < 5:
                continue
            # Numerical derivative
            dy = np.gradient(y, s)
            # Find peak of |derivative| (susceptibility proxy)
            peak_idx = np.argmax(np.abs(dy))
            deriv_peaks.append(float(s[peak_idx]))
            deriv_maxvals.append(float(np.abs(dy[peak_idx])))

        if deriv_peaks:
            results[L] = {
                "mean_deriv_peak_s": float(np.mean(deriv_peaks)),
                "std_deriv_peak_s": float(np.std(deriv_peaks)),
                "mean_max_deriv": float(np.mean(deriv_maxvals)),
                "n_curves": len(deriv_peaks),
            }

    # Check if peak location is stable across L
    if len(results) >= 2:
        Ls = sorted(results.keys())
        peaks = [results[L]["mean_deriv_peak_s"] for L in Ls]
        print(f"  Derivative peak locations: {dict(zip(Ls, [f'{p:.3f}' for p in peaks]))}")

        # If peaks converge to same s_c, that's evidence for a fixed critical point
        peak_range = max(peaks) - min(peaks)
        print(f"  Peak range across depths: {peak_range:.3f}")
        print(f"  {'CONVERGING' if peak_range < 0.15 else 'DIVERGING'}: {'consistent s_c' if peak_range < 0.15 else 'depth-dependent s_c'}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RESULTS_DIR / "cti_atlas_fit.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "cti_scaling_collapse.json"))
    args = parser.parse_args()

    with open(args.input) as f:
        atlas = json.load(f)

    curves_by_L, all_curves = extract_curves(atlas)
    print(f"Loaded {len(all_curves)} curves across depth scales: {sorted(curves_by_L.keys())}")
    for L in sorted(curves_by_L.keys()):
        models = set(c["model"] for c in curves_by_L[L])
        print(f"  L={L}: {len(curves_by_L[L])} curves from {len(models)} models ({', '.join(sorted(models))})")

    # ── 1. Non-monotonicity scaling with L ───────────────────────────
    print("\n" + "=" * 70)
    print("[1] Non-Monotonicity vs Depth (L)")
    print("=" * 70)
    nm_scaling = test_non_monotonicity_scaling(curves_by_L)
    if "interpretation" in nm_scaling:
        print(f"\n  {nm_scaling['interpretation']}")
        for L in nm_scaling["L_values"]:
            info = nm_scaling["per_L"][L]
            print(f"    L={L:2d}: rho={info['mean_rho']:.3f} +/- {info['std_rho']:.3f}, peak_drop={info['mean_peak_drop']:.4f}, n={info['n_curves']}")

    # ── 2. Within-family scaling ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("[2] Within-Family Scaling (BGE, E5)")
    print("=" * 70)
    family_scaling = test_family_scaling(curves_by_L)
    for family, data in family_scaling.items():
        print(f"\n  {family}: exponent={data['non_monotonicity_exponent']:.2f}, R2={data['r_squared']:.3f}")
        for s, r in zip(data["sizes"], data["mean_rho"]):
            print(f"    dim={s}: mean_rho={r:.3f}")

    # ── 3. Order parameter proxy (derivative peak) ───────────────────
    print("\n" + "=" * 70)
    print("[3] Order Parameter Proxy: Derivative Peak Location")
    print("=" * 70)
    order_param = compute_order_parameter_proxy(curves_by_L)

    # ── 4. Finite-size scaling collapse ──────────────────────────────
    print("\n" + "=" * 70)
    print("[4] Finite-Size Scaling Collapse (grid search)")
    print("=" * 70)
    collapse = grid_search_collapse(curves_by_L, all_curves)
    print(f"\n  Best s_c = {collapse['best_s_c']:.3f}")
    print(f"  Best nu  = {collapse['best_nu']:.3f}")
    print(f"  Best collapse MSE = {collapse['best_collapse_mse']:.6f}")

    # Compare to no-rescaling baseline
    baseline_mse = collapse_quality_interp(curves_by_L, 0.5, 1e10)  # nu=inf -> no rescaling
    if baseline_mse < float("inf") and collapse['best_collapse_mse'] < float("inf"):
        improvement = (baseline_mse - collapse['best_collapse_mse']) / max(baseline_mse, 1e-10)
        print(f"  Baseline MSE (no rescaling): {baseline_mse:.6f}")
        print(f"  Improvement from rescaling: {100*improvement:.1f}%")

    # ── 5. Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Evidence for Depth-Criticality")
    print("=" * 70)

    evidence_for = 0
    evidence_against = 0

    # Check non-monotonicity scales with L
    if "power_law_exponent" in nm_scaling:
        if nm_scaling["p_value"] < 0.05:
            print(f"  [+] Non-monotonicity scales with depth: L^{nm_scaling['power_law_exponent']:.2f} (p={nm_scaling['p_value']:.4f})")
            evidence_for += 1
        else:
            print(f"  [-] Non-monotonicity vs depth: not significant (p={nm_scaling['p_value']:.4f})")
            evidence_against += 1

    # Check derivative peaks converge
    if len(order_param) >= 2:
        Ls = sorted(order_param.keys())
        peaks = [order_param[L]["mean_deriv_peak_s"] for L in Ls]
        peak_range = max(peaks) - min(peaks)
        if peak_range < 0.15:
            print(f"  [+] Derivative peaks converge: range={peak_range:.3f} (consistent s_c)")
            evidence_for += 1
        else:
            print(f"  [-] Derivative peaks diverge: range={peak_range:.3f}")
            evidence_against += 1

    # Check collapse improvement
    if baseline_mse < float("inf") and collapse['best_collapse_mse'] < float("inf"):
        if improvement > 0.1:
            print(f"  [+] Scaling collapse improves fit by {100*improvement:.1f}%")
            evidence_for += 1
        else:
            print(f"  [-] Scaling collapse shows minimal improvement ({100*improvement:.1f}%)")
            evidence_against += 1

    print(f"\n  Score: {evidence_for} for / {evidence_against} against depth-criticality")

    # Save
    results = {
        "non_monotonicity_scaling": {k: v for k, v in nm_scaling.items() if k != "per_L"},
        "family_scaling": family_scaling,
        "order_parameter_proxy": {str(k): v for k, v in order_param.items()},
        "collapse": {
            "best_s_c": collapse["best_s_c"],
            "best_nu": collapse["best_nu"],
            "best_collapse_mse": collapse["best_collapse_mse"],
        },
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
    }

    # Don't save full grid (too big)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
