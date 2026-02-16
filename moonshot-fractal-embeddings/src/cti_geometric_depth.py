#!/usr/bin/env python
"""
cti_geometric_depth.py

Core hypothesis test for "Geometric Control Laws of Representation Dynamics":

  Depth is not an independent axis. Quality peaks depend on the coupling
  between depth, width, and representation geometry. There exists an
  "effective depth" d_eff = f(layer, hidden_dim, L) that predicts quality
  better than raw layer index s = layer/L.

Tests:
  1. Does hidden_dim predict quality peak location?
  2. Does the dim/layer ratio predict peak quality?
  3. Can we build a universal quality predictor from architecture params?
  4. Cross-architecture validation: do geometric features generalize?

Usage:
    python -u src/cti_geometric_depth.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats, optimize

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_all_curves():
    """Load depth-quality curves from both atlas and Pythia data."""
    curves = []

    # Load atlas
    atlas_path = RESULTS_DIR / "cti_atlas_fit.json"
    if atlas_path.exists():
        with open(atlas_path) as f:
            atlas = json.load(f)

        for model_key, model_data in atlas.items():
            if "datasets" not in model_data:
                continue
            num_layers = model_data.get("num_layers", 0)
            hidden_dim = model_data.get("hidden_dim", 0)
            if num_layers == 0 or hidden_dim == 0:
                continue

            for ds_name, ds_data in model_data["datasets"].items():
                layers = ds_data.get("layers", {})
                if not layers:
                    continue

                sorted_layers = sorted(layers.items(), key=lambda x: int(x[0]))
                s = np.array([int(k) / num_layers for k, _ in sorted_layers])
                y_l0 = np.array([v["knn_l0"] for _, v in sorted_layers])
                y_l1 = np.array([v["knn_l1"] for _, v in sorted_layers])

                curves.append({
                    "model": model_key,
                    "dataset": ds_name,
                    "source": "atlas",
                    "s": s,
                    "y_l0": y_l0,
                    "y_l1": y_l1,
                    "L": num_layers,
                    "D": hidden_dim,
                    "params_proxy": num_layers * hidden_dim ** 2,  # rough param count proxy
                })

    # Load Pythia
    pythia_path = RESULTS_DIR / "cti_pythia_depth_series.json"
    if pythia_path.exists():
        with open(pythia_path) as f:
            pythia = json.load(f)

        for model_key, model_data in pythia.items():
            if "datasets" not in model_data:
                continue
            num_layers = model_data.get("num_layers", 0)
            hidden_dim = model_data.get("hidden_dim", 0)
            if num_layers == 0 or hidden_dim == 0:
                continue

            for ds_name, ds_data in model_data["datasets"].items():
                layers = ds_data.get("layers", {})
                if not layers:
                    continue

                sorted_layers = sorted(layers.items(), key=lambda x: int(x[0]))
                s = np.array([v["C_relative"] for _, v in sorted_layers])
                y_l0 = np.array([v["knn_l0"] for _, v in sorted_layers])
                y_l1 = np.array([v["knn_l1"] for _, v in sorted_layers])

                curves.append({
                    "model": model_key,
                    "dataset": ds_name,
                    "source": "pythia",
                    "s": s,
                    "y_l0": y_l0,
                    "y_l1": y_l1,
                    "L": num_layers,
                    "D": hidden_dim,
                    "params_proxy": num_layers * hidden_dim ** 2,
                })

    return curves


def extract_curve_features(curves):
    """Extract per-curve features for regression."""
    records = []
    for c in curves:
        y = c["y_l1"]
        if len(y) < 3:
            continue

        # Peak location and value
        peak_idx = np.argmax(y)
        peak_s = c["s"][peak_idx]
        peak_val = y[peak_idx]
        final_val = y[-1]
        peak_drop = peak_val - final_val  # Non-monotonicity

        # Monotonicity
        rho_mono, _ = stats.spearmanr(np.arange(len(y)), y)

        # Shape features: derivative
        dy = np.gradient(y, c["s"]) if len(y) > 1 else np.zeros_like(y)
        max_deriv = np.max(np.abs(dy))
        max_deriv_s = c["s"][np.argmax(np.abs(dy))]

        # AUC (area under curve, normalized)
        auc = np.trapz(y, c["s"]) if len(y) > 1 else 0.0

        # Architecture features
        L = c["L"]
        D = c["D"]

        records.append({
            "model": c["model"],
            "dataset": c["dataset"],
            "source": c["source"],
            # Architecture
            "L": L,
            "D": D,
            "D_over_L": D / L,
            "log_L": np.log(L),
            "log_D": np.log(D),
            "log_D_over_L": np.log(D / L),
            "aspect_ratio": D / (D + L),
            "params_proxy": c["params_proxy"],
            "log_params": np.log(c["params_proxy"]),
            # Quality features
            "peak_s": peak_s,
            "peak_val": peak_val,
            "final_val": final_val,
            "peak_drop": peak_drop,
            "rho_mono": rho_mono,
            "max_deriv": max_deriv,
            "max_deriv_s": max_deriv_s,
            "auc": auc,
        })

    return records


def test_geometric_predictors(records):
    """Test what predicts peak location and quality."""
    print("\n" + "=" * 70)
    print("[1] What Predicts Peak Quality (peak_val)?")
    print("=" * 70)

    peak_vals = np.array([r["peak_val"] for r in records])
    predictors = {
        "log(D)": np.array([r["log_D"] for r in records]),
        "log(L)": np.array([r["log_L"] for r in records]),
        "log(D/L)": np.array([r["log_D_over_L"] for r in records]),
        "log(params)": np.array([r["log_params"] for r in records]),
        "aspect_ratio": np.array([r["aspect_ratio"] for r in records]),
    }

    results_quality = {}
    for name, x in sorted(predictors.items()):
        rho, p = stats.spearmanr(x, peak_vals)
        r_pearson, p_pearson = stats.pearsonr(x, peak_vals)
        results_quality[name] = {"rho": float(rho), "p": float(p),
                                 "r_pearson": float(r_pearson)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:18s}: rho={rho:+.3f} (p={p:.4f}) {sig}  r={r_pearson:+.3f}")

    # =====================================================================
    print("\n" + "=" * 70)
    print("[2] What Predicts Peak LOCATION (peak_s)?")
    print("=" * 70)

    peak_ss = np.array([r["peak_s"] for r in records])

    results_location = {}
    for name, x in sorted(predictors.items()):
        rho, p = stats.spearmanr(x, peak_ss)
        r_pearson, p_pearson = stats.pearsonr(x, peak_ss)
        results_location[name] = {"rho": float(rho), "p": float(p),
                                  "r_pearson": float(r_pearson)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:18s}: rho={rho:+.3f} (p={p:.4f}) {sig}  r={r_pearson:+.3f}")

    # =====================================================================
    print("\n" + "=" * 70)
    print("[3] What Predicts Non-Monotonicity (peak_drop)?")
    print("=" * 70)

    peak_drops = np.array([r["peak_drop"] for r in records])

    results_nonmono = {}
    for name, x in sorted(predictors.items()):
        rho, p = stats.spearmanr(x, peak_drops)
        r_pearson, p_pearson = stats.pearsonr(x, peak_drops)
        results_nonmono[name] = {"rho": float(rho), "p": float(p),
                                 "r_pearson": float(r_pearson)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:18s}: rho={rho:+.3f} (p={p:.4f}) {sig}  r={r_pearson:+.3f}")

    return results_quality, results_location, results_nonmono


def test_effective_depth(records, curves):
    """Test: is there an effective depth d_eff that universally predicts quality?

    d_eff(l, L, D) = (l/L)^alpha * (D/D_ref)^beta

    where alpha, beta are universal constants.
    """
    print("\n" + "=" * 70)
    print("[4] Effective Depth: d_eff = (l/L)^alpha * (D/D_ref)^beta")
    print("=" * 70)

    # Collect all (s, D, quality) triples
    all_s = []
    all_D = []
    all_L = []
    all_y = []

    for c in curves:
        y = c["y_l1"]
        for i, (si, yi) in enumerate(zip(c["s"], y)):
            if si == 0:
                continue  # skip embedding layer
            all_s.append(si)
            all_D.append(c["D"])
            all_L.append(c["L"])
            all_y.append(yi)

    all_s = np.array(all_s)
    all_D = np.array(all_D)
    all_L = np.array(all_L)
    all_y = np.array(all_y)

    D_ref = np.median(all_D)
    print(f"  D_ref (median) = {D_ref}")

    # Fit: y = a + b * s^alpha * (D/D_ref)^beta
    def model_eff_depth(params, s, D):
        a, b, alpha, beta = params
        d_eff = (s ** alpha) * ((D / D_ref) ** beta)
        return a + b * d_eff

    def residuals(params):
        pred = model_eff_depth(params, all_s, all_D)
        return pred - all_y

    # Multiple random starts
    best_result = None
    best_cost = float("inf")
    for _ in range(20):
        x0 = [0.1, 0.3, np.random.uniform(0.5, 2.0), np.random.uniform(-0.5, 0.5)]
        try:
            result = optimize.least_squares(residuals, x0, method="trf",
                                           bounds=([-1, -1, 0.01, -3], [1, 1, 5, 3]))
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            continue

    if best_result is not None:
        a, b, alpha, beta = best_result.x
        y_pred = model_eff_depth(best_result.x, all_s, all_D)
        ss_res = np.sum((all_y - y_pred) ** 2)
        ss_tot = np.sum((all_y - np.mean(all_y)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((all_y - y_pred) ** 2))

        print(f"  Effective depth: d_eff = s^{alpha:.3f} * (D/D_ref)^{beta:.3f}")
        print(f"  Q(d_eff) = {a:.4f} + {b:.4f} * d_eff")
        print(f"  R2 = {r_squared:.4f}")
        print(f"  RMSE = {rmse:.4f}")

        # Compare to naive s-only model
        def residuals_naive(params):
            a_n, b_n, alpha_n = params
            return a_n + b_n * (all_s ** alpha_n) - all_y

        naive_result = optimize.least_squares(
            residuals_naive, [0.1, 0.3, 1.0], method="trf",
            bounds=([-1, -1, 0.01], [1, 1, 5])
        )
        y_naive = naive_result.x[0] + naive_result.x[1] * (all_s ** naive_result.x[2])
        r2_naive = 1 - np.sum((all_y - y_naive) ** 2) / ss_tot

        print(f"\n  Naive model (s only): R2 = {r2_naive:.4f}")
        print(f"  Geometric model:      R2 = {r_squared:.4f}")
        print(f"  R2 improvement: {r_squared - r2_naive:+.4f}")

        improvement = r_squared - r2_naive
        if improvement > 0.02:
            print(f"  --> GEOMETRY HELPS: +{improvement:.4f} R2 from width information")
        elif improvement > 0:
            print(f"  --> Marginal improvement: +{improvement:.4f}")
        else:
            print(f"  --> No improvement from width information")

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "a": float(a),
            "b": float(b),
            "r_squared": float(r_squared),
            "rmse": float(rmse),
            "r2_naive": float(r2_naive),
            "r2_improvement": float(improvement),
            "D_ref": float(D_ref),
            "n_points": len(all_y),
        }

    return {"error": "Optimization failed"}


def test_per_dataset_universality(records, curves):
    """For each dataset, test if D predicts quality peak across models."""
    print("\n" + "=" * 70)
    print("[5] Per-Dataset: Does Width Predict Quality Across Models?")
    print("=" * 70)

    datasets = set(r["dataset"] for r in records)
    results = {}

    for ds in sorted(datasets):
        ds_records = [r for r in records if r["dataset"] == ds]
        if len(ds_records) < 5:
            continue

        # Test: does log(D) predict peak_val?
        log_D = np.array([r["log_D"] for r in ds_records])
        peak_val = np.array([r["peak_val"] for r in ds_records])
        rho, p = stats.spearmanr(log_D, peak_val)

        # Also test log(params)
        log_p = np.array([r["log_params"] for r in ds_records])
        rho_p, p_p = stats.spearmanr(log_p, peak_val)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {ds:20s}: log(D)->peak: rho={rho:+.3f} {sig}  "
              f"log(params)->peak: rho={rho_p:+.3f} (n={len(ds_records)})")

        results[ds] = {
            "rho_D_peak": float(rho),
            "p_D_peak": float(p),
            "rho_params_peak": float(rho_p),
            "p_params_peak": float(p_p),
            "n": len(ds_records),
        }

    return results


def test_width_depth_decoupling(curves):
    """KEY TEST: After controlling for width (D), does depth (L) still matter?

    If width explains most of the variance, depth is secondary.
    If both matter independently, they're truly coupled.
    """
    print("\n" + "=" * 70)
    print("[6] Width-Depth Decoupling: Partial Correlation Analysis")
    print("=" * 70)

    # Collect model-level features
    model_features = {}
    for c in curves:
        key = f"{c['model']}|{c['dataset']}"
        if key not in model_features:
            model_features[key] = {
                "log_D": np.log(c["D"]),
                "log_L": np.log(c["L"]),
                "log_DL": np.log(c["D"] * c["L"]),
                "peak_val": float(np.max(c["y_l1"])),
                "peak_s": float(c["s"][np.argmax(c["y_l1"])]),
                "peak_drop": float(np.max(c["y_l1"]) - c["y_l1"][-1]),
                "model": c["model"],
            }

    if len(model_features) < 10:
        print("  Not enough model-dataset pairs for partial correlation")
        return {}

    records = list(model_features.values())
    log_D = np.array([r["log_D"] for r in records])
    log_L = np.array([r["log_L"] for r in records])
    peak_val = np.array([r["peak_val"] for r in records])
    peak_drop = np.array([r["peak_drop"] for r in records])

    # Simple correlations
    rho_D_quality, p_D = stats.spearmanr(log_D, peak_val)
    rho_L_quality, p_L = stats.spearmanr(log_L, peak_val)
    rho_D_drop, p_D_drop = stats.spearmanr(log_D, peak_drop)
    rho_L_drop, p_L_drop = stats.spearmanr(log_L, peak_drop)

    print(f"  Simple correlations (n={len(records)}):")
    print(f"    log(D) -> peak quality: rho={rho_D_quality:+.3f} (p={p_D:.4f})")
    print(f"    log(L) -> peak quality: rho={rho_L_quality:+.3f} (p={p_L:.4f})")
    print(f"    log(D) -> non-monotonicity: rho={rho_D_drop:+.3f} (p={p_D_drop:.4f})")
    print(f"    log(L) -> non-monotonicity: rho={rho_L_drop:+.3f} (p={p_L_drop:.4f})")

    # Partial correlations: D controlling for L, and L controlling for D
    def partial_corr(x, y, z):
        """Partial Spearman correlation of x,y controlling for z."""
        # Rank-based partial correlation
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        rz = stats.rankdata(z)

        # Residualize
        slope_xz = np.polyfit(rz, rx, 1)
        slope_yz = np.polyfit(rz, ry, 1)
        rx_resid = rx - np.polyval(slope_xz, rz)
        ry_resid = ry - np.polyval(slope_yz, rz)

        r, p = stats.pearsonr(rx_resid, ry_resid)
        return float(r), float(p)

    rho_D_partial, p_D_partial = partial_corr(log_D, peak_val, log_L)
    rho_L_partial, p_L_partial = partial_corr(log_L, peak_val, log_D)
    rho_D_drop_partial, p_D_drop_partial = partial_corr(log_D, peak_drop, log_L)
    rho_L_drop_partial, p_L_drop_partial = partial_corr(log_L, peak_drop, log_D)

    print(f"\n  Partial correlations (controlling for the other):")
    print(f"    log(D) -> peak quality | L: rho={rho_D_partial:+.3f} (p={p_D_partial:.4f})")
    print(f"    log(L) -> peak quality | D: rho={rho_L_partial:+.3f} (p={p_L_partial:.4f})")
    print(f"    log(D) -> non-mono | L:     rho={rho_D_drop_partial:+.3f} (p={p_D_drop_partial:.4f})")
    print(f"    log(L) -> non-mono | D:     rho={rho_L_drop_partial:+.3f} (p={p_L_drop_partial:.4f})")

    # Interpretation
    D_dominates_quality = abs(rho_D_partial) > abs(rho_L_partial) and p_D_partial < 0.05
    L_dominates_quality = abs(rho_L_partial) > abs(rho_D_partial) and p_L_partial < 0.05
    both_matter = p_D_partial < 0.05 and p_L_partial < 0.05

    print(f"\n  Interpretation:")
    if both_matter:
        print(f"    BOTH D and L independently predict quality -> truly COUPLED")
    elif D_dominates_quality:
        print(f"    WIDTH (D) dominates -> depth is secondary to geometry")
    elif L_dominates_quality:
        print(f"    DEPTH (L) dominates -> traditional view holds")
    else:
        print(f"    NEITHER is significant alone -> complex interaction")

    return {
        "simple": {
            "rho_D_quality": float(rho_D_quality),
            "rho_L_quality": float(rho_L_quality),
            "rho_D_drop": float(rho_D_drop),
            "rho_L_drop": float(rho_L_drop),
        },
        "partial": {
            "rho_D_quality_given_L": float(rho_D_partial),
            "p_D_quality_given_L": float(p_D_partial),
            "rho_L_quality_given_D": float(rho_L_partial),
            "p_L_quality_given_D": float(p_L_partial),
            "rho_D_drop_given_L": float(rho_D_drop_partial),
            "rho_L_drop_given_D": float(rho_L_drop_partial),
        },
        "interpretation": ("coupled" if both_matter
                          else "width_dominates" if D_dominates_quality
                          else "depth_dominates" if L_dominates_quality
                          else "complex"),
        "n": len(records),
    }


def test_quality_from_capacity(curves):
    """Test: Does quality scale with capacity = D * L (total representational volume)?

    If quality ~ f(D*L), then depth and width are INTERCHANGEABLE for capacity.
    If quality depends on D/L ratio, then SHAPE matters, not just total capacity.
    """
    print("\n" + "=" * 70)
    print("[7] Capacity vs Shape: What Matters for Quality?")
    print("=" * 70)

    model_data = defaultdict(list)
    for c in curves:
        peak_val = float(np.max(c["y_l1"]))
        model_data[c["model"]].append(peak_val)

    # Average peak quality per model
    models = []
    for c in curves:
        m = c["model"]
        if m not in [x["model"] for x in models]:
            models.append({
                "model": m,
                "D": c["D"],
                "L": c["L"],
                "DL": c["D"] * c["L"],
                "D_over_L": c["D"] / c["L"],
                "mean_peak": np.mean(model_data[m]),
            })

    if len(models) < 5:
        print("  Not enough unique models")
        return {}

    DL = np.array([m["DL"] for m in models])
    D_over_L = np.array([m["D_over_L"] for m in models])
    peak = np.array([m["mean_peak"] for m in models])

    rho_capacity, p_cap = stats.spearmanr(np.log(DL), peak)
    rho_shape, p_shape = stats.spearmanr(np.log(D_over_L), peak)

    print(f"  Unique models: {len(models)}")
    for m in sorted(models, key=lambda x: x["DL"]):
        print(f"    {m['model']:25s}: D={m['D']:5d}, L={m['L']:3d}, "
              f"D*L={m['DL']:8d}, D/L={m['D_over_L']:6.1f}, peak={m['mean_peak']:.3f}")

    print(f"\n  log(D*L) -> peak quality: rho={rho_capacity:+.3f} (p={p_cap:.4f})")
    print(f"  log(D/L) -> peak quality: rho={rho_shape:+.3f} (p={p_shape:.4f})")

    if abs(rho_capacity) > abs(rho_shape):
        print(f"  --> CAPACITY (D*L) matters more than SHAPE (D/L)")
    else:
        print(f"  --> SHAPE (D/L ratio) matters more than raw CAPACITY")

    return {
        "rho_capacity": float(rho_capacity),
        "p_capacity": float(p_cap),
        "rho_shape": float(rho_shape),
        "p_shape": float(p_shape),
        "models": [{k: float(v) if isinstance(v, (int, float, np.floating, np.integer))
                     else v for k, v in m.items()} for m in models],
    }


def main():
    print("=" * 70)
    print("GEOMETRIC CONTROL LAWS: Effective Depth Hypothesis Test")
    print("=" * 70)

    curves = load_all_curves()
    print(f"\nLoaded {len(curves)} curves")
    models = set(c["model"] for c in curves)
    print(f"Models: {sorted(models)}")
    print(f"Depth range: {sorted(set(c['L'] for c in curves))}")
    print(f"Width range: {sorted(set(c['D'] for c in curves))}")

    records = extract_curve_features(curves)
    print(f"Extracted features for {len(records)} curves")

    # Run all tests
    results = {}

    r_quality, r_location, r_nonmono = test_geometric_predictors(records)
    results["predictors_quality"] = r_quality
    results["predictors_location"] = r_location
    results["predictors_nonmono"] = r_nonmono

    results["effective_depth"] = test_effective_depth(records, curves)

    results["per_dataset"] = test_per_dataset_universality(records, curves)

    results["decoupling"] = test_width_depth_decoupling(curves)

    results["capacity_vs_shape"] = test_quality_from_capacity(curves)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Geometric Control Laws")
    print("=" * 70)

    findings = []

    # 1. Does geometry improve prediction?
    eff = results.get("effective_depth", {})
    if eff.get("r2_improvement", 0) > 0.02:
        findings.append(f"[+] Geometry improves quality prediction: R2 +{eff['r2_improvement']:.4f}")
    else:
        findings.append(f"[-] Geometry does not improve prediction: R2 +{eff.get('r2_improvement', 0):.4f}")

    # 2. Width vs depth
    dec = results.get("decoupling", {})
    interp = dec.get("interpretation", "unknown")
    findings.append(f"[=] Width-depth relationship: {interp}")

    # 3. Capacity vs shape
    cs = results.get("capacity_vs_shape", {})
    if abs(cs.get("rho_shape", 0)) > abs(cs.get("rho_capacity", 0)):
        findings.append(f"[+] Shape (D/L) matters more than capacity (D*L)")
    else:
        findings.append(f"[-] Capacity (D*L) matters more than shape (D/L)")

    for f in findings:
        print(f"  {f}")

    # Save
    output_path = RESULTS_DIR / "cti_geometric_depth.json"
    with open(output_path, "w") as f_out:
        json.dump(results, f_out, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
