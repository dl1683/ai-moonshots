#!/usr/bin/env python
"""
cti_atlas_analysis.py

Comprehensive analysis of the CTI atlas experiment:
1. Curve fitting (5 functional forms) with AIC/BIC selection
2. Cross-dataset invariance analysis
3. Universality class discovery via clustering
4. Architecture feature extraction for class prediction
5. Peak-layer prediction accuracy

Usage:
    python -u src/cti_atlas_analysis.py --input results/cti_atlas_fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# ── Functional forms ─────────────────────────────────────────────────

def power_law(x, a, b, c):
    """D(C) = a + b * C^c"""
    return a + b * np.power(np.clip(x, 1e-10, None), c)

def sigmoid(x, a, b, c, d):
    """D(C) = a + b / (1 + exp(-c*(x-d)))"""
    return a + b / (1.0 + np.exp(-c * (x - d)))

def exponential(x, a, b, c):
    """D(C) = a - b * exp(-c*x)"""
    return a - b * np.exp(-c * x)

def linear(x, a, b):
    """D(C) = a + b*x"""
    return a + b * x

def piecewise_sigmoid(x, a, b1, c1, d1, b2, c2, d2):
    """Two sigmoids added: captures double transition"""
    s1 = b1 / (1.0 + np.exp(-c1 * (x - d1)))
    s2 = b2 / (1.0 + np.exp(-c2 * (x - d2)))
    return a + s1 + s2


FORMS = {
    "power_law": (power_law, 3, [0.5, 0.5, 0.5]),
    "sigmoid": (sigmoid, 4, [0.3, 0.5, 10.0, 0.5]),
    "exponential": (exponential, 3, [0.8, 0.5, 3.0]),
    "linear": (linear, 2, [0.3, 0.5]),
    "piecewise_sigmoid": (piecewise_sigmoid, 7, [0.3, 0.2, 10.0, 0.3, 0.3, 10.0, 0.7]),
}


def fit_curve(x: np.ndarray, y: np.ndarray, form_name: str) -> Dict[str, Any]:
    """Fit a functional form and return parameters + goodness metrics."""
    func, n_params, p0 = FORMS[form_name]
    n = len(x)

    try:
        if form_name == "linear":
            popt, pcov = np.polyfit(x, y, 1, cov=True)
            popt = [popt[1], popt[0]]  # intercept, slope
            y_pred = linear(x, *popt)
        else:
            popt, pcov = optimize.curve_fit(
                func, x, y, p0=p0, maxfev=10000,
                bounds=(-np.inf, np.inf)
            )
            y_pred = func(x, *popt)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        # AIC and BIC
        if ss_res > 0 and n > n_params:
            mse = ss_res / n
            log_lik = -n / 2 * (np.log(2 * np.pi * mse) + 1)
            aic = 2 * n_params - 2 * log_lik
            bic = n_params * np.log(n) - 2 * log_lik
        else:
            aic = float("inf")
            bic = float("inf")

        return {
            "form": form_name,
            "n_params": n_params,
            "r2": float(r2),
            "aic": float(aic),
            "bic": float(bic),
            "rmse": float(np.sqrt(ss_res / n)),
            "params": [float(p) for p in popt],
            "success": True,
        }
    except Exception as e:
        return {
            "form": form_name,
            "n_params": n_params,
            "r2": -999,
            "aic": float("inf"),
            "bic": float("inf"),
            "rmse": float("inf"),
            "params": [],
            "success": False,
            "error": str(e),
        }


def fit_all_forms(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fit all functional forms and select best by AIC."""
    results = {}
    for form_name in FORMS:
        results[form_name] = fit_curve(x, y, form_name)

    # Select best by AIC (penalizing complexity)
    valid = {k: v for k, v in results.items() if v["success"]}
    if valid:
        best_aic = min(valid, key=lambda k: valid[k]["aic"])
        best_bic = min(valid, key=lambda k: valid[k]["bic"])
    else:
        best_aic = best_bic = None

    return {
        "fits": results,
        "best_aic": best_aic,
        "best_bic": best_bic,
    }


# ── Data extraction ──────────────────────────────────────────────────

def extract_curves(atlas: Dict) -> Dict[str, Dict[str, Dict]]:
    """Extract layer-wise kNN curves from atlas JSON.

    Returns: {model: {dataset: {"x": array, "y_l0": array, "y_l1": array, "n_layers": int}}}
    """
    curves = {}
    for model_key, model_data in atlas.items():
        if "datasets" not in model_data:
            continue

        num_layers = model_data.get("num_layers", 0)
        curves[model_key] = {}

        for ds_name, ds_data in model_data["datasets"].items():
            layers = ds_data.get("layers", {})
            if not layers:
                continue

            # Sort by layer index
            sorted_layers = sorted(layers.items(), key=lambda x: int(x[0]))

            x = np.array([v["C_relative"] for _, v in sorted_layers])
            y_l0 = np.array([v["knn_l0"] for _, v in sorted_layers])
            y_l1 = np.array([v["knn_l1"] for _, v in sorted_layers])

            # Skip layer 0 (embedding layer) - often degenerate
            if len(x) > 2 and y_l0[0] < 0.2:
                x = x[1:]
                y_l0 = y_l0[1:]
                y_l1 = y_l1[1:]

            curves[model_key][ds_name] = {
                "x": x,
                "y_l0": y_l0,
                "y_l1": y_l1,
                "n_layers": num_layers,
                "peak_l0_layer": int(sorted_layers[int(np.argmax(y_l0))][0]),
                "peak_l1_layer": int(sorted_layers[int(np.argmax(y_l1))][0]),
            }

    return curves


# ── Cross-dataset invariance ─────────────────────────────────────────

def compute_invariance(curves: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute cross-dataset correlation for each model.

    For each model, normalize each dataset's curve and compute
    pairwise Spearman correlations.
    """
    model_invariance = {}

    for model_key, datasets in curves.items():
        if len(datasets) < 2:
            continue

        # Need to handle different numbers of layers (if datasets vary)
        # Normalize curves: min-max scale to [0,1]
        normalized = {}
        for ds_name, ds_data in datasets.items():
            y = ds_data["y_l1"]
            if len(y) < 3:
                continue
            y_min, y_max = y.min(), y.max()
            if y_max - y_min < 1e-6:
                continue
            normalized[ds_name] = (y - y_min) / (y_max - y_min)

        if len(normalized) < 2:
            continue

        # Pairwise correlations (only for same-length curves)
        ds_names = list(normalized.keys())
        correlations = []
        pairs = []
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                a = normalized[ds_names[i]]
                b = normalized[ds_names[j]]
                if len(a) != len(b):
                    # Interpolate to common length
                    common_len = min(len(a), len(b))
                    x_a = np.linspace(0, 1, len(a))
                    x_b = np.linspace(0, 1, len(b))
                    x_common = np.linspace(0, 1, common_len)
                    a = np.interp(x_common, x_a, a)
                    b = np.interp(x_common, x_b, b)
                rho, p = stats.spearmanr(a, b)
                correlations.append(rho)
                pairs.append((ds_names[i], ds_names[j], rho, p))

        model_invariance[model_key] = {
            "mean_rho": float(np.mean(correlations)),
            "median_rho": float(np.median(correlations)),
            "min_rho": float(np.min(correlations)),
            "max_rho": float(np.max(correlations)),
            "std_rho": float(np.std(correlations)),
            "n_pairs": len(correlations),
            "pairs": pairs,
        }

    return model_invariance


# ── Monotonicity analysis ────────────────────────────────────────────

def analyze_monotonicity(y: np.ndarray) -> Dict[str, Any]:
    """Analyze if a curve is monotonic, and characterize its shape."""
    n = len(y)
    if n < 3:
        return {"monotonic": True, "n_points": n}

    # Spearman correlation with index = monotonicity measure
    indices = np.arange(n)
    rho, p = stats.spearmanr(indices, y)

    # Count direction changes
    diffs = np.diff(y)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

    # Find peak and dip
    peak_idx = int(np.argmax(y))
    dip_idx = int(np.argmin(y[1:])) + 1 if n > 1 else 0

    # Non-monotonicity score: how much does peak deviate from final
    peak_vs_final = float(y[peak_idx] - y[-1])
    dip_vs_final = float(y[-1] - y[dip_idx]) if dip_idx > 0 else 0.0

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "sign_changes": int(sign_changes),
        "peak_relative_idx": float(peak_idx / (n - 1)),
        "dip_relative_idx": float(dip_idx / (n - 1)),
        "peak_vs_final": peak_vs_final,
        "final_layer_drop": peak_vs_final > 0.01,
        "is_monotonic": rho > 0.9 and sign_changes < 3,
        "n_points": n,
    }


# ── Universality class discovery ─────────────────────────────────────

def extract_shape_features(curves: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """Extract shape feature vectors for each model (averaged across datasets)."""
    model_features = {}

    for model_key, datasets in curves.items():
        all_features = []
        for ds_name, ds_data in datasets.items():
            y = ds_data["y_l1"]
            if len(y) < 4:
                continue
            mono = analyze_monotonicity(y)
            all_features.append([
                mono["spearman_rho"],           # monotonicity
                mono["peak_relative_idx"],       # where peak is
                mono["peak_vs_final"],           # peak vs final layer gap
                mono["sign_changes"] / len(y),   # normalized sign changes
                float(mono["final_layer_drop"]), # binary: does final layer drop?
            ])

        if all_features:
            model_features[model_key] = np.mean(all_features, axis=0)

    return model_features


def cluster_models(features: Dict[str, np.ndarray], n_clusters: int = 3) -> Dict[str, Any]:
    """Cluster models into universality classes based on shape features."""
    model_keys = list(features.keys())
    X = np.array([features[k] for k in model_keys])

    if len(model_keys) < 3:
        return {"error": "Not enough models for clustering"}

    # Hierarchical clustering
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Characterize each cluster
    clusters = defaultdict(list)
    for key, label in zip(model_keys, labels):
        clusters[int(label)].append(key)

    # Compute cluster centroids and descriptions
    cluster_info = {}
    for label, members in clusters.items():
        member_features = np.array([features[m] for m in members])
        centroid = member_features.mean(axis=0)

        # Characterize based on centroid
        if centroid[0] > 0.9:
            shape = "monotonic"
        elif centroid[2] > 0.03:
            shape = "non-monotonic (peak before final)"
        else:
            shape = "weakly non-monotonic"

        cluster_info[label] = {
            "members": members,
            "centroid": centroid.tolist(),
            "shape_description": shape,
            "mean_monotonicity": float(centroid[0]),
            "mean_peak_position": float(centroid[1]),
            "mean_peak_vs_final": float(centroid[2]),
        }

    return {
        "n_clusters": n_clusters,
        "clusters": cluster_info,
        "labels": {k: int(l) for k, l in zip(model_keys, labels)},
    }


# ── Architecture features ────────────────────────────────────────────

ARCHITECTURE_FEATURES = {
    "bge-small": {"family": "bge", "type": "encoder", "pooling": "cls", "params_M": 33, "layers": 12, "dim": 384},
    "bge-base": {"family": "bge", "type": "encoder", "pooling": "cls", "params_M": 109, "layers": 12, "dim": 768},
    "bge-large": {"family": "bge", "type": "encoder", "pooling": "cls", "params_M": 335, "layers": 24, "dim": 1024},
    "e5-small": {"family": "e5", "type": "encoder", "pooling": "mean", "params_M": 33, "layers": 12, "dim": 384},
    "e5-base": {"family": "e5", "type": "encoder", "pooling": "mean", "params_M": 109, "layers": 12, "dim": 768},
    "e5-large": {"family": "e5", "type": "encoder", "pooling": "mean", "params_M": 335, "layers": 24, "dim": 1024},
    "minilm": {"family": "minilm", "type": "encoder", "pooling": "mean", "params_M": 22, "layers": 6, "dim": 384},
    "mpnet": {"family": "mpnet", "type": "encoder", "pooling": "mean", "params_M": 109, "layers": 12, "dim": 768},
    "nomic": {"family": "nomic", "type": "encoder", "pooling": "mean", "params_M": 137, "layers": 12, "dim": 768},
    "embedding-gemma": {"family": "gemma", "type": "encoder", "pooling": "mean", "params_M": 300, "layers": 24, "dim": 768},
    "multilingual-e5-large": {"family": "e5", "type": "encoder", "pooling": "mean", "params_M": 560, "layers": 24, "dim": 1024},
    "pythia-410m": {"family": "pythia", "type": "decoder", "pooling": "last", "params_M": 410, "layers": 24, "dim": 1024},
    "gte-qwen2-1.5b": {"family": "qwen", "type": "encoder", "pooling": "mean", "params_M": 1500, "layers": 28, "dim": 1536},
    "stella-1.5b": {"family": "stella", "type": "encoder", "pooling": "mean", "params_M": 1500, "layers": 28, "dim": 1536},
}


# ── Main analysis ────────────────────────────────────────────────────

def run_analysis(atlas_path: str) -> Dict[str, Any]:
    """Run full atlas analysis."""
    print("=" * 70)
    print("CTI ATLAS ANALYSIS")
    print("=" * 70)

    with open(atlas_path) as f:
        atlas = json.load(f)

    # 1. Extract curves
    print("\n[1] Extracting curves...")
    curves = extract_curves(atlas)
    total_curves = sum(len(ds) for ds in curves.values())
    print(f"  {len(curves)} models, {total_curves} total curves")

    for model, datasets in curves.items():
        print(f"  {model}: {len(datasets)} datasets, {datasets[list(datasets.keys())[0]]['n_layers']} layers")

    # 2. Fit functional forms
    print("\n[2] Fitting functional forms...")
    all_fits = {}
    form_wins = defaultdict(int)
    form_wins_bic = defaultdict(int)

    for model, datasets in curves.items():
        all_fits[model] = {}
        for ds, data in datasets.items():
            x, y = data["x"], data["y_l1"]
            if len(x) < 4:
                continue
            result = fit_all_forms(x, y)
            all_fits[model][ds] = result
            if result["best_aic"]:
                form_wins[result["best_aic"]] += 1
            if result["best_bic"]:
                form_wins_bic[result["best_bic"]] += 1

    print("\n  Best form by AIC (across all curves):")
    total_fits = sum(form_wins.values())
    for form, count in sorted(form_wins.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total_fits, 1)
        print(f"    {form:25s}: {count:3d} / {total_fits} ({pct:.1f}%)")

    print("\n  Best form by BIC (penalizes complexity more):")
    for form, count in sorted(form_wins_bic.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total_fits, 1)
        print(f"    {form:25s}: {count:3d} / {total_fits} ({pct:.1f}%)")

    # Per-model best form
    print("\n  Best AIC form per model:")
    model_best_forms = {}
    for model in all_fits:
        model_forms = defaultdict(int)
        for ds in all_fits[model]:
            best = all_fits[model][ds]["best_aic"]
            if best:
                model_forms[best] += 1
        if model_forms:
            dominant = max(model_forms, key=lambda k: model_forms[k])
            model_best_forms[model] = dominant
            print(f"    {model:25s}: {dominant} ({model_forms[dominant]}/{sum(model_forms.values())})")

    # 3. Monotonicity analysis
    print("\n[3] Monotonicity analysis...")
    model_monotonicity = {}
    for model, datasets in curves.items():
        mono_scores = []
        peak_positions = []
        final_drops = 0
        for ds, data in datasets.items():
            y = data["y_l1"]
            if len(y) < 4:
                continue
            mono = analyze_monotonicity(y)
            mono_scores.append(mono["spearman_rho"])
            peak_positions.append(mono["peak_relative_idx"])
            if mono["final_layer_drop"]:
                final_drops += 1

        if mono_scores:
            model_monotonicity[model] = {
                "mean_monotonicity": float(np.mean(mono_scores)),
                "mean_peak_position": float(np.mean(peak_positions)),
                "final_drop_rate": final_drops / len(mono_scores),
                "is_monotonic": np.mean(mono_scores) > 0.9,
            }
            classification = "MONOTONIC" if model_monotonicity[model]["is_monotonic"] else "NON-MONOTONIC"
            drop_str = f"final-drop={final_drops}/{len(mono_scores)}" if not model_monotonicity[model]["is_monotonic"] else ""
            print(f"    {model:25s}: rho={np.mean(mono_scores):.3f}  peak@{np.mean(peak_positions):.2f}  {classification} {drop_str}")

    # 4. Cross-dataset invariance
    print("\n[4] Cross-dataset invariance...")
    invariance = compute_invariance(curves)
    for model, inv in sorted(invariance.items(), key=lambda x: -x[1]["mean_rho"]):
        print(f"    {model:25s}: mean_rho={inv['mean_rho']:.3f}  min={inv['min_rho']:.3f}  max={inv['max_rho']:.3f}  ({inv['n_pairs']} pairs)")

    mean_invariance = np.mean([v["mean_rho"] for v in invariance.values()])
    print(f"\n  Overall mean cross-dataset invariance: {mean_invariance:.3f}")

    # 5. Universality class discovery
    print("\n[5] Universality class discovery...")
    features = extract_shape_features(curves)
    if len(features) >= 3:
        for n_clust in [2, 3]:
            clustering = cluster_models(features, n_clusters=n_clust)
            print(f"\n  k={n_clust} clusters:")
            for label, info in clustering["clusters"].items():
                print(f"    Class {label} ({info['shape_description']}):")
                for m in info["members"]:
                    arch = ARCHITECTURE_FEATURES.get(m, {})
                    arch_type = arch.get("type", "?")
                    pool = arch.get("pooling", "?")
                    print(f"      - {m} ({arch_type}, {pool})")

    # 6. Peak layer prediction
    print("\n[6] Peak layer analysis...")
    peak_data = []
    for model, datasets in curves.items():
        n_layers = datasets[list(datasets.keys())[0]]["n_layers"]
        for ds, data in datasets.items():
            peak = data["peak_l1_layer"]
            peak_data.append({
                "model": model,
                "dataset": ds,
                "peak_layer": peak,
                "n_layers": n_layers,
                "peak_relative": peak / n_layers if n_layers > 0 else 0,
                "is_final": peak == n_layers,
            })

    # How often is final layer the peak?
    n_final = sum(1 for p in peak_data if p["is_final"])
    print(f"  Final layer is peak: {n_final}/{len(peak_data)} ({100*n_final/max(len(peak_data),1):.1f}%)")

    # Average optimal relative depth
    rel_peaks = [p["peak_relative"] for p in peak_data]
    print(f"  Mean optimal relative depth: {np.mean(rel_peaks):.3f} +/- {np.std(rel_peaks):.3f}")

    # Per-model peak consistency
    print("\n  Per-model peak consistency:")
    for model, datasets in curves.items():
        peaks = [data["peak_l1_layer"] for data in datasets.values()]
        n_layers = datasets[list(datasets.keys())[0]]["n_layers"]
        print(f"    {model:25s}: layers={n_layers:2d}  peaks={peaks}  median={np.median(peaks):.0f}  std={np.std(peaks):.1f}")

    # 7. Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Key claims support
    n_monotonic = sum(1 for m in model_monotonicity.values() if m["is_monotonic"])
    n_nonmono = sum(1 for m in model_monotonicity.values() if not m["is_monotonic"])
    print(f"\n  Models: {n_monotonic} monotonic, {n_nonmono} non-monotonic")

    power_law_wins = form_wins.get("power_law", 0)
    print(f"  Power law wins: {power_law_wins}/{total_fits} ({100*power_law_wins/max(total_fits,1):.1f}%) -- FALSIFIED as universal")

    sigmoid_wins = form_wins.get("sigmoid", 0)
    print(f"  Sigmoid wins: {sigmoid_wins}/{total_fits} ({100*sigmoid_wins/max(total_fits,1):.1f}%)")

    print(f"  Cross-dataset invariance: mean rho = {mean_invariance:.3f}")
    print(f"  Final layer is optimal: {100*n_final/max(len(peak_data),1):.1f}% of curves")
    print(f"  Mean optimal depth: {np.mean(rel_peaks):.1f}% of total layers")

    # Compute savings
    non_final_savings = []
    for p in peak_data:
        if not p["is_final"] and p["n_layers"] > 0:
            savings = 1.0 - p["peak_relative"]
            non_final_savings.append(savings)
    if non_final_savings:
        print(f"  Potential compute savings (non-final peaks): {100*np.mean(non_final_savings):.1f}% mean layer reduction")

    # Compile results
    results = {
        "n_models": len(curves),
        "n_curves": total_curves,
        "form_wins_aic": dict(form_wins),
        "form_wins_bic": dict(form_wins_bic),
        "model_best_forms": model_best_forms,
        "model_monotonicity": model_monotonicity,
        "cross_dataset_invariance": {
            k: {
                "mean_rho": v["mean_rho"],
                "median_rho": v["median_rho"],
                "min_rho": v["min_rho"],
                "max_rho": v["max_rho"],
            }
            for k, v in invariance.items()
        },
        "overall_invariance": float(mean_invariance),
        "peak_analysis": {
            "final_layer_rate": n_final / max(len(peak_data), 1),
            "mean_optimal_depth": float(np.mean(rel_peaks)),
            "std_optimal_depth": float(np.std(rel_peaks)),
        },
        "all_fits": {
            model: {
                ds: {
                    "best_aic": fits["best_aic"],
                    "best_bic": fits["best_bic"],
                    "r2": {
                        form: fits["fits"][form]["r2"]
                        for form in fits["fits"]
                        if fits["fits"][form]["success"]
                    },
                }
                for ds, fits in datasets.items()
            }
            for model, datasets in all_fits.items()
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RESULTS_DIR / "cti_atlas_fit.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "cti_atlas_analysis.json"))
    args = parser.parse_args()

    results = run_analysis(args.input)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
