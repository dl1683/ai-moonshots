#!/usr/bin/env python
"""
cti_oos_test.py

Out-of-sample layer prediction test for piecewise sigmoid defense.
Fit on even-indexed layers, predict odd-indexed layers.
Compare RMSE across functional forms.

Also: bootstrap confidence intervals for change points.

Usage:
    python -u src/cti_oos_test.py --input results/cti_atlas_fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import optimize, stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=optimize.OptimizeWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


# ── Functional forms (same as atlas analysis) ────────────────────────

def power_law(x, a, b, c):
    return a + b * np.power(np.clip(x, 1e-10, None), c)

def sigmoid_func(x, a, b, c, d):
    return a + b / (1.0 + np.exp(-c * (x - d)))

def exponential_func(x, a, b, c):
    return a - b * np.exp(-c * x)

def linear_func(x, a, b):
    return a + b * x

def piecewise_sigmoid(x, a, b1, c1, d1, b2, c2, d2):
    s1 = b1 / (1.0 + np.exp(-c1 * (x - d1)))
    s2 = b2 / (1.0 + np.exp(-c2 * (x - d2)))
    return a + s1 + s2


FORMS = {
    "power_law": (power_law, 3, [0.5, 0.5, 0.5]),
    "sigmoid": (sigmoid_func, 4, [0.3, 0.5, 10.0, 0.5]),
    "exponential": (exponential_func, 3, [0.8, 0.5, 3.0]),
    "linear": (linear_func, 2, [0.3, 0.5]),
    "piecewise_sigmoid": (piecewise_sigmoid, 7, [0.3, 0.2, 10.0, 0.3, 0.3, 10.0, 0.7]),
}


def fit_and_predict(x_train, y_train, x_test, form_name):
    """Fit on train, predict on test. Return RMSE and predictions."""
    func, n_params, p0 = FORMS[form_name]

    if len(x_train) <= n_params:
        return {"success": False, "rmse": float("inf")}

    try:
        if form_name == "linear":
            popt = np.polyfit(x_train, y_train, 1)
            popt = [popt[1], popt[0]]
            y_pred = linear_func(x_test, *popt)
        else:
            popt, _ = optimize.curve_fit(
                func, x_train, y_train, p0=p0, maxfev=10000,
            )
            y_pred = func(x_test, *popt)

        rmse = float(np.sqrt(np.mean((y_pred - np.array([0])) ** 2))) if len(x_test) == 0 else 0
        # Actually compute against test
        return {"success": True, "rmse": float("inf"), "params": popt.tolist() if hasattr(popt, 'tolist') else list(popt)}
    except Exception:
        return {"success": False, "rmse": float("inf")}


def oos_evaluate_curve(x, y, form_name):
    """Even-odd split: fit on even, predict odd."""
    func, n_params, p0 = FORMS[form_name]

    even_mask = np.arange(len(x)) % 2 == 0
    odd_mask = ~even_mask

    x_train, y_train = x[even_mask], y[even_mask]
    x_test, y_test = x[odd_mask], y[odd_mask]

    if len(x_train) <= n_params or len(x_test) == 0:
        return {"success": False, "rmse": float("inf"), "form": form_name}

    try:
        if form_name == "linear":
            popt = np.polyfit(x_train, y_train, 1)
            y_pred = np.polyval(popt, x_test)
        else:
            popt, _ = optimize.curve_fit(func, x_train, y_train, p0=p0, maxfev=10000)
            y_pred = func(x_test, *popt)

        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_test - y_pred)))

        return {
            "success": True,
            "form": form_name,
            "rmse": rmse,
            "mae": mae,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
        }
    except Exception as e:
        return {"success": False, "rmse": float("inf"), "form": form_name, "error": str(e)}


def bootstrap_change_points(x, y, n_bootstrap=200):
    """Bootstrap confidence intervals for piecewise sigmoid change points."""
    change_points_1 = []
    change_points_2 = []
    n = len(x)

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        idx.sort()
        x_boot = x[idx]
        y_boot = y[idx]

        try:
            popt, _ = optimize.curve_fit(
                piecewise_sigmoid, x_boot, y_boot,
                p0=[0.3, 0.2, 10.0, 0.3, 0.3, 10.0, 0.7],
                maxfev=10000,
            )
            # Change points are d1 and d2
            change_points_1.append(popt[3])
            change_points_2.append(popt[6])
        except Exception:
            pass

    if len(change_points_1) < 10:
        return None

    cp1 = np.array(change_points_1)
    cp2 = np.array(change_points_2)

    return {
        "change_point_1": {
            "mean": float(np.mean(cp1)),
            "std": float(np.std(cp1)),
            "ci_95": [float(np.percentile(cp1, 2.5)), float(np.percentile(cp1, 97.5))],
        },
        "change_point_2": {
            "mean": float(np.mean(cp2)),
            "std": float(np.std(cp2)),
            "ci_95": [float(np.percentile(cp2, 2.5)), float(np.percentile(cp2, 97.5))],
        },
        "n_successful": len(change_points_1),
    }


def extract_curves(atlas):
    """Extract curves from atlas JSON."""
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
            sorted_layers = sorted(layers.items(), key=lambda x: int(x[0]))
            x = np.array([v["C_relative"] for _, v in sorted_layers])
            y_l1 = np.array([v["knn_l1"] for _, v in sorted_layers])
            # Skip layer 0 if degenerate
            if len(x) > 2 and y_l1[0] < 0.05:
                x = x[1:]
                y_l1 = y_l1[1:]
            curves[model_key][ds_name] = {"x": x, "y": y_l1, "n_layers": num_layers}
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RESULTS_DIR / "cti_atlas_fit.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "cti_oos_results.json"))
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()

    with open(args.input) as f:
        atlas = json.load(f)

    curves = extract_curves(atlas)
    total_curves = sum(len(ds) for ds in curves.values())
    print(f"Loaded {len(curves)} models, {total_curves} curves")

    # ── 1. Out-of-sample RMSE comparison ─────────────────────────────
    print("\n" + "=" * 70)
    print("[1] Out-of-Sample Layer Prediction (even->odd split)")
    print("=" * 70)

    form_rmses = defaultdict(list)
    form_maes = defaultdict(list)
    form_wins = defaultdict(int)
    curve_results = {}

    for model, datasets in curves.items():
        curve_results[model] = {}
        for ds, data in datasets.items():
            x, y = data["x"], data["y"]
            if len(x) < 6:  # Need enough points for split
                continue

            best_rmse = float("inf")
            best_form = None
            ds_results = {}

            for form_name in FORMS:
                result = oos_evaluate_curve(x, y, form_name)
                ds_results[form_name] = result
                if result["success"]:
                    form_rmses[form_name].append(result["rmse"])
                    form_maes[form_name].append(result["mae"])
                    if result["rmse"] < best_rmse:
                        best_rmse = result["rmse"]
                        best_form = form_name

            if best_form:
                form_wins[best_form] += 1
            curve_results[model][ds] = ds_results

    print("\n  Form           | Mean RMSE | Mean MAE  | Wins  | Win%")
    print("  " + "-" * 60)
    total_wins = sum(form_wins.values())
    for form_name in ["piecewise_sigmoid", "sigmoid", "exponential", "power_law", "linear"]:
        if form_rmses[form_name]:
            mr = np.mean(form_rmses[form_name])
            ma = np.mean(form_maes[form_name])
            w = form_wins[form_name]
            wp = 100 * w / max(total_wins, 1)
            print(f"  {form_name:17s}| {mr:.6f} | {ma:.6f} | {w:5d} | {wp:.1f}%")

    # ── 2. Bootstrap change points ───────────────────────────────────
    print("\n" + "=" * 70)
    print("[2] Bootstrap Change Point Confidence Intervals (n=200)")
    print("=" * 70)

    bootstrap_results = {}
    n_stable = 0
    n_tested = 0

    for model, datasets in curves.items():
        bootstrap_results[model] = {}
        for ds, data in datasets.items():
            x, y = data["x"], data["y"]
            if len(x) < 8:  # Need enough points for bootstrap
                continue

            n_tested += 1
            cp = bootstrap_change_points(x, y, n_bootstrap=args.bootstrap)
            bootstrap_results[model][ds] = cp

            if cp:
                cp1_width = cp["change_point_1"]["ci_95"][1] - cp["change_point_1"]["ci_95"][0]
                cp2_width = cp["change_point_2"]["ci_95"][1] - cp["change_point_2"]["ci_95"][0]
                is_stable = cp1_width < 0.3 and cp2_width < 0.3
                if is_stable:
                    n_stable += 1
                print(f"  {model:20s} | {ds:15s} | CP1={cp['change_point_1']['mean']:.2f} [{cp['change_point_1']['ci_95'][0]:.2f},{cp['change_point_1']['ci_95'][1]:.2f}] | CP2={cp['change_point_2']['mean']:.2f} [{cp['change_point_2']['ci_95'][0]:.2f},{cp['change_point_2']['ci_95'][1]:.2f}] | {'STABLE' if is_stable else 'wide'}")

    print(f"\n  Stable change points (CI width < 0.3): {n_stable}/{n_tested} ({100*n_stable/max(n_tested,1):.1f}%)")

    # ── 3. k-sensitivity quick test ──────────────────────────────────
    print("\n" + "=" * 70)
    print("[3] Summary Statistics")
    print("=" * 70)

    # Power law vs piecewise comparison
    if form_rmses["power_law"] and form_rmses["piecewise_sigmoid"]:
        pl_rmse = np.mean(form_rmses["power_law"])
        ps_rmse = np.mean(form_rmses["piecewise_sigmoid"])
        reduction = (pl_rmse - ps_rmse) / pl_rmse * 100
        print(f"\n  Power law mean RMSE:     {pl_rmse:.6f}")
        print(f"  Piecewise sigmoid RMSE:  {ps_rmse:.6f}")
        print(f"  RMSE reduction:          {reduction:.1f}%")

        # Paired test
        pl_vals = np.array(form_rmses["power_law"])
        ps_vals = np.array(form_rmses["piecewise_sigmoid"])
        min_len = min(len(pl_vals), len(ps_vals))
        if min_len > 5:
            t_stat, p_val = stats.wilcoxon(pl_vals[:min_len], ps_vals[:min_len])
            print(f"  Wilcoxon test: p = {p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

    # Save results
    results = {
        "oos_rmse": {form: float(np.mean(vals)) for form, vals in form_rmses.items()},
        "oos_mae": {form: float(np.mean(vals)) for form, vals in form_maes.items()},
        "oos_wins": dict(form_wins),
        "total_curves_tested": total_wins,
        "bootstrap_stable_rate": n_stable / max(n_tested, 1),
        "n_bootstrap_tested": n_tested,
        "n_bootstrap_stable": n_stable,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
