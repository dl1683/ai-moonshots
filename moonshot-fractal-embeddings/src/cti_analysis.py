#!/usr/bin/env python
"""
CTI Pilot Analysis — Evaluate against Codex-designed success criteria.

Reads cti_pilot_results.json and evaluates:
1. Monotonicity: Spearman(C, D) <= -0.9 on >= 80% curves
2. Fit quality: Median R^2 >= 0.93, 25th pct >= 0.85
3. Exponent stability: |alpha_i - alpha_j| <= 0.10, pooled CI width <= 0.10
4. Predictive power: Fit first 4 layers, predict last 2, MAPE <= 7%
5. Rival laws: Power law beats exp/log by dAIC >= 6 on >= 70%
6. Parameter sanity: 0 <= D_inf <= 0.25, 0.15 <= alpha <= 1.2, k > 0

Outputs: results/cti_pilot_evaluation.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


# ── Model functions ──────────────────────────────────────────────────────

def power_law(c, d_inf, k, alpha):
    return d_inf + k * np.power(c + 1e-12, -alpha)


def exponential_decay(c, d_inf, k, lam):
    return d_inf + k * np.exp(-lam * c)


def log_decay(c, d_inf, k, b):
    return d_inf + k / (np.log(c + 1e-12) + b)


RIVAL_MODELS = {
    "exponential": exponential_decay,
    "logarithmic": log_decay,
}


# ── Helpers ──────────────────────────────────────────────────────────────

def aic(n, k_params, rss):
    if n <= 0 or rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + 2 * k_params


def fit_model(func, c, d, p0=None, bounds=None):
    try:
        popt, pcov = curve_fit(func, c, d, p0=p0, bounds=bounds, maxfev=10000)
        residuals = d - func(c, *popt)
        rss = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((d - np.mean(d)) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 1e-12 else 0.0
        n = len(d)
        k_params = len(popt)
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k_params - 1, 1)
        return {
            "status": "ok",
            "params": [float(x) for x in popt],
            "pcov": pcov.tolist() if pcov is not None else None,
            "rss": rss,
            "r2": r2,
            "adj_r2": adj_r2,
            "aic": aic(n, k_params, rss),
            "n": n,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def extract_curves(state: Dict) -> List[Dict]:
    """Extract (model, dataset, C[], D[]) curves from CTI state.

    Handles nested format: results[model][dataset][layer_str][seed_str] = {...}
    Also handles flat pipe-delimited keys as fallback.
    """
    results = state.get("results", {})
    total_layers = state.get("config", {}).get("total_layers", 12)

    # Group by (model, dataset)
    grouped = {}

    # Try nested format first (results[model][dataset][layer][seed])
    for model_key, model_val in results.items():
        if not isinstance(model_val, dict):
            continue
        for dataset_key, dataset_val in model_val.items():
            if not isinstance(dataset_val, dict):
                continue
            for layer_key, layer_val in dataset_val.items():
                if not isinstance(layer_val, dict):
                    continue
                try:
                    layer = int(layer_key)
                except (ValueError, TypeError):
                    continue
                for seed_key, run in layer_val.items():
                    if not isinstance(run, dict) or run.get("status") != "ok":
                        continue
                    gk = f"{model_key}|{dataset_key}"
                    if gk not in grouped:
                        grouped[gk] = {}
                    if layer not in grouped[gk]:
                        grouped[gk][layer] = []
                    s_val = run.get("steerability", run.get("S"))
                    if s_val is not None:
                        grouped[gk][layer].append(float(s_val))

    # Fallback: flat pipe-delimited keys
    if not grouped:
        for key, val in results.items():
            if not isinstance(val, dict) or val.get("status") != "ok":
                continue
            parts = key.split("|")
            if len(parts) < 4:
                continue
            model, dataset = parts[0], parts[1]
            layer = int(parts[2].replace("L", ""))
            gk = f"{model}|{dataset}"
            if gk not in grouped:
                grouped[gk] = {}
            if layer not in grouped[gk]:
                grouped[gk][layer] = []
            s_val = val.get("steerability", val.get("S"))
            if s_val is not None:
                grouped[gk][layer].append(float(s_val))

    curves = []
    for gk, layer_data in grouped.items():
        model, dataset = gk.split("|")
        layers_sorted = sorted(layer_data.keys())
        if len(layers_sorted) < 3:
            continue

        # Mean steerability per layer
        S_by_layer = {l: np.mean(layer_data[l]) for l in layers_sorted}
        S_full = S_by_layer.get(max(layers_sorted), 0)
        S_chance = 0.0  # Baseline

        C_vals = []
        D_vals = []
        for l in layers_sorted:
            c = l / total_layers
            d = (S_full - S_by_layer[l]) / (S_full - S_chance + 1e-6) if S_full > S_chance else 0
            C_vals.append(c)
            D_vals.append(d)

        curves.append({
            "model": model,
            "dataset": dataset,
            "layers": layers_sorted,
            "C": np.array(C_vals),
            "D": np.array(D_vals),
            "S_full": S_full,
            "S_by_layer": S_by_layer,
        })

    return curves


# ── Criterion evaluators ─────────────────────────────────────────────────

def check_monotonicity(curves: List[Dict]) -> Dict:
    rhos = []
    for c in curves:
        rho, _ = stats.spearmanr(c["C"], c["D"])
        rhos.append({"model": c["model"], "dataset": c["dataset"], "rho": float(rho)})

    passing = sum(1 for r in rhos if r["rho"] <= -0.9)
    frac = passing / len(rhos) if rhos else 0
    violating = sum(1 for r in rhos if r["rho"] > -0.5)
    frac_violating = violating / len(rhos) if rhos else 0

    return {
        "criterion": "monotonicity",
        "pass": frac >= 0.80,
        "falsified": frac_violating > 0.30,
        "fraction_passing": frac,
        "fraction_violating": frac_violating,
        "details": rhos,
    }


def check_fit_quality(curves: List[Dict]) -> Dict:
    fits = []
    for c in curves:
        result = fit_model(
            power_law, c["C"], c["D"],
            p0=[0.0, 1.0, 0.5],
            bounds=([0, 0, 0.01], [1.0, 100, 5.0]),
        )
        result["model"] = c["model"]
        result["dataset"] = c["dataset"]
        fits.append(result)

    adj_r2s = [f["adj_r2"] for f in fits if f["status"] == "ok"]
    if not adj_r2s:
        return {"criterion": "fit_quality", "pass": False, "falsified": True, "detail": "no successful fits"}

    median_r2 = float(np.median(adj_r2s))
    pct25_r2 = float(np.percentile(adj_r2s, 25))

    return {
        "criterion": "fit_quality",
        "pass": median_r2 >= 0.93 and pct25_r2 >= 0.85,
        "falsified": median_r2 < 0.85,
        "median_adj_r2": median_r2,
        "p25_adj_r2": pct25_r2,
        "fits": fits,
    }


def check_exponent_stability(curves: List[Dict]) -> Dict:
    alphas = {}
    for c in curves:
        result = fit_model(
            power_law, c["C"], c["D"],
            p0=[0.0, 1.0, 0.5],
            bounds=([0, 0, 0.01], [1.0, 100, 5.0]),
        )
        if result["status"] == "ok":
            alphas[c["model"]] = alphas.get(c["model"], [])
            alphas[c["model"]].append(result["params"][2])  # alpha

    if len(alphas) < 2:
        return {"criterion": "exponent_stability", "pass": False, "falsified": True, "detail": "not enough models"}

    # Per-model mean alpha
    model_alphas = {m: float(np.mean(a)) for m, a in alphas.items()}
    models = list(model_alphas.keys())

    # Pairwise differences
    max_diff = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            diff = abs(model_alphas[models[i]] - model_alphas[models[j]])
            max_diff = max(max_diff, diff)

    # Pooled CI
    all_alphas = [a for lst in alphas.values() for a in lst]
    mean_alpha = float(np.mean(all_alphas))
    se_alpha = float(np.std(all_alphas, ddof=1) / np.sqrt(len(all_alphas))) if len(all_alphas) > 1 else float("inf")
    ci_width = 2 * 1.96 * se_alpha

    return {
        "criterion": "exponent_stability",
        "pass": max_diff <= 0.10 and ci_width <= 0.10,
        "falsified": max_diff > 0.20 or (ci_width > 0.10 and 0 >= mean_alpha - 1.96 * se_alpha),
        "model_alphas": model_alphas,
        "max_pairwise_diff": max_diff,
        "pooled_mean_alpha": mean_alpha,
        "pooled_ci_width": ci_width,
        "pooled_ci": [mean_alpha - 1.96 * se_alpha, mean_alpha + 1.96 * se_alpha],
    }


def check_predictive_power(curves: List[Dict]) -> Dict:
    results = []
    for c in curves:
        if len(c["C"]) < 5:
            continue
        # Fit on first 4 points
        c_fit = c["C"][:4]
        d_fit = c["D"][:4]
        c_test = c["C"][4:]
        d_test = c["D"][4:]

        fit = fit_model(
            power_law, c_fit, d_fit,
            p0=[0.0, 1.0, 0.5],
            bounds=([0, 0, 0.01], [1.0, 100, 5.0]),
        )
        if fit["status"] != "ok":
            continue

        d_pred = power_law(c_test, *fit["params"])
        abs_errors = np.abs(d_test - d_pred)
        pct_errors = abs_errors / (np.abs(d_test) + 1e-6) * 100

        results.append({
            "model": c["model"],
            "dataset": c["dataset"],
            "mape": float(np.mean(pct_errors)),
            "max_abs_error": float(np.max(abs_errors)),
            "predictions": d_pred.tolist(),
            "actuals": d_test.tolist(),
        })

    if not results:
        return {"criterion": "predictive_power", "pass": False, "falsified": True, "detail": "no curves with 5+ points"}

    mapes = [r["mape"] for r in results]
    max_errs = [r["max_abs_error"] for r in results]

    return {
        "criterion": "predictive_power",
        "pass": float(np.median(mapes)) <= 7.0 and float(np.median(max_errs)) <= 0.02,
        "falsified": float(np.median(mapes)) > 12.0,
        "median_mape": float(np.median(mapes)),
        "median_max_abs_error": float(np.median(max_errs)),
        "details": results,
    }


def check_rival_laws(curves: List[Dict]) -> Dict:
    results = []
    for c in curves:
        pl_fit = fit_model(
            power_law, c["C"], c["D"],
            p0=[0.0, 1.0, 0.5],
            bounds=([0, 0, 0.01], [1.0, 100, 5.0]),
        )
        rival_fits = {}
        for name, func in RIVAL_MODELS.items():
            rival_fits[name] = fit_model(
                func, c["C"], c["D"],
                p0=[0.0, 1.0, 0.5],
                bounds=([0, 0, 0.001], [1.0, 100, 50.0]),
            )

        if pl_fit["status"] != "ok":
            continue

        pl_aic = pl_fit["aic"]
        wins_by_6 = True
        rival_dominant = False
        for name, rf in rival_fits.items():
            if rf["status"] != "ok":
                continue
            delta = rf["aic"] - pl_aic
            if delta < 6:
                wins_by_6 = False
            if delta < -10:
                rival_dominant = True

        results.append({
            "model": c["model"],
            "dataset": c["dataset"],
            "pl_aic": pl_aic,
            "rival_aics": {n: rf["aic"] for n, rf in rival_fits.items() if rf["status"] == "ok"},
            "pl_wins_by_6": wins_by_6,
            "rival_dominant": rival_dominant,
        })

    if not results:
        return {"criterion": "rival_laws", "pass": False, "falsified": True}

    frac_wins = sum(1 for r in results if r["pl_wins_by_6"]) / len(results)
    frac_rival_dom = sum(1 for r in results if r["rival_dominant"]) / len(results)

    return {
        "criterion": "rival_laws",
        "pass": frac_wins >= 0.70,
        "falsified": frac_rival_dom > 0.50,
        "frac_pl_wins": frac_wins,
        "frac_rival_dominant": frac_rival_dom,
        "details": results,
    }


def check_parameter_sanity(curves: List[Dict]) -> Dict:
    results = []
    for c in curves:
        fit = fit_model(
            power_law, c["C"], c["D"],
            p0=[0.0, 1.0, 0.5],
            bounds=([0, 0, 0.01], [1.0, 100, 5.0]),
        )
        if fit["status"] != "ok":
            continue
        d_inf, k, alpha = fit["params"]
        sane = (0 <= d_inf <= 0.25) and (0.15 <= alpha <= 1.2) and (k > 0)
        results.append({
            "model": c["model"],
            "dataset": c["dataset"],
            "d_inf": d_inf,
            "k": k,
            "alpha": alpha,
            "sane": sane,
        })

    if not results:
        return {"criterion": "parameter_sanity", "pass": False, "falsified": True}

    frac_sane = sum(1 for r in results if r["sane"]) / len(results)

    return {
        "criterion": "parameter_sanity",
        "pass": frac_sane >= 0.80,
        "falsified": frac_sane < 0.50,
        "frac_sane": frac_sane,
        "details": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def evaluate_cti_pilot(state_path: Path) -> Dict:
    state = json.loads(state_path.read_text())
    curves = extract_curves(state)

    if not curves:
        return {
            "status": "no_data",
            "n_curves": 0,
            "decision": "RED",
            "reason": "No valid curves extracted from CTI results",
        }

    checks = [
        check_monotonicity(curves),
        check_fit_quality(curves),
        check_exponent_stability(curves),
        check_predictive_power(curves),
        check_rival_laws(curves),
        check_parameter_sanity(curves),
    ]

    n_pass = sum(1 for c in checks if c.get("pass"))
    n_falsified = sum(1 for c in checks if c.get("falsified"))

    if n_pass >= 5 and n_falsified == 0:
        decision = "GREEN"
    elif n_pass >= 4:
        decision = "YELLOW"
    else:
        decision = "RED"

    return {
        "status": "evaluated",
        "n_curves": len(curves),
        "n_pass": n_pass,
        "n_falsified": n_falsified,
        "decision": decision,
        "criteria": {c["criterion"]: c for c in checks},
    }


def main():
    state_path = RESULTS_DIR / "cti_pilot_results.json"
    if not state_path.exists():
        print(f"CTI results not found at {state_path}")
        sys.exit(1)

    result = evaluate_cti_pilot(state_path)
    out_path = RESULTS_DIR / "cti_pilot_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))

    print(f"\nCTI Pilot Evaluation")
    print(f"====================")
    print(f"Curves analyzed: {result['n_curves']}")

    if result.get("status") == "no_data":
        print(f"DECISION: {result['decision']}")
        print(f"Reason: {result.get('reason', 'insufficient data')}")
        print(f"\nNeed at least 3 layers per (model, dataset) curve.")
        print(f"Saved to: {out_path}")
        return

    print(f"Criteria passed: {result['n_pass']}/6")
    print(f"Falsified: {result['n_falsified']}")
    print(f"DECISION: {result['decision']}")
    print()

    for name, check in result.get("criteria", {}).items():
        status = "PASS" if check.get("pass") else ("FALSIFIED" if check.get("falsified") else "FAIL")
        print(f"  {name}: {status}")
        for k in ["median_adj_r2", "p25_adj_r2", "max_pairwise_diff", "pooled_mean_alpha",
                   "pooled_ci_width", "median_mape", "frac_pl_wins", "frac_sane",
                   "fraction_passing"]:
            if k in check:
                print(f"    {k}: {check[k]:.4f}")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
