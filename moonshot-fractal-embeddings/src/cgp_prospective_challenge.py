#!/usr/bin/env python
"""
cgp_prospective_challenge.py

CGP Prospective Causal-Sufficiency Challenge

Phase 1: Fit universal G -> kNN curve from Week 2 + Week 3 data
Phase 2: Generate pre-registered predictions for held-out conditions
Phase 3: Run held-out conditions and compare predictions vs actuals

Usage:
    # Phase 1+2: Fit curve and generate predictions (no GPU needed)
    python -u src/cgp_prospective_challenge.py --phase fit

    # Phase 3: Run held-out conditions and evaluate (GPU needed)
    python -u src/cgp_prospective_challenge.py --phase evaluate
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))


# ── Curve Fitting Functions ────────────────────────────────────────

def saturating_exp(G, a, b, c):
    """kNN = a * (1 - exp(-b * G)) + c"""
    return a * (1 - np.exp(-b * np.clip(G, 0, 500))) + c


def logarithmic(G, a, b):
    """kNN = a * log(G + 1) + b"""
    return a * np.log(np.clip(G, 0, None) + 1) + b


def power_law(G, a, b, c):
    """kNN = a * G^b + c"""
    return a * np.power(np.clip(G, 1e-8, None), b) + c


CURVE_FORMS = {
    "saturating_exp": {
        "func": saturating_exp,
        "p0": [0.5, 0.01, 0.3],
        "bounds": ([0, 0, 0], [1.0, 1.0, 1.0]),
    },
    "logarithmic": {
        "func": logarithmic,
        "p0": [0.1, 0.3],
        "bounds": ([-1, 0], [1, 1]),
    },
    "power_law": {
        "func": power_law,
        "p0": [0.1, 0.3, 0.2],
        "bounds": ([0, 0, 0], [1.0, 1.0, 1.0]),
    },
}


# ── Data Loading ───────────────────────────────────────────────────

def load_week2_data():
    """Load Week 2 data and extract (G, kNN_l1) pairs."""
    path = RESULTS_DIR / "cgp_week2_control_study.json"
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    points = []
    for c in data.get("conditions", []):
        obj = c.get("objective", "")
        if obj == "baseline":
            continue

        for ds_name, ds_metrics in c.get("metrics", {}).items():
            if isinstance(ds_metrics, dict):
                # Get deepest layer
                int_keys = [int(k) for k in ds_metrics if k.isdigit()]
                if not int_keys:
                    continue
                m = ds_metrics[str(max(int_keys))]
                knn_l1 = m.get("knn_l1", None)
                sep = m.get("class_sep_l1", None)
                if knn_l1 is not None and sep is not None:
                    # Week 2 doesn't have G directly; use sep as proxy
                    points.append({
                        "source": "week2",
                        "model": "pythia-160m",
                        "dataset": ds_name,
                        "objective": obj,
                        "lambda_sep": c.get("lambda_sep", 0),
                        "sep_l1": sep,
                        "knn_l1": knn_l1,
                        "G": None,  # Will compute if needed
                    })
    return points


def load_week3_data():
    """Load Week 3 data and extract (G, kNN_l1) pairs."""
    # Try full results first, then partial
    for fname in ["cgp_week3_cross_arch.json", "cgp_week3_cross_arch_partial.json"]:
        path = RESULTS_DIR / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            break
    else:
        return []

    points = []
    for r in data.get("results", []):
        model = r.get("model", "")
        lam_sep = r.get("lambda_sep", 0)
        seed = r.get("seed", 0)
        is_baseline = r.get("is_baseline", False)

        for ds_name, metrics in r.get("eval", {}).items():
            if "error" in metrics:
                continue
            knn_l1 = metrics.get("knn_l1", None)
            G = metrics.get("composite_G", None)
            sep_l1 = metrics.get("class_sep_l1", None)

            if knn_l1 is not None and G is not None:
                points.append({
                    "source": "week3",
                    "model": model,
                    "dataset": ds_name,
                    "lambda_sep": lam_sep,
                    "seed": seed,
                    "is_baseline": is_baseline,
                    "sep_l1": sep_l1,
                    "knn_l1": knn_l1,
                    "G": G,
                    "kappa": metrics.get("kappa", None),
                    "fisher_q": metrics.get("fisher_q", None),
                })
    return points


# ── Phase 1: Fit Curves ───────────────────────────────────────────

def fit_universal_curve(points):
    """Fit multiple curve forms and select best by LOOCV R2."""
    # Filter to points with valid G
    valid = [p for p in points if p["G"] is not None and
             np.isfinite(p["G"]) and np.isfinite(p["knn_l1"]) and
             p["G"] > 0]

    if len(valid) < 10:
        print(f"ERROR: Only {len(valid)} valid points. Need >= 10.")
        return None

    G = np.array([p["G"] for p in valid])
    y = np.array([p["knn_l1"] for p in valid])

    print(f"\nFitting on {len(valid)} data points")
    print(f"  G range: [{G.min():.1f}, {G.max():.1f}]")
    print(f"  kNN_l1 range: [{y.min():.3f}, {y.max():.3f}]")

    best_form = None
    best_r2 = -np.inf
    best_params = None
    results = {}

    for form_name, form_cfg in CURVE_FORMS.items():
        try:
            popt, pcov = optimize.curve_fit(
                form_cfg["func"], G, y,
                p0=form_cfg["p0"],
                bounds=form_cfg["bounds"],
                maxfev=10000,
            )
            y_pred = form_cfg["func"](G, *popt)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            # Leave-one-model-out CV
            models = list(set(p["model"] for p in valid))
            cv_r2s = []
            for held_out_model in models:
                train_idx = [i for i, p in enumerate(valid) if p["model"] != held_out_model]
                test_idx = [i for i, p in enumerate(valid) if p["model"] == held_out_model]
                if len(train_idx) < 5 or len(test_idx) < 3:
                    continue
                try:
                    popt_cv, _ = optimize.curve_fit(
                        form_cfg["func"], G[train_idx], y[train_idx],
                        p0=form_cfg["p0"],
                        bounds=form_cfg["bounds"],
                        maxfev=10000,
                    )
                    y_pred_cv = form_cfg["func"](G[test_idx], *popt_cv)
                    cv_r2 = r2_score(y[test_idx], y_pred_cv)
                    cv_r2s.append(cv_r2)
                except Exception:
                    pass

            mean_cv_r2 = np.mean(cv_r2s) if cv_r2s else -1.0

            results[form_name] = {
                "params": popt.tolist(),
                "r2": float(r2),
                "rmse": float(rmse),
                "cv_r2s": {m: float(r) for m, r in zip(models, cv_r2s)},
                "mean_cv_r2": float(mean_cv_r2),
            }

            print(f"\n  {form_name}:")
            print(f"    params = {popt}")
            print(f"    R2 = {r2:.4f}, RMSE = {rmse:.4f}")
            print(f"    CV R2 = {mean_cv_r2:.4f} (per-model: {cv_r2s})")

            if mean_cv_r2 > best_r2:
                best_r2 = mean_cv_r2
                best_form = form_name
                best_params = popt

        except Exception as e:
            print(f"  {form_name}: FAILED - {e}")
            results[form_name] = {"error": str(e)}

    print(f"\n  BEST: {best_form} (CV R2 = {best_r2:.4f})")

    return {
        "best_form": best_form,
        "best_params": best_params.tolist() if best_params is not None else None,
        "best_cv_r2": float(best_r2),
        "all_results": results,
        "n_points": len(valid),
        "G_range": [float(G.min()), float(G.max())],
        "knn_range": [float(y.min()), float(y.max())],
    }


def test_mediation(points):
    """Test if adding model indicator improves R2 beyond G alone."""
    valid = [p for p in points if p["G"] is not None and
             np.isfinite(p["G"]) and np.isfinite(p["knn_l1"]) and
             p["G"] > 0]

    if len(valid) < 10:
        return {"error": "insufficient data"}

    G = np.array([p["G"] for p in valid]).reshape(-1, 1)
    y = np.array([p["knn_l1"] for p in valid])

    # Model 1: G only
    reg_g = LinearRegression().fit(G, y)
    r2_g = reg_g.score(G, y)

    # Model 2: G + model indicators
    models = sorted(set(p["model"] for p in valid))
    model_dummies = np.zeros((len(valid), len(models)))
    for i, p in enumerate(valid):
        model_dummies[i, models.index(p["model"])] = 1.0

    X_full = np.hstack([G, model_dummies])
    reg_full = LinearRegression().fit(X_full, y)
    r2_full = reg_full.score(X_full, y)

    # F-test for model indicator contribution
    n = len(valid)
    p1 = 1  # G only
    p2 = 1 + len(models)  # G + model dummies
    if r2_full > r2_g and n > p2 + 1:
        f_stat = ((r2_full - r2_g) / (p2 - p1)) / ((1 - r2_full) / (n - p2))
        f_p = 1 - stats.f.cdf(f_stat, p2 - p1, n - p2)
    else:
        f_stat = 0.0
        f_p = 1.0

    return {
        "r2_G_only": float(r2_g),
        "r2_G_plus_model": float(r2_full),
        "delta_r2": float(r2_full - r2_g),
        "f_stat": float(f_stat),
        "f_p": float(f_p),
        "mediation_holds": bool(f_p > 0.05),
        "n_models": len(models),
        "models": models,
        "n_points": n,
    }


def generate_predictions(curve_fit, G_values):
    """Generate predictions for given G values using the fitted curve."""
    form_name = curve_fit["best_form"]
    params = curve_fit["best_params"]
    func = CURVE_FORMS[form_name]["func"]

    predictions = {}
    for label, G in G_values.items():
        pred = float(func(G, *params))
        predictions[label] = {
            "G": float(G),
            "predicted_knn_l1": pred,
        }

    return predictions


# ── Phase 3: Run Held-Out Conditions ─────────────────────────────

def run_heldout_conditions():
    """Run bge-base on held-out conditions. Returns results dict."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from hierarchical_datasets import load_hierarchical_dataset

    # Import from Week 3 script
    from cgp_week3_cross_arch import (
        LoRALinear, apply_lora_encoder, contrastive_loss,
        class_separation_loss, extract_embeddings, compute_metrics,
        train_encoder_with_sep,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Held-out model
    hf_path = "BAAI/bge-base-en-v1.5"
    lambda_seps = [0.0, 0.1, 0.3, 1.0]
    seeds = [42, 123, 456]
    eval_datasets = ["clinc", "dbpedia_classes", "agnews", "trec"]
    steps = 500
    eval_samples = 800

    # Load training data
    train_data = load_hierarchical_dataset("clinc", split="train", max_samples=5000)
    texts = [s.text for s in train_data.samples]
    l1_labels = [s.level1_label for s in train_data.samples]

    all_results = []

    for lam_sep, seed in product(lambda_seps, seeds):
        print(f"\n  --- bge-base lam_sep={lam_sep} seed={seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        base_model = AutoModel.from_pretrained(
            hf_path, torch_dtype=torch.float32
        ).to(device)

        if lam_sep == 0.0 and seed == seeds[0]:
            model = base_model
            train_time = 0.0
            is_baseline = True
        else:
            model = apply_lora_encoder(base_model, 16, 32)
            t0 = time.time()
            train_encoder_with_sep(
                model, tokenizer, texts, l1_labels, device,
                lambda_sep=lam_sep, num_steps=steps,
            )
            train_time = time.time() - t0
            is_baseline = False

        # Evaluate
        eval_results = {}
        for ds_name in eval_datasets:
            try:
                ds = load_hierarchical_dataset(ds_name, split="test",
                                               max_samples=eval_samples)
                eval_texts = [s.text for s in ds.samples]
                eval_l0 = [s.level0_label for s in ds.samples]
                eval_l1 = [s.level1_label for s in ds.samples]

                embs = extract_embeddings(model, tokenizer, eval_texts, device)
                metrics = compute_metrics(embs, eval_l0, eval_l1)
                eval_results[ds_name] = metrics
                print(f"    {ds_name}: knn_l1={metrics['knn_l1']:.3f} "
                      f"G={metrics['composite_G']:.1f}")
            except Exception as e:
                print(f"    {ds_name}: ERROR - {e}")
                eval_results[ds_name] = {"error": str(e)}

        all_results.append({
            "model": "bge-base",
            "lambda_sep": lam_sep,
            "seed": seed,
            "is_baseline": is_baseline,
            "train_time_sec": train_time,
            "eval": eval_results,
        })

        del model, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def evaluate_predictions(predictions, actual_results):
    """Compare predictions vs actuals."""
    pred_vals = []
    actual_vals = []
    G_vals = []

    for r in actual_results:
        for ds_name, metrics in r.get("eval", {}).items():
            if "error" in metrics:
                continue
            G = metrics.get("composite_G", None)
            knn_l1 = metrics.get("knn_l1", None)
            if G is not None and knn_l1 is not None:
                # Find matching prediction
                label = f"bge-base_{ds_name}_lam{r['lambda_sep']}_s{r['seed']}"
                if label in predictions:
                    pred_vals.append(predictions[label]["predicted_knn_l1"])
                    actual_vals.append(knn_l1)
                    G_vals.append(G)

    if len(pred_vals) < 5:
        return {"error": "insufficient matching predictions"}

    pred_arr = np.array(pred_vals)
    actual_arr = np.array(actual_vals)

    rmse = np.sqrt(mean_squared_error(actual_arr, pred_arr))
    r2 = r2_score(actual_arr, pred_arr)
    mae = np.mean(np.abs(actual_arr - pred_arr))

    return {
        "n_predictions": len(pred_vals),
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae),
        "pass_rmse": bool(rmse < 0.10),
        "pass_r2": bool(r2 > 0.3),
        "predictions": [
            {"predicted": float(p), "actual": float(a), "G": float(g)}
            for p, a, g in zip(pred_vals, actual_vals, G_vals)
        ],
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["fit", "evaluate", "both"],
                       default="fit")
    args = parser.parse_args()

    if args.phase in ("fit", "both"):
        print("=" * 60)
        print("  Phase 1: Fit Universal G -> kNN Curve")
        print("=" * 60)

        # Load data
        w2_points = load_week2_data()
        w3_points = load_week3_data()
        print(f"Week 2 points: {len(w2_points)} (no G, sep only)")
        print(f"Week 3 points: {len(w3_points)} (with G)")

        # Only use Week 3 for curve fitting (has proper G values)
        all_points = w3_points

        # Fit curves
        curve_fit = fit_universal_curve(all_points)

        # Test mediation
        print("\n" + "=" * 60)
        print("  Mediation Test: Does G Absorb Model Effects?")
        print("=" * 60)
        mediation = test_mediation(all_points)
        print(f"  R2(G only) = {mediation.get('r2_G_only', 0):.4f}")
        print(f"  R2(G + model) = {mediation.get('r2_G_plus_model', 0):.4f}")
        print(f"  Delta R2 = {mediation.get('delta_r2', 0):.4f}")
        print(f"  F-test p = {mediation.get('f_p', 0):.4f}")
        print(f"  Mediation holds: {mediation.get('mediation_holds', False)}")

        # Save results
        output = {
            "phase": "fit",
            "curve_fit": curve_fit,
            "mediation": mediation,
            "n_week2_points": len(w2_points),
            "n_week3_points": len(w3_points),
        }

        out_path = RESULTS_DIR / "cgp_prospective_fit.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2,
                     default=lambda o: float(o) if hasattr(o, 'item') else str(o))
        print(f"\nSaved to {out_path}")

    if args.phase in ("evaluate", "both"):
        print("\n" + "=" * 60)
        print("  Phase 3: Run Held-Out Conditions")
        print("=" * 60)

        # Load curve fit
        fit_path = RESULTS_DIR / "cgp_prospective_fit.json"
        if not fit_path.exists():
            print("ERROR: Run --phase fit first")
            sys.exit(1)

        with open(fit_path) as f:
            fit_data = json.load(f)

        # Run held-out conditions
        actual_results = run_heldout_conditions()

        # Generate predictions using fitted curve
        curve_fit = fit_data["curve_fit"]
        G_values = {}
        for r in actual_results:
            for ds_name, metrics in r.get("eval", {}).items():
                if "error" not in metrics and "composite_G" in metrics:
                    label = f"bge-base_{ds_name}_lam{r['lambda_sep']}_s{r['seed']}"
                    G_values[label] = metrics["composite_G"]

        predictions = generate_predictions(curve_fit, G_values)

        # Evaluate
        print("\n" + "=" * 60)
        print("  Evaluation: Predictions vs Actuals")
        print("=" * 60)
        eval_results = evaluate_predictions(predictions, actual_results)
        print(f"  RMSE = {eval_results.get('rmse', 0):.4f} (pass: {eval_results.get('pass_rmse', False)})")
        print(f"  R2 = {eval_results.get('r2', 0):.4f} (pass: {eval_results.get('pass_r2', False)})")

        # Save
        output = {
            "phase": "evaluate",
            "curve_fit_used": curve_fit,
            "predictions": predictions,
            "actual_results": actual_results,
            "evaluation": eval_results,
        }

        out_path = RESULTS_DIR / "cgp_prospective_evaluation.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2,
                     default=lambda o: float(o) if hasattr(o, 'item') else str(o))
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
