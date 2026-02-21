#!/usr/bin/env python
"""
cti_fit_universal_law.py

Fit the CTI universal law to checkpoint sweep data.

Universal law (Codex design):
  logit(Q) = b_d + alpha * log(C/N^gamma) - beta * (x - x*)^2
  where x* = mu_0 + mu_1 * log(C/N^gamma)

This predicts:
  1. Quality Q at every layer for every checkpoint
  2. Optimal layer x* shifts with log(compute/capacity)

Success criteria (Codex):
  - Pooled MAE < 0.05
  - Optimal layer l* MAE < 2.5 layers
  - If these pass on fast-path, expand to full fit

Usage:
    python -u src/cti_fit_universal_law.py [--holdout]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_sweep_data(path=None):
    """Load checkpoint sweep results."""
    if path is None:
        path = RESULTS_DIR / "cti_checkpoint_sweep.json"
    with open(path) as f:
        data = json.load(f)
    return data


def extract_observations(data, metric="knn_l0"):
    """Extract (x, log_ratio, Q, dataset, model, step) tuples from sweep data."""
    obs = []
    for result in data["results"]:
        if "error" in result:
            continue
        model = result["model"]
        step = result["step"]
        N = float(result["N_params"])
        C = float(result["C_flops"])
        L = int(result["num_layers"])

        if C <= 0:
            # step 0: use a small epsilon to avoid log(0)
            log_ratio = -40.0  # ~log(1e-17)
        else:
            log_ratio = np.log(C) - 1.0 * np.log(N)  # gamma=1 initial

        for ds_name, ds_data in result["datasets"].items():
            for li_str, layer_data in ds_data["layers"].items():
                li = int(li_str)
                x = layer_data["x"]
                Q = layer_data[metric]

                # Skip extreme values
                if Q <= 0.001 or Q >= 0.999:
                    continue

                obs.append({
                    "x": x,
                    "log_ratio": log_ratio,
                    "Q": Q,
                    "dataset": ds_name,
                    "model": model,
                    "step": step,
                    "layer": li,
                    "L": L,
                    "N": N,
                    "C": C,
                })

    return obs


def universal_law_predict(params, obs_list, gamma=None):
    """Predict Q for each observation using the universal law.

    Parameters:
      params = [alpha, beta, mu_0, mu_1, b_agnews, b_trec, ...]
      gamma is either fixed or part of params
    """
    # Unpack
    if gamma is None:
        alpha, beta, mu_0, mu_1, gam = params[:5]
        b_d = dict(zip(sorted(set(o["dataset"] for o in obs_list)),
                       params[5:]))
    else:
        alpha, beta, mu_0, mu_1 = params[:4]
        gam = gamma
        b_d = dict(zip(sorted(set(o["dataset"] for o in obs_list)),
                       params[4:]))

    predictions = []
    for o in obs_list:
        N = o["N"]
        C = o["C"]
        x = o["x"]
        ds = o["dataset"]

        if C <= 0:
            log_r = -40.0
        else:
            log_r = np.log(C) - gam * np.log(N)

        x_star = mu_0 + mu_1 * log_r
        logit_Q = b_d.get(ds, 0.0) + alpha * log_r - beta * (x - x_star) ** 2

        # Clip to avoid overflow
        logit_Q = np.clip(logit_Q, -20, 20)
        Q_pred = expit(logit_Q)
        predictions.append(Q_pred)

    return np.array(predictions)


def fit_universal_law(obs_list, fix_gamma=None):
    """Fit the universal law parameters via least-squares on logit(Q)."""
    datasets = sorted(set(o["dataset"] for o in obs_list))
    n_ds = len(datasets)
    Q_obs = np.array([o["Q"] for o in obs_list])

    def loss(params):
        Q_pred = universal_law_predict(params, obs_list, gamma=fix_gamma)
        # MSE in probability space
        residuals = Q_obs - Q_pred
        return np.mean(residuals ** 2)

    # Initial guesses
    if fix_gamma is not None:
        # [alpha, beta, mu_0, mu_1, b_d1, b_d2, ...]
        x0 = [0.01, 1.0, 0.5, 0.01] + [0.0] * n_ds
        bounds = [
            (-1, 1),      # alpha
            (0.01, 50),    # beta (positive, quadratic penalty)
            (-2, 2),       # mu_0
            (-0.5, 0.5),   # mu_1
        ] + [(-10, 10)] * n_ds
    else:
        # [alpha, beta, mu_0, mu_1, gamma, b_d1, b_d2, ...]
        x0 = [0.01, 1.0, 0.5, 0.01, 1.0] + [0.0] * n_ds
        bounds = [
            (-1, 1),      # alpha
            (0.01, 50),    # beta
            (-2, 2),       # mu_0
            (-0.5, 0.5),   # mu_1
            (0.5, 2.0),    # gamma
        ] + [(-10, 10)] * n_ds

    # Multiple random restarts
    best_result = None
    best_loss = float("inf")

    for trial in range(20):
        if trial == 0:
            x_init = x0
        else:
            rng = np.random.RandomState(trial)
            x_init = [
                rng.uniform(b[0], b[1]) for b in bounds
            ]

        try:
            result = minimize(loss, x_init, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 5000, "ftol": 1e-12})
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        except Exception:
            continue

    return best_result, datasets


def evaluate_fit(obs_list, params, datasets, fix_gamma=None):
    """Evaluate fit quality: MAE, R2, optimal layer prediction."""
    Q_obs = np.array([o["Q"] for o in obs_list])
    Q_pred = universal_law_predict(params, obs_list, gamma=fix_gamma)

    residuals = Q_obs - Q_pred
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals ** 2)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Q_obs - np.mean(Q_obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Optimal layer prediction
    if fix_gamma is not None:
        alpha, beta, mu_0, mu_1 = params[:4]
        gam = fix_gamma
    else:
        alpha, beta, mu_0, mu_1, gam = params[:5]

    # Group by (model, step, dataset) to find actual vs predicted optimal layer
    groups = {}
    for i, o in enumerate(obs_list):
        key = (o["model"], o["step"], o["dataset"])
        if key not in groups:
            groups[key] = {"obs": [], "pred": [], "layers": [], "L": o["L"]}
        groups[key]["obs"].append(Q_obs[i])
        groups[key]["pred"].append(Q_pred[i])
        groups[key]["layers"].append(o["layer"])

    l_star_errors = []
    l_star_details = []
    for key, g in groups.items():
        model, step, ds = key
        L = g["L"]
        obs_arr = np.array(g["obs"])
        pred_arr = np.array(g["pred"])
        layers = np.array(g["layers"])

        actual_best = layers[np.argmax(obs_arr)]

        N_model = [o for o in obs_list if o["model"] == model][0]["N"]
        C_model = [o for o in obs_list if o["model"] == model and o["step"] == step][0]["C"]

        if C_model <= 0:
            log_r = -40.0
        else:
            log_r = np.log(C_model) - gam * np.log(N_model)

        x_star_pred = mu_0 + mu_1 * log_r
        l_star_pred = x_star_pred * L
        l_star_pred = np.clip(l_star_pred, 0, L)

        error = abs(actual_best - l_star_pred)
        l_star_errors.append(error)
        l_star_details.append({
            "model": model,
            "step": step,
            "dataset": ds,
            "actual_best_layer": int(actual_best),
            "predicted_best_layer": round(float(l_star_pred), 2),
            "error_layers": round(float(error), 2),
            "L": L,
        })

    l_star_mae = np.mean(l_star_errors)

    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "n_obs": len(obs_list),
        "l_star_mae": float(l_star_mae),
        "l_star_details": l_star_details,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", action="store_true",
                       help="Test on holdout data")
    parser.add_argument("--metric", type=str, default="knn_l0",
                       help="Quality metric to fit (knn_l0 or knn_l1)")
    parser.add_argument("--sweep-file", type=str, default=None,
                       help="Path to sweep JSON")
    args = parser.parse_args()

    # Load data
    sweep_path = args.sweep_file or (RESULTS_DIR / "cti_checkpoint_sweep.json")
    data = load_sweep_data(sweep_path)

    print("=" * 70)
    print("  CTI Universal Law Fit")
    print("=" * 70)
    print(f"Metric: {args.metric}")
    print(f"Data: {sweep_path}")

    # Extract observations
    obs = extract_observations(data, metric=args.metric)
    print(f"Total observations: {len(obs)}")

    if len(obs) == 0:
        print("ERROR: No valid observations found!")
        sys.exit(1)

    # Separate models for potential holdout
    models_in_data = sorted(set(o["model"] for o in obs))
    print(f"Models: {models_in_data}")

    datasets_in_data = sorted(set(o["dataset"] for o in obs))
    print(f"Datasets: {datasets_in_data}")

    # Show step 0 stats
    step0 = [o for o in obs if o["step"] == 0]
    if step0:
        q0 = [o["Q"] for o in step0]
        print(f"Step 0 quality: mean={np.mean(q0):.3f}, range=[{np.min(q0):.3f}, {np.max(q0):.3f}]")

    # Fit with gamma fixed at 1.0
    print("\n--- Fit 1: gamma=1.0 (fixed) ---")
    result1, ds_names1 = fit_universal_law(obs, fix_gamma=1.0)
    if result1 is not None:
        print(f"Converged: loss={result1.fun:.6f}")
        alpha, beta, mu_0, mu_1 = result1.x[:4]
        b_d = {ds: result1.x[4 + i] for i, ds in enumerate(ds_names1)}
        print(f"  alpha={alpha:.4f}, beta={beta:.4f}")
        print(f"  mu_0={mu_0:.4f}, mu_1={mu_1:.6f}")
        for ds, b in b_d.items():
            print(f"  b_{ds}={b:.4f}")

        eval1 = evaluate_fit(obs, result1.x, ds_names1, fix_gamma=1.0)
        print(f"\n  MAE = {eval1['mae']:.4f}  (threshold: < 0.05)")
        print(f"  RMSE = {eval1['rmse']:.4f}")
        print(f"  R2 = {eval1['r2']:.4f}")
        print(f"  l* MAE = {eval1['l_star_mae']:.2f} layers  (threshold: < 2.5)")

        # Pass/fail
        pass_mae = eval1["mae"] < 0.05
        pass_lstar = eval1["l_star_mae"] < 2.5
        print(f"\n  PASS MAE: {pass_mae}")
        print(f"  PASS l*:  {pass_lstar}")
        print(f"  OVERALL:  {'GREEN' if pass_mae and pass_lstar else 'RED'}")

    # Fit with gamma free
    print("\n--- Fit 2: gamma free ---")
    result2, ds_names2 = fit_universal_law(obs, fix_gamma=None)
    if result2 is not None:
        print(f"Converged: loss={result2.fun:.6f}")
        alpha, beta, mu_0, mu_1, gamma = result2.x[:5]
        b_d = {ds: result2.x[5 + i] for i, ds in enumerate(ds_names2)}
        print(f"  alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")
        print(f"  mu_0={mu_0:.4f}, mu_1={mu_1:.6f}")
        for ds, b in b_d.items():
            print(f"  b_{ds}={b:.4f}")

        eval2 = evaluate_fit(obs, result2.x, ds_names2, fix_gamma=None)
        print(f"\n  MAE = {eval2['mae']:.4f}  (threshold: < 0.05)")
        print(f"  RMSE = {eval2['rmse']:.4f}")
        print(f"  R2 = {eval2['r2']:.4f}")
        print(f"  l* MAE = {eval2['l_star_mae']:.2f} layers  (threshold: < 2.5)")

        pass_mae = eval2["mae"] < 0.05
        pass_lstar = eval2["l_star_mae"] < 2.5
        print(f"\n  PASS MAE: {pass_mae}")
        print(f"  PASS l*:  {pass_lstar}")
        print(f"  OVERALL:  {'GREEN' if pass_mae and pass_lstar else 'RED'}")

    # Per-model analysis
    print("\n--- Per-Model Breakdown ---")
    for model in models_in_data:
        model_obs = [o for o in obs if o["model"] == model]
        Q_model = np.array([o["Q"] for o in model_obs])
        steps = sorted(set(o["step"] for o in model_obs))

        # Show quality at each step for first dataset
        ds0 = datasets_in_data[0]
        for step in [steps[0], steps[-1]]:
            step_obs = [o for o in model_obs if o["step"] == step and o["dataset"] == ds0]
            if step_obs:
                Qs = [o["Q"] for o in step_obs]
                best_layer = step_obs[np.argmax(Qs)]["layer"]
                print(f"  {model} step={step} {ds0}: best_layer={best_layer}, "
                      f"best_Q={max(Qs):.3f}, final_Q={Qs[-1]:.3f}")

    # Save results
    out = {
        "experiment": "CTI Universal Law Fit",
        "metric": args.metric,
        "n_observations": len(obs),
        "models": models_in_data,
        "datasets": datasets_in_data,
    }

    if result1 is not None:
        out["fit_gamma_fixed"] = {
            "gamma": 1.0,
            "alpha": float(result1.x[0]),
            "beta": float(result1.x[1]),
            "mu_0": float(result1.x[2]),
            "mu_1": float(result1.x[3]),
            "b_d": {ds: float(result1.x[4 + i]) for i, ds in enumerate(ds_names1)},
            "loss": float(result1.fun),
            **eval1,
            "pass_mae": eval1["mae"] < 0.05,
            "pass_lstar": eval1["l_star_mae"] < 2.5,
        }

    if result2 is not None:
        out["fit_gamma_free"] = {
            "gamma": float(result2.x[4]),
            "alpha": float(result2.x[0]),
            "beta": float(result2.x[1]),
            "mu_0": float(result2.x[2]),
            "mu_1": float(result2.x[3]),
            "b_d": {ds: float(result2.x[5 + i]) for i, ds in enumerate(ds_names2)},
            "loss": float(result2.fun),
            **eval2,
            "pass_mae": eval2["mae"] < 0.05,
            "pass_lstar": eval2["l_star_mae"] < 2.5,
        }

    out_path = RESULTS_DIR / "cti_universal_law_fit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                 default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
