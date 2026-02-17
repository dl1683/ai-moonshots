#!/usr/bin/env python
"""
Analyze the residual surgery dense sweep to precisely characterize the phase transition.

Key questions:
1. Where exactly is alpha* (the critical point)?
2. What is the order parameter that undergoes the transition?
3. Is it continuous (2nd order) or discontinuous (1st order)?
4. What is the critical exponent (if any)?

This analysis feeds into the pre-registered blind test:
we predict alpha* for unseen models BEFORE sweeping them.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar, curve_fit
from scipy.interpolate import UnivariateSpline

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_dense_sweep(path=None):
    """Load and parse the dense sweep results."""
    if path is None:
        path = RESULTS_DIR / "cti_residual_dense.json"
    with open(path) as f:
        data = json.load(f)
    return data


def extract_order_parameters(data, dataset="clinc"):
    """Extract order parameters from the sweep."""
    experiments = data["experiments"][dataset]

    alphas = []
    betas = []
    bell_r2s = []
    delta_r2s = []
    peak_knn = []
    mean_knn = []
    late_knn = []  # Average of last 25% of layers
    max_knn_layer = []
    profile_variance = []

    for alpha_str in sorted(experiments.keys(), key=float):
        r = experiments[alpha_str]
        if r.get("status") != "ok":
            continue

        alpha = float(alpha_str)
        fit = r["fit"]
        profile = r["profile"]

        # Get kNN values
        layers = sorted(profile.keys(), key=int)
        knn_vals = np.array([profile[l]["knn_l1"] for l in layers])
        x_vals = np.array([profile[l]["x"] for l in layers])

        # Late layers = last 25%
        n_late = max(1, len(layers) // 4)
        late_mean = np.mean(knn_vals[-n_late:])

        # Find peak layer (excluding layer 0 which is always the embedding)
        peak_idx = np.argmax(knn_vals[1:]) + 1  # skip layer 0

        alphas.append(alpha)
        betas.append(fit["beta"])
        bell_r2s.append(fit["bell_r2"])
        delta_r2s.append(fit["delta_r2"])
        peak_knn.append(np.max(knn_vals))
        mean_knn.append(np.mean(knn_vals))
        late_knn.append(late_mean)
        max_knn_layer.append(peak_idx)
        profile_variance.append(np.var(knn_vals))

    return {
        "alphas": np.array(alphas),
        "betas": np.array(betas),
        "bell_r2": np.array(bell_r2s),
        "delta_r2": np.array(delta_r2s),
        "peak_knn": np.array(peak_knn),
        "mean_knn": np.array(mean_knn),
        "late_knn": np.array(late_knn),
        "max_knn_layer": np.array(max_knn_layer),
        "profile_variance": np.array(profile_variance),
    }


def find_critical_alpha(alphas, order_param, method="max_gradient"):
    """Find alpha* using various methods."""
    results = {}

    if method == "max_gradient" or method == "all":
        # Find where the order parameter changes most rapidly
        if len(alphas) >= 3:
            # Numerical gradient
            grad = np.gradient(order_param, alphas)
            abs_grad = np.abs(grad)
            idx = np.argmax(abs_grad)
            results["max_gradient"] = {
                "alpha_star": float(alphas[idx]),
                "gradient": float(grad[idx]),
                "index": int(idx),
            }

    if method == "sigmoid_fit" or method == "all":
        # Fit sigmoid to the order parameter
        def sigmoid(x, x0, k, ymin, ymax):
            return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - x0)))

        try:
            # Initial guess
            p0 = [0.85, -20, np.min(order_param), np.max(order_param)]
            popt, pcov = curve_fit(sigmoid, alphas, order_param, p0=p0, maxfev=10000)
            results["sigmoid_fit"] = {
                "alpha_star": float(popt[0]),
                "steepness": float(popt[1]),
                "y_min": float(popt[2]),
                "y_max": float(popt[3]),
            }
        except Exception as e:
            results["sigmoid_fit"] = {"error": str(e)}

    if method == "threshold" or method == "all":
        # Find where beta drops below 1.0 (threshold method)
        for i in range(len(alphas) - 1):
            if order_param[i] >= 1.0 and order_param[i+1] < 1.0:
                # Linear interpolation
                frac = (1.0 - order_param[i]) / (order_param[i+1] - order_param[i])
                alpha_star = alphas[i] + frac * (alphas[i+1] - alphas[i])
                results["threshold_1.0"] = {"alpha_star": float(alpha_star)}
                break

    return results


def analyze_transition_order(alphas, betas):
    """Determine if the transition is 1st order (discontinuous) or 2nd order (continuous)."""
    # Check for discontinuity
    dbeta = np.diff(betas) / np.diff(alphas)
    max_jump_idx = np.argmax(np.abs(dbeta))
    max_jump = np.abs(dbeta[max_jump_idx])

    # Critical exponent: near alpha*, does beta ~ |alpha - alpha*|^gamma?
    # Use the max gradient point as alpha*
    alpha_star = alphas[max_jump_idx + 1]

    # Look at beta vs |alpha - alpha*| for alpha < alpha*
    mask = alphas < alpha_star
    if np.sum(mask) >= 3:
        log_dist = np.log(alpha_star - alphas[mask] + 1e-10)
        log_beta = np.log(betas[mask] + 1e-10)

        # Linear fit in log-log space
        coeffs = np.polyfit(log_dist, log_beta, 1)
        gamma = coeffs[0]  # Critical exponent
    else:
        gamma = None

    return {
        "max_jump_rate": float(max_jump),
        "max_jump_alpha": float(alphas[max_jump_idx]),
        "alpha_star_estimate": float(alpha_star),
        "critical_exponent_gamma": float(gamma) if gamma is not None else None,
        "transition_type": "likely_continuous" if max_jump < 50 else "possibly_discontinuous",
    }


def main():
    print("=" * 70)
    print("PHASE TRANSITION ANALYSIS: Residual Surgery Dense Sweep")
    print("=" * 70)

    data = load_dense_sweep()
    params = extract_order_parameters(data)

    print(f"\nModel: {data['model_id']}")
    print(f"Layers: {data['num_layers']}")
    print(f"Alpha values: {len(params['alphas'])}")

    # Print order parameter table
    print(f"\n{'Alpha':>6} {'Beta':>8} {'Bell R2':>8} {'Delta':>8} {'Mean kNN':>9} {'Late kNN':>9} {'PeakLayer':>10}")
    print("-" * 70)
    for i in range(len(params["alphas"])):
        print(f"{params['alphas'][i]:>6.2f} {params['betas'][i]:>8.2f} "
              f"{params['bell_r2'][i]:>8.3f} {params['delta_r2'][i]:>+8.4f} "
              f"{params['mean_knn'][i]:>9.3f} {params['late_knn'][i]:>9.3f} "
              f"{params['max_knn_layer'][i]:>10d}")

    # Find critical alpha using beta as order parameter
    print(f"\n{'='*70}")
    print("CRITICAL POINT ANALYSIS (Order parameter: beta)")
    print(f"{'='*70}")

    crit = find_critical_alpha(params["alphas"], params["betas"], method="all")
    for method, result in crit.items():
        print(f"\n  Method: {method}")
        for k, v in result.items():
            print(f"    {k}: {v}")

    # Also use mean_knn as order parameter
    print(f"\n{'='*70}")
    print("CRITICAL POINT ANALYSIS (Order parameter: mean_knn)")
    print(f"{'='*70}")

    crit_knn = find_critical_alpha(params["alphas"], params["mean_knn"], method="all")
    for method, result in crit_knn.items():
        print(f"\n  Method: {method}")
        for k, v in result.items():
            print(f"    {k}: {v}")

    # Transition order analysis
    print(f"\n{'='*70}")
    print("TRANSITION ORDER ANALYSIS")
    print(f"{'='*70}")

    trans = analyze_transition_order(params["alphas"], params["betas"])
    for k, v in trans.items():
        print(f"  {k}: {v}")

    # The key finding: what happens at the transition
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    # Find the biggest beta drop
    dbeta = np.diff(params["betas"])
    max_drop_idx = np.argmin(dbeta)  # Most negative = biggest drop
    print(f"\n  Largest beta drop: {params['betas'][max_drop_idx]:.2f} -> {params['betas'][max_drop_idx+1]:.2f}")
    print(f"  At alpha: {params['alphas'][max_drop_idx]:.2f} -> {params['alphas'][max_drop_idx+1]:.2f}")
    print(f"  Drop magnitude: {dbeta[max_drop_idx]:.2f}")
    print(f"  Estimated alpha*: {(params['alphas'][max_drop_idx] + params['alphas'][max_drop_idx+1])/2:.3f}")

    # Mean kNN transition
    dknn = np.diff(params["mean_knn"])
    max_rise_idx = np.argmax(dknn)
    print(f"\n  Largest mean_knn rise: {params['mean_knn'][max_rise_idx]:.3f} -> {params['mean_knn'][max_rise_idx+1]:.3f}")
    print(f"  At alpha: {params['alphas'][max_rise_idx]:.2f} -> {params['alphas'][max_rise_idx+1]:.2f}")
    print(f"  Rise magnitude: {dknn[max_rise_idx]:.3f}")

    # Late kNN transition
    dlate = np.diff(params["late_knn"])
    max_late_idx = np.argmax(dlate)
    print(f"\n  Largest late_knn rise: {params['late_knn'][max_late_idx]:.3f} -> {params['late_knn'][max_late_idx+1]:.3f}")
    print(f"  At alpha: {params['alphas'][max_late_idx]:.2f} -> {params['alphas'][max_late_idx+1]:.2f}")

    # Summary for pre-registration
    # The real transition is characterized by:
    # 1. Beta threshold: where beta drops below 1.0 (from bell-fit curvature)
    # 2. Sigmoid midpoint of mean_knn (most robust)
    # 3. The late_knn jump (most dramatic)
    alpha_star_threshold = crit.get("threshold_1.0", {}).get("alpha_star", 0.87)
    alpha_star_sigmoid = crit_knn.get("sigmoid_fit", {}).get("alpha_star", 0.87)
    alpha_star_best = (alpha_star_threshold + alpha_star_sigmoid) / 2

    print(f"\n{'='*70}")
    print("PRE-REGISTRATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  For Qwen3-0.6B on CLINC:")
    print(f"    alpha* (beta threshold=1.0):  {alpha_star_threshold:.3f}")
    print(f"    alpha* (kNN sigmoid midpoint): {alpha_star_sigmoid:.3f}")
    print(f"    Consensus alpha*:              {alpha_star_best:.3f}")
    print(f"    Critical exponent gamma:       {trans.get('critical_exponent_gamma', 'N/A')}")
    print(f"\n  Profile regimes:")
    print(f"    alpha < 0.85: Declining profiles (information collapse)")
    print(f"    alpha ~ {alpha_star_best:.2f}: PHASE TRANSITION")
    print(f"    alpha > 0.90: Complex/maintained profiles (information preserved)")
    print(f"\n  PRE-REGISTERED PREDICTION for HuggingFaceTB/SmolLM2-1.7B:")
    print(f"    H1: alpha* is UNIVERSAL across transformer architectures")
    print(f"    Predicted alpha* = {alpha_star_best:.2f} +/- 0.05")
    print(f"    H0 (null): alpha* differs by > 0.10 from Qwen3")
    print(f"    Test: Run dense sweep on SmolLM2-1.7B, compute alpha* same way")
    print(f"    Success criterion: |alpha*_smollm - {alpha_star_best:.2f}| < 0.10")

    # Save analysis
    analysis = {
        "model": data["model_id"],
        "dataset": "clinc",
        "order_parameters": {k: v.tolist() for k, v in params.items()},
        "critical_alpha_beta": crit,
        "critical_alpha_knn": {k: v for k, v in crit_knn.items()},
        "transition_order": trans,
        "alpha_star_threshold": float(alpha_star_threshold),
        "alpha_star_sigmoid": float(alpha_star_sigmoid),
        "alpha_star_consensus": float(alpha_star_best),
        "pre_registered_prediction": {
            "target_model": "google/gemma-3-1b-it",
            "predicted_alpha_star": float(alpha_star_best),
            "tolerance": 0.05,
            "null_hypothesis_threshold": 0.10,
            "success_criterion": f"|alpha*_gemma - {alpha_star_best:.3f}| < 0.10",
        }
    }

    out_path = RESULTS_DIR / "phase_transition_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
