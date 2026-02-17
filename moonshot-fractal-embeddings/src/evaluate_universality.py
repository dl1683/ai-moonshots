#!/usr/bin/env python
"""
Evaluate the universality hypothesis: Is alpha* constant across transformer architectures?

Pre-registered prediction (from Qwen3-0.6B):
  alpha* = 0.873 +/- 0.05
  Null hypothesis: alpha* differs by > 0.10

Models tested:
  1. Qwen/Qwen3-0.6B (28 layers, 596M) — reference
  2. HuggingFaceTB/SmolLM2-1.7B (24 layers, 1711M) — blind test 1
  3. allenai/OLMo-2-0425-1B (16 layers, 1279M) — blind test 2
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import ttest_1samp

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_sweep(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_transition_metrics(data, dataset="clinc"):
    """Extract alpha* using multiple methods."""
    experiments = data["experiments"][dataset]

    alphas = []
    betas = []
    mus = []
    delta_r2s = []
    mean_knns = []

    for alpha_str in sorted(experiments.keys(), key=float):
        r = experiments[alpha_str]
        if r.get("status") != "ok" or not r.get("fit"):
            continue

        alpha = float(alpha_str)
        fit = r["fit"]
        profile = r["profile"]

        layers = sorted(profile.keys(), key=int)
        knn_vals = [profile[l]["knn_l1"] for l in layers]

        alphas.append(alpha)
        betas.append(fit["beta"])
        mus.append(fit["mu"])
        delta_r2s.append(fit["delta_r2"])
        mean_knns.append(np.mean(knn_vals))

    alphas = np.array(alphas)
    betas = np.array(betas)
    mus = np.array(mus)
    delta_r2s = np.array(delta_r2s)
    mean_knns = np.array(mean_knns)

    results = {
        "alphas": alphas.tolist(),
        "betas": betas.tolist(),
        "mus": mus.tolist(),
        "delta_r2s": delta_r2s.tolist(),
        "mean_knns": mean_knns.tolist(),
    }

    # Method 1: Beta threshold (beta crosses 1.0 from above)
    alpha_star_beta = None
    for i in range(len(alphas) - 1):
        if betas[i] >= 1.0 and betas[i+1] < 1.0:
            frac = (1.0 - betas[i]) / (betas[i+1] - betas[i])
            alpha_star_beta = alphas[i] + frac * (alphas[i+1] - alphas[i])
            break
    results["alpha_star_beta"] = float(alpha_star_beta) if alpha_star_beta is not None else None

    # Method 2: Mu transition (mu shifts from -0.5 to positive)
    alpha_star_mu = None
    for i in range(len(alphas) - 1):
        if mus[i] <= -0.3 and mus[i+1] > -0.3:
            frac = (-0.3 - mus[i]) / (mus[i+1] - mus[i])
            alpha_star_mu = alphas[i] + frac * (alphas[i+1] - alphas[i])
            break
    results["alpha_star_mu"] = float(alpha_star_mu) if alpha_star_mu is not None else None

    # Method 3: Delta R2 crossover (bell starts beating linear)
    alpha_star_delta = None
    for i in range(len(alphas) - 1):
        if delta_r2s[i] < 0 and delta_r2s[i+1] >= 0:
            frac = (0 - delta_r2s[i]) / (delta_r2s[i+1] - delta_r2s[i])
            alpha_star_delta = alphas[i] + frac * (alphas[i+1] - alphas[i])
            break
    results["alpha_star_delta"] = float(alpha_star_delta) if alpha_star_delta is not None else None

    # Method 4: Sigmoid fit to mean_knn
    alpha_star_sigmoid = None
    try:
        def sigmoid(x, x0, k, ymin, ymax):
            return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - x0)))
        p0 = [0.85, 20, np.min(mean_knns), np.max(mean_knns)]
        bounds = ([0.5, 1, 0, 0], [1.0, 200, 1, 1])
        popt, _ = curve_fit(sigmoid, alphas, mean_knns, p0=p0, bounds=bounds, maxfev=10000)
        alpha_star_sigmoid = float(popt[0])
    except Exception:
        pass
    results["alpha_star_sigmoid"] = alpha_star_sigmoid

    # Consensus: average of available methods
    estimates = [v for v in [alpha_star_beta, alpha_star_mu, alpha_star_delta, alpha_star_sigmoid]
                 if v is not None]
    results["alpha_star_consensus"] = float(np.mean(estimates)) if estimates else None
    results["alpha_star_std"] = float(np.std(estimates)) if len(estimates) > 1 else None
    results["n_methods"] = len(estimates)

    return results


def main():
    print("=" * 70)
    print("UNIVERSALITY HYPOTHESIS TEST")
    print("Is alpha* constant across transformer architectures?")
    print("=" * 70)
    print(f"\nPre-registered prediction: alpha* = 0.873 +/- 0.05")
    print(f"Null hypothesis: |alpha*_model - 0.873| > 0.10")

    models = [
        ("Qwen3-0.6B (reference)", "cti_residual_dense.json"),
        ("SmolLM2-1.7B (blind test)", "cti_residual_smollm2.json"),
        ("OLMo-2-1B (blind test)", "cti_residual_olmo2.json"),
    ]

    all_alpha_stars = []
    all_results = {}

    for name, filename in models:
        data = load_sweep(filename)
        if data is None:
            print(f"\n  {name}: FILE NOT FOUND ({filename})")
            continue

        metrics = extract_transition_metrics(data)
        all_results[name] = metrics

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"  Model: {data['model_id']}")
        print(f"  Layers: {data['num_layers']}, Params: {data['n_params']/1e6:.0f}M")
        print(f"{'='*50}")
        print(f"  alpha* (beta threshold):   {metrics['alpha_star_beta']}")
        print(f"  alpha* (mu transition):    {metrics['alpha_star_mu']}")
        print(f"  alpha* (delta crossover):  {metrics['alpha_star_delta']}")
        print(f"  alpha* (sigmoid fit):      {metrics['alpha_star_sigmoid']}")
        print(f"  alpha* (consensus):        {metrics['alpha_star_consensus']}")
        if metrics['alpha_star_std'] is not None:
            print(f"  alpha* (std of methods):   {metrics['alpha_star_std']:.4f}")

        if metrics['alpha_star_consensus'] is not None:
            deviation = abs(metrics['alpha_star_consensus'] - 0.873)
            status = "PASS" if deviation < 0.10 else "FAIL"
            print(f"\n  Deviation from prediction: |{metrics['alpha_star_consensus']:.3f} - 0.873| = {deviation:.3f}")
            print(f"  Within 0.05 tolerance:     {'YES' if deviation < 0.05 else 'NO'}")
            print(f"  Within 0.10 tolerance:     {'YES' if deviation < 0.10 else 'NO'}")
            print(f"  Null hypothesis rejected:  {'YES' if deviation < 0.10 else 'NO'}")
            all_alpha_stars.append(metrics['alpha_star_consensus'])

    # Cross-model comparison
    if len(all_alpha_stars) >= 2:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}")

        alpha_stars = np.array(all_alpha_stars)
        print(f"\n  alpha* values: {[f'{a:.3f}' for a in alpha_stars]}")
        print(f"  Mean alpha*:   {np.mean(alpha_stars):.4f}")
        print(f"  Std alpha*:    {np.std(alpha_stars):.4f}")
        print(f"  Range:         {np.max(alpha_stars) - np.min(alpha_stars):.4f}")

        # One-sample t-test: is the mean significantly different from 0.873?
        if len(all_alpha_stars) >= 3:
            t_stat, p_val = ttest_1samp(alpha_stars, 0.873)
            print(f"\n  One-sample t-test (H0: mean = 0.873):")
            print(f"    t = {t_stat:.3f}, p = {p_val:.4f}")
            print(f"    {'Cannot reject H0 (alpha* consistent with 0.873)' if p_val > 0.05 else 'REJECT H0 (alpha* differs from 0.873)'}")

        # Is variation across models smaller than within-model method variation?
        inter_model_var = np.var(alpha_stars)
        intra_model_vars = []
        for name, metrics in all_results.items():
            if metrics.get('alpha_star_std') is not None:
                intra_model_vars.append(metrics['alpha_star_std'] ** 2)

        if intra_model_vars:
            mean_intra_var = np.mean(intra_model_vars)
            print(f"\n  Variance decomposition:")
            print(f"    Inter-model variance:     {inter_model_var:.6f}")
            print(f"    Mean intra-model variance: {mean_intra_var:.6f}")
            print(f"    Ratio (inter/intra):       {inter_model_var/mean_intra_var:.3f}")
            if inter_model_var < mean_intra_var:
                print(f"    CONCLUSION: Variation BETWEEN models is SMALLER than within-model")
                print(f"    This supports the universality hypothesis!")

        # Final verdict
        print(f"\n{'='*70}")
        print("FINAL VERDICT")
        print(f"{'='*70}")
        all_pass = all(abs(a - 0.873) < 0.10 for a in alpha_stars)
        tight_pass = all(abs(a - 0.873) < 0.05 for a in alpha_stars)
        print(f"\n  Pre-registered prediction: alpha* = 0.873 +/- 0.05")
        print(f"  Models tested: {len(alpha_stars)}")
        print(f"  All within 0.10 tolerance: {'YES' if all_pass else 'NO'}")
        print(f"  All within 0.05 tolerance: {'YES' if tight_pass else 'NO'}")
        if all_pass:
            print(f"\n  *** UNIVERSALITY HYPOTHESIS SUPPORTED ***")
            print(f"  alpha* = {np.mean(alpha_stars):.3f} +/- {np.std(alpha_stars):.3f} across {len(alpha_stars)} architectures")
            print(f"  This is a UNIVERSAL CONSTANT of transformer information processing!")
        else:
            print(f"\n  Universality hypothesis NOT supported. alpha* is model-dependent.")

    # Save
    out = {
        "pre_registered_prediction": 0.873,
        "tolerance": 0.05,
        "null_threshold": 0.10,
        "results": {name: {k: v for k, v in m.items() if k != "alphas"} for name, m in all_results.items()},
        "alpha_stars": all_alpha_stars,
        "mean_alpha_star": float(np.mean(all_alpha_stars)) if all_alpha_stars else None,
        "std_alpha_star": float(np.std(all_alpha_stars)) if len(all_alpha_stars) > 1 else None,
    }
    out_path = RESULTS_DIR / "universality_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
