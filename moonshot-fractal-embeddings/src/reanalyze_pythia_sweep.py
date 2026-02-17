#!/usr/bin/env python
"""
Re-analyze Pythia depth sweep with SCALE-INDEPENDENT transition metrics.

The beta=1.0 threshold failed because Pythia models have lower absolute
representation quality than modern models. We need metrics that detect
the transition RELATIVE to each model's quality ceiling.

Scale-independent order parameters:
1. Normalized mean_knn: mean_knn(alpha) / mean_knn(alpha=1.0)
2. Alpha at which 50% of quality is preserved (alpha_50)
3. Max derivative of normalized quality (steepest transition)
4. Profile variance ratio: var(profile@alpha) / var(profile@alpha=1)
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_all_sweeps():
    """Load all available sweep results."""
    sweeps = {}

    # Pythia models
    pythia_path = RESULTS_DIR / "cti_pythia_depth_sweep.json"
    if pythia_path.exists():
        with open(pythia_path) as f:
            pythia_data = json.load(f)
        for model_id, result in pythia_data["results"].items():
            sweeps[model_id] = {
                "num_layers": result["num_layers"],
                "n_params": result["n_params"],
                "family": "pythia",
                "fits": result["fits"],
            }

    # Original 3 models (full profiles available)
    for filename, family in [
        ("cti_residual_dense.json", "qwen3"),
        ("cti_residual_smollm2.json", "smollm2"),
        ("cti_residual_olmo2.json", "olmo2"),
    ]:
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        model_id = data["model_id"]
        sweeps[model_id] = {
            "num_layers": data["num_layers"],
            "n_params": data["n_params"],
            "family": family,
            "experiments": data["experiments"],
        }

    return sweeps


def extract_mean_knn_curve(sweep_data, dataset="clinc"):
    """Extract mean_knn as function of alpha."""
    alphas = []
    mean_knns = []

    if "experiments" in sweep_data:
        # Full profile data (original models)
        experiments = sweep_data["experiments"].get(dataset, {})
        for alpha_str in sorted(experiments.keys(), key=float):
            r = experiments[alpha_str]
            if r.get("status") != "ok":
                continue
            profile = r["profile"]
            layers = sorted(profile.keys(), key=int)
            knn_vals = [profile[l]["knn_l1"] for l in layers]
            alphas.append(float(alpha_str))
            mean_knns.append(np.mean(knn_vals))
    elif "fits" in sweep_data:
        # Only fit data (Pythia models) - can't compute mean_knn
        # Use beta as proxy
        return None, None

    if not alphas:
        return None, None
    return np.array(alphas), np.array(mean_knns)


def extract_beta_curve(sweep_data, dataset="clinc"):
    """Extract beta as function of alpha."""
    alphas = []
    betas = []

    if "experiments" in sweep_data:
        experiments = sweep_data["experiments"].get(dataset, {})
        for alpha_str in sorted(experiments.keys(), key=float):
            r = experiments[alpha_str]
            if r.get("status") != "ok" or not r.get("fit"):
                continue
            alphas.append(float(alpha_str))
            betas.append(r["fit"]["beta"])
    elif "fits" in sweep_data:
        fits = sweep_data["fits"].get(dataset, {})
        for alpha_str in sorted(fits.keys(), key=float):
            alphas.append(float(alpha_str))
            betas.append(fits[alpha_str]["beta"])

    if not alphas:
        return None, None
    return np.array(alphas), np.array(betas)


def find_transition_scale_free(alphas, values):
    """Find transition point using scale-free methods."""
    results = {}

    # Normalize to [0, 1] range
    v_min, v_max = np.min(values), np.max(values)
    if v_max - v_min < 1e-10:
        return {"normalized_range": 0, "alpha_50": None, "max_deriv_alpha": None}

    v_norm = (values - v_min) / (v_max - v_min)

    # Method 1: Alpha at 50% of range (midpoint of transition)
    alpha_50 = None
    for i in range(len(alphas) - 1):
        if v_norm[i] <= 0.5 and v_norm[i+1] > 0.5:
            frac = (0.5 - v_norm[i]) / (v_norm[i+1] - v_norm[i])
            alpha_50 = alphas[i] + frac * (alphas[i+1] - alphas[i])
            break
    results["alpha_50"] = float(alpha_50) if alpha_50 is not None else None

    # Method 2: Maximum derivative (steepest point)
    if len(alphas) >= 3:
        deriv = np.gradient(v_norm, alphas)
        max_deriv_idx = np.argmax(np.abs(deriv))
        results["max_deriv_alpha"] = float(alphas[max_deriv_idx])
        results["max_deriv_value"] = float(deriv[max_deriv_idx])

    # Method 3: Sigmoid fit
    try:
        def sigmoid(x, x0, k, ymin, ymax):
            return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - x0)))
        p0 = [0.8, 10, np.min(values), np.max(values)]
        popt, _ = curve_fit(sigmoid, alphas, values, p0=p0, maxfev=10000)
        results["sigmoid_midpoint"] = float(popt[0])
        results["sigmoid_steepness"] = float(popt[1])
    except Exception:
        results["sigmoid_midpoint"] = None

    # Normalized range (how much variation exists)
    results["normalized_range"] = float(v_max - v_min)
    results["quality_at_alpha1"] = float(values[-1]) if len(values) > 0 else None

    return results


def main():
    print("=" * 70)
    print("SCALE-INDEPENDENT PHASE TRANSITION ANALYSIS")
    print("=" * 70)

    sweeps = load_all_sweeps()
    print(f"\nLoaded {len(sweeps)} models")

    # For each model, compute transition metrics using beta
    print(f"\n{'Model':<35} {'L':>3} {'Family':>8} {'beta_range':>10} {'alpha_50':>9} "
          f"{'max_d_a':>8} {'sigm_mid':>9}")
    print("-" * 95)

    depth_transition = []  # (depth, alpha_50, model_id, family)

    for model_id, data in sorted(sweeps.items(), key=lambda x: x[1]["num_layers"]):
        L = data["num_layers"]
        family = data["family"]

        alphas, betas = extract_beta_curve(data, "clinc")
        if alphas is None or len(alphas) < 3:
            print(f"{model_id:<35} {L:>3} {family:>8} -- NO DATA --")
            continue

        metrics = find_transition_scale_free(alphas, betas)

        a50 = metrics.get("alpha_50")
        max_da = metrics.get("max_deriv_alpha")
        sig = metrics.get("sigmoid_midpoint")
        br = metrics.get("normalized_range", 0)

        print(f"{model_id:<35} {L:>3} {family:>8} {br:>10.3f} "
              f"{f'{a50:.3f}' if a50 is not None else 'N/A':>9} "
              f"{f'{max_da:.3f}' if max_da is not None else 'N/A':>8} "
              f"{f'{sig:.3f}' if sig is not None else 'N/A':>9}")

        # Use best available transition estimate
        best_alpha = a50 or sig or max_da
        if best_alpha is not None and br > 0.1:  # Only include if meaningful variation
            depth_transition.append((L, best_alpha, model_id, family))

    # Scaling law analysis
    if len(depth_transition) >= 3:
        print(f"\n{'='*70}")
        print("SCALING LAW (scale-free alpha_50)")
        print(f"{'='*70}")

        depths = np.array([d[0] for d in depth_transition])
        alpha_50s = np.array([d[1] for d in depth_transition])
        names = [d[2] for d in depth_transition]
        families = [d[3] for d in depth_transition]

        for i in range(len(depths)):
            print(f"  L={depths[i]:>3}, alpha_50={alpha_50s[i]:.3f}  ({names[i]}, {families[i]})")

        if len(set(depths)) >= 3:
            rho, p_rho = spearmanr(depths, alpha_50s)
            r, p_r = pearsonr(depths, alpha_50s)
            print(f"\n  Spearman rho = {rho:.3f} (p = {p_rho:.4f})")
            print(f"  Pearson r = {r:.3f} (p = {p_r:.4f})")

        # Try power law
        try:
            def power_law(L, a, b):
                return 1 - a * L ** (-b)
            popt, _ = curve_fit(power_law, depths, alpha_50s, p0=[1, 1], maxfev=10000)
            pred = power_law(depths, *popt)
            ss_res = np.sum((alpha_50s - pred) ** 2)
            ss_tot = np.sum((alpha_50s - alpha_50s.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"\n  Power law: alpha_50 = 1 - {popt[0]:.3f} * L^(-{popt[1]:.3f})")
            print(f"  R2 = {r2:.4f}")
        except Exception as e:
            print(f"\n  Power law fit failed: {e}")

        # Separate Pythia from other models
        pythia_mask = np.array([f == "pythia" for f in families])
        other_mask = ~pythia_mask

        if np.sum(pythia_mask) >= 2:
            print(f"\n  Pythia only (n={np.sum(pythia_mask)}):")
            if len(set(depths[pythia_mask])) >= 2:
                rho_p, p_p = spearmanr(depths[pythia_mask], alpha_50s[pythia_mask])
                print(f"    Spearman rho = {rho_p:.3f} (p = {p_p:.4f})")

        if np.sum(other_mask) >= 2:
            print(f"\n  Modern models only (n={np.sum(other_mask)}):")
            if len(set(depths[other_mask])) >= 2:
                rho_o, p_o = spearmanr(depths[other_mask], alpha_50s[other_mask])
                print(f"    Spearman rho = {rho_o:.3f} (p = {p_o:.4f})")

    # Save
    out = {
        "analysis": "scale_independent_transition",
        "depth_transition": [
            {"depth": int(d[0]), "alpha_50": float(d[1]), "model": d[2], "family": d[3]}
            for d in depth_transition
        ],
    }
    out_path = RESULTS_DIR / "scale_free_transition_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
