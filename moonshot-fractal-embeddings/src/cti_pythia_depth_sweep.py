#!/usr/bin/env python
"""
CTI Pythia Depth Sweep: Test scaling law alpha* = 1 - a*L^(-b) using
the Pythia family (same training recipe, different depths).

This is the highest-impact experiment per Codex review:
"Controlled depth sweep in one family (same recipe, many depths, many seeds)"

Models: pythia-{70m, 160m, 410m, 1b, 1.4b, 2.8b}
Depths: 6, 12, 24, 16, 24, 32
Alpha values: 13 points (0.0 to 1.0)
Datasets: clinc, trec (two independent measurements)
"""

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
]

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
DATASETS = ["clinc", "trec"]
MAX_SAMPLES = 2000
BATCH_SIZE = 32


def run_model_sweep(model_id, alphas, datasets, max_samples, batch_size, device="cuda"):
    """Run full alpha sweep for one model."""
    from cti_residual_surgery import (
        load_model, extract_depth_profile, fit_bell,
        load_hierarchical_dataset
    )
    from hierarchical_datasets import load_hierarchical_dataset

    print(f"\n{'#'*70}")
    print(f"MODEL: {model_id}")
    print(f"{'#'*70}")

    model, tokenizer, num_layers, n_params = load_model(model_id, device)

    model_results = {
        "model_id": model_id,
        "num_layers": num_layers,
        "n_params": int(n_params),
        "experiments": {},
    }

    for ds_name in datasets:
        print(f"\n  Dataset: {ds_name}")
        try:
            data = load_hierarchical_dataset(ds_name, split="test", max_samples=max_samples)
        except Exception as e:
            print(f"    SKIP ({e})")
            continue

        texts = [s.text for s in data.samples]
        l1_labels = np.array([s.level1_label for s in data.samples])
        n_classes = len(data.level1_names)

        ds_results = {}
        for alpha in alphas:
            t0 = time.time()
            profile = extract_depth_profile(
                model, tokenizer, texts, l1_labels, n_classes,
                alpha=alpha, batch_size=batch_size, device=device,
            )
            if profile is None:
                ds_results[str(alpha)] = {"status": "error"}
                continue

            xs = np.array([profile[l]["x"] for l in sorted(profile.keys())])
            qs = np.array([profile[l]["Q_norm"] for l in sorted(profile.keys())])
            fit = fit_bell(xs, qs)
            elapsed = time.time() - t0

            if fit:
                print(f"    a={alpha:.2f}: bell={fit['bell_r2']:.3f} mu={fit['mu']:.2f} "
                      f"beta={fit['beta']:.2f} delta={fit['delta_r2']:+.3f} [{elapsed:.1f}s]")
            ds_results[str(alpha)] = {
                "status": "ok",
                "profile": {str(k): v for k, v in profile.items()},
                "fit": fit,
                "runtime_sec": round(elapsed, 1),
            }

        model_results["experiments"][ds_name] = ds_results

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return model_results


def compute_alpha_star(model_results, dataset="clinc"):
    """Compute alpha* from a model's sweep results."""
    experiments = model_results["experiments"].get(dataset, {})
    if not experiments:
        return None

    alphas = []
    betas = []
    for alpha_str in sorted(experiments.keys(), key=float):
        r = experiments[alpha_str]
        if r.get("status") != "ok" or not r.get("fit"):
            continue
        alphas.append(float(alpha_str))
        betas.append(r["fit"]["beta"])

    if len(alphas) < 3:
        return None

    alphas = np.array(alphas)
    betas = np.array(betas)

    # Method: beta threshold (beta crosses 1.0)
    for i in range(len(alphas) - 1):
        if betas[i] >= 1.0 and betas[i+1] < 1.0:
            frac = (1.0 - betas[i]) / (betas[i+1] - betas[i])
            return float(alphas[i] + frac * (alphas[i+1] - alphas[i]))

    # If beta never crosses 1.0, try finding where it drops most
    if max(betas) > 1.0:
        # Beta starts above 1 but never crosses cleanly
        return None
    elif max(betas) < 1.0:
        # Beta never reaches 1.0 — transition is at very low alpha
        # Use the point where beta is at its maximum
        return float(alphas[np.argmax(betas)])

    return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("PYTHIA DEPTH SWEEP: Scaling Law Validation")
    print("alpha* = 1 - a * L^(-b)")
    print("=" * 70)
    print(f"Models: {len(PYTHIA_MODELS)}")
    print(f"Alphas: {len(ALPHAS)}")
    print(f"Datasets: {DATASETS}")
    print(f"Device: {device}")

    all_results = {}
    for model_id in PYTHIA_MODELS:
        result = run_model_sweep(model_id, ALPHAS, DATASETS, MAX_SAMPLES, BATCH_SIZE, device)
        all_results[model_id] = result

    # Compute alpha* for each model
    print(f"\n{'='*70}")
    print("SCALING LAW RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Layers':>6} {'alpha*(clinc)':>14} {'alpha*(trec)':>14}")
    print("-" * 70)

    depth_alpha_pairs = []
    for model_id, result in all_results.items():
        L = result["num_layers"]
        a_clinc = compute_alpha_star(result, "clinc")
        a_trec = compute_alpha_star(result, "trec")
        print(f"{model_id:<30} {L:>6} "
              f"{f'{a_clinc:.3f}' if a_clinc is not None else 'N/A':>14} "
              f"{f'{a_trec:.3f}' if a_trec is not None else 'N/A':>14}")

        if a_clinc is not None:
            depth_alpha_pairs.append((L, a_clinc, model_id))

    # Fit scaling law
    if len(depth_alpha_pairs) >= 3:
        from scipy.optimize import curve_fit
        from scipy.stats import spearmanr, pearsonr

        depths = np.array([p[0] for p in depth_alpha_pairs])
        alpha_stars = np.array([p[1] for p in depth_alpha_pairs])

        rho, p_rho = spearmanr(depths, alpha_stars)
        r, p_r = pearsonr(depths, alpha_stars)

        print(f"\nCorrelation (depth vs alpha*):")
        print(f"  Spearman rho = {rho:.3f} (p = {p_rho:.4f})")
        print(f"  Pearson r = {r:.3f} (p = {p_r:.4f})")

        # Power law fit: alpha* = 1 - a * L^(-b)
        def power_law(L, a, b):
            return 1 - a * L ** (-b)

        try:
            popt, pcov = curve_fit(power_law, depths, alpha_stars, p0=[1, 1], maxfev=10000)
            pred = power_law(depths, *popt)
            ss_res = np.sum((alpha_stars - pred) ** 2)
            ss_tot = np.sum((alpha_stars - alpha_stars.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            print(f"\nPower law fit: alpha* = 1 - {popt[0]:.3f} * L^(-{popt[1]:.3f})")
            print(f"  R2 = {r2:.4f}")
            print(f"  Residuals: {[f'{r:.3f}' for r in (alpha_stars - pred)]}")

            # Predictions
            print(f"\nPredictions:")
            for L_test in [48, 64, 96, 128]:
                print(f"  L={L_test}: alpha* = {power_law(L_test, *popt):.3f}")
        except Exception as e:
            print(f"\nPower law fit failed: {e}")
            popt = None

        # Linear fit
        coeffs = np.polyfit(depths, alpha_stars, 1)
        pred_lin = np.polyval(coeffs, depths)
        r2_lin = 1 - np.sum((alpha_stars - pred_lin)**2) / ss_tot if ss_tot > 0 else 0
        print(f"\nLinear fit: alpha* = {coeffs[0]:.5f} * L + {coeffs[1]:.3f}")
        print(f"  R2 = {r2_lin:.4f}")

    # Save all results
    output = {
        "experiment": "pythia_depth_sweep",
        "models": PYTHIA_MODELS,
        "alphas": ALPHAS,
        "datasets": DATASETS,
        "results": {},
        "scaling_law": {},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    for model_id, result in all_results.items():
        # Slim down profiles to save space
        slim_result = {
            "model_id": result["model_id"],
            "num_layers": result["num_layers"],
            "n_params": result["n_params"],
            "fits": {},
        }
        for ds_name, ds_res in result["experiments"].items():
            ds_fits = {}
            for alpha_str, r in ds_res.items():
                if r.get("status") == "ok" and r.get("fit"):
                    ds_fits[alpha_str] = r["fit"]
            slim_result["fits"][ds_name] = ds_fits
        output["results"][model_id] = slim_result

    if depth_alpha_pairs:
        output["scaling_law"] = {
            "depths": [int(p[0]) for p in depth_alpha_pairs],
            "alpha_stars": [float(p[1]) for p in depth_alpha_pairs],
            "models": [p[2] for p in depth_alpha_pairs],
        }

    out_path = RESULTS_DIR / "cti_pythia_depth_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
