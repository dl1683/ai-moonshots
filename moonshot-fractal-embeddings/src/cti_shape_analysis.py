#!/usr/bin/env python
"""Analyze the universal quality profile shape across models and datasets."""

import json
import numpy as np
from numpy.polynomial import polynomial as P
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    with open(RESULTS_DIR / "cti_checkpoint_sweep_all.json") as f:
        data = json.load(f)

    x_common = np.linspace(0, 1, 21)
    all_profiles = []

    for result in data["results"]:
        if "error" in result or result["step"] == 0:
            continue
        L = int(result["num_layers"])

        for ds_name, ds_data in result["datasets"].items():
            layers = ds_data["layers"]
            qs = []
            xs = []
            for li in sorted(layers.keys(), key=int):
                qs.append(layers[li]["knn_l1"])
                xs.append(layers[li]["x"])

            qs = np.array(qs)
            xs = np.array(xs)
            q_min, q_max = qs.min(), qs.max()
            if q_max > q_min:
                qs_norm = (qs - q_min) / (q_max - q_min)
                y = np.interp(x_common, xs, qs_norm)
                all_profiles.append(y)

    all_profiles = np.array(all_profiles)
    mean_profile = np.mean(all_profiles, axis=0)
    std_profile = np.std(all_profiles, axis=0)

    print("=== Universal Quality Profile Shape ===")
    print("x (normalized depth) | mean normalized Q | std")
    for i, x in enumerate(x_common):
        print(f"  {x:.2f}                  {mean_profile[i]:.3f}            {std_profile[i]:.3f}")

    # Fit polynomial
    coeffs2 = P.polyfit(x_common, mean_profile, 2)
    y_fit2 = P.polyval(x_common, coeffs2)
    r2_2 = 1 - np.sum((mean_profile - y_fit2) ** 2) / np.sum((mean_profile - mean_profile.mean()) ** 2)

    coeffs3 = P.polyfit(x_common, mean_profile, 3)
    y_fit3 = P.polyval(x_common, coeffs3)
    r2_3 = 1 - np.sum((mean_profile - y_fit3) ** 2) / np.sum((mean_profile - mean_profile.mean()) ** 2)

    print(f"\nQuadratic: Q(x) = {coeffs2[0]:.4f} + {coeffs2[1]:.4f}*x + {coeffs2[2]:.4f}*x^2, R2={r2_2:.4f}")
    print(f"Cubic: R2={r2_3:.4f}")

    print(f"\nProfile characteristics:")
    print(f"  At x=0 (embedding): {mean_profile[0]:.3f} +/- {std_profile[0]:.3f}")
    print(f"  At x=0.5 (mid):     {mean_profile[10]:.3f} +/- {std_profile[10]:.3f}")
    print(f"  At x=1.0 (final):   {mean_profile[20]:.3f} +/- {std_profile[20]:.3f}")
    print(f"  Peak position:      x={x_common[np.argmax(mean_profile)]:.2f}")

    # Profile evolution with training
    print("\n=== Profile Evolution with Training ===")
    steps = sorted(set(r["step"] for r in data["results"]
                       if "error" not in r and r["step"] > 0))

    for step in [steps[0], steps[len(steps) // 4], steps[len(steps) // 2], steps[-1]]:
        step_profiles = []
        for result in data["results"]:
            if "error" in result or result["step"] != step:
                continue
            L = int(result["num_layers"])
            for ds_name, ds_data in result["datasets"].items():
                layers = ds_data["layers"]
                qs = []
                xs = []
                for li in sorted(layers.keys(), key=int):
                    qs.append(layers[li]["knn_l1"])
                    xs.append(layers[li]["x"])
                qs = np.array(qs)
                xs = np.array(xs)
                q_min, q_max = qs.min(), qs.max()
                if q_max > q_min:
                    qs_norm = (qs - q_min) / (q_max - q_min)
                    y = np.interp(x_common, xs, qs_norm)
                    step_profiles.append(y)

        if step_profiles:
            mean_sp = np.mean(step_profiles, axis=0)
            peak_x = x_common[np.argmax(mean_sp)]
            print(f"  Step {step:>6d}: peak_x={peak_x:.2f}, "
                  f"profile at [0,0.5,1.0] = [{mean_sp[0]:.3f}, {mean_sp[10]:.3f}, {mean_sp[20]:.3f}]")

    # The key result: shape universality stats
    print("\n=== SHAPE UNIVERSALITY SUMMARY ===")
    from scipy.stats import spearmanr

    # Cross-model shape correlation
    models = sorted(set(r["model"] for r in data["results"] if "error" not in r))
    datasets = sorted(set(ds for r in data["results"] if "error" not in r
                          for ds in r.get("datasets", {}).keys()))

    profiles_dict = {}
    for result in data["results"]:
        if "error" in result or result["step"] == 0:
            continue
        model = result["model"]
        step = result["step"]
        L = int(result["num_layers"])
        for ds_name, ds_data in result["datasets"].items():
            layers = ds_data["layers"]
            qs = []
            xs = []
            for li in sorted(layers.keys(), key=int):
                qs.append(layers[li]["knn_l1"])
                xs.append(layers[li]["x"])
            qs = np.array(qs)
            xs = np.array(xs)
            q_min, q_max = qs.min(), qs.max()
            if q_max > q_min:
                qs_norm = (qs - q_min) / (q_max - q_min)
                profiles_dict[(model, step, ds_name)] = {
                    "xs": xs, "qs_norm": qs_norm, "L": L
                }

    # All pairwise model comparisons at same (step, dataset)
    cross_model_rhos = []
    for step in steps:
        for ds in datasets:
            interp = {}
            for model in models:
                key = (model, step, ds)
                if key in profiles_dict:
                    p = profiles_dict[key]
                    interp[model] = np.interp(x_common, p["xs"], p["qs_norm"])

            model_list = sorted(interp.keys())
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    rho, _ = spearmanr(interp[model_list[i]], interp[model_list[j]])
                    cross_model_rhos.append(rho)

    # All pairwise dataset comparisons at same (model, step)
    cross_ds_rhos = []
    for step in steps:
        for model in models:
            interp = {}
            for ds in datasets:
                key = (model, step, ds)
                if key in profiles_dict:
                    p = profiles_dict[key]
                    interp[ds] = np.interp(x_common, p["xs"], p["qs_norm"])

            ds_list = sorted(interp.keys())
            for i in range(len(ds_list)):
                for j in range(i + 1, len(ds_list)):
                    rho, _ = spearmanr(interp[ds_list[i]], interp[ds_list[j]])
                    cross_ds_rhos.append(rho)

    cross_model_rhos = np.array(cross_model_rhos)
    cross_ds_rhos = np.array(cross_ds_rhos)

    print(f"Cross-model shape correlation:")
    print(f"  Mean rho = {cross_model_rhos.mean():.3f}, Median = {np.median(cross_model_rhos):.3f}")
    print(f"  Min = {cross_model_rhos.min():.3f}, >0.7: {np.mean(cross_model_rhos > 0.7):.1%}")

    print(f"Cross-dataset shape correlation:")
    print(f"  Mean rho = {cross_ds_rhos.mean():.3f}, Median = {np.median(cross_ds_rhos):.3f}")
    print(f"  Min = {cross_ds_rhos.min():.3f}, >0.7: {np.mean(cross_ds_rhos > 0.7):.1%}")

    # Save
    out = {
        "universal_profile": {
            "x": x_common.tolist(),
            "mean": mean_profile.tolist(),
            "std": std_profile.tolist(),
        },
        "quadratic_fit": {
            "coeffs": coeffs2.tolist(),
            "r2": float(r2_2),
        },
        "cross_model_shape_rho": {
            "mean": float(cross_model_rhos.mean()),
            "median": float(np.median(cross_model_rhos)),
            "min": float(cross_model_rhos.min()),
            "frac_above_07": float(np.mean(cross_model_rhos > 0.7)),
            "n": int(len(cross_model_rhos)),
        },
        "cross_dataset_shape_rho": {
            "mean": float(cross_ds_rhos.mean()),
            "median": float(np.median(cross_ds_rhos)),
            "min": float(cross_ds_rhos.min()),
            "frac_above_07": float(np.mean(cross_ds_rhos > 0.7)),
            "n": int(len(cross_ds_rhos)),
        },
    }

    out_path = RESULTS_DIR / "cti_shape_universality.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
