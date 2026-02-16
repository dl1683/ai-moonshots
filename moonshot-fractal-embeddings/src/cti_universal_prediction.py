#!/usr/bin/env python
"""CTI Universal Prediction: test Pythia-fitted law across 5 model families.

This is the comprehensive cross-family validation:
1. Load Pythia-fitted parameters (from holdout prediction)
2. Load depth profiles from OLMo-2, Cerebras-GPT, OPT, GPT-2
3. Predict quality for each model at each layer (ZERO target-family fitting)
4. Compute per-family MAE, R2, shape correlation
5. Leave-one-family-out cross-validation
6. Summary table for paper

Families (5 total, 13 models):
  - Pythia: 160M, 410M, 1B, 1.4B (training source)
  - OLMo-2: 1B (11 checkpoints)
  - Cerebras-GPT: 256M, 590M, 1.3B
  - OPT: 350M, 1.3B
  - GPT-2: 124M, 355M, 774M
"""

from __future__ import annotations

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_pythia_observations(exclude_step0=True):
    """Load Pythia checkpoint sweep data."""
    path = RESULTS_DIR / "cti_checkpoint_sweep_all.json"
    with open(path) as f:
        data = json.load(f)

    obs = []
    for result in data["results"]:
        if "error" in result:
            continue
        if exclude_step0 and result["step"] == 0:
            continue

        model = result["model"]
        N = float(result["N_params"])
        C = float(result["C_flops"])
        L = int(result["num_layers"])
        step = result["step"]

        for ds_name, ds_data in result["datasets"].items():
            n_classes = DS_CLASSES.get(ds_name, 100)
            Q_chance = 1.0 / n_classes

            for li_str, layer_data in ds_data["layers"].items():
                li = int(li_str)
                x = li / L
                Q_raw = layer_data["knn_l1"]
                Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)

                obs.append({
                    "x": x, "Q": Q_norm, "dataset": ds_name,
                    "model": model, "family": "pythia",
                    "step": step, "layer": li, "L": L,
                    "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                })
    return obs


def load_olmo2_observations():
    """Load OLMo-2 sweep data."""
    path = RESULTS_DIR / "cti_olmo2_sweep.json"
    with open(path) as f:
        data = json.load(f)

    obs = []
    for result in data["results"]:
        if "error" in result:
            continue

        model = result["model"]
        N = float(result["N_params"])
        C = float(result["C_flops"])
        L = int(result["num_layers"])
        step = result["step"]

        for ds_name, ds_data in result["datasets"].items():
            n_classes = DS_CLASSES.get(ds_name, 100)
            Q_chance = 1.0 / n_classes

            for li_str, layer_data in ds_data["layers"].items():
                li = int(li_str)
                x = li / L
                Q_raw = layer_data["knn_l1"]
                Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)

                obs.append({
                    "x": x, "Q": Q_norm, "dataset": ds_name,
                    "model": model, "family": "olmo2",
                    "step": step, "layer": li, "L": L,
                    "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                })
    return obs


def load_multi_family_observations():
    """Load multi-family (Cerebras-GPT, OPT, GPT-2) depth profile data."""
    path = RESULTS_DIR / "cti_multi_family.json"
    with open(path) as f:
        data = json.load(f)

    obs = []
    for result in data["results"]:
        if "error" in result:
            continue

        model = result["model"]
        family = result["family"]
        N = float(result["N_params"])
        C = float(result["C_flops"])
        L = int(result["num_layers"])

        for ds_name, ds_data in result["datasets"].items():
            n_classes = DS_CLASSES.get(ds_name, 100)
            Q_chance = 1.0 / n_classes

            for li_str, layer_data in ds_data["layers"].items():
                li = int(li_str)
                x = li / L
                Q_raw = layer_data["knn_l1"]
                Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)

                obs.append({
                    "x": x, "Q": Q_norm, "dataset": ds_name,
                    "model": model, "family": family,
                    "step": -1, "layer": li, "L": L,
                    "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                })
    return obs


def predict_ds(params, obs_list, ds_list):
    """Predict Q_norm given fitted parameters."""
    alpha, beta, mu_0, mu_1 = params[:4]
    b_d = {ds: params[4 + i] for i, ds in enumerate(ds_list)}
    preds = []
    for o in obs_list:
        x_star = mu_0 + mu_1 * o["log_r"]
        logit_Q = b_d.get(o["dataset"], 0) + alpha * o["log_r"] - beta * (o["x"] - x_star) ** 2
        preds.append(expit(np.clip(logit_Q, -20, 20)))
    return np.array(preds)


def fit_gaussian(train_obs, datasets, n_restarts=20):
    """Fit Gaussian bell + dataset offsets."""
    Q_tr = np.array([o["Q"] for o in train_obs])
    n_ds = len(datasets)
    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds

    def loss(params):
        return np.mean((Q_tr - predict_ds(params, train_obs, datasets)) ** 2)

    best, best_loss = None, float("inf")
    for trial in range(n_restarts):
        rng = np.random.RandomState(trial)
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_loss:
                best_loss = res.fun
                best = res
        except Exception:
            continue
    return best


def eval_fit(Q_obs, Q_pred):
    """Compute MAE and R2."""
    residuals = Q_obs - Q_pred
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return mae, r2


def shape_correlation(obs_list, pred_array):
    """Compute Spearman correlation between observed and predicted rank profiles
    per (model, dataset, step) group."""
    groups = {}
    for i, o in enumerate(obs_list):
        key = (o["model"], o["dataset"], o["step"])
        if key not in groups:
            groups[key] = []
        groups[key].append((o["x"], o["Q"], pred_array[i]))

    rhos = []
    for key, entries in groups.items():
        if len(entries) < 4:
            continue
        entries.sort(key=lambda e: e[0])
        obs_vals = [e[1] for e in entries]
        pred_vals = [e[2] for e in entries]
        rho, _ = spearmanr(obs_vals, pred_vals)
        if not np.isnan(rho):
            rhos.append(rho)

    return rhos


def main():
    # Load all data
    pythia_obs = load_pythia_observations()
    olmo2_obs = load_olmo2_observations()
    multi_obs = load_multi_family_observations()
    all_obs = pythia_obs + olmo2_obs + multi_obs

    datasets = sorted(set(o["dataset"] for o in all_obs))
    families = sorted(set(o["family"] for o in all_obs))

    print("=" * 70)
    print("  CTI UNIVERSAL PREDICTION: 5-FAMILY ANALYSIS")
    print("=" * 70)
    print(f"Total observations: {len(all_obs)}")
    print(f"Families: {families}")
    for fam in families:
        fam_obs = [o for o in all_obs if o["family"] == fam]
        models = sorted(set(o["model"] for o in fam_obs))
        print(f"  {fam}: {len(fam_obs)} obs, models={models}")

    # === PART 1: Pythia-fitted params -> predict ALL other families ===
    print("\n" + "=" * 70)
    print("  PART 1: PYTHIA-FITTED PARAMS -> ALL FAMILIES (zero-shot)")
    print("=" * 70)

    # Load Pythia params from holdout prediction
    with open(RESULTS_DIR / "cti_holdout_prediction.json") as f:
        holdout = json.load(f)
    pythia_params = holdout["fit_params"]
    p = [pythia_params["alpha"], pythia_params["beta"],
         pythia_params["mu_0"], pythia_params["mu_1"]]
    for ds in datasets:
        p.append(pythia_params["b_d"][ds])
    p = np.array(p)

    # Predict for each non-Pythia family
    non_pythia = [o for o in all_obs if o["family"] != "pythia"]
    Q_obs_all = np.array([o["Q"] for o in non_pythia])
    Q_pred_all = predict_ds(p, non_pythia, datasets)

    overall_mae, overall_r2 = eval_fit(Q_obs_all, Q_pred_all)
    shape_rhos = shape_correlation(non_pythia, Q_pred_all)

    print(f"\nOverall (all non-Pythia): MAE={overall_mae:.4f}, R2={overall_r2:.4f}, N={len(non_pythia)}")
    print(f"Shape correlation: mean_rho={np.mean(shape_rhos):.3f}, median={np.median(shape_rhos):.3f}, "
          f"frac>0.7={np.mean(np.array(shape_rhos)>0.7):.3f}")

    # Per-family breakdown
    per_family_results = {}
    for fam in families:
        if fam == "pythia":
            continue
        fam_obs = [o for o in non_pythia if o["family"] == fam]
        fam_Q = np.array([o["Q"] for o in fam_obs])
        fam_pred = predict_ds(p, fam_obs, datasets)
        mae, r2 = eval_fit(fam_Q, fam_pred)
        fam_rhos = shape_correlation(fam_obs, fam_pred)

        per_family_results[fam] = {
            "mae": mae, "r2": r2, "n": len(fam_obs),
            "shape_mean_rho": float(np.mean(fam_rhos)) if fam_rhos else 0,
            "shape_frac_above_07": float(np.mean(np.array(fam_rhos) > 0.7)) if fam_rhos else 0,
        }
        print(f"  {fam:15s}: MAE={mae:.4f}, R2={r2:.4f}, N={len(fam_obs)}, "
              f"shape_rho={np.mean(fam_rhos):.3f}" if fam_rhos else "  no shape data")

    # Per-dataset breakdown
    per_dataset_results = {}
    for ds in datasets:
        ds_obs = [o for o in non_pythia if o["dataset"] == ds]
        ds_Q = np.array([o["Q"] for o in ds_obs])
        ds_pred = predict_ds(p, ds_obs, datasets)
        mae, r2 = eval_fit(ds_Q, ds_pred)
        per_dataset_results[ds] = {"mae": mae, "r2": r2, "n": len(ds_obs)}
        print(f"  {ds:20s}: MAE={mae:.4f}, R2={r2:.4f}, N={len(ds_obs)}")

    # === PART 2: LEAVE-ONE-FAMILY-OUT CROSS-VALIDATION ===
    print("\n" + "=" * 70)
    print("  PART 2: LEAVE-ONE-FAMILY-OUT CROSS-VALIDATION")
    print("=" * 70)

    lofo_results = {}
    for holdout_fam in families:
        train_obs = [o for o in all_obs if o["family"] != holdout_fam]
        test_obs = [o for o in all_obs if o["family"] == holdout_fam]

        if not train_obs or not test_obs:
            continue

        # Fit on all other families
        fit = fit_gaussian(train_obs, datasets, n_restarts=20)
        if fit is None:
            print(f"  {holdout_fam}: FIT FAILED")
            continue

        Q_test = np.array([o["Q"] for o in test_obs])
        Q_pred = predict_ds(fit.x, test_obs, datasets)
        mae, r2 = eval_fit(Q_test, Q_pred)
        test_rhos = shape_correlation(test_obs, Q_pred)

        params_dict = {
            "alpha": float(fit.x[0]),
            "beta": float(fit.x[1]),
            "mu_0": float(fit.x[2]),
            "mu_1": float(fit.x[3]),
        }
        for i, ds in enumerate(datasets):
            params_dict[f"b_{ds}"] = float(fit.x[4 + i])

        lofo_results[holdout_fam] = {
            "mae": mae, "r2": r2, "n": len(test_obs),
            "n_train": len(train_obs),
            "shape_mean_rho": float(np.mean(test_rhos)) if test_rhos else 0,
            "fit_params": params_dict,
            "fit_mse": float(fit.fun),
        }

        print(f"  Holdout={holdout_fam:15s}: MAE={mae:.4f}, R2={r2:.4f}, "
              f"N_test={len(test_obs)}, N_train={len(train_obs)}, "
              f"shape_rho={np.mean(test_rhos):.3f}" if test_rhos else "")

    # Summary
    print("\n" + "=" * 70)
    print("  LEAVE-ONE-FAMILY-OUT SUMMARY")
    print("=" * 70)

    lofo_maes = [v["mae"] for v in lofo_results.values()]
    lofo_r2s = [v["r2"] for v in lofo_results.values()]
    lofo_rhos = [v["shape_mean_rho"] for v in lofo_results.values()]

    print(f"  Mean MAE: {np.mean(lofo_maes):.4f} +/- {np.std(lofo_maes):.4f}")
    print(f"  Mean R2:  {np.mean(lofo_r2s):.4f} +/- {np.std(lofo_r2s):.4f}")
    print(f"  Mean shape rho: {np.mean(lofo_rhos):.3f}")
    print(f"  All R2 > 0: {all(r2 > 0 for r2 in lofo_r2s)}")
    print(f"  All R2 > 0.5: {all(r2 > 0.5 for r2 in lofo_r2s)}")

    # === PART 3: DEGRADATION UNIVERSALITY ===
    print("\n" + "=" * 70)
    print("  PART 3: LATE-TRAINING DEGRADATION UNIVERSALITY")
    print("=" * 70)

    # For multi-family models (final checkpoint only), check best_x < 1.0
    degradation_count = 0
    total_profiles = 0
    degradation_by_family = {}

    for fam in families:
        fam_obs = [o for o in all_obs if o["family"] == fam]
        models_in_fam = sorted(set(o["model"] for o in fam_obs))
        steps_per_model = {}
        for o in fam_obs:
            key = (o["model"], o["step"])
            if key not in steps_per_model:
                steps_per_model[key] = {}
            ds = o["dataset"]
            if ds not in steps_per_model[key]:
                steps_per_model[key][ds] = []
            steps_per_model[key][ds].append(o)

        fam_degrade = 0
        fam_total = 0

        for (model, step), ds_dict in steps_per_model.items():
            for ds, obs_list in ds_dict.items():
                obs_list.sort(key=lambda o: o["x"])
                best_x = max(obs_list, key=lambda o: o["Q"])["x"]
                final_Q = obs_list[-1]["Q"]
                best_Q = max(o["Q"] for o in obs_list)
                gap = best_Q - final_Q

                fam_total += 1
                total_profiles += 1
                if gap > 0.01 and best_x < 0.95:
                    fam_degrade += 1
                    degradation_count += 1

        degradation_by_family[fam] = {
            "degraded": fam_degrade,
            "total": fam_total,
            "frac": fam_degrade / fam_total if fam_total > 0 else 0,
        }
        print(f"  {fam:15s}: {fam_degrade}/{fam_total} profiles show degradation "
              f"({fam_degrade/fam_total*100:.0f}%)")

    print(f"\n  Overall: {degradation_count}/{total_profiles} "
          f"({degradation_count/total_profiles*100:.0f}%)")

    # === PART 4: PARAMETER STABILITY ACROSS FOLDS ===
    print("\n" + "=" * 70)
    print("  PART 4: FITTED PARAMETER STABILITY")
    print("=" * 70)

    if lofo_results:
        param_names = ["alpha", "beta", "mu_0", "mu_1"]
        for pname in param_names:
            vals = [v["fit_params"][pname] for v in lofo_results.values()]
            print(f"  {pname:6s}: mean={np.mean(vals):+.4f}, std={np.std(vals):.4f}, "
                  f"range=[{min(vals):.4f}, {max(vals):.4f}]")

    # === SAVE RESULTS ===
    output = {
        "experiment": "CTI Universal 5-Family Prediction",
        "families": families,
        "total_observations": len(all_obs),
        "pythia_zero_shot": {
            "description": "Pythia-fitted params applied to all non-Pythia families",
            "overall": {"mae": overall_mae, "r2": overall_r2, "n": len(non_pythia)},
            "shape_correlation": {
                "mean_rho": float(np.mean(shape_rhos)),
                "median_rho": float(np.median(shape_rhos)),
                "frac_above_07": float(np.mean(np.array(shape_rhos) > 0.7)),
            },
            "per_family": per_family_results,
            "per_dataset": per_dataset_results,
        },
        "leave_one_family_out": {
            "per_family": lofo_results,
            "summary": {
                "mean_mae": float(np.mean(lofo_maes)),
                "std_mae": float(np.std(lofo_maes)),
                "mean_r2": float(np.mean(lofo_r2s)),
                "std_r2": float(np.std(lofo_r2s)),
                "mean_shape_rho": float(np.mean(lofo_rhos)),
                "all_r2_above_zero": all(r2 > 0 for r2 in lofo_r2s),
                "all_r2_above_05": all(r2 > 0.5 for r2 in lofo_r2s),
            },
        },
        "degradation_universality": {
            "total_degraded": degradation_count,
            "total_profiles": total_profiles,
            "frac_degraded": degradation_count / total_profiles if total_profiles > 0 else 0,
            "per_family": degradation_by_family,
        },
    }

    out_path = RESULTS_DIR / "cti_universal_prediction.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\nSaved to {out_path}")

    # === FINAL VERDICT ===
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)

    mean_r2 = np.mean(lofo_r2s)
    mean_rho = np.mean(lofo_rhos)

    if mean_r2 > 0.7 and mean_rho > 0.8:
        print("  STRONG: Cross-family transfer works well (R2>0.7, rho>0.8)")
    elif mean_r2 > 0.5 and mean_rho > 0.7:
        print("  MODERATE: Cross-family transfer has signal (R2>0.5, rho>0.7)")
    elif mean_r2 > 0.3:
        print("  WEAK: Some cross-family signal but noisy (R2>0.3)")
    else:
        print("  FAIL: Cross-family transfer does not work (R2<0.3)")

    print(f"  LOFO Mean R2 = {mean_r2:.4f}")
    print(f"  LOFO Mean shape rho = {mean_rho:.3f}")
    print(f"  Degradation universality = {degradation_count}/{total_profiles} "
          f"({degradation_count/total_profiles*100:.0f}%)")


if __name__ == "__main__":
    main()
