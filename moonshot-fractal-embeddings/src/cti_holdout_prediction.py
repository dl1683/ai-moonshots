#!/usr/bin/env python
"""CTI 1.4B Holdout Prediction Test.

Pre-registered falsification criteria:
  - Cross-model MAE must stay under 0.05
  - Cross-model R2 must stay above 0.7

Fits Gaussian+b_d model on 160M+410M+1B, predicts 1.4B quality profiles.
Also tests shape universality: does 1.4B profile shape correlate with training models?
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}

TRAIN_MODELS = {"pythia-160m", "pythia-410m", "pythia-1b"}
HOLDOUT_MODEL = "pythia-1.4b"

# Pre-registered thresholds
MAE_THRESHOLD = 0.05
R2_THRESHOLD = 0.70


def load_observations(path, exclude_step0=True, metric="knn_l1"):
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
                Q_raw = layer_data[metric]
                Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)

                obs.append({
                    "x": x, "Q": Q_norm, "Q_raw": Q_raw, "dataset": ds_name,
                    "model": model, "step": step, "layer": li, "L": L,
                    "N": N, "C": C,
                })
    return obs


def predict_ds(params, obs_list, ds_list):
    alpha, beta, mu_0, mu_1 = params[:4]
    b_d = {ds: params[4 + i] for i, ds in enumerate(ds_list)}
    preds = []
    for o in obs_list:
        log_r = np.log(o["C"]) - np.log(o["N"])
        x_star = mu_0 + mu_1 * log_r
        logit_Q = b_d.get(o["dataset"], 0) + alpha * log_r - beta * (o["x"] - x_star) ** 2
        preds.append(expit(np.clip(logit_Q, -20, 20)))
    return np.array(preds)


def fit_model(obs_list, Q_obs, datasets, n_restarts=30):
    n_ds = len(datasets)
    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds

    def loss(params):
        return np.mean((Q_obs - predict_ds(params, obs_list, datasets)) ** 2)

    best, best_loss = None, float("inf")
    for trial in range(n_restarts):
        rng = np.random.RandomState(trial)
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 5000, "ftol": 1e-12})
            if res.fun < best_loss:
                best_loss = res.fun
                best = res
        except Exception:
            continue
    return best


def eval_fit(Q_obs, Q_pred):
    residuals = Q_obs - Q_pred
    mae = np.mean(np.abs(residuals))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((Q_obs - Q_obs.mean()) ** 2)
    return mae, r2


def main():
    path = RESULTS_DIR / "cti_checkpoint_sweep_all.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run merge first.")
        return

    all_obs = load_observations(path)
    datasets = sorted(set(o["dataset"] for o in all_obs))

    train_obs = [o for o in all_obs if o["model"] in TRAIN_MODELS]
    test_obs = [o for o in all_obs if o["model"] == HOLDOUT_MODEL]

    Q_train = np.array([o["Q"] for o in train_obs])
    Q_test = np.array([o["Q"] for o in test_obs])

    print("=" * 70)
    print("  CTI 1.4B HOLDOUT PREDICTION TEST")
    print("=" * 70)
    print(f"Training models: {sorted(TRAIN_MODELS)}")
    print(f"Holdout model: {HOLDOUT_MODEL}")
    print(f"Training observations: {len(train_obs)}")
    print(f"Test observations: {len(test_obs)}")
    print(f"Datasets: {datasets}")

    # ========================
    # FIT ON TRAINING MODELS
    # ========================
    print("\n--- Fitting Gaussian+b_d on training models ---")
    result = fit_model(train_obs, Q_train, datasets)
    alpha, beta, mu_0, mu_1 = result.x[:4]
    print(f"alpha={alpha:.4f}, beta={beta:.4f}, mu_0={mu_0:.4f}, mu_1={mu_1:.6f}")
    for i, ds in enumerate(datasets):
        print(f"  b_{ds}={result.x[4+i]:.4f}")

    Q_train_pred = predict_ds(result.x, train_obs, datasets)
    mae_train, r2_train = eval_fit(Q_train, Q_train_pred)
    print(f"Training fit: MAE={mae_train:.4f}, R2={r2_train:.4f}")

    # ========================
    # PREDICT ON 1.4B
    # ========================
    print("\n--- Predicting 1.4B (holdout) ---")
    Q_test_pred = predict_ds(result.x, test_obs, datasets)
    mae_test, r2_test = eval_fit(Q_test, Q_test_pred)
    print(f"Holdout prediction: MAE={mae_test:.4f}, R2={r2_test:.4f}")

    # Per-dataset breakdown
    print("\nPer-dataset holdout results:")
    ds_results = {}
    for ds in datasets:
        mask = np.array([o["dataset"] == ds for o in test_obs])
        if mask.sum() == 0:
            continue
        Q_te = Q_test[mask]
        Q_pr = Q_test_pred[mask]
        mae_d, r2_d = eval_fit(Q_te, Q_pr)
        ds_results[ds] = {"mae": float(mae_d), "r2": float(r2_d), "n": int(mask.sum())}
        print(f"  {ds:>16s}: MAE={mae_d:.4f}, R2={r2_d:.4f} (n={mask.sum()})")

    # ========================
    # FALSIFICATION CRITERIA
    # ========================
    print("\n" + "=" * 70)
    print("  FALSIFICATION CRITERIA")
    print("=" * 70)
    mae_pass = mae_test < MAE_THRESHOLD
    r2_pass = r2_test > R2_THRESHOLD
    print(f"  MAE = {mae_test:.4f} < {MAE_THRESHOLD} ? {'PASS' if mae_pass else 'FAIL'}")
    print(f"  R2  = {r2_test:.4f} > {R2_THRESHOLD} ? {'PASS' if r2_pass else 'FAIL'}")
    overall = "PASS" if (mae_pass and r2_pass) else "FAIL"
    print(f"  Overall: {overall}")

    # ========================
    # SHAPE UNIVERSALITY TEST
    # ========================
    print("\n" + "=" * 70)
    print("  SHAPE UNIVERSALITY: Training vs 1.4B")
    print("=" * 70)

    # Compute normalized quality profiles for each (model, step, dataset)
    x_common = np.linspace(0, 1, 21)
    profiles = {}
    for o in all_obs:
        key = (o["model"], o["step"], o["dataset"])
        if key not in profiles:
            profiles[key] = []
        profiles[key].append((o["x"], o["Q"]))

    # Interpolate to common grid
    interp_profiles = {}
    for key, pts in profiles.items():
        pts.sort()
        xs = np.array([p[0] for p in pts])
        qs = np.array([p[1] for p in pts])
        if qs.max() > qs.min():
            qs_norm = (qs - qs.min()) / (qs.max() - qs.min())
            interp_profiles[key] = np.interp(x_common, xs, qs_norm)

    # Cross-model shape: 1.4B vs each training model
    shape_rhos = []
    steps = sorted(set(o["step"] for o in all_obs))
    for step in steps:
        for ds in datasets:
            holdout_key = (HOLDOUT_MODEL, step, ds)
            if holdout_key not in interp_profiles:
                continue
            holdout_prof = interp_profiles[holdout_key]

            for tm in sorted(TRAIN_MODELS):
                train_key = (tm, step, ds)
                if train_key not in interp_profiles:
                    continue
                rho, _ = spearmanr(holdout_prof, interp_profiles[train_key])
                shape_rhos.append({
                    "train_model": tm, "step": step, "dataset": ds, "rho": float(rho)
                })

    if shape_rhos:
        rhos = np.array([r["rho"] for r in shape_rhos])
        print(f"1.4B vs training models shape correlation:")
        print(f"  Mean rho = {rhos.mean():.3f}")
        print(f"  Median rho = {np.median(rhos):.3f}")
        print(f"  Min rho = {rhos.min():.3f}")
        print(f"  >0.7: {np.mean(rhos > 0.7):.1%}")
        print(f"  n comparisons = {len(rhos)}")

        # Per-dataset
        for ds in datasets:
            ds_rhos = [r["rho"] for r in shape_rhos if r["dataset"] == ds]
            if ds_rhos:
                print(f"  {ds}: mean_rho={np.mean(ds_rhos):.3f}, min={np.min(ds_rhos):.3f}")

    # ========================
    # OPTIMAL LAYER PREDICTION
    # ========================
    print("\n" + "=" * 70)
    print("  OPTIMAL LAYER PREDICTION FOR 1.4B")
    print("=" * 70)

    final_step = 143000
    test_final = [o for o in test_obs if o["step"] == final_step]
    if test_final:
        L = test_final[0]["L"]
        N = test_final[0]["N"]
        C = test_final[0]["C"]
        log_r = np.log(C) - np.log(N)
        x_star = mu_0 + mu_1 * log_r
        l_star = np.clip(x_star * L, 0, L)

        for ds in datasets:
            ds_obs = [o for o in test_final if o["dataset"] == ds]
            if not ds_obs:
                continue
            Qs = np.array([o["Q"] for o in ds_obs])
            layers = np.array([o["layer"] for o in ds_obs])
            actual_best = layers[np.argmax(Qs)]
            error = abs(actual_best - l_star)
            print(f"  {ds:>16s}: actual=L{actual_best}/{L}, predicted=L{l_star:.1f}, error={error:.1f}")

    # ========================
    # LATE-TRAINING DEGRADATION
    # ========================
    print("\n" + "=" * 70)
    print("  LATE-TRAINING QUALITY DEGRADATION (1.4B)")
    print("=" * 70)

    for ds in datasets:
        print(f"\n  {ds}:")
        for step in sorted(steps):
            step_obs = [o for o in test_obs if o["step"] == step and o["dataset"] == ds]
            if not step_obs:
                continue
            Qs = np.array([o["Q"] for o in step_obs])
            layers = np.array([o["layer"] for o in step_obs])
            L = step_obs[0]["L"]
            best_layer = layers[np.argmax(Qs)]
            final_q = Qs[layers == L][0] if len(Qs[layers == L]) > 0 else float("nan")
            best_q = Qs.max()
            gap = best_q - final_q
            marker = " <<< DEGRADATION" if best_layer < L and gap > 0.01 else ""
            print(f"    step {step:>6d}: best=L{best_layer:>2d}/{L} (Q={best_q:.4f}), "
                  f"final_Q={final_q:.4f}, gap={gap:.4f}{marker}")

    # ========================
    # SAVE RESULTS
    # ========================
    out = {
        "experiment": "CTI 1.4B Holdout Prediction Test",
        "train_models": sorted(TRAIN_MODELS),
        "holdout_model": HOLDOUT_MODEL,
        "n_train": len(train_obs),
        "n_test": len(test_obs),
        "fit_params": {
            "alpha": float(alpha), "beta": float(beta),
            "mu_0": float(mu_0), "mu_1": float(mu_1),
            "b_d": {ds: float(result.x[4+i]) for i, ds in enumerate(datasets)},
        },
        "train_fit": {"mae": float(mae_train), "r2": float(r2_train)},
        "holdout_fit": {"mae": float(mae_test), "r2": float(r2_test)},
        "per_dataset": ds_results,
        "falsification": {
            "mae_threshold": MAE_THRESHOLD,
            "r2_threshold": R2_THRESHOLD,
            "mae_pass": bool(mae_pass),
            "r2_pass": bool(r2_pass),
            "overall": overall,
        },
        "shape_universality": {
            "mean_rho": float(rhos.mean()) if shape_rhos else None,
            "median_rho": float(np.median(rhos)) if shape_rhos else None,
            "frac_above_07": float(np.mean(rhos > 0.7)) if shape_rhos else None,
            "n": len(shape_rhos),
        },
    }

    out_path = RESULTS_DIR / "cti_holdout_prediction.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
