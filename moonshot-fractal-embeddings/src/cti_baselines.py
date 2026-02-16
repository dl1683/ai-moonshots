#!/usr/bin/env python
"""CTI Baselines: compare our Gaussian law against simpler/nonparametric alternatives.

Baselines:
1. Intercept-only: Q = mean(Q) per dataset (no depth/compute info)
2. Linear-in-depth: logit(Q) = b_d + c*x (no compute, just depth)
3. Monotone spline: natural cubic spline per (model, dataset) — oracle nonparametric
4. kNN regression in (x, log(C/N)) space — nonparametric smooth
5. Our Gaussian + b_d

All evaluated via leave-one-model-out CV (fit on 3 models, predict 4th).
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit as sp_logit
from scipy.interpolate import CubicSpline
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_observations(path=None, exclude_step0=True, metric="knn_l1"):
    if path is None:
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
                Q_raw = layer_data[metric]
                Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)

                obs.append({
                    "x": x, "Q": Q_norm, "Q_raw": Q_raw, "dataset": ds_name,
                    "model": model, "step": step, "layer": li, "L": L,
                    "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                })
    return obs


def eval_fit(Q_obs, Q_pred):
    residuals = Q_obs - Q_pred
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Q_obs - Q_obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return mae, r2


# ==========================================
# BASELINE 1: Dataset mean (intercept only)
# ==========================================
def baseline_intercept(train_obs, test_obs):
    ds_means = {}
    for o in train_obs:
        ds_means.setdefault(o["dataset"], []).append(o["Q"])
    ds_means = {k: np.mean(v) for k, v in ds_means.items()}

    preds = np.array([ds_means.get(o["dataset"], 0.3) for o in test_obs])
    return preds


# ==========================================
# BASELINE 2: Linear in depth + dataset intercept
# ==========================================
def baseline_linear_depth(train_obs, test_obs, datasets):
    Q_tr = np.array([o["Q"] for o in train_obs])
    n_ds = len(datasets)

    def predict(params, obs_list):
        c = params[0]
        b_d = {ds: params[1 + i] for i, ds in enumerate(datasets)}
        preds = []
        for o in obs_list:
            logit_Q = b_d.get(o["dataset"], 0) + c * o["x"]
            preds.append(expit(np.clip(logit_Q, -20, 20)))
        return np.array(preds)

    def loss(params):
        return np.mean((Q_tr - predict(params, train_obs)) ** 2)

    bounds = [(-5, 5)] + [(-10, 10)] * n_ds
    best, best_loss = None, float("inf")
    for trial in range(20):
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

    return predict(best.x, test_obs)


# ==========================================
# BASELINE 3: kNN regression (nonparametric)
# ==========================================
def baseline_knn(train_obs, test_obs, k=10):
    X_tr = np.array([[o["x"], o["log_r"]] for o in train_obs])
    y_tr = np.array([o["Q"] for o in train_obs])
    X_te = np.array([[o["x"], o["log_r"]] for o in test_obs])

    # Per-dataset kNN (more fair since datasets have different scales)
    datasets = sorted(set(o["dataset"] for o in train_obs))
    preds = np.zeros(len(test_obs))

    for ds in datasets:
        tr_mask = np.array([o["dataset"] == ds for o in train_obs])
        te_mask = np.array([o["dataset"] == ds for o in test_obs])

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        knn = KNeighborsRegressor(n_neighbors=min(k, tr_mask.sum()), weights="distance")
        knn.fit(X_tr[tr_mask], y_tr[tr_mask])
        preds[te_mask] = knn.predict(X_te[te_mask])

    return preds


# ==========================================
# OUR MODEL: Gaussian + b_d
# ==========================================
def our_gaussian(train_obs, test_obs, datasets):
    Q_tr = np.array([o["Q"] for o in train_obs])
    n_ds = len(datasets)

    def predict(params, obs_list):
        alpha, beta, mu_0, mu_1 = params[:4]
        b_d = {ds: params[4 + i] for i, ds in enumerate(datasets)}
        preds = []
        for o in obs_list:
            x_star = mu_0 + mu_1 * o["log_r"]
            logit_Q = b_d.get(o["dataset"], 0) + alpha * o["log_r"] - beta * (o["x"] - x_star) ** 2
            preds.append(expit(np.clip(logit_Q, -20, 20)))
        return np.array(preds)

    def loss(params):
        return np.mean((Q_tr - predict(params, train_obs)) ** 2)

    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    best, best_loss = None, float("inf")
    for trial in range(30):
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

    return predict(best.x, test_obs), best.x


def main():
    obs = load_observations()
    models = sorted(set(o["model"] for o in obs))
    datasets = sorted(set(o["dataset"] for o in obs))

    print("=" * 70)
    print("  CTI BASELINE COMPARISON (Leave-One-Model-Out CV)")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Total observations: {len(obs)}")

    results = {m: {} for m in ["intercept", "linear_depth", "knn_k5", "knn_k10", "gaussian_bd"]}

    for holdout in models:
        train = [o for o in obs if o["model"] != holdout]
        test = [o for o in obs if o["model"] == holdout]
        Q_test = np.array([o["Q"] for o in test])

        print(f"\n--- Holdout: {holdout} (n_test={len(test)}) ---")

        # Baseline 1: Intercept
        pred_int = baseline_intercept(train, test)
        mae_int, r2_int = eval_fit(Q_test, pred_int)
        results["intercept"][holdout] = {"mae": float(mae_int), "r2": float(r2_int)}
        print(f"  Intercept only:     MAE={mae_int:.4f}, R2={r2_int:.4f}")

        # Baseline 2: Linear depth
        pred_lin = baseline_linear_depth(train, test, datasets)
        mae_lin, r2_lin = eval_fit(Q_test, pred_lin)
        results["linear_depth"][holdout] = {"mae": float(mae_lin), "r2": float(r2_lin)}
        print(f"  Linear depth+b_d:   MAE={mae_lin:.4f}, R2={r2_lin:.4f}")

        # Baseline 3a: kNN k=5
        pred_knn5 = baseline_knn(train, test, k=5)
        mae_k5, r2_k5 = eval_fit(Q_test, pred_knn5)
        results["knn_k5"][holdout] = {"mae": float(mae_k5), "r2": float(r2_k5)}
        print(f"  kNN (k=5, per-ds):  MAE={mae_k5:.4f}, R2={r2_k5:.4f}")

        # Baseline 3b: kNN k=10
        pred_knn10 = baseline_knn(train, test, k=10)
        mae_k10, r2_k10 = eval_fit(Q_test, pred_knn10)
        results["knn_k10"][holdout] = {"mae": float(mae_k10), "r2": float(r2_k10)}
        print(f"  kNN (k=10, per-ds): MAE={mae_k10:.4f}, R2={r2_k10:.4f}")

        # Our model
        pred_gauss, params = our_gaussian(train, test, datasets)
        mae_g, r2_g = eval_fit(Q_test, pred_gauss)
        results["gaussian_bd"][holdout] = {"mae": float(mae_g), "r2": float(r2_g)}
        print(f"  Gaussian + b_d:     MAE={mae_g:.4f}, R2={r2_g:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Mean across holdout models")
    print("=" * 70)
    summary = {}
    for name in results:
        maes = [v["mae"] for v in results[name].values()]
        r2s = [v["r2"] for v in results[name].values()]
        mean_mae = np.mean(maes)
        mean_r2 = np.mean(r2s)
        summary[name] = {"mean_mae": float(mean_mae), "mean_r2": float(mean_r2)}
        print(f"  {name:>20s}: mean MAE={mean_mae:.4f}, mean R2={mean_r2:.4f}")

    # Delta over best baseline
    best_baseline_r2 = max(summary[k]["mean_r2"] for k in summary if k != "gaussian_bd")
    our_r2 = summary["gaussian_bd"]["mean_r2"]
    print(f"\n  Gaussian R2 advantage over best baseline: {our_r2 - best_baseline_r2:+.4f}")

    # Save
    out = {
        "experiment": "CTI Baseline Comparison",
        "cv_type": "leave-one-model-out",
        "per_model": results,
        "summary": summary,
    }
    out_path = RESULTS_DIR / "cti_baselines.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
