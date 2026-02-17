#!/usr/bin/env python
"""CTI Ablations + Additional Nonparametric Baselines.

1. Ablations: remove log(C/N), remove mu1, depth-only Gaussian
2. Additional baselines: GP, MLP, Gradient-Boosted Trees
All in LOFO cross-validation.
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_all_observations():
    sources = [
        RESULTS_DIR / "cti_checkpoint_sweep_all.json",
        RESULTS_DIR / "cti_multi_family.json",
        RESULTS_DIR / "cti_olmo2_sweep.json",
    ]
    obs = []
    for src_path in sources:
        with open(src_path) as f:
            data = json.load(f)
        for result in data["results"]:
            if "error" in result:
                continue
            if result.get("step", -1) == 0:
                continue
            model = result["model"]
            N = float(result["N_params"])
            C = float(result["C_flops"])
            L = int(result["num_layers"])
            step = result.get("step", -1)
            family = result.get("family", model.split("-")[0])
            for ds_name, ds_data in result["datasets"].items():
                n_classes = DS_CLASSES.get(ds_name, 100)
                Q_chance = 1.0 / n_classes
                if not isinstance(ds_data, dict) or "layers" not in ds_data:
                    continue
                for li_str, layer_data in ds_data["layers"].items():
                    li = int(li_str)
                    x = li / L
                    Q_raw = layer_data.get("knn_l1")
                    if Q_raw is None:
                        continue
                    Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                    Q_norm = np.clip(Q_norm, 0.001, 0.999)
                    obs.append({
                        "x": x, "Q": Q_norm, "dataset": ds_name,
                        "model": model, "step": step, "layer": li, "L": L,
                        "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                        "family": family,
                    })
    return obs


def eval_fit(Q_obs, Q_pred):
    residuals = Q_obs - Q_pred
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return mae, r2


# ==========================================
# ABLATED GAUSSIAN MODELS
# ==========================================

def fit_gaussian_full(train_obs, test_obs, datasets):
    """Full Gaussian: alpha, beta, mu0, mu1 + b_d."""
    Q_tr = np.array([o["Q"] for o in train_obs])
    n_ds = len(datasets)

    def predict(params, obs_list):
        alpha, beta, mu_0, mu_1 = params[:4]
        b_d = {ds: params[4 + i] for i, ds in enumerate(datasets)}
        return np.array([
            expit(np.clip(b_d.get(o["dataset"], 0) + alpha * o["log_r"]
                          - beta * (o["x"] - mu_0 - mu_1 * o["log_r"]) ** 2, -20, 20))
            for o in obs_list
        ])

    def loss(params):
        return np.mean((Q_tr - predict(params, train_obs)) ** 2)

    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    best, best_loss = None, float("inf")
    for trial in range(20):
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
    return predict(best.x, test_obs) if best else np.full(len(test_obs), 0.5)


def fit_gaussian_no_compute(train_obs, test_obs, datasets):
    """Ablation: remove log(C/N) entirely. beta, mu0 + b_d only."""
    Q_tr = np.array([o["Q"] for o in train_obs])
    n_ds = len(datasets)

    def predict(params, obs_list):
        beta, mu_0 = params[:2]
        b_d = {ds: params[2 + i] for i, ds in enumerate(datasets)}
        return np.array([
            expit(np.clip(b_d.get(o["dataset"], 0)
                          - beta * (o["x"] - mu_0) ** 2, -20, 20))
            for o in obs_list
        ])

    def loss(params):
        return np.mean((Q_tr - predict(params, train_obs)) ** 2)

    bounds = [(0.01, 50), (-2, 2)] + [(-10, 10)] * n_ds
    best, best_loss = None, float("inf")
    for trial in range(20):
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
    return predict(best.x, test_obs) if best else np.full(len(test_obs), 0.5)


def fit_gaussian_no_mu1(train_obs, test_obs, datasets):
    """Ablation: fix mu1=0 (no compute-dependent peak shift)."""
    Q_tr = np.array([o["Q"] for o in train_obs])
    n_ds = len(datasets)

    def predict(params, obs_list):
        alpha, beta, mu_0 = params[:3]
        b_d = {ds: params[3 + i] for i, ds in enumerate(datasets)}
        return np.array([
            expit(np.clip(b_d.get(o["dataset"], 0) + alpha * o["log_r"]
                          - beta * (o["x"] - mu_0) ** 2, -20, 20))
            for o in obs_list
        ])

    def loss(params):
        return np.mean((Q_tr - predict(params, train_obs)) ** 2)

    bounds = [(-1, 1), (0.01, 50), (-2, 2)] + [(-10, 10)] * n_ds
    best, best_loss = None, float("inf")
    for trial in range(20):
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
    return predict(best.x, test_obs) if best else np.full(len(test_obs), 0.5)


def fit_gaussian_no_beta(train_obs, test_obs, datasets):
    """Ablation: remove beta (no depth curvature, just alpha*log_r + b_d)."""
    Q_tr = np.array([o["Q"] for o in train_obs])
    n_ds = len(datasets)

    def predict(params, obs_list):
        alpha = params[0]
        b_d = {ds: params[1 + i] for i, ds in enumerate(datasets)}
        return np.array([
            expit(np.clip(b_d.get(o["dataset"], 0) + alpha * o["log_r"], -20, 20))
            for o in obs_list
        ])

    def loss(params):
        return np.mean((Q_tr - predict(params, train_obs)) ** 2)

    bounds = [(-1, 1)] + [(-10, 10)] * n_ds
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
    return predict(best.x, test_obs) if best else np.full(len(test_obs), 0.5)


# ==========================================
# NONPARAMETRIC BASELINES
# ==========================================

def fit_gp(train_obs, test_obs, datasets):
    """Gaussian Process per dataset."""
    preds = np.zeros(len(test_obs))
    for ds in datasets:
        tr_idx = [i for i, o in enumerate(train_obs) if o["dataset"] == ds]
        te_idx = [i for i, o in enumerate(test_obs) if o["dataset"] == ds]
        if not tr_idx or not te_idx:
            mean_q = np.mean([o["Q"] for o in train_obs]) if train_obs else 0.5
            for j in te_idx:
                preds[j] = mean_q
            continue
        X_tr = np.array([[train_obs[i]["x"], train_obs[i]["log_r"]] for i in tr_idx])
        y_tr = np.array([train_obs[i]["Q"] for i in tr_idx])
        X_te = np.array([[test_obs[j]["x"], test_obs[j]["log_r"]] for j in te_idx])
        try:
            kernel = ConstantKernel(1.0) * RBF(length_scale=[0.2, 2.0]) + WhiteKernel(noise_level=0.01)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                           normalize_y=True, alpha=1e-3)
            gp.fit(X_tr, y_tr)
            pred_vals = gp.predict(X_te)
            pred_vals = np.clip(pred_vals, 0.001, 0.999)
            for k, j in enumerate(te_idx):
                preds[j] = pred_vals[k]
        except Exception:
            mean_q = np.mean(y_tr)
            for j in te_idx:
                preds[j] = mean_q
    return preds


def fit_mlp(train_obs, test_obs, datasets):
    """MLP per dataset."""
    preds = np.zeros(len(test_obs))
    for ds in datasets:
        tr_idx = [i for i, o in enumerate(train_obs) if o["dataset"] == ds]
        te_idx = [i for i, o in enumerate(test_obs) if o["dataset"] == ds]
        if not tr_idx or not te_idx:
            mean_q = np.mean([o["Q"] for o in train_obs]) if train_obs else 0.5
            for j in te_idx:
                preds[j] = mean_q
            continue
        X_tr = np.array([[train_obs[i]["x"], train_obs[i]["log_r"]] for i in tr_idx])
        y_tr = np.array([train_obs[i]["Q"] for i in tr_idx])
        X_te = np.array([[test_obs[j]["x"], test_obs[j]["log_r"]] for j in te_idx])
        try:
            mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                               early_stopping=True, random_state=42,
                               learning_rate_init=0.001)
            mlp.fit(X_tr, y_tr)
            pred_vals = mlp.predict(X_te)
            pred_vals = np.clip(pred_vals, 0.001, 0.999)
            for k, j in enumerate(te_idx):
                preds[j] = pred_vals[k]
        except Exception:
            mean_q = np.mean(y_tr)
            for j in te_idx:
                preds[j] = mean_q
    return preds


def fit_gbrt(train_obs, test_obs, datasets):
    """Gradient-Boosted Regression Trees per dataset."""
    preds = np.zeros(len(test_obs))
    for ds in datasets:
        tr_idx = [i for i, o in enumerate(train_obs) if o["dataset"] == ds]
        te_idx = [i for i, o in enumerate(test_obs) if o["dataset"] == ds]
        if not tr_idx or not te_idx:
            mean_q = np.mean([o["Q"] for o in train_obs]) if train_obs else 0.5
            for j in te_idx:
                preds[j] = mean_q
            continue
        X_tr = np.array([[train_obs[i]["x"], train_obs[i]["log_r"]] for i in tr_idx])
        y_tr = np.array([train_obs[i]["Q"] for i in tr_idx])
        X_te = np.array([[test_obs[j]["x"], test_obs[j]["log_r"]] for j in te_idx])
        try:
            gbrt = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                              learning_rate=0.1, random_state=42)
            gbrt.fit(X_tr, y_tr)
            pred_vals = gbrt.predict(X_te)
            pred_vals = np.clip(pred_vals, 0.001, 0.999)
            for k, j in enumerate(te_idx):
                preds[j] = pred_vals[k]
        except Exception:
            mean_q = np.mean(y_tr)
            for j in te_idx:
                preds[j] = mean_q
    return preds


def fit_gam(train_obs, test_obs, datasets):
    """GAM (thin-plate RBF) per dataset."""
    preds = np.zeros(len(test_obs))
    for ds in datasets:
        tr_idx = [i for i, o in enumerate(train_obs) if o["dataset"] == ds]
        te_idx = [i for i, o in enumerate(test_obs) if o["dataset"] == ds]
        if not tr_idx or not te_idx:
            mean_q = np.mean([o["Q"] for o in train_obs]) if train_obs else 0.5
            for j in te_idx:
                preds[j] = mean_q
            continue
        X_tr = np.array([[train_obs[i]["x"], train_obs[i]["log_r"]] for i in tr_idx])
        y_tr = np.array([train_obs[i]["Q"] for i in tr_idx])
        X_te = np.array([[test_obs[j]["x"], test_obs[j]["log_r"]] for j in te_idx])
        try:
            rbf = RBFInterpolator(X_tr, y_tr, kernel="thin_plate_spline",
                                  smoothing=len(X_tr) * 0.001)
            pred_vals = rbf(X_te)
            pred_vals = np.clip(pred_vals, 0.001, 0.999)
            for k, j in enumerate(te_idx):
                preds[j] = pred_vals[k]
        except Exception:
            mean_q = np.mean(y_tr)
            for j in te_idx:
                preds[j] = mean_q
    return preds


def run_lofo(obs, model_name, fitter):
    """Run LOFO and return per-family R2 and mean."""
    families = sorted(set(o["family"] for o in obs))
    datasets = sorted(set(o["dataset"] for o in obs))
    results = {}

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]
        Q_test = np.array([o["Q"] for o in test])

        try:
            pred = fitter(train, test, datasets)
            mae, r2 = eval_fit(Q_test, pred)
            results[holdout_fam] = {"r2": r2, "mae": mae}
        except Exception as e:
            results[holdout_fam] = {"r2": float("nan"), "mae": float("nan")}

    r2s = [v["r2"] for v in results.values() if not np.isnan(v["r2"])]
    maes = [v["mae"] for v in results.values() if not np.isnan(v["mae"])]
    mean_r2 = float(np.mean(r2s)) if r2s else float("nan")
    mean_mae = float(np.mean(maes)) if maes else float("nan")

    return {
        "mean_r2": mean_r2,
        "mean_mae": mean_mae,
        "per_family": results,
    }


def main():
    print("Loading observations...")
    obs = load_all_observations()
    print(f"  Total: {len(obs)} obs")

    # ==========================================
    # ABLATIONS
    # ==========================================
    print("\n" + "=" * 70)
    print("  ABLATION STUDY")
    print("=" * 70)

    ablations = {
        "Full Gaussian (8p)": fit_gaussian_full,
        "No compute (6p)": fit_gaussian_no_compute,
        "No mu1 (7p)": fit_gaussian_no_mu1,
        "No beta (5p)": fit_gaussian_no_beta,
    }

    ablation_results = {}
    for name, fitter in ablations.items():
        print(f"\n  {name}:")
        result = run_lofo(obs, name, fitter)
        ablation_results[name] = result
        print(f"    Mean R2={result['mean_r2']:.4f}, MAE={result['mean_mae']:.4f}")
        for fam, fdata in result["per_family"].items():
            print(f"      {fam:>15s}: R2={fdata['r2']:.4f}")

    # ==========================================
    # NONPARAMETRIC BASELINES
    # ==========================================
    print("\n" + "=" * 70)
    print("  NONPARAMETRIC BASELINES (LOFO)")
    print("=" * 70)

    nonparam = {
        "GAM (RBF)": fit_gam,
        "Gaussian Process": fit_gp,
        "MLP (32,16)": fit_mlp,
        "GBRT (100 trees)": fit_gbrt,
    }

    nonparam_results = {}
    for name, fitter in nonparam.items():
        print(f"\n  {name}:")
        result = run_lofo(obs, name, fitter)
        nonparam_results[name] = result
        print(f"    Mean R2={result['mean_r2']:.4f}, MAE={result['mean_mae']:.4f}")
        for fam, fdata in result["per_family"].items():
            print(f"      {fam:>15s}: R2={fdata['r2']:.4f}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("  FULL COMPARISON SUMMARY")
    print("=" * 70)

    all_models = {}
    all_models.update(ablation_results)
    all_models.update(nonparam_results)

    for name in sorted(all_models.keys(), key=lambda k: all_models[k]["mean_r2"], reverse=True):
        r2 = all_models[name]["mean_r2"]
        mae = all_models[name]["mean_mae"]
        print(f"  {name:>25s}: R2={r2:.4f}, MAE={mae:.4f}")

    # Save
    output = {
        "ablations": ablation_results,
        "nonparametric_baselines": nonparam_results,
    }

    out_path = RESULTS_DIR / "cti_ablations_baselines.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
