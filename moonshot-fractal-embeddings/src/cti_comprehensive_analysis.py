#!/usr/bin/env python
"""Comprehensive CTI analysis: chance normalization, CV, polynomial form."""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Dataset L1 class counts for chance normalization
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_observations(path=None, exclude_step0=True, metric="knn_l1"):
    if path is None:
        path = RESULTS_DIR / "cti_checkpoint_sweep.json"
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


def predict_universal(params, obs_list):
    alpha, beta, mu_0, mu_1, b = params[:5]
    preds = []
    for o in obs_list:
        log_r = np.log(o["C"]) - np.log(o["N"])
        x_star = mu_0 + mu_1 * log_r
        logit_Q = b + alpha * log_r - beta * (o["x"] - x_star) ** 2
        preds.append(expit(np.clip(logit_Q, -20, 20)))
    return np.array(preds)


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


def predict_poly(params, obs_list, ds_list):
    """Polynomial form: logit(Q) = b_d + (a0 + a1*xi) + (a2 + a3*xi)*x + (a4 + a5*xi)*x^2"""
    a0, a1, a2, a3, a4, a5 = params[:6]
    b_d = {ds: params[6 + i] for i, ds in enumerate(ds_list)}
    preds = []
    for o in obs_list:
        xi = np.log(o["C"]) - np.log(o["N"])
        x = o["x"]
        logit_Q = b_d.get(o["dataset"], 0) + (a0 + a1 * xi) + (a2 + a3 * xi) * x + (a4 + a5 * xi) * x ** 2
        preds.append(expit(np.clip(logit_Q, -20, 20)))
    return np.array(preds)


def fit_model(predict_fn, obs_list, Q_obs, n_params, bounds, n_restarts=30, **kwargs):
    def loss(params):
        return np.mean((Q_obs - predict_fn(params, obs_list, **kwargs)) ** 2)

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
    obs = load_observations()
    datasets = sorted(set(o["dataset"] for o in obs))
    models = sorted(set(o["model"] for o in obs))
    Q_all = np.array([o["Q"] for o in obs])

    print("=" * 70)
    print("  CTI Comprehensive Analysis (4 datasets, 3 models)")
    print("=" * 70)
    print(f"Observations: {len(obs)}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Chance levels: {', '.join(f'{ds}={1/DS_CLASSES[ds]:.4f}' for ds in datasets)}")

    # Q_norm stats
    print("\nQ_norm mean by dataset:")
    for ds in datasets:
        mask = np.array([o["dataset"] == ds for o in obs])
        Qs = Q_all[mask]
        print(f"  {ds}: mean={Qs.mean():.4f}, std={Qs.std():.4f}, range=[{Qs.min():.4f}, {Qs.max():.4f}]")

    # === FIT 1: Universal (no dataset intercept) ===
    print("\n" + "=" * 70)
    print("FIT 1: Universal Gaussian (no dataset intercept)")
    bounds_univ = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5), (-10, 10)]
    result_univ = fit_model(predict_universal, obs, Q_all, 5, bounds_univ)
    Q_pred_univ = predict_universal(result_univ.x, obs)
    mae_univ, r2_univ = eval_fit(Q_all, Q_pred_univ)
    a, b, m0, m1, b0 = result_univ.x
    print(f"alpha={a:.4f}, beta={b:.4f}, mu_0={m0:.4f}, mu_1={m1:.6f}, b={b0:.4f}")
    print(f"MAE = {mae_univ:.4f}, R2 = {r2_univ:.4f}")

    # === FIT 2: With dataset intercept ===
    print("\n" + "=" * 70)
    print("FIT 2: Gaussian with dataset intercept")
    n_ds = len(datasets)
    bounds_ds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    result_ds = fit_model(lambda p, o: predict_ds(p, o, datasets), obs, Q_all,
                          4 + n_ds, bounds_ds)
    Q_pred_ds = predict_ds(result_ds.x, obs, datasets)
    mae_ds, r2_ds = eval_fit(Q_all, Q_pred_ds)
    a2, b2, m02, m12 = result_ds.x[:4]
    print(f"alpha={a2:.4f}, beta={b2:.4f}, mu_0={m02:.4f}, mu_1={m12:.6f}")
    for i, ds in enumerate(datasets):
        print(f"  b_{ds}={result_ds.x[4 + i]:.4f}")
    print(f"MAE = {mae_ds:.4f}, R2 = {r2_ds:.4f}")
    print(f"Delta R2 from dataset intercept: {r2_ds - r2_univ:.4f}")

    # === FIT 3: Polynomial with dataset intercept (Codex suggestion) ===
    print("\n" + "=" * 70)
    print("FIT 3: Polynomial (Codex) with dataset intercept")
    bounds_poly = [(-1, 1)] * 6 + [(-10, 10)] * n_ds
    result_poly = fit_model(lambda p, o: predict_poly(p, o, datasets), obs, Q_all,
                            6 + n_ds, bounds_poly)
    Q_pred_poly = predict_poly(result_poly.x, obs, datasets)
    mae_poly, r2_poly = eval_fit(Q_all, Q_pred_poly)
    a0, a1, a2p, a3, a4, a5 = result_poly.x[:6]
    print(f"a0={a0:.4f}, a1={a1:.6f}, a2={a2p:.4f}, a3={a3:.6f}, a4={a4:.4f}, a5={a5:.6f}")
    for i, ds in enumerate(datasets):
        print(f"  b_{ds}={result_poly.x[6 + i]:.4f}")
    print(f"MAE = {mae_poly:.4f}, R2 = {r2_poly:.4f}")

    # === Leave-One-Dataset-Out CV (universal model) ===
    print("\n" + "=" * 70)
    print("Leave-One-Dataset-Out CV (universal model, no dataset intercept)")
    for holdout_ds in datasets:
        train = [o for o in obs if o["dataset"] != holdout_ds]
        test = [o for o in obs if o["dataset"] == holdout_ds]
        Q_tr = np.array([o["Q"] for o in train])
        Q_te = np.array([o["Q"] for o in test])

        res = fit_model(predict_universal, train, Q_tr, 5, bounds_univ)
        Q_pred = predict_universal(res.x, test)
        mae, r2 = eval_fit(Q_te, Q_pred)
        print(f"  Holdout {holdout_ds:>16s}: MAE={mae:.4f}, R2={r2:.4f}")

    # === Leave-One-Model-Out CV (dataset intercept) ===
    print("\n" + "=" * 70)
    print("Leave-One-Model-Out CV (with dataset intercept)")
    for holdout_model in models:
        train = [o for o in obs if o["model"] != holdout_model]
        test = [o for o in obs if o["model"] == holdout_model]
        Q_tr = np.array([o["Q"] for o in train])
        Q_te = np.array([o["Q"] for o in test])

        res = fit_model(lambda p, o: predict_ds(p, o, datasets), train, Q_tr,
                        4 + n_ds, bounds_ds)
        Q_pred = predict_ds(res.x, test, datasets)
        mae, r2 = eval_fit(Q_te, Q_pred)
        a, b, m0, m1 = res.x[:4]
        print(f"  Holdout {holdout_model:>12s}: MAE={mae:.4f}, R2={r2:.4f} "
              f"(alpha={a:.4f}, beta={b:.4f}, mu_0={m0:.4f}, mu_1={m1:.6f})")

    # === Leave-One-Dataset-Out CV (polynomial, with remaining datasets intercept) ===
    print("\n" + "=" * 70)
    print("Leave-One-Dataset-Out CV (polynomial form, with remaining dataset intercepts)")
    for holdout_ds in datasets:
        train = [o for o in obs if o["dataset"] != holdout_ds]
        test = [o for o in obs if o["dataset"] == holdout_ds]
        Q_tr = np.array([o["Q"] for o in train])
        Q_te = np.array([o["Q"] for o in test])

        train_ds = sorted(set(o["dataset"] for o in train))

        # Fit on training datasets
        bounds_cv = [(-1, 1)] * 6 + [(-10, 10)] * len(train_ds)
        res = fit_model(lambda p, o: predict_poly(p, o, train_ds), train, Q_tr,
                        6 + len(train_ds), bounds_cv)

        # For test: use mean of trained b_d as proxy
        mean_b = np.mean(res.x[6:])
        test_params = list(res.x[:6]) + [mean_b]
        Q_pred = predict_poly(test_params, test, [holdout_ds])
        mae, r2 = eval_fit(Q_te, Q_pred)
        print(f"  Holdout {holdout_ds:>16s}: MAE={mae:.4f}, R2={r2:.4f} (b_proxy={mean_b:.4f})")

    # === Optimal layer analysis ===
    print("\n" + "=" * 70)
    print("Optimal Layer at Final Checkpoint (step 143000)")
    a2, b2, m02, m12 = result_ds.x[:4]
    l_errors = []
    for model in models:
        model_obs = [o for o in obs if o["model"] == model]
        N = model_obs[0]["N"]
        L = model_obs[0]["L"]

        for ds in datasets:
            step_obs = [o for o in model_obs if o["step"] == 143000 and o["dataset"] == ds]
            if not step_obs:
                continue
            Qs = np.array([o["Q"] for o in step_obs])
            layers = np.array([o["layer"] for o in step_obs])
            actual_best = layers[np.argmax(Qs)]

            C = step_obs[0]["C"]
            log_r = np.log(C) - np.log(N)
            x_star = m02 + m12 * log_r
            l_star = np.clip(x_star * L, 0, L)

            error = abs(actual_best - l_star)
            l_errors.append(error)
            print(f"  {model:>12s} {ds:>16s}: actual=L{actual_best:>2d}/{L}, "
                  f"predicted=L{l_star:.1f}, err={error:.1f}")

    print(f"\n  Mean l* error at final step: {np.mean(l_errors):.2f} layers")

    # === Save comprehensive results ===
    out = {
        "experiment": "CTI Comprehensive Analysis",
        "n_observations": len(obs),
        "models": models,
        "datasets": datasets,
        "fit_universal": {
            "params": {"alpha": float(result_univ.x[0]), "beta": float(result_univ.x[1]),
                       "mu_0": float(result_univ.x[2]), "mu_1": float(result_univ.x[3]),
                       "b": float(result_univ.x[4])},
            "mae": float(mae_univ), "r2": float(r2_univ),
        },
        "fit_ds_intercept": {
            "params": {"alpha": float(result_ds.x[0]), "beta": float(result_ds.x[1]),
                       "mu_0": float(result_ds.x[2]), "mu_1": float(result_ds.x[3]),
                       "b_d": {ds: float(result_ds.x[4 + i]) for i, ds in enumerate(datasets)}},
            "mae": float(mae_ds), "r2": float(r2_ds),
        },
        "fit_polynomial": {
            "params": {"a0": float(result_poly.x[0]), "a1": float(result_poly.x[1]),
                       "a2": float(result_poly.x[2]), "a3": float(result_poly.x[3]),
                       "a4": float(result_poly.x[4]), "a5": float(result_poly.x[5]),
                       "b_d": {ds: float(result_poly.x[6 + i]) for i, ds in enumerate(datasets)}},
            "mae": float(mae_poly), "r2": float(r2_poly),
        },
    }

    out_path = RESULTS_DIR / "cti_comprehensive_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
