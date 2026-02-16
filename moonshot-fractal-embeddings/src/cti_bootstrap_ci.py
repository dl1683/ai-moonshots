#!/usr/bin/env python
"""CTI Bootstrap Confidence Intervals.

For each leave-one-model-out fold, compute 95% CIs on MAE and R2 via
block bootstrap (resampling by checkpoint) to account for within-checkpoint
correlation.
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}
N_BOOT = 200
SEED = 42


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
                    "x": x, "Q": Q_norm, "dataset": ds_name,
                    "model": model, "step": step, "layer": li, "L": L,
                    "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                })
    return obs


def predict_ds(params, obs_list, ds_list):
    alpha, beta, mu_0, mu_1 = params[:4]
    b_d = {ds: params[4 + i] for i, ds in enumerate(ds_list)}
    preds = []
    for o in obs_list:
        x_star = mu_0 + mu_1 * o["log_r"]
        logit_Q = b_d.get(o["dataset"], 0) + alpha * o["log_r"] - beta * (o["x"] - x_star) ** 2
        preds.append(expit(np.clip(logit_Q, -20, 20)))
    return np.array(preds)


def fit_gaussian(train_obs, datasets, n_restarts=20):
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
    residuals = Q_obs - Q_pred
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Q_obs - Q_obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return mae, r2


def bootstrap_cv(obs, holdout_model, datasets, rng):
    """Bootstrap a single leave-one-model-out fold by resampling checkpoints."""
    # Get unique (model, step) blocks for training data
    train_obs = [o for o in obs if o["model"] != holdout_model]
    test_obs = [o for o in obs if o["model"] == holdout_model]

    # Block bootstrap: resample by (model, step) blocks
    train_blocks = list(set((o["model"], o["step"]) for o in train_obs))
    boot_blocks = [train_blocks[i] for i in rng.randint(0, len(train_blocks), len(train_blocks))]

    # Build bootstrap training set
    block_map = {}
    for o in train_obs:
        key = (o["model"], o["step"])
        if key not in block_map:
            block_map[key] = []
        block_map[key].append(o)

    boot_train = []
    for block in boot_blocks:
        boot_train.extend(block_map[block])

    # Also resample test blocks
    test_blocks = list(set((o["model"], o["step"]) for o in test_obs))
    boot_test_blocks = [test_blocks[i] for i in rng.randint(0, len(test_blocks), len(test_blocks))]

    test_block_map = {}
    for o in test_obs:
        key = (o["model"], o["step"])
        if key not in test_block_map:
            test_block_map[key] = []
        test_block_map[key].append(o)

    boot_test = []
    for block in boot_test_blocks:
        boot_test.extend(test_block_map[block])

    if not boot_train or not boot_test:
        return None, None

    # Fit on bootstrap training
    result = fit_gaussian(boot_train, datasets, n_restarts=5)
    if result is None:
        return None, None

    # Predict on bootstrap test
    Q_test = np.array([o["Q"] for o in boot_test])
    Q_pred = predict_ds(result.x, boot_test, datasets)
    mae, r2 = eval_fit(Q_test, Q_pred)
    return mae, r2


def main():
    obs = load_observations()
    models = sorted(set(o["model"] for o in obs))
    datasets = sorted(set(o["dataset"] for o in obs))

    print("=" * 70)
    print("  CTI BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"N_bootstrap: {N_BOOT}")

    results = {}

    for holdout in models:
        print(f"\n--- Holdout: {holdout} ---")
        rng = np.random.RandomState(SEED)

        boot_maes = []
        boot_r2s = []

        for b in range(N_BOOT):
            mae, r2 = bootstrap_cv(obs, holdout, datasets, rng)
            if mae is not None:
                boot_maes.append(mae)
                boot_r2s.append(r2)

            if (b + 1) % 20 == 0:
                print(f"  Bootstrap {b+1}/{N_BOOT}...")

        boot_maes = np.array(boot_maes)
        boot_r2s = np.array(boot_r2s)

        mae_ci = (float(np.percentile(boot_maes, 2.5)), float(np.percentile(boot_maes, 97.5)))
        r2_ci = (float(np.percentile(boot_r2s, 2.5)), float(np.percentile(boot_r2s, 97.5)))
        mae_mean = float(boot_maes.mean())
        r2_mean = float(boot_r2s.mean())

        results[holdout] = {
            "mae_mean": mae_mean,
            "mae_95ci": mae_ci,
            "r2_mean": r2_mean,
            "r2_95ci": r2_ci,
            "n_successful": len(boot_maes),
        }

        print(f"  MAE: {mae_mean:.4f} [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]")
        print(f"  R2:  {r2_mean:.4f} [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_maes = [v["mae_mean"] for v in results.values()]
    all_r2s = [v["r2_mean"] for v in results.values()]
    print(f"  Mean MAE across folds: {np.mean(all_maes):.4f}")
    print(f"  Mean R2 across folds: {np.mean(all_r2s):.4f}")

    # Check if all R2 CIs exclude 0
    all_r2_above_zero = all(v["r2_95ci"][0] > 0 for v in results.values())
    print(f"  All R2 95% CIs exclude zero: {all_r2_above_zero}")

    # Save
    out = {
        "experiment": "CTI Bootstrap CIs",
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "per_model": results,
    }
    out_path = RESULTS_DIR / "cti_bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
