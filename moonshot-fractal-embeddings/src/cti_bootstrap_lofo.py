#!/usr/bin/env python
"""CTI Bootstrap CIs for Leave-One-Family-Out predictions.

Fast approach: bootstrap by resampling (model, step) blocks within each
LOFO fold. For each bootstrap, refit on resampled training blocks and
predict on resampled test blocks. Report 95% CIs on MAE and R2.

Uses 5 restarts (not 20) for speed since each bootstrap just needs a
reasonable fit, not the global optimum.
"""

from __future__ import annotations

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}
N_BOOT = 500
SEED = 42


def load_all_observations():
    """Load observations from all sources."""
    obs = []

    # Pythia
    with open(RESULTS_DIR / "cti_checkpoint_sweep_all.json") as f:
        data = json.load(f)
    for r in data["results"]:
        if "error" in r or r["step"] == 0:
            continue
        for ds_name, ds_data in r["datasets"].items():
            n_classes = DS_CLASSES.get(ds_name, 100)
            Q_chance = 1.0 / n_classes
            for li_str, ld in ds_data["layers"].items():
                li = int(li_str)
                L = int(r["num_layers"])
                N = float(r["N_params"])
                C = float(r["C_flops"])
                Q_norm = (ld["knn_l1"] - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)
                obs.append({
                    "x": li / L, "Q": Q_norm, "dataset": ds_name,
                    "model": r["model"], "family": "pythia",
                    "step": r["step"], "layer": li, "L": L,
                    "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                })

    # OLMo-2
    with open(RESULTS_DIR / "cti_olmo2_sweep.json") as f:
        data = json.load(f)
    for r in data["results"]:
        if "error" in r:
            continue
        for ds_name, ds_data in r["datasets"].items():
            n_classes = DS_CLASSES.get(ds_name, 100)
            Q_chance = 1.0 / n_classes
            for li_str, ld in ds_data["layers"].items():
                li = int(li_str)
                L = int(r["num_layers"])
                N = float(r["N_params"])
                C = float(r["C_flops"])
                Q_norm = (ld["knn_l1"] - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)
                obs.append({
                    "x": li / L, "Q": Q_norm, "dataset": ds_name,
                    "model": r["model"], "family": "olmo2",
                    "step": r["step"], "layer": li, "L": L,
                    "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                })

    # Multi-family
    with open(RESULTS_DIR / "cti_multi_family.json") as f:
        data = json.load(f)
    for r in data["results"]:
        if "error" in r:
            continue
        for ds_name, ds_data in r["datasets"].items():
            n_classes = DS_CLASSES.get(ds_name, 100)
            Q_chance = 1.0 / n_classes
            for li_str, ld in ds_data["layers"].items():
                li = int(li_str)
                L = int(r["num_layers"])
                N = float(r["N_params"])
                C = float(r["C_flops"])
                Q_norm = (ld["knn_l1"] - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)
                obs.append({
                    "x": li / L, "Q": Q_norm, "dataset": ds_name,
                    "model": r["model"], "family": r["family"],
                    "step": -1, "layer": li, "L": L,
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


def fit_gaussian(train_obs, datasets, n_restarts=5):
    Q_tr = np.array([o["Q"] for o in train_obs])
    n_ds = len(datasets)
    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds

    def loss(params):
        return np.mean((Q_tr - predict_ds(params, train_obs, datasets)) ** 2)

    best, best_loss = None, float("inf")
    for trial in range(n_restarts):
        rng = np.random.RandomState(trial * 100 + 7)
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-10})
            if res.fun < best_loss:
                best_loss = res.fun
                best = res
        except Exception:
            continue
    return best


def eval_fit(Q_obs, Q_pred):
    residuals = Q_obs - Q_pred
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return mae, r2


def block_bootstrap(obs_list, rng):
    """Resample by (model, step) blocks."""
    block_map = {}
    for o in obs_list:
        key = (o["model"], o["step"])
        if key not in block_map:
            block_map[key] = []
        block_map[key].append(o)

    blocks = list(block_map.keys())
    boot_blocks = [blocks[i] for i in rng.randint(0, len(blocks), len(blocks))]

    boot_obs = []
    for block in boot_blocks:
        boot_obs.extend(block_map[block])
    return boot_obs


def main():
    obs = load_all_observations()
    families = sorted(set(o["family"] for o in obs))
    datasets = sorted(set(o["dataset"] for o in obs))

    print("=" * 70)
    print("  CTI BOOTSTRAP CIs: LEAVE-ONE-FAMILY-OUT")
    print("=" * 70)
    print(f"Total obs: {len(obs)}, Families: {families}")
    print(f"N_bootstrap: {N_BOOT}")

    results = {}

    for holdout_fam in families:
        print(f"\n--- Holdout: {holdout_fam} ---")
        rng = np.random.RandomState(SEED)

        train_all = [o for o in obs if o["family"] != holdout_fam]
        test_all = [o for o in obs if o["family"] == holdout_fam]

        boot_maes = []
        boot_r2s = []

        for b in range(N_BOOT):
            # Bootstrap train and test separately
            boot_train = block_bootstrap(train_all, rng)
            boot_test = block_bootstrap(test_all, rng)

            if not boot_train or not boot_test:
                continue

            fit = fit_gaussian(boot_train, datasets, n_restarts=3)
            if fit is None:
                continue

            Q_test = np.array([o["Q"] for o in boot_test])
            Q_pred = predict_ds(fit.x, boot_test, datasets)
            mae, r2 = eval_fit(Q_test, Q_pred)
            boot_maes.append(mae)
            boot_r2s.append(r2)

            if (b + 1) % 50 == 0:
                print(f"  Bootstrap {b+1}/{N_BOOT}...")

        boot_maes = np.array(boot_maes)
        boot_r2s = np.array(boot_r2s)

        mae_ci = (float(np.percentile(boot_maes, 2.5)), float(np.percentile(boot_maes, 97.5)))
        r2_ci = (float(np.percentile(boot_r2s, 2.5)), float(np.percentile(boot_r2s, 97.5)))

        results[holdout_fam] = {
            "mae_mean": float(boot_maes.mean()),
            "mae_95ci": mae_ci,
            "r2_mean": float(boot_r2s.mean()),
            "r2_95ci": r2_ci,
            "n_successful": len(boot_maes),
        }

        print(f"  MAE: {boot_maes.mean():.4f} [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]")
        print(f"  R2:  {boot_r2s.mean():.4f} [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_r2_means = [v["r2_mean"] for v in results.values()]
    all_mae_means = [v["mae_mean"] for v in results.values()]
    all_r2_lower = [v["r2_95ci"][0] for v in results.values()]

    print(f"  Mean MAE: {np.mean(all_mae_means):.4f}")
    print(f"  Mean R2:  {np.mean(all_r2_means):.4f}")
    print(f"  All R2 CIs exclude zero: {all(lb > 0 for lb in all_r2_lower)}")
    print(f"  Tightest R2 lower bound: {min(all_r2_lower):.4f}")

    # Save
    output = {
        "experiment": "CTI Bootstrap CIs (LOFO)",
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "per_family": results,
        "summary": {
            "mean_mae": float(np.mean(all_mae_means)),
            "mean_r2": float(np.mean(all_r2_means)),
            "all_r2_above_zero": all(lb > 0 for lb in all_r2_lower),
            "tightest_r2_lower": float(min(all_r2_lower)),
        },
    }

    out_path = RESULTS_DIR / "cti_bootstrap_lofo.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
