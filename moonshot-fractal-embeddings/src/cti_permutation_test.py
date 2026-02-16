#!/usr/bin/env python
"""CTI Permutation Test: is cross-family transfer significant?

Instead of bootstrap CIs (which fail due to optimization sensitivity),
we use a permutation test: shuffle family labels and re-run LOFO.
This creates a null distribution of "R2 from random family assignment"
and tests whether our real LOFO R2 is significantly above chance.

This is the RIGHT statistical test: it directly answers "does the law
generalize cross-family better than chance?"
"""

from __future__ import annotations

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}
N_PERM = 200
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


def lofo_r2(obs, datasets):
    """Compute mean LOFO R2 across all families."""
    families = sorted(set(o["family"] for o in obs))
    r2s = []

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]

        if not train or not test:
            continue

        fit = fit_gaussian(train, datasets, n_restarts=20)
        if fit is None:
            r2s.append(0.0)
            continue

        Q_test = np.array([o["Q"] for o in test])
        Q_pred = predict_ds(fit.x, test, datasets)
        residuals = Q_test - Q_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Q_test - Q_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2s.append(r2)

    return float(np.mean(r2s)), r2s


def main():
    obs = load_all_observations()
    datasets = sorted(set(o["dataset"] for o in obs))
    families = sorted(set(o["family"] for o in obs))

    print("=" * 70)
    print("  CTI PERMUTATION TEST: IS CROSS-FAMILY TRANSFER SIGNIFICANT?")
    print("=" * 70)
    print(f"Obs: {len(obs)}, Families: {families}, N_perm: {N_PERM}")

    # 1. Real LOFO R2
    print("\nComputing real LOFO R2...")
    real_mean_r2, real_r2s = lofo_r2(obs, datasets)
    print(f"  Real LOFO R2: {real_mean_r2:.4f}")
    for fam, r2 in zip(families, real_r2s):
        print(f"    {fam}: R2={r2:.4f}")

    # 2. Permutation null distribution
    print(f"\nRunning {N_PERM} permutations...")
    rng = np.random.RandomState(SEED)
    null_r2s = []

    # Get unique models and their family assignments
    model_families = {}
    for o in obs:
        model_families[o["model"]] = o["family"]
    models = sorted(model_families.keys())
    original_families = [model_families[m] for m in models]

    for p in range(N_PERM):
        # Shuffle family labels across models (keeping model->obs mapping intact)
        shuffled_families = list(original_families)
        rng.shuffle(shuffled_families)

        # Create new obs with shuffled family assignments
        model_to_new_fam = dict(zip(models, shuffled_families))
        perm_obs = []
        for o in obs:
            o_new = dict(o)
            o_new["family"] = model_to_new_fam[o["model"]]
            perm_obs.append(o_new)

        perm_r2, _ = lofo_r2(perm_obs, datasets)
        null_r2s.append(perm_r2)

        if (p + 1) % 10 == 0:
            print(f"  Perm {p+1}/{N_PERM}: R2={perm_r2:.4f}")

    null_r2s = np.array(null_r2s)

    # 3. Compute p-value
    p_value = float(np.mean(null_r2s >= real_mean_r2))

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Real LOFO R2:  {real_mean_r2:.4f}")
    print(f"  Null mean R2:  {null_r2s.mean():.4f} +/- {null_r2s.std():.4f}")
    print(f"  Null max R2:   {null_r2s.max():.4f}")
    print(f"  p-value:       {p_value:.4f}")
    print(f"  Significant at 0.05: {p_value < 0.05}")
    print(f"  Significant at 0.01: {p_value < 0.01}")

    # Save
    output = {
        "experiment": "CTI Permutation Test",
        "description": "Shuffle family labels, re-run LOFO, test if real R2 exceeds null",
        "n_permutations": N_PERM,
        "seed": SEED,
        "real_lofo_r2": real_mean_r2,
        "real_per_family": dict(zip(families, [float(r) for r in real_r2s])),
        "null_distribution": {
            "mean": float(null_r2s.mean()),
            "std": float(null_r2s.std()),
            "min": float(null_r2s.min()),
            "max": float(null_r2s.max()),
            "percentile_95": float(np.percentile(null_r2s, 95)),
        },
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }

    out_path = RESULTS_DIR / "cti_permutation_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
