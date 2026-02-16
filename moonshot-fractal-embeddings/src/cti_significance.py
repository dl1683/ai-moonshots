#!/usr/bin/env python
"""CTI Significance Tests: F-test + comparison to null models.

For each LOFO fold, compute:
1. F-statistic and p-value (is R2 significantly > 0?)
2. Comparison to intercept-only model (nested F-test)
3. Leave-one-family-out variance as uncertainty measure

This replaces the failed bootstrap approach with standard,
well-understood statistical tests.
"""

from __future__ import annotations

import json
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_all_observations():
    """Load all CTI observations."""
    obs = []

    for source, path, family_key in [
        ("pythia", "cti_checkpoint_sweep_all.json", lambda r: "pythia"),
        ("olmo2", "cti_olmo2_sweep.json", lambda r: "olmo2"),
        ("multi", "cti_multi_family.json", lambda r: r["family"]),
    ]:
        with open(RESULTS_DIR / path) as f:
            data = json.load(f)
        for r in data["results"]:
            if "error" in r:
                continue
            if source == "pythia" and r.get("step", 0) == 0:
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
                        "model": r["model"], "family": family_key(r),
                        "step": r.get("step", -1), "layer": li, "L": L,
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


def main():
    obs = load_all_observations()
    datasets = sorted(set(o["dataset"] for o in obs))
    families = sorted(set(o["family"] for o in obs))
    p_model = 4 + len(datasets)  # alpha, beta, mu_0, mu_1 + b_d per dataset

    print("=" * 70)
    print("  CTI SIGNIFICANCE TESTS")
    print("=" * 70)
    print(f"Obs: {len(obs)}, Families: {families}, Model params: {p_model}")

    results = {}

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]
        n_test = len(test)

        fit = fit_gaussian(train, datasets, n_restarts=20)
        if fit is None:
            print(f"  {holdout_fam}: FIT FAILED")
            continue

        Q_test = np.array([o["Q"] for o in test])
        Q_pred = predict_ds(fit.x, test, datasets)
        residuals = Q_test - Q_pred

        # Metrics
        mae = float(np.mean(np.abs(residuals)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((Q_test - Q_test.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # F-statistic: F = (R2/p) / ((1-R2)/(n-p-1))
        if r2 > 0 and n_test > p_model + 1:
            f_stat = (r2 / p_model) / ((1 - r2) / (n_test - p_model - 1))
            f_pvalue = 1 - stats.f.cdf(f_stat, p_model, n_test - p_model - 1)
        else:
            f_stat = 0
            f_pvalue = 1.0

        # Intercept-only R2 (baseline: predict mean of training data for each dataset)
        Q_train = np.array([o["Q"] for o in train])
        ds_means = {}
        for o in train:
            ds = o["dataset"]
            if ds not in ds_means:
                ds_means[ds] = []
            ds_means[ds].append(o["Q"])
        ds_means = {ds: np.mean(vals) for ds, vals in ds_means.items()}

        Q_null = np.array([ds_means.get(o["dataset"], Q_train.mean()) for o in test])
        ss_null = float(np.sum((Q_test - Q_null) ** 2))
        r2_null = 1 - ss_null / ss_tot if ss_tot > 0 else 0

        # Nested F-test: our model vs intercept-only
        p_null = len(datasets)  # just b_d terms
        p_extra = p_model - p_null  # alpha, beta, mu_0, mu_1
        if ss_res < ss_null and p_extra > 0 and n_test > p_model + 1:
            f_nested = ((ss_null - ss_res) / p_extra) / (ss_res / (n_test - p_model - 1))
            f_nested_p = 1 - stats.f.cdf(f_nested, p_extra, n_test - p_model - 1)
        else:
            f_nested = 0
            f_nested_p = 1.0

        results[holdout_fam] = {
            "n_test": n_test,
            "mae": mae,
            "r2": r2,
            "f_stat": float(f_stat),
            "f_pvalue": float(f_pvalue),
            "r2_null": float(r2_null),
            "f_nested": float(f_nested),
            "f_nested_pvalue": float(f_nested_p),
        }

        print(f"\n  Holdout: {holdout_fam} (N={n_test})")
        print(f"    R2={r2:.4f}, MAE={mae:.4f}")
        print(f"    F-stat={f_stat:.1f}, p={f_pvalue:.2e} (vs R2=0)")
        print(f"    R2_null (intercept)={r2_null:.4f}")
        print(f"    Nested F={f_nested:.1f}, p={f_nested_p:.2e} (vs intercept)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    r2_vals = [v["r2"] for v in results.values()]
    mae_vals = [v["mae"] for v in results.values()]
    f_pvals = [v["f_pvalue"] for v in results.values()]
    nested_pvals = [v["f_nested_pvalue"] for v in results.values()]

    print(f"  LOFO R2: mean={np.mean(r2_vals):.4f}, range=[{min(r2_vals):.4f}, {max(r2_vals):.4f}]")
    print(f"  LOFO MAE: mean={np.mean(mae_vals):.4f}")
    print(f"  All F-test p < 0.01: {all(p < 0.01 for p in f_pvals)}")
    print(f"  All nested F p < 0.01: {all(p < 0.01 for p in nested_pvals)}")
    print(f"  Max F-test p: {max(f_pvals):.2e}")
    print(f"  Max nested F p: {max(nested_pvals):.2e}")

    # Combined p-value (Fisher's method)
    chi2_stat = -2 * sum(np.log(max(p, 1e-300)) for p in f_pvals)
    combined_p = 1 - stats.chi2.cdf(chi2_stat, 2 * len(f_pvals))
    print(f"  Fisher's combined p-value: {combined_p:.2e}")

    # Save
    output = {
        "experiment": "CTI Significance Tests",
        "model_params": p_model,
        "per_family": results,
        "summary": {
            "mean_r2": float(np.mean(r2_vals)),
            "r2_range": [float(min(r2_vals)), float(max(r2_vals))],
            "mean_mae": float(np.mean(mae_vals)),
            "all_f_significant": all(p < 0.01 for p in f_pvals),
            "all_nested_significant": all(p < 0.01 for p in nested_pvals),
            "fishers_combined_p": float(combined_p),
        },
    }

    out_path = RESULTS_DIR / "cti_significance.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
