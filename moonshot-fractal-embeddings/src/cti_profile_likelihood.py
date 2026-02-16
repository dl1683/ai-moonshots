#!/usr/bin/env python
"""CTI Profile Likelihood Confidence Intervals.

For each physics parameter (alpha, beta, mu_0, mu_1), compute profile
likelihood CIs by:
1. Fix the parameter at a grid of values
2. Re-optimize all other parameters
3. Compute deviance = -2*(loglik_profile - loglik_best)
4. CI boundary where deviance = chi2(1, 0.95) = 3.84

This is much more stable than bootstrap because each fit is a full
optimization from the MLE, not from a random starting point.
"""

from __future__ import annotations

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import chi2
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


def neg_loglik(params, obs_list, ds_list):
    """Negative Gaussian log-likelihood (MSE up to constant)."""
    Q_obs = np.array([o["Q"] for o in obs_list])
    Q_pred = predict_ds(params, obs_list, ds_list)
    return 0.5 * np.sum((Q_obs - Q_pred) ** 2)


def fit_full(obs, datasets, n_restarts=20):
    """Fit all parameters."""
    n_ds = len(datasets)
    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds

    best, best_loss = None, float("inf")
    for trial in range(n_restarts):
        rng = np.random.RandomState(trial)
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(neg_loglik, x0, args=(obs, datasets),
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_loss:
                best_loss = res.fun
                best = res
        except Exception:
            continue
    return best


def profile_ci(param_idx, param_name, best_params, best_nll, obs, datasets,
               n_grid=40, n_restarts=5):
    """Compute profile likelihood CI for one parameter."""
    n_ds = len(datasets)
    all_bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    threshold = chi2.ppf(0.95, df=1)  # 3.84

    best_val = best_params[param_idx]
    param_bounds = all_bounds[param_idx]

    # Search range: expand from best value
    lo, hi = param_bounds
    grid = np.linspace(max(lo, best_val - 3 * abs(best_val) - 0.1),
                       min(hi, best_val + 3 * abs(best_val) + 0.1),
                       n_grid)

    profile_nll = []

    for val in grid:
        # Fix param_idx at val, optimize rest
        other_bounds = list(all_bounds)
        other_bounds[param_idx] = (val, val)  # fix this param

        best_fit = None
        best_loss = float("inf")
        for trial in range(n_restarts):
            x0 = list(best_params)
            rng = np.random.RandomState(trial * 100 + 13)
            # Perturb other params slightly
            for i in range(len(x0)):
                if i != param_idx:
                    x0[i] += rng.normal(0, 0.01 * max(abs(x0[i]), 0.01))
                    x0[i] = np.clip(x0[i], all_bounds[i][0], all_bounds[i][1])
            x0[param_idx] = val

            try:
                res = minimize(neg_loglik, x0, args=(obs, datasets),
                              method="L-BFGS-B", bounds=other_bounds,
                              options={"maxiter": 2000, "ftol": 1e-12})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_fit = res
            except Exception:
                continue

        deviance = 2 * (best_loss - best_nll) if best_fit else float("inf")
        profile_nll.append((val, deviance))

    # Find CI boundaries (where deviance crosses threshold)
    vals = np.array([p[0] for p in profile_nll])
    devs = np.array([p[1] for p in profile_nll])

    # Lower bound: leftmost point where deviance < threshold
    in_ci = devs < threshold
    if any(in_ci):
        ci_lo = vals[in_ci].min()
        ci_hi = vals[in_ci].max()
    else:
        ci_lo = best_val
        ci_hi = best_val

    return ci_lo, ci_hi, list(zip(vals.tolist(), devs.tolist()))


def main():
    obs = load_all_observations()
    datasets = sorted(set(o["dataset"] for o in obs))

    print("=" * 70)
    print("  CTI PROFILE LIKELIHOOD CONFIDENCE INTERVALS")
    print("=" * 70)
    print(f"Obs: {len(obs)}, Datasets: {datasets}")

    # Fit on all data
    print("\nFitting full model...")
    fit = fit_full(obs, datasets, n_restarts=20)
    best_nll = fit.fun
    best_params = fit.x.tolist()

    print(f"Best NLL: {best_nll:.2f}")
    param_names = ["alpha", "beta", "mu_0", "mu_1"] + [f"b_{ds}" for ds in datasets]
    for i, name in enumerate(param_names):
        print(f"  {name}: {best_params[i]:.4f}")

    # Profile CIs for physics parameters
    results = {}
    for idx, name in enumerate(["alpha", "beta", "mu_0", "mu_1"]):
        print(f"\nProfiling {name}...")
        ci_lo, ci_hi, profile = profile_ci(idx, name, best_params, best_nll,
                                            obs, datasets)
        results[name] = {
            "mle": best_params[idx],
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "ci_width": ci_hi - ci_lo,
        }
        print(f"  MLE = {best_params[idx]:.4f}")
        print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Width = {ci_hi - ci_lo:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: 95% Profile Likelihood CIs")
    print(f"{'='*70}")
    for name, r in results.items():
        print(f"  {name:6s}: {r['mle']:+.4f}  [{r['ci_95_lo']:+.4f}, {r['ci_95_hi']:+.4f}]")

    # Check: do CIs for alpha exclude 0?
    alpha_excludes_zero = results["alpha"]["ci_95_lo"] > 0 or results["alpha"]["ci_95_hi"] < 0
    print(f"\n  alpha CI excludes zero: {alpha_excludes_zero}")
    # Check: does beta CI exclude 0?
    beta_above_zero = results["beta"]["ci_95_lo"] > 0
    print(f"  beta CI above zero: {beta_above_zero}")

    # Save
    output = {
        "experiment": "CTI Profile Likelihood CIs",
        "n_observations": len(obs),
        "best_nll": best_nll,
        "best_params": dict(zip(param_names, best_params)),
        "profile_cis": results,
    }

    out_path = RESULTS_DIR / "cti_profile_likelihood.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
