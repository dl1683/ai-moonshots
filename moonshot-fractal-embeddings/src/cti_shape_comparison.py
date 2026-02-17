#!/usr/bin/env python
"""CTI Alternative Interpretable Shapes (vectorized).

Tests whether Gaussian depth profile is uniquely good or if other
interpretable parametric forms fit equally well in LOFO cross-validation.

Shapes tested:
  1. Symmetric Gaussian (baseline): -beta*(x - mu)^2
  2. Skewed Gaussian: -beta*(x - mu)^2 + gamma*(x - mu)^3
  3. Asymmetric Gaussian: -beta_L*(x-mu)^2 / -beta_R*(x-mu)^2
  4. Beta-shape: a*log(x) + b*log(1-x)
  5. Free cubic: c1*x + c2*x^2 + c3*x^3
  6. Quartic polynomial: c1*x + c2*x^2 + c3*x^3 + c4*x^4
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
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
            N = float(result["N_params"])
            C = float(result["C_flops"])
            L = int(result["num_layers"])
            family = result.get("family", result["model"].split("-")[0])
            for ds_name, ds_data in result["datasets"].items():
                n_classes = DS_CLASSES.get(ds_name, 100)
                Q_chance = 1.0 / n_classes
                if not isinstance(ds_data, dict) or "layers" not in ds_data:
                    continue
                for li_str, layer_data in ds_data["layers"].items():
                    li = int(li_str)
                    Q_raw = layer_data.get("knn_l1")
                    if Q_raw is None:
                        continue
                    Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                    Q_norm = np.clip(Q_norm, 0.001, 0.999)
                    obs.append({
                        "x": li / L, "Q": Q_norm, "dataset": ds_name,
                        "log_r": np.log(C) - np.log(N), "family": family,
                    })
    return obs


def vectorize_obs(obs_list, datasets):
    """Pre-extract arrays for vectorized operations."""
    x = np.array([o["x"] for o in obs_list])
    Q = np.array([o["Q"] for o in obs_list])
    log_r = np.array([o["log_r"] for o in obs_list])
    # Dataset indicator: ds_idx[i] = index of obs i's dataset in datasets list
    ds_map = {ds: i for i, ds in enumerate(datasets)}
    ds_idx = np.array([ds_map[o["dataset"]] for o in obs_list])
    return x, Q, log_r, ds_idx


def eval_fit(Q_obs, Q_pred):
    residuals = Q_obs - Q_pred
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return mae, r2


def _optimize(loss_fn, bounds, n_restarts=15, maxiter=3000):
    best, best_loss = None, float("inf")
    for trial in range(n_restarts):
        rng = np.random.RandomState(trial)
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(loss_fn, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": maxiter, "ftol": 1e-11})
            if res.fun < best_loss:
                best_loss = res.fun
                best = res
        except Exception:
            continue
    return best


# All predict functions are VECTORIZED: operate on pre-extracted arrays

def fit_symmetric_gaussian(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                           x_te, log_r_te, ds_idx_te, n_ds):
    """logit(Q) = b_d[ds] + alpha*log_r - beta*(x - mu0 - mu1*log_r)^2"""
    def predict(params, x, log_r, ds_idx):
        alpha, beta, mu_0, mu_1 = params[:4]
        b = np.array(params[4:4+n_ds])
        dev = x - mu_0 - mu_1 * log_r
        logit_val = b[ds_idx] + alpha * log_r - beta * dev ** 2
        return expit(np.clip(logit_val, -20, 20))

    def loss(params):
        return np.mean((Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)) ** 2)

    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    best = _optimize(loss, bounds)
    return predict(best.x, x_te, log_r_te, ds_idx_te) if best else np.full(len(x_te), 0.5)


def fit_skewed_gaussian(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                        x_te, log_r_te, ds_idx_te, n_ds):
    """logit(Q) = b_d + alpha*log_r - beta*(x-mu)^2 + gamma*(x-mu)^3"""
    def predict(params, x, log_r, ds_idx):
        alpha, beta, mu_0, mu_1, gamma = params[:5]
        b = np.array(params[5:5+n_ds])
        dev = x - mu_0 - mu_1 * log_r
        logit_val = b[ds_idx] + alpha * log_r - beta * dev**2 + gamma * dev**3
        return expit(np.clip(logit_val, -20, 20))

    def loss(params):
        return np.mean((Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)) ** 2)

    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5), (-20, 20)] + [(-10, 10)] * n_ds
    best = _optimize(loss, bounds)
    return predict(best.x, x_te, log_r_te, ds_idx_te) if best else np.full(len(x_te), 0.5)


def fit_asymmetric_gaussian(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                            x_te, log_r_te, ds_idx_te, n_ds):
    """logit(Q) = b_d + alpha*log_r - beta_L*(x-mu)^2 [x<mu] / -beta_R*(x-mu)^2 [x>=mu]"""
    def predict(params, x, log_r, ds_idx):
        alpha, beta_L, beta_R, mu_0, mu_1 = params[:5]
        b = np.array(params[5:5+n_ds])
        mu = mu_0 + mu_1 * log_r
        dev = x - mu
        beta = np.where(dev < 0, beta_L, beta_R)
        logit_val = b[ds_idx] + alpha * log_r - beta * dev**2
        return expit(np.clip(logit_val, -20, 20))

    def loss(params):
        return np.mean((Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)) ** 2)

    bounds = [(-1, 1), (0.01, 50), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    best = _optimize(loss, bounds)
    return predict(best.x, x_te, log_r_te, ds_idx_te) if best else np.full(len(x_te), 0.5)


def fit_beta_shape(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                   x_te, log_r_te, ds_idx_te, n_ds):
    """logit(Q) = b_d + alpha*log_r + a*log(x) + b*log(1-x)"""
    def predict(params, x, log_r, ds_idx):
        alpha, a, b_coef = params[:3]
        b = np.array(params[3:3+n_ds])
        x_c = np.clip(x, 0.01, 0.99)
        logit_val = b[ds_idx] + alpha * log_r + a * np.log(x_c) + b_coef * np.log(1 - x_c)
        return expit(np.clip(logit_val, -20, 20))

    def loss(params):
        return np.mean((Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)) ** 2)

    bounds = [(-1, 1), (0.01, 20), (0.01, 20)] + [(-10, 10)] * n_ds
    best = _optimize(loss, bounds)
    return predict(best.x, x_te, log_r_te, ds_idx_te) if best else np.full(len(x_te), 0.5)


def fit_cubic(x_tr, Q_tr, log_r_tr, ds_idx_tr,
              x_te, log_r_te, ds_idx_te, n_ds):
    """logit(Q) = b_d + alpha*log_r + c1*x + c2*x^2 + c3*x^3"""
    def predict(params, x, log_r, ds_idx):
        alpha, c1, c2, c3 = params[:4]
        b = np.array(params[4:4+n_ds])
        logit_val = b[ds_idx] + alpha * log_r + c1*x + c2*x**2 + c3*x**3
        return expit(np.clip(logit_val, -20, 20))

    def loss(params):
        return np.mean((Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)) ** 2)

    bounds = [(-1, 1), (-20, 20), (-50, 50), (-50, 50)] + [(-10, 10)] * n_ds
    best = _optimize(loss, bounds)
    return predict(best.x, x_te, log_r_te, ds_idx_te) if best else np.full(len(x_te), 0.5)


def fit_quartic(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                x_te, log_r_te, ds_idx_te, n_ds):
    """logit(Q) = b_d + alpha*log_r + c1*x + c2*x^2 + c3*x^3 + c4*x^4"""
    def predict(params, x, log_r, ds_idx):
        alpha, c1, c2, c3, c4 = params[:5]
        b = np.array(params[5:5+n_ds])
        logit_val = b[ds_idx] + alpha * log_r + c1*x + c2*x**2 + c3*x**3 + c4*x**4
        return expit(np.clip(logit_val, -20, 20))

    def loss(params):
        return np.mean((Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)) ** 2)

    bounds = [(-1, 1), (-30, 30), (-80, 80), (-80, 80), (-80, 80)] + [(-10, 10)] * n_ds
    best = _optimize(loss, bounds)
    return predict(best.x, x_te, log_r_te, ds_idx_te) if best else np.full(len(x_te), 0.5)


def run_lofo(obs, name, fitter_fn):
    """LOFO cross-validation with vectorized fitters."""
    families = sorted(set(o["family"] for o in obs))
    datasets = sorted(set(o["dataset"] for o in obs))
    n_ds = len(datasets)
    results = {}

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]

        x_tr, Q_tr, log_r_tr, ds_idx_tr = vectorize_obs(train, datasets)
        x_te, Q_te, log_r_te, ds_idx_te = vectorize_obs(test, datasets)

        try:
            pred = fitter_fn(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                           x_te, log_r_te, ds_idx_te, n_ds)
            mae, r2 = eval_fit(Q_te, pred)
            results[holdout_fam] = {"r2": r2, "mae": mae}
        except Exception as e:
            print(f"    ERROR on {holdout_fam}: {e}")
            results[holdout_fam] = {"r2": float("nan"), "mae": float("nan")}

    r2s = [v["r2"] for v in results.values() if not np.isnan(v["r2"])]
    maes = [v["mae"] for v in results.values() if not np.isnan(v["mae"])]
    return {
        "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "per_family": results,
    }


def main():
    print("Loading observations...")
    obs = load_all_observations()
    print(f"  Total: {len(obs)} obs")

    n_datasets = len(set(o["dataset"] for o in obs))

    shapes = {
        "Symmetric Gaussian": (fit_symmetric_gaussian, 4 + n_datasets),
        "Skewed Gaussian (+gamma)": (fit_skewed_gaussian, 5 + n_datasets),
        "Asymmetric Gaussian (beta_L/R)": (fit_asymmetric_gaussian, 5 + n_datasets),
        "Beta shape (a,b)": (fit_beta_shape, 3 + n_datasets),
        "Free cubic": (fit_cubic, 4 + n_datasets),
        "Quartic polynomial": (fit_quartic, 5 + n_datasets),
    }

    results = {}

    print("\n" + "=" * 70)
    print("  ALTERNATIVE SHAPE COMPARISON (LOFO)")
    print("=" * 70)

    for name, (fitter, n_params) in shapes.items():
        print(f"\n  {name} ({n_params}p):", flush=True)
        result = run_lofo(obs, name, fitter)
        result["n_params"] = n_params
        results[name] = result
        print(f"    Mean R2={result['mean_r2']:.4f}, MAE={result['mean_mae']:.4f}")
        for fam, fdata in result["per_family"].items():
            print(f"      {fam:>15s}: R2={fdata['r2']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("  SHAPE COMPARISON SUMMARY")
    print("=" * 70)

    sorted_names = sorted(results.keys(), key=lambda k: results[k]["mean_r2"], reverse=True)
    for name in sorted_names:
        r = results[name]
        print(f"  {name:>35s} ({r['n_params']}p): R2={r['mean_r2']:.4f}, MAE={r['mean_mae']:.4f}")

    # Family-level variance
    print("\n" + "=" * 70)
    print("  FAMILY-LEVEL VARIANCE (SD of R2 across families)")
    print("=" * 70)

    for name in sorted_names:
        r2_vals = [v["r2"] for v in results[name]["per_family"].values()
                   if not np.isnan(v["r2"])]
        sd = np.std(r2_vals) if len(r2_vals) > 1 else 0
        min_r2 = min(r2_vals) if r2_vals else float("nan")
        max_r2 = max(r2_vals) if r2_vals else float("nan")
        print(f"  {name:>35s}: SD={sd:.4f}, min={min_r2:.4f}, max={max_r2:.4f}")

    # Delta from baseline
    baseline_r2 = results["Symmetric Gaussian"]["mean_r2"]
    baseline_mae = results["Symmetric Gaussian"]["mean_mae"]

    print("\n" + "=" * 70)
    print("  DELTA FROM SYMMETRIC GAUSSIAN BASELINE")
    print("=" * 70)

    for name in sorted_names:
        if name == "Symmetric Gaussian":
            continue
        dr2 = results[name]["mean_r2"] - baseline_r2
        dmae = results[name]["mean_mae"] - baseline_mae
        extra_p = results[name]["n_params"] - results["Symmetric Gaussian"]["n_params"]
        sign = "+" if extra_p >= 0 else ""
        print(f"  {name:>35s}: dR2={dr2:+.4f}, dMAE={dmae:+.4f}, {sign}{extra_p}p")

    # Save
    out_path = RESULTS_DIR / "cti_shape_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
