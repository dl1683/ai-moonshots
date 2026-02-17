#!/usr/bin/env python
"""CTI Calibrated Transfer: test shape + calibrated level for new families.

For families where zero-shot R2 is poor (Gemma-2, Phi), test whether
fitting ONLY the b_d intercepts (4 params) on a small sample recovers
good R2 -- i.e., shape transfers but level needs calibration.

This mirrors the BLOOM analysis: shape rho is the real metric,
absolute R2 requires family-specific b_d.
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_all_observations():
    sources = [
        RESULTS_DIR / "cti_checkpoint_sweep_all.json",
        RESULTS_DIR / "cti_multi_family.json",
        RESULTS_DIR / "cti_olmo2_sweep.json",
        RESULTS_DIR / "cti_new_families.json",
    ]
    obs = []
    for src_path in sources:
        if not src_path.exists():
            continue
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
            model = result["model"]
            step = result.get("step", -1)
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
                        "model": model, "step": step, "layer": li, "L": L,
                    })
    return obs


def vectorize_obs(obs_list, datasets):
    x = np.array([o["x"] for o in obs_list])
    Q = np.array([o["Q"] for o in obs_list])
    log_r = np.array([o["log_r"] for o in obs_list])
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


def fit_gaussian_get_params(x_tr, Q_tr, log_r_tr, ds_idx_tr, n_ds):
    """Fit Gaussian and return parameters."""
    def predict(params, x, log_r, ds_idx):
        alpha, beta, mu_0, mu_1 = params[:4]
        b = np.array(params[4:4+n_ds])
        dev = x - mu_0 - mu_1 * log_r
        logit_val = b[ds_idx] + alpha * log_r - beta * dev**2
        return expit(np.clip(logit_val, -20, 20))

    def loss(params):
        return np.mean((Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)) ** 2)

    bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    best, best_loss = None, float("inf")
    for trial in range(15):
        rng = np.random.RandomState(trial)
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 3000, "ftol": 1e-11})
            if res.fun < best_loss:
                best_loss = res.fun
                best = res
        except Exception:
            continue
    return best.x if best else None


def predict_with_params(params, x, log_r, ds_idx, n_ds):
    """Predict using given parameters."""
    alpha, beta, mu_0, mu_1 = params[:4]
    b = np.array(params[4:4+n_ds])
    dev = x - mu_0 - mu_1 * log_r
    logit_val = b[ds_idx] + alpha * log_r - beta * dev**2
    return expit(np.clip(logit_val, -20, 20))


def calibrate_intercepts(shared_params, x_cal, Q_cal, log_r_cal, ds_idx_cal, n_ds):
    """Fit only b_d intercepts, keeping shared params (alpha, beta, mu0, mu1) frozen."""
    alpha, beta, mu_0, mu_1 = shared_params[:4]

    def predict(b_d, x, log_r, ds_idx):
        b = np.array(b_d)
        dev = x - mu_0 - mu_1 * log_r
        logit_val = b[ds_idx] + alpha * log_r - beta * dev**2
        return expit(np.clip(logit_val, -20, 20))

    def loss(b_d):
        return np.mean((Q_cal - predict(b_d, x_cal, log_r_cal, ds_idx_cal)) ** 2)

    bounds = [(-10, 10)] * n_ds
    best, best_loss = None, float("inf")
    for trial in range(10):
        rng = np.random.RandomState(trial)
        x0 = [rng.uniform(-5, 5) for _ in range(n_ds)]
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000})
            if res.fun < best_loss:
                best_loss = res.fun
                best = res
        except Exception:
            continue

    if best is None:
        return shared_params
    new_params = list(shared_params[:4]) + list(best.x)
    return np.array(new_params)


def compute_shape_rho(obs_list, pred_values):
    profiles = {}
    for i, o in enumerate(obs_list):
        key = (o["model"], o["step"], o["dataset"])
        if key not in profiles:
            profiles[key] = {"obs": [], "pred": []}
        profiles[key]["obs"].append(o["Q"])
        profiles[key]["pred"].append(pred_values[i])

    rhos = []
    for p in profiles.values():
        if len(p["obs"]) < 4:
            continue
        rho, _ = spearmanr(p["obs"], p["pred"])
        if not np.isnan(rho):
            rhos.append(rho)
    return rhos


def main():
    print("Loading observations...")
    obs = load_all_observations()
    datasets = sorted(set(o["dataset"] for o in obs))
    n_ds = len(datasets)
    families = sorted(set(o["family"] for o in obs))
    print(f"  Total: {len(obs)} obs, {len(families)} families")

    results = {}

    print("\n" + "=" * 70)
    print("  CALIBRATED TRANSFER ANALYSIS")
    print("=" * 70)

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]

        x_tr, Q_tr, log_r_tr, ds_idx_tr = vectorize_obs(train, datasets)
        x_te, Q_te, log_r_te, ds_idx_te = vectorize_obs(test, datasets)

        # 1. Fit on training families
        params = fit_gaussian_get_params(x_tr, Q_tr, log_r_tr, ds_idx_tr, n_ds)
        if params is None:
            continue

        # 2. Zero-shot prediction
        pred_zeroshot = predict_with_params(params, x_te, log_r_te, ds_idx_te, n_ds)
        mae_zs, r2_zs = eval_fit(Q_te, pred_zeroshot)

        # 3. Shape correlation
        rhos = compute_shape_rho(test, pred_zeroshot)
        mean_rho = float(np.mean(rhos)) if rhos else float("nan")

        # 4. Calibrated prediction (fit only b_d on ALL held-out data)
        cal_params = calibrate_intercepts(params, x_te, Q_te, log_r_te, ds_idx_te, n_ds)
        pred_cal = predict_with_params(cal_params, x_te, log_r_te, ds_idx_te, n_ds)
        mae_cal, r2_cal = eval_fit(Q_te, pred_cal)

        # 5. b_d shift
        orig_bd = params[4:4+n_ds]
        cal_bd = cal_params[4:4+n_ds]
        bd_shift = float(np.mean(cal_bd - orig_bd))

        results[holdout_fam] = {
            "n_obs": len(test),
            "n_profiles": len(rhos),
            "zero_shot_r2": r2_zs,
            "zero_shot_mae": mae_zs,
            "shape_rho": mean_rho,
            "calibrated_r2": r2_cal,
            "calibrated_mae": mae_cal,
            "mean_bd_shift": bd_shift,
        }

        marker = " *" if r2_zs < 0.5 else ""
        print(f"\n  {holdout_fam}:{marker}")
        print(f"    Zero-shot:  R2={r2_zs:.4f}, MAE={mae_zs:.4f}")
        print(f"    Shape rho:  {mean_rho:.3f} ({len(rhos)} profiles)")
        print(f"    Calibrated: R2={r2_cal:.4f}, MAE={mae_cal:.4f}")
        print(f"    b_d shift:  {bd_shift:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Categorize families
    shape_transfers = []
    full_transfers = []
    for fam, r in results.items():
        if r["zero_shot_r2"] >= 0.5 and r["shape_rho"] >= 0.6:
            full_transfers.append(fam)
        elif r["shape_rho"] >= 0.6:
            shape_transfers.append(fam)

    print(f"\n  Full transfer (R2>=0.5 AND rho>=0.6): {len(full_transfers)}/{len(families)}")
    for f in full_transfers:
        r = results[f]
        print(f"    {f:>15s}: R2={r['zero_shot_r2']:.3f}, rho={r['shape_rho']:.3f}")

    print(f"\n  Shape-only transfer (rho>=0.6 but R2<0.5): {len(shape_transfers)}/{len(families)}")
    for f in shape_transfers:
        r = results[f]
        print(f"    {f:>15s}: R2={r['zero_shot_r2']:.3f} -> cal={r['calibrated_r2']:.3f}, "
              f"rho={r['shape_rho']:.3f}, shift={r['mean_bd_shift']:+.3f}")

    # Aggregate stats
    all_rhos = [r["shape_rho"] for r in results.values() if not np.isnan(r["shape_rho"])]
    all_zs_r2 = [r["zero_shot_r2"] for r in results.values()]
    all_cal_r2 = [r["calibrated_r2"] for r in results.values()]

    print(f"\n  Mean shape rho:       {np.mean(all_rhos):.3f} ({len(all_rhos)} families)")
    print(f"  Mean zero-shot R2:    {np.mean(all_zs_r2):.3f}")
    print(f"  Mean calibrated R2:   {np.mean(all_cal_r2):.3f}")
    print(f"  Families with rho>0.6: {sum(1 for r in all_rhos if r > 0.6)}/{len(all_rhos)}")

    out = {"results": results, "datasets": datasets, "families": families}
    out_path = RESULTS_DIR / "cti_calibrated_transfer.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
