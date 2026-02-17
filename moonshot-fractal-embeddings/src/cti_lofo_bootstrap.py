#!/usr/bin/env python
"""CTI LOFO Bootstrap: held-out per-profile deltas with proper uncertainty.

Fixes the in-sample issue from cti_robust_analysis.py:
- For each LOFO fold, fit models on train families, predict on held-out family
- Compute per-profile R2 deltas on held-out predictions only
- Bootstrap the 252 held-out profile deltas for CI
- Also adds GAM (thin-plate spline) baseline with depth-compute interaction
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.interpolate import RBFInterpolator
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_all_observations(exclude_step0=True, metric="knn_l1"):
    """Load observations from ALL three data sources (5 families, 5004 obs)."""
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
            if exclude_step0 and result.get("step", -1) == 0:
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
                    Q_raw = layer_data.get(metric)
                    if Q_raw is None:
                        continue
                    Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                    Q_norm = np.clip(Q_norm, 0.001, 0.999)

                    obs.append({
                        "x": x, "Q": Q_norm, "dataset": ds_name,
                        "model": model, "step": step, "layer": li, "L": L,
                        "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                        "family": family,
                        "profile_id": f"{model}_{step}_{ds_name}",
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
# MODEL FITTERS
# ==========================================

def fit_linear(train_obs, test_obs, datasets):
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
    return predict(best.x, test_obs) if best else np.full(len(test_obs), 0.5)


def fit_gaussian(train_obs, test_obs, datasets):
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
    return predict(best.x, test_obs) if best else np.full(len(test_obs), 0.5)


def fit_gam(train_obs, test_obs, datasets):
    """GAM-like: RBF interpolation in (x, log_r) per dataset.
    Uses thin-plate spline kernel - the smooth competitor Codex asked for.
    """
    preds = np.zeros(len(test_obs))

    for ds in datasets:
        tr_idx = [i for i, o in enumerate(train_obs) if o["dataset"] == ds]
        te_idx = [i for i, o in enumerate(test_obs) if o["dataset"] == ds]

        if not tr_idx or not te_idx:
            # No training data for this dataset, use global mean
            mean_q = np.mean([train_obs[i]["Q"] for i in range(len(train_obs))])
            for j in te_idx:
                preds[j] = mean_q
            continue

        X_tr = np.array([[train_obs[i]["x"], train_obs[i]["log_r"]] for i in tr_idx])
        y_tr = np.array([train_obs[i]["Q"] for i in tr_idx])
        X_te = np.array([[test_obs[j]["x"], test_obs[j]["log_r"]] for j in te_idx])

        try:
            # Thin-plate spline RBF with smoothing
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


# ==========================================
# MAIN: LOFO with held-out per-profile deltas
# ==========================================

def main():
    print("Loading observations...")
    obs = load_all_observations()

    families = sorted(set(o["family"] for o in obs))
    datasets = sorted(set(o["dataset"] for o in obs))
    profiles = sorted(set(o["profile_id"] for o in obs))

    print(f"  Total: {len(obs)} obs, {len(families)} families, {len(profiles)} profiles")
    for fam in families:
        n = sum(1 for o in obs if o["family"] == fam)
        n_prof = len(set(o["profile_id"] for o in obs if o["family"] == fam))
        print(f"    {fam}: {n} obs, {n_prof} profiles")

    # Map profile -> observations
    profile_obs = {}
    profile_family = {}
    for o in obs:
        profile_obs.setdefault(o["profile_id"], []).append(o)
        profile_family[o["profile_id"]] = o["family"]

    model_names = ["Linear", "Gaussian", "GAM"]
    model_fitters = [fit_linear, fit_gaussian, fit_gam]

    # Store per-profile held-out predictions for each model
    # Key: (model_name, profile_id) -> (Q_obs_array, Q_pred_array)
    held_out_predictions = {name: {} for name in model_names}

    # Also store fold-level R2s
    fold_results = {name: {} for name in model_names}

    print("\n" + "=" * 70)
    print("  LOFO CV with held-out per-profile analysis")
    print("=" * 70)

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]
        Q_test = np.array([o["Q"] for o in test])

        print(f"\n  Holdout: {holdout_fam} (n_test={len(test)})")

        for name, fitter in zip(model_names, model_fitters):
            try:
                if name == "GAM":
                    pred = fitter(train, test, datasets)
                else:
                    pred = fitter(train, test, datasets)
                mae, r2 = eval_fit(Q_test, pred)
                fold_results[name][holdout_fam] = {"r2": r2, "mae": mae}
                print(f"    {name:>10s}: R2={r2:.4f}, MAE={mae:.4f}")

                # Store per-profile predictions
                for prof in set(o["profile_id"] for o in test):
                    prof_idx = [i for i, o in enumerate(test) if o["profile_id"] == prof]
                    Q_obs = np.array([Q_test[i] for i in prof_idx])
                    Q_pred = np.array([pred[i] for i in prof_idx])
                    held_out_predictions[name][prof] = (Q_obs, Q_pred)

            except Exception as e:
                print(f"    {name:>10s}: FAILED ({e})")
                fold_results[name][holdout_fam] = {"r2": float("nan"), "mae": float("nan")}

    # ==========================================
    # Summary: LOFO R2 per model
    # ==========================================
    print("\n" + "=" * 70)
    print("  LOFO SUMMARY")
    print("=" * 70)

    for name in model_names:
        r2s = [v["r2"] for v in fold_results[name].values() if not np.isnan(v["r2"])]
        maes = [v["mae"] for v in fold_results[name].values() if not np.isnan(v["mae"])]
        if r2s:
            print(f"  {name:>10s}: mean R2={np.mean(r2s):.4f}, mean MAE={np.mean(maes):.4f}")

    # ==========================================
    # Held-out per-profile delta analysis
    # ==========================================
    print("\n" + "=" * 70)
    print("  HELD-OUT PER-PROFILE DELTA R2 (Gaussian vs Linear)")
    print("=" * 70)

    profile_deltas_gl = []  # Gaussian - Linear
    profile_deltas_gg = []  # Gaussian - GAM
    profile_mae_deltas = []  # MAE deltas (Gaussian - Linear, negative = Gaussian better)
    profile_families_list = []

    for prof in profiles:
        if prof not in held_out_predictions["Gaussian"] or prof not in held_out_predictions["Linear"]:
            continue

        Q_obs_g, Q_pred_g = held_out_predictions["Gaussian"][prof]
        Q_obs_l, Q_pred_l = held_out_predictions["Linear"][prof]

        if len(Q_obs_g) < 3:
            continue

        ss_tot = np.sum((Q_obs_g - Q_obs_g.mean()) ** 2)
        if ss_tot < 1e-10:
            continue

        r2_g = 1 - np.sum((Q_obs_g - Q_pred_g) ** 2) / ss_tot
        r2_l = 1 - np.sum((Q_obs_l - Q_pred_l) ** 2) / ss_tot
        mae_g = np.mean(np.abs(Q_obs_g - Q_pred_g))
        mae_l = np.mean(np.abs(Q_obs_l - Q_pred_l))

        profile_deltas_gl.append(r2_g - r2_l)
        profile_mae_deltas.append(mae_g - mae_l)  # negative = Gaussian better
        profile_families_list.append(profile_family[prof])

        if prof in held_out_predictions["GAM"]:
            Q_obs_gam, Q_pred_gam = held_out_predictions["GAM"][prof]
            r2_gam = 1 - np.sum((Q_obs_gam - Q_pred_gam) ** 2) / ss_tot
            profile_deltas_gg.append(r2_g - r2_gam)

    profile_deltas_gl = np.array(profile_deltas_gl)
    profile_mae_deltas = np.array(profile_mae_deltas)
    profile_deltas_gg = np.array(profile_deltas_gg) if profile_deltas_gg else np.array([])

    n_pos = (profile_deltas_gl > 0).sum()
    n_neg = (profile_deltas_gl < 0).sum()

    print(f"  N profiles: {len(profile_deltas_gl)}")
    print(f"  Delta R2 (Gaussian - Linear):")
    print(f"    Mean:   {profile_deltas_gl.mean():+.4f}")
    print(f"    Median: {np.median(profile_deltas_gl):+.4f}")
    print(f"    SD:     {profile_deltas_gl.std():.4f}")
    print(f"    +/-:    {n_pos}/{n_neg}")

    print(f"  Delta MAE (Gaussian - Linear, negative = Gaussian better):")
    print(f"    Mean:   {profile_mae_deltas.mean():+.4f}")
    print(f"    Median: {np.median(profile_mae_deltas):+.4f}")

    # Per-family breakdown
    per_family_deltas = {}
    for fam in families:
        fam_d = [profile_deltas_gl[i] for i in range(len(profile_deltas_gl))
                 if profile_families_list[i] == fam]
        if fam_d:
            fam_d = np.array(fam_d)
            per_family_deltas[fam] = fam_d.tolist()
            print(f"    {fam:>15s}: n={len(fam_d)}, mean={fam_d.mean():+.4f}, "
                  f"median={np.median(fam_d):+.4f}, pct+={100*(fam_d>0).mean():.0f}%")

    # Cluster-robust paired tests (on held-out predictions)
    from scipy.stats import ttest_1samp, wilcoxon

    t_stat, t_pval = ttest_1samp(profile_deltas_gl, 0)
    try:
        w_stat, w_pval = wilcoxon(profile_deltas_gl)
    except Exception:
        w_stat, w_pval = float("nan"), float("nan")

    print(f"\n  Paired t-test (held-out): t={t_stat:.3f}, p={t_pval:.2e}")
    print(f"  Wilcoxon (held-out): W={w_stat}, p={w_pval:.2e}")

    # ==========================================
    # Bootstrap CI (held-out profile deltas)
    # ==========================================
    print("\n" + "=" * 70)
    print("  BOOTSTRAP CI (held-out, stratified by family)")
    print("=" * 70)

    n_boot = 10000
    rng = np.random.RandomState(42)
    boot_means = []
    boot_lofo_means = []
    boot_mae_means = []

    for b in range(n_boot):
        if b % 2000 == 0:
            print(f"  Bootstrap {b}/{n_boot}...")

        # Stratified bootstrap
        boot_deltas = []
        boot_lofo = []
        boot_mae = []
        for fam in families:
            if fam not in per_family_deltas:
                continue
            fam_d = np.array(per_family_deltas[fam])
            boot_sample = rng.choice(fam_d, size=len(fam_d), replace=True)
            boot_deltas.extend(boot_sample)
            boot_lofo.append(boot_sample.mean())

            # MAE deltas for this family
            fam_mae_d = np.array([profile_mae_deltas[i] for i in range(len(profile_mae_deltas))
                                   if profile_families_list[i] == fam])
            boot_mae_sample = rng.choice(fam_mae_d, size=len(fam_mae_d), replace=True)
            boot_mae.extend(boot_mae_sample)

        boot_means.append(np.mean(boot_deltas))
        boot_lofo_means.append(np.mean(boot_lofo))
        boot_mae_means.append(np.mean(boot_mae))

    boot_means = np.array(boot_means)
    boot_lofo_means = np.array(boot_lofo_means)
    boot_mae_means = np.array(boot_mae_means)

    ci_025, ci_975 = np.percentile(boot_means, [2.5, 97.5])
    ci_05, ci_95 = np.percentile(boot_means, [5, 95])
    lofo_025, lofo_975 = np.percentile(boot_lofo_means, [2.5, 97.5])
    mae_025, mae_975 = np.percentile(boot_mae_means, [2.5, 97.5])

    print(f"\n  Profile-level delta R2 (held-out):")
    print(f"    Mean:     {boot_means.mean():+.4f}")
    print(f"    95% CI:   [{ci_025:+.4f}, {ci_975:+.4f}]")
    print(f"    P(>0):    {(boot_means > 0).mean():.3f}")

    print(f"\n  LOFO-style delta R2 (equal family weight):")
    print(f"    Mean:     {boot_lofo_means.mean():+.4f}")
    print(f"    95% CI:   [{lofo_025:+.4f}, {lofo_975:+.4f}]")
    print(f"    P(>0):    {(boot_lofo_means > 0).mean():.3f}")

    print(f"\n  MAE delta (Gaussian - Linear, negative = Gaussian better):")
    print(f"    Mean:     {boot_mae_means.mean():+.4f}")
    print(f"    95% CI:   [{mae_025:+.4f}, {mae_975:+.4f}]")

    # GAM comparison
    if len(profile_deltas_gg) > 0:
        print(f"\n  Gaussian vs GAM (held-out profiles):")
        print(f"    Mean delta R2: {profile_deltas_gg.mean():+.4f}")
        print(f"    Median:        {np.median(profile_deltas_gg):+.4f}")
        print(f"    Pct Gaussian > GAM: {100*(profile_deltas_gg>0).mean():.0f}%")

    # ==========================================
    # Save results
    # ==========================================
    results = {
        "lofo_summary": {
            name: {
                "mean_r2": float(np.mean([v["r2"] for v in fold_results[name].values()
                                          if not np.isnan(v["r2"])])),
                "mean_mae": float(np.mean([v["mae"] for v in fold_results[name].values()
                                           if not np.isnan(v["mae"])])),
                "per_family": {k: v for k, v in fold_results[name].items()},
            }
            for name in model_names
        },
        "held_out_profile_delta": {
            "n_profiles": int(len(profile_deltas_gl)),
            "gaussian_vs_linear": {
                "mean": float(profile_deltas_gl.mean()),
                "median": float(np.median(profile_deltas_gl)),
                "sd": float(profile_deltas_gl.std()),
                "n_positive": int(n_pos),
                "n_negative": int(n_neg),
                "paired_t_stat": float(t_stat),
                "paired_t_pval": float(t_pval),
                "wilcoxon_pval": float(w_pval) if not np.isnan(w_pval) else None,
            },
            "gaussian_vs_gam": {
                "mean": float(profile_deltas_gg.mean()) if len(profile_deltas_gg) > 0 else None,
                "median": float(np.median(profile_deltas_gg)) if len(profile_deltas_gg) > 0 else None,
                "pct_gaussian_wins": float((profile_deltas_gg > 0).mean()) if len(profile_deltas_gg) > 0 else None,
            },
            "mae_delta": {
                "mean": float(profile_mae_deltas.mean()),
                "median": float(np.median(profile_mae_deltas)),
            },
        },
        "bootstrap_ci": {
            "n_boot": n_boot,
            "profile_mean_r2_delta": {
                "mean": float(boot_means.mean()),
                "ci95": [float(ci_025), float(ci_975)],
                "prob_positive": float((boot_means > 0).mean()),
            },
            "lofo_mean_r2_delta": {
                "mean": float(boot_lofo_means.mean()),
                "ci95": [float(lofo_025), float(lofo_975)],
                "prob_positive": float((boot_lofo_means > 0).mean()),
            },
            "mae_delta": {
                "mean": float(boot_mae_means.mean()),
                "ci95": [float(mae_025), float(mae_975)],
            },
        },
        "per_family": {fam: {"n": len(d), "mean": float(np.mean(d)), "median": float(np.median(d))}
                       for fam, d in per_family_deltas.items()},
    }

    out_path = RESULTS_DIR / "cti_lofo_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
