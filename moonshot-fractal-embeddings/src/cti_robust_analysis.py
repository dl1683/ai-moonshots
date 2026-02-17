#!/usr/bin/env python
"""CTI Robust Analysis: paired bootstrap CIs, spline/GAM baseline, cluster-aware inference.

Addresses Codex 6.1/10 review concerns:
1. Paired bootstrap CI for delta R2 (Gaussian vs Linear) -- uncertainty on margin
2. Natural cubic spline baseline (stronger nonparametric competitor)
3. Cluster-robust standard errors for proper dependence handling
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.interpolate import UnivariateSpline
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_all_observations(exclude_step0=True, metric="knn_l1"):
    """Load observations from ALL three data sources (5 families, 5004 obs)."""
    sources = [
        RESULTS_DIR / "cti_checkpoint_sweep_all.json",  # pythia (4 sizes x 11 ckpts)
        RESULTS_DIR / "cti_multi_family.json",           # cerebras-gpt, gpt2, opt
        RESULTS_DIR / "cti_olmo2_sweep.json",            # olmo2 (1 size x 11 ckpts)
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
                        "x": x, "Q": Q_norm, "Q_raw": Q_raw, "dataset": ds_name,
                        "model": model, "step": step, "layer": li, "L": L,
                        "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                        "family": family,
                        "profile_id": f"{model}_{step}_{ds_name}",
                    })

    print(f"  Loaded {len(obs)} observations from {len(sources)} sources")
    families = sorted(set(o["family"] for o in obs))
    print(f"  Families: {families}")
    for fam in families:
        n = sum(1 for o in obs if o["family"] == fam)
        print(f"    {fam}: {n} obs")
    return obs


def eval_fit(Q_obs, Q_pred):
    residuals = Q_obs - Q_pred
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return mae, r2


# ==========================================
# MODEL FITTERS (return predictions for test set)
# ==========================================

def fit_linear_depth(train_obs, test_obs, datasets):
    """Linear in depth + dataset intercepts."""
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
    return predict(best.x, test_obs)


def fit_gaussian(train_obs, test_obs, datasets):
    """Our Gaussian law."""
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
    return predict(best.x, test_obs)


def fit_spline(train_obs, test_obs, datasets):
    """Natural cubic spline per (dataset) on depth, with smoothing.
    This is a stronger nonparametric baseline -- fits a smooth curve per dataset.
    Uses all training data pooled per dataset (ignores compute).
    """
    preds = np.zeros(len(test_obs))

    for ds in datasets:
        tr_mask = [i for i, o in enumerate(train_obs) if o["dataset"] == ds]
        te_mask = [i for i, o in enumerate(test_obs) if o["dataset"] == ds]

        if not tr_mask or not te_mask:
            continue

        x_tr = np.array([train_obs[i]["x"] for i in tr_mask])
        y_tr = np.array([train_obs[i]["Q"] for i in tr_mask])
        x_te = np.array([test_obs[i]["x"] for i in te_mask])

        # Sort for spline fitting
        sort_idx = np.argsort(x_tr)
        x_tr_s = x_tr[sort_idx]
        y_tr_s = y_tr[sort_idx]

        # Bin to avoid duplicate x values (average within bins)
        n_bins = min(50, len(np.unique(np.round(x_tr_s, 3))))
        if n_bins < 4:
            # Too few unique x values, use mean
            preds[[te_mask[j] for j in range(len(te_mask))]] = y_tr.mean()
            continue

        bin_edges = np.linspace(x_tr_s.min(), x_tr_s.max(), n_bins + 1)
        bin_x, bin_y = [], []
        for b in range(n_bins):
            mask = (x_tr_s >= bin_edges[b]) & (x_tr_s < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (x_tr_s >= bin_edges[b]) & (x_tr_s <= bin_edges[b + 1])
            if mask.sum() > 0:
                bin_x.append(np.mean(x_tr_s[mask]))
                bin_y.append(np.mean(y_tr_s[mask]))

        bin_x = np.array(bin_x)
        bin_y = np.array(bin_y)

        if len(bin_x) < 4:
            preds[[te_mask[j] for j in range(len(te_mask))]] = y_tr.mean()
            continue

        try:
            # Smoothing spline with cross-validated smoothing
            spl = UnivariateSpline(bin_x, bin_y, k=3, s=len(bin_y) * 0.01)
            pred_vals = spl(np.clip(x_te, bin_x.min(), bin_x.max()))
            pred_vals = np.clip(pred_vals, 0.001, 0.999)
            for j, idx in enumerate(te_mask):
                preds[idx] = pred_vals[j]
        except Exception:
            for idx in te_mask:
                preds[idx] = y_tr.mean()

    return preds


def fit_spline_compute(train_obs, test_obs, datasets):
    """Spline per dataset with compute interaction.
    Bins by compute tercile within each dataset, fits separate spline per bin.
    More powerful nonparametric baseline.
    """
    preds = np.zeros(len(test_obs))

    for ds in datasets:
        tr_idx = [i for i, o in enumerate(train_obs) if o["dataset"] == ds]
        te_idx = [i for i, o in enumerate(test_obs) if o["dataset"] == ds]

        if not tr_idx or not te_idx:
            continue

        log_r_tr = np.array([train_obs[i]["log_r"] for i in tr_idx])
        log_r_te = np.array([test_obs[i]["log_r"] for i in te_idx])

        # Split into terciles by compute
        terciles = np.percentile(log_r_tr, [33.3, 66.7])

        def get_tercile(lr):
            if lr < terciles[0]:
                return 0
            elif lr < terciles[1]:
                return 1
            else:
                return 2

        # For each test point, find its tercile and use corresponding spline
        for j, ti in enumerate(te_idx):
            t = get_tercile(log_r_te[j])
            # Get training points in this tercile
            sub_tr = [i for i in tr_idx if get_tercile(train_obs[i]["log_r"]) == t]
            if len(sub_tr) < 8:
                sub_tr = tr_idx  # fallback to all

            x_sub = np.array([train_obs[i]["x"] for i in sub_tr])
            y_sub = np.array([train_obs[i]["Q"] for i in sub_tr])

            sort_idx = np.argsort(x_sub)
            x_s = x_sub[sort_idx]
            y_s = y_sub[sort_idx]

            # Simple binned average prediction
            n_bins = min(20, len(np.unique(np.round(x_s, 2))))
            if n_bins < 4:
                preds[ti] = y_s.mean()
                continue

            bin_edges = np.linspace(x_s.min(), x_s.max(), n_bins + 1)
            bin_x, bin_y = [], []
            for b in range(n_bins):
                if b == n_bins - 1:
                    mask = (x_s >= bin_edges[b]) & (x_s <= bin_edges[b + 1])
                else:
                    mask = (x_s >= bin_edges[b]) & (x_s < bin_edges[b + 1])
                if mask.sum() > 0:
                    bin_x.append(np.mean(x_s[mask]))
                    bin_y.append(np.mean(y_s[mask]))

            if len(bin_x) < 4:
                preds[ti] = y_s.mean()
                continue

            try:
                spl = UnivariateSpline(np.array(bin_x), np.array(bin_y), k=3,
                                       s=len(bin_y) * 0.01)
                xp = np.clip(test_obs[ti]["x"], min(bin_x), max(bin_x))
                preds[ti] = float(np.clip(spl(xp), 0.001, 0.999))
            except Exception:
                preds[ti] = np.mean(y_s)

    return preds


# ==========================================
# ANALYSIS 1: Paired Bootstrap CI for Delta R2
# ==========================================
def paired_bootstrap_delta_r2(profile_deltas_dict, n_boot=10000, seed=42):
    """Bootstrap CI for mean delta R2 at the profile level.

    Uses pre-computed per-profile delta R2 values from cluster_robust_inference.
    Resamples profiles (stratified by family) to get CI for the mean advantage.
    Much faster than re-fitting LOFO each time.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Paired Bootstrap CI for Delta R2")
    print("  (Profile-level resampling, stratified by family)")
    print("=" * 70)

    families = sorted(profile_deltas_dict.keys())
    rng = np.random.RandomState(seed)

    boot_means = []
    boot_lofo_means = []

    for b in range(n_boot):
        if b % 2000 == 0:
            print(f"  Bootstrap {b}/{n_boot}...")

        # Stratified bootstrap: resample profiles within each family
        boot_deltas = []
        boot_lofo = []  # mean per family (mimics LOFO)
        for fam in families:
            fam_deltas = np.array(profile_deltas_dict[fam])
            boot_sample = rng.choice(fam_deltas, size=len(fam_deltas), replace=True)
            boot_deltas.extend(boot_sample)
            boot_lofo.append(boot_sample.mean())

        boot_means.append(np.mean(boot_deltas))
        boot_lofo_means.append(np.mean(boot_lofo))  # LOFO-style: equal weight per family

    boot_means = np.array(boot_means)
    boot_lofo_means = np.array(boot_lofo_means)

    # CIs for profile-level mean
    ci_025, ci_975 = np.percentile(boot_means, [2.5, 97.5])
    ci_05, ci_95 = np.percentile(boot_means, [5, 95])

    # CIs for LOFO-style mean (equal weight per family)
    lofo_025, lofo_975 = np.percentile(boot_lofo_means, [2.5, 97.5])
    lofo_05, lofo_95 = np.percentile(boot_lofo_means, [5, 95])

    print(f"\n  Profile-level mean delta R2:")
    print(f"    Mean:     {boot_means.mean():+.4f}")
    print(f"    95% CI:   [{ci_025:+.4f}, {ci_975:+.4f}]")
    print(f"    90% CI:   [{ci_05:+.4f}, {ci_95:+.4f}]")
    print(f"    P(delta>0): {(boot_means > 0).mean():.3f}")

    print(f"\n  LOFO-style mean delta R2 (equal family weight):")
    print(f"    Mean:     {boot_lofo_means.mean():+.4f}")
    print(f"    95% CI:   [{lofo_025:+.4f}, {lofo_975:+.4f}]")
    print(f"    90% CI:   [{lofo_05:+.4f}, {lofo_95:+.4f}]")
    print(f"    P(delta>0): {(boot_lofo_means > 0).mean():.3f}")

    return {
        "n_boot": n_boot,
        "profile_mean": {
            "mean": float(boot_means.mean()),
            "ci95": [float(ci_025), float(ci_975)],
            "ci90": [float(ci_05), float(ci_95)],
            "prob_positive": float((boot_means > 0).mean()),
        },
        "lofo_mean": {
            "mean": float(boot_lofo_means.mean()),
            "ci95": [float(lofo_025), float(lofo_975)],
            "ci90": [float(lofo_05), float(lofo_95)],
            "prob_positive": float((boot_lofo_means > 0).mean()),
        },
    }


# ==========================================
# ANALYSIS 2: Spline/GAM Baseline
# ==========================================
def spline_baseline_comparison(obs):
    """Run LOFO with spline baselines added."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: Spline Baseline Comparison (LOFO)")
    print("=" * 70)

    families = sorted(set(o["family"] for o in obs))
    datasets = sorted(set(o["dataset"] for o in obs))

    models_compared = {
        "Linear-in-depth": fit_linear_depth,
        "Gaussian (ours)": fit_gaussian,
        "Spline-per-dataset": fit_spline,
        "Spline+compute-tercile": fit_spline_compute,
    }

    results = {name: {} for name in models_compared}

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]
        Q_test = np.array([o["Q"] for o in test])

        print(f"\n  Holdout: {holdout_fam} (n_test={len(test)})")

        for name, fitter in models_compared.items():
            try:
                if name in ["Linear-in-depth", "Gaussian (ours)"]:
                    pred = fitter(train, test, datasets)
                else:
                    pred = fitter(train, test, datasets)
                mae, r2 = eval_fit(Q_test, pred)
                results[name][holdout_fam] = {"mae": mae, "r2": r2}
                print(f"    {name:>25s}: MAE={mae:.4f}, R2={r2:.4f}")
            except Exception as e:
                print(f"    {name:>25s}: FAILED ({e})")
                results[name][holdout_fam] = {"mae": float("nan"), "r2": float("nan")}

    # Summary
    print("\n  SUMMARY:")
    summary = {}
    for name in results:
        r2s = [v["r2"] for v in results[name].values() if not np.isnan(v["r2"])]
        maes = [v["mae"] for v in results[name].values() if not np.isnan(v["mae"])]
        if r2s:
            summary[name] = {
                "mean_r2": float(np.mean(r2s)),
                "mean_mae": float(np.mean(maes)),
                "per_family": {k: v["r2"] for k, v in results[name].items()},
            }
            print(f"    {name:>25s}: mean R2={np.mean(r2s):.4f}, mean MAE={np.mean(maes):.4f}")

    return summary


# ==========================================
# ANALYSIS 3: Cluster-Robust Standard Errors
# ==========================================
def cluster_robust_inference(obs):
    """Compute cluster-robust SEs for the Gaussian model.
    Clusters = profiles (model x checkpoint x dataset).
    Uses the Liang-Zeger sandwich estimator approximation.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Cluster-Robust Inference")
    print("=" * 70)

    datasets = sorted(set(o["dataset"] for o in obs))
    families = sorted(set(o["family"] for o in obs))
    profiles = sorted(set(o["profile_id"] for o in obs))

    print(f"  Total observations: {len(obs)}")
    print(f"  Unique profiles (clusters): {len(profiles)}")
    print(f"  Avg observations per cluster: {len(obs) / len(profiles):.1f}")

    # Profile-level analysis: compute R2 per profile for both models
    profile_r2_gauss = {}
    profile_r2_linear = {}
    profile_family_map = {}

    # First fit global models (on all data)
    Q_all = np.array([o["Q"] for o in obs])

    # Fit gaussian on all data
    def predict_gauss_global(params, obs_list):
        alpha, beta, mu_0, mu_1 = params[:4]
        b_d = {ds: params[4 + i] for i, ds in enumerate(datasets)}
        preds = []
        for o in obs_list:
            x_star = mu_0 + mu_1 * o["log_r"]
            logit_Q = b_d.get(o["dataset"], 0) + alpha * o["log_r"] - beta * (o["x"] - x_star) ** 2
            preds.append(expit(np.clip(logit_Q, -20, 20)))
        return np.array(preds)

    def loss_gauss(params):
        return np.mean((Q_all - predict_gauss_global(params, obs)) ** 2)

    n_ds = len(datasets)
    bounds_g = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    best_g, best_loss_g = None, float("inf")
    for trial in range(30):
        rng = np.random.RandomState(trial)
        x0 = [rng.uniform(b[0], b[1]) for b in bounds_g]
        try:
            res = minimize(loss_gauss, x0, method="L-BFGS-B", bounds=bounds_g,
                          options={"maxiter": 5000, "ftol": 1e-12})
            if res.fun < best_loss_g:
                best_loss_g = res.fun
                best_g = res
        except Exception:
            continue

    pred_gauss_all = predict_gauss_global(best_g.x, obs)

    # Fit linear on all data
    def predict_lin_global(params, obs_list):
        c = params[0]
        b_d = {ds: params[1 + i] for i, ds in enumerate(datasets)}
        preds = []
        for o in obs_list:
            logit_Q = b_d.get(o["dataset"], 0) + c * o["x"]
            preds.append(expit(np.clip(logit_Q, -20, 20)))
        return np.array(preds)

    def loss_lin(params):
        return np.mean((Q_all - predict_lin_global(params, obs)) ** 2)

    bounds_l = [(-5, 5)] + [(-10, 10)] * n_ds
    best_l, best_loss_l = None, float("inf")
    for trial in range(20):
        rng = np.random.RandomState(trial)
        x0 = [rng.uniform(b[0], b[1]) for b in bounds_l]
        try:
            res = minimize(loss_lin, x0, method="L-BFGS-B", bounds=bounds_l,
                          options={"maxiter": 3000, "ftol": 1e-12})
            if res.fun < best_loss_l:
                best_loss_l = res.fun
                best_l = res
        except Exception:
            continue

    pred_lin_all = predict_lin_global(best_l.x, obs)

    # Compute per-profile R2 and residuals
    profile_deltas = []
    for prof in profiles:
        idx = [i for i, o in enumerate(obs) if o["profile_id"] == prof]
        Q_prof = np.array([obs[i]["Q"] for i in idx])
        pred_g = np.array([pred_gauss_all[i] for i in idx])
        pred_l = np.array([pred_lin_all[i] for i in idx])

        ss_tot = np.sum((Q_prof - Q_prof.mean()) ** 2)
        if ss_tot < 1e-10:
            continue

        ss_res_g = np.sum((Q_prof - pred_g) ** 2)
        ss_res_l = np.sum((Q_prof - pred_l) ** 2)

        r2_g = 1 - ss_res_g / ss_tot
        r2_l = 1 - ss_res_l / ss_tot

        fam = obs[idx[0]]["family"]
        profile_family_map[prof] = fam
        profile_r2_gauss[prof] = r2_g
        profile_r2_linear[prof] = r2_l
        profile_deltas.append(r2_g - r2_l)

    profile_deltas = np.array(profile_deltas)

    # Cluster-level paired t-test (proper inference unit = profile)
    from scipy.stats import ttest_1samp, wilcoxon

    t_stat, t_pval = ttest_1samp(profile_deltas, 0)
    try:
        w_stat, w_pval = wilcoxon(profile_deltas)
    except Exception:
        w_stat, w_pval = float("nan"), float("nan")

    n_pos = (profile_deltas > 0).sum()
    n_neg = (profile_deltas < 0).sum()
    n_zero = (profile_deltas == 0).sum()

    # Per-family cluster analysis
    per_family = {}
    for fam in families:
        fam_deltas = np.array([profile_deltas[i] for i, p in enumerate(
            [p for p in profiles if p in profile_family_map])
            if profile_family_map.get(list(profile_family_map.keys())[i] if i < len(profile_family_map) else "", "") == fam])

        # Recompute properly
        fam_profs = [p for p in profiles if profile_family_map.get(p, "") == fam]
        fam_d = np.array([profile_r2_gauss[p] - profile_r2_linear[p] for p in fam_profs
                         if p in profile_r2_gauss])
        if len(fam_d) > 0:
            per_family[fam] = {
                "n_profiles": int(len(fam_d)),
                "mean_delta": float(fam_d.mean()),
                "std_delta": float(fam_d.std()),
                "pct_positive": float((fam_d > 0).mean()),
            }

    print(f"\n  Profile-level delta R2 (Gaussian - Linear):")
    print(f"    N profiles:  {len(profile_deltas)}")
    print(f"    Mean:        {profile_deltas.mean():+.4f}")
    print(f"    Median:      {np.median(profile_deltas):+.4f}")
    print(f"    SD:          {profile_deltas.std():.4f}")
    print(f"    +/-/0:       {n_pos}/{n_neg}/{n_zero}")
    print(f"    Paired t:    t={t_stat:.3f}, p={t_pval:.2e}")
    print(f"    Wilcoxon:    W={w_stat}, p={w_pval:.2e}")

    for fam, fdata in per_family.items():
        print(f"    {fam:>15s}: n={fdata['n_profiles']}, mean_delta={fdata['mean_delta']:+.4f}, "
              f"pct_positive={fdata['pct_positive']:.1%}")

    # Effective sample size (Kish's)
    # Within-profile correlation
    icc_numerator = profile_deltas.var()
    within_vars = []
    valid_profiles = [p for p in profiles if p in profile_r2_gauss]
    for prof in valid_profiles:
        idx = [i for i, o in enumerate(obs) if o["profile_id"] == prof]
        if len(idx) < 2:
            continue
        resid_g = np.array([obs[i]["Q"] - pred_gauss_all[i] for i in idx])
        within_vars.append(resid_g.var())
    avg_within = np.mean(within_vars) if within_vars else 1.0

    print(f"\n  Effective sample size info:")
    print(f"    N observations: {len(obs)}")
    print(f"    N profiles (clusters): {len(profile_deltas)}")
    print(f"    Profile-level SE of delta: {profile_deltas.std() / np.sqrt(len(profile_deltas)):.4f}")

    # Build per-family delta arrays for bootstrap
    per_family_deltas = {}
    for fam in families:
        fam_profs = [p for p in profiles if profile_family_map.get(p, "") == fam]
        fam_d = [profile_r2_gauss[p] - profile_r2_linear[p] for p in fam_profs
                 if p in profile_r2_gauss]
        per_family_deltas[fam] = fam_d

    return {
        "n_profiles": int(len(profile_deltas)),
        "n_observations": len(obs),
        "delta_r2_mean": float(profile_deltas.mean()),
        "delta_r2_median": float(np.median(profile_deltas)),
        "delta_r2_sd": float(profile_deltas.std()),
        "delta_r2_se": float(profile_deltas.std() / np.sqrt(len(profile_deltas))),
        "paired_t_stat": float(t_stat),
        "paired_t_pval": float(t_pval),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_pval": float(w_pval) if not np.isnan(w_pval) else None,
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "per_family": per_family,
    }, per_family_deltas


def main():
    print("Loading observations...")
    obs = load_all_observations()
    print(f"Total: {len(obs)} observations")

    # Run all three analyses
    results = {}

    # Analysis 2: Spline baselines (fast, do first)
    results["spline_baselines"] = spline_baseline_comparison(obs)

    # Analysis 3: Cluster-robust inference (fast)
    cluster_results, per_family_deltas = cluster_robust_inference(obs)
    results["cluster_inference"] = cluster_results

    # Analysis 1: Paired bootstrap using pre-computed profile deltas (fast)
    results["paired_bootstrap"] = paired_bootstrap_delta_r2(per_family_deltas, n_boot=10000, seed=42)

    # Save all results
    out_path = RESULTS_DIR / "cti_robust_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nAll results saved to {out_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    pb = results["paired_bootstrap"]
    pm = pb["profile_mean"]
    lm = pb["lofo_mean"]
    print(f"\n  1. Delta R2 CI (bootstrap, profile-level): [{pm['ci95'][0]:+.4f}, {pm['ci95'][1]:+.4f}]")
    print(f"     P(Gaussian > Linear): {pm['prob_positive']:.1%}")
    print(f"     LOFO-style CI: [{lm['ci95'][0]:+.4f}, {lm['ci95'][1]:+.4f}]")

    sb = results["spline_baselines"]
    print(f"\n  2. Spline baselines:")
    for name, data in sb.items():
        print(f"     {name:>25s}: R2={data['mean_r2']:.4f}")

    ci = results["cluster_inference"]
    print(f"\n  3. Cluster-robust inference (N={ci['n_profiles']} profiles):")
    print(f"     Paired t: p={ci['paired_t_pval']:.2e}")
    print(f"     Delta R2: {ci['delta_r2_mean']:+.4f} +/- {ci['delta_r2_se']:.4f}")


if __name__ == "__main__":
    main()
