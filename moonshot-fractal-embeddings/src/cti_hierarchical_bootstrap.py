#!/usr/bin/env python
"""CTI Hierarchical Bootstrap: family-level block bootstrap for proper inference.

Addresses Codex's key weakness: profile-level independence assumption is too weak.
This implements a block bootstrap at the family level (the LOFO holdout unit),
properly respecting the family/model-checkpoint/profile hierarchy.

Protocol:
1. Pre-compute per-family LOFO statistics (R2, shape_rho, delta R2)
2. Block-bootstrap: resample families with replacement
3. Compute bootstrap distribution of mean statistics
4. Report family-level CIs (more conservative than profile-level)

The effective sample size is the number of families (5 for primary, 8 for extended),
not the number of profiles (252 or 268). This is the honest inference unit.
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_all_observations(include_new=False):
    sources = [
        RESULTS_DIR / "cti_checkpoint_sweep_all.json",
        RESULTS_DIR / "cti_multi_family.json",
        RESULTS_DIR / "cti_olmo2_sweep.json",
    ]
    if include_new:
        sources.append(RESULTS_DIR / "cti_new_families.json")
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


def fit_model(x_tr, Q_tr, log_r_tr, ds_idx_tr, n_ds, model_type="gaussian"):
    """Fit Gaussian or Linear model."""
    if model_type == "gaussian":
        def predict(params, x, log_r, ds_idx):
            alpha, beta, mu_0, mu_1 = params[:4]
            b = np.array(params[4:4+n_ds])
            dev = x - mu_0 - mu_1 * log_r
            logit_val = b[ds_idx] + alpha * log_r - beta * dev**2
            return expit(np.clip(logit_val, -20, 20))
        bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
    else:
        def predict(params, x, log_r, ds_idx):
            c1 = params[0]
            b = np.array(params[1:1+n_ds])
            logit_val = b[ds_idx] + c1 * x
            return expit(np.clip(logit_val, -20, 20))
        bounds = [(-10, 10)] + [(-10, 10)] * n_ds

    def loss(params):
        return np.mean((Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)) ** 2)

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
    return best.x if best else None, predict


def eval_r2(Q_obs, Q_pred):
    ss_res = np.sum((Q_obs - Q_pred) ** 2)
    ss_tot = np.sum((Q_obs - Q_obs.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def compute_lofo_stats(obs, families, datasets):
    """Pre-compute per-family LOFO statistics."""
    n_ds = len(datasets)
    family_stats = {}

    for holdout in families:
        train = [o for o in obs if o["family"] != holdout]
        test = [o for o in obs if o["family"] == holdout]
        if not test or not train:
            continue

        x_tr, Q_tr, lr_tr, di_tr = vectorize_obs(train, datasets)
        x_te, Q_te, lr_te, di_te = vectorize_obs(test, datasets)

        # Gaussian
        gp, gpred = fit_model(x_tr, Q_tr, lr_tr, di_tr, n_ds, "gaussian")
        if gp is None:
            continue
        pred_g = gpred(gp, x_te, lr_te, di_te)
        r2_g = eval_r2(Q_te, pred_g)

        # Linear
        lp, lpred = fit_model(x_tr, Q_tr, lr_tr, di_tr, n_ds, "linear")
        if lp is None:
            continue
        pred_l = lpred(lp, x_te, lr_te, di_te)
        r2_l = eval_r2(Q_te, pred_l)

        # Shape correlation
        profiles = {}
        for i, o in enumerate(test):
            key = (o["model"], o["step"], o["dataset"])
            if key not in profiles:
                profiles[key] = {"obs": [], "pred": []}
            profiles[key]["obs"].append(o["Q"])
            profiles[key]["pred"].append(pred_g[i])

        rhos = []
        for p in profiles.values():
            if len(p["obs"]) >= 4:
                rho, _ = spearmanr(p["obs"], p["pred"])
                if not np.isnan(rho):
                    rhos.append(rho)

        family_stats[holdout] = {
            "r2_gaussian": r2_g,
            "r2_linear": r2_l,
            "delta_r2": r2_g - r2_l,
            "shape_rho": np.mean(rhos) if rhos else float("nan"),
            "n_profiles": len(rhos),
            "n_obs": len(test),
        }

    return family_stats


def block_bootstrap(family_stats, n_boot=10000, seed=42):
    """Block bootstrap at the family level."""
    families = list(family_stats.keys())
    n_fam = len(families)
    rng = np.random.RandomState(seed)

    # Pre-extract arrays
    r2s = np.array([family_stats[f]["r2_gaussian"] for f in families])
    deltas = np.array([family_stats[f]["delta_r2"] for f in families])
    rhos = np.array([family_stats[f]["shape_rho"] for f in families])

    boot_r2 = np.empty(n_boot)
    boot_delta = np.empty(n_boot)
    boot_rho = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.randint(0, n_fam, size=n_fam)
        boot_r2[i] = np.mean(r2s[idx])
        boot_delta[i] = np.mean(deltas[idx])
        valid_rhos = rhos[idx]
        valid_mask = ~np.isnan(valid_rhos)
        boot_rho[i] = np.mean(valid_rhos[valid_mask]) if valid_mask.any() else np.nan

    return boot_r2, boot_delta, boot_rho


def main():
    print("CTI Hierarchical Bootstrap")
    print("=" * 60)

    # ==================
    # 5-family primary
    # ==================
    print("\n--- 5-Family Primary ---")
    obs5 = load_all_observations(include_new=False)
    datasets = sorted(set(o["dataset"] for o in obs5))
    families5 = sorted(set(o["family"] for o in obs5))
    print(f"  {len(obs5)} obs, {len(families5)} families")

    print("  Computing LOFO statistics (Gaussian + Linear)...")
    stats5 = compute_lofo_stats(obs5, families5, datasets)
    for fam, s in stats5.items():
        print(f"    {fam:>15s}: R2_g={s['r2_gaussian']:.4f}, R2_l={s['r2_linear']:.4f}, "
              f"delta={s['delta_r2']:.4f}, rho={s['shape_rho']:.3f}")

    print(f"\n  Running 10000 family-level block bootstrap...")
    boot_r2_5, boot_delta_5, boot_rho_5 = block_bootstrap(stats5, n_boot=10000)

    r2_mean_5 = np.mean([s["r2_gaussian"] for s in stats5.values()])
    delta_mean_5 = np.mean([s["delta_r2"] for s in stats5.values()])

    print(f"\n  RESULTS (5 families, N_eff=5):")
    print(f"    R2:       {r2_mean_5:.4f}  95% CI [{np.percentile(boot_r2_5, 2.5):.4f}, {np.percentile(boot_r2_5, 97.5):.4f}]")
    print(f"    Delta R2: {delta_mean_5:.4f}  95% CI [{np.percentile(boot_delta_5, 2.5):.4f}, {np.percentile(boot_delta_5, 97.5):.4f}]")
    print(f"    Shape rho: {np.mean([s['shape_rho'] for s in stats5.values()]):.3f}  "
          f"95% CI [{np.percentile(boot_rho_5, 2.5):.3f}, {np.percentile(boot_rho_5, 97.5):.3f}]")
    print(f"    P(delta <= 0): {np.mean(boot_delta_5 <= 0):.4f}")
    print(f"    Delta CI excludes zero: {np.percentile(boot_delta_5, 2.5) > 0}")

    # ==================
    # 8-family extended
    # ==================
    print("\n--- 8-Family Extended ---")
    obs8 = load_all_observations(include_new=True)
    families8 = sorted(set(o["family"] for o in obs8))
    print(f"  {len(obs8)} obs, {len(families8)} families")

    print("  Computing LOFO statistics...")
    stats8 = compute_lofo_stats(obs8, families8, datasets)
    for fam, s in stats8.items():
        print(f"    {fam:>15s}: R2_g={s['r2_gaussian']:.4f}, R2_l={s['r2_linear']:.4f}, "
              f"delta={s['delta_r2']:.4f}, rho={s['shape_rho']:.3f}")

    print(f"\n  Running 10000 family-level block bootstrap...")
    boot_r2_8, boot_delta_8, boot_rho_8 = block_bootstrap(stats8, n_boot=10000)

    r2_mean_8 = np.mean([s["r2_gaussian"] for s in stats8.values()])
    delta_mean_8 = np.mean([s["delta_r2"] for s in stats8.values()])
    rho_mean_8 = np.nanmean([s["shape_rho"] for s in stats8.values()])

    print(f"\n  RESULTS (8 families, N_eff=8):")
    print(f"    R2:       {r2_mean_8:.4f}  95% CI [{np.percentile(boot_r2_8, 2.5):.4f}, {np.percentile(boot_r2_8, 97.5):.4f}]")
    print(f"    Delta R2: {delta_mean_8:.4f}  95% CI [{np.percentile(boot_delta_8, 2.5):.4f}, {np.percentile(boot_delta_8, 97.5):.4f}]")
    print(f"    Shape rho: {rho_mean_8:.3f}  95% CI [{np.nanpercentile(boot_rho_8, 2.5):.3f}, {np.nanpercentile(boot_rho_8, 97.5):.3f}]")
    print(f"    P(delta <= 0): {np.mean(boot_delta_8 <= 0):.4f}")
    print(f"    Delta CI excludes zero: {np.percentile(boot_delta_8, 2.5) > 0}")

    # Save
    output = {
        "primary_5_family": {
            "n_families": len(families5),
            "n_observations": len(obs5),
            "n_bootstrap": 10000,
            "per_family": {f: stats5[f] for f in families5},
            "mean_r2": float(r2_mean_5),
            "r2_ci_95": [float(np.percentile(boot_r2_5, 2.5)), float(np.percentile(boot_r2_5, 97.5))],
            "mean_delta_r2": float(delta_mean_5),
            "delta_ci_95": [float(np.percentile(boot_delta_5, 2.5)), float(np.percentile(boot_delta_5, 97.5))],
            "delta_p_value": float(np.mean(boot_delta_5 <= 0)),
            "delta_excludes_zero": bool(np.percentile(boot_delta_5, 2.5) > 0),
            "mean_shape_rho": float(np.mean([s["shape_rho"] for s in stats5.values()])),
            "rho_ci_95": [float(np.percentile(boot_rho_5, 2.5)), float(np.percentile(boot_rho_5, 97.5))],
        },
        "extended_8_family": {
            "n_families": len(families8),
            "n_observations": len(obs8),
            "n_bootstrap": 10000,
            "per_family": {f: stats8[f] for f in families8 if f in stats8},
            "mean_r2": float(r2_mean_8),
            "r2_ci_95": [float(np.percentile(boot_r2_8, 2.5)), float(np.percentile(boot_r2_8, 97.5))],
            "mean_delta_r2": float(delta_mean_8),
            "delta_ci_95": [float(np.percentile(boot_delta_8, 2.5)), float(np.percentile(boot_delta_8, 97.5))],
            "delta_p_value": float(np.mean(boot_delta_8 <= 0)),
            "delta_excludes_zero": bool(np.percentile(boot_delta_8, 2.5) > 0),
            "mean_shape_rho": float(rho_mean_8),
            "rho_ci_95": [float(np.nanpercentile(boot_rho_8, 2.5)), float(np.nanpercentile(boot_rho_8, 97.5))],
        },
    }

    out_path = RESULTS_DIR / "cti_hierarchical_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
