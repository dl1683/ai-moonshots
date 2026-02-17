#!/usr/bin/env python
"""CTI Expanded LOFO: 8 families (original 5 + Gemma-2, Phi, Qwen2.5).

Merges all data sources and re-runs full LOFO cross-validation with 8 families.
This addresses Codex's #1 recommendation for 7/10: more diverse families.
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
    """Load all observations from all sources including new families."""
    sources = [
        RESULTS_DIR / "cti_checkpoint_sweep_all.json",
        RESULTS_DIR / "cti_multi_family.json",
        RESULTS_DIR / "cti_olmo2_sweep.json",
        RESULTS_DIR / "cti_new_families.json",  # NEW
    ]
    obs = []
    for src_path in sources:
        if not src_path.exists():
            print(f"  WARNING: {src_path.name} not found, skipping")
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


def fit_gaussian(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                 x_te, log_r_te, ds_idx_te, n_ds):
    """Full Gaussian model, vectorized."""
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

    if best is None:
        return np.full(len(x_te), 0.5), None

    return predict(best.x, x_te, log_r_te, ds_idx_te), best.x


def compute_shape_correlation(obs_list, pred_values):
    """Compute per-profile Spearman shape correlation."""
    # Group by profile (model, step, dataset)
    profiles = {}
    for i, o in enumerate(obs_list):
        key = (o["model"], o["step"], o["dataset"])
        if key not in profiles:
            profiles[key] = {"obs": [], "pred": [], "layers": []}
        profiles[key]["obs"].append(o["Q"])
        profiles[key]["pred"].append(pred_values[i])
        profiles[key]["layers"].append(o["layer"])

    rhos = []
    for key, p in profiles.items():
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
    print(f"  Total: {len(obs)} obs")

    # Family distribution
    fam_counts = {}
    for o in obs:
        fam_counts[o["family"]] = fam_counts.get(o["family"], 0) + 1
    print("\n  Family distribution:")
    for fam in sorted(fam_counts.keys()):
        pct = 100.0 * fam_counts[fam] / len(obs)
        print(f"    {fam:>15s}: {fam_counts[fam]:5d} ({pct:.1f}%)")

    families = sorted(fam_counts.keys())
    n_families = len(families)
    print(f"\n  Total families: {n_families}")

    # Count profiles
    profiles = set()
    for o in obs:
        profiles.add((o["model"], o.get("step", -1), o["dataset"]))
    print(f"  Total profiles: {len(profiles)}")

    # ==========================================
    # EXPANDED LOFO (8 families)
    # ==========================================
    print("\n" + "=" * 70)
    print(f"  LEAVE-ONE-FAMILY-OUT ({n_families} families)")
    print("=" * 70)

    lofo_results = {}
    all_params = {}

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]

        x_tr, Q_tr, log_r_tr, ds_idx_tr = vectorize_obs(train, datasets)
        x_te, Q_te, log_r_te, ds_idx_te = vectorize_obs(test, datasets)

        pred, params = fit_gaussian(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                                     x_te, log_r_te, ds_idx_te, n_ds)
        mae, r2 = eval_fit(Q_te, pred)

        # Shape correlation
        rhos = compute_shape_correlation(test, pred)
        mean_rho = float(np.mean(rhos)) if rhos else float("nan")
        n_profiles = len(rhos)

        lofo_results[holdout_fam] = {
            "r2": r2, "mae": mae,
            "n_obs": len(test), "n_profiles": n_profiles,
            "mean_shape_rho": mean_rho,
        }

        if params is not None:
            all_params[holdout_fam] = {
                "alpha": float(params[0]),
                "beta": float(params[1]),
                "mu_0": float(params[2]),
                "mu_1": float(params[3]),
            }

        print(f"    {holdout_fam:>15s}: R2={r2:.4f}, MAE={mae:.4f}, "
              f"rho={mean_rho:.3f} ({n_profiles} profiles, {len(test)} obs)")

    # Summary
    r2s = [v["r2"] for v in lofo_results.values()]
    maes = [v["mae"] for v in lofo_results.values()]
    rhos = [v["mean_shape_rho"] for v in lofo_results.values() if not np.isnan(v["mean_shape_rho"])]
    n_total_profiles = sum(v["n_profiles"] for v in lofo_results.values())

    mean_r2 = np.mean(r2s)
    se_r2 = np.std(r2s) / np.sqrt(len(r2s))
    mean_mae = np.mean(maes)
    mean_rho = np.mean(rhos) if rhos else float("nan")

    print(f"\n    {'MEAN':>15s}: R2={mean_r2:.4f} +/- {se_r2:.4f}, "
          f"MAE={mean_mae:.4f}, rho={mean_rho:.3f} ({n_total_profiles} profiles)")

    # Parameter stability
    print("\n" + "=" * 70)
    print("  PARAMETER STABILITY ACROSS FOLDS")
    print("=" * 70)

    for param_name in ["alpha", "beta", "mu_0", "mu_1"]:
        vals = [v[param_name] for v in all_params.values()]
        print(f"    {param_name:>5s}: mean={np.mean(vals):.4f}, "
              f"sd={np.std(vals):.4f}, range=[{min(vals):.4f}, {max(vals):.4f}]")

    # Comparison: original 5 vs expanded 8
    print("\n" + "=" * 70)
    print("  ORIGINAL 5 vs EXPANDED 8 COMPARISON")
    print("=" * 70)

    original_families = {"cerebras-gpt", "gpt2", "olmo2", "opt", "pythia"}
    new_families = set(families) - original_families

    orig_r2s = [v["r2"] for k, v in lofo_results.items() if k in original_families]
    new_r2s = [v["r2"] for k, v in lofo_results.items() if k in new_families]

    print(f"    Original 5 families: mean R2 = {np.mean(orig_r2s):.4f}")
    print(f"    New 3 families:      mean R2 = {np.mean(new_r2s):.4f}")
    print(f"    All 8 families:      mean R2 = {mean_r2:.4f}")

    # Save
    output = {
        "n_families": n_families,
        "n_observations": len(obs),
        "n_profiles": n_total_profiles,
        "families": families,
        "lofo_results": lofo_results,
        "parameter_stability": all_params,
        "summary": {
            "mean_r2": float(mean_r2),
            "se_r2": float(se_r2),
            "mean_mae": float(mean_mae),
            "mean_shape_rho": float(mean_rho),
            "original_5_mean_r2": float(np.mean(orig_r2s)),
            "new_3_mean_r2": float(np.mean(new_r2s)),
        }
    }

    out_path = RESULTS_DIR / "cti_expanded_lofo.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
