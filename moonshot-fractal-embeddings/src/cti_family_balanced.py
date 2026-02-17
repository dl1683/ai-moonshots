#!/usr/bin/env python
"""CTI Family-Balanced Analysis (vectorized).

Addresses Codex concern: Pythia dominates (3520/5004 = 70% of observations).
Tests whether results hold when:
  1. Families are weighted equally (inverse-frequency weighting)
  2. Subsample to minimum family size
  3. Report family-level mean +/- SE for R2
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


def fit_gaussian_vectorized(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                            x_te, log_r_te, ds_idx_te, n_ds,
                            weights=None):
    """Full Gaussian model, vectorized, with optional weights."""
    if weights is None:
        weights = np.ones(len(x_tr))

    def predict(params, x, log_r, ds_idx):
        alpha, beta, mu_0, mu_1 = params[:4]
        b = np.array(params[4:4+n_ds])
        dev = x - mu_0 - mu_1 * log_r
        logit_val = b[ds_idx] + alpha * log_r - beta * dev**2
        return expit(np.clip(logit_val, -20, 20))

    def loss(params):
        residuals = Q_tr - predict(params, x_tr, log_r_tr, ds_idx_tr)
        return np.mean(weights * residuals**2)

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
    return predict(best.x, x_te, log_r_te, ds_idx_te) if best else np.full(len(x_te), 0.5)


def run_lofo_standard(obs, datasets):
    """Standard (unbalanced) LOFO."""
    families = sorted(set(o["family"] for o in obs))
    n_ds = len(datasets)
    results = {}

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]
        x_tr, Q_tr, log_r_tr, ds_idx_tr = vectorize_obs(train, datasets)
        x_te, Q_te, log_r_te, ds_idx_te = vectorize_obs(test, datasets)

        pred = fit_gaussian_vectorized(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                                        x_te, log_r_te, ds_idx_te, n_ds)
        mae, r2 = eval_fit(Q_te, pred)
        results[holdout_fam] = {"r2": r2, "mae": mae, "n_test": len(test)}

    return results


def run_lofo_weighted(obs, datasets):
    """Inverse-frequency weighted LOFO."""
    families = sorted(set(o["family"] for o in obs))
    n_ds = len(datasets)
    results = {}

    for holdout_fam in families:
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]
        x_tr, Q_tr, log_r_tr, ds_idx_tr = vectorize_obs(train, datasets)
        x_te, Q_te, log_r_te, ds_idx_te = vectorize_obs(test, datasets)

        # Compute weights
        train_fam_counts = {}
        for o in train:
            train_fam_counts[o["family"]] = train_fam_counts.get(o["family"], 0) + 1
        n_train_families = len(train_fam_counts)
        weights = np.array([
            1.0 / (n_train_families * train_fam_counts[o["family"]])
            for o in train
        ])
        weights = weights * len(train) / weights.sum()

        pred = fit_gaussian_vectorized(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                                        x_te, log_r_te, ds_idx_te, n_ds,
                                        weights=weights)
        mae, r2 = eval_fit(Q_te, pred)
        results[holdout_fam] = {"r2": r2, "mae": mae, "n_test": len(test)}

    return results


def run_lofo_subsampled(obs, datasets, n_boot=5, seed=42):
    """Subsample each training family to min size, repeat n_boot times."""
    families = sorted(set(o["family"] for o in obs))
    n_ds = len(datasets)

    fam_obs = {}
    for o in obs:
        fam_obs.setdefault(o["family"], []).append(o)
    min_size = min(len(v) for v in fam_obs.values())

    all_results = {fam: [] for fam in families}
    rng = np.random.RandomState(seed)

    for boot in range(n_boot):
        print(f"    Bootstrap {boot+1}/{n_boot}...", flush=True)
        for holdout_fam in families:
            train = []
            for fam in families:
                if fam == holdout_fam:
                    continue
                fam_data = fam_obs[fam]
                if len(fam_data) > min_size:
                    idx = rng.choice(len(fam_data), min_size, replace=False)
                    train.extend([fam_data[i] for i in idx])
                else:
                    train.extend(fam_data)

            test = fam_obs[holdout_fam]
            x_tr, Q_tr, log_r_tr, ds_idx_tr = vectorize_obs(train, datasets)
            x_te, Q_te, log_r_te, ds_idx_te = vectorize_obs(test, datasets)

            pred = fit_gaussian_vectorized(x_tr, Q_tr, log_r_tr, ds_idx_tr,
                                            x_te, log_r_te, ds_idx_te, n_ds)
            mae, r2 = eval_fit(Q_te, pred)
            all_results[holdout_fam].append({"r2": r2, "mae": mae})

    results = {}
    for fam in families:
        r2s = [r["r2"] for r in all_results[fam]]
        maes = [r["mae"] for r in all_results[fam]]
        results[fam] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_se": float(np.std(r2s) / np.sqrt(len(r2s))),
            "mae_mean": float(np.mean(maes)),
            "mae_se": float(np.std(maes) / np.sqrt(len(maes))),
            "n_boot": n_boot,
        }

    return results, min_size


def main():
    print("Loading observations...")
    obs = load_all_observations()
    datasets = sorted(set(o["dataset"] for o in obs))
    print(f"  Total: {len(obs)} obs")

    fam_counts = {}
    for o in obs:
        fam_counts[o["family"]] = fam_counts.get(o["family"], 0) + 1
    print("\n  Family distribution:")
    for fam in sorted(fam_counts.keys()):
        pct = 100.0 * fam_counts[fam] / len(obs)
        print(f"    {fam:>15s}: {fam_counts[fam]:5d} ({pct:.1f}%)")

    output = {"family_counts": fam_counts}

    # 1. Standard LOFO
    print("\n" + "=" * 70)
    print("  1. STANDARD LOFO (unbalanced)")
    print("=" * 70)

    standard = run_lofo_standard(obs, datasets)
    r2s = [v["r2"] for v in standard.values()]
    mean_r2 = np.mean(r2s)
    se_r2 = np.std(r2s) / np.sqrt(len(r2s))

    for fam, v in sorted(standard.items()):
        print(f"    {fam:>15s}: R2={v['r2']:.4f}, MAE={v['mae']:.4f} (n={v['n_test']})")
    print(f"\n    Mean R2 = {mean_r2:.4f} +/- {se_r2:.4f}")

    output["standard_lofo"] = {
        "per_family": standard,
        "mean_r2": float(mean_r2),
        "se_r2": float(se_r2),
    }

    # 2. Weighted LOFO
    print("\n" + "=" * 70)
    print("  2. WEIGHTED LOFO (inverse-frequency)")
    print("=" * 70)

    weighted = run_lofo_weighted(obs, datasets)
    r2s_w = [v["r2"] for v in weighted.values()]
    mean_r2_w = np.mean(r2s_w)
    se_r2_w = np.std(r2s_w) / np.sqrt(len(r2s_w))

    for fam, v in sorted(weighted.items()):
        print(f"    {fam:>15s}: R2={v['r2']:.4f}, MAE={v['mae']:.4f}")
    print(f"\n    Mean R2 = {mean_r2_w:.4f} +/- {se_r2_w:.4f}")

    output["weighted_lofo"] = {
        "per_family": weighted,
        "mean_r2": float(mean_r2_w),
        "se_r2": float(se_r2_w),
    }

    # 3. Subsampled LOFO
    print("\n" + "=" * 70)
    print("  3. SUBSAMPLED LOFO (min family size)")
    print("=" * 70)

    subsampled, min_size = run_lofo_subsampled(obs, datasets, n_boot=5)
    print(f"    Subsampled to {min_size} obs per family")

    r2_means = [v["r2_mean"] for v in subsampled.values()]
    mean_r2_s = np.mean(r2_means)

    for fam, v in sorted(subsampled.items()):
        print(f"    {fam:>15s}: R2={v['r2_mean']:.4f} +/- {v['r2_se']:.4f}")
    print(f"\n    Mean R2 = {mean_r2_s:.4f}")

    output["subsampled_lofo"] = {
        "per_family": subsampled,
        "mean_r2": float(mean_r2_s),
        "min_family_size": min_size,
    }

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"    Standard LOFO:    R2 = {mean_r2:.4f} +/- {se_r2:.4f}")
    print(f"    Weighted LOFO:    R2 = {mean_r2_w:.4f} +/- {se_r2_w:.4f}")
    print(f"    Subsampled LOFO:  R2 = {mean_r2_s:.4f}")
    max_delta = max(abs(mean_r2_w - mean_r2), abs(mean_r2_s - mean_r2))
    print(f"\n    Max delta from standard: {max_delta:.4f}")

    output["summary"] = {
        "standard_r2": float(mean_r2),
        "weighted_r2": float(mean_r2_w),
        "subsampled_r2": float(mean_r2_s),
        "max_delta": float(max_delta),
        "conclusion": "robust" if max_delta < 0.05 else "sensitive"
    }

    out_path = RESULTS_DIR / "cti_family_balanced.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
