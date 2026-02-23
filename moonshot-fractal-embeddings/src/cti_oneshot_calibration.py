"""
One-shot calibration demo: practical utility of the CTI universal law.

Given ONE reference architecture's (kappa, q) on a dataset, predict q for all other
architectures using the FROZEN universal alpha=3.598.

Formula:
  C_cal = logit(q_norm_ref) - alpha_universal * kappa_ref
  logit(q_norm_pred) = alpha_universal * kappa_new + C_cal

Pre-registered:
  - For each dataset, iterate over all reference architectures
  - Measure Pearson r(q_pred, q_obs) and MAE on the remaining architectures
  - Criterion: mean r >= 0.80 across datasets (LOAO average over reference architectures)

This directly tests: "the universal alpha lets you predict all architecture performances
from a SINGLE reference measurement"

Output: results/cti_oneshot_calibration.json
"""

import json
import os
import numpy as np
from scipy.stats import pearsonr
from scipy.special import expit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_oneshot_calibration.json")

ALPHA_UNIVERSAL = 3.598   # from comprehensive universality fit
BETA_UNIVERSAL  = 0.478   # sparse competition: log(K-1) coefficient (positive sign)

MIN_ARCHS_PER_DATASET = 5
PEARSON_THRESHOLD = 0.80


def logit(q):
    return np.log(q / (1 - q))


def load_cache():
    pts = []
    for fname in os.listdir(RESULTS_DIR):
        if not (fname.startswith("kappa_near_cache_") and fname.endswith(".json")):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for entry in data:
            q = entry.get("q")
            kappa = entry.get("kappa_nearest")
            K = entry.get("K")
            model = entry.get("model", "")
            dataset = entry.get("dataset", "")
            if q is None or kappa is None or K is None:
                continue
            if q <= 0 or q >= 1.0 or kappa <= 0:
                continue
            q_norm = (q - 1.0 / K) / (1.0 - 1.0 / K)
            if q_norm <= 0 or q_norm >= 1.0:
                continue
            pts.append({
                "model": model,
                "dataset": dataset,
                "K": int(K),
                "q": float(q),
                "q_norm": float(q_norm),
                "kappa": float(kappa),
            })
    return pts


def mean_per_model_dataset(pts):
    """Average kappa and q across layers per (model, dataset)."""
    groups = {}
    for p in pts:
        key = (p["model"], p["dataset"])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)
    result = []
    for (model, dataset), plist in groups.items():
        K = plist[0]["K"]
        q_vals = [p["q"] for p in plist]
        q_norm_vals = [p["q_norm"] for p in plist]
        kappa_vals = [p["kappa"] for p in plist]
        result.append({
            "model": model,
            "dataset": dataset,
            "K": K,
            "q": float(np.mean(q_vals)),
            "q_norm": float(np.mean(q_norm_vals)),
            "kappa": float(np.mean(kappa_vals)),
        })
    return result


def oneshot_cv(plist, use_beta=True):
    """
    Leave-one-out over reference architectures.
    For each ref architecture, fit C_cal from (kappa_ref, q_norm_ref),
    then predict all others.
    Returns mean Pearson r and MAE over all (ref, other) pairs.
    """
    K = plist[0]["K"]
    log_km1 = np.log(K - 1)
    n = len(plist)
    r_vals = []
    mae_vals = []

    for i_ref in range(n):
        ref = plist[i_ref]
        others = [plist[j] for j in range(n) if j != i_ref]
        if len(others) < 2:
            continue

        # Calibrate C from reference
        logit_q_ref = float(logit(ref["q_norm"]))
        if use_beta:
            C_cal = logit_q_ref - ALPHA_UNIVERSAL * ref["kappa"] + BETA_UNIVERSAL * log_km1
        else:
            C_cal = logit_q_ref - ALPHA_UNIVERSAL * ref["kappa"]

        # Predict others
        q_pred = []
        q_obs = []
        for o in others:
            if use_beta:
                logit_pred = ALPHA_UNIVERSAL * o["kappa"] - BETA_UNIVERSAL * log_km1 + C_cal
            else:
                logit_pred = ALPHA_UNIVERSAL * o["kappa"] + C_cal
            q_pred_val = float(expit(logit_pred))
            q_pred.append(q_pred_val)
            q_obs.append(o["q_norm"])

        q_pred = np.array(q_pred)
        q_obs = np.array(q_obs)

        if len(q_pred) >= 2 and np.std(q_pred) > 1e-10 and np.std(q_obs) > 1e-10:
            r, _ = pearsonr(q_pred, q_obs)
            r_vals.append(r)

        mae = float(np.mean(np.abs(q_pred - q_obs)))
        mae_vals.append(mae)

    return np.mean(r_vals) if r_vals else 0.0, np.mean(mae_vals) if mae_vals else 1.0


def main():
    print("Loading cache points...")
    all_pts = load_cache()
    pts = mean_per_model_dataset(all_pts)
    print(f"Loaded {len(all_pts)} raw -> {len(pts)} model-dataset pairs")

    by_dataset = {}
    for p in pts:
        d = p["dataset"]
        if d not in by_dataset:
            by_dataset[d] = []
        by_dataset[d].append(p)

    results = {}
    r_vals_all = []

    print(f"\nOne-shot calibration (alpha={ALPHA_UNIVERSAL}, beta={BETA_UNIVERSAL} fixed):")
    print(f"{'Dataset':>20} {'K':>4} {'N_arch':>7} {'LOAO_r':>8} {'LOAO_MAE':>10} {'PR':>6}")

    for dataset, plist in sorted(by_dataset.items()):
        if len(plist) < MIN_ARCHS_PER_DATASET:
            continue
        K = plist[0]["K"]

        mean_r, mean_mae = oneshot_cv(plist, use_beta=True)
        pr = mean_r >= PEARSON_THRESHOLD

        print(f"{dataset:>20} {K:>4} {len(plist):>7} {mean_r:>8.4f} {mean_mae:>10.4f} {'PASS' if pr else 'FAIL':>6}")
        results[dataset] = {
            "K": K,
            "n_architectures": len(plist),
            "loao_pearson_r": float(mean_r),
            "loao_mae": float(mean_mae),
            "pr_pass": bool(pr),
        }
        r_vals_all.append(mean_r)

    mean_r_global = np.mean(r_vals_all) if r_vals_all else 0.0
    n_pass = sum(1 for d in results.values() if d["pr_pass"])
    n_total = len(results)

    print(f"\nGlobal summary:")
    print(f"  Mean LOAO Pearson r: {mean_r_global:.4f}")
    print(f"  Datasets passing r >= {PEARSON_THRESHOLD}: {n_pass}/{n_total}")
    print(f"  Pre-registered: mean r >= {PEARSON_THRESHOLD} -> "
          f"{'PASS' if mean_r_global >= PEARSON_THRESHOLD else 'FAIL'}")

    output = {
        "experiment": "oneshot_calibration",
        "alpha_universal": ALPHA_UNIVERSAL,
        "beta_universal": BETA_UNIVERSAL,
        "pre_registered_threshold": PEARSON_THRESHOLD,
        "mean_loao_r": float(mean_r_global),
        "n_datasets": n_total,
        "n_pass": n_pass,
        "pr_pass": bool(mean_r_global >= PEARSON_THRESHOLD),
        "per_dataset": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
