#!/usr/bin/env python -u
"""
K-SCALING PROSPECTIVE PREDICTION TEST (Session 43)
===================================================
Codex recommendation: fit A(K) on 3 datasets, predict A on held-out 4th dataset,
evaluate out-of-sample q prediction quality.

PRE-REGISTERED PROTOCOL:
  Law: logit(q) = A(K) * kappa_nearest + C_arch
  K-scaling: A(K) = a / log(K) + b
  Fit A(K) on 3 datasets, predict on 4th (LODO)
  C_arch: per-architecture intercept from training datasets

PRE-REGISTERED DATA:
  Datasets: agnews (K=4), dbpedia (K=14), 20newsgroups (K=20), go_emotions (K=28)
  Architectures: 19 models from kappa_near_cache files
  Layers: 4 per model (proportional depth)

PRE-REGISTERED CRITERIA:
  H1: LODO rho(q_pred, q_actual) >= 0.85 (mean across 4 folds)
  H2: LODO MAE(q_pred, q_actual) <= 0.08
  H3: A_predicted within 50% of A_actual for each dataset
  H4: Global cross-dataset pooled rho >= 0.75

FAIRNESS CONSTRAINTS:
  - A(K) fitted ONLY on training datasets (not test)
  - C_arch fitted ONLY on training datasets (not test)
  - Per-architecture intercept allowed (known from LOAO: A universal, C varies)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_JSON = RESULTS_DIR / "cti_k_scaling_prediction.json"

# Dataset K values
DATASETS = {
    "agnews":        4,
    "dbpedia":       14,
    "20newsgroups":  20,
    "go_emotions":   28,
}

# Pre-registered thresholds
LODO_RHO_THRESH = 0.85
LODO_MAE_THRESH = 0.08
A_RATIO_THRESH  = 0.50  # max |A_pred/A_actual - 1|
GLOBAL_RHO_THRESH = 0.75


def load_all_caches():
    """Load all kappa_near_cache JSON files into a unified list.
    Uses NORMALIZED q = (acc - 1/K) / (1 - 1/K) for cross-K comparability.
    This matches the LOAO convention (A_NLP=1.054 is for normalized q).
    """
    points = []
    for ds_name, K in DATASETS.items():
        pattern = f"kappa_near_cache_{ds_name}_*.json"
        files = sorted(RESULTS_DIR.glob(pattern))
        print(f"  {ds_name} (K={K}): {len(files)} files")
        for fpath in files:
            with open(fpath) as f:
                data = json.load(f)
            for pt in data:
                q_raw = float(pt["q"])
                kappa = float(pt["kappa_nearest"])
                # Normalize: q_norm = (acc - 1/K) / (1 - 1/K)
                q_norm = (q_raw - 1.0/K) / (1.0 - 1.0/K)
                q_norm = float(np.clip(q_norm, 1e-5, 1 - 1e-5))
                logit_q_norm = float(np.log(q_norm / (1.0 - q_norm)))
                if not np.isfinite(logit_q_norm) or not np.isfinite(kappa) or kappa <= 0:
                    continue
                if q_raw < 1.0/K:  # below chance: degenerate model, skip
                    continue
                points.append({
                    "dataset": ds_name,
                    "K": K,
                    "model": str(pt.get("model", fpath.stem.split("_")[-1])),
                    "layer": int(pt.get("layer", 0)),
                    "q_raw": q_raw,
                    "q": q_norm,  # normalized q
                    "kappa_nearest": kappa,
                    "logit_q": logit_q_norm,  # logit of normalized q
                })
    return points


def fit_slope_per_dataset(points, datasets_to_use):
    """
    Fit logit(q) = A * kappa + C_arch via CROSS-ARCHITECTURE OLS.
    This matches the LOAO approach: for each dataset, pool all architectures
    and regress logit_q vs kappa with per-architecture intercept (within-arch demeaning).

    Uses per-architecture demeaning to remove architecture-specific offsets,
    then regresses the WITHIN-architecture variation of kappa vs logit_q.

    Returns dict: dataset_name -> {A_pooled, std_A, n_points}
    """
    from collections import defaultdict
    result = {}
    for ds_name in datasets_to_use:
        ds_pts = [p for p in points if p["dataset"] == ds_name]
        if not ds_pts:
            continue
        by_model = defaultdict(list)
        for p in ds_pts:
            by_model[p["model"]].append(p)

        # Two estimates: (1) within-arch slope (across layers), (2) cross-arch slope (across models)
        # For K-scaling, we want (2): which architecture has higher q given higher kappa
        # Use simple pooled OLS across all 76 points with single intercept
        kappas_all = np.array([p["kappa_nearest"] for p in ds_pts])
        logits_all = np.array([p["logit_q"] for p in ds_pts])

        # Simple OLS (single intercept): matches LOAO's universal A concept
        # logit = A * kappa + C  (C universal across architectures)
        coeffs = np.polyfit(kappas_all, logits_all, 1)
        A_simple = float(coeffs[0])

        # Also within-arch demeaned (per-arch C):
        kc_list, lc_list = [], []
        for model, mpts in by_model.items():
            if len(mpts) < 2:
                continue
            kappas = np.array([p["kappa_nearest"] for p in mpts])
            logits = np.array([p["logit_q"] for p in mpts])
            kc_list.extend((kappas - kappas.mean()).tolist())
            lc_list.extend((logits - logits.mean()).tolist())
        kc = np.array(kc_list)
        lc = np.array(lc_list)
        A_within = float(np.dot(kc, lc) / (np.dot(kc, kc) + 1e-12))

        # Bootstrap std
        n = len(kappas_all)
        boot_A = []
        rng_b = np.random.default_rng(42)
        for _ in range(200):
            idx = rng_b.integers(0, n, size=n)
            boot_A.append(float(np.polyfit(kappas_all[idx], logits_all[idx], 1)[0]))

        result[ds_name] = {
            "A_pooled": A_simple,       # simple OLS (cross-arch, used for K-scaling)
            "A_within": A_within,       # within-arch demeaned (across layers)
            "A_std":    float(np.std(boot_A)),
            "n_models": len(by_model),
            "n_points": len(ds_pts),
            "K": DATASETS[ds_name],
        }
    return result


def fit_k_scaling(slope_dict):
    """
    Fit A(K) = a / log(K) + b using the per-dataset slopes.
    Returns (a, b, r_fit).
    """
    ds_names = list(slope_dict.keys())
    K_vals = np.array([slope_dict[d]["K"] for d in ds_names], dtype=float)
    A_vals = np.array([slope_dict[d]["A_pooled"] for d in ds_names], dtype=float)

    # Model: A = a / log(K) + b
    # Linearize: A = a * (1/log(K)) + b -> OLS
    X = 1.0 / np.log(K_vals)
    coeffs = np.polyfit(X, A_vals, 1)  # [a, b]
    a, b = coeffs
    A_pred = a * X + b
    r, _ = pearsonr(A_vals, A_pred)
    return float(a), float(b), float(r)


def predict_q_for_dataset(test_points, A_pred_test, C_arch_map):
    """
    Predict q for each test point using A_pred_test and per-arch C.
    Returns arrays: q_pred, q_actual.
    """
    q_pred_list = []
    q_actual_list = []
    for p in test_points:
        model = p["model"]
        C = C_arch_map.get(model, 0.0)
        logit_pred = A_pred_test * p["kappa_nearest"] + C
        q_pred = 1.0 / (1.0 + np.exp(-logit_pred))
        q_pred_list.append(float(np.clip(q_pred, 1e-6, 1 - 1e-6)))
        q_actual_list.append(p["q"])
    return np.array(q_pred_list), np.array(q_actual_list)


def compute_c_per_arch(points, datasets, A_fixed):
    """
    For each architecture, compute C_arch = mean(logit_q - A*kappa) across training data.
    """
    from collections import defaultdict
    residuals = defaultdict(list)
    for p in points:
        if p["dataset"] in datasets:
            resid = p["logit_q"] - A_fixed * p["kappa_nearest"]
            residuals[p["model"]].append(resid)
    return {m: float(np.mean(rs)) for m, rs in residuals.items()}


def json_default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    print("=" * 70)
    print("K-SCALING PROSPECTIVE PREDICTION TEST")
    print("=" * 70)
    print(f"Pre-registered: H1 rho>={LODO_RHO_THRESH}, H2 MAE<={LODO_MAE_THRESH}, "
          f"H3 A_ratio<={A_RATIO_THRESH}, H4 global_rho>={GLOBAL_RHO_THRESH}")

    # Load all caches
    print("\nLoading caches...")
    points = load_all_caches()
    print(f"Total points: {len(points)}")
    ds_counts = {}
    for p in points:
        ds_counts[p["dataset"]] = ds_counts.get(p["dataset"], 0) + 1
    for ds, n in sorted(ds_counts.items()):
        print(f"  {ds}: {n} points (K={DATASETS[ds]})")

    # Step 1: Fit per-dataset slopes on ALL data (for reference)
    print("\nStep 1: Fit per-dataset slopes (all data)")
    all_slopes = fit_slope_per_dataset(points, list(DATASETS.keys()))
    for ds, info in sorted(all_slopes.items()):
        print(f"  {ds} K={info['K']}: A_pooled={info['A_pooled']:.4f} +/- {info['A_std']:.4f} ({info['n_models']} models)")

    # Fit global A(K) model
    a_global, b_global, r_fit = fit_k_scaling(all_slopes)
    print(f"\nGlobal A(K) = {a_global:.4f}/log(K) + {b_global:.4f}  (r_fit={r_fit:.4f})")

    # Step 2: LODO validation
    print("\nStep 2: Leave-one-dataset-out (LODO)")
    ds_list = list(DATASETS.keys())

    lodo_results = []
    all_pred = []
    all_actual = []

    for ds_out in ds_list:
        ds_train = [d for d in ds_list if d != ds_out]
        K_out = DATASETS[ds_out]

        # Fit A(K) on training datasets
        train_slopes = fit_slope_per_dataset(points, ds_train)
        a_fit, b_fit, r_train = fit_k_scaling(train_slopes)
        A_pred_out = a_fit / np.log(K_out) + b_fit
        A_actual_out = all_slopes[ds_out]["A_pooled"]
        A_ratio = abs(A_pred_out / (A_actual_out + 1e-9) - 1.0)

        print(f"\n  Held-out: {ds_out} (K={K_out})")
        print(f"    A(K) fit: a={a_fit:.4f}, b={b_fit:.4f} (r={r_train:.4f})")
        print(f"    A_pred={A_pred_out:.4f}, A_actual={A_actual_out:.4f}, ratio_error={A_ratio:.3f}")

        # Compute C_arch from training datasets using the GLOBAL A (fairness: not refitting A)
        # We use A_pred_out to be truly prospective
        train_pts = [p for p in points if p["dataset"] in ds_train]
        C_arch_map = compute_c_per_arch(train_pts, ds_train, A_pred_out)

        # Predict on test set
        test_pts = [p for p in points if p["dataset"] == ds_out]
        q_pred, q_actual = predict_q_for_dataset(test_pts, A_pred_out, C_arch_map)

        if len(q_pred) >= 4:
            rho, p_rho = spearmanr(q_pred, q_actual)
            r, p_r = pearsonr(q_pred, q_actual)
            mae = float(np.mean(np.abs(q_pred - q_actual)))
        else:
            rho = r = mae = 0.0
            p_rho = p_r = 1.0

        pass_H1 = bool(rho >= LODO_RHO_THRESH)
        pass_H2 = bool(mae <= LODO_MAE_THRESH)
        pass_H3 = bool(A_ratio <= A_RATIO_THRESH)
        print(f"    n_test={len(test_pts)}: rho={rho:.4f} {'PASS' if pass_H1 else 'FAIL'}, "
              f"MAE={mae:.4f} {'PASS' if pass_H2 else 'FAIL'}, "
              f"A_error={A_ratio:.3f} {'PASS' if pass_H3 else 'FAIL'}")

        all_pred.extend(q_pred.tolist())
        all_actual.extend(q_actual)

        lodo_results.append({
            "held_out_dataset": ds_out,
            "K_out": K_out,
            "a_fit": float(a_fit),
            "b_fit": float(b_fit),
            "r_train": float(r_train),
            "A_pred": float(A_pred_out),
            "A_actual": float(A_actual_out),
            "A_ratio_error": float(A_ratio),
            "n_test": len(test_pts),
            "rho": float(rho),
            "r": float(r),
            "mae": float(mae),
            "pass_H1": pass_H1,
            "pass_H2": pass_H2,
            "pass_H3": pass_H3,
        })

    # Global pooled
    all_pred = np.array(all_pred)
    all_actual = np.array(all_actual)
    global_rho, _ = spearmanr(all_pred, all_actual)
    global_r, _ = pearsonr(all_pred, all_actual)
    global_mae = float(np.mean(np.abs(all_pred - all_actual)))
    mean_lodo_rho = float(np.mean([r["rho"] for r in lodo_results]))
    mean_lodo_mae = float(np.mean([r["mae"] for r in lodo_results]))
    pass_H4 = bool(global_rho >= GLOBAL_RHO_THRESH)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':>14} | {'A_pred':>8} {'A_actual':>9} {'A_error':>8} | {'rho':>7} {'pass':>5} | {'MAE':>7} {'pass':>5}")
    for r in lodo_results:
        p1 = "PASS" if r["pass_H1"] else "FAIL"
        p2 = "PASS" if r["pass_H2"] else "FAIL"
        p3 = "PASS" if r["pass_H3"] else "FAIL"
        print(f"  {r['held_out_dataset']:>12} | {r['A_pred']:>8.4f} {r['A_actual']:>9.4f} {r['A_ratio_error']:>8.3f} | "
              f"{r['rho']:>7.4f} {p1:>5} | {r['mae']:>7.4f} {p2:>5}")
    print(f"\nMean LODO rho: {mean_lodo_rho:.4f} {'PASS' if mean_lodo_rho >= LODO_RHO_THRESH else 'FAIL'}")
    print(f"Mean LODO MAE: {mean_lodo_mae:.4f} {'PASS' if mean_lodo_mae <= LODO_MAE_THRESH else 'FAIL'}")
    print(f"Global pooled rho: {global_rho:.4f} {'PASS' if pass_H4 else 'FAIL'}")

    # Final pass count
    n_H1_pass = sum(1 for r in lodo_results if r["pass_H1"])
    n_H2_pass = sum(1 for r in lodo_results if r["pass_H2"])
    n_H3_pass = sum(1 for r in lodo_results if r["pass_H3"])
    print(f"\nH1 (rho>={LODO_RHO_THRESH}): {n_H1_pass}/4 LODO folds pass")
    print(f"H2 (MAE<={LODO_MAE_THRESH}): {n_H2_pass}/4 LODO folds pass")
    print(f"H3 (A_error<={A_RATIO_THRESH}): {n_H3_pass}/4 LODO folds pass")
    print(f"H4 (global_rho>={GLOBAL_RHO_THRESH}): {'PASS' if pass_H4 else 'FAIL'}")

    output = {
        "experiment": "k_scaling_prediction",
        "session": 43,
        "preregistered": {
            "H1_rho_threshold": LODO_RHO_THRESH,
            "H2_mae_threshold": LODO_MAE_THRESH,
            "H3_A_ratio_threshold": A_RATIO_THRESH,
            "H4_global_rho_threshold": GLOBAL_RHO_THRESH,
            "datasets": list(DATASETS.keys()),
            "K_values": list(DATASETS.values()),
            "A_K_model": "A(K) = a/log(K) + b",
            "C_fitting": "per-architecture, trained on training datasets only",
        },
        "global_A_fit": {
            "a": a_global,
            "b": b_global,
            "r_fit": r_fit,
        },
        "per_dataset_slopes_all": {
            ds: {
                "K": info["K"],
                "A_pooled": info["A_pooled"],
                "A_std": info["A_std"],
                "n_models": info["n_models"],
            }
            for ds, info in all_slopes.items()
        },
        "lodo": lodo_results,
        "summary": {
            "mean_lodo_rho": mean_lodo_rho,
            "mean_lodo_mae": mean_lodo_mae,
            "global_rho": float(global_rho),
            "global_mae": global_mae,
            "n_H1_pass": n_H1_pass,
            "n_H2_pass": n_H2_pass,
            "n_H3_pass": n_H3_pass,
            "pass_H4": pass_H4,
            "pass_H1_mean": bool(mean_lodo_rho >= LODO_RHO_THRESH),
            "pass_H2_mean": bool(mean_lodo_mae <= LODO_MAE_THRESH),
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
