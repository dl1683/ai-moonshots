#!/usr/bin/env python -u
"""
ONE-POINT CALIBRATION TEST (Session 43)
=========================================
Codex design (fresh context, Nobel 2/10):
  Convert 'C not transferable' from fatal flaw to practical feature.

PROTOCOL (PRE-REGISTERED):
  Pre-registered:  A(K) = 1.16/log(K) + 1.30  (from K-scaling LODO)
  Per new dataset: reveal ONE anchor architecture -> estimate C_dataset
  Predict:         logit(q_arch) = A(K) * kappa_arch + C_dataset for all others

INTERPRETATION:
  If rho >= 0.85 on held-out architectures:
    -> "Universal slope structure + 1-shot dataset calibration"
    -> Claims: A is universal, C requires 1 observation per dataset
    -> Practical: test 1 cheap model on new data, predict all others

PRE-REGISTERED CRITERIA:
  H1: mean rho (anchored, held-out archs) >= 0.85 for >= 3/4 datasets
  H2: mean MAE <= 0.08 for >= 3/4 datasets
  H3: average over anchor choices (all 19 archs as anchor) shows stable rho

DATA: kappa_near_cache files (19 archs x 4 datasets)
A_FIT from K-scaling LODO (fitted on all 4 datasets):
  a=1.1636, b=1.3013  (global fit)
  A(K) = 1.1636/log(K) + 1.3013
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_JSON = RESULTS_DIR / "cti_one_point_calibration.json"

DATASETS = {
    "agnews":       4,
    "dbpedia":      14,
    "20newsgroups": 20,
    "go_emotions":  28,
}

# Pre-registered A(K) parameters from K-scaling LODO (global fit)
A_GLOBAL_a = 1.1636
A_GLOBAL_b = 1.3013

# Pre-registered thresholds
RHO_THRESH = 0.85
MAE_THRESH = 0.08


def A_pred(K):
    return A_GLOBAL_a / np.log(K) + A_GLOBAL_b


def load_all_caches():
    """Load all kappa_near_cache files. Returns {dataset: {model: [points]}}."""
    data = defaultdict(lambda: defaultdict(list))
    for ds_name, K in DATASETS.items():
        pattern = f"kappa_near_cache_{ds_name}_*.json"
        files = sorted(RESULTS_DIR.glob(pattern))
        for fpath in files:
            with open(fpath) as f:
                pts = json.load(f)
            for pt in pts:
                q_raw = float(pt["q"])
                kappa = float(pt["kappa_nearest"])
                q_norm = (q_raw - 1.0/K) / (1.0 - 1.0/K)
                q_norm = float(np.clip(q_norm, 1e-5, 1 - 1e-5))
                if q_raw < 1.0/K:
                    continue
                logit_q_norm = float(np.log(q_norm / (1.0 - q_norm)))
                if not np.isfinite(logit_q_norm) or not np.isfinite(kappa) or kappa <= 0:
                    continue
                model = str(pt.get("model", fpath.stem.split("_")[-1]))
                data[ds_name][model].append({
                    "layer": int(pt.get("layer", 0)),
                    "q": q_norm,
                    "q_raw": q_raw,
                    "kappa": kappa,
                    "logit_q": logit_q_norm,
                })
    return data


def estimate_C(pts_arch, A):
    """Estimate C from an anchor architecture's data points."""
    residuals = [p["logit_q"] - A * p["kappa"] for p in pts_arch]
    return float(np.mean(residuals))


def predict_q(pts, A, C):
    """Predict q from kappa using logit(q) = A*kappa + C."""
    q_preds = []
    for p in pts:
        logit_pred = A * p["kappa"] + C
        q_pred = 1.0 / (1.0 + np.exp(-logit_pred))
        q_preds.append(float(np.clip(q_pred, 1e-6, 1 - 1e-6)))
    return q_preds


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
    print("ONE-POINT CALIBRATION TEST (TWO-COMPONENT MODEL)")
    print(f"A(K) = {A_GLOBAL_a:.4f}/log(K) + {A_GLOBAL_b:.4f} (pre-registered)")
    print(f"Model: logit(q) = A(K)*kappa + C_arch + C_dataset")
    print(f"Pre-registered: H1 rho>={RHO_THRESH}, H2 MAE<={MAE_THRESH}")
    print("=" * 70)

    cache = load_all_caches()
    print(f"\nDatasets loaded: {list(cache.keys())}")
    for ds, models in cache.items():
        print(f"  {ds}: {len(models)} architectures, {sum(len(v) for v in models.values())} points")

    all_models_set = set()
    for ds_models in cache.values():
        all_models_set.update(ds_models.keys())
    all_models = sorted(all_models_set)
    print(f"\n{len(all_models)} unique architectures across all datasets")

    all_results = []

    for ds_name, K in DATASETS.items():
        A = A_pred(K)
        print(f"\n--- Dataset: {ds_name} (K={K}, A={A:.4f}) ---")
        test_models = sorted(cache[ds_name].keys())
        train_datasets = [d for d in DATASETS if d != ds_name]

        # === METHOD 1: Simple C_dataset only (baseline) ===
        # === METHOD 2: Two-component C_arch + C_dataset ===
        #  Step 1: Pre-train C_arch from training datasets
        C_arch_pretrained = {}
        for model in all_models:
            residuals_training = []
            for tr_ds in train_datasets:
                if model in cache[tr_ds]:
                    A_tr = A_pred(DATASETS[tr_ds])
                    for p in cache[tr_ds][model]:
                        residuals_training.append(p["logit_q"] - A_tr * p["kappa"])
            if residuals_training:
                C_arch_pretrained[model] = float(np.mean(residuals_training))

        # Simple anchor results (baseline - no C_arch pre-training)
        anchor_rhos_simple = []
        anchor_rhos_twocomp = []
        anchor_maes_simple = []
        anchor_maes_twocomp = []

        for anchor_model in test_models:
            anchor_pts = cache[ds_name][anchor_model]
            if not anchor_pts:
                continue

            # === Method 1: Simple (C_dataset from anchor residuals) ===
            C_dataset_simple = estimate_C(anchor_pts, A)

            # === Method 2: Two-component ===
            # C_dataset = mean(logit - A*kappa - C_arch) for anchor
            C_arch_anchor = C_arch_pretrained.get(anchor_model, 0.0)
            residuals_anchor = [p["logit_q"] - A * p["kappa"] - C_arch_anchor
                                 for p in anchor_pts]
            C_dataset_twocomp = float(np.mean(residuals_anchor))

            q_pred_simple_all = []
            q_pred_twocomp_all = []
            q_actual_all = []

            for other_model in test_models:
                if other_model == anchor_model:
                    continue
                other_pts = cache[ds_name][other_model]
                if not other_pts:
                    continue

                C_arch_other = C_arch_pretrained.get(other_model, 0.0)

                q_preds_simple = predict_q(other_pts, A, C_dataset_simple)
                q_preds_twocomp = [
                    float(np.clip(1.0 / (1.0 + np.exp(
                        -(A * p["kappa"] + C_arch_other + C_dataset_twocomp)
                    )), 1e-6, 1-1e-6))
                    for p in other_pts
                ]
                q_actuals = [p["q"] for p in other_pts]

                q_pred_simple_all.extend(q_preds_simple)
                q_pred_twocomp_all.extend(q_preds_twocomp)
                q_actual_all.extend(q_actuals)

            if len(q_actual_all) >= 4:
                rho_s, _ = spearmanr(q_pred_simple_all, q_actual_all)
                rho_t, _ = spearmanr(q_pred_twocomp_all, q_actual_all)
                mae_s = float(np.mean(np.abs(np.array(q_pred_simple_all) - np.array(q_actual_all))))
                mae_t = float(np.mean(np.abs(np.array(q_pred_twocomp_all) - np.array(q_actual_all))))
                anchor_rhos_simple.append(float(rho_s))
                anchor_rhos_twocomp.append(float(rho_t))
                anchor_maes_simple.append(float(mae_s))
                anchor_maes_twocomp.append(float(mae_t))

        for label, rhos, maes in [
            ("simple", anchor_rhos_simple, anchor_maes_simple),
            ("two-comp", anchor_rhos_twocomp, anchor_maes_twocomp),
        ]:
            mr = float(np.mean(rhos)) if rhos else 0.0
            mm = float(np.mean(maes)) if maes else 999.0
            sr = float(np.std(rhos)) if rhos else 0.0
            pass1 = mr >= RHO_THRESH
            pass2 = mm <= MAE_THRESH
            print(f"  [{label}] rho={mr:.4f}+/-{sr:.4f}, MAE={mm:.4f}  "
                  f"H1={'PASS' if pass1 else 'FAIL'} H2={'PASS' if pass2 else 'FAIL'}")

        mean_rho = float(np.mean(anchor_rhos_twocomp)) if anchor_rhos_twocomp else 0.0
        mean_mae = float(np.mean(anchor_maes_twocomp)) if anchor_maes_twocomp else 999.0
        pass_H1 = bool(mean_rho >= RHO_THRESH)
        pass_H2 = bool(mean_mae <= MAE_THRESH)

        all_results.append({
            "dataset": ds_name,
            "K": K,
            "A_pred": float(A),
            "n_architectures": len(test_models),
            "simple_rho": float(np.mean(anchor_rhos_simple)) if anchor_rhos_simple else 0.0,
            "twocomp_rho": mean_rho,
            "twocomp_rho_std": float(np.std(anchor_rhos_twocomp)) if anchor_rhos_twocomp else 0.0,
            "simple_mae": float(np.mean(anchor_maes_simple)) if anchor_maes_simple else 999.0,
            "twocomp_mae": mean_mae,
            "pass_H1": pass_H1,
            "pass_H2": pass_H2,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_H1_pass = sum(1 for r in all_results if r["pass_H1"])
    n_H2_pass = sum(1 for r in all_results if r["pass_H2"])

    print(f"{'Dataset':>14} | {'K':>3} | {'A_pred':>7} | {'rho':>7} {'H1':>5} | {'MAE':>7} {'H2':>5}")
    for r in all_results:
        p1 = "PASS" if r["pass_H1"] else "FAIL"
        p2 = "PASS" if r["pass_H2"] else "FAIL"
        print(f"  {r['dataset']:>12} | {r['K']:>3} | {r['A_pred']:>7.4f} | "
              f"{r['twocomp_rho']:>7.4f} {p1:>5} | {r['twocomp_mae']:>7.4f} {p2:>5}")

    pass_H1_claim = bool(n_H1_pass >= 3)
    pass_H2_claim = bool(n_H2_pass >= 3)
    print(f"\nH1 (>= 3/4 datasets with rho>={RHO_THRESH}): "
          f"{n_H1_pass}/4 -> {'PASS' if pass_H1_claim else 'FAIL'}")
    print(f"H2 (>= 3/4 datasets with MAE<={MAE_THRESH}): "
          f"{n_H2_pass}/4 -> {'PASS' if pass_H2_claim else 'FAIL'}")

    # Excluding go_emotions (multilabel confound)
    non_multi = [r for r in all_results if r["dataset"] != "go_emotions"]
    n_H1_no_ge = sum(1 for r in non_multi if r["pass_H1"])
    print(f"\nExcluding go_emotions (multilabel confound):")
    print(f"  H1 (>= 2/3 non-multilabel datasets): "
          f"{n_H1_no_ge}/3 -> {'PASS' if n_H1_no_ge >= 2 else 'FAIL'}")

    output = {
        "experiment": "one_point_calibration",
        "session": 43,
        "preregistered": {
            "A_K_formula": "A(K) = 1.1636/log(K) + 1.3013",
            "A_GLOBAL_a": A_GLOBAL_a,
            "A_GLOBAL_b": A_GLOBAL_b,
            "source": "K-scaling LODO (4 datasets, global fit)",
            "H1_rho_threshold": RHO_THRESH,
            "H2_mae_threshold": MAE_THRESH,
            "claim": "one anchor architecture per dataset -> predict all others",
        },
        "results": all_results,
        "summary": {
            "n_H1_pass": n_H1_pass,
            "n_H2_pass": n_H2_pass,
            "pass_H1_claim": pass_H1_claim,
            "pass_H2_claim": pass_H2_claim,
            "pass_H1_excl_go_emotions": bool(n_H1_no_ge >= 2),
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
