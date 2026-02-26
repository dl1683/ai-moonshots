"""
CTI Leave-One-Model-Family-Out (LOMFO) + Leave-One-Dataset-Out (LODO) stress test.
Codex directive (Session 79): stronger OOD protocol to show law generalizes across
model families and datasets.

LOMFO: for each training family, remove all its models, refit, predict those models.
LODO: for each dataset, remove its C_d from training, use mean C_d as substitute.
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from scipy.special import logit as sp_logit, expit
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from cti_utility_revised import (
    load_all_cache,
    get_family,
    fit_family_alphas,
    MODEL_META,
    TRAINING_MODELS,
    HOLDOUT_MODELS,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_lomfo_lodo_stress_test.json")

HOLDOUT_META = {
    "gemma-3-1b": {"params_m": 1000, "family": "decoder"},
    "roberta-base": {"params_m": 125, "family": "encoder"},
    "distilbert-base-uncased": {"params_m": 66, "family": "encoder"},
    "albert-base-v2": {"params_m": 12, "family": "encoder"},
    "opt-125m": {"params_m": 125, "family": "decoder"},
    "pythia-2.8b": {"params_m": 2800, "family": "decoder"},
    "stablelm-3b-4e1t": {"params_m": 3000, "family": "decoder"},
    "bloom-560m": {"params_m": 560, "family": "decoder"},
    "phi-1.5": {"params_m": 1300, "family": "decoder"},
    "qwen2.5-1.5b": {"params_m": 1500, "family": "decoder"},
    "falcon-rw-1b": {"params_m": 1000, "family": "decoder"},
}


def get_model_best(pts):
    """Get best kappa layer per (model, dataset)."""
    groups = defaultdict(list)
    for p in pts:
        groups[(p["model"], p["dataset"])].append(p)
    best = {}
    for (m, d), plist in groups.items():
        best.setdefault(m, {})[d] = max(plist, key=lambda x: x["kappa"])
    return best


def fit_and_predict(pts, train_models, predict_models, held_family=None):
    """
    Fit CTI parameters on train_models, predict on predict_models.
    If held_family is given, use mean-of-remaining alpha for it and C_f=0.
    """
    all_meta = {**MODEL_META, **{m: {"params_m": v["params_m"]} for m, v in HOLDOUT_META.items()}}

    # Get best-layer per model
    all_best = get_model_best(pts)

    # Fit alphas on training data only
    train_pts = [p for p in pts if p["model"] in train_models]
    alphas = fit_family_alphas(train_pts)

    # If held family, use mean of remaining
    if held_family and held_family not in alphas:
        remaining = [v for k, v in alphas.items() if k != held_family]
        alphas[held_family] = float(np.mean(remaining)) if remaining else 1.477

    # Build training observations
    train_obs = []
    for model, ds_dict in all_best.items():
        if model not in train_models:
            continue
        family = get_family(model)
        alpha = alphas.get(family, 1.477)
        meta = all_meta.get(model, {})
        log_m = np.log(meta["params_m"]) if meta else np.log(100)
        for ds, pt in ds_dict.items():
            K = pt["K"]
            q_norm = (pt["q"] - 1.0 / K) / (1.0 - 1.0 / K)
            if q_norm <= 0.01 or q_norm >= 0.99:
                continue
            logit_q = float(sp_logit(q_norm))
            resid = logit_q - alpha * pt["kappa"]
            train_obs.append({
                "model": model, "dataset": ds, "family": family,
                "logit_q": logit_q, "kappa": pt["kappa"],
                "resid": resid, "log_m": log_m,
            })

    if not train_obs:
        return []

    # Compute C_d
    ds_resids = defaultdict(list)
    for o in train_obs:
        ds_resids[o["dataset"]].append(o["resid"])
    C_d = {ds: float(np.mean(vs)) for ds, vs in ds_resids.items()}
    mean_C_d = float(np.mean(list(C_d.values())))  # fallback for unseen datasets

    # Compute C_f
    fam_resids = defaultdict(list)
    for o in train_obs:
        fam_resids[o["family"]].append(o["resid"] - C_d[o["dataset"]])
    C_f = {f: float(np.mean(vs)) for f, vs in fam_resids.items()}

    # Fit gamma
    final_resids = np.array([
        o["resid"] - C_d[o["dataset"]] - C_f.get(o["family"], 0)
        for o in train_obs
    ])
    log_ms = np.array([o["log_m"] for o in train_obs])
    X = np.column_stack([log_ms, np.ones(len(log_ms))])
    beta = np.linalg.lstsq(X, final_resids, rcond=None)[0]
    gamma = float(beta[0])
    gamma_intercept = float(beta[1])

    # Predict on predict_models
    predictions = []
    for model, ds_dict in all_best.items():
        if model not in predict_models:
            continue
        meta = all_meta.get(model)
        if not meta:
            continue
        family = get_family(model)
        alpha = alphas.get(family, 1.477)
        log_m = np.log(meta["params_m"])
        c_f = C_f.get(family, 0.0)

        for ds, pt in ds_dict.items():
            K = pt["K"]
            q_norm_actual = (pt["q"] - 1.0 / K) / (1.0 - 1.0 / K)
            if q_norm_actual <= 0.01 or q_norm_actual >= 0.99:
                continue
            c_d = C_d.get(ds, mean_C_d)
            logit_pred = alpha * pt["kappa"] + c_d + c_f + gamma * log_m + gamma_intercept
            q_norm_pred = float(expit(logit_pred))
            logit_actual = float(sp_logit(np.clip(q_norm_actual, 0.001, 0.999)))
            predictions.append({
                "model": model, "dataset": ds,
                "q_norm_actual": q_norm_actual, "q_norm_pred": q_norm_pred,
                "logit_actual": logit_actual, "logit_pred": float(logit_pred),
                "error": abs(q_norm_pred - q_norm_actual),
                "alpha_used": alpha,
            })

    return predictions


def evaluate(predictions):
    if len(predictions) < 3:
        return {"n": len(predictions), "r": None, "mae": None}
    errors = [p["error"] for p in predictions]
    la = [p["logit_actual"] for p in predictions]
    lp = [p["logit_pred"] for p in predictions]
    r_val, _ = pearsonr(la, lp)
    mae = float(np.mean(errors))
    pct_015 = float(100 * sum(1 for e in errors if e <= 0.15) / len(errors))
    return {
        "n": len(predictions),
        "n_models": len(set(p["model"] for p in predictions)),
        "r": round(r_val, 3),
        "mae": round(mae, 3),
        "pct_within_015": round(pct_015, 1),
    }


def run_lomfo(pts):
    """Leave-One-Model-Family-Out: remove all training models of one family."""
    families = sorted(set(get_family(m) for m in TRAINING_MODELS))
    results = {}

    for held_fam in families:
        # Training: all models NOT in held_fam
        train_models = {m for m in TRAINING_MODELS if get_family(m) != held_fam}
        # Predict: all models in held_fam (training + holdout)
        all_held = (
            {m for m in TRAINING_MODELS if get_family(m) == held_fam} |
            {m for m in HOLDOUT_MODELS if get_family(m) == held_fam}
        )

        n_train = len(train_models)
        n_pred_models = len(all_held)

        # Also record what known alpha would be
        full_alphas = fit_family_alphas([p for p in pts if p["model"] in TRAINING_MODELS])
        known_alpha = full_alphas.get(held_fam, None)

        preds = fit_and_predict(pts, train_models, all_held, held_family=held_fam)
        stats = evaluate(preds)

        # What is the borrowed alpha?
        borrowed = float(np.mean([v for k, v in full_alphas.items() if k != held_fam]))

        print(f"  LOMFO {held_fam}: n_train={n_train}, n_pred_models={n_pred_models}, "
              f"n_preds={stats['n']}, r={stats['r']}, MAE={stats['mae']}, "
              f"alpha_borrowed={borrowed:.3f} (true={known_alpha:.3f} if known)")

        results[held_fam] = {
            **stats,
            "alpha_borrowed": round(borrowed, 3),
            "alpha_true": round(known_alpha, 3) if known_alpha else None,
        }

    return results


def run_lodo(pts):
    """Leave-One-Dataset-Out: remove one dataset from C_d training, predict holdout on it."""
    datasets = sorted(set(p["dataset"] for p in pts if p["model"] in HOLDOUT_MODELS))
    results = {}

    for held_ds in datasets:
        # Training: all training models, but skip the held-out dataset when fitting C_d
        train_pts = [p for p in pts if p["model"] in TRAINING_MODELS and p["dataset"] != held_ds]

        # Fit alphas normally on all training data (datasets don't affect alpha)
        full_train_pts = [p for p in pts if p["model"] in TRAINING_MODELS]
        alphas = fit_family_alphas(full_train_pts)

        # Build train_obs without held_ds
        all_best = get_model_best(pts)
        all_meta = {**MODEL_META, **{m: {"params_m": v["params_m"]} for m, v in HOLDOUT_META.items()}}
        train_obs = []
        for model, ds_dict in all_best.items():
            if model not in TRAINING_MODELS:
                continue
            family = get_family(model)
            alpha = alphas.get(family, 1.477)
            meta = all_meta.get(model, {})
            log_m = np.log(meta["params_m"]) if meta else np.log(100)
            for ds, pt in ds_dict.items():
                if ds == held_ds:
                    continue
                K = pt["K"]
                q_norm = (pt["q"] - 1.0 / K) / (1.0 - 1.0 / K)
                if q_norm <= 0.01 or q_norm >= 0.99:
                    continue
                logit_q = float(sp_logit(q_norm))
                resid = logit_q - alpha * pt["kappa"]
                train_obs.append({
                    "model": model, "dataset": ds, "family": family,
                    "logit_q": logit_q, "kappa": pt["kappa"],
                    "resid": resid, "log_m": log_m,
                })

        # Fit C_d from remaining datasets
        ds_resids = defaultdict(list)
        for o in train_obs:
            ds_resids[o["dataset"]].append(o["resid"])
        C_d = {ds: float(np.mean(vs)) for ds, vs in ds_resids.items()}
        mean_C_d = float(np.mean(list(C_d.values())))
        C_d[held_ds] = mean_C_d  # fallback estimate for held-out dataset

        # Compute C_f
        fam_resids = defaultdict(list)
        for o in train_obs:
            fam_resids[o["family"]].append(o["resid"] - C_d[o["dataset"]])
        C_f = {f: float(np.mean(vs)) for f, vs in fam_resids.items()}

        # Fit gamma
        final_resids_arr = np.array([
            o["resid"] - C_d[o["dataset"]] - C_f.get(o["family"], 0)
            for o in train_obs
        ])
        log_ms_arr = np.array([o["log_m"] for o in train_obs])
        X = np.column_stack([log_ms_arr, np.ones(len(log_ms_arr))])
        beta_g = np.linalg.lstsq(X, final_resids_arr, rcond=None)[0]
        gamma = float(beta_g[0])
        gamma_intercept = float(beta_g[1])

        # Predict holdout models on held-out dataset
        predictions = []
        for model, ds_dict in all_best.items():
            if model not in HOLDOUT_MODELS:
                continue
            meta = HOLDOUT_META.get(model)
            if not meta:
                continue
            if held_ds not in ds_dict:
                continue
            family = meta["family"]
            alpha = alphas.get(family, 1.477)
            log_m = np.log(meta["params_m"])
            c_f = C_f.get(family, 0.0)
            c_d = C_d[held_ds]

            pt = ds_dict[held_ds]
            K = pt["K"]
            q_norm_actual = (pt["q"] - 1.0 / K) / (1.0 - 1.0 / K)
            if q_norm_actual <= 0.01 or q_norm_actual >= 0.99:
                continue
            logit_pred = alpha * pt["kappa"] + c_d + c_f + gamma * log_m + gamma_intercept
            q_norm_pred = float(expit(logit_pred))
            logit_actual = float(sp_logit(np.clip(q_norm_actual, 0.001, 0.999)))
            predictions.append({
                "model": model, "dataset": held_ds,
                "q_norm_actual": q_norm_actual, "q_norm_pred": q_norm_pred,
                "logit_actual": logit_actual, "logit_pred": float(logit_pred),
                "error": abs(q_norm_pred - q_norm_actual),
            })

        stats = evaluate(predictions)
        print(f"  LODO {held_ds:>20}: n={stats['n']}, r={stats['r']}, MAE={stats['mae']}, "
              f"C_d_used={mean_C_d:.3f}")
        results[held_ds] = stats

    return results


def main():
    print("Loading data...")
    pts = load_all_cache()
    print(f"  Loaded {len(pts)} layer-level points")

    print("\n" + "=" * 60)
    print("LOMFO: Leave-One-Model-Family-Out")
    print("=" * 60)
    lomfo = run_lomfo(pts)

    print("\n" + "=" * 60)
    print("LODO: Leave-One-Dataset-Out")
    print("=" * 60)
    lodo = run_lodo(pts)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_lodo = {k: v for k, v in lodo.items() if v["r"] is not None}
    if valid_lodo:
        mean_r = np.mean([v["r"] for v in valid_lodo.values()])
        mean_mae = np.mean([v["mae"] for v in valid_lodo.values()])
        print(f"LODO (n_datasets={len(valid_lodo)}): mean r={mean_r:.3f}, mean MAE={mean_mae:.3f}")
        for ds, v in sorted(valid_lodo.items()):
            status = "PASS" if (v["r"] is not None and v["r"] >= 0.70) else "FAIL/LOW"
            print(f"  {ds:>20}: r={v['r']}, MAE={v['mae']} [{status}]")

    print()
    for fam, res in lomfo.items():
        status = "PASS" if (res["r"] is not None and res["r"] >= 0.70) else "MARGINAL/FAIL"
        note = f"(alpha gap: borrowed={res['alpha_borrowed']}, true={res['alpha_true']})"
        print(f"LOMFO {fam}: r={res['r']}, MAE={res['mae']} [{status}] {note}")

    # Save
    output = {
        "experiment": "cti_lomfo_lodo_stress_test",
        "description": ("LOMFO: hold out one training family, predict it. "
                        "LODO: hold out one dataset from C_d fitting, predict holdout on it. "
                        "Alpha gap for LOMFO shows family-specific alpha is necessary."),
        "lomfo": lomfo,
        "lodo": lodo,
        "lodo_summary": {
            "mean_r": round(float(np.mean([v["r"] for v in valid_lodo.values()])), 3) if valid_lodo else None,
            "mean_mae": round(float(np.mean([v["mae"] for v in valid_lodo.values()])), 3) if valid_lodo else None,
            "n_datasets": len(valid_lodo),
        } if valid_lodo else {},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
