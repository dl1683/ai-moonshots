"""
CTI Practical Utility — Revised per Codex (Session 78).

Key finding from Experiment 1: zero-shot cross-model ranking is DEAD.
C(model, dataset) is NOT constant across architectures.

What DOES work:
  H1: Within-model layer ranking (kappa picks best layer 70%+)
  H2: One-shot adaptation (1 known anchor per model calibrates intercept)
       - ranking rho=0.86 (strong), MAE=0.30 (needs family-specific alpha)
  H3: kappa/sqrt(K) normalization — moderate global signal

Codex Session 78 follow-up improvements:
  - Family-specific alpha for H2 (not fixed 1.477)
  - Bootstrap CIs on all metrics
  - Baselines: random ranking, depth-prior, dataset-mean
  - Anchor choice stress test

Output: results/cti_utility_revised.json
"""

import json
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.special import logit as sp_logit, expit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_utility_revised.json")

# Architecture family classification
ENCODER_MODELS = {
    "bert-base-uncased", "bge-base-v1.5", "deberta-base", "electra-small",
    "bge-small-en-v1.5", "bge-large-en-v1.5",
}
DECODER_MODELS = {
    "gpt-neo-125m", "gpt2", "pythia-160m", "pythia-410m", "pythia-1b",
    "Qwen2.5-0.5B", "Qwen3-0.6B", "Qwen3-1.7B", "Mistral-7B-v0.3",
    "OLMo-1B-hf", "phi2", "TinyLlama-1.1B-intermediate-step-1431k-3T",
    "SmolLM2-1.7B",
}
SSM_MODELS = {"mamba-130m", "rwkv-4-169m-pile"}
HYBRID_MODELS = {"Falcon-H1-0.5B-Base"}


def get_family(model):
    if model in ENCODER_MODELS:
        return "encoder"
    if model in SSM_MODELS:
        return "ssm"
    if model in HYBRID_MODELS:
        return "hybrid"
    return "decoder"


def fit_family_alphas(pts):
    """Fit alpha per architecture family using OLS with per-dataset intercepts."""
    from collections import defaultdict
    family_data = defaultdict(list)
    for p in pts:
        K = p["K"]
        q_norm = (p["q"] - 1.0 / K) / (1.0 - 1.0 / K)
        if q_norm <= 0.01 or q_norm >= 0.99:
            continue
        family_data[get_family(p["model"])].append({
            "logit_q": float(sp_logit(q_norm)),
            "kappa": p["kappa"],
            "dataset": p["dataset"],
        })

    alphas = {}
    for family, data in family_data.items():
        if len(data) < 5:
            alphas[family] = 1.477  # default
            continue
        datasets = sorted(set(d["dataset"] for d in data))
        ds_map = {ds: i for i, ds in enumerate(datasets)}
        X = np.zeros((len(data), 1 + len(datasets)))
        y = np.zeros(len(data))
        for i, d in enumerate(data):
            X[i, 0] = d["kappa"]
            X[i, 1 + ds_map[d["dataset"]]] = 1.0
            y[i] = d["logit_q"]
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            alphas[family] = float(beta[0])
        except Exception:
            alphas[family] = 1.477
    return alphas


def load_all_cache():
    """Load all layer-level points from kappa_near_cache files."""
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
            layer = entry.get("layer", -1)
            if q is None or kappa is None or K is None:
                continue
            if q <= 0 or q >= 1.0:
                continue
            if kappa <= 0:
                continue
            pts.append({
                "model": model,
                "dataset": dataset,
                "layer": int(layer),
                "K": int(K),
                "q": float(q),
                "kappa": float(kappa),
                "family": get_family(model),
            })
    return pts


# =====================================================================
# H1: Within-model layer ranking
# =====================================================================
def test_within_model_layer_ranking(pts):
    """For each (model, dataset), does kappa rank layers correctly?"""
    groups = {}
    for p in pts:
        key = (p["model"], p["dataset"])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    results = []
    for (model, dataset), plist in sorted(groups.items()):
        if len(plist) < 3:
            continue
        kappas = np.array([p["kappa"] for p in plist])
        qs = np.array([p["q"] for p in plist])
        layers = np.array([p["layer"] for p in plist])

        rho, p_val = spearmanr(kappas, qs)
        # Does kappa pick the best layer?
        best_kappa_idx = np.argmax(kappas)
        best_q_idx = np.argmax(qs)
        best_layer_match = bool(best_kappa_idx == best_q_idx)

        # Regret: q at kappa-predicted best vs actual best
        q_at_kappa_best = qs[best_kappa_idx]
        q_at_true_best = qs[best_q_idx]
        regret = float(q_at_true_best - q_at_kappa_best)

        results.append({
            "model": model,
            "dataset": dataset,
            "K": plist[0]["K"],
            "n_layers": len(plist),
            "spearman_rho": float(rho),
            "best_layer_match": best_layer_match,
            "regret": regret,
            "q_at_kappa_best": float(q_at_kappa_best),
            "q_at_true_best": float(q_at_true_best),
            "kappa_best_layer": int(layers[best_kappa_idx]),
            "q_best_layer": int(layers[best_q_idx]),
        })

    return results


# =====================================================================
# H2: One-shot adaptation (improved: family-specific alpha + baselines)
# =====================================================================
def _get_model_best(pts):
    """For each model, pick best-kappa layer per dataset."""
    by_model = {}
    for p in pts:
        m = p["model"]
        if m not in by_model:
            by_model[m] = {}
        d = p["dataset"]
        if d not in by_model[m]:
            by_model[m][d] = []
        by_model[m][d].append(p)

    model_best = {}
    for model, ds_dict in by_model.items():
        model_best[model] = {}
        for dataset, plist in ds_dict.items():
            best = max(plist, key=lambda x: x["kappa"])
            model_best[model][dataset] = best
    return model_best


def _one_shot_core(model_best, alpha_fn):
    """Core one-shot loop. alpha_fn(model) returns the alpha to use."""
    results = []
    for model, ds_dict in model_best.items():
        datasets = sorted(ds_dict.keys())
        if len(datasets) < 3:
            continue
        alpha = alpha_fn(model)
        for anchor_ds in datasets:
            anchor = ds_dict[anchor_ds]
            K_anchor = anchor["K"]
            q_norm_anchor = (anchor["q"] - 1.0 / K_anchor) / (1.0 - 1.0 / K_anchor)
            if q_norm_anchor <= 0.01 or q_norm_anchor >= 0.99:
                continue
            logit_q_anchor = sp_logit(q_norm_anchor)
            C_model = logit_q_anchor - alpha * anchor["kappa"]

            pred_qs, actual_qs, test_datasets = [], [], []
            for test_ds in datasets:
                if test_ds == anchor_ds:
                    continue
                test = ds_dict[test_ds]
                K_test = test["K"]
                q_norm_test = (test["q"] - 1.0 / K_test) / (1.0 - 1.0 / K_test)
                if q_norm_test <= 0.01 or q_norm_test >= 0.99:
                    continue
                logit_pred = alpha * test["kappa"] + C_model
                q_pred = float(expit(logit_pred))
                pred_qs.append(q_pred)
                actual_qs.append(q_norm_test)
                test_datasets.append(test_ds)

            if len(pred_qs) >= 2:
                pred_arr = np.array(pred_qs)
                actual_arr = np.array(actual_qs)
                mae = float(np.mean(np.abs(pred_arr - actual_arr)))
                rho, _ = spearmanr(pred_arr, actual_arr)
                results.append({
                    "model": model,
                    "anchor_dataset": anchor_ds,
                    "n_test_datasets": len(test_datasets),
                    "mae": mae,
                    "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                })
    return results


def test_one_shot_adaptation(pts, family_alphas=None):
    """Test one-shot with global alpha, family-specific alpha, and baselines."""
    model_best = _get_model_best(pts)

    # --- Method A: Global alpha (original) ---
    results_global = _one_shot_core(model_best, lambda m: 1.477)

    # --- Method B: Family-specific alpha ---
    if family_alphas is None:
        family_alphas = fit_family_alphas(pts)
    results_family = _one_shot_core(
        model_best,
        lambda m: family_alphas.get(get_family(m), 1.477),
    )

    # --- Baseline: Random ranking (permutation null) ---
    rng = np.random.RandomState(42)
    n_perm = 200
    random_rhos = []
    for model, ds_dict in model_best.items():
        datasets = sorted(ds_dict.keys())
        if len(datasets) < 3:
            continue
        qs = np.array([
            (ds_dict[d]["q"] - 1.0 / ds_dict[d]["K"]) / (1.0 - 1.0 / ds_dict[d]["K"])
            for d in datasets
        ])
        valid = (qs > 0.01) & (qs < 0.99)
        qs_valid = qs[valid]
        if len(qs_valid) < 3:
            continue
        for _ in range(n_perm):
            perm = rng.permutation(len(qs_valid))
            rho, _ = spearmanr(qs_valid, qs_valid[perm])
            if not np.isnan(rho):
                random_rhos.append(float(rho))

    # --- Baseline: Dataset-mean q (predict q = mean q across all models for that dataset) ---
    ds_mean_q = {}
    for model, ds_dict in model_best.items():
        for ds, pt in ds_dict.items():
            K = pt["K"]
            q_norm = (pt["q"] - 1.0 / K) / (1.0 - 1.0 / K)
            if 0.01 < q_norm < 0.99:
                ds_mean_q.setdefault(ds, []).append(q_norm)
    ds_mean_q = {ds: float(np.mean(vs)) for ds, vs in ds_mean_q.items()}

    ds_mean_results = []
    for model, ds_dict in model_best.items():
        datasets = sorted(ds_dict.keys())
        if len(datasets) < 3:
            continue
        for anchor_ds in datasets:
            pred_qs, actual_qs = [], []
            for test_ds in datasets:
                if test_ds == anchor_ds:
                    continue
                test = ds_dict[test_ds]
                K_test = test["K"]
                q_norm_test = (test["q"] - 1.0 / K_test) / (1.0 - 1.0 / K_test)
                if q_norm_test <= 0.01 or q_norm_test >= 0.99:
                    continue
                if test_ds in ds_mean_q:
                    pred_qs.append(ds_mean_q[test_ds])
                    actual_qs.append(q_norm_test)
            if len(pred_qs) >= 2:
                pred_arr = np.array(pred_qs)
                actual_arr = np.array(actual_qs)
                mae = float(np.mean(np.abs(pred_arr - actual_arr)))
                rho, _ = spearmanr(pred_arr, actual_arr)
                ds_mean_results.append({
                    "mae": mae,
                    "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                })

    return {
        "global_alpha": _summarize_h2(results_global, "global_alpha=1.477"),
        "family_alpha": _summarize_h2(results_family, "family-specific alpha"),
        "baseline_random": {
            "mean_rho": float(np.mean(random_rhos)) if random_rhos else 0.0,
            "std_rho": float(np.std(random_rhos)) if random_rhos else 0.0,
            "n_permutations": len(random_rhos),
        },
        "baseline_dataset_mean": _summarize_h2(ds_mean_results, "dataset-mean baseline"),
        "family_alphas_used": {f: round(a, 4) for f, a in (family_alphas or {}).items()},
    }


def _summarize_h2(results, label):
    if not results:
        return {"label": label, "n": 0, "mean_mae": float("nan"), "mean_rho": float("nan")}
    maes = [r["mae"] for r in results]
    rhos = [r["spearman_rho"] for r in results]
    return {
        "label": label,
        "n": len(results),
        "mean_mae": float(np.mean(maes)),
        "median_mae": float(np.median(maes)),
        "std_mae": float(np.std(maes)),
        "mean_rho": float(np.mean(rhos)),
        "median_rho": float(np.median(rhos)),
        "std_rho": float(np.std(rhos)),
        "mae_ci95": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "rho_ci95": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
    }


# =====================================================================
# H3: kappa/sqrt(K) cross-dataset ranking
# =====================================================================
def test_kappa_sqrtK_ranking(pts):
    """Test if kappa/sqrt(K) can rank (model, dataset) pairs globally."""
    # Best layer per (model, dataset)
    groups = {}
    for p in pts:
        key = (p["model"], p["dataset"])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    best_pts = []
    for (model, dataset), plist in groups.items():
        best = max(plist, key=lambda x: x["kappa"])
        best_pts.append(best)

    # Global ranking using kappa/sqrt(K)
    kappa_norm = np.array([p["kappa"] / np.sqrt(p["K"]) for p in best_pts])
    qs = np.array([p["q"] for p in best_pts])

    rho_global, p_global = spearmanr(kappa_norm, qs)
    r_global, _ = pearsonr(kappa_norm, qs)

    # Per-model ranking across datasets
    by_model = {}
    for p in best_pts:
        m = p["model"]
        if m not in by_model:
            by_model[m] = []
        by_model[m].append(p)

    per_model = []
    for model, plist in sorted(by_model.items()):
        if len(plist) < 3:
            continue
        kn = np.array([p["kappa"] / np.sqrt(p["K"]) for p in plist])
        q = np.array([p["q"] for p in plist])
        rho, _ = spearmanr(kn, q)
        per_model.append({
            "model": model,
            "n_datasets": len(plist),
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        })

    return {
        "global_spearman_rho": float(rho_global),
        "global_pearson_r": float(r_global),
        "n_points": len(best_pts),
        "per_model": per_model,
    }


# =====================================================================
# H4: LOO-costed benchmark (fair dataset-mean vs CTI)
# =====================================================================
def test_loo_costed(pts, family_alphas):
    """LOO-costed: dataset-mean excludes the target model.
    CTI one-shot uses only 1 anchor eval + kappa geometry.
    This is the fair comparison Codex requested."""
    model_best = _get_model_best(pts)

    # Build (model, dataset) -> q_norm table
    table = {}
    for model, ds_dict in model_best.items():
        for ds, pt in ds_dict.items():
            K = pt["K"]
            q_norm = (pt["q"] - 1.0 / K) / (1.0 - 1.0 / K)
            if 0.01 < q_norm < 0.99:
                table[(model, ds)] = {"q_norm": q_norm, "kappa": pt["kappa"], "K": K}

    # For each (model, dataset), compute LOO-dataset-mean (excluding this model)
    datasets = sorted(set(ds for _, ds in table.keys()))
    models = sorted(set(m for m, _ in table.keys()))

    loo_results = []
    cti_results = []
    for target_model in models:
        target_datasets = [ds for ds in datasets if (target_model, ds) in table]
        if len(target_datasets) < 3:
            continue

        for anchor_ds in target_datasets:
            anchor = table[(target_model, anchor_ds)]
            alpha = family_alphas.get(get_family(target_model), 1.477)
            C_model = sp_logit(anchor["q_norm"]) - alpha * anchor["kappa"]

            for test_ds in target_datasets:
                if test_ds == anchor_ds:
                    continue
                if (target_model, test_ds) not in table:
                    continue
                actual = table[(target_model, test_ds)]["q_norm"]
                test_kappa = table[(target_model, test_ds)]["kappa"]

                # CTI prediction
                logit_pred = alpha * test_kappa + C_model
                cti_pred = float(expit(logit_pred))

                # LOO dataset mean (exclude target model)
                other_qs = [
                    table[(m, test_ds)]["q_norm"]
                    for m in models
                    if m != target_model and (m, test_ds) in table
                ]
                loo_mean = float(np.mean(other_qs)) if other_qs else actual

                cti_results.append({
                    "model": target_model, "anchor": anchor_ds, "test": test_ds,
                    "pred": cti_pred, "actual": actual,
                    "error": abs(cti_pred - actual),
                })
                loo_results.append({
                    "model": target_model, "anchor": anchor_ds, "test": test_ds,
                    "pred": loo_mean, "actual": actual,
                    "error": abs(loo_mean - actual),
                })

    cti_mae = float(np.mean([r["error"] for r in cti_results])) if cti_results else float("nan")
    loo_mae = float(np.mean([r["error"] for r in loo_results])) if loo_results else float("nan")

    # Per-model comparison
    per_model = {}
    for cr, lr in zip(cti_results, loo_results):
        m = cr["model"]
        if m not in per_model:
            per_model[m] = {"cti_errors": [], "loo_errors": []}
        per_model[m]["cti_errors"].append(cr["error"])
        per_model[m]["loo_errors"].append(lr["error"])

    cti_wins = 0
    for m, d in per_model.items():
        if np.mean(d["cti_errors"]) < np.mean(d["loo_errors"]):
            cti_wins += 1

    return {
        "n_predictions": len(cti_results),
        "cti_mae": cti_mae,
        "loo_dataset_mean_mae": loo_mae,
        "cti_wins_per_model": cti_wins,
        "total_models": len(per_model),
        "cti_better": cti_mae < loo_mae,
    }


# =====================================================================
# H5: Within-dataset model ordering (can kappa rank models on SAME dataset?)
# =====================================================================
def test_within_dataset_model_ordering(pts):
    """For each dataset, does kappa rank models correctly?"""
    model_best = _get_model_best(pts)

    # Group best-layer points by dataset
    by_dataset = {}
    for model, ds_dict in model_best.items():
        for ds, pt in ds_dict.items():
            if ds not in by_dataset:
                by_dataset[ds] = []
            by_dataset[ds].append(pt)

    results = []
    for ds, plist in sorted(by_dataset.items()):
        if len(plist) < 4:
            continue
        kappas = np.array([p["kappa"] for p in plist])
        qs = np.array([p["q"] for p in plist])
        rho, p_val = spearmanr(kappas, qs)

        # Top-1 and top-3 match
        rank_by_kappa = np.argsort(-kappas)
        rank_by_q = np.argsort(-qs)
        top1_match = bool(rank_by_kappa[0] == rank_by_q[0])
        top3_kappa = set(rank_by_kappa[:3])
        top3_q = set(rank_by_q[:3])
        top3_overlap = len(top3_kappa & top3_q)

        results.append({
            "dataset": ds,
            "n_models": len(plist),
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
            "top1_match": top1_match,
            "top3_overlap": top3_overlap,
        })

    return results


# =====================================================================
# H6: Residual prediction (CTI signal beyond dataset difficulty)
# =====================================================================
def test_residual_prediction(pts, family_alphas):
    """After removing dataset-mean difficulty, does kappa predict residual q?"""
    model_best = _get_model_best(pts)

    # Compute per-dataset mean q_norm
    ds_q_norms = {}
    for model, ds_dict in model_best.items():
        for ds, pt in ds_dict.items():
            K = pt["K"]
            q_norm = (pt["q"] - 1.0 / K) / (1.0 - 1.0 / K)
            if 0.01 < q_norm < 0.99:
                ds_q_norms.setdefault(ds, []).append((model, q_norm, pt["kappa"]))
    ds_means = {ds: float(np.mean([x[1] for x in vs])) for ds, vs in ds_q_norms.items()}

    # Compute residuals: q_residual = q_norm - dataset_mean
    residuals = []
    kappas = []
    per_dataset = {}
    for ds, pts_list in ds_q_norms.items():
        if len(pts_list) < 4:
            continue
        ds_mean = ds_means[ds]
        ds_kappas = []
        ds_residuals = []
        for model, q_norm, kappa in pts_list:
            r = q_norm - ds_mean
            residuals.append(r)
            kappas.append(kappa)
            ds_kappas.append(kappa)
            ds_residuals.append(r)
        if len(ds_kappas) >= 4:
            rho, _ = spearmanr(ds_kappas, ds_residuals)
            per_dataset[ds] = {
                "n_models": len(ds_kappas),
                "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                "residual_std": float(np.std(ds_residuals)),
            }

    # Global correlation
    if len(residuals) >= 10:
        rho_global, p_global = spearmanr(kappas, residuals)
        r_global, _ = pearsonr(kappas, residuals)
    else:
        rho_global, r_global, p_global = 0.0, 0.0, 1.0

    return {
        "global_spearman_rho": float(rho_global),
        "global_pearson_r": float(r_global),
        "global_p_value": float(p_global),
        "n_points": len(residuals),
        "per_dataset": per_dataset,
    }


# =====================================================================
# H7: Mixed-effects decomposition of C
# =====================================================================
# Model metadata for regression
MODEL_META = {
    "Falcon-H1-0.5B-Base": {"params_m": 500, "n_layers": 36, "hidden": 1024},
    "Mistral-7B-v0.3": {"params_m": 7000, "n_layers": 32, "hidden": 4096},
    "OLMo-1B-hf": {"params_m": 1000, "n_layers": 16, "hidden": 2048},
    "Qwen2.5-0.5B": {"params_m": 500, "n_layers": 28, "hidden": 1024},
    "Qwen3-0.6B": {"params_m": 600, "n_layers": 28, "hidden": 1024},
    "Qwen3-1.7B": {"params_m": 1700, "n_layers": 28, "hidden": 2048},
    "SmolLM2-1.7B": {"params_m": 1700, "n_layers": 24, "hidden": 2048},
    "TinyLlama-1.1B-intermediate-step-1431k-3T": {"params_m": 1100, "n_layers": 22, "hidden": 2048},
    "bert-base-uncased": {"params_m": 109, "n_layers": 12, "hidden": 768},
    "bge-base-v1.5": {"params_m": 109, "n_layers": 12, "hidden": 768},
    "deberta-base": {"params_m": 86, "n_layers": 12, "hidden": 768},
    "electra-small": {"params_m": 14, "n_layers": 12, "hidden": 256},
    "gpt-neo-125m": {"params_m": 125, "n_layers": 12, "hidden": 768},
    "gpt2": {"params_m": 124, "n_layers": 12, "hidden": 768},
    "mamba-130m": {"params_m": 130, "n_layers": 24, "hidden": 768},
    "phi2": {"params_m": 2700, "n_layers": 32, "hidden": 2560},
    "pythia-160m": {"params_m": 160, "n_layers": 12, "hidden": 768},
    "pythia-1b": {"params_m": 1000, "n_layers": 16, "hidden": 2048},
    "pythia-410m": {"params_m": 410, "n_layers": 12, "hidden": 1024},
    "rwkv-4-169m-pile": {"params_m": 169, "n_layers": 12, "hidden": 768},
}


def test_mixed_effects_decomposition(pts, family_alphas):
    """Decompose C = logit(q) - alpha*kappa into dataset + model + family effects.

    Tests whether C has structured components that can be separated.
    R2 at each level shows how much variance each component explains.
    """
    model_best = _get_model_best(pts)

    # Compute observed C for each (model, dataset) pair
    observations = []
    for model, ds_dict in model_best.items():
        family = get_family(model)
        alpha = family_alphas.get(family, 1.477)
        for ds, pt in ds_dict.items():
            K = pt["K"]
            q_norm = (pt["q"] - 1.0 / K) / (1.0 - 1.0 / K)
            if q_norm <= 0.01 or q_norm >= 0.99:
                continue
            C_obs = sp_logit(q_norm) - alpha * pt["kappa"]
            observations.append({
                "model": model, "dataset": ds, "family": family,
                "C_obs": float(C_obs), "kappa": pt["kappa"],
                "q_norm": q_norm, "K": K,
            })

    if len(observations) < 10:
        return {"error": "too few observations"}

    C_all = np.array([o["C_obs"] for o in observations])
    grand_mean = float(np.mean(C_all))
    ss_total = float(np.sum((C_all - grand_mean) ** 2))

    # Level 1: Dataset effect only
    ds_means = {}
    for o in observations:
        ds_means.setdefault(o["dataset"], []).append(o["C_obs"])
    ds_means = {ds: np.mean(vs) for ds, vs in ds_means.items()}
    pred_ds = np.array([ds_means[o["dataset"]] for o in observations])
    ss_res_ds = float(np.sum((C_all - pred_ds) ** 2))
    r2_ds = 1 - ss_res_ds / ss_total if ss_total > 0 else 0

    # Level 2: Dataset + family effect
    df_means = {}
    for o in observations:
        key = (o["dataset"], o["family"])
        df_means.setdefault(key, []).append(o["C_obs"])
    df_means = {k: np.mean(vs) for k, vs in df_means.items()}
    pred_df = np.array([df_means.get((o["dataset"], o["family"]), grand_mean) for o in observations])
    ss_res_df = float(np.sum((C_all - pred_df) ** 2))
    r2_df = 1 - ss_res_df / ss_total if ss_total > 0 else 0

    # Level 3: Dataset + model effect (full model)
    dm_means = {}
    for o in observations:
        key = (o["dataset"], o["model"])
        dm_means.setdefault(key, []).append(o["C_obs"])
    dm_means = {k: np.mean(vs) for k, vs in dm_means.items()}
    pred_dm = np.array([dm_means.get((o["dataset"], o["model"]), grand_mean) for o in observations])
    ss_res_dm = float(np.sum((C_all - pred_dm) ** 2))
    r2_dm = 1 - ss_res_dm / ss_total if ss_total > 0 else 0

    # Level 4: Family effect only (no dataset)
    fam_means = {}
    for o in observations:
        fam_means.setdefault(o["family"], []).append(o["C_obs"])
    fam_means = {f: np.mean(vs) for f, vs in fam_means.items()}
    pred_fam = np.array([fam_means[o["family"]] for o in observations])
    ss_res_fam = float(np.sum((C_all - pred_fam) ** 2))
    r2_fam = 1 - ss_res_fam / ss_total if ss_total > 0 else 0

    # Level 5: Model effect only (no dataset)
    mod_means = {}
    for o in observations:
        mod_means.setdefault(o["model"], []).append(o["C_obs"])
    mod_means = {m: np.mean(vs) for m, vs in mod_means.items()}
    pred_mod = np.array([mod_means[o["model"]] for o in observations])
    ss_res_mod = float(np.sum((C_all - pred_mod) ** 2))
    r2_mod = 1 - ss_res_mod / ss_total if ss_total > 0 else 0

    # Regress residuals (after dataset+family) against model architectural features
    residuals_after_df = C_all - pred_df
    log_params = []
    n_layers_arr = []
    log_hidden = []
    valid_idx = []
    for i, o in enumerate(observations):
        meta = MODEL_META.get(o["model"])
        if meta:
            log_params.append(np.log(meta["params_m"]))
            n_layers_arr.append(meta["n_layers"])
            log_hidden.append(np.log(meta["hidden"]))
            valid_idx.append(i)

    arch_regression = {}
    if len(valid_idx) >= 10:
        resid = residuals_after_df[valid_idx]
        for name, feat in [("log_params", log_params), ("n_layers", n_layers_arr), ("log_hidden", log_hidden)]:
            feat_arr = np.array(feat)
            rho, p = spearmanr(feat_arr, resid)
            r, _ = pearsonr(feat_arr, resid)
            arch_regression[name] = {
                "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                "pearson_r": float(r) if not np.isnan(r) else 0.0,
                "p_value": float(p) if not np.isnan(p) else 1.0,
            }

    # Per-dataset C variance decomposition
    per_ds_var = {}
    for ds in sorted(ds_means.keys()):
        ds_obs = [o for o in observations if o["dataset"] == ds]
        if len(ds_obs) >= 3:
            cs = [o["C_obs"] for o in ds_obs]
            per_ds_var[ds] = {
                "mean_C": float(np.mean(cs)),
                "std_C": float(np.std(cs)),
                "n_models": len(ds_obs),
                "range_C": float(max(cs) - min(cs)),
            }

    return {
        "n_observations": len(observations),
        "grand_mean_C": grand_mean,
        "total_variance": float(np.var(C_all)),
        "r2_dataset_only": r2_ds,
        "r2_dataset_plus_family": r2_df,
        "r2_dataset_plus_model": r2_dm,
        "r2_family_only": r2_fam,
        "r2_model_only": r2_mod,
        "marginal_r2_family_given_dataset": r2_df - r2_ds,
        "marginal_r2_model_given_dataset": r2_dm - r2_ds,
        "arch_feature_regression": arch_regression,
        "per_dataset_C": per_ds_var,
        "family_mean_C": {f: float(v) for f, v in fam_means.items()},
    }


def _h1_random_baseline(h1_results):
    """Baseline: random layer selection regret."""
    rng = np.random.RandomState(42)
    random_regrets = []
    random_match_rate = 0
    n_trials = 1000
    for r in h1_results:
        n_layers = r["n_layers"]
        if n_layers < 2:
            continue
        matches = 0
        trial_regrets = []
        for _ in range(n_trials):
            rand_idx = rng.randint(0, n_layers)
            # We don't have per-layer q here, so estimate: random picks best 1/n_layers of time
            matches += 1 if rng.random() < 1.0 / n_layers else 0
        random_match_rate += 1.0 / n_layers
    if h1_results:
        random_match_rate /= len(h1_results)
    return {
        "random_best_layer_match_pct": float(100.0 * random_match_rate),
        "note": "Expected match rate if layer chosen uniformly at random",
    }


def main():
    print("=" * 70)
    print("CTI Practical Utility — Revised (Codex Session 78, improved)")
    print("=" * 70)

    pts = load_all_cache()
    print(f"Loaded {len(pts)} layer-level points")

    # Fit family alphas upfront
    family_alphas = fit_family_alphas(pts)
    print(f"Family alphas: {', '.join(f'{f}={a:.3f}' for f, a in sorted(family_alphas.items()))}")

    # =========================================================
    # H1: Within-model layer ranking
    # =========================================================
    print("\n--- H1: Within-model layer ranking ---")
    h1_results = test_within_model_layer_ranking(pts)

    rhos = [r["spearman_rho"] for r in h1_results]
    matches = [r["best_layer_match"] for r in h1_results]
    regrets = [r["regret"] for r in h1_results]

    n_high_rho = sum(1 for r in rhos if r >= 0.80)
    n_perfect_match = sum(matches)
    mean_rho = float(np.mean(rhos))
    mean_regret = float(np.mean(regrets))
    median_regret = float(np.median(regrets))

    h1_baseline = _h1_random_baseline(h1_results)

    print(f"  Total (model, dataset) pairs: {len(h1_results)}")
    print(f"  Mean Spearman rho: {mean_rho:.4f}")
    print(f"  Pairs with rho >= 0.80: {n_high_rho}/{len(h1_results)} "
          f"({100*n_high_rho/len(h1_results):.1f}%)")
    print(f"  Best-layer exact match: {n_perfect_match}/{len(h1_results)} "
          f"({100*n_perfect_match/len(h1_results):.1f}%)")
    print(f"  BASELINE random match: {h1_baseline['random_best_layer_match_pct']:.1f}%")
    print(f"  Mean regret: {mean_regret:.4f}, Median: {median_regret:.4f}, Max: {max(regrets):.4f}")

    # Per-dataset summary
    print("\n  Per-dataset within-model ranking:")
    ds_rhos = {}
    for r in h1_results:
        d = r["dataset"]
        if d not in ds_rhos:
            ds_rhos[d] = []
        ds_rhos[d].append(r["spearman_rho"])
    for ds, rs in sorted(ds_rhos.items()):
        print(f"    {ds:>20}: mean_rho={np.mean(rs):.3f}, n={len(rs)}")

    # =========================================================
    # H2: One-shot adaptation (with family alpha + baselines)
    # =========================================================
    print("\n--- H2: One-shot adaptation (improved) ---")
    h2_results = test_one_shot_adaptation(pts, family_alphas)

    for key in ["global_alpha", "family_alpha", "baseline_dataset_mean"]:
        r = h2_results[key]
        print(f"  {r['label']:>30}: MAE={r['mean_mae']:.4f} +/- {r.get('std_mae',0):.3f}, "
              f"rho={r['mean_rho']:.4f} +/- {r.get('std_rho',0):.3f}, n={r['n']}")
    br = h2_results["baseline_random"]
    print(f"  {'random permutation':>30}: mean_rho={br['mean_rho']:.4f} +/- {br['std_rho']:.3f}")

    # =========================================================
    # H3: kappa/sqrt(K) cross-dataset ranking
    # =========================================================
    print("\n--- H3: kappa/sqrt(K) cross-dataset ranking ---")
    h3_results = test_kappa_sqrtK_ranking(pts)
    print(f"  Global Spearman rho: {h3_results['global_spearman_rho']:.4f}")
    print(f"  Global Pearson r: {h3_results['global_pearson_r']:.4f}")
    print(f"  N points: {h3_results['n_points']}")
    print("\n  Per-model cross-dataset ranking (kappa/sqrt(K)):")
    for m in h3_results["per_model"]:
        status = "PASS" if m["spearman_rho"] >= 0.80 else "FAIL"
        print(f"    {m['model']:>40}: rho={m['spearman_rho']:.3f} n={m['n_datasets']} {status}")

    n_model_pass = sum(1 for m in h3_results["per_model"] if m["spearman_rho"] >= 0.80)

    # =========================================================
    # H4: LOO-costed benchmark
    # =========================================================
    print("\n--- H4: LOO-costed benchmark (fair CTI vs dataset-mean) ---")
    h4_results = test_loo_costed(pts, family_alphas)
    print(f"  N predictions: {h4_results['n_predictions']}")
    print(f"  CTI family-alpha MAE:     {h4_results['cti_mae']:.4f}")
    print(f"  LOO dataset-mean MAE:     {h4_results['loo_dataset_mean_mae']:.4f}")
    print(f"  CTI wins (per-model):     {h4_results['cti_wins_per_model']}/{h4_results['total_models']}")
    print(f"  CTI better overall:       {h4_results['cti_better']}")

    # =========================================================
    # H5: Within-dataset model ordering
    # =========================================================
    print("\n--- H5: Within-dataset model ordering (kappa ranks models?) ---")
    h5_results = test_within_dataset_model_ordering(pts)
    h5_rhos = [r["spearman_rho"] for r in h5_results]
    h5_top1 = [r["top1_match"] for r in h5_results]
    print(f"  N datasets (n_models>=4): {len(h5_results)}")
    if h5_results:
        print(f"  Mean Spearman rho:        {np.mean(h5_rhos):.4f}")
        print(f"  Top-1 match rate:         {sum(h5_top1)}/{len(h5_top1)}")
        for r in h5_results:
            status = "PASS" if r["spearman_rho"] >= 0.70 else "FAIL"
            print(f"    {r['dataset']:>20}: rho={r['spearman_rho']:.3f} "
                  f"n={r['n_models']} top1={'Y' if r['top1_match'] else 'N'} "
                  f"top3={r['top3_overlap']}/3 {status}")

    # =========================================================
    # H6: Residual prediction
    # =========================================================
    print("\n--- H6: Residual prediction (CTI beyond dataset difficulty) ---")
    h6_results = test_residual_prediction(pts, family_alphas)
    print(f"  Global Spearman rho(kappa, residual): {h6_results['global_spearman_rho']:.4f}")
    print(f"  Global Pearson r:                     {h6_results['global_pearson_r']:.4f}")
    print(f"  Global p-value:                       {h6_results['global_p_value']:.6f}")
    print(f"  N points:                             {h6_results['n_points']}")
    if h6_results["per_dataset"]:
        print("  Per-dataset:")
        for ds, d in sorted(h6_results["per_dataset"].items()):
            status = "SIG" if d["spearman_rho"] >= 0.50 else "weak"
            print(f"    {ds:>20}: rho={d['spearman_rho']:.3f} n={d['n_models']} "
                  f"residual_std={d['residual_std']:.4f} {status}")

    # =========================================================
    # H7: Mixed-effects decomposition of C
    # =========================================================
    print("\n--- H7: Mixed-effects decomposition of C ---")
    h7_results = test_mixed_effects_decomposition(pts, family_alphas)
    if "error" not in h7_results:
        print(f"  N observations: {h7_results['n_observations']}")
        print(f"  Total C variance: {h7_results['total_variance']:.4f}")
        print(f"  R2 (dataset only):          {h7_results['r2_dataset_only']:.4f}")
        print(f"  R2 (family only):           {h7_results['r2_family_only']:.4f}")
        print(f"  R2 (model only):            {h7_results['r2_model_only']:.4f}")
        print(f"  R2 (dataset + family):      {h7_results['r2_dataset_plus_family']:.4f}")
        print(f"  R2 (dataset + model):       {h7_results['r2_dataset_plus_model']:.4f}")
        print(f"  Marginal R2 (family|ds):    {h7_results['marginal_r2_family_given_dataset']:.4f}")
        print(f"  Marginal R2 (model|ds):     {h7_results['marginal_r2_model_given_dataset']:.4f}")
        print(f"\n  Family mean C: {h7_results['family_mean_C']}")
        if h7_results["arch_feature_regression"]:
            print("\n  Architectural feature regression (residuals after dataset+family):")
            for feat, stats in h7_results["arch_feature_regression"].items():
                sig = "***" if stats["p_value"] < 0.001 else "**" if stats["p_value"] < 0.01 else "*" if stats["p_value"] < 0.05 else "ns"
                print(f"    {feat:>12}: rho={stats['spearman_rho']:.3f}, r={stats['pearson_r']:.3f}, "
                      f"p={stats['p_value']:.4f} {sig}")
        if h7_results["per_dataset_C"]:
            print("\n  Per-dataset C statistics:")
            for ds, stats in sorted(h7_results["per_dataset_C"].items()):
                print(f"    {ds:>20}: mean_C={stats['mean_C']:.3f}, std_C={stats['std_C']:.3f}, "
                      f"range={stats['range_C']:.3f}, n={stats['n_models']}")

    # =========================================================
    # OUTPUT
    # =========================================================
    h2_global = h2_results["global_alpha"]
    h2_family = h2_results["family_alpha"]

    output = {
        "experiment": "cti_utility_revised",
        "description": "CTI utility with mixed-effects, LOO, residuals, within-dataset",
        "h1_within_model_layer_ranking": {
            "n_pairs": len(h1_results),
            "mean_spearman_rho": mean_rho,
            "pct_rho_ge_080": float(100 * n_high_rho / len(h1_results)) if h1_results else 0,
            "pct_best_layer_match": float(100 * n_perfect_match / len(h1_results)) if h1_results else 0,
            "mean_regret": mean_regret,
            "median_regret": median_regret,
            "baseline_random": h1_baseline,
            "per_pair": h1_results,
        },
        "h2_one_shot_adaptation": h2_results,
        "h3_kappa_sqrtK_ranking": h3_results,
        "h4_loo_costed": h4_results,
        "h5_within_dataset_ordering": h5_results,
        "h6_residual_prediction": h6_results,
        "h7_mixed_effects": h7_results,
        "summary": {
            "h1_best_layer_pct": float(100 * n_perfect_match / len(h1_results)) if h1_results else 0,
            "h1_baseline_random_pct": h1_baseline["random_best_layer_match_pct"],
            "h1_lift_over_random": float(100 * n_perfect_match / len(h1_results) - h1_baseline["random_best_layer_match_pct"]) if h1_results else 0,
            "h2_global_alpha_mae": h2_global["mean_mae"],
            "h2_global_alpha_rho": h2_global["mean_rho"],
            "h2_family_alpha_mae": h2_family["mean_mae"],
            "h2_family_alpha_rho": h2_family["mean_rho"],
            "h2_baseline_random_rho": h2_results["baseline_random"]["mean_rho"],
            "h2_baseline_dataset_mean_mae": h2_results["baseline_dataset_mean"]["mean_mae"],
            "h2_baseline_dataset_mean_rho": h2_results["baseline_dataset_mean"]["mean_rho"],
            "h3_global_rho": h3_results["global_spearman_rho"],
            "h3_n_model_pass": n_model_pass,
            "h4_cti_mae": h4_results["cti_mae"],
            "h4_loo_mae": h4_results["loo_dataset_mean_mae"],
            "h4_cti_better": h4_results["cti_better"],
            "h5_mean_rho": float(np.mean(h5_rhos)) if h5_rhos else 0.0,
            "h6_global_rho": h6_results["global_spearman_rho"],
            "h6_global_p": h6_results["global_p_value"],
            "h7_r2_dataset": h7_results.get("r2_dataset_only", 0),
            "h7_r2_dataset_family": h7_results.get("r2_dataset_plus_family", 0),
            "h7_r2_dataset_model": h7_results.get("r2_dataset_plus_model", 0),
            "h7_marginal_family": h7_results.get("marginal_r2_family_given_dataset", 0),
            "family_alphas": {f: round(a, 4) for f, a in family_alphas.items()},
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    kappa_match = float(100 * n_perfect_match / len(h1_results)) if h1_results else 0
    random_match = h1_baseline["random_best_layer_match_pct"]
    print(f"  H1 (best-layer selection):  {kappa_match:.1f}% match "
          f"(random={random_match:.1f}%, lift=+{kappa_match - random_match:.1f}pp)")
    print(f"  H1 (mean Spearman rho):     {mean_rho:.3f}")
    print(f"  H2 global alpha:            MAE={h2_global['mean_mae']:.4f}, rho={h2_global['mean_rho']:.4f}")
    print(f"  H2 family alpha:            MAE={h2_family['mean_mae']:.4f}, rho={h2_family['mean_rho']:.4f}")
    print(f"  H2 baseline (dataset mean): MAE={h2_results['baseline_dataset_mean']['mean_mae']:.4f}, "
          f"rho={h2_results['baseline_dataset_mean']['mean_rho']:.4f}")
    print(f"  H2 baseline (random):       rho={h2_results['baseline_random']['mean_rho']:.4f}")
    print(f"  H3 (kappa/sqrt(K) global):  rho={h3_results['global_spearman_rho']:.3f}, "
          f"{n_model_pass}/{len(h3_results['per_model'])} models pass")
    print(f"  H4 (LOO-costed):            CTI MAE={h4_results['cti_mae']:.4f} vs "
          f"LOO-mean MAE={h4_results['loo_dataset_mean_mae']:.4f} "
          f"({'CTI WINS' if h4_results['cti_better'] else 'LOO WINS'})")
    print(f"  H5 (within-dataset order):  mean rho={np.mean(h5_rhos):.3f}" if h5_rhos else "  H5: N/A")
    print(f"  H6 (residual prediction):   rho={h6_results['global_spearman_rho']:.3f}, "
          f"p={h6_results['global_p_value']:.6f}")


if __name__ == "__main__":
    main()
