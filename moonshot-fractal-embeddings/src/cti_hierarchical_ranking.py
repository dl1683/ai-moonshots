"""
Codex Experiment 1: Hierarchical calibration for architecture ranking.

Diagnoses WHY kappa ranking fails (3/10) and fixes it with:
1. Best-layer selection (not mean across layers)
2. Family-aware alpha correction (encoder vs decoder alpha differs ~3-5x)
3. Regime diagnosis (small-K compression, saturation effects)
4. Calibrated predicted-q ranking (logit(q) = alpha_family * kappa + C_dataset)

Codex design: hierarchical partial-pooling where alpha is family-specific,
C is dataset-specific, and ranking uses predicted logit(q) not raw kappa.

Success criteria (Codex):
  - Pass rate increases from 3/10 to >= 7/10 at Spearman rho >= 0.80
  - agnews and emotion Spearman increase by >= +0.25 absolute each
"""

import json
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.special import expit, logit as sp_logit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_hierarchical_ranking.json")

SPEARMAN_THRESHOLD = 0.80

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


def load_cache():
    """Load all valid points from kappa_near_cache files with layer info."""
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


def best_layer_per_model_dataset(pts):
    """For each (model, dataset), pick the layer with highest kappa."""
    groups = {}
    for p in pts:
        key = (p["model"], p["dataset"])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    best_pts = []
    for (model, dataset), plist in groups.items():
        best = max(plist, key=lambda x: x["kappa"])
        best["n_layers"] = len(plist)
        best_pts.append(best)
    return best_pts


def fit_family_alpha(pts):
    """Fit alpha per family using logit(q_norm) = alpha * kappa + C_dataset."""
    from collections import defaultdict

    family_data = defaultdict(list)
    for p in pts:
        K = p["K"]
        q_norm = (p["q"] - 1.0/K) / (1.0 - 1.0/K)
        if q_norm <= 0.01 or q_norm >= 0.99:
            continue
        family_data[p["family"]].append({
            "logit_q": float(sp_logit(q_norm)),
            "kappa": p["kappa"],
            "dataset": p["dataset"],
            "model": p["model"],
        })

    alphas = {}
    for family, data in family_data.items():
        if len(data) < 5:
            alphas[family] = {"alpha": 1.477, "n": len(data), "note": "too few, using default"}
            continue

        # Group by dataset for per-dataset intercept
        datasets = sorted(set(d["dataset"] for d in data))
        ds_map = {ds: i for i, ds in enumerate(datasets)}
        n_ds = len(datasets)

        # Build design matrix: logit(q) = alpha * kappa + C_d
        X = np.zeros((len(data), 1 + n_ds))
        y = np.zeros(len(data))
        for i, d in enumerate(data):
            X[i, 0] = d["kappa"]
            X[i, 1 + ds_map[d["dataset"]]] = 1.0
            y[i] = d["logit_q"]

        # OLS
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            alpha = float(beta[0])
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            intercepts = {ds: float(beta[1 + ds_map[ds]]) for ds in datasets}

            alphas[family] = {
                "alpha": alpha,
                "r2": r2,
                "n": len(data),
                "n_datasets": n_ds,
                "intercepts": intercepts,
            }
        except Exception as e:
            alphas[family] = {"alpha": 1.477, "n": len(data), "note": f"fit failed: {e}"}

    return alphas


def predict_logit_q(kappa, family, dataset, family_alphas):
    """Predict logit(q_norm) using family-specific alpha + dataset intercept."""
    fa = family_alphas.get(family, family_alphas.get("decoder", {"alpha": 1.477, "intercepts": {}}))
    alpha = fa["alpha"]
    C = fa.get("intercepts", {}).get(dataset, 0.0)
    return alpha * kappa + C


def rank_and_evaluate(pts, family_alphas, method_name):
    """Rank architectures per dataset and compute Spearman rho."""
    by_dataset = {}
    for p in pts:
        d = p["dataset"]
        if d not in by_dataset:
            by_dataset[d] = []
        by_dataset[d].append(p)

    results = {}
    spearman_rhos = []

    for dataset, plist in sorted(by_dataset.items()):
        if len(plist) < 5:
            continue

        K = plist[0]["K"]
        qs = np.array([p["q"] for p in plist])
        kappas = np.array([p["kappa"] for p in plist])
        models = [p["model"] for p in plist]
        families = [p["family"] for p in plist]

        # Predicted ranking score
        scores = np.array([
            predict_logit_q(p["kappa"], p["family"], dataset, family_alphas)
            for p in plist
        ])

        rho_pred, p_pred = spearmanr(scores, qs)
        rho_raw, p_raw = spearmanr(kappas, qs)

        results[dataset] = {
            "K": K,
            "n_architectures": len(plist),
            "spearman_rho_calibrated": float(rho_pred),
            "spearman_rho_raw_kappa": float(rho_raw),
            "improvement": float(rho_pred - rho_raw),
            "pr_pass_calibrated": bool(abs(rho_pred) >= SPEARMAN_THRESHOLD),
            "pr_pass_raw": bool(abs(rho_raw) >= SPEARMAN_THRESHOLD),
            "families_present": sorted(set(families)),
            "n_families": len(set(families)),
        }
        spearman_rhos.append(rho_pred)

    n_pass = sum(1 for d in results.values() if d["pr_pass_calibrated"])
    n_total = len(results)
    mean_rho = float(np.mean(spearman_rhos)) if spearman_rhos else 0.0

    return {
        "method": method_name,
        "n_datasets": n_total,
        "mean_spearman_rho": mean_rho,
        "n_pass": n_pass,
        "pr_mean_pass": bool(mean_rho >= SPEARMAN_THRESHOLD),
        "per_dataset": results,
    }


def main():
    print("=" * 70)
    print("CTI Hierarchical Ranking: Codex Experiment 1")
    print("=" * 70)

    # Load data
    print("\nLoading cache points...")
    all_pts = load_cache()
    print(f"Loaded {len(all_pts)} raw layer-level points")

    families = {}
    for p in all_pts:
        f = p["family"]
        families[f] = families.get(f, 0) + 1
    print(f"Family distribution: {families}")

    # =========================================================
    # BASELINE: Original method (mean across layers, raw kappa)
    # =========================================================
    print("\n--- BASELINE: Mean-layer, raw kappa ranking ---")
    groups = {}
    for p in all_pts:
        key = (p["model"], p["dataset"])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    mean_pts = []
    for (model, dataset), plist in groups.items():
        mean_pts.append({
            "model": model,
            "dataset": dataset,
            "K": plist[0]["K"],
            "q": float(np.mean([p["q"] for p in plist])),
            "kappa": float(np.mean([p["kappa"] for p in plist])),
            "family": plist[0]["family"],
            "n_layers": len(plist),
        })

    baseline_alphas = {"decoder": {"alpha": 1.477, "intercepts": {}},
                       "encoder": {"alpha": 1.477, "intercepts": {}},
                       "ssm": {"alpha": 1.477, "intercepts": {}},
                       "hybrid": {"alpha": 1.477, "intercepts": {}}}
    baseline_result = rank_and_evaluate(mean_pts, baseline_alphas, "baseline_mean_raw_kappa")
    print(f"  Pass: {baseline_result['n_pass']}/{baseline_result['n_datasets']}, "
          f"mean rho: {baseline_result['mean_spearman_rho']:.4f}")

    # =========================================================
    # FIX 1: Best-layer selection (not mean)
    # =========================================================
    print("\n--- FIX 1: Best-layer kappa, raw ranking ---")
    best_pts = best_layer_per_model_dataset(all_pts)
    print(f"  Best-layer points: {len(best_pts)}")

    fix1_result = rank_and_evaluate(best_pts, baseline_alphas, "best_layer_raw_kappa")
    print(f"  Pass: {fix1_result['n_pass']}/{fix1_result['n_datasets']}, "
          f"mean rho: {fix1_result['mean_spearman_rho']:.4f}")

    # =========================================================
    # FIX 2: Best-layer + family-aware alpha (LOAO within-family)
    # =========================================================
    print("\n--- FIX 2: Best-layer + family-aware alpha ---")
    family_alphas = fit_family_alpha(best_pts)
    for fam, info in sorted(family_alphas.items()):
        print(f"  {fam}: alpha={info['alpha']:.3f}, n={info['n']}, "
              f"r2={info.get('r2', 'N/A')}")

    fix2_result = rank_and_evaluate(best_pts, family_alphas, "best_layer_family_alpha")
    print(f"  Pass: {fix2_result['n_pass']}/{fix2_result['n_datasets']}, "
          f"mean rho: {fix2_result['mean_spearman_rho']:.4f}")

    # =========================================================
    # FIX 3: LODO (Leave-One-Dataset-Out) family alpha
    # =========================================================
    print("\n--- FIX 3: LODO family alpha (no data leakage) ---")
    datasets = sorted(set(p["dataset"] for p in best_pts))
    lodo_results = {}
    lodo_rhos = []

    for held_out in datasets:
        train = [p for p in best_pts if p["dataset"] != held_out]
        test = [p for p in best_pts if p["dataset"] == held_out]
        if len(test) < 5:
            continue

        lodo_alphas = fit_family_alpha(train)
        K = test[0]["K"]
        qs = np.array([p["q"] for p in test])
        kappas = np.array([p["kappa"] for p in test])

        # For held-out dataset, we don't have its intercept.
        # Use mean intercept from training datasets as proxy.
        for fam in lodo_alphas:
            intercepts = lodo_alphas[fam].get("intercepts", {})
            if intercepts:
                lodo_alphas[fam]["intercepts"][held_out] = float(np.mean(list(intercepts.values())))

        scores = np.array([
            predict_logit_q(p["kappa"], p["family"], held_out, lodo_alphas)
            for p in test
        ])

        rho, p_val = spearmanr(scores, qs)
        rho_raw, _ = spearmanr(kappas, qs)

        lodo_results[held_out] = {
            "K": K,
            "n_architectures": len(test),
            "spearman_rho_calibrated": float(rho),
            "spearman_rho_raw_kappa": float(rho_raw),
            "improvement": float(rho - rho_raw),
            "pr_pass": bool(abs(rho) >= SPEARMAN_THRESHOLD),
        }
        lodo_rhos.append(rho)
        status = "PASS" if abs(rho) >= SPEARMAN_THRESHOLD else "FAIL"
        print(f"  {held_out:>20} K={K:>3} rho_cal={rho:.3f} rho_raw={rho_raw:.3f} "
              f"delta={rho-rho_raw:+.3f} {status}")

    lodo_n_pass = sum(1 for d in lodo_results.values() if d["pr_pass"])
    lodo_mean = float(np.mean(lodo_rhos)) if lodo_rhos else 0.0
    print(f"  LODO Pass: {lodo_n_pass}/{len(lodo_results)}, mean rho: {lodo_mean:.4f}")

    # =========================================================
    # FIX 4: Decoder-only ranking (alpha IS constant within family)
    # =========================================================
    print("\n--- FIX 4: Decoder-only ranking (constant alpha) ---")
    decoder_pts = [p for p in best_pts if p["family"] == "decoder"]
    print(f"  Decoder-only points: {len(decoder_pts)}")

    by_ds_dec = {}
    for p in decoder_pts:
        d = p["dataset"]
        if d not in by_ds_dec:
            by_ds_dec[d] = []
        by_ds_dec[d].append(p)

    dec_results = {}
    dec_rhos = []
    for dataset, plist in sorted(by_ds_dec.items()):
        if len(plist) < 5:
            continue
        K = plist[0]["K"]
        kappas = np.array([p["kappa"] for p in plist])
        qs = np.array([p["q"] for p in plist])
        rho, p_val = spearmanr(kappas, qs)
        kappa_cv = float(np.std(kappas) / np.mean(kappas)) if np.mean(kappas) > 0 else 0
        pr = abs(rho) >= SPEARMAN_THRESHOLD
        dec_results[dataset] = {
            "K": K, "n": len(plist), "spearman_rho": float(rho),
            "kappa_cv": kappa_cv, "pr_pass": bool(pr),
        }
        dec_rhos.append(rho)
        print(f"  {dataset:>20} K={K:>3} n={len(plist):>2} rho={rho:.3f} "
              f"kappa_cv={kappa_cv:.3f} {'PASS' if pr else 'FAIL'}")

    dec_n_pass = sum(1 for d in dec_results.values() if d["pr_pass"])
    dec_mean = float(np.mean(dec_rhos)) if dec_rhos else 0.0
    print(f"  Decoder-only: {dec_n_pass}/{len(dec_results)} pass, mean rho: {dec_mean:.4f}")

    # =========================================================
    # FIX 5: All models, best layer with q (pick layer with highest q)
    # =========================================================
    print("\n--- FIX 5: Best-q-layer selection (oracle layer) ---")
    best_q_pts = []
    for (model, dataset), plist in groups.items():
        best = max(plist, key=lambda x: x["q"])
        best["n_layers"] = len(plist)
        best_q_pts.append(best)

    fix5_result = rank_and_evaluate(best_q_pts, baseline_alphas, "best_q_layer_raw_kappa")
    print(f"  Pass: {fix5_result['n_pass']}/{fix5_result['n_datasets']}, "
          f"mean rho: {fix5_result['mean_spearman_rho']:.4f}")

    # =========================================================
    # CORRELATION: kappa_cv predicts rankability
    # =========================================================
    print("\n--- KAPPA CV vs RANKABILITY ---")
    all_cv = []
    all_rho = []
    for ds in datasets:
        ds_pts_all = [p for p in best_pts if p["dataset"] == ds]
        if len(ds_pts_all) < 5:
            continue
        kappas = np.array([p["kappa"] for p in ds_pts_all])
        qs = np.array([p["q"] for p in ds_pts_all])
        cv = float(np.std(kappas) / np.mean(kappas)) if np.mean(kappas) > 0 else 0
        rho, _ = spearmanr(kappas, qs)
        all_cv.append(cv)
        all_rho.append(rho)
        print(f"  {ds:>20}: kappa_cv={cv:.3f}, spearman_rho={rho:.3f}")

    if len(all_cv) >= 3:
        cv_rho_corr, cv_rho_p = spearmanr(all_cv, all_rho)
        print(f"  Correlation(kappa_cv, rankability): Spearman={cv_rho_corr:.3f}, p={cv_rho_p:.4f}")
        print(f"  => Kappa spread {'PREDICTS' if cv_rho_corr > 0.5 else 'does NOT predict'} rankability")

    # =========================================================
    # DIAGNOSIS: Per-dataset breakdown
    # =========================================================
    print("\n--- DIAGNOSIS ---")
    for ds in datasets:
        ds_pts = [p for p in best_pts if p["dataset"] == ds]
        if len(ds_pts) < 5:
            continue
        K = ds_pts[0]["K"]
        kappas = np.array([p["kappa"] for p in ds_pts])
        qs = np.array([p["q"] for p in ds_pts])
        families_present = set(p["family"] for p in ds_pts)

        kappa_range = float(kappas.max() - kappas.min())
        kappa_cv = float(np.std(kappas) / np.mean(kappas)) if np.mean(kappas) > 0 else 0
        q_range = float(qs.max() - qs.min())
        mean_q = float(np.mean(qs))
        saturated = mean_q > 0.85

        print(f"  {ds:>20}: K={K:>3}, n={len(ds_pts):>2}, families={sorted(families_present)}")
        print(f"    kappa: range={kappa_range:.3f}, CV={kappa_cv:.3f}")
        print(f"    q:     range={q_range:.3f}, mean={mean_q:.3f}, saturated={saturated}")

    # =========================================================
    # OUTPUT
    # =========================================================
    output = {
        "experiment": "cti_hierarchical_ranking",
        "description": "Codex Experiment 1: Hierarchical calibration for architecture ranking",
        "codex_success_criteria": {
            "pass_rate_target": ">=7/10",
            "agnews_delta": ">=+0.25",
            "emotion_delta": ">=+0.25",
        },
        "baseline": baseline_result,
        "fix1_best_layer": fix1_result,
        "fix2_family_alpha": fix2_result,
        "fix3_lodo": {
            "method": "lodo_family_alpha",
            "n_datasets": len(lodo_results),
            "mean_spearman_rho": lodo_mean,
            "n_pass": lodo_n_pass,
            "per_dataset": lodo_results,
        },
        "fix4_decoder_only": {
            "method": "decoder_only_best_layer_kappa",
            "n_datasets": len(dec_results),
            "mean_spearman_rho": dec_mean,
            "n_pass": dec_n_pass,
            "per_dataset": dec_results,
        },
        "fix5_best_q_layer": fix5_result,
        "family_alphas": {k: {kk: vv for kk, vv in v.items() if kk != "intercepts"}
                          for k, v in family_alphas.items()},
        "kappa_cv_vs_rankability": {
            "correlation": float(cv_rho_corr) if len(all_cv) >= 3 else None,
            "p_value": float(cv_rho_p) if len(all_cv) >= 3 else None,
        },
        "codex_criteria_met": {
            "pass_rate_7_of_10": lodo_n_pass >= 7,
        },
    }

    # Check agnews/emotion improvement
    for ds_name in ["agnews", "emotion"]:
        baseline_rho = baseline_result["per_dataset"].get(ds_name, {}).get("spearman_rho_raw_kappa", 0)
        lodo_rho = lodo_results.get(ds_name, {}).get("spearman_rho_calibrated", 0)
        delta = lodo_rho - baseline_rho
        output["codex_criteria_met"][f"{ds_name}_delta_ge_025"] = delta >= 0.25
        print(f"\n{ds_name}: baseline={baseline_rho:.3f}, LODO={lodo_rho:.3f}, delta={delta:+.3f} "
              f"{'PASS' if delta >= 0.25 else 'FAIL'}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline (mean-layer, raw kappa): {baseline_result['n_pass']}/{baseline_result['n_datasets']} pass, "
          f"mean rho={baseline_result['mean_spearman_rho']:.3f}")
    print(f"  Fix 1 (best-layer, raw kappa):    {fix1_result['n_pass']}/{fix1_result['n_datasets']} pass, "
          f"mean rho={fix1_result['mean_spearman_rho']:.3f}")
    print(f"  Fix 2 (best-layer, family alpha): {fix2_result['n_pass']}/{fix2_result['n_datasets']} pass, "
          f"mean rho={fix2_result['mean_spearman_rho']:.3f}")
    print(f"  Fix 3 (LODO, no leakage):         {lodo_n_pass}/{len(lodo_results)} pass, "
          f"mean rho={lodo_mean:.3f}")
    print(f"  Fix 4 (decoder-only, best-layer):  {dec_n_pass}/{len(dec_results)} pass, "
          f"mean rho={dec_mean:.3f}")
    print(f"  Fix 5 (best-q-layer, raw kappa):  {fix5_result['n_pass']}/{fix5_result['n_datasets']} pass, "
          f"mean rho={fix5_result['mean_spearman_rho']:.3f}")


if __name__ == "__main__":
    main()
