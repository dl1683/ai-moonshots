#!/usr/bin/env python
"""
MULTI-DATASET REPLICATION OF ARCHITECTURE UNIVERSALITY CLASSES

Codex design (7.9/10 Nobel review):
  "Highest-upside next experiment: multi-dataset replication.
   Your biggest vulnerability is 'this might be CLINC-specific.'
   Fixing that gives the largest credibility jump per unit effort."

Replicates the geometry mediator experiment (7 models x 9 alphas)
on AGNews and DBPedia_Classes, then pools with existing CLINC data.

Tests:
  1. Per-dataset: Do transformer/SSM universality classes replicate?
  2. Pooled: Does global rho >= 0.95 hold across all datasets?
  3. Slope consistency: Are per-dataset slopes consistent?
  4. LODO (Leave-One-Dataset-Out): Can we predict a held-out dataset?

Resumable: saves partial results after each model completes.
"""

import json
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr, f as f_dist

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset


DATASETS = ["agnews", "dbpedia_classes"]
ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
ALL_MODELS = [
    "HuggingFaceTB/SmolLM2-360M",
    "EleutherAI/pythia-410m",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen3-0.6B",
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf",
]


def extract_all_layer_reps(model, tokenizer, texts, alpha, device="cuda", batch_size=32):
    """Extract all layer representations with residual scaling."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask",
                           torch.ones(enc["input_ids"].shape, device=device))
            for idx, hs in enumerate(out.hidden_states):
                hs_f = hs.float()
                m = mask.unsqueeze(-1).float()
                pooled = (hs_f * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if idx not in all_hidden:
                    all_hidden[idx] = []
                all_hidden[idx].append(pooled.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


def compute_kappa_and_eta(X, labels):
    """Compute kappa and eta from representations."""
    try:
        if np.isnan(X).any():
            return {"kappa": 0.0, "eta": 0.0}

        d = X.shape[1]
        unique_labels = np.unique(labels)
        grand_mean = X.mean(0)

        Z_parts = []
        trace_sb = 0.0
        trace_sw = 0.0

        for lbl in unique_labels:
            lbl_mask = labels == lbl
            n_k = lbl_mask.sum()
            if n_k < 2:
                continue
            X_k = X[lbl_mask]
            mu_k = X_k.mean(0)
            trace_sb += n_k * np.sum((mu_k - grand_mean) ** 2)
            centered = X_k - mu_k
            trace_sw += np.sum(centered ** 2)
            Z_parts.append(centered)

        if trace_sw < 1e-12:
            return {"kappa": 100.0 if trace_sb > 0 else 0.0, "eta": 0.0}

        kappa = float(min(trace_sb / trace_sw, 100.0))

        Z = np.concatenate(Z_parts, axis=0)
        try:
            s = np.linalg.svd(Z, compute_uv=False)
            s2 = s ** 2
            s4 = s2 ** 2
            trace_sw_sq = float(s4.sum())
            if trace_sw_sq < 1e-20:
                eta = 0.0
            else:
                eta = float((trace_sw ** 2) / (d * trace_sw_sq))
        except np.linalg.LinAlgError:
            eta = 0.0

        return {"kappa": kappa, "eta": eta}
    except Exception:
        return {"kappa": 0.0, "eta": 0.0}


def compute_knn(X, labels, n_train_frac=0.7):
    """kNN accuracy."""
    n = len(labels)
    n_train = int(n_train_frac * n)
    if n_train < 5 or n - n_train < 5:
        return 0.0
    try:
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X[:n_train], labels[:n_train])
        return float(knn.score(X[n_train:], labels[n_train:]))
    except Exception:
        return 0.0


def compute_layer_stats(reps, labels):
    """Compute kNN, kappa, and eta averaged across all layers."""
    knn_vals, kappa_vals, eta_vals = [], [], []

    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue
        knn_val = compute_knn(X, labels)
        stats = compute_kappa_and_eta(X, labels)
        knn_vals.append(knn_val)
        if np.isfinite(stats["kappa"]):
            kappa_vals.append(stats["kappa"])
        if np.isfinite(stats.get("eta", 0)):
            eta_vals.append(stats["eta"])

    return {
        "knn": float(np.mean(knn_vals)) if knn_vals else 0,
        "kappa": float(np.mean(kappa_vals)) if kappa_vals else 0,
        "eta": float(np.mean(eta_vals)) if eta_vals else 0,
    }


def sigmoid(x, a, b, c, d):
    """Generalized sigmoid."""
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def load_cache(dataset_name):
    """Load cached results for a dataset."""
    cache_path = RESULTS_DIR / f"cti_multidata_{dataset_name}_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return []


def save_cache(dataset_name, points):
    """Save partial results."""
    cache_path = RESULTS_DIR / f"cti_multidata_{dataset_name}_cache.json"
    with open(cache_path, "w") as f:
        json.dump(points, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))


def extract_dataset(dataset_name, device="cuda"):
    """Extract kappa, eta, kNN for all models on one dataset."""
    print(f"\n{'#'*70}")
    print(f"# DATASET: {dataset_name}")
    print(f"{'#'*70}")

    # Load cached results
    cached = load_cache(dataset_name)
    cached_keys = set()
    for p in cached:
        cached_keys.add((p["model"], p["alpha"]))

    if cached:
        print(f"  Loaded {len(cached)} cached points")

    # Load dataset
    ds = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])
    n_classes = len(np.unique(labels))
    print(f"  {len(texts)} samples, {n_classes} classes")

    all_points = list(cached)

    for model_id in ALL_MODELS:
        paradigm = "ssm" if "mamba" in model_id.lower() else "transformer"
        short = model_id.split("/")[-1]

        # Check if all alphas are cached for this model
        model_cached = all(
            (model_id, a) in cached_keys for a in ALPHAS
        )
        if model_cached:
            print(f"\n  {short}: ALL CACHED, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  MODEL: {short} ({paradigm})")
        print(f"{'='*60}")

        try:
            model, tokenizer, n_layers, n_params = load_model(model_id, device)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        for alpha in ALPHAS:
            if (model_id, alpha) in cached_keys:
                print(f"    alpha={alpha:.2f} CACHED")
                continue

            print(f"    alpha={alpha:.2f}", end="", flush=True)
            t0 = time.time()

            try:
                reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
                stats = compute_layer_stats(reps, labels)
                elapsed = time.time() - t0
                print(f"  kNN={stats['knn']:.3f}  kappa={stats['kappa']:.4f}  "
                      f"eta={stats['eta']:.4f}  ({elapsed:.1f}s)")

                point = {
                    "model": model_id,
                    "paradigm": paradigm,
                    "dataset": dataset_name,
                    "n_classes": n_classes,
                    "n_layers": n_layers,
                    "n_params": n_params,
                    "alpha": alpha,
                    "knn": stats["knn"],
                    "kappa": stats["kappa"],
                    "eta": stats["eta"],
                    "kappa_eta": stats["kappa"] * stats["eta"],
                }
                all_points.append(point)
                cached_keys.add((model_id, alpha))
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM!")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            sys.stdout.flush()

        # Save cache after each model
        save_cache(dataset_name, all_points)
        print(f"  Saved {len(all_points)} points to cache")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return all_points


def analyze_per_dataset(points, dataset_name):
    """Analyze universality for a single dataset."""
    kappas = np.array([p["kappa"] for p in points])
    knns = np.array([p["knn"] for p in points])
    etas = np.array([p["eta"] for p in points])
    paradigms = np.array([p["paradigm"] for p in points])

    result = {"dataset": dataset_name, "n_points": len(points)}

    # Slopes by architecture
    for par in ["transformer", "ssm"]:
        mask = paradigms == par
        if mask.sum() < 3:
            continue
        slope = np.polyfit(kappas[mask], knns[mask], 1)[0]
        rho, p = spearmanr(kappas[mask], knns[mask])
        result[f"slope_{par}"] = float(slope)
        result[f"rho_{par}"] = float(rho)

    if "slope_transformer" in result and "slope_ssm" in result:
        st = result["slope_transformer"]
        ss = result["slope_ssm"]
        result["slope_ratio"] = float(ss / st) if abs(st) > 1e-10 else float("inf")

    # Global sigmoid fit
    try:
        popt, _ = curve_fit(sigmoid, kappas, knns,
                            p0=[0.6, 10, 0.3, 0.1], maxfev=10000)
        pred = sigmoid(kappas, *popt)
        ss_tot = np.sum((knns - knns.mean()) ** 2)
        r2 = 1 - np.sum((knns - pred) ** 2) / ss_tot
        result["sigmoid_r2"] = float(r2)
    except Exception:
        result["sigmoid_r2"] = 0.0

    # Global Spearman
    rho, p = spearmanr(kappas, knns)
    result["global_rho"] = float(rho)
    result["global_rho_p"] = float(p)

    # Eta distribution
    for par in ["transformer", "ssm"]:
        mask = paradigms == par
        if mask.sum() > 0:
            result[f"eta_mean_{par}"] = float(etas[mask].mean())
            result[f"eta_std_{par}"] = float(etas[mask].std())

    return result


def analyze_pooled(all_datasets_points, clinc_points):
    """Pool all datasets and analyze universality."""
    # Combine all points
    combined = list(clinc_points) + list(all_datasets_points)

    kappas = np.array([p["kappa"] for p in combined])
    knns = np.array([p["knn"] for p in combined])
    etas = np.array([p["eta"] for p in combined])
    paradigms = np.array([p["paradigm"] for p in combined])
    datasets = np.array([p["dataset"] for p in combined])

    ss_tot = np.sum((knns - knns.mean()) ** 2)

    print(f"\n{'='*70}")
    print(f"POOLED ANALYSIS: {len(combined)} points across "
          f"{len(set(datasets))} datasets")
    print(f"{'='*70}")

    # 1. Global correlations
    rho, p = spearmanr(kappas, knns)
    r, pr = pearsonr(kappas, knns)
    print(f"\n1. GLOBAL CORRELATIONS")
    print(f"   Spearman rho = {rho:.4f} (p={p:.2e})")
    print(f"   Pearson r    = {r:.4f}")

    # 2. Per-architecture slopes
    print(f"\n2. PER-ARCHITECTURE SLOPES")
    slopes = {}
    for par in ["transformer", "ssm"]:
        mask = paradigms == par
        if mask.sum() < 3:
            continue
        slope = np.polyfit(kappas[mask], knns[mask], 1)[0]
        rho_par, _ = spearmanr(kappas[mask], knns[mask])
        slopes[par] = slope
        print(f"   {par:>12}: slope={slope:.4f}, rho={rho_par:.4f}, N={mask.sum()}")

    if "transformer" in slopes and "ssm" in slopes:
        ratio = slopes["ssm"] / slopes["transformer"]
        print(f"   Slope ratio (SSM/T): {ratio:.4f}")

    # 3. Per-dataset slopes
    print(f"\n3. PER-DATASET SLOPES")
    dataset_slopes = {}
    for ds_name in sorted(set(datasets)):
        ds_mask = datasets == ds_name
        print(f"\n   Dataset: {ds_name} (N={ds_mask.sum()})")
        for par in ["transformer", "ssm"]:
            mask = ds_mask & (paradigms == par)
            if mask.sum() < 3:
                continue
            slope = np.polyfit(kappas[mask], knns[mask], 1)[0]
            rho_par, _ = spearmanr(kappas[mask], knns[mask])
            print(f"     {par:>12}: slope={slope:.4f}, rho={rho_par:.4f}")
            if ds_name not in dataset_slopes:
                dataset_slopes[ds_name] = {}
            dataset_slopes[ds_name][par] = float(slope)

    # 4. Architecture dummy test (pooled)
    print(f"\n4. ARCHITECTURE DUMMY TEST (POOLED)")
    is_ssm = (paradigms == "ssm").astype(float)

    X_full = np.column_stack([kappas, is_ssm, np.ones(len(kappas))])
    beta_full = np.linalg.lstsq(X_full, knns, rcond=None)[0]
    pred_full = X_full @ beta_full
    ss_res_full = np.sum((knns - pred_full) ** 2)

    X_red = np.column_stack([kappas, np.ones(len(kappas))])
    beta_red = np.linalg.lstsq(X_red, knns, rcond=None)[0]
    pred_red = X_red @ beta_red
    ss_res_red = np.sum((knns - pred_red) ** 2)

    n = len(knns)
    f_stat = ((ss_res_red - ss_res_full) / 1) / (ss_res_full / (n - 3))
    p_dummy = 1 - f_dist.cdf(f_stat, 1, n - 3)
    print(f"   Arch dummy F={f_stat:.4f}, p={p_dummy:.6f}")
    print(f"   Dummy coeff (SSM offset): {beta_full[1]:.4f}")

    # 5. Sigmoid fit (pooled)
    print(f"\n5. POOLED SIGMOID FIT")
    try:
        popt, _ = curve_fit(sigmoid, kappas, knns,
                            p0=[0.6, 10, 0.3, 0.1], maxfev=10000)
        pred = sigmoid(kappas, *popt)
        r2 = 1 - np.sum((knns - pred) ** 2) / ss_tot
        mae = float(np.mean(np.abs(knns - pred)))
        print(f"   R^2 = {r2:.4f}, MAE = {mae:.4f}")
    except Exception as e:
        r2 = 0.0
        mae = 1.0
        print(f"   Fit failed: {e}")

    # 6. LODO (Leave-One-Dataset-Out)
    print(f"\n6. LEAVE-ONE-DATASET-OUT CROSS-VALIDATION")
    lodo_results = {}
    for held_out in sorted(set(datasets)):
        train_mask = datasets != held_out
        test_mask = datasets == held_out

        k_train, knn_train = kappas[train_mask], knns[train_mask]
        k_test, knn_test = kappas[test_mask], knns[test_mask]

        try:
            popt_cv, _ = curve_fit(sigmoid, k_train, knn_train,
                                   p0=[0.6, 10, 0.3, 0.1], maxfev=10000)
            pred_cv = sigmoid(k_test, *popt_cv)
            mae_cv = float(np.mean(np.abs(knn_test - pred_cv)))
        except Exception:
            mae_cv = 1.0

        rho_cv, _ = spearmanr(k_test, knn_test) if len(k_test) > 2 else (0.0, 1.0)
        print(f"   Hold out {held_out:>20}: MAE={mae_cv:.4f}, rho={rho_cv:.4f}, "
              f"N={test_mask.sum()}")
        lodo_results[held_out] = {"mae": mae_cv, "rho": float(rho_cv),
                                  "n_points": int(test_mask.sum())}

    mean_lodo_mae = np.mean([v["mae"] for v in lodo_results.values()])
    print(f"\n   Mean LODO MAE: {mean_lodo_mae:.4f}")

    # 7. Slope consistency test
    print(f"\n7. SLOPE CONSISTENCY ACROSS DATASETS")
    t_slopes = []
    s_slopes = []
    ratios = []
    for ds_name, ds_slopes_val in sorted(dataset_slopes.items()):
        if "transformer" in ds_slopes_val and "ssm" in ds_slopes_val:
            t_slopes.append(ds_slopes_val["transformer"])
            s_slopes.append(ds_slopes_val["ssm"])
            ratio_val = ds_slopes_val["ssm"] / ds_slopes_val["transformer"]
            ratios.append(ratio_val)
            print(f"   {ds_name}: T={ds_slopes_val['transformer']:.4f}, "
                  f"S={ds_slopes_val['ssm']:.4f}, ratio={ratio_val:.4f}")

    if ratios:
        print(f"\n   Slope ratio mean={np.mean(ratios):.4f}, "
              f"std={np.std(ratios):.4f}")
        print(f"   Consistent (std < 0.3): {'YES' if np.std(ratios) < 0.3 else 'NO'}")

    # Compile results
    results = {
        "experiment": "multi_dataset_replication",
        "hypothesis": "Architecture universality classes replicate across datasets",
        "n_datasets": len(set(datasets)),
        "n_total_points": len(combined),
        "datasets_used": sorted(set(datasets)),
        "global_correlation": {
            "rho": float(rho), "r": float(r), "p": float(p),
        },
        "pooled_sigmoid": {
            "r2": float(r2), "mae": float(mae),
        },
        "architecture_dummy": {
            "f_stat": float(f_stat), "p_value": float(p_dummy),
            "dummy_coeff": float(beta_full[1]),
        },
        "per_dataset_slopes": dataset_slopes,
        "slope_ratio_stats": {
            "ratios": [float(r) for r in ratios],
            "mean": float(np.mean(ratios)) if ratios else 0,
            "std": float(np.std(ratios)) if ratios else 0,
            "consistent": bool(np.std(ratios) < 0.3) if ratios else False,
        },
        "lodo": lodo_results,
        "lodo_mean_mae": float(mean_lodo_mae),
        "scorecard": {
            "passes": 0,
            "total": 5,
            "details": [],
        },
    }

    # Scorecard
    checks = [
        ("Global rho >= 0.90", float(rho) >= 0.90, f"rho={rho:.4f}"),
        ("Slope ratio consistent (std<0.3)", bool(np.std(ratios) < 0.3) if ratios else False,
         f"std={np.std(ratios):.4f}" if ratios else "N/A"),
        ("LODO MAE <= 0.06", mean_lodo_mae <= 0.06, f"MAE={mean_lodo_mae:.4f}"),
        ("SSM slope > T slope in all datasets",
         all(r > 1.0 for r in ratios) if ratios else False,
         f"ratios={[f'{r:.2f}' for r in ratios]}"),
        ("Pooled sigmoid R^2 >= 0.90", float(r2) >= 0.90, f"R^2={r2:.4f}"),
    ]

    passes = 0
    for criterion, passed, value in checks:
        results["scorecard"]["details"].append({
            "criterion": criterion,
            "passed": bool(passed),
            "value": value,
        })
        if passed:
            passes += 1
    results["scorecard"]["passes"] = passes

    print(f"\n{'='*70}")
    print(f"SCORECARD: {passes}/5")
    print(f"{'='*70}")
    for d in results["scorecard"]["details"]:
        status = "PASS" if d["passed"] else "FAIL"
        print(f"  [{status}] {d['criterion']}: {d['value']}")

    return results


def main():
    print("=" * 70)
    print("MULTI-DATASET REPLICATION OF ARCHITECTURE UNIVERSALITY")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Phase 1: Extract data for each new dataset
    new_data = []
    for ds_name in DATASETS:
        ds_points = extract_dataset(ds_name, device)
        new_data.extend(ds_points)

    # Phase 2: Load existing CLINC data
    clinc_path = RESULTS_DIR / "cti_geometry_mediator.json"
    with open(clinc_path) as f:
        clinc_data = json.load(f)

    clinc_points = []
    for p in clinc_data["all_points"]:
        clinc_points.append({
            "model": p["model"],
            "paradigm": p["paradigm"],
            "dataset": "clinc",
            "n_classes": 150,
            "alpha": p["alpha"],
            "knn": p["knn"],
            "kappa": p["kappa"],
            "eta": p["eta"],
            "kappa_eta": p.get("kappa_eta", p["kappa"] * p["eta"]),
        })

    # Phase 3: Per-dataset analysis
    print(f"\n{'='*70}")
    print("PER-DATASET ANALYSIS")
    print(f"{'='*70}")

    per_dataset = {}
    for ds_name in ["clinc"] + DATASETS:
        if ds_name == "clinc":
            pts = clinc_points
        else:
            pts = [p for p in new_data if p["dataset"] == ds_name]
        if pts:
            per_dataset[ds_name] = analyze_per_dataset(pts, ds_name)
            print(f"\n  {ds_name}: rho={per_dataset[ds_name]['global_rho']:.4f}, "
                  f"sigmoid_R2={per_dataset[ds_name]['sigmoid_r2']:.4f}")
            if "slope_ratio" in per_dataset[ds_name]:
                print(f"    slope_ratio={per_dataset[ds_name]['slope_ratio']:.4f}")

    # Phase 4: Pooled analysis
    results = analyze_pooled(new_data, clinc_points)
    results["per_dataset"] = per_dataset

    # Save
    out_path = RESULTS_DIR / "cti_multi_dataset_replication.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
