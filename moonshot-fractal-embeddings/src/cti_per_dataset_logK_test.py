#!/usr/bin/env python
"""
PER-DATASET log(K) TEST: Within each dataset (fixed K), kappa predicts q.
Cross-dataset: does the SLOPE vary with K as log(K) or sqrt(K)?

Key idea: the theory says q = Phi(a*kappa*d + b*log(K) + c).
At fixed K, this is just q = Phi(a'*kappa + c').
The slope a' should be the same across datasets if theory is correct.
If slopes differ, the K-dependence must be captured by how
the slope varies with K.

This test fits per-dataset sigmoids and checks whether the slopes
vary as 1/log(K) or 1/sqrt(K).
"""

import json
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

D_HIDDEN = {
    "EleutherAI/pythia-160m": 768,
    "EleutherAI/pythia-410m": 1024,
    "EleutherAI/pythia-1b": 2048,
    "EleutherAI/pythia-1.4b": 2048,
    "EleutherAI/pythia-2.8b": 2560,
    "HuggingFaceTB/SmolLM2-360M": 960,
    "Qwen/Qwen2-0.5B": 896,
    "Qwen/Qwen3-0.6B": 1024,
    "state-spaces/mamba-130m-hf": 768,
    "state-spaces/mamba-370m-hf": 1024,
    "state-spaces/mamba-790m-hf": 1536,
    "state-spaces/mamba-2.8b-hf": 2560,
    "tiiuae/falcon-h1-0.5b-base": 1024,
}


def load_all_points():
    all_points = []
    try:
        with open(RESULTS_DIR / "cti_geometry_mediator.json") as f:
            data = json.load(f)
        for p in data["all_points"]:
            d = D_HIDDEN.get(p["model"], None)
            if d is None:
                continue
            K = 150
            knn = p["knn"]
            q = (knn - 1.0/K) / (1.0 - 1.0/K)
            all_points.append({"model": p["model"], "dataset": "clinc", "K": K,
                              "kappa": p["kappa"], "q": q, "d": d, "alpha": p["alpha"]})
    except Exception as e:
        print(f"  Mediator: {e}")

    for ds in ["agnews", "dbpedia_classes"]:
        try:
            with open(RESULTS_DIR / f"cti_multidata_{ds}_cache.json") as f:
                data = json.load(f)
            for p in data:
                d = D_HIDDEN.get(p["model"], None)
                if d is None:
                    continue
                K = p["n_classes"]
                knn = p["knn"]
                q = (knn - 1.0/K) / (1.0 - 1.0/K)
                all_points.append({"model": p["model"], "dataset": ds, "K": K,
                                  "kappa": p["kappa"], "q": q, "d": d, "alpha": p["alpha"]})
        except Exception as e:
            print(f"  {ds}: {e}")

    try:
        with open(RESULTS_DIR / "cti_blind_prediction.json") as f:
            data = json.load(f)
        for p in data["blind_points"]:
            d = D_HIDDEN.get(p["model"], None)
            if d is None:
                continue
            K = p["K"]
            knn = p["knn"]
            q = (knn - 1.0/K) / (1.0 - 1.0/K)
            all_points.append({"model": p["model"], "dataset": p["dataset"], "K": K,
                              "kappa": p["kappa"], "q": q, "d": d, "alpha": p["alpha"]})
    except Exception as e:
        print(f"  Blind: {e}")

    return all_points


def main():
    print("=" * 70)
    print("PER-DATASET SLOPE ANALYSIS: How does sigmoid slope vary with K?")
    print("=" * 70)

    all_points = load_all_points()
    valid = [p for p in all_points if 0.01 < p["q"] < 0.99 and p["kappa"] > 0]
    print(f"Loaded {len(valid)} valid points")

    # Group by dataset
    datasets = {}
    for p in valid:
        ds = p["dataset"]
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(p)

    print(f"\n{'='*70}")
    print("PER-DATASET SIGMOID FITS")
    print("=" * 70)

    def sig(x, a, b):
        return expit(a * x + b)

    per_ds_results = []
    for ds_name in sorted(datasets.keys()):
        pts = datasets[ds_name]
        if len(pts) < 5:
            continue

        kappas = np.array([p["kappa"] for p in pts])
        qs = np.array([p["q"] for p in pts])
        K = int(pts[0]["K"])

        try:
            popt, _ = curve_fit(sig, kappas, qs, p0=[10.0, -1.0], maxfev=10000)
            a_fit, b_fit = popt
            q_pred = sig(kappas, a_fit, b_fit)
            ss_res = np.sum((qs - q_pred)**2)
            ss_tot = np.sum((qs - qs.mean())**2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            mae = np.mean(np.abs(qs - q_pred))

            per_ds_results.append({
                "dataset": ds_name, "K": K, "n": len(pts),
                "a": float(a_fit), "b": float(b_fit),
                "r2": float(r2), "mae": float(mae),
            })

            print(f"\n  {ds_name} (K={K}, n={len(pts)}):")
            print(f"    sigmoid({a_fit:.3f}*kappa + {b_fit:.3f})")
            print(f"    R^2 = {r2:.4f}, MAE = {mae:.4f}")
        except Exception as e:
            print(f"\n  {ds_name}: fit failed: {e}")

    if len(per_ds_results) < 3:
        print("Not enough per-dataset fits")
        return

    # Now analyze: does the slope 'a' vary as 1/log(K), 1/sqrt(K), or something else?
    print(f"\n{'='*70}")
    print("SLOPE vs K ANALYSIS")
    print("=" * 70)

    Ks = np.array([r["K"] for r in per_ds_results])
    slopes = np.array([r["a"] for r in per_ds_results])
    intercepts = np.array([r["b"] for r in per_ds_results])

    print(f"\n  Dataset       K     slope(a)  intercept(b)")
    print(f"  {'-'*50}")
    for r in per_ds_results:
        print(f"  {r['dataset']:>15} {r['K']:>5} {r['a']:>10.3f} {r['b']:>10.3f}")

    # Test: a ~ C/log(K) vs a ~ C/sqrt(K) vs a ~ C/K
    # i.e., a*log(K) should be constant if a ~ 1/log(K)
    a_logK = slopes * np.log(Ks + 1)
    a_sqrtK = slopes * np.sqrt(Ks)
    a_K = slopes * Ks

    print(f"\n  slope * log(K+1): {a_logK} -- CV = {np.std(a_logK)/np.mean(a_logK):.3f}")
    print(f"  slope * sqrt(K):  {a_sqrtK} -- CV = {np.std(a_sqrtK)/np.mean(a_sqrtK):.3f}")
    print(f"  slope * K:        {a_K} -- CV = {np.std(a_K)/np.mean(a_K):.3f}")

    cv_logK = np.std(a_logK) / max(np.mean(a_logK), 1e-10)
    cv_sqrtK = np.std(a_sqrtK) / max(np.mean(a_sqrtK), 1e-10)
    cv_K = np.std(a_K) / max(np.mean(a_K), 1e-10)

    print(f"\n  BEST COLLAPSE: ", end="")
    if cv_logK < cv_sqrtK and cv_logK < cv_K:
        print(f"log(K) (CV={cv_logK:.3f})")
    elif cv_sqrtK < cv_logK and cv_sqrtK < cv_K:
        print(f"sqrt(K) (CV={cv_sqrtK:.3f})")
    else:
        print(f"K (CV={cv_K:.3f})")

    # Also check: does slope correlate with log(K) or sqrt(K)?
    if len(Ks) >= 3:
        r_logK, p_logK = pearsonr(np.log(Ks), slopes)
        r_sqrtK, p_sqrtK = pearsonr(np.sqrt(Ks), slopes)
        print(f"\n  Correlation of slope with log(K): r={r_logK:.3f}, p={p_logK:.4f}")
        print(f"  Correlation of slope with sqrt(K): r={r_sqrtK:.3f}, p={p_sqrtK:.4f}")

    # MIXED-EFFECTS MODEL: per-dataset intercept, shared slope
    print(f"\n{'='*70}")
    print("MIXED-EFFECTS: shared kappa slope, per-dataset intercept")
    print("=" * 70)

    # Fit: q_i = sigmoid(a_shared * kappa_i / f(K_i) + b_dataset_i)
    kappas_all = np.array([p["kappa"] for p in valid])
    qs_all = np.array([p["q"] for p in valid])
    Ks_all = np.array([float(p["K"]) for p in valid])
    ds_names = [p["dataset"] for p in valid]
    unique_ds = sorted(set(ds_names))
    ds_idx = np.array([unique_ds.index(d) for d in ds_names])

    # Model: sigmoid(a*kappa/f(K) + b_j) where j is dataset index
    # Test with f(K) = sqrt(K) and f(K) = log(K)
    for f_name, f_func in [("sqrt(K)", np.sqrt), ("log(K+1)", lambda x: np.log(x+1))]:
        n_ds = len(unique_ds)

        def loss_mixed(params):
            a = params[0]
            bs = params[1:1+n_ds]
            x = a * kappas_all / f_func(Ks_all) + bs[ds_idx]
            pred = expit(x)
            return np.sum((qs_all - pred)**2)

        best = None
        best_loss = float("inf")
        for a0 in [5.0, 10.0, 20.0, 50.0]:
            p0 = [a0] + [-1.0]*n_ds
            try:
                res = minimize(loss_mixed, p0, method="Nelder-Mead",
                              options={"maxiter": 20000})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best = res.x
            except:
                pass

        if best is not None:
            a_m = best[0]
            bs_m = best[1:1+n_ds]
            pred_m = expit(a_m * kappas_all / f_func(Ks_all) + bs_m[ds_idx])
            r2_m = 1 - np.sum((qs_all - pred_m)**2) / np.sum((qs_all - qs_all.mean())**2)
            mae_m = np.mean(np.abs(qs_all - pred_m))
            print(f"\n  {f_name}: a={a_m:.3f}, R^2={r2_m:.4f}, MAE={mae_m:.4f}")
            print(f"    Per-dataset intercepts: ", end="")
            for i, d in enumerate(unique_ds):
                print(f"{d}={bs_m[i]:.3f} ", end="")
            print()

    # Also mixed-effects with ADDITIVE log(K)
    def loss_mixed_add(params):
        a = params[0]
        b_logK = params[1]
        bs = params[2:2+len(unique_ds)]
        x = a * kappas_all + b_logK * np.log(Ks_all + 1) + bs[ds_idx]
        pred = expit(x)
        return np.sum((qs_all - pred)**2)

    best_add = None
    best_add_loss = float("inf")
    for a0 in [5.0, 10.0, 20.0]:
        for b0 in [-1.0, -0.5, 0.0]:
            p0 = [a0, b0] + [-1.0]*len(unique_ds)
            try:
                res = minimize(loss_mixed_add, p0, method="Nelder-Mead",
                              options={"maxiter": 20000})
                if res.fun < best_add_loss:
                    best_add_loss = res.fun
                    best_add = res.x
            except:
                pass

    if best_add is not None:
        a_add = best_add[0]
        b_add = best_add[1]
        bs_add = best_add[2:2+len(unique_ds)]
        pred_add = expit(a_add * kappas_all + b_add * np.log(Ks_all + 1) + bs_add[ds_idx])
        r2_add = 1 - np.sum((qs_all - pred_add)**2) / np.sum((qs_all - qs_all.mean())**2)
        mae_add = np.mean(np.abs(qs_all - pred_add))
        print(f"\n  ADDITIVE+intercepts: a={a_add:.3f}, b_logK={b_add:.3f}, R^2={r2_add:.4f}, MAE={mae_add:.4f}")

    print(f"\n{'='*70}")
    print("SCORECARD")
    print("=" * 70)

    checks = [
        ("Per-dataset sigmoid R^2 > 0.80 in >= 3 datasets",
         sum(1 for r in per_ds_results if r["r2"] > 0.80) >= 3,
         f"{sum(1 for r in per_ds_results if r['r2'] > 0.80)} datasets"),
        ("Slope varies with K (correlation |r| > 0.7)",
         abs(r_logK) > 0.7 or abs(r_sqrtK) > 0.7 if len(Ks) >= 3 else False,
         f"r_logK={r_logK:.3f}, r_sqrtK={r_sqrtK:.3f}" if len(Ks) >= 3 else "N/A"),
        ("Mixed-effects R^2 > 0.85",
         r2_m > 0.85 if best is not None else False,
         f"R^2={r2_m:.4f}" if best is not None else "N/A"),
        ("log(K) collapse CV < sqrt(K) collapse CV",
         cv_logK < cv_sqrtK,
         f"logK CV={cv_logK:.3f}, sqrtK CV={cv_sqrtK:.3f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "per_dataset_logK_test",
        "per_dataset": per_ds_results,
        "collapse_cv": {
            "logK": float(cv_logK),
            "sqrtK": float(cv_sqrtK),
            "K": float(cv_K),
        },
        "passes": passes,
    }

    out_path = RESULTS_DIR / "cti_per_dataset_logK.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
