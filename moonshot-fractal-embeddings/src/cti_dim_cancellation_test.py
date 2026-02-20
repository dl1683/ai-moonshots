#!/usr/bin/env python
"""
DIMENSION CANCELLATION TEST: Does adding d_hidden improve the fit on REAL data?

If kappa already encodes d through tr(S_B)/tr(S_W) normalization,
then adding d_hidden to the sigmoid fit should NOT improve R^2.

If it DOES improve, then kappa doesn't fully cancel d, and the
empirical law is merely approximate.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr, spearmanr, norm

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Hidden dimensions for each model family
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
    """Load all available (model, dataset) points with kappa, knn, K."""
    all_points = []

    # CLINC from geometry mediator
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
            all_points.append({
                "model": p["model"], "dataset": "clinc", "K": K,
                "kappa": p["kappa"], "eta": p.get("eta", 1.0),
                "knn": knn, "q": q, "d": d, "alpha": p["alpha"],
            })
    except Exception as e:
        print(f"  Mediator load failed: {e}")

    # AGNews, DBPedia from multi-dataset cache
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
                all_points.append({
                    "model": p["model"], "dataset": ds, "K": K,
                    "kappa": p["kappa"], "eta": p.get("eta", 1.0),
                    "knn": knn, "q": q, "d": d, "alpha": p["alpha"],
                })
        except Exception as e:
            print(f"  {ds} load failed: {e}")

    # Yahoo, arXiv from blind prediction
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
            all_points.append({
                "model": p["model"], "dataset": p["dataset"], "K": K,
                "kappa": p["kappa"], "eta": p.get("eta", 1.0),
                "knn": knn, "q": q, "d": d, "alpha": p["alpha"],
            })
    except Exception as e:
        print(f"  Blind load failed: {e}")

    return all_points


def main():
    print("=" * 70)
    print("DIMENSION CANCELLATION TEST: Does d_hidden improve the fit?")
    print("=" * 70)

    all_points = load_all_points()
    print(f"\nLoaded {len(all_points)} points")

    # Filter valid points
    valid = [p for p in all_points if 0 < p["q"] < 1 and p["kappa"] > 0]
    print(f"Valid points (0 < q < 1): {len(valid)}")

    if len(valid) < 10:
        print("Not enough valid points!")
        return

    kappas = np.array([p["kappa"] for p in valid])
    etas = np.array([p["eta"] for p in valid])
    qs = np.array([p["q"] for p in valid])
    Ks = np.array([float(p["K"]) for p in valid])
    ds = np.array([float(p["d"]) for p in valid])

    print(f"\nData summary:")
    print(f"  K range: {Ks.min():.0f} - {Ks.max():.0f}")
    print(f"  d range: {ds.min():.0f} - {ds.max():.0f}")
    print(f"  kappa range: {kappas.min():.4f} - {kappas.max():.4f}")
    print(f"  q range: {qs.min():.4f} - {qs.max():.4f}")

    def sig(x, a, b):
        return expit(a * x + b)

    # ================================================================
    # MODEL 1: sigmoid(a * kappa/sqrt(K) + b) [the empirical law]
    # ================================================================
    x1 = kappas / np.sqrt(Ks)
    try:
        p1, _ = curve_fit(sig, x1, qs, p0=[5.0, -1.0], maxfev=10000)
        q1 = sig(x1, *p1)
        r2_1 = 1 - np.sum((qs - q1)**2) / np.sum((qs - qs.mean())**2)
        mae_1 = np.mean(np.abs(qs - q1))
        print(f"\n  Model 1: sigmoid({p1[0]:.3f}*kappa/sqrt(K) + {p1[1]:.3f})")
        print(f"    R^2 = {r2_1:.6f}, MAE = {mae_1:.4f}")
    except:
        r2_1 = 0
        print("  Model 1: fit failed")

    # ================================================================
    # MODEL 2: sigmoid(a * kappa*d/sqrt(K) + b) [theory prediction]
    # ================================================================
    x2 = kappas * ds / np.sqrt(Ks)
    try:
        p2, _ = curve_fit(sig, x2, qs, p0=[0.01, -1.0], maxfev=10000)
        q2 = sig(x2, *p2)
        r2_2 = 1 - np.sum((qs - q2)**2) / np.sum((qs - qs.mean())**2)
        mae_2 = np.mean(np.abs(qs - q2))
        print(f"\n  Model 2: sigmoid({p2[0]:.6f}*kappa*d/sqrt(K) + {p2[1]:.3f})")
        print(f"    R^2 = {r2_2:.6f}, MAE = {mae_2:.4f}")
    except:
        r2_2 = 0
        print("  Model 2: fit failed")

    # ================================================================
    # MODEL 3: sigmoid(a * kappa*sqrt(d)/sqrt(K) + b)
    # ================================================================
    x3 = kappas * np.sqrt(ds) / np.sqrt(Ks)
    try:
        p3, _ = curve_fit(sig, x3, qs, p0=[0.1, -1.0], maxfev=10000)
        q3 = sig(x3, *p3)
        r2_3 = 1 - np.sum((qs - q3)**2) / np.sum((qs - qs.mean())**2)
        mae_3 = np.mean(np.abs(qs - q3))
        print(f"\n  Model 3: sigmoid({p3[0]:.5f}*kappa*sqrt(d)/sqrt(K) + {p3[1]:.3f})")
        print(f"    R^2 = {r2_3:.6f}, MAE = {mae_3:.4f}")
    except:
        r2_3 = 0
        print("  Model 3: fit failed")

    # ================================================================
    # MODEL 4: probit(a * kappa/sqrt(K) + b) [exact Gaussian theory]
    # ================================================================
    def probit(x, a, b):
        return norm.cdf(a * x + b)

    try:
        p4, _ = curve_fit(probit, x1, qs, p0=[5.0, -1.0], maxfev=10000)
        q4 = probit(x1, *p4)
        r2_4 = 1 - np.sum((qs - q4)**2) / np.sum((qs - qs.mean())**2)
        mae_4 = np.mean(np.abs(qs - q4))
        print(f"\n  Model 4: probit({p4[0]:.3f}*kappa/sqrt(K) + {p4[1]:.3f})")
        print(f"    R^2 = {r2_4:.6f}, MAE = {mae_4:.4f}")
    except:
        r2_4 = 0
        print("  Model 4: fit failed")

    # ================================================================
    # MODEL 5: Free d-power — sigmoid(a * kappa * d^gamma / sqrt(K) + b)
    # ================================================================
    def loss_free_d(params):
        a, b, gamma = params
        x = kappas * np.power(ds, gamma) / np.sqrt(Ks)
        pred = expit(a * x + b)
        return np.sum((qs - pred) ** 2)

    best_loss = float("inf")
    best_params = None
    for gamma_init in [-0.5, -0.2, 0.0, 0.2, 0.5, 1.0]:
        for a_init in [0.001, 0.01, 0.1, 1.0, 5.0]:
            try:
                res = minimize(loss_free_d, [a_init, -1.0, gamma_init],
                              method="Nelder-Mead", options={"maxiter": 5000})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_params = res.x
            except:
                pass

    if best_params is not None:
        a5, b5, gamma5 = best_params
        x5 = kappas * np.power(ds, gamma5) / np.sqrt(Ks)
        q5 = expit(a5 * x5 + b5)
        r2_5 = 1 - np.sum((qs - q5)**2) / np.sum((qs - qs.mean())**2)
        mae_5 = np.mean(np.abs(qs - q5))
        print(f"\n  Model 5 (FREE d): sigmoid({a5:.5f}*kappa*d^{gamma5:.4f}/sqrt(K) + {b5:.3f})")
        print(f"    R^2 = {r2_5:.6f}, MAE = {mae_5:.4f}")
        print(f"    Best-fit gamma = {gamma5:.4f}")
        print(f"    (gamma=0 means d doesn't matter; gamma=1 means kappa*d)")
    else:
        r2_5, gamma5 = 0, 0
        print("  Model 5: fit failed")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: DIMENSION CANCELLATION")
    print("=" * 70)

    models = [
        ("kappa/sqrt(K)", r2_1, "empirical law"),
        ("kappa*d/sqrt(K)", r2_2, "theory: kappa*d"),
        ("kappa*sqrt(d)/sqrt(K)", r2_3, "theory: kappa*sqrt(d)"),
        ("probit(kappa/sqrt(K))", r2_4, "Gaussian theory"),
        (f"kappa*d^{gamma5:.3f}/sqrt(K)", r2_5, "free d-power"),
    ]

    print(f"\n  {'Model':>35} {'R^2':>10} {'Notes':>20}")
    print(f"  {'-'*35} {'-'*10} {'-'*20}")
    for name, r2, notes in sorted(models, key=lambda x: -x[1]):
        print(f"  {name:>35} {r2:>10.6f} {notes:>20}")

    # Key test
    d_improves = r2_2 > r2_1 + 0.01 or r2_3 > r2_1 + 0.01
    print(f"\n  DOES d IMPROVE THE FIT? {'YES' if d_improves else 'NO'}")
    print(f"  Best gamma for d: {gamma5:.4f}")
    if abs(gamma5) < 0.1:
        print(f"  gamma ~ 0: DIMENSION CANCELLATION CONFIRMED!")
    elif abs(gamma5 - 1.0) < 0.2:
        print(f"  gamma ~ 1: Theory (kappa*d) is correct")
    elif abs(gamma5 - 0.5) < 0.15:
        print(f"  gamma ~ 0.5: kappa*sqrt(d) is the right variable")
    else:
        print(f"  gamma = {gamma5:.4f}: intermediate case")

    # ================================================================
    # SCORECARD
    # ================================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print("=" * 70)

    checks = [
        ("kappa/sqrt(K) R^2 > 0.90 on real data",
         r2_1 > 0.90, f"R^2={r2_1:.4f}"),
        ("Adding d does NOT improve R^2 by > 0.01",
         not d_improves, f"d-free:{r2_1:.4f}, kappa*d:{r2_2:.4f}"),
        ("Free gamma for d is close to 0 (within 0.2)",
         abs(gamma5) < 0.2, f"gamma={gamma5:.4f}"),
        ("probit fits as well as sigmoid",
         abs(r2_4 - r2_1) < 0.02, f"probit:{r2_4:.4f}, sigmoid:{r2_1:.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "dimension_cancellation_test",
        "n_points": len(valid),
        "models": {name: {"r2": float(r2)} for name, r2, _ in models},
        "free_gamma": float(gamma5),
        "d_improves": d_improves,
        "passes": passes,
    }

    out_path = RESULTS_DIR / "cti_dim_cancellation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
