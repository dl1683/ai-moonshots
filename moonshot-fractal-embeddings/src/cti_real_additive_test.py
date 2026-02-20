#!/usr/bin/env python
"""
REAL DATA ADDITIVE TEST: Does the additive form work on real networks?

Theory (confirmed on synthetic Gaussians):
  q = Phi(a*kappa*sqrt(d) + b*log(K) + c)    [additive, WINS]
  NOT: q = Phi(a*kappa*sqrt(d)/log(K) + c)     [divisive, loses]

But on real networks, we expect d_eff to cancel through kappa normalization.
So the real-network law should be:
  q = Phi(a*kappa + b*log(K) + c)              [additive, d-free]
  OR: q = sigmoid(a*kappa + b*log(K) + c)      [sigmoid approximation]

This test compares additive vs divisive on ALL real data points.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import expit

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
                "kappa": p["kappa"], "knn": knn, "q": q, "d": d,
                "alpha": p["alpha"],
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
                    "kappa": p["kappa"], "knn": knn, "q": q, "d": d,
                    "alpha": p["alpha"],
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
                "kappa": p["kappa"], "knn": knn, "q": q, "d": d,
                "alpha": p["alpha"],
            })
    except Exception as e:
        print(f"  Blind load failed: {e}")

    return all_points


def main():
    print("=" * 70)
    print("REAL DATA: ADDITIVE vs DIVISIVE log(K) on neural networks")
    print("=" * 70)

    all_points = load_all_points()
    print(f"\nLoaded {len(all_points)} points")

    # Filter valid points
    valid = [p for p in all_points if 0.01 < p["q"] < 0.99 and p["kappa"] > 0]
    print(f"Valid points (0.01 < q < 0.99): {len(valid)}")

    if len(valid) < 10:
        print("Not enough valid points!")
        return

    kappas = np.array([p["kappa"] for p in valid])
    qs = np.array([p["q"] for p in valid])
    Ks = np.array([float(p["K"]) for p in valid])
    ds = np.array([float(p["d"]) for p in valid])

    datasets = set(p["dataset"] for p in valid)
    print(f"Datasets: {datasets}")
    print(f"K range: {Ks.min():.0f} - {Ks.max():.0f}")
    print(f"d range: {ds.min():.0f} - {ds.max():.0f}")
    print(f"kappa range: {kappas.min():.4f} - {kappas.max():.4f}")

    print(f"\n{'='*70}")
    print("MODEL COMPARISON (d-free, since real networks)")
    print("=" * 70)

    # =============================================================
    # MODEL 1: sigmoid(a*kappa + b*log(K) + c) [ADDITIVE, d-free]
    # =============================================================
    def loss_add_sig(params):
        a, b, c = params
        x = a * kappas + b * np.log(Ks + 1) + c
        pred = expit(x)
        return np.sum((qs - pred) ** 2)

    best_add = None
    best_add_loss = float("inf")
    for a0 in [1.0, 5.0, 10.0, 20.0]:
        for b0 in [-2.0, -1.0, -0.5, 0.0]:
            for c0 in [-5.0, -3.0, -1.0, 0.0]:
                try:
                    res = minimize(loss_add_sig, [a0, b0, c0],
                                   method="Nelder-Mead", options={"maxiter": 10000})
                    if res.fun < best_add_loss:
                        best_add_loss = res.fun
                        best_add = res.x
                except:
                    pass

    a1, b1, c1 = best_add
    q1 = expit(a1 * kappas + b1 * np.log(Ks + 1) + c1)
    r2_add = 1 - np.sum((qs - q1)**2) / np.sum((qs - qs.mean())**2)
    mae_add = np.mean(np.abs(qs - q1))
    print(f"\n  ADDITIVE: sigmoid({a1:.3f}*kappa + {b1:.3f}*log(K+1) + {c1:.3f})")
    print(f"    R^2 = {r2_add:.6f}, MAE = {mae_add:.4f}")

    # =============================================================
    # MODEL 2: sigmoid(a*kappa/log(K+1) + c) [DIVISIVE, d-free]
    # =============================================================
    def loss_div_sig(params):
        a, c = params
        x = a * kappas / np.log(Ks + 1) + c
        pred = expit(x)
        return np.sum((qs - pred) ** 2)

    best_div = None
    best_div_loss = float("inf")
    for a0 in [1.0, 5.0, 10.0, 20.0, 50.0]:
        for c0 in [-5.0, -3.0, -1.0, 0.0]:
            try:
                res = minimize(loss_div_sig, [a0, c0],
                               method="Nelder-Mead", options={"maxiter": 10000})
                if res.fun < best_div_loss:
                    best_div_loss = res.fun
                    best_div = res.x
            except:
                pass

    a2, c2 = best_div
    q2 = expit(a2 * kappas / np.log(Ks + 1) + c2)
    r2_div = 1 - np.sum((qs - q2)**2) / np.sum((qs - qs.mean())**2)
    mae_div = np.mean(np.abs(qs - q2))
    print(f"\n  DIVISIVE: sigmoid({a2:.3f}*kappa/log(K+1) + {c2:.3f})")
    print(f"    R^2 = {r2_div:.6f}, MAE = {mae_div:.4f}")

    # =============================================================
    # MODEL 3: sigmoid(a*kappa/sqrt(K) + c) [OLD empirical law]
    # =============================================================
    def loss_sqrtK(params):
        a, c = params
        x = a * kappas / np.sqrt(Ks) + c
        pred = expit(x)
        return np.sum((qs - pred) ** 2)

    best_sqrtK = None
    best_sqrtK_loss = float("inf")
    for a0 in [1.0, 5.0, 10.0, 20.0, 50.0]:
        for c0 in [-5.0, -3.0, -1.0, 0.0]:
            try:
                res = minimize(loss_sqrtK, [a0, c0],
                               method="Nelder-Mead", options={"maxiter": 10000})
                if res.fun < best_sqrtK_loss:
                    best_sqrtK_loss = res.fun
                    best_sqrtK = res.x
            except:
                pass

    a3, c3 = best_sqrtK
    q3 = expit(a3 * kappas / np.sqrt(Ks) + c3)
    r2_sqrtK = 1 - np.sum((qs - q3)**2) / np.sum((qs - qs.mean())**2)
    mae_sqrtK = np.mean(np.abs(qs - q3))
    print(f"\n  SQRT(K): sigmoid({a3:.3f}*kappa/sqrt(K) + {c3:.3f})")
    print(f"    R^2 = {r2_sqrtK:.6f}, MAE = {mae_sqrtK:.4f}")

    # =============================================================
    # MODEL 4: sigmoid(a*kappa + b*log(K) + c*sqrt(d) + e) [WITH d]
    # =============================================================
    def loss_add_d(params):
        a, b, c, e = params
        x = a * kappas + b * np.log(Ks + 1) + c * np.sqrt(ds) + e
        pred = expit(x)
        return np.sum((qs - pred) ** 2)

    best_add_d = None
    best_add_d_loss = float("inf")
    for a0 in [5.0, 10.0, 20.0]:
        for b0 in [-1.0, -0.5]:
            for c0 in [-0.1, 0.0, 0.1]:
                for e0 in [-3.0, -1.0]:
                    try:
                        res = minimize(loss_add_d, [a0, b0, c0, e0],
                                       method="Nelder-Mead", options={"maxiter": 10000})
                        if res.fun < best_add_d_loss:
                            best_add_d_loss = res.fun
                            best_add_d = res.x
                    except:
                        pass

    a4, b4, c4, e4 = best_add_d
    q4 = expit(a4 * kappas + b4 * np.log(Ks + 1) + c4 * np.sqrt(ds) + e4)
    r2_add_d = 1 - np.sum((qs - q4)**2) / np.sum((qs - qs.mean())**2)
    mae_add_d = np.mean(np.abs(qs - q4))
    print(f"\n  ADDITIVE+d: sigmoid({a4:.3f}*kappa + {b4:.3f}*log(K+1) + {c4:.4f}*sqrt(d) + {e4:.3f})")
    print(f"    R^2 = {r2_add_d:.6f}, MAE = {mae_add_d:.4f}")
    print(f"    (d coefficient: {c4:.4f} -- if ~0, dimension cancellation confirmed)")

    # =============================================================
    # MODEL 5: Probit additive: Phi(a*kappa + b*log(K) + c)
    # =============================================================
    def loss_probit_add(params):
        a, b, c = params
        x = a * kappas + b * np.log(Ks + 1) + c
        pred = norm.cdf(x)
        return np.sum((qs - pred) ** 2)

    best_probit = None
    best_probit_loss = float("inf")
    for a0 in [1.0, 3.0, 5.0, 10.0]:
        for b0 in [-1.0, -0.5, 0.0]:
            for c0 in [-3.0, -1.0, 0.0]:
                try:
                    res = minimize(loss_probit_add, [a0, b0, c0],
                                   method="Nelder-Mead", options={"maxiter": 10000})
                    if res.fun < best_probit_loss:
                        best_probit_loss = res.fun
                        best_probit = res.x
                except:
                    pass

    a5, b5, c5 = best_probit
    q5 = norm.cdf(a5 * kappas + b5 * np.log(Ks + 1) + c5)
    r2_probit = 1 - np.sum((qs - q5)**2) / np.sum((qs - qs.mean())**2)
    mae_probit = np.mean(np.abs(qs - q5))
    print(f"\n  PROBIT-ADD: Phi({a5:.3f}*kappa + {b5:.3f}*log(K+1) + {c5:.3f})")
    print(f"    R^2 = {r2_probit:.6f}, MAE = {mae_probit:.4f}")

    # =============================================================
    # PER-DATASET ANALYSIS
    # =============================================================
    print(f"\n{'='*70}")
    print("PER-DATASET BREAKDOWN")
    print("=" * 70)

    for ds_name in sorted(datasets):
        ds_mask = np.array([p["dataset"] == ds_name for p in valid])
        if ds_mask.sum() < 3:
            continue
        kap_ds = kappas[ds_mask]
        qs_ds = qs[ds_mask]
        Ks_ds = Ks[ds_mask]

        # Additive prediction
        q_add_ds = expit(a1 * kap_ds + b1 * np.log(Ks_ds + 1) + c1)
        r2_ds_add = 1 - np.sum((qs_ds - q_add_ds)**2) / max(np.sum((qs_ds - qs_ds.mean())**2), 1e-10)
        mae_ds_add = np.mean(np.abs(qs_ds - q_add_ds))

        # Divisive prediction
        q_div_ds = expit(a2 * kap_ds / np.log(Ks_ds + 1) + c2)
        r2_ds_div = 1 - np.sum((qs_ds - q_div_ds)**2) / max(np.sum((qs_ds - qs_ds.mean())**2), 1e-10)
        mae_ds_div = np.mean(np.abs(qs_ds - q_div_ds))

        K_val = int(Ks_ds[0])
        print(f"\n  {ds_name} (K={K_val}, n={ds_mask.sum()}):")
        print(f"    ADDITIVE  R^2={r2_ds_add:.4f}, MAE={mae_ds_add:.4f}")
        print(f"    DIVISIVE  R^2={r2_ds_div:.4f}, MAE={mae_ds_div:.4f}")

    # =============================================================
    # SUMMARY
    # =============================================================
    print(f"\n{'='*70}")
    print("SUMMARY: REAL NETWORK RESULTS")
    print("=" * 70)

    models = [
        ("ADDITIVE: sig(a*k + b*logK + c)", r2_add, mae_add, 3),
        ("DIVISIVE: sig(a*k/logK + c)", r2_div, mae_div, 2),
        ("OLD: sig(a*k/sqrtK + c)", r2_sqrtK, mae_sqrtK, 2),
        ("ADDITIVE+d: sig(a*k + b*logK + c*sqrtd + e)", r2_add_d, mae_add_d, 4),
        ("PROBIT-ADD: Phi(a*k + b*logK + c)", r2_probit, mae_probit, 3),
    ]

    print(f"\n  {'Model':>50} {'R^2':>8} {'MAE':>8} {'#p':>4}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*4}")
    for name, r2, mae, np_ in sorted(models, key=lambda x: -x[1]):
        print(f"  {name:>50} {r2:>8.4f} {mae:>8.4f} {np_:>4}")

    # Key comparisons
    print(f"\n  log(K) vs sqrt(K) on REAL data:")
    if r2_add > r2_sqrtK:
        print(f"    log(K) WINS: R^2 {r2_add:.4f} vs {r2_sqrtK:.4f}")
    else:
        print(f"    sqrt(K) WINS: R^2 {r2_sqrtK:.4f} vs {r2_add:.4f}")

    print(f"\n  Additive vs divisive on REAL data:")
    if r2_add > r2_div + 0.005:
        print(f"    ADDITIVE WINS: R^2 {r2_add:.4f} vs {r2_div:.4f}")
    elif r2_div > r2_add + 0.005:
        print(f"    DIVISIVE WINS: R^2 {r2_div:.4f} vs {r2_add:.4f}")
    else:
        print(f"    INDISTINGUISHABLE: R^2 {r2_add:.4f} vs {r2_div:.4f}")

    print(f"\n  Dimension cancellation:")
    print(f"    sqrt(d) coefficient: {c4:.4f}")
    if abs(c4) < 0.01:
        print(f"    CONFIRMED: d has negligible effect on real networks")
    else:
        print(f"    PARTIAL: d still contributes (R^2 with d: {r2_add_d:.4f} vs without: {r2_add:.4f})")

    # Scorecard
    print(f"\n{'='*70}")
    print("SCORECARD")
    print("=" * 70)

    checks = [
        ("Best real-data R^2 > 0.80",
         max(r2_add, r2_div, r2_sqrtK) > 0.80,
         f"best={max(r2_add, r2_div, r2_sqrtK):.4f}"),
        ("Additive beats divisive (delta > 0.005)",
         r2_add > r2_div + 0.005,
         f"add={r2_add:.4f}, div={r2_div:.4f}"),
        ("log(K) beats sqrt(K) (delta > 0.005)",
         r2_add > r2_sqrtK + 0.005,
         f"logK={r2_add:.4f}, sqrtK={r2_sqrtK:.4f}"),
        ("d coefficient < 0.01 (cancellation)",
         abs(c4) < 0.01,
         f"coeff={c4:.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "real_data_additive_test",
        "n_points": len(valid),
        "datasets": list(datasets),
        "models": {
            "additive_sig": {"r2": float(r2_add), "mae": float(mae_add),
                             "params": [float(a1), float(b1), float(c1)]},
            "divisive_sig": {"r2": float(r2_div), "mae": float(mae_div),
                             "params": [float(a2), float(c2)]},
            "sqrtK_sig": {"r2": float(r2_sqrtK), "mae": float(mae_sqrtK),
                          "params": [float(a3), float(c3)]},
            "additive_with_d": {"r2": float(r2_add_d), "mae": float(mae_add_d),
                                "params": [float(a4), float(b4), float(c4), float(e4)]},
            "probit_additive": {"r2": float(r2_probit), "mae": float(mae_probit),
                                "params": [float(a5), float(b5), float(c5)]},
        },
        "d_coefficient": float(c4),
        "passes": passes,
    }

    out_path = RESULTS_DIR / "cti_real_additive_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
