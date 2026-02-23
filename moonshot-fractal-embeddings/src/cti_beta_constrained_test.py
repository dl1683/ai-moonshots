"""
CTI Beta-Constrained Model Comparison
Pre-registered: Feb 23, 2026

Tests whether sparse competition (beta=-0.5) or dense ETF competition (beta=-1)
better fits the real-network kappa_near_cache data.

Three models:
  M0: unconstrained -- alpha, beta, C all free
  M1: beta=-1 (dense ETF: N_eff = K-1, logit = A*kappa + (-1)*log(K-1) + C)
  M2: beta=-0.5 (sparse competition: N_eff ~ sqrt(K), logit = A*kappa + (-0.5)*log(K-1) + C)
  M3: beta=0 (K-independent, pure kappa law)

Evaluation:
  - 5-fold LOAO (leave-one-architecture-out) MAE
  - 5-fold LODO (leave-one-dataset-out) MAE
  - WINNER: model with lowest LOAO MAE

Pre-registered thresholds:
  PR_CONSTRAINED_WIN: M2 (beta=-0.5) has LOAO MAE <= M1 (beta=-1) LOAO MAE
  PR_CLOSE: |M2_MAE - M1_MAE| / M1_MAE >= 0.03 (3% relative improvement)
  PR_UNCON_GAP: M0 LOAO MAE within 5% of winner (unconstrained doesn't help much)
"""

import json
import os
import numpy as np
from scipy.special import logit as scipy_logit
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CACHE_PATTERN = "kappa_near_cache_"

TIMESTAMP = datetime.now().isoformat()

print("=" * 70)
print("CTI BETA-CONSTRAINED MODEL COMPARISON")
print(f"Timestamp: {TIMESTAMP}")
print("=" * 70)
print()
print("Pre-registered (Feb 23, 2026 -- committed BEFORE running):")
print("  M0: unconstrained (alpha, beta, C free)")
print("  M1: beta=-1 (dense ETF)")
print("  M2: beta=-0.5 (sparse competition)")
print("  M3: beta=0 (K-independent)")
print("  PR_WIN: M2 LOAO_MAE <= M1 LOAO_MAE")
print("  PR_CLOSE: |M2 - M1| / M1 >= 0.03 (3% relative improvement)")
print()


def load_all_points():
    """Load all valid (kappa, q, K, model, dataset) points."""
    points = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith(CACHE_PATTERN) or not fname.endswith(".json"):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        model = data.get("model", "")
        dataset = data.get("dataset", "")

        # Two formats
        if "layers" in data:
            for layer_data in data["layers"].values():
                kappa = layer_data.get("kappa_nearest") or layer_data.get("kappa_min")
                K = layer_data.get("K")
                q_raw = layer_data.get("q_raw") or layer_data.get("q_norm")
                if kappa is None or K is None or q_raw is None:
                    continue
                if kappa <= 0 or K < 3 or q_raw <= 1.0 / K or q_raw >= 1.0:
                    continue
                logit_q = float(scipy_logit(q_raw))
                log_km1 = float(np.log(K - 1))
                points.append({
                    "kappa": float(kappa),
                    "logit_q": logit_q,
                    "log_km1": log_km1,
                    "K": K,
                    "model": model,
                    "dataset": dataset,
                })
        elif "data" in data:
            for entry in data["data"]:
                kappa = entry.get("kappa_nearest") or entry.get("kappa_min")
                K = entry.get("K")
                q_raw = entry.get("q_raw") or entry.get("q_norm")
                if kappa is None or K is None or q_raw is None:
                    continue
                if kappa <= 0 or K < 3 or q_raw <= 1.0 / K or q_raw >= 1.0:
                    continue
                logit_q = float(scipy_logit(q_raw))
                log_km1 = float(np.log(K - 1))
                points.append({
                    "kappa": float(kappa),
                    "logit_q": logit_q,
                    "log_km1": log_km1,
                    "K": K,
                    "model": model,
                    "dataset": dataset,
                })
    return points


def fit_unconstrained(kappas, log_km1s, logit_qs):
    """Fit logit(q) = alpha*kappa + beta*log(K-1) + C (all free)."""
    X = np.column_stack([kappas, log_km1s, np.ones(len(kappas))])
    y = np.array(logit_qs)
    result = np.linalg.lstsq(X, y, rcond=None)
    coeffs = result[0]
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])  # alpha, beta, C


def fit_beta_fixed(kappas, log_km1s, logit_qs, beta_fixed):
    """Fit logit(q) = alpha*kappa + C, with beta pre-subtracted."""
    # logit(q) - beta_fixed * log(K-1) = alpha*kappa + C
    y_adj = np.array(logit_qs) - beta_fixed * np.array(log_km1s)
    X = np.column_stack([kappas, np.ones(len(kappas))])
    result = np.linalg.lstsq(X, y_adj, rcond=None)
    coeffs = result[0]
    return float(coeffs[0]), float(coeffs[1])  # alpha, C


def predict_unconstrained(alpha, beta, C, kappas, log_km1s):
    return alpha * np.array(kappas) + beta * np.array(log_km1s) + C


def predict_beta_fixed(alpha, beta_fixed, C, kappas, log_km1s):
    return alpha * np.array(kappas) + beta_fixed * np.array(log_km1s) + C


def loao_cv(points, model_key="model"):
    """Leave-one-architecture-out CV. Returns per-model MAE for each model."""
    groups = sorted(set(p[model_key] for p in points))
    results = {"M0": [], "M1": [], "M2": [], "M3": []}

    for g in groups:
        train = [p for p in points if p[model_key] != g]
        test = [p for p in points if p[model_key] == g]
        if len(train) < 10 or len(test) < 2:
            continue

        kappas_tr = [p["kappa"] for p in train]
        log_km1s_tr = [p["log_km1"] for p in train]
        logit_qs_tr = [p["logit_q"] for p in train]

        kappas_te = np.array([p["kappa"] for p in test])
        log_km1s_te = np.array([p["log_km1"] for p in test])
        logit_qs_te = np.array([p["logit_q"] for p in test])

        # M0: unconstrained
        a0, b0, c0 = fit_unconstrained(kappas_tr, log_km1s_tr, logit_qs_tr)
        pred0 = predict_unconstrained(a0, b0, c0, kappas_te, log_km1s_te)
        results["M0"].append(float(np.mean(np.abs(pred0 - logit_qs_te))))

        # M1: beta=-1
        a1, c1 = fit_beta_fixed(kappas_tr, log_km1s_tr, logit_qs_tr, -1.0)
        pred1 = predict_beta_fixed(a1, -1.0, c1, kappas_te, log_km1s_te)
        results["M1"].append(float(np.mean(np.abs(pred1 - logit_qs_te))))

        # M2: beta=-0.5
        a2, c2 = fit_beta_fixed(kappas_tr, log_km1s_tr, logit_qs_tr, -0.5)
        pred2 = predict_beta_fixed(a2, -0.5, c2, kappas_te, log_km1s_te)
        results["M2"].append(float(np.mean(np.abs(pred2 - logit_qs_te))))

        # M3: beta=0
        a3, c3 = fit_beta_fixed(kappas_tr, log_km1s_tr, logit_qs_tr, 0.0)
        pred3 = predict_beta_fixed(a3, 0.0, c3, kappas_te, log_km1s_te)
        results["M3"].append(float(np.mean(np.abs(pred3 - logit_qs_te))))

    return {
        k: {
            "mean_mae": float(np.mean(v)),
            "std_mae": float(np.std(v)),
            "n_groups": len(v),
            "per_group": v,
        }
        for k, v in results.items()
        if v
    }, groups


def lodo_cv(points):
    """Leave-one-dataset-out CV."""
    datasets = sorted(set(p["dataset"] for p in points))
    results = {"M0": [], "M1": [], "M2": [], "M3": []}

    for d in datasets:
        train = [p for p in points if p["dataset"] != d]
        test = [p for p in points if p["dataset"] == d]
        if len(train) < 10 or len(test) < 2:
            continue

        kappas_tr = [p["kappa"] for p in train]
        log_km1s_tr = [p["log_km1"] for p in train]
        logit_qs_tr = [p["logit_q"] for p in train]

        kappas_te = np.array([p["kappa"] for p in test])
        log_km1s_te = np.array([p["log_km1"] for p in test])
        logit_qs_te = np.array([p["logit_q"] for p in test])

        a0, b0, c0 = fit_unconstrained(kappas_tr, log_km1s_tr, logit_qs_tr)
        pred0 = predict_unconstrained(a0, b0, c0, kappas_te, log_km1s_te)
        results["M0"].append(float(np.mean(np.abs(pred0 - logit_qs_te))))

        a1, c1 = fit_beta_fixed(kappas_tr, log_km1s_tr, logit_qs_tr, -1.0)
        pred1 = predict_beta_fixed(a1, -1.0, c1, kappas_te, log_km1s_te)
        results["M1"].append(float(np.mean(np.abs(pred1 - logit_qs_te))))

        a2, c2 = fit_beta_fixed(kappas_tr, log_km1s_tr, logit_qs_tr, -0.5)
        pred2 = predict_beta_fixed(a2, -0.5, c2, kappas_te, log_km1s_te)
        results["M2"].append(float(np.mean(np.abs(pred2 - logit_qs_te))))

        a3, c3 = fit_beta_fixed(kappas_tr, log_km1s_tr, logit_qs_tr, 0.0)
        pred3 = predict_beta_fixed(a3, 0.0, c3, kappas_te, log_km1s_te)
        results["M3"].append(float(np.mean(np.abs(pred3 - logit_qs_te))))

    return {
        k: {
            "mean_mae": float(np.mean(v)),
            "std_mae": float(np.std(v)),
            "n_groups": len(v),
            "per_group": v,
        }
        for k, v in results.items()
        if v
    }, datasets


# ============================================================
# MAIN
# ============================================================
print("Loading kappa_near_cache data...")
points = load_all_points()
print(f"  Loaded {len(points)} valid points from cache files")

if len(points) < 50:
    print("ERROR: Too few points loaded. Check cache files.")
    import sys
    sys.exit(1)

kappas_all = [p["kappa"] for p in points]
log_km1s_all = [p["log_km1"] for p in points]
logit_qs_all = [p["logit_q"] for p in points]

# ============================================================
# GLOBAL FIT (all data)
# ============================================================
print("\nGlobal fits (all data):")
a0_g, b0_g, c0_g = fit_unconstrained(kappas_all, log_km1s_all, logit_qs_all)
print(f"  M0 (unconstrained): alpha={a0_g:.4f}, beta={b0_g:.4f}, C={c0_g:.4f}")

a1_g, c1_g = fit_beta_fixed(kappas_all, log_km1s_all, logit_qs_all, -1.0)
print(f"  M1 (beta=-1):       alpha={a1_g:.4f}, C={c1_g:.4f}")

a2_g, c2_g = fit_beta_fixed(kappas_all, log_km1s_all, logit_qs_all, -0.5)
print(f"  M2 (beta=-0.5):     alpha={a2_g:.4f}, C={c2_g:.4f}")

a3_g, c3_g = fit_beta_fixed(kappas_all, log_km1s_all, logit_qs_all, 0.0)
print(f"  M3 (beta=0):        alpha={a3_g:.4f}, C={c3_g:.4f}")

# Global MAE
pred_all = {
    "M0": predict_unconstrained(a0_g, b0_g, c0_g, kappas_all, log_km1s_all),
    "M1": predict_beta_fixed(a1_g, -1.0, c1_g, kappas_all, log_km1s_all),
    "M2": predict_beta_fixed(a2_g, -0.5, c2_g, kappas_all, log_km1s_all),
    "M3": predict_beta_fixed(a3_g, 0.0, c3_g, kappas_all, log_km1s_all),
}
logit_arr = np.array(logit_qs_all)
global_mae = {}
for name, pred in pred_all.items():
    mae = float(np.mean(np.abs(np.array(pred) - logit_arr)))
    r2 = float(1 - np.var(np.array(pred) - logit_arr) / np.var(logit_arr))
    global_mae[name] = mae
    print(f"  {name} global MAE={mae:.4f}, R2={r2:.4f}")

# ============================================================
# LOAO CV
# ============================================================
print("\nRunning LOAO CV...")
loao_results, loao_groups = loao_cv(points)
print(f"  Groups tested: {loao_results['M0']['n_groups']} architectures")
print("\n  LOAO MAE:")
for name in ["M0", "M1", "M2", "M3"]:
    r = loao_results[name]
    print(f"    {name}: {r['mean_mae']:.4f} +/- {r['std_mae']:.4f}")

# ============================================================
# LODO CV
# ============================================================
print("\nRunning LODO CV...")
lodo_results, lodo_datasets = lodo_cv(points)
print(f"  Datasets tested: {lodo_results['M0']['n_groups']}")
print("\n  LODO MAE:")
for name in ["M0", "M1", "M2", "M3"]:
    r = lodo_results[name]
    print(f"    {name}: {r['mean_mae']:.4f} +/- {r['std_mae']:.4f}")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

loao_m1 = loao_results["M1"]["mean_mae"]
loao_m2 = loao_results["M2"]["mean_mae"]
loao_m0 = loao_results["M0"]["mean_mae"]
loao_m3 = loao_results["M3"]["mean_mae"]

pr_win = loao_m2 <= loao_m1
rel_improvement = (loao_m1 - loao_m2) / loao_m1
pr_close = abs(rel_improvement) >= 0.03
pr_uncon_gap = (loao_m0 - min(loao_m1, loao_m2)) / min(loao_m1, loao_m2) <= 0.05

winner_name = min(["M0", "M1", "M2", "M3"],
                  key=lambda x: loao_results[x]["mean_mae"])

print(f"\nLOAO MAE ranking:")
print(f"  M2 (beta=-0.5): {loao_m2:.4f}")
print(f"  M1 (beta=-1.0): {loao_m1:.4f}")
print(f"  M0 (unconstrained): {loao_m0:.4f}")
print(f"  M3 (beta=0): {loao_m3:.4f}")
print(f"\nOverall winner (lowest LOAO MAE): {winner_name}")
print(f"\nPR_WIN (M2 <= M1): {'PASS' if pr_win else 'FAIL'}")
print(f"  M2 MAE={loao_m2:.4f} vs M1 MAE={loao_m1:.4f}")
if pr_win:
    print(f"  Relative improvement: {rel_improvement*100:.1f}%")
else:
    print(f"  Relative gap (M2 worse by): {-rel_improvement*100:.1f}%")

print(f"PR_CLOSE (>=3% improvement): {'PASS' if pr_close else 'FAIL'}")
print(f"PR_UNCON_GAP (unconstrained within 5% of winner): {'PASS' if pr_uncon_gap else 'FAIL'}")

# Per-K breakdown
print("\nPer-K model comparison (global fit MAE):")
for K_target in sorted(set(p["K"] for p in points)):
    pts_k = [p for p in points if p["K"] == K_target]
    if len(pts_k) < 5:
        continue
    kk = [p["kappa"] for p in pts_k]
    lk = [p["log_km1"] for p in pts_k]
    qq = [p["logit_q"] for p in pts_k]

    mae_m1 = float(np.mean(np.abs(predict_beta_fixed(a1_g, -1.0, c1_g, kk, lk) - np.array(qq))))
    mae_m2 = float(np.mean(np.abs(predict_beta_fixed(a2_g, -0.5, c2_g, kk, lk) - np.array(qq))))
    winner = "M2" if mae_m2 < mae_m1 else "M1"
    print(f"  K={K_target:3d} (n={len(pts_k):3d}): M1={mae_m1:.3f}, M2={mae_m2:.3f} -> {winner}")

# ============================================================
# OUTPUT JSON
# ============================================================
output = {
    "experiment": "beta_constrained_model_comparison",
    "timestamp": TIMESTAMP,
    "n_total_points": len(points),
    "n_models": len(set(p["model"] for p in points)),
    "n_datasets": len(set(p["dataset"] for p in points)),
    "pre_registered": {
        "PR_WIN": "M2 (beta=-0.5) LOAO MAE <= M1 (beta=-1) LOAO MAE",
        "PR_CLOSE": "relative improvement >= 3%",
        "PR_UNCON_GAP": "unconstrained within 5% of winner",
    },
    "global_fits": {
        "M0": {"alpha": a0_g, "beta": b0_g, "C": c0_g, "global_mae": global_mae["M0"]},
        "M1": {"alpha": a1_g, "beta": -1.0, "C": c1_g, "global_mae": global_mae["M1"]},
        "M2": {"alpha": a2_g, "beta": -0.5, "C": c2_g, "global_mae": global_mae["M2"]},
        "M3": {"alpha": a3_g, "beta": 0.0, "C": c3_g, "global_mae": global_mae["M3"]},
    },
    "loao_cv": loao_results,
    "lodo_cv": lodo_results,
    "verdict": {
        "winner": winner_name,
        "PR_WIN": pr_win,
        "PR_CLOSE": pr_close,
        "PR_UNCON_GAP": pr_uncon_gap,
        "loao_m1": loao_m1,
        "loao_m2": loao_m2,
        "loao_m0": loao_m0,
        "loao_m3": loao_m3,
        "m2_relative_improvement_over_m1": rel_improvement,
    },
}

out_path = os.path.join(RESULTS_DIR, "cti_beta_constrained.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {out_path}")
