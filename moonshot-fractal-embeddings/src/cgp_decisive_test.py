#!/usr/bin/env python
"""
cgp_decisive_test.py

Decisive test: Does delta_G calibrated from small models predict delta_kNN on bge-base?

This is the final go/no-go test for CGP's invariant universality claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_training_deltas():
    """Load Week 3 data and compute deltas from frozen baseline."""
    path = RESULTS_DIR / "cgp_week3_cross_arch.json"
    with open(path) as f:
        data = json.load(f)

    # Get frozen baselines (lambda_sep=0, seed=42)
    baselines = {}
    for r in data["results"]:
        if r.get("is_baseline"):
            for ds, m in r["eval"].items():
                if "error" not in m:
                    baselines[(r["model"], ds)] = {
                        "knn_l1": m["knn_l1"],
                        "G": m["composite_G"],
                    }

    # Compute deltas for all trained conditions
    deltas = []
    for r in data["results"]:
        if r.get("is_baseline"):
            continue
        for ds, m in r["eval"].items():
            if "error" in m:
                continue
            key = (r["model"], ds)
            if key not in baselines:
                continue
            b = baselines[key]
            G = m.get("composite_G", 0)
            knn = m.get("knn_l1", 0)
            deltas.append({
                "model": r["model"],
                "dataset": ds,
                "lambda_sep": r["lambda_sep"],
                "seed": r["seed"],
                "delta_G": G - b["G"],
                "delta_knn": knn - b["knn_l1"],
                "G_abs": G,
                "knn_abs": knn,
            })
    return deltas


def load_bge_base_results():
    """Load bge-base evaluation results."""
    path = RESULTS_DIR / "cgp_prospective_evaluation.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("actual_results", [])


def main():
    print("=" * 70)
    print("  DECISIVE TEST: Cross-Architecture Relative Universality")
    print("=" * 70)

    # Load training data (small models)
    train_deltas = load_training_deltas()
    print(f"\nTraining deltas: {len(train_deltas)} conditions from 3 small models")

    # Fit delta_G -> delta_kNN on training data
    dG_train = np.array([d["delta_G"] for d in train_deltas])
    dk_train = np.array([d["delta_knn"] for d in train_deltas])

    # Simple linear: delta_knn = slope * delta_G + intercept
    reg = LinearRegression().fit(dG_train.reshape(-1, 1), dk_train)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    train_r2 = reg.score(dG_train.reshape(-1, 1), dk_train)
    train_rmse = np.sqrt(mean_squared_error(dk_train, reg.predict(dG_train.reshape(-1, 1))))

    print(f"\nTraining fit: delta_knn = {slope:.6f} * delta_G + {intercept:.4f}")
    print(f"  R2 = {train_r2:.4f}, RMSE = {train_rmse:.4f}")
    print(f"  Training rho: {stats.spearmanr(dG_train, dk_train)[0]:.4f}")

    # Load bge-base results
    bge_results = load_bge_base_results()
    if bge_results is None:
        print("\nERROR: No bge-base results found. Run Phase 3 first.")
        sys.exit(1)

    # Find bge-base frozen baseline
    baseline = None
    for r in bge_results:
        if r.get("is_baseline"):
            baseline = r
            break

    if baseline is None:
        print("ERROR: No bge-base baseline found")
        sys.exit(1)

    base_metrics = {}
    for ds, m in baseline["eval"].items():
        if "error" not in m:
            base_metrics[ds] = {
                "knn_l1": m["knn_l1"],
                "G": m["composite_G"],
            }

    print(f"\nbge-base baseline:")
    for ds, bm in base_metrics.items():
        print(f"  {ds}: knn_l1={bm['knn_l1']:.3f}, G={bm['G']:.1f}")

    # Compute deltas for bge-base
    test_deltas = []
    for r in bge_results:
        if r.get("is_baseline"):
            continue
        for ds, m in r["eval"].items():
            if "error" in m or ds not in base_metrics:
                continue
            b = base_metrics[ds]
            G = m.get("composite_G", 0)
            knn = m.get("knn_l1", 0)
            delta_G = G - b["G"]
            delta_knn = knn - b["knn_l1"]
            predicted_delta = slope * delta_G + intercept

            test_deltas.append({
                "dataset": ds,
                "lambda_sep": r["lambda_sep"],
                "seed": r["seed"],
                "delta_G": delta_G,
                "actual_delta_knn": delta_knn,
                "predicted_delta_knn": predicted_delta,
                "residual": delta_knn - predicted_delta,
            })

    if len(test_deltas) < 5:
        print(f"ERROR: Only {len(test_deltas)} test conditions. Need >= 5.")
        sys.exit(1)

    actual = np.array([d["actual_delta_knn"] for d in test_deltas])
    predicted = np.array([d["predicted_delta_knn"] for d in test_deltas])
    delta_Gs = np.array([d["delta_G"] for d in test_deltas])

    # Primary metrics
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(actual - predicted))
    rho, rho_p = stats.spearmanr(delta_Gs, actual)

    print(f"\n{'='*70}")
    print(f"  RESULTS: bge-base Held-Out Test (n={len(test_deltas)})")
    print(f"{'='*70}")
    print(f"  Prediction R2:  {r2:.4f}  {'PASS' if r2 > 0.0 else 'FAIL'} (>0 = better than mean)")
    print(f"  Prediction RMSE: {rmse:.4f}")
    print(f"  Prediction MAE:  {mae:.4f}")
    print(f"  DeltaG->DeltakNN rho: {rho:.4f} (p={rho_p:.4e})")

    # Direction accuracy: does sign(predicted delta) match sign(actual delta)?
    sign_match = np.sum(np.sign(predicted) == np.sign(actual)) / len(actual)
    print(f"  Direction accuracy: {sign_match:.1%}")

    # Breakdown by dataset
    print(f"\n  Per-dataset breakdown:")
    for ds in sorted(set(d["dataset"] for d in test_deltas)):
        pts = [d for d in test_deltas if d["dataset"] == ds]
        act = [d["actual_delta_knn"] for d in pts]
        pred = [d["predicted_delta_knn"] for d in pts]
        print(f"    {ds:18s}: actual=[{min(act):.3f}, {max(act):.3f}], "
              f"pred=[{min(pred):.3f}, {max(pred):.3f}]")

    # Pre-registered success criteria
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    pass_r2 = r2 > 0.0
    pass_rho = rho > 0.3 and rho_p < 0.05
    pass_direction = sign_match > 0.6

    print(f"  R2 > 0 (better than baseline): {'PASS' if pass_r2 else 'FAIL'}")
    print(f"  rho > 0.3 with p < 0.05:       {'PASS' if pass_rho else 'FAIL'}")
    print(f"  Direction accuracy > 60%:       {'PASS' if pass_direction else 'FAIL'}")

    overall = pass_r2 and pass_rho and pass_direction
    print(f"\n  OVERALL: {'PASS - Relative universality has signal' if overall else 'FAIL - Stop invariant line'}")

    # Save results
    output = {
        "test": "decisive_cross_architecture",
        "train_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(train_r2),
            "rmse": float(train_rmse),
            "n_train": len(train_deltas),
        },
        "bge_base_test": {
            "n_test": len(test_deltas),
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "rho": float(rho),
            "rho_p": float(rho_p),
            "direction_accuracy": float(sign_match),
            "pass_r2": bool(pass_r2),
            "pass_rho": bool(pass_rho),
            "pass_direction": bool(pass_direction),
            "overall_pass": bool(overall),
        },
        "test_deltas": test_deltas,
    }

    out_path = RESULTS_DIR / "cgp_decisive_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
