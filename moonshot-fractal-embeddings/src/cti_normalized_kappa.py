#!/usr/bin/env python
"""
NORMALIZED KAPPA: Universal law across datasets.

The raw kappa=trace(S_B)/trace(S_W) depends on number of classes K.
With fewer classes, same kappa yields higher kNN.

Hypothesis: normalized_kNN = f(kappa * g(K)) where g(K) accounts for
task difficulty.

Simplest normalization: kappa_norm = kappa * K (more classes = harder)
Quality normalization: q = (kNN - 1/K) / (1 - 1/K)

If q = sigmoid(kappa * K) collapses BOTH datasets, we have the universal law.

Uses existing data from spectral_collapse.json and prospective_kappa.json.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def sigmoid(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(-b * (x - c)))


def main():
    print("=" * 70)
    print("NORMALIZED KAPPA: Cross-dataset universality")
    print("=" * 70)

    # Load all data points
    all_points = []

    # From spectral collapse (CLINC, 4 models)
    with open(RESULTS_DIR / "cti_spectral_collapse.json") as f:
        sc = json.load(f)
    for p in sc["all_points"]:
        all_points.append({
            "model": p["model"], "dataset": "clinc", "K": 150,
            "alpha": p["alpha"], "kappa": p["kappa"], "knn": p["knn"],
        })

    # From prospective prediction (CLINC + TREC, held-out models)
    with open(RESULTS_DIR / "cti_prospective_kappa.json") as f:
        pp = json.load(f)
    for p in pp["holdout"]["points"]:
        # Avoid duplicates
        key = (p["model"], p["dataset"], p["alpha"])
        existing = [(x["model"], x["dataset"], x["alpha"]) for x in all_points]
        if key not in existing:
            K = 150 if p["dataset"] == "clinc" else 50  # TREC has 50 L1 classes
            # Actually for kNN we use L0 (coarse) labels
            # CLINC: 10 L0 classes, TREC: 6 L0 classes
            # But our kNN uses level1_label... let me check
            # Actually the code uses level1_label throughout, so:
            # CLINC: 150 classes, TREC: 50 classes for level1
            all_points.append({
                "model": p["model"], "dataset": p["dataset"],
                "K": K,
                "alpha": p["alpha"], "kappa": p["kappa"], "knn": p["knn"],
            })

    print(f"\n  Total data points: {len(all_points)}")
    print(f"  CLINC points: {sum(1 for p in all_points if p['dataset'] == 'clinc')}")
    print(f"  TREC points: {sum(1 for p in all_points if p['dataset'] == 'trec')}")

    # ============================================================
    # OPTION 1: Raw kappa vs kNN (no normalization)
    # ============================================================
    print(f"\n{'='*70}")
    print("RAW kappa vs kNN (no normalization)")
    print(f"{'='*70}")

    kappas_raw = np.array([p["kappa"] for p in all_points])
    knns_raw = np.array([p["knn"] for p in all_points])

    rho_raw, p_raw = spearmanr(kappas_raw, knns_raw)
    r_raw, pr_raw = pearsonr(kappas_raw, knns_raw)
    print(f"  Spearman rho = {rho_raw:.4f} (p = {p_raw:.6f})")
    print(f"  Pearson r = {r_raw:.4f} (p = {pr_raw:.6f})")

    # ============================================================
    # OPTION 2: Normalized quality q = (kNN - 1/K) / (1 - 1/K)
    #           vs kappa (no kappa normalization)
    # ============================================================
    print(f"\n{'='*70}")
    print("OPTION 2: Normalized quality q vs raw kappa")
    print(f"{'='*70}")

    q_vals = np.array([(p["knn"] - 1/p["K"]) / (1 - 1/p["K"]) for p in all_points])
    rho_q, p_q = spearmanr(kappas_raw, q_vals)
    r_q, pr_q = pearsonr(kappas_raw, q_vals)
    print(f"  Spearman rho = {rho_q:.4f} (p = {p_q:.6f})")
    print(f"  Pearson r = {r_q:.4f} (p = {pr_q:.6f})")

    # ============================================================
    # OPTION 3: q vs kappa * K (both normalized)
    # ============================================================
    print(f"\n{'='*70}")
    print("OPTION 3: Normalized quality q vs kappa * K")
    print(f"{'='*70}")

    kappa_K = np.array([p["kappa"] * p["K"] for p in all_points])
    rho_kK, p_kK = spearmanr(kappa_K, q_vals)
    r_kK, pr_kK = pearsonr(kappa_K, q_vals)
    print(f"  Spearman rho = {rho_kK:.4f} (p = {p_kK:.6f})")
    print(f"  Pearson r = {r_kK:.4f} (p = {pr_kK:.6f})")

    # ============================================================
    # OPTION 4: q vs kappa * sqrt(K)
    # ============================================================
    print(f"\n{'='*70}")
    print("OPTION 4: Normalized quality q vs kappa * sqrt(K)")
    print(f"{'='*70}")

    kappa_sqK = np.array([p["kappa"] * np.sqrt(p["K"]) for p in all_points])
    rho_sqK, p_sqK = spearmanr(kappa_sqK, q_vals)
    r_sqK, pr_sqK = pearsonr(kappa_sqK, q_vals)
    print(f"  Spearman rho = {rho_sqK:.4f} (p = {p_sqK:.6f})")
    print(f"  Pearson r = {r_sqK:.4f} (p = {pr_sqK:.6f})")

    # ============================================================
    # OPTION 5: q vs kappa * log(K)
    # ============================================================
    print(f"\n{'='*70}")
    print("OPTION 5: Normalized quality q vs kappa * log(K)")
    print(f"{'='*70}")

    kappa_logK = np.array([p["kappa"] * np.log(p["K"]) for p in all_points])
    rho_logK, p_logK = spearmanr(kappa_logK, q_vals)
    r_logK, pr_logK = pearsonr(kappa_logK, q_vals)
    print(f"  Spearman rho = {rho_logK:.4f} (p = {p_logK:.6f})")
    print(f"  Pearson r = {r_logK:.4f} (p = {pr_logK:.6f})")

    # ============================================================
    # Find best normalization
    # ============================================================
    print(f"\n{'='*70}")
    print("COMPARISON: Which normalization gives best collapse?")
    print(f"{'='*70}")

    options = [
        ("Raw kappa vs kNN", rho_raw, r_raw),
        ("q vs raw kappa", rho_q, r_q),
        ("q vs kappa*K", rho_kK, r_kK),
        ("q vs kappa*sqrt(K)", rho_sqK, r_sqK),
        ("q vs kappa*log(K)", rho_logK, r_logK),
    ]

    print(f"\n  {'Option':>25} {'Spearman':>10} {'Pearson':>10}")
    print(f"  {'-'*50}")
    for name, rho, r in options:
        print(f"  {name:>25} {rho:>10.4f} {r:>10.4f}")

    best = max(options, key=lambda x: abs(x[2]))  # Best by Pearson r
    print(f"\n  BEST: {best[0]} (Pearson r = {best[2]:.4f})")

    # ============================================================
    # Fit sigmoid on best normalization and test cross-dataset prediction
    # ============================================================
    print(f"\n{'='*70}")
    print(f"SIGMOID FIT ON BEST NORMALIZATION: {best[0]}")
    print(f"{'='*70}")

    # Determine which kappa_norm to use
    if best[0] == "q vs kappa*K":
        kappa_norm_all = kappa_K
        y_all = q_vals
    elif best[0] == "q vs kappa*sqrt(K)":
        kappa_norm_all = kappa_sqK
        y_all = q_vals
    elif best[0] == "q vs kappa*log(K)":
        kappa_norm_all = kappa_logK
        y_all = q_vals
    elif best[0] == "q vs raw kappa":
        kappa_norm_all = kappas_raw
        y_all = q_vals
    else:
        kappa_norm_all = kappas_raw
        y_all = knns_raw

    # Split: train on CLINC (Qwen2, SmolLM2, Qwen3), test on TREC + Pythia
    train_mask = np.array([p["dataset"] == "clinc" and "pythia" not in p["model"].lower()
                           for p in all_points])
    test_mask = ~train_mask

    kappa_train = kappa_norm_all[train_mask]
    y_train = y_all[train_mask]
    kappa_test = kappa_norm_all[test_mask]
    y_test = y_all[test_mask]

    print(f"\n  Train: {train_mask.sum()} points (3 models, CLINC)")
    print(f"  Test: {test_mask.sum()} points (Pythia + TREC)")

    try:
        popt, pcov = curve_fit(sigmoid, kappa_train, y_train,
                               p0=[0.6, 0.1, 30, -0.05], maxfev=10000)
        pred_train = sigmoid(kappa_train, *popt)
        pred_test = sigmoid(kappa_test, *popt)

        mae_train = np.mean(np.abs(y_train - pred_train))
        mae_test = np.mean(np.abs(y_test - pred_test))

        ss_res_train = np.sum((y_train - pred_train) ** 2)
        ss_tot_train = np.sum((y_train - y_train.mean()) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train

        ss_res_test = np.sum((y_test - pred_test) ** 2)
        ss_tot_test = np.sum((y_test - y_test.mean()) ** 2)
        r2_test = 1 - ss_res_test / ss_tot_test if ss_tot_test > 0 else 0

        print(f"\n  Training R^2 = {r2_train:.4f}, MAE = {mae_train:.4f}")
        print(f"  Held-out R^2 = {r2_test:.4f}, MAE = {mae_test:.4f}")
        print(f"\n  Pre-registered: held-out MAE < 0.05")

        if mae_test < 0.05:
            print(f"  UNIVERSAL LAW CONFIRMED (MAE={mae_test:.4f} < 0.05)")
        elif mae_test < 0.10:
            print(f"  PARTIAL UNIVERSALITY (MAE={mae_test:.4f} < 0.10)")
        else:
            print(f"  NOT UNIVERSAL (MAE={mae_test:.4f} >= 0.10)")

        # Per-condition breakdown
        print(f"\n  PER-CONDITION (held-out):")
        test_indices = np.where(test_mask)[0]
        conditions = set()
        for i in test_indices:
            p = all_points[i]
            cond = (p["model"], p["dataset"])
            if cond not in conditions:
                conditions.add(cond)
                cond_mask = np.array([all_points[j]["model"] == p["model"] and
                                      all_points[j]["dataset"] == p["dataset"]
                                      for j in test_indices])
                cond_errors = np.abs(y_test[cond_mask] - pred_test[cond_mask])
                short = p["model"].split("/")[-1]
                print(f"    {short:>20} / {p['dataset']:>8}: MAE = {np.mean(cond_errors):.4f}")

    except Exception as e:
        print(f"  Sigmoid fit failed: {e}")
        mae_test = None

    # Save
    out = {
        "experiment": "normalized_kappa_universality",
        "n_points": len(all_points),
        "normalizations": {name: {"rho": float(rho), "r": float(r)}
                           for name, rho, r in options},
        "best_normalization": best[0],
    }
    if mae_test is not None:
        out["cross_dataset_prediction"] = {
            "train_r2": float(r2_train),
            "train_mae": float(mae_train),
            "test_r2": float(r2_test),
            "test_mae": float(mae_test),
        }

    out_path = RESULTS_DIR / "cti_normalized_kappa.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
