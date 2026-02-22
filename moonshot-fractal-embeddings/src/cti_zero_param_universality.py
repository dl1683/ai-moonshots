#!/usr/bin/env python
"""
cti_zero_param_universality.py

PREREGISTERED: Zero-parameter universality test for CTI law.

Hypothesis: logit(q) = C_corr * sqrt(d_eff * log(K)) * kappa_nearest + C_0
where C_corr=1.075 is a UNIVERSAL constant (locked before testing).

Protocol:
1. Compute kappa_nearest, d_eff, q from raw embeddings (NPZ files)
2. Calibrate C_0 on ONE cell: (gpt-neo-125m, agnews)
3. Predict ALL OTHER cells without refitting
4. Report out-of-cell R2

Pass criteria (preregistered):
  - Overall R2 >= 0.85
  - Each dataset R2 >= 0.70
  - Residual correlation with K: |r| < 0.2
  - Allowing per-task A improves R2 by < 0.05
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from scipy.special import logit as sp_logit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# ==================== LOCKED CONSTANT (do not change after preregistration) ====================
C_CORR = 1.075  # Universal coefficient, fit on synthetic data (locked)
# ================================================================================================

# NPZ files available (model, dataset, K, path)
NPZ_FILES = [
    ("gpt-neo-125m", "agnews",       4,  "do_int_embs_gpt-neo-125m_agnews.npz"),
    ("gpt-neo-125m", "dbpedia",      14, "do_int_embs_gpt-neo-125m_dbpedia.npz"),
    ("gpt-neo-125m", "20newsgroups", 20, "do_int_embs_v3_gpt-neo-125m_20newsgroups.npz"),
    ("pythia-160m",  "agnews",       4,  "do_int_embs_pythia-160m_agnews.npz"),
    ("pythia-160m",  "dbpedia",      14, "do_int_embs_pythia-160m_dbpedia.npz"),
    ("pythia-160m",  "20newsgroups", 20, "do_int_embs_v3_pythia-160m_20newsgroups.npz"),
]

# Calibration cell (locked before testing)
CALIBRATION_CELL = ("gpt-neo-125m", "agnews")


def compute_d_eff_W(X, y):
    """Within-class effective dimensionality: tr(Sigma_W)^2 / tr(Sigma_W^2)."""
    classes = np.unique(y)
    n = len(y)
    # Class-center each class
    X_c = np.vstack([X[y == c] - X[y == c].mean(0) for c in classes])
    W = X_c.T @ X_c / n  # d x d
    tr_W = np.trace(W)
    tr_W2 = float(np.sum(W * W))  # tr(W^2) = sum(W_ij^2)
    if tr_W2 < 1e-12:
        return 1.0
    return float(tr_W ** 2 / tr_W2)


def compute_kappa_nearest(X, y):
    """kappa_nearest = delta_min / (sigma_W * sqrt(d))."""
    classes = np.unique(y)
    d = X.shape[1]
    # Class means
    means = np.array([X[y == c].mean(0) for c in classes])
    # Min inter-class centroid distance
    dists = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dists.append(np.linalg.norm(means[i] - means[j]))
    delta_min = float(min(dists))
    # sigma_W = sqrt(tr(Sigma_W) / d) where Sigma_W is pooled within-class cov
    n = len(y)
    X_c = np.vstack([X[y == c] - means[ci] for ci, c in enumerate(classes)])
    tr_W = float(np.trace(X_c.T @ X_c / n))
    sigma_W = float(np.sqrt(tr_W / d))
    if sigma_W < 1e-10:
        return 0.0
    return float(delta_min / (sigma_W * np.sqrt(d)))


def compute_q_nc(X, y):
    """Normalized nearest-centroid accuracy: q = (acc - 1/K) / (1 - 1/K)."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    classes = np.unique(y_tr)
    K = len(classes)
    means = np.array([X_tr[y_tr == c].mean(0) for c in classes])
    dists = np.array([[np.linalg.norm(x - means[j]) for j in range(K)] for x in X_te])
    pred = classes[np.argmin(dists, axis=1)]
    acc = float(np.mean(pred == y_te))
    return float((acc - 1.0 / K) / (1.0 - 1.0 / K))


def load_cell(npz_path: Path, K: int):
    """Load embeddings, compute kappa_nearest, d_eff_W, q."""
    npz = np.load(npz_path)
    X, y = npz["X"].astype(float), npz["y"]

    # Subsample if large (500 per class for d_eff stability)
    rng = np.random.RandomState(42)
    classes = np.unique(y)
    max_per_class = 500
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        if len(ci) > max_per_class:
            ci = rng.choice(ci, max_per_class, replace=False)
        idx.extend(ci.tolist())
    idx = np.array(idx)
    X, y = X[idx], y[idx]

    d_eff_W = compute_d_eff_W(X, y)
    kappa = compute_kappa_nearest(X, y)
    q_nc = compute_q_nc(X, y)
    logit_q = float(sp_logit(np.clip(q_nc, 1e-6, 1 - 1e-6)))

    return {
        "d_eff_W": d_eff_W,
        "kappa_nearest": kappa,
        "q_nc": q_nc,
        "logit_q": logit_q,
        "K": K,
        "n": len(y),
        "d": X.shape[1],
    }


def formula_predict(kappa, d_eff, K, C_corr, C_0):
    """Predict logit(q) = C_corr * sqrt(d_eff * log(K)) * kappa + C_0."""
    A = C_corr * np.sqrt(d_eff * np.log(float(K)))
    return A * kappa + C_0


def r2_score(y_true, y_pred):
    ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def main():
    print("=" * 70)
    print("ZERO-PARAMETER CTI UNIVERSALITY TEST")
    print("PREREGISTERED: C_corr=1.075, C_0 from one cell, no other params")
    print("=" * 70)
    print()

    # Step 1: Load all cells
    cells = []
    for model, dataset, K, fname in NPZ_FILES:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"MISSING: {fname} - skipping")
            continue
        print(f"Loading: {model} + {dataset} (K={K})")
        cell = load_cell(path, K)
        cell["model"] = model
        cell["dataset"] = dataset
        print(f"  kappa={cell['kappa_nearest']:.4f}, d_eff_W={cell['d_eff_W']:.2f}, "
              f"q_nc={cell['q_nc']:.4f}, logit_q={cell['logit_q']:.4f}")
        cells.append(cell)

    print(f"\nTotal cells loaded: {len(cells)}")

    # Step 2: Calibrate C_0 on ONE cell (preregistered)
    calib = [c for c in cells
             if c["model"] == CALIBRATION_CELL[0] and c["dataset"] == CALIBRATION_CELL[1]]
    if not calib:
        print(f"ERROR: Calibration cell {CALIBRATION_CELL} not found!")
        return
    calib = calib[0]
    A_calib = C_CORR * np.sqrt(calib["d_eff_W"] * np.log(float(calib["K"])))
    C_0 = calib["logit_q"] - A_calib * calib["kappa_nearest"]
    print(f"\nCalibration cell: {CALIBRATION_CELL}")
    print(f"  A_calib = C_corr * sqrt(d_eff * log(K)) = {A_calib:.4f}")
    print(f"  C_0 = {C_0:.4f} (LOCKED for all predictions)")

    # Step 3: Predict held-out cells
    held_out = [c for c in cells
                if not (c["model"] == CALIBRATION_CELL[0]
                        and c["dataset"] == CALIBRATION_CELL[1])]
    print(f"\n{'='*70}")
    print(f"HELD-OUT PREDICTIONS (n={len(held_out)}):")
    print(f"{'Model':<20} {'Dataset':<15} {'K':>4} {'d_eff_W':>8} {'kappa':>8} "
          f"{'logit_actual':>14} {'logit_pred':>12} {'err':>8}")
    print("-" * 95)

    actuals, preds = [], []
    for c in cells:
        is_calib = (c["model"] == CALIBRATION_CELL[0]
                    and c["dataset"] == CALIBRATION_CELL[1])
        pred = formula_predict(c["kappa_nearest"], c["d_eff_W"], c["K"], C_CORR, C_0)
        err = c["logit_q"] - pred
        flag = " [CALIB]" if is_calib else ""
        print(f"{c['model']:<20} {c['dataset']:<15} {c['K']:>4} "
              f"{c['d_eff_W']:>8.2f} {c['kappa_nearest']:>8.4f} "
              f"{c['logit_q']:>14.4f} {pred:>12.4f} {err:>8.4f}{flag}")
        if not is_calib:
            actuals.append(c["logit_q"])
            preds.append(pred)

    # Step 4: Compute metrics
    actuals = np.array(actuals)
    preds = np.array(preds)
    overall_r2 = r2_score(actuals, preds)
    overall_r, _ = pearsonr(actuals, preds)
    residuals = actuals - preds
    Ks = np.array([c["K"] for c in held_out])
    r_resid_K, _ = pearsonr(Ks, residuals) if len(Ks) > 2 else (0.0, 1.0)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY:")
    print(f"  Overall out-of-cell R2 = {overall_r2:.4f}")
    print(f"  Overall out-of-cell r  = {overall_r:.4f}")
    print(f"  Residual corr with K   = {r_resid_K:.4f}")

    # Per-dataset R2
    datasets = sorted(set(c["dataset"] for c in held_out))
    dataset_r2 = {}
    for ds in datasets:
        ds_cells = [c for c in held_out if c["dataset"] == ds]
        if len(ds_cells) < 2:
            continue
        ds_act = [c["logit_q"] for c in ds_cells]
        ds_pred = [formula_predict(c["kappa_nearest"], c["d_eff_W"], c["K"], C_CORR, C_0)
                   for c in ds_cells]
        dataset_r2[ds] = r2_score(ds_act, ds_pred)
        print(f"  Dataset {ds}: R2 = {dataset_r2[ds]:.4f} (n={len(ds_cells)})")

    # Step 5: Compare to per-task A (assess whether per-task A helps significantly)
    # Fit per-task A: logit(q) = A_task * kappa_nearest + C_0_task
    # Use leave-one-task-out: train on other tasks, test on left-out
    print("\n  Per-task A comparison (leave-one-task-out):")
    all_datasets = sorted(set(c["dataset"] for c in cells))
    looo_actuals, looo_preds_pertask = [], []
    for test_ds in all_datasets:
        train_cells = [c for c in cells if c["dataset"] != test_ds]
        test_cells  = [c for c in cells if c["dataset"] == test_ds]
        if not train_cells or not test_cells:
            continue
        # Fit A and C from train (one parameter per task type)
        X_train = np.array([c["kappa_nearest"] for c in train_cells])
        y_train = np.array([c["logit_q"] for c in train_cells])
        # Simple linear fit
        A_fit = float(np.polyfit(X_train, y_train, 1)[0])
        C_fit = float(np.polyfit(X_train, y_train, 1)[1])
        for c in test_cells:
            looo_actuals.append(c["logit_q"])
            looo_preds_pertask.append(A_fit * c["kappa_nearest"] + C_fit)
    if looo_actuals:
        r2_pertask = r2_score(looo_actuals, looo_preds_pertask)
        print(f"  Per-task A LOOO R2 = {r2_pertask:.4f}")
        r2_improvement = r2_pertask - overall_r2
        print(f"  Improvement from per-task A: {r2_improvement:+.4f}")

    # Step 6: VERDICT
    print("\n" + "=" * 70)
    print("PREREGISTERED PASS CRITERIA:")
    pass1 = overall_r2 >= 0.85
    pass2 = all(r2 >= 0.70 for r2 in dataset_r2.values()) if dataset_r2 else False
    pass3 = abs(r_resid_K) < 0.20
    # pass4: improvement from per-task A < 0.05
    pass4 = (r2_pertask - overall_r2) < 0.05 if looo_actuals else True

    print(f"  1. Overall R2 >= 0.85:          {overall_r2:.4f} [{('PASS' if pass1 else 'FAIL')}]")
    print(f"  2. Each dataset R2 >= 0.70:     {min(dataset_r2.values()) if dataset_r2 else float('nan'):.4f} "
          f"[{'PASS' if pass2 else 'FAIL'}]")
    print(f"  3. |r(resid, K)| < 0.20:        {abs(r_resid_K):.4f} [{'PASS' if pass3 else 'FAIL'}]")
    print(f"  4. Per-task improvement < 0.05: {(r2_pertask - overall_r2) if looo_actuals else float('nan'):+.4f} "
          f"[{'PASS' if pass4 else 'FAIL'}]")

    n_pass = sum([pass1, pass2, pass3, pass4])
    verdict = "PASS" if n_pass >= 4 else ("PARTIAL" if n_pass >= 2 else "FAIL")
    print(f"\nVERDICT: {verdict} ({n_pass}/4 criteria met)")

    # Save
    result = {
        "preregistered": {
            "C_corr": C_CORR,
            "C_0_from": CALIBRATION_CELL,
            "criteria": {
                "r2_overall": 0.85,
                "r2_per_dataset": 0.70,
                "r_resid_K": 0.20,
                "pertask_improvement": 0.05,
            },
        },
        "calibration": {
            "model": CALIBRATION_CELL[0],
            "dataset": CALIBRATION_CELL[1],
            "C_0": float(C_0),
            "A_calib": float(A_calib),
        },
        "cells": cells,
        "metrics": {
            "overall_r2": float(overall_r2),
            "overall_r": float(overall_r),
            "r_resid_K": float(r_resid_K),
            "dataset_r2": {k: float(v) for k, v in dataset_r2.items()},
            "pertask_r2": float(r2_pertask) if looo_actuals else None,
        },
        "criteria_pass": {
            "r2_overall": pass1,
            "r2_per_dataset": pass2,
            "r_resid_K_low": pass3,
            "pertask_improvement_small": pass4,
        },
        "verdict": verdict,
        "n_pass": n_pass,
    }

    out_path = RESULTS_DIR / "cti_zero_param_universality.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
