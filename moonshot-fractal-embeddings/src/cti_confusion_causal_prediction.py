#!/usr/bin/env python -u
"""
Confusion-Matrix Causal Prediction Test (Session 41)
=====================================================
Codex recommendation: causal centroid intervention predicting full
confusion-matrix shift out-of-sample with FIXED (A, C, tau*), no refit.

PROTOCOL:
  For each class ci, shift its nearest competitor j1 by delta in
  {1.0, 2.0, 3.0} embedding units. Using ONLY pre-registered parameters,
  predict the new full confusion vector C_new[ci,:]. Measure actual.
  Test: r(C_predicted, C_actual) > threshold.

PRE-REGISTERED PARAMETERS (all from prior sessions, NOT refitted here):
  A_NLP = 1.054  (LOAO 12-architecture mean)
  tau_local = 0.026  (confusion matrix Gumbel test per-class median)
  tau_star = 0.20    (phi_upgrade_pooled best tau)

PRE-REGISTERED CRITERIA:
  H1: pooled r(C_pred_tau_star, C_actual) > 0.50
  H2: pooled r(C_pred_tau_local, C_actual) > 0.50
  H3: sign_accuracy(delta_C[ci,j1] predicted vs actual) >= 0.80
      (j1 confusion fraction should decrease when j1 moves away)

Dataset: pythia-160m step512, K=14 DBpedia14, N=7000
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_NPZ = RESULTS_DIR / "checkpoint_embs_pythia-160m_step512.npz"
OUT_JSON = RESULTS_DIR / "cti_confusion_causal_prediction.json"

# PRE-REGISTERED (DO NOT CHANGE AFTER RUNNING)
A_NLP = 1.054       # LOAO constant
TAU_STAR = 0.20     # phi_upgrade_pooled
TAU_LOCAL = 0.026   # confusion_gumbel_test per-class median
N_SPLITS = 5

# Shifts to test (embedding space units)
DELTAS = [1.0, 2.0, 3.0]

# Thresholds
R_POOLED_THRESH = 0.50
SIGN_THRESH = 0.80


def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    for c in classes:
        centroids[c] = X[y == c].mean(0)
    R = np.vstack([X[y == c] - centroids[c] for c in classes])
    sigma_W = float(np.sqrt(np.mean(R ** 2)))
    return centroids, sigma_W


def kappa_matrix(centroids, sigma_W, d):
    classes = sorted(centroids.keys())
    kap = {}
    for ci in classes:
        kap[ci] = {}
        for cj in classes:
            if ci == cj:
                kap[ci][cj] = 0.0
            else:
                dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
                kap[ci][cj] = dist / (sigma_W * np.sqrt(d) + 1e-12)
    return kap


def apply_shift(X, y, centroids, ci, j1, delta):
    """Shift all points in class j1 AWAY from class ci by delta embedding units."""
    mu_i, mu_j = centroids[ci], centroids[j1]
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == j1] += delta * direction
    return X_new


def confusion_vector_5fold(X, y, ci, n_splits=N_SPLITS):
    """Compute per-fold average confusion P(classified as j | true = ci)."""
    classes = sorted(np.unique(y).tolist())
    K = len(classes)
    idx_ci = classes.index(ci)
    C = np.zeros(K)
    count = 0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X_tr, y_tr)

        mask = y_te == ci
        if mask.sum() == 0:
            continue
        preds = knn.predict(X_te[mask])
        for pred in preds:
            j_idx = classes.index(pred)
            C[j_idx] += 1
            count += 1

    if count > 0:
        C /= count
    return C, classes


def predict_confusion(kap_ci, tau, q_ci, classes, ci):
    """
    Predict P(classified as j | true = ci) for all j.
    Uses: C[ci,j] = (1 - q_ci) * softmax(-kappa_j/tau)[j]
    where softmax is over j != ci.
    """
    K = len(classes)
    idx_ci = classes.index(ci)

    # Competition weights
    kappas = np.array([kap_ci.get(cj, 0.0) for cj in classes])
    kappas[idx_ci] = np.inf  # exclude self

    log_w = -kappas / tau
    log_w[idx_ci] = -np.inf
    log_w -= np.max(log_w[log_w > -np.inf])  # numerical stability
    w = np.exp(log_w)
    w[idx_ci] = 0.0
    Z = w.sum()
    if Z < 1e-12:
        w = np.ones(K) / (K - 1)
        w[idx_ci] = 0.0
        Z = w.sum()
    w_norm = w / Z  # normalized competition weights (sum to 1)

    # Predicted confusion vector
    C_pred = np.zeros(K)
    C_pred[idx_ci] = q_ci  # diagonal = correct
    C_pred += (1.0 - q_ci) * w_norm  # off-diagonal = competition-weighted
    return C_pred


def json_default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    print("=" * 70)
    print("CONFUSION MATRIX CAUSAL PREDICTION TEST")
    print(f"A={A_NLP}, tau*={TAU_STAR}, tau_local={TAU_LOCAL}")
    print("=" * 70)

    data = np.load(str(CACHE_NPZ))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    K = len(classes)
    print(f"Data: N={len(X)}, d={d}, K={K}")

    # Baseline geometry
    centroids, sigma_W = compute_class_stats(X, y)
    kap = kappa_matrix(centroids, sigma_W, d)

    # Find nearest competitor (j1) for each class
    j1_map = {}
    kappa_j1_map = {}
    C0_map = {}  # baseline logit(q_ci) intercept
    for ci in classes:
        sorted_kap = sorted([(kap[ci][cj], cj) for cj in classes if cj != ci])
        kappa_j1_ci, j1_ci = sorted_kap[0]
        j1_map[ci] = j1_ci
        kappa_j1_map[ci] = kappa_j1_ci

    print("\nBaseline confusion vectors (computing 5-fold)...")
    baseline_C = {}
    baseline_q = {}
    for ci in classes:
        C_base, cls_list = confusion_vector_5fold(X, y, ci)
        baseline_C[ci] = C_base
        baseline_q[ci] = float(C_base[cls_list.index(ci)])
        print(f"  ci={ci}: q={baseline_q[ci]:.4f}, kappa_j1={kappa_j1_map[ci]:.4f}, "
              f"j1={j1_map[ci]}")

    # Compute baseline intercept C0 for each class: C0 = logit(q) - A * kappa_j1
    for ci in classes:
        q = baseline_q[ci]
        q_clip = float(np.clip(q, 1e-5, 1 - 1e-5))
        logit_q = float(np.log(q_clip / (1 - q_clip)))
        C0_map[ci] = logit_q - A_NLP * kappa_j1_map[ci]

    print("\nRunning causal prediction across deltas...")
    results_per_delta = []

    for delta in DELTAS:
        print(f"\n  Delta = {delta:.1f}")
        delta_results = []

        all_pred_star = []
        all_pred_local = []
        all_actual = []
        sign_correct = 0
        sign_total = 0

        for ci in classes:
            j1 = j1_map[ci]
            kappa_j1_old = kappa_j1_map[ci]

            # Apply shift: move j1 away from ci
            X_new = apply_shift(X, y, centroids, ci, j1, delta)
            c_new, sw_new = compute_class_stats(X_new, y)
            kap_new = kappa_matrix(c_new, sw_new, d)

            # New kappa_j1 for ci (and all other kappas)
            kappa_j1_new = kap_new[ci][j1]
            delta_kappa = kappa_j1_new - kappa_j1_old

            # Predict new q_ci using A_NLP and per-class C0
            logit_q_new = C0_map[ci] + A_NLP * kappa_j1_new
            q_ci_new = float(1.0 / (1.0 + np.exp(-logit_q_new)))
            q_ci_new = float(np.clip(q_ci_new, 1e-5, 1 - 1e-5))

            # Predict confusion using tau_star
            kap_ci_new = {cj: kap_new[ci][cj] for cj in classes}
            C_pred_star = predict_confusion(kap_ci_new, TAU_STAR, q_ci_new, classes, ci)

            # Predict confusion using tau_local
            C_pred_local = predict_confusion(kap_ci_new, TAU_LOCAL, q_ci_new, classes, ci)

            # Measure actual confusion
            C_actual_arr, cls_list = confusion_vector_5fold(X_new, y, ci)

            # Accumulate
            # Only off-diagonal (exclude self-prediction for fairness)
            idx_ci = cls_list.index(ci)
            offdiag = [j for j in range(K) if j != idx_ci]

            for j in offdiag:
                all_pred_star.append(C_pred_star[j])
                all_pred_local.append(C_pred_local[j])
                all_actual.append(C_actual_arr[j])

            # Sign test: does C[ci,j1] decrease after shift? (delta_C_j1 < 0)
            j1_idx = cls_list.index(j1)
            pred_delta_j1_star = C_pred_star[j1_idx] - baseline_C[ci][j1_idx]
            actual_delta_j1 = C_actual_arr[j1_idx] - baseline_C[ci][j1_idx]
            sign_match = bool(np.sign(pred_delta_j1_star) == np.sign(actual_delta_j1))
            sign_correct += int(sign_match)
            sign_total += 1

            delta_results.append({
                "ci": int(ci),
                "j1": int(j1),
                "delta_kappa": float(delta_kappa),
                "q_pred": float(q_ci_new),
                "q_actual": float(C_actual_arr[idx_ci]),
                "delta_C_j1_pred_star": float(pred_delta_j1_star),
                "delta_C_j1_actual": float(actual_delta_j1),
                "sign_match": sign_match,
            })

            print(f"    ci={ci}: q_pred={q_ci_new:.3f}, q_actual={C_actual_arr[idx_ci]:.3f}, "
                  f"dC_j1_pred={pred_delta_j1_star:+.4f}, dC_j1_act={actual_delta_j1:+.4f}, "
                  f"sign={'OK' if sign_match else 'FAIL'}")

        # Aggregate
        all_pred_star = np.array(all_pred_star)
        all_pred_local = np.array(all_pred_local)
        all_actual = np.array(all_actual)

        r_star, p_star = pearsonr(all_pred_star, all_actual) if len(all_pred_star) >= 4 else (0.0, 1.0)
        r_local, p_local = pearsonr(all_pred_local, all_actual) if len(all_pred_local) >= 4 else (0.0, 1.0)
        sign_acc = float(sign_correct / sign_total) if sign_total > 0 else 0.0

        pass_H1 = bool(r_star > R_POOLED_THRESH)
        pass_H2 = bool(r_local > R_POOLED_THRESH)
        pass_H3 = bool(sign_acc >= SIGN_THRESH)

        print(f"\n  delta={delta}: r_tau_star={r_star:.4f} {'PASS' if pass_H1 else 'FAIL'}, "
              f"r_tau_local={r_local:.4f} {'PASS' if pass_H2 else 'FAIL'}, "
              f"sign_acc={sign_acc:.3f} {'PASS' if pass_H3 else 'FAIL'}")

        results_per_delta.append({
            "delta": float(delta),
            "n_pairs": len(all_pred_star),
            "r_tau_star": float(r_star),
            "p_tau_star": float(p_star),
            "r_tau_local": float(r_local),
            "p_tau_local": float(p_local),
            "sign_accuracy": float(sign_acc),
            "pass_H1": pass_H1,
            "pass_H2": pass_H2,
            "pass_H3": pass_H3,
            "per_class": delta_results,
        })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Delta':>6} | {'r_tau*':>9} {'pass':>5} | {'r_local':>8} {'pass':>5} | "
          f"{'sign_acc':>9} {'pass':>5}")
    for res in results_per_delta:
        p1 = "PASS" if res["pass_H1"] else "FAIL"
        p2 = "PASS" if res["pass_H2"] else "FAIL"
        p3 = "PASS" if res["pass_H3"] else "FAIL"
        print(f"{res['delta']:>6.1f} | {res['r_tau_star']:>9.4f} {p1:>5} | "
              f"{res['r_tau_local']:>8.4f} {p2:>5} | {res['sign_accuracy']:>9.4f} {p3:>5}")

    output = {
        "experiment": "confusion_causal_prediction",
        "session": 41,
        "preregistered": {
            "A_NLP": A_NLP,
            "tau_star": TAU_STAR,
            "tau_local": TAU_LOCAL,
            "H1_threshold": R_POOLED_THRESH,
            "H2_threshold": R_POOLED_THRESH,
            "H3_sign_threshold": SIGN_THRESH,
            "deltas": DELTAS,
        },
        "results": results_per_delta,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
