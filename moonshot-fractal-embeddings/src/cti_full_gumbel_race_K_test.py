#!/usr/bin/env python -u
"""
FULL GUMBEL RACE K-TEST (Feb 22 2026)
======================================
Codex recommendation (Feb 22): "Run one pre-registered K x T intervention
using the full competitor sum (Gumbel/softmax race), not a nearest-only truncation."

CORRECT UNIVERSAL LAW (Codex formulation):
  q_i = 1 / (1 + sum_{j != i} exp[-A * sqrt(d_eff) * Delta_kappa_ij])
  Delta_kappa_ij = kappa_nearest_i - kappa_ij   [note: kappa_ij = dist(mu_i, mu_j)/sigma_W*sqrt(d)]

  For i = true class:
    kappa_ij is the SNR between class i and class j from i's perspective
    When class j is closer to i's centroid: kappa_ij is smaller (closer) -> higher confusion

Equivalent form:
  logit(q_i) = A * sqrt(d_eff) * kappa_nearest_i - log(sum_{j != i} exp[-A*sqrt(d_eff)*kappa_ij])

MODEL COMPARISON (pre-registered):
  M0 (nearest-only): logit(q_i) = A * sqrt(d_eff) * kappa_nearest_i + C
  M1 (full-sum):     logit(q_i) = A * sqrt(d_eff) * kappa_nearest_i
                                  - log(sum_{j != i} exp[-A * sqrt(d_eff) * kappa_ij]) + C_adj
  M0 is the APPROXIMATION (ignores all but nearest)
  M1 is the FULL Gumbel Race (includes ALL competitors)

PREDICTION:
  M1 should fit BETTER than M0 when K is large (dense: CIFAR-10 K=10)
  M0 should be adequate when K=2 (binary)
  Test at K in {2, 3, 5, 7, 10} using random class subsets

ARM C RESOLUTION:
  Arm C (ViT): farthest class still causally affects q (r=0.637)
  M1 PREDICTS this: for K=10 dense CIFAR-10, jK weight = exp(-0.63 * 0.3 * sqrt(32)) = 0.34 (non-negligible)
  M1 ARM C PREDICTION: r(kappa_jK, q) should be ~ 0.34 (non-zero) for CIFAR-10

PRE-REGISTERED CRITERIA:
  1. M1 R2 > M0 R2 at K=10 by at least 0.05
  2. M1 R2 > 0.90 at K=10
  3. M0 R2 ≥ M1 R2 - 0.05 at K=2 (both adequate for binary)
  4. A_fit from M1 matches A_LOAO within 30% for both NLP and ViT

DATASETS:
  ViT-Large CIFAR-10 layer 12 (K=10)
  Pythia-160m DBpedia layer 12 (K=14)
"""

import json
import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from scipy import stats, optimize

# ================================================================
# CONFIG
# ================================================================
VIT_EMBS_FILE  = "results/vit_loao_embs_vit-large-patch16-224_cifar10.npz"
VIT_LAYER_KEY  = "12"
VIT_SUBSAMPLE  = 500
VIT_A_LOAO     = 0.63          # LOCKED from LOAO

NLP_EMBS_FILE  = "results/do_int_embs_pythia-160m_dbpedia.npz"
NLP_A_LOAO     = 1.054         # LOCKED from LOAO

OUT_JSON  = "results/cti_full_gumbel_race_K_test.json"
OUT_LOG   = "results/cti_full_gumbel_race_K_test_log.txt"

K_SUBSETS_VIT = [2, 3, 5, 7, 10]      # K values to test (CIFAR-10 has 10)
K_SUBSETS_NLP = [2, 3, 5, 7, 14]      # K values to test (DBpedia has 14)
N_REPEATS     = 10                     # random class subsets per K value
N_CV_SPLITS   = 5
RANDOM_SEED   = 42

# PRE-REGISTERED thresholds
DELTA_R2_THRESHOLD = 0.05     # M1 must exceed M0 by at least this at K=10
M1_R2_THRESHOLD    = 0.90     # M1 R2 must reach this at K=10
A_TOLERANCE        = 0.30     # A_fit must be within 30% of A_LOAO

CIFAR10_NAMES = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
                 5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}


# ================================================================
# HELPERS
# ================================================================
def compute_kappa_matrix(X, y, classes=None):
    """Return centroids, sigma_W, tr_W, kappa_matrix[i,j]."""
    if classes is None:
        classes = sorted(np.unique(y).tolist())
    K = len(classes)
    d = X.shape[1]
    cents = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        cents[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R**2)))
    tr_W = float(np.mean(R**2) * d)

    # kappa_ij = dist(mu_i, mu_j) / (sigma_W * sqrt(d))
    kappa = np.zeros((K, K))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if ci == cj:
                kappa[i, j] = 0.0
            else:
                dist_ij = float(np.linalg.norm(cents[ci] - cents[cj]))
                kappa[i, j] = dist_ij / (sigma_W * np.sqrt(d) + 1e-10)

    # d_eff per class (in direction of nearest competitor)
    d_eff = []
    for i, ci in enumerate(classes):
        kappas_i = [(kappa[i, j], j, classes[j]) for j in range(K) if j != i]
        kappas_i.sort()
        j1_idx = kappas_i[0][1]
        cj1 = classes[j1_idx]
        dir_j1 = cents[cj1] - cents[ci]
        dist_j1 = np.linalg.norm(dir_j1)
        if dist_j1 < 1e-10:
            d_eff.append(float('nan'))
            continue
        dir_j1 = dir_j1 / dist_j1
        R_ci = X[y == ci] - cents[ci]
        var_cdir = float(np.var(R_ci @ dir_j1))
        d_eff.append(tr_W / var_cdir if var_cdir > 1e-10 else float('nan'))

    return cents, sigma_W, tr_W, kappa, d_eff, classes


def compute_actual_q(X, y, classes, n_splits=N_CV_SPLITS):
    """Compute per-class 1-NN accuracy (normalized) via CV."""
    K = len(classes)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    q_dict = {c: [] for c in classes}
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
        knn.fit(X_tr, y_tr)
        preds = knn.predict(X_te)
        for c in classes:
            mask = (y_te == c)
            if mask.sum() == 0:
                continue
            recall = float((preds[mask] == c).mean())
            q_dict[c].append(recall)
    q = {}
    for c in classes:
        if q_dict[c]:
            q_raw = float(np.mean(q_dict[c]))
            K_inv = 1.0 / K
            q[c] = float((q_raw - K_inv) / (1.0 - K_inv)) if abs(1.0 - K_inv) > 1e-10 else 0.0
    return q


def logit_fn(q):
    q = float(np.clip(q, 1e-5, 1-1e-5))
    return float(np.log(q / (1.0 - q)))


def predict_M0(kappa_matrix, classes, A, d_eff_per_class):
    """M0: logit(q) = A * sqrt(d_eff) * kappa_nearest + C."""
    K = len(classes)
    preds = []
    for i in range(K):
        kappas_i = [kappa_matrix[i, j] for j in range(K) if j != i]
        kappa_nearest = min(kappas_i)
        d_eff_i = d_eff_per_class[i] if not np.isnan(d_eff_per_class[i]) else np.nanmean(d_eff_per_class)
        preds.append(A * np.sqrt(d_eff_i) * kappa_nearest)
    return np.array(preds)


def predict_M1(kappa_matrix, classes, A, d_eff_per_class):
    """M1: logit(q) = A*sqrt(d_eff)*kappa_nearest - log(sum_j exp[-A*sqrt(d_eff)*kappa_ij]) + const."""
    K = len(classes)
    preds = []
    for i in range(K):
        kappas_i = [kappa_matrix[i, j] for j in range(K) if j != i]
        kappa_nearest = min(kappas_i)
        d_eff_i = d_eff_per_class[i] if not np.isnan(d_eff_per_class[i]) else np.nanmean(d_eff_per_class)
        # Full sum: log(sum_{j != i} exp[-A * sqrt(d_eff) * kappa_ij])
        log_sum = np.log(sum(np.exp(-A * np.sqrt(d_eff_i) * kij) for kij in kappas_i))
        # M1: A * sqrt(d_eff) * kappa_nearest - log_sum
        preds.append(A * np.sqrt(d_eff_i) * kappa_nearest - log_sum)
    return np.array(preds)


def fit_R2(logit_actual, logit_pred):
    """Fit intercept-only linear correction and return R2."""
    if len(logit_actual) < 2 or np.std(logit_pred) < 1e-10 or np.std(logit_actual) < 1e-10:
        return 0.0
    # R^2 between logit_pred and logit_actual after best-fit affine
    reg = LinearRegression()
    reg.fit(logit_pred.reshape(-1, 1), logit_actual)
    logit_fitted = reg.predict(logit_pred.reshape(-1, 1))
    ss_res = np.sum((logit_actual - logit_fitted)**2)
    ss_tot = np.sum((logit_actual - logit_actual.mean())**2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0


def fit_A_from_M1(logit_actual, kappa_matrix, classes, d_eff_per_class):
    """Fit A that maximizes R2 of M1 predictions."""
    def neg_r2(A_val):
        if A_val[0] <= 0:
            return 1.0
        pred = predict_M1(kappa_matrix, classes, A_val[0], d_eff_per_class)
        return -fit_R2(logit_actual, pred)
    res = optimize.minimize(neg_r2, [1.0], method='Nelder-Mead',
                            options={'xatol': 1e-3, 'fatol': 1e-4, 'maxiter': 100})
    return float(res.x[0]), float(-res.fun)


# ================================================================
# RUN ONE DATASET
# ================================================================
def run_dataset(X_sub, y_sub, A_LOAO, K_subsets, class_names, log_fn, dataset_name):
    """Run full K-variation test on one dataset."""
    all_classes = sorted(np.unique(y_sub).tolist())
    K_max = len(all_classes)
    rng = np.random.RandomState(RANDOM_SEED)

    results = {}

    for K_val in K_subsets:
        if K_val > K_max:
            log_fn(f"  Skip K={K_val} (K_max={K_max})")
            continue

        log_fn(f"\n  [K={K_val}] Running {N_REPEATS} random subsets...")
        K_results = []

        for rep in range(N_REPEATS):
            chosen_classes = sorted(rng.choice(all_classes, K_val, replace=False).tolist())

            # Subsample data to only those classes
            mask = np.isin(y_sub, chosen_classes)
            X_k = X_sub[mask]
            y_k = y_sub[mask]

            # Re-label classes 0..K-1
            class_map = {c: i for i, c in enumerate(chosen_classes)}
            y_k_relabeled = np.array([class_map[c] for c in y_k])
            classes_k = list(range(K_val))

            # Compute geometry
            try:
                cents, sigma_W, tr_W, kappa_mat, d_eff_list, _ = compute_kappa_matrix(
                    X_k, y_k_relabeled, classes_k)
            except Exception as e:
                log_fn(f"  K={K_val} rep={rep} geometry error: {e}")
                continue

            # Compute actual q
            q_dict = compute_actual_q(X_k, y_k_relabeled, classes_k)
            if len(q_dict) < K_val:
                continue

            logit_actual = np.array([logit_fn(q_dict[c]) for c in classes_k
                                     if c in q_dict and q_dict[c] is not None])
            if len(logit_actual) < 2:
                continue

            valid_idx = [c for c in classes_k if c in q_dict and q_dict[c] is not None]

            # M0 predictions
            m0_pred = predict_M0(kappa_mat, classes_k, A_LOAO, d_eff_list)
            m0_pred_valid = m0_pred[[c for c in valid_idx]]
            m0_R2 = fit_R2(logit_actual, m0_pred_valid)

            # M1 predictions
            m1_pred = predict_M1(kappa_mat, classes_k, A_LOAO, d_eff_list)
            m1_pred_valid = m1_pred[[c for c in valid_idx]]
            m1_R2 = fit_R2(logit_actual, m1_pred_valid)

            # Fit A from M1
            try:
                A_fit, A_R2 = fit_A_from_M1(logit_actual, kappa_mat, classes_k, d_eff_list)
            except Exception:
                A_fit, A_R2 = float('nan'), float('nan')

            K_results.append({
                "rep": rep,
                "classes": [class_names.get(chosen_classes[c], str(chosen_classes[c])) for c in classes_k],
                "m0_R2": m0_R2,
                "m1_R2": m1_R2,
                "delta_R2": m1_R2 - m0_R2,
                "A_fit": A_fit,
                "A_loao": A_LOAO,
            })

        if K_results:
            mean_m0_R2 = float(np.mean([r["m0_R2"] for r in K_results]))
            mean_m1_R2 = float(np.mean([r["m1_R2"] for r in K_results]))
            mean_delta = float(np.mean([r["delta_R2"] for r in K_results]))
            A_fits = [r["A_fit"] for r in K_results if not np.isnan(r["A_fit"])]
            mean_A_fit = float(np.mean(A_fits)) if A_fits else float('nan')

            pass_delta = mean_delta >= DELTA_R2_THRESHOLD if K_val == K_max else None
            pass_M1 = mean_m1_R2 >= M1_R2_THRESHOLD if K_val == K_max else None
            pass_A = abs(mean_A_fit - A_LOAO) / A_LOAO <= A_TOLERANCE if not np.isnan(mean_A_fit) else None

            log_fn(f"    K={K_val}: M0 R2={mean_m0_R2:.3f}, M1 R2={mean_m1_R2:.3f}, "
                   f"delta={mean_delta:+.3f}, A_fit={mean_A_fit:.3f} (LOAO={A_LOAO})")
            if K_val == K_max:
                log_fn(f"    PRE-REG K={K_val}: delta>={DELTA_R2_THRESHOLD}? {'PASS' if pass_delta else 'FAIL'}")
                log_fn(f"    PRE-REG K={K_val}: M1_R2>={M1_R2_THRESHOLD}? {'PASS' if pass_M1 else 'FAIL'}")
            if not np.isnan(mean_A_fit):
                log_fn(f"    A_fit within {A_TOLERANCE*100:.0f}%: {'PASS' if pass_A else 'FAIL'} "
                       f"(fit={mean_A_fit:.3f} vs LOAO={A_LOAO})")

            results[K_val] = {
                "K": K_val,
                "n_reps": len(K_results),
                "mean_m0_R2": mean_m0_R2,
                "mean_m1_R2": mean_m1_R2,
                "mean_delta_R2": mean_delta,
                "mean_A_fit": mean_A_fit,
                "std_A_fit": float(np.std(A_fits)) if A_fits else float('nan'),
                "pre_reg_delta": bool(pass_delta) if pass_delta is not None else None,
                "pre_reg_M1_R2": bool(pass_M1) if pass_M1 is not None else None,
                "pre_reg_A": bool(pass_A) if pass_A is not None else None,
                "reps": K_results,
            }

    return results


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs("results", exist_ok=True)
    log_file = open(OUT_LOG, "w", buffering=1)
    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log("=" * 70)
    log("FULL GUMBEL RACE K-TEST")
    log("=" * 70)
    log("Law: q = 1/(1 + sum_j exp[-A*sqrt(d_eff)*Delta_kappa_ij])")
    log("M0: nearest-only (logit(q) = A*sqrt(d)*kappa_nearest + C)")
    log("M1: full sum (logit(q) = A*sqrt(d)*kappa_nearest - log(sum_j exp[-A*sqrt(d)*kappa_ij]) + C)")
    log(f"PRE-REGISTERED:")
    log(f"  1. M1 R2 > M0 R2 by >= {DELTA_R2_THRESHOLD} at K_max")
    log(f"  2. M1 R2 >= {M1_R2_THRESHOLD} at K_max")
    log(f"  3. A_fit within {A_TOLERANCE*100:.0f}% of A_LOAO")
    log("=" * 70)

    all_results = {}

    # ----------------------------------------------------------------
    # VIT
    # ----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("DATASET 1: ViT-Large CIFAR-10 layer 12")
    log("=" * 70)

    vit_data = np.load(VIT_EMBS_FILE)
    X_vit_all = vit_data[VIT_LAYER_KEY].astype(np.float64)
    y_vit_all = vit_data["y"].astype(np.int64)
    rng = np.random.RandomState(RANDOM_SEED)
    keep = []
    for c in range(10):
        idx_c = np.where(y_vit_all == c)[0]
        chosen = rng.choice(idx_c, size=min(VIT_SUBSAMPLE, len(idx_c)), replace=False)
        keep.append(chosen)
    X_vit = X_vit_all[np.sort(np.concatenate(keep))]
    y_vit = y_vit_all[np.sort(np.concatenate(keep))]
    log(f"Subsampled ViT: N={X_vit.shape[0]}, d={X_vit.shape[1]}")

    vit_K_results = run_dataset(X_vit, y_vit, VIT_A_LOAO, K_SUBSETS_VIT,
                                CIFAR10_NAMES, log, "ViT")
    all_results["vit"] = {
        "dataset": "ViT-Large CIFAR-10 layer 12",
        "A_loao": VIT_A_LOAO,
        "K_subsets": K_SUBSETS_VIT,
        "K_results": {str(k): v for k, v in vit_K_results.items()},
    }

    # ----------------------------------------------------------------
    # NLP
    # ----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("DATASET 2: Pythia-160m DBpedia layer 12")
    log("=" * 70)

    try:
        nlp_data = np.load(NLP_EMBS_FILE)
        X_nlp = nlp_data["X"].astype(np.float64)
        y_nlp = nlp_data["y"].astype(np.int64)
        log(f"NLP: N={X_nlp.shape[0]}, d={X_nlp.shape[1]}")
        nlp_class_names = {i: f"dbp{i}" for i in range(20)}
        K_subsets_nlp = [k for k in K_SUBSETS_NLP if k <= len(np.unique(y_nlp))]
        nlp_K_results = run_dataset(X_nlp, y_nlp, NLP_A_LOAO, K_subsets_nlp,
                                    nlp_class_names, log, "NLP")
        all_results["nlp"] = {
            "dataset": "Pythia-160m DBpedia layer 12",
            "A_loao": NLP_A_LOAO,
            "K_subsets": K_subsets_nlp,
            "K_results": {str(k): v for k, v in nlp_K_results.items()},
        }
    except Exception as e:
        log(f"NLP failed: {e}")

    # ----------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("FINAL VERDICT")
    log("=" * 70)
    log("Pre-registered: M1 (full Gumbel) beats M0 (nearest-only) at K_max")
    log()
    log(f"{'Dataset':>8} {'K':>4} {'M0_R2':>8} {'M1_R2':>8} {'Delta':>8} {'A_fit':>8} {'A_LOAO':>8} {'DeltaPass':>10}")
    log("-" * 75)
    for ds_name, ds_data in all_results.items():
        A_loao = ds_data["A_loao"]
        for K_str, K_data in ds_data["K_results"].items():
            K_val = K_data["K"]
            m0 = K_data["mean_m0_R2"]
            m1 = K_data["mean_m1_R2"]
            delta = K_data["mean_delta_R2"]
            A_fit = K_data["mean_A_fit"]
            dp = K_data.get("pre_reg_delta")
            log(f"{ds_name:>8} {K_val:>4} {m0:>8.3f} {m1:>8.3f} {delta:>+8.3f} "
                f"{A_fit:>8.3f} {A_loao:>8.3f} "
                f"{'PASS' if dp else ('FAIL' if dp is False else 'N/A'):>10}")

    # Summary
    log()
    log("KEY QUESTION: Does M1 (full sum) beat M0 (nearest-only)?")
    for ds_name, ds_data in all_results.items():
        K_max = max(int(k) for k in ds_data["K_results"].keys())
        if str(K_max) in ds_data["K_results"]:
            K_data = ds_data["K_results"][str(K_max)]
            delta = K_data["mean_delta_R2"]
            m1 = K_data["mean_m1_R2"]
            log(f"  {ds_name} K={K_max}: M1 delta={delta:+.3f}, M1 R2={m1:.3f}")
            pass_delta = delta >= DELTA_R2_THRESHOLD
            pass_M1 = m1 >= M1_R2_THRESHOLD
            log(f"    delta>={DELTA_R2_THRESHOLD}: {'PASS' if pass_delta else 'FAIL'}")
            log(f"    M1_R2>={M1_R2_THRESHOLD}: {'PASS' if pass_M1 else 'FAIL'}")

    out = {
        "experiment": "full_gumbel_race_K_test",
        "description": "Test whether full Gumbel Race (M1) beats nearest-only (M0) across K values",
        "pre_registered": {
            "delta_R2_threshold": DELTA_R2_THRESHOLD,
            "M1_R2_threshold": M1_R2_THRESHOLD,
            "A_tolerance": A_TOLERANCE,
        },
        **all_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    log(f"\nSaved to {OUT_JSON}")
    log_file.close()


if __name__ == "__main__":
    main()
