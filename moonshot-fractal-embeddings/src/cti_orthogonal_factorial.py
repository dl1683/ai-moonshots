#!/usr/bin/env python -u
"""
ORTHOGONAL CAUSAL FACTORIAL (Feb 21 2026)
==========================================
Codex recommendation: highest-leverage missing experiment.

DESIGN:
  Two surgery arms on FROZEN pre-trained embeddings (Pythia-160m, DBpedia K=14):

  ARM A (kappa intervention): Apply centroid_shift on (ci, j1) — moves BOTH ci and j1
    -> kappa_nearest(ci) changes
    -> K_eff also changes (j1 was the dominant competitor)

  ARM B (K_eff intervention, ORTHOGONAL): Move ONLY j2 (second nearest to ci)
    -> ci stays put -> kappa_nearest(ci) to j1: UNCHANGED
    -> j2 becomes farther -> K_eff decreases
    -> This is the ORTHOGONAL intervention on K_eff

  ARM C (negative control): Move ONLY jK (farthest competitor from ci)
    -> kappa_nearest: UNCHANGED
    -> K_eff: minimally changed (jK was already irrelevant)
    -> Should have near-ZERO effect on q_ci

PRE-REGISTERED CRITERIA:
  1. Arm A: r(delta_kappa, delta_logit_q_ci) > 0.90 [kappa causes q]
  2. Arm B:
     - r(delta_j2_dist, delta_logit_q_ci) > 0.50 -> SUPPORTS 2-layer law (K_eff matters)
     - r(delta_j2_dist, delta_logit_q_ci) < 0.20 -> SUPPORTS 1-layer law (K_eff irrelevant)
  3. Arm C: |r(delta_jK_dist, delta_logit_q_ci)| < 0.20 [farthest competitor is irrelevant]

NOTE: We run all 14 focus classes and aggregate to reduce 1NN noise.
Aggregation: Fisher z-transform of per-class r values, then mean and CI.
"""

import json
import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# ================================================================
# CONFIG
# ================================================================
EMBS_FILE   = "results/do_int_embs_pythia-160m_dbpedia.npz"
OUT_JSON    = "results/cti_orthogonal_factorial.json"
OUT_LOG     = "results/cti_orthogonal_factorial_log.txt"

N_DELTA     = 13                           # delta points per arm
DELTA_MAX_A = 3.0                          # Arm A: push nearest +-3 embedding units
DELTA_MAX_B = 5.0                          # Arm B: push 2nd-nearest up to +5 units
DELTA_B_RANGE = np.linspace(0.0, DELTA_MAX_B, N_DELTA)   # B: only push farther
DELTA_A_RANGE = np.linspace(-DELTA_MAX_A, DELTA_MAX_A, N_DELTA)

N_CV_SPLITS = 10       # 10-fold stratified CV for q_ci estimation (more stable)
MIN_SAMPLES = 10       # minimum class samples to attempt surgery

# PRE-REGISTERED thresholds
PRE_REG_ARM_A_R  = 0.90
PRE_REG_ARM_B_R  = 0.50   # threshold above which 2-layer law is supported
PRE_REG_ARM_C_R  = 0.20   # threshold below which negative control passes


# ================================================================
# GEOMETRY HELPERS
# ================================================================
def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        centroids[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R**2) * X.shape[1]))   # tr(Sigma_W)/d * d = tr(Sigma_W)
    # sigma_W = sqrt(mean_squared_residual * d) = sqrt(tr(Sigma_W)/d * d) = sqrt(tr(Sigma_W))
    # Actually standard: sigma_W = sqrt( sum ||x - mu_c||^2 / (N*d) )  (per-dim std)
    # We want kappa = ||mu_i - mu_j|| / (sigma_W * sqrt(d))
    # So sigma_W should be the per-dim pooled within-class std
    sigma_W = float(np.sqrt(np.mean(R**2)))   # per-dim std
    return centroids, sigma_W


def get_competitor_ranking(centroids, sigma_W, d, ci):
    """Return sorted list of (kappa_ij, j) for all j != ci, ascending."""
    mu_i = centroids[ci]
    ranking = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa_ij = dist / (sigma_W * np.sqrt(d) + 1e-10)
        ranking.append((kappa_ij, cj, dist))
    ranking.sort(key=lambda x: x[0])   # ascending: j1 is nearest (smallest kappa)
    return ranking   # (kappa_ij, j, dist)


def compute_per_class_q(X, y, ci, n_splits=N_CV_SPLITS):
    """10-fold stratified CV, return per-class i recall."""
    classes = np.unique(y)
    K = len(classes)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    recalls = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # Check minimum class samples
        if (y_tr == ci).sum() < 2:
            continue
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X_tr, y_tr)
        mask = (y_te == ci)
        if mask.sum() == 0:
            continue
        preds = knn.predict(X_te[mask])
        recall = float((preds == ci).mean())
        recalls.append(recall)
    if not recalls:
        return None
    q_raw = float(np.mean(recalls))
    K_inv = 1.0 / K
    if abs(1.0 - K_inv) < 1e-10:
        return None
    return float((q_raw - K_inv) / (1.0 - K_inv))


def logit(q):
    q = float(np.clip(q, 1e-5, 1-1e-5))
    return float(np.log(q / (1.0 - q)))


# ================================================================
# SURGERY FUNCTIONS
# ================================================================
def apply_centroid_shift(X, y, centroids, cj, ck, delta):
    """Standard: move cj and ck apart/together by delta (both move by delta/2)."""
    mu_j, mu_k = centroids[cj].copy(), centroids[ck].copy()
    diff = mu_k - mu_j
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] -= (delta / 2) * direction
    X_new[y == ck] += (delta / 2) * direction
    return X_new


def apply_competitor_shift(X, y, centroids, ci, cj, delta):
    """ORTHOGONAL: move ONLY competitor cj away from ci by delta.
    Focus class ci DOES NOT MOVE -> kappa_nearest(ci, j1) unchanged.
    delta > 0: push cj farther; delta < 0: pull cj closer.
    """
    mu_i = centroids[ci]
    mu_j = centroids[cj].copy()
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] += delta * direction   # ONLY cj moves
    return X_new


# ================================================================
# DOSE-RESPONSE
# ================================================================
def run_arm(X, y, centroids, sigma_W, ci, j_target, delta_range, arm_name, log_fn,
            surgery_fn):
    """Run one arm: sweep delta, measure geometry and q_ci."""
    d = X.shape[1]
    ranking = get_competitor_ranking(centroids, sigma_W, d, ci)
    kappa_j1_orig = ranking[0][0]
    kappa_jtgt_orig = None
    for k, j, dist in ranking:
        if j == j_target:
            kappa_jtgt_orig = k
            break

    records = []
    for delta in delta_range:
        X_new = surgery_fn(X, y, centroids, ci, j_target, delta)
        new_cents, new_sw = compute_class_stats(X_new, y)
        new_rank = get_competitor_ranking(new_cents, new_sw, d, ci)
        new_kappa_j1 = new_rank[0][0]   # kappa to current nearest
        new_j1 = new_rank[0][1]

        q_ci = compute_per_class_q(X_new, y, ci)
        if q_ci is None:
            continue
        lq = logit(q_ci)

        # Also compute K_eff proxy: effective rank of {kappa_ij}
        kappas_ij = np.array([k for k, j, d_ in new_rank])
        K_eff_kappa = float((kappas_ij.sum())**2 / (kappas_ij**2).sum()) if len(kappas_ij) > 0 else 0.0

        # Find new kappa of j_target
        new_kappa_jtgt = None
        for k, j, d_ in new_rank:
            if j == j_target:
                new_kappa_jtgt = k
                break

        rec = {
            "delta": float(delta),
            "kappa_nearest_new": float(new_kappa_j1),
            "new_j1": int(new_j1),
            "kappa_jtgt_new": float(new_kappa_jtgt) if new_kappa_jtgt is not None else None,
            "K_eff_kappa": K_eff_kappa,
            "q_ci": float(q_ci),
            "logit_q_ci": float(lq),
        }
        records.append(rec)
        log_fn(f"    [{arm_name} ci={ci}] delta={delta:+.2f}: kappa_j1={new_kappa_j1:.3f} "
               f"(orig={kappa_j1_orig:.3f}), K_eff={K_eff_kappa:.2f}, q_ci={q_ci:.4f}")

    return records, float(kappa_j1_orig), float(kappa_jtgt_orig) if kappa_jtgt_orig is not None else None


def fit_arm_r(records, x_key, y_key="logit_q_ci"):
    xs = np.array([r[x_key] for r in records if r.get(x_key) is not None])
    ys = np.array([r[y_key] for r in records if r.get(x_key) is not None])
    if len(xs) < 4 or np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
        return 0.0, 1.0
    r, p = stats.pearsonr(xs, ys)
    return float(r), float(p)


def fisher_z_mean(rs):
    """Mean r via Fisher z-transform."""
    zs = np.arctanh(np.clip(rs, -0.9999, 0.9999))
    return float(np.tanh(zs.mean()))


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
    log("ORTHOGONAL CAUSAL FACTORIAL")
    log("=" * 70)
    log(f"Embeddings: {EMBS_FILE}")
    log(f"PRE-REGISTERED:")
    log(f"  Arm A: r(delta_kappa_j1, delta_logit_q) > {PRE_REG_ARM_A_R}")
    log(f"  Arm B: r(delta_j2_dist, delta_logit_q) > {PRE_REG_ARM_B_R} -> 2-layer supported")
    log(f"         r(delta_j2_dist, delta_logit_q) < {PRE_REG_ARM_B_R} -> 1-layer supported")
    log(f"  Arm C: |r(delta_jK_dist, delta_logit_q)| < {PRE_REG_ARM_C_R} [neg control]")
    log("=" * 70)

    # Load embeddings
    data = np.load(EMBS_FILE)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    K = len(classes)
    log(f"\nLoaded: N={X.shape[0]}, d={d}, K={K}")

    centroids, sigma_W = compute_class_stats(X, y)
    log(f"Baseline sigma_W={sigma_W:.4f}")

    # Compute baseline geometry
    log("\n[BASELINE GEOMETRY]")
    for ci in classes:
        ranking = get_competitor_ranking(centroids, sigma_W, d, ci)
        j1 = ranking[0][1]; k1 = ranking[0][0]
        j2 = ranking[1][1]; k2 = ranking[1][0]
        jK = ranking[-1][1]; kK = ranking[-1][0]
        log(f"  class {ci}: j1={j1} (kappa={k1:.3f}), j2={j2} (kappa={k2:.3f}), "
            f"jK={jK} (kappa={kK:.3f}), margin={k2/k1:.2f}x")

    # Per-class baseline q
    log("\n[BASELINE q_ci]")
    baseline_q = {}
    baseline_logit = {}
    for ci in classes:
        q = compute_per_class_q(X, y, ci)
        if q is not None:
            baseline_q[ci] = q
            baseline_logit[ci] = logit(q)
            log(f"  class {ci}: q_ci={q:.4f}, logit_q={logit(q):.4f}")
        else:
            log(f"  class {ci}: SKIP (too few samples)")

    # ----------------------------------------------------------------
    # RUN ARMS
    # ----------------------------------------------------------------
    all_results = {}

    for ci in classes:
        if ci not in baseline_q:
            continue
        ranking = get_competitor_ranking(centroids, sigma_W, d, ci)
        j1 = ranking[0][1]
        j2 = ranking[1][1]
        jK = ranking[-1][1]

        if (y == ci).sum() < MIN_SAMPLES:
            log(f"\n  SKIP class {ci} (< {MIN_SAMPLES} samples)")
            continue

        log(f"\n{'='*60}")
        log(f"FOCUS CLASS {ci} | j1={j1} j2={j2} jK={jK}")
        log(f"{'='*60}")

        ci_results = {"class": int(ci), "j1": int(j1), "j2": int(j2), "jK": int(jK)}

        # ARM A: standard centroid shift on (ci, j1)
        log(f"\n  [ARM A] kappa intervention (ci={ci} <-> j1={j1})")
        recs_A, kappa_j1_orig, _ = run_arm(
            X, y, centroids, sigma_W, ci, j1,
            DELTA_A_RANGE, "A",
            log,
            apply_centroid_shift
        )
        r_A_kappa, p_A_kappa = fit_arm_r(recs_A, "kappa_nearest_new")
        r_A_Keff, p_A_Keff   = fit_arm_r(recs_A, "K_eff_kappa")
        log(f"  ARM A: r(kappa_j1, logit_q)={r_A_kappa:.3f} p={p_A_kappa:.4f} | "
            f"r(K_eff, logit_q)={r_A_Keff:.3f} p={p_A_Keff:.4f}")
        ci_results["arm_A"] = {
            "records": recs_A,
            "r_kappa_j1_logit": r_A_kappa, "p_kappa_j1_logit": p_A_kappa,
            "r_Keff_logit": r_A_Keff, "p_Keff_logit": p_A_Keff,
            "kappa_j1_orig": kappa_j1_orig,
        }

        # ARM B: ORTHOGONAL — move ONLY j2, class ci fixed
        log(f"\n  [ARM B] K_eff intervention (ONLY j2={j2} moves, ci={ci} fixed)")
        recs_B, _, kappa_j2_orig = run_arm(
            X, y, centroids, sigma_W, ci, j2,
            DELTA_B_RANGE, "B",
            log,
            apply_competitor_shift
        )
        r_B_kappa, p_B_kappa = fit_arm_r(recs_B, "kappa_nearest_new")
        r_B_Keff,  p_B_Keff  = fit_arm_r(recs_B, "K_eff_kappa")
        r_B_jtgt,  p_B_jtgt  = fit_arm_r(recs_B, "kappa_jtgt_new")
        log(f"  ARM B: r(kappa_j1, logit_q)={r_B_kappa:.3f} p={p_B_kappa:.4f} | "
            f"r(K_eff, logit_q)={r_B_Keff:.3f} p={p_B_Keff:.4f} | "
            f"r(kappa_j2, logit_q)={r_B_jtgt:.3f} p={p_B_jtgt:.4f}")
        ci_results["arm_B"] = {
            "records": recs_B,
            "r_kappa_j1_logit": r_B_kappa, "p_kappa_j1_logit": p_B_kappa,
            "r_Keff_logit": r_B_Keff, "p_Keff_logit": p_B_Keff,
            "r_kappa_j2_logit": r_B_jtgt, "p_kappa_j2_logit": p_B_jtgt,
            "kappa_j2_orig": kappa_j2_orig,
        }

        # ARM C: negative control — move ONLY jK (farthest), ci fixed
        log(f"\n  [ARM C] Neg control (ONLY jK={jK} moves, ci={ci} fixed)")
        recs_C, _, kappa_jK_orig = run_arm(
            X, y, centroids, sigma_W, ci, jK,
            DELTA_B_RANGE, "C",
            log,
            apply_competitor_shift
        )
        r_C_kappa, p_C_kappa = fit_arm_r(recs_C, "kappa_nearest_new")
        r_C_Keff,  p_C_Keff  = fit_arm_r(recs_C, "K_eff_kappa")
        r_C_jtgt,  p_C_jtgt  = fit_arm_r(recs_C, "kappa_jtgt_new")
        log(f"  ARM C: r(kappa_j1, logit_q)={r_C_kappa:.3f} p={p_C_kappa:.4f} | "
            f"r(K_eff, logit_q)={r_C_Keff:.3f} p={p_C_Keff:.4f} | "
            f"r(kappa_jK, logit_q)={r_C_jtgt:.3f} p={p_C_jtgt:.4f}")
        ci_results["arm_C"] = {
            "records": recs_C,
            "r_kappa_j1_logit": r_C_kappa, "p_kappa_j1_logit": p_C_kappa,
            "r_Keff_logit": r_C_Keff, "p_Keff_logit": p_C_Keff,
            "r_kappa_jK_logit": r_C_jtgt, "p_kappa_jK_logit": p_C_jtgt,
            "kappa_jK_orig": kappa_jK_orig,
        }

        all_results[str(ci)] = ci_results

    # ----------------------------------------------------------------
    # AGGREGATE ACROSS FOCUS CLASSES
    # ----------------------------------------------------------------
    log("\n" + "="*70)
    log("AGGREGATE RESULTS (Fisher z-mean across focus classes)")
    log("="*70)

    rs_A_kappa = [all_results[str(ci)]["arm_A"]["r_kappa_j1_logit"]
                  for ci in classes if str(ci) in all_results]
    rs_B_Keff  = [all_results[str(ci)]["arm_B"]["r_Keff_logit"]
                  for ci in classes if str(ci) in all_results]
    rs_B_j2    = [all_results[str(ci)]["arm_B"]["r_kappa_j2_logit"]
                  for ci in classes if str(ci) in all_results]
    rs_C_jK    = [all_results[str(ci)]["arm_C"]["r_kappa_jK_logit"]
                  for ci in classes if str(ci) in all_results]
    rs_B_kappa = [all_results[str(ci)]["arm_B"]["r_kappa_j1_logit"]
                  for ci in classes if str(ci) in all_results]

    mean_r_A_kappa = fisher_z_mean(np.array(rs_A_kappa)) if rs_A_kappa else 0.0
    mean_r_B_Keff  = fisher_z_mean(np.array(rs_B_Keff))  if rs_B_Keff else 0.0
    mean_r_B_j2    = fisher_z_mean(np.array(rs_B_j2))    if rs_B_j2 else 0.0
    mean_r_C_jK    = fisher_z_mean(np.array(rs_C_jK))    if rs_C_jK else 0.0
    mean_r_B_kappa = fisher_z_mean(np.array(rs_B_kappa)) if rs_B_kappa else 0.0

    log(f"\nArm A (kappa intervention, n={len(rs_A_kappa)} classes):")
    log(f"  r(kappa_j1, logit_q) = {mean_r_A_kappa:.3f}  "
        f"[pre-reg threshold > {PRE_REG_ARM_A_R}]")
    log(f"  PASS" if abs(mean_r_A_kappa) >= PRE_REG_ARM_A_R else f"  FAIL")

    log(f"\nArm B (K_eff intervention, ORTHOGONAL, n={len(rs_B_Keff)} classes):")
    log(f"  r(kappa_j1, logit_q) = {mean_r_B_kappa:.3f}  [should be ~0: kappa_j1 unchanged]")
    log(f"  r(K_eff_kappa, logit_q) = {mean_r_B_Keff:.3f}")
    log(f"  r(kappa_j2, logit_q)  = {mean_r_B_j2:.3f}  "
        f"[pre-reg: > {PRE_REG_ARM_B_R} = 2-layer; < {PRE_REG_ARM_B_R} = 1-layer]")
    if abs(mean_r_B_j2) >= PRE_REG_ARM_B_R:
        log(f"  -> 2-LAYER LAW SUPPORTED (K_eff matters)")
    else:
        log(f"  -> 1-LAYER LAW SUPPORTED (K_eff irrelevant)")

    log(f"\nArm C (neg control, n={len(rs_C_jK)} classes):")
    log(f"  r(kappa_jK, logit_q) = {mean_r_C_jK:.3f}  "
        f"[pre-reg: < {PRE_REG_ARM_C_R}]")
    log(f"  PASS" if abs(mean_r_C_jK) < PRE_REG_ARM_C_R else f"  FAIL")

    # ----------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------
    log("\n" + "="*70)
    log("VERDICT")
    log("="*70)
    verdict_A = abs(mean_r_A_kappa) >= PRE_REG_ARM_A_R
    verdict_B_2layer = abs(mean_r_B_j2) >= PRE_REG_ARM_B_R
    verdict_C = abs(mean_r_C_jK) < PRE_REG_ARM_C_R
    log(f"  Arm A (kappa causal): {'PASS' if verdict_A else 'FAIL'} (r={mean_r_A_kappa:.3f})")
    log(f"  Arm B (K_eff matters): {'2-LAYER SUPPORTED' if verdict_B_2layer else '1-LAYER SUPPORTED'} "
        f"(r={mean_r_B_j2:.3f})")
    log(f"  Arm C (neg control): {'PASS' if verdict_C else 'FAIL'} (r={mean_r_C_jK:.3f})")

    # Save
    out = {
        "experiment": "orthogonal_causal_factorial",
        "description": "Arm A: kappa intervention. Arm B: K_eff ORTHOGONAL (only j2 moves). Arm C: neg control.",
        "pre_registered": {
            "arm_A_r_threshold": PRE_REG_ARM_A_R,
            "arm_B_r_threshold_2layer": PRE_REG_ARM_B_R,
            "arm_C_r_threshold": PRE_REG_ARM_C_R,
        },
        "aggregate": {
            "n_focus_classes": len(rs_A_kappa),
            "arm_A_r_kappa_j1": mean_r_A_kappa,
            "arm_A_pass": bool(verdict_A),
            "arm_B_r_kappa_j1_unchanged": mean_r_B_kappa,
            "arm_B_r_Keff": mean_r_B_Keff,
            "arm_B_r_kappa_j2": mean_r_B_j2,
            "arm_B_verdict": "2-layer" if verdict_B_2layer else "1-layer",
            "arm_C_r_kappa_jK": mean_r_C_jK,
            "arm_C_pass": bool(verdict_C),
        },
        "per_class": all_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    log(f"\nSaved to {OUT_JSON}")
    log_file.close()


if __name__ == "__main__":
    main()
