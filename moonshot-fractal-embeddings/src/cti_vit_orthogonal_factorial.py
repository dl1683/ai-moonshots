#!/usr/bin/env python -u
"""
VIT CROSS-MODALITY ORTHOGONAL CAUSAL FACTORIAL (Feb 21-22 2026)
===============================================================
SAME protocol as NLP (cti_orthogonal_factorial.py) but using
ViT-Large-16-224 CIFAR-10 frozen layer-12 embeddings.

PURPOSE: Replicate the orthogonal causal factorial in vision modality.
Tests whether kappa_nearest is the CAUSAL driver in vision as well as NLP.

DESIGN (identical to NLP test):
  ARM A: Apply centroid_shift on (ci, j1) -- BOTH move
    -> kappa_nearest(ci) changes
  ARM B: Move ONLY j2 (ORTHOGONAL) -- ci fixed, kappa_j1 unchanged
    -> tests whether 2nd-nearest competitor causally affects q
  ARM C: Move ONLY jK (farthest) -- negative control

PRE-REGISTERED CRITERIA:
  1. Arm A: r(delta_kappa, delta_logit_q) > 0.90 [kappa causes q, cross-modality]
  2. Arm B: r(kappa_j2, logit_q) < 0.50 -> 1-layer; > 0.50 -> 2-layer
  3. Arm C: |r| < 0.20 [negative control]

EXPECTED RESULTS:
  - Class 4: margin 1.05x (VERY tight), predict B_j2_r ~ 0.95 (high j2 effect)
  - Class 6: margin 1.03x (VERY tight), predict B_j2_r ~ 0.97
  - Class 3/5: margin 1.88/1.94x, predict B_j2_r ~ 0.45 (moderate)
  - Aggregate Arm B r predicted > 0.50 (2-layer law in vision domain)

EMBEDDINGS: vit_loao_embs_vit-large-patch16-224_cifar10.npz, layer 12
  - N=10000, d=1024, K=10 (CIFAR-10), 1000 per class
  - Subsample 500/class (5000 total) for computational feasibility
  - q_baseline ~ 0.678 (not at ceiling), kappa_baseline ~ 0.206
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
EMBS_FILE   = "results/vit_loao_embs_vit-large-patch16-224_cifar10.npz"
LAYER_KEY   = "12"               # Layer 12: q~0.678, kappa~0.206, not at ceiling
SUBSAMPLE   = 500                # samples per class (5000 total)
RANDOM_SEED = 42

OUT_JSON    = "results/cti_vit_orthogonal_factorial.json"
OUT_LOG     = "results/cti_vit_orthogonal_factorial_log.txt"

N_DELTA       = 11
DELTA_MAX_A   = 3.0              # Arm A: push nearest +-3 embedding units
DELTA_MAX_B   = 4.0              # Arm B: push 2nd-nearest up to +4 units
DELTA_A_RANGE = np.linspace(-DELTA_MAX_A, DELTA_MAX_A, N_DELTA)
DELTA_B_RANGE = np.linspace(0.0, DELTA_MAX_B, N_DELTA)

N_CV_SPLITS = 5                  # 5-fold CV (faster than 10-fold)
MIN_SAMPLES = 10

# PRE-REGISTERED thresholds
PRE_REG_ARM_A_R  = 0.90
PRE_REG_ARM_B_R  = 0.50
PRE_REG_ARM_C_R  = 0.20

CIFAR10_NAMES = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat',
                 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}


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
    sigma_W = float(np.sqrt(np.mean(R**2)))   # per-dim pooled within-class std
    return centroids, sigma_W


def get_competitor_ranking(centroids, sigma_W, d, ci):
    """Return sorted list of (kappa_ij, j, dist) for all j != ci, ascending kappa."""
    mu_i = centroids[ci]
    ranking = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa_ij = dist / (sigma_W * np.sqrt(d) + 1e-10)
        ranking.append((kappa_ij, cj, dist))
    ranking.sort(key=lambda x: x[0])
    return ranking


def compute_per_class_q(X, y, ci, n_splits=N_CV_SPLITS):
    """n-fold stratified CV, return per-class i normalized recall."""
    classes = np.unique(y)
    K = len(classes)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    recalls = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
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
def apply_centroid_shift(X, y, centroids, ci, j_target, delta):
    """Standard: move ci and j_target apart/together by delta (both move by delta/2)."""
    mu_i, mu_j = centroids[ci].copy(), centroids[j_target].copy()
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == ci]       -= (delta / 2) * direction
    X_new[y == j_target] += (delta / 2) * direction
    return X_new


def apply_competitor_shift(X, y, centroids, ci, cj, delta):
    """ORTHOGONAL: move ONLY competitor cj away from ci by delta.
    Focus class ci DOES NOT MOVE -> kappa_nearest(ci, j1) unchanged.
    """
    mu_i = centroids[ci]
    mu_j = centroids[cj].copy()
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] += delta * direction
    return X_new


# ================================================================
# DOSE-RESPONSE RUNNER
# ================================================================
def run_arm(X, y, centroids, sigma_W, ci, j_target, delta_range, arm_name, log_fn,
            surgery_fn):
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
        new_kappa_j1 = new_rank[0][0]
        new_j1 = new_rank[0][1]

        q_ci = compute_per_class_q(X_new, y, ci)
        if q_ci is None:
            continue
        lq = logit(q_ci)

        kappas_ij = np.array([k for k, j, d_ in new_rank])
        K_eff_kappa = float((kappas_ij.sum())**2 / (kappas_ij**2).sum()) if len(kappas_ij) > 0 else 0.0

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
        log_fn(f"    [{arm_name} ci={ci}({CIFAR10_NAMES[ci]})] delta={delta:+.2f}: "
               f"kappa_j1={new_kappa_j1:.3f} (orig={kappa_j1_orig:.3f}), q_ci={q_ci:.4f}")

    return records, float(kappa_j1_orig), float(kappa_jtgt_orig) if kappa_jtgt_orig is not None else None


def fit_arm_r(records, x_key, y_key="logit_q_ci"):
    xs = np.array([r[x_key] for r in records if r.get(x_key) is not None])
    ys = np.array([r[y_key] for r in records if r.get(x_key) is not None])
    if len(xs) < 4 or np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
        return 0.0, 1.0
    r, p = stats.pearsonr(xs, ys)
    return float(r), float(p)


def fisher_z_mean(rs):
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
    log("VIT CROSS-MODALITY ORTHOGONAL CAUSAL FACTORIAL")
    log("=" * 70)
    log(f"Embeddings: {EMBS_FILE}, layer {LAYER_KEY}")
    log(f"Subsample: {SUBSAMPLE}/class")
    log(f"PRE-REGISTERED:")
    log(f"  Arm A: r(kappa_j1, logit_q) > {PRE_REG_ARM_A_R}")
    log(f"  Arm B: r(kappa_j2, logit_q) > {PRE_REG_ARM_B_R} -> 2-layer; < {PRE_REG_ARM_B_R} -> 1-layer")
    log(f"  Arm C: |r| < {PRE_REG_ARM_C_R} [neg control]")
    log("=" * 70)

    # Load embeddings
    data = np.load(EMBS_FILE)
    X_all = data[LAYER_KEY].astype(np.float64)
    y_all = data["y"].astype(np.int64)
    classes = sorted(np.unique(y_all).tolist())
    K = len(classes)
    log(f"\nFull dataset: N={X_all.shape[0]}, d={X_all.shape[1]}, K={K}")

    # Subsample
    rng = np.random.RandomState(RANDOM_SEED)
    keep = []
    for c in classes:
        idx_c = np.where(y_all == c)[0]
        chosen = rng.choice(idx_c, size=min(SUBSAMPLE, len(idx_c)), replace=False)
        keep.append(chosen)
    keep = np.concatenate(keep)
    keep = np.sort(keep)
    X = X_all[keep]
    y = y_all[keep]
    d = X.shape[1]
    log(f"Subsampled: N={X.shape[0]}, d={d}, K={K} ({SUBSAMPLE}/class)")

    # Baseline geometry
    centroids, sigma_W = compute_class_stats(X, y)
    log(f"Baseline sigma_W={sigma_W:.4f}")

    log("\n[BASELINE GEOMETRY]")
    for ci in classes:
        ranking = get_competitor_ranking(centroids, sigma_W, d, ci)
        j1, k1, dist1 = ranking[0][1], ranking[0][0], ranking[0][2]
        j2, k2 = ranking[1][1], ranking[1][0]
        jK, kK = ranking[-1][1], ranking[-1][0]
        margin = k2 / k1
        name_i = CIFAR10_NAMES[ci]
        name_j1 = CIFAR10_NAMES[j1]
        name_j2 = CIFAR10_NAMES[j2]
        log(f"  class {ci}({name_i}): j1={j1}({name_j1}) kappa={k1:.3f}, "
            f"j2={j2}({name_j2}) kappa={k2:.3f}, margin={margin:.2f}x")

    log("\n[BASELINE q_ci]")
    baseline_q = {}
    baseline_logit = {}
    for ci in classes:
        q = compute_per_class_q(X, y, ci)
        if q is not None:
            baseline_q[ci] = q
            baseline_logit[ci] = logit(q)
            name_i = CIFAR10_NAMES[ci]
            log(f"  class {ci}({name_i}): q_ci={q:.4f}, logit_q={logit(q):.4f}")
        else:
            log(f"  class {ci}: SKIP")

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
        margin = ranking[1][0] / ranking[0][0]

        if (y == ci).sum() < MIN_SAMPLES:
            log(f"\n  SKIP class {ci} (< {MIN_SAMPLES} samples)")
            continue

        name_i = CIFAR10_NAMES[ci]
        name_j1 = CIFAR10_NAMES[j1]
        name_j2 = CIFAR10_NAMES[j2]
        name_jK = CIFAR10_NAMES[jK]

        log(f"\n{'='*60}")
        log(f"FOCUS CLASS {ci}({name_i}) | j1={j1}({name_j1}) j2={j2}({name_j2}) "
            f"jK={jK}({name_jK}) margin={margin:.2f}x")
        log(f"{'='*60}")

        ci_results = {
            "class": int(ci),
            "class_name": name_i,
            "j1": int(j1), "j1_name": name_j1,
            "j2": int(j2), "j2_name": name_j2,
            "jK": int(jK), "jK_name": name_jK,
            "margin_j2_j1": float(margin),
            "baseline_q": float(baseline_q[ci]),
        }

        # ARM A
        log(f"\n  [ARM A] kappa intervention ({name_i} <-> {name_j1})")
        recs_A, kappa_j1_orig, _ = run_arm(
            X, y, centroids, sigma_W, ci, j1,
            DELTA_A_RANGE, "A", log, apply_centroid_shift)
        r_A_kappa, p_A_kappa = fit_arm_r(recs_A, "kappa_nearest_new")
        r_A_Keff,  p_A_Keff  = fit_arm_r(recs_A, "K_eff_kappa")
        log(f"  ARM A: r(kappa_j1, logit_q)={r_A_kappa:.3f} p={p_A_kappa:.4f}")
        ci_results["arm_A"] = {
            "records": recs_A,
            "r_kappa_j1_logit": r_A_kappa, "p_kappa_j1_logit": p_A_kappa,
            "r_Keff_logit": r_A_Keff,      "p_Keff_logit": p_A_Keff,
            "kappa_j1_orig": kappa_j1_orig,
        }

        # ARM B: ORTHOGONAL
        log(f"\n  [ARM B] ORTHOGONAL — only {name_j2} moves, {name_i} fixed")
        recs_B, _, kappa_j2_orig = run_arm(
            X, y, centroids, sigma_W, ci, j2,
            DELTA_B_RANGE, "B", log, apply_competitor_shift)
        r_B_kappa, p_B_kappa = fit_arm_r(recs_B, "kappa_nearest_new")
        r_B_Keff,  p_B_Keff  = fit_arm_r(recs_B, "K_eff_kappa")
        r_B_jtgt,  p_B_jtgt  = fit_arm_r(recs_B, "kappa_jtgt_new")
        log(f"  ARM B: r(kappa_j1_unchanged, logit_q)={r_B_kappa:.3f}, "
            f"r(kappa_j2, logit_q)={r_B_jtgt:.3f} p={p_B_jtgt:.4f}")
        ci_results["arm_B"] = {
            "records": recs_B,
            "r_kappa_j1_logit": r_B_kappa, "p_kappa_j1_logit": p_B_kappa,
            "r_Keff_logit": r_B_Keff,      "p_Keff_logit": p_B_Keff,
            "r_kappa_j2_logit": r_B_jtgt,  "p_kappa_j2_logit": p_B_jtgt,
            "kappa_j2_orig": kappa_j2_orig,
        }

        # ARM C: negative control
        log(f"\n  [ARM C] Neg control — only {name_jK} moves, {name_i} fixed")
        recs_C, _, kappa_jK_orig = run_arm(
            X, y, centroids, sigma_W, ci, jK,
            DELTA_B_RANGE, "C", log, apply_competitor_shift)
        r_C_kappa, p_C_kappa = fit_arm_r(recs_C, "kappa_nearest_new")
        r_C_Keff,  p_C_Keff  = fit_arm_r(recs_C, "K_eff_kappa")
        r_C_jtgt,  p_C_jtgt  = fit_arm_r(recs_C, "kappa_jtgt_new")
        log(f"  ARM C: r(kappa_jK, logit_q)={r_C_jtgt:.3f} p={p_C_jtgt:.4f}")
        ci_results["arm_C"] = {
            "records": recs_C,
            "r_kappa_j1_logit": r_C_kappa, "p_kappa_j1_logit": p_C_kappa,
            "r_Keff_logit": r_C_Keff,      "p_Keff_logit": p_C_Keff,
            "r_kappa_jK_logit": r_C_jtgt,  "p_kappa_jK_logit": p_C_jtgt,
            "kappa_jK_orig": kappa_jK_orig,
        }

        all_results[str(ci)] = ci_results

    # ----------------------------------------------------------------
    # AGGREGATE
    # ----------------------------------------------------------------
    log("\n" + "="*70)
    log("AGGREGATE RESULTS (Fisher z-mean across focus classes)")
    log("="*70)

    rs_A_kappa = [all_results[str(ci)]["arm_A"]["r_kappa_j1_logit"]
                  for ci in classes if str(ci) in all_results]
    rs_B_j2    = [all_results[str(ci)]["arm_B"]["r_kappa_j2_logit"]
                  for ci in classes if str(ci) in all_results]
    rs_B_kappa = [all_results[str(ci)]["arm_B"]["r_kappa_j1_logit"]
                  for ci in classes if str(ci) in all_results]
    rs_C_jK    = [all_results[str(ci)]["arm_C"]["r_kappa_jK_logit"]
                  for ci in classes if str(ci) in all_results]

    mean_r_A_kappa = fisher_z_mean(np.array(rs_A_kappa)) if rs_A_kappa else 0.0
    mean_r_B_j2    = fisher_z_mean(np.array(rs_B_j2))    if rs_B_j2 else 0.0
    mean_r_B_kappa = fisher_z_mean(np.array(rs_B_kappa)) if rs_B_kappa else 0.0
    mean_r_C_jK    = fisher_z_mean(np.array(rs_C_jK))    if rs_C_jK else 0.0

    verdict_A        = abs(mean_r_A_kappa) >= PRE_REG_ARM_A_R
    verdict_B_2layer = abs(mean_r_B_j2) >= PRE_REG_ARM_B_R
    verdict_C        = abs(mean_r_C_jK) < PRE_REG_ARM_C_R

    log(f"\nArm A (kappa causal, n={len(rs_A_kappa)} classes):")
    log(f"  r(kappa_j1, logit_q) = {mean_r_A_kappa:.3f}  [threshold > {PRE_REG_ARM_A_R}]")
    log(f"  {'PASS' if verdict_A else 'FAIL'}")

    log(f"\nArm B (ORTHOGONAL — only j2 moves, n={len(rs_B_j2)} classes):")
    log(f"  r(kappa_j1_UNCHANGED, logit_q) = {mean_r_B_kappa:.3f}  [should be ~0]")
    log(f"  r(kappa_j2, logit_q) = {mean_r_B_j2:.3f}  [> {PRE_REG_ARM_B_R} -> 2-layer; < -> 1-layer]")
    if verdict_B_2layer:
        log(f"  -> 2-LAYER LAW SUPPORTED (j2 causally affects q)")
    else:
        log(f"  -> 1-LAYER LAW SUPPORTED (j2 irrelevant)")

    log(f"\nArm C (neg control, n={len(rs_C_jK)} classes):")
    log(f"  r(kappa_jK, logit_q) = {mean_r_C_jK:.3f}  [threshold < {PRE_REG_ARM_C_R}]")
    log(f"  {'PASS' if verdict_C else 'FAIL'}")

    # Per-class summary table
    log("\n[PER-CLASS SUMMARY]")
    log(f"{'class':>5} {'name':>10} {'margin':>8} {'q_base':>7} "
        f"{'A_r':>6} {'B_j1_r':>8} {'B_j2_r':>8} {'C_r':>6}")
    log("-" * 65)
    for ci in classes:
        if str(ci) not in all_results:
            continue
        r = all_results[str(ci)]
        margin = r["margin_j2_j1"]
        q_base = r["baseline_q"]
        A_r = r["arm_A"]["r_kappa_j1_logit"]
        B_j1_r = r["arm_B"]["r_kappa_j1_logit"]
        B_j2_r = r["arm_B"]["r_kappa_j2_logit"]
        C_r = r["arm_C"]["r_kappa_jK_logit"]
        log(f"{ci:>5} {r['class_name']:>10} {margin:>8.2f}x {q_base:>7.4f} "
            f"{A_r:>6.3f} {B_j1_r:>8.3f} {B_j2_r:>8.3f} {C_r:>6.3f}")

    # ----------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------
    log("\n" + "="*70)
    log("FINAL VERDICT (cross-modality replication)")
    log("="*70)
    log(f"  Arm A (kappa causal):  {'PASS' if verdict_A else 'FAIL'} r={mean_r_A_kappa:.3f}")
    log(f"  Arm B (j2 effect):     {'2-LAYER' if verdict_B_2layer else '1-LAYER'} r={mean_r_B_j2:.3f}")
    log(f"  Arm C (neg control):   {'PASS' if verdict_C else 'FAIL'} r={mean_r_C_jK:.3f}")
    log(f"\n  NLP ref: Arm A r=0.899, Arm B r=0.450 (1-layer), Arm C r=0.000")
    if verdict_A and verdict_C:
        log(f"  CROSS-MODALITY CAUSAL REPLICATION: PASS")
    else:
        log(f"  CROSS-MODALITY CAUSAL REPLICATION: PARTIAL/FAIL")

    # Save
    out = {
        "experiment": "vit_cross_modality_orthogonal_factorial",
        "description": "ViT-Large CIFAR-10 layer 12, same protocol as NLP orthogonal factorial",
        "embeddings": EMBS_FILE,
        "layer": LAYER_KEY,
        "subsample_per_class": SUBSAMPLE,
        "pre_registered": {
            "arm_A_r_threshold": PRE_REG_ARM_A_R,
            "arm_B_r_threshold_2layer": PRE_REG_ARM_B_R,
            "arm_C_r_threshold": PRE_REG_ARM_C_R,
        },
        "nlp_reference": {
            "arm_A_r": 0.899, "arm_B_r": 0.450, "arm_C_r": 0.000,
            "verdict_B": "1-layer"
        },
        "aggregate": {
            "n_focus_classes": len(rs_A_kappa),
            "arm_A_r_kappa_j1": mean_r_A_kappa,
            "arm_A_pass": bool(verdict_A),
            "arm_B_r_kappa_j1_unchanged": mean_r_B_kappa,
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
