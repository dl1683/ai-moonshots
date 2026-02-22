#!/usr/bin/env python -u
"""
Causal Sufficiency RCT: kappa_nearest as sole causal variable
==============================================================
Pre-registration: results/cti_causal_sufficiency_rct_prereg.json

DESIGN (6 arms on frozen pythia-160m/DBpedia K=14):

  j1_shift (m=1): standard centroid-pair shift of (ci, j1) -- both move by +/-delta/2
    -> Delta_kappa_j1 changes. Tests direct causal path.

  j2_shift (orthogonal): move ONLY j2 outward from ci (ci and j1 FIXED)
    -> Delta_kappa_j1 ~ 0. Tests if j2 independently causes q.
    -> PRE-REG: r(delta_j2_dist, delta_logit) < 0.20 [kappa_nearest IS the causal var]

  jK_shift (neg control): move ONLY jK (farthest) outward from ci (ci and j1 FIXED)
    -> PRE-REG: r(delta_jK_dist, delta_logit) < 0.20

  top2_shift (m=2): move j1 AND j2 outward by delta each (ci FIXED)
  top4_shift (m=4): move top-4 competitors outward by delta each (ci FIXED)
  top_all_shift (m=K-1): move ALL K-1 competitors outward by delta each (ci FIXED)

PRE-REGISTERED PASS CRITERIA (from cti_causal_sufficiency_rct_prereg.json):
  1. arm_j1_r:       r(kappa_j1,   delta_logit) > 0.90
  2. arm_j2_r:       r(delta_j2,   delta_logit) < 0.20  [orthogonal control]
  3. arm_jK_r:       r(delta_jK,   delta_logit) < 0.20  [neg control]
  4. arm_top_all_r:  r(kappa_j1,   delta_logit) > 0.90  [j1 dominates even in top_all]
  5. fixed_slope_r2_pooled: R2(fixed slope=alpha_arch, x=total_delta_kappa) >= 0.85
       pooled across j1+top2+top4+top_all arms
  6. epsilon_invariance: per-arm mean(delta_logit - alpha_arch*delta_kappa_j1)
       within +/-0.10 for j1/j2/jK arms
  7. top_all_alpha_ratio: alpha_topall / alpha_j1 in [11.1, 15.1] = [0.85*(K-1), 1.15*(K-1)]
"""

import json
import os
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# ================================================================
# CONFIG
# ================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
EMBS_FILE = REPO_ROOT / "results" / "dointerv_multi_pythia-160m_l12.npz"
OUT_JSON  = REPO_ROOT / "results" / "cti_causal_sufficiency_rct.json"

# Pre-registered constants
ALPHA_ARCH = 1.052      # pythia-160m/DBpedia single-pair result (Session 37)
A_RENORM   = 1.0535     # canonical universal constant
K          = 14         # DBpedia14

# Delta sweep ranges
DELTA_J1    = np.linspace(-3.0, 3.0, 21)   # centroid-pair: both directions
DELTA_OUT   = np.linspace(0.0,  5.0, 11)   # competitor-only: outward only

# Pre-registered pass thresholds
THRESH_J1_R       = 0.90
THRESH_J2_R       = 0.20
THRESH_JK_R       = 0.20
THRESH_TOP_ALL_R  = 0.90
THRESH_R2_POOLED  = 0.85
THRESH_EPSILON    = 0.10
K_MINUS_1         = K - 1
ALPHA_RATIO_LO    = 0.85 * K_MINUS_1   # 11.05 for K=14
ALPHA_RATIO_HI    = 1.15 * K_MINUS_1   # 14.95 for K=14

N_CV_SPLITS = 5


# ================================================================
# GEOMETRY HELPERS
# ================================================================

def compute_class_stats(X, y):
    """Return centroids dict and per-dim pooled within-class std."""
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        centroids[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R**2)))   # per-dim std
    return centroids, sigma_W


def get_competitor_ranking(centroids, sigma_W, d, ci):
    """Return sorted list of (kappa_ij, j) ascending (j1 = nearest)."""
    mu_i = centroids[ci]
    ranking = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa = dist / (sigma_W * np.sqrt(d) + 1e-10)
        ranking.append((kappa, cj, dist))
    ranking.sort(key=lambda x: x[0])
    return ranking   # ascending: index 0 = nearest (j1)


def compute_per_class_q(X, y, ci, n_splits=N_CV_SPLITS):
    """Stratified k-fold CV, return normalised per-class recall."""
    n_c = (y == ci).sum()
    if n_c < n_splits:
        return None
    K_local = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    recalls = []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if (y_tr == ci).sum() < 1:
            continue
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X_tr, y_tr)
        mask = (y_te == ci)
        if mask.sum() == 0:
            continue
        preds = knn.predict(X_te[mask])
        recalls.append(float((preds == ci).mean()))
    if not recalls:
        return None
    q_raw = float(np.mean(recalls))
    return float((q_raw - 1.0/K_local) / (1.0 - 1.0/K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1-1e-5))
    return float(np.log(q / (1.0 - q)))


# ================================================================
# SURGERY FUNCTIONS
# ================================================================

def apply_j1_pair_shift(X, y, centroids, ci, j1, delta):
    """Centroid-pair shift: ci and j1 both move by +/-delta/2 apart."""
    mu_i, mu_j = centroids[ci], centroids[j1]
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == ci] -= (delta / 2) * direction
    X_new[y == j1] += (delta / 2) * direction
    return X_new


def apply_competitor_shift(X, y, centroids, ci, cj, delta):
    """Move ONLY competitor cj outward from ci by delta (ci FIXED)."""
    mu_i, mu_j = centroids[ci], centroids[cj]
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] += delta * direction
    return X_new


def apply_top_m_shift(X, y, centroids, ci, top_m_list, delta):
    """Move ALL competitors in top_m_list outward from ci by delta each (ci FIXED)."""
    X_new = X.copy()
    mu_i = centroids[ci]
    for cj in top_m_list:
        mu_j = centroids[cj]
        diff = mu_j - mu_i
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            continue
        direction = diff / dist
        X_new[y == cj] += delta * direction
    return X_new


# ================================================================
# ARM SWEEP
# ================================================================

def run_sweep(X, y, centroids, sigma_W, d, ci, delta_range,
              surgery_fn, surgery_kwargs, arm_name, log_fn,
              baseline_kappa_j1, baseline_logit_q, moved_competitors):
    """
    Sweep delta for one arm. Returns list of records.
    surgery_fn(X, y, centroids, ci, **surgery_kwargs, delta=delta) -> X_new
    moved_competitors: list of class labels that are moved (for total_delta_kappa)
    """
    # Pre-compute original kappa for each moved competitor
    orig_rank = get_competitor_ranking(centroids, sigma_W, d, ci)
    orig_kappa = {cj: k for k, cj, _ in orig_rank}

    records = []
    for delta in delta_range:
        X_new = surgery_fn(X, y, centroids, ci, delta=delta, **surgery_kwargs)
        new_cents, new_sw = compute_class_stats(X_new, y)
        new_rank = get_competitor_ranking(new_cents, new_sw, d, ci)
        new_kappa = {cj: k for k, cj, _ in new_rank}

        kappa_j1_new = float(new_rank[0][0])

        total_kappa_change = sum(
            new_kappa.get(cj, 0.0) - orig_kappa.get(cj, 0.0)
            for cj in moved_competitors
        )

        q_ci = compute_per_class_q(X_new, y, ci)
        if q_ci is None:
            continue
        lq = safe_logit(q_ci)

        rec = {
            "delta": float(delta),
            "kappa_j1_new": kappa_j1_new,
            "delta_kappa_j1": kappa_j1_new - baseline_kappa_j1,
            "total_delta_kappa": total_kappa_change,
            "q_ci": float(q_ci),
            "logit_q_ci": float(lq),
            "delta_logit_q": float(lq - baseline_logit_q),
        }
        records.append(rec)
        log_fn(f"    [{arm_name} ci={ci}] delta={delta:+.2f}: "
               f"kappa_j1={kappa_j1_new:.3f} (d={rec['delta_kappa_j1']:+.3f}), "
               f"q={q_ci:.4f}")

    return records


def fit_arm(records, x_key, y_key="delta_logit_q"):
    """Fit Pearson r and free slope for (x_key, y_key) from records.
    Returns (r=0, p=1, alpha=0, C=0) when input is constant or too few points.
    """
    pairs = [(r[x_key], r[y_key]) for r in records
             if r.get(x_key) is not None and r.get(y_key) is not None]
    if len(pairs) < 4:
        return 0.0, 1.0, 0.0, 0.0
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 4 or np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
        # Constant input: r is undefined, interpret as r=0 (no correlation)
        return 0.0, 1.0, 0.0, float(np.mean(ys)) if len(ys) > 0 else 0.0
    r, p = stats.pearsonr(xs, ys)
    coeffs = np.polyfit(xs, ys, 1)
    return float(r), float(p), float(coeffs[0]), float(coeffs[1])


def fixed_slope_r2(xs, ys, slope):
    """R2 for y = slope*x (no intercept correction, just fixed slope fit)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 3:
        return 0.0
    y_pred = slope * xs
    ss_res = float(np.sum((ys - y_pred)**2))
    ss_tot = float(np.sum((ys - ys.mean())**2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def fisher_z_mean(rs):
    arr = np.array(rs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    zs = np.arctanh(np.clip(arr, -0.9999, 0.9999))
    return float(np.tanh(np.mean(zs)))


# ================================================================
# MAIN
# ================================================================

def main():
    os.makedirs(str(REPO_ROOT / "results"), exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    log("=" * 70)
    log("CAUSAL SUFFICIENCY RCT: kappa_nearest is causally sufficient")
    log("=" * 70)
    log(f"Embeddings: {EMBS_FILE.name}")
    log(f"alpha_arch={ALPHA_ARCH}, A_renorm={A_RENORM}, K={K}")
    log(f"Pass criteria:")
    log(f"  j1_r > {THRESH_J1_R}, j2_r < {THRESH_J2_R}, jK_r < {THRESH_JK_R}")
    log(f"  top_all_r > {THRESH_TOP_ALL_R}")
    log(f"  fixed_slope_R2 >= {THRESH_R2_POOLED}")
    log(f"  epsilon_invariance within +/-{THRESH_EPSILON}")
    log(f"  alpha_topall/alpha_j1 in [{ALPHA_RATIO_LO:.1f}, {ALPHA_RATIO_HI:.1f}]")
    log("=" * 70)

    # Load frozen embeddings
    data = np.load(str(EMBS_FILE))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    log(f"\nLoaded: N={X.shape[0]}, d={d}, K={len(classes)}")

    centroids, sigma_W = compute_class_stats(X, y)
    log(f"Baseline sigma_W={sigma_W:.4f}")

    # Per-class baseline
    log("\n[BASELINE]")
    baseline_q = {}
    baseline_logit = {}
    baseline_kappa_j1 = {}
    for ci in classes:
        ranking = get_competitor_ranking(centroids, sigma_W, d, ci)
        k1, j1, _ = ranking[0]
        k2, j2, _ = ranking[1]
        jK_cls = ranking[-1][1]
        log(f"  class {ci}: j1={j1} kappa={k1:.3f}, j2={j2} kappa={k2:.3f}, "
            f"jK={jK_cls} kappa={ranking[-1][0]:.3f}, margin={k2/k1:.2f}x")
        q = compute_per_class_q(X, y, ci)
        if q is None:
            continue
        baseline_q[ci] = q
        baseline_logit[ci] = safe_logit(q)
        baseline_kappa_j1[ci] = float(k1)
        log(f"    q_ci={q:.4f}, logit_q={baseline_logit[ci]:.4f}")

    # ----------------------------------------------------------------
    # RUN ARMS
    # ----------------------------------------------------------------
    all_results = {}

    ARM_NAMES = ["j1_shift", "j2_shift", "jK_shift", "top2_shift", "top4_shift", "top_all_shift"]

    for ci in classes:
        if ci not in baseline_q:
            continue
        ranking = get_competitor_ranking(centroids, sigma_W, d, ci)
        j1 = ranking[0][1]
        j2 = ranking[1][1]
        j4_list = [ranking[r][1] for r in range(min(4, len(ranking)))]
        jK = ranking[-1][1]
        all_competitors = [ranking[r][1] for r in range(len(ranking))]

        log(f"\n{'='*60}")
        log(f"FOCUS CLASS {ci} | j1={j1} j2={j2} jK={jK}")
        log(f"{'='*60}")

        ci_result = {
            "class": int(ci),
            "j1": int(j1), "j2": int(j2), "jK": int(jK),
            "baseline_kappa_j1": baseline_kappa_j1[ci],
            "baseline_logit_q": baseline_logit[ci],
        }

        bk1 = baseline_kappa_j1[ci]
        blq = baseline_logit[ci]

        # ARM j1_shift
        log(f"\n  [ARM j1_shift] centroid-pair shift ci={ci} <-> j1={j1}")
        recs_j1 = run_sweep(
            X, y, centroids, sigma_W, d, ci,
            DELTA_J1, apply_j1_pair_shift, {"j1": j1}, "j1_shift", log, bk1, blq,
            moved_competitors=[j1]
        )
        r_j1, p_j1, alpha_j1, _ = fit_arm(recs_j1, "delta_kappa_j1")
        log(f"  j1_shift: r={r_j1:.3f} p={p_j1:.4f}, alpha_j1={alpha_j1:.4f}")

        # ARM j2_shift (orthogonal control)
        log(f"\n  [ARM j2_shift] ONLY j2={j2} moves (ci={ci} and j1={j1} FIXED)")
        recs_j2 = run_sweep(
            X, y, centroids, sigma_W, d, ci,
            DELTA_OUT, apply_competitor_shift, {"cj": j2}, "j2_shift", log, bk1, blq,
            moved_competitors=[j2]
        )
        r_j2, p_j2, alpha_j2, _ = fit_arm(recs_j2, "total_delta_kappa")
        r_j2_kj1, _, _, _ = fit_arm(recs_j2, "delta_kappa_j1")
        log(f"  j2_shift: r(total_kappa_j2, logit)={r_j2:.3f}, r(kappa_j1, logit)={r_j2_kj1:.3f}")

        # ARM jK_shift (negative control)
        log(f"\n  [ARM jK_shift] ONLY jK={jK} moves (neg control)")
        recs_jK = run_sweep(
            X, y, centroids, sigma_W, d, ci,
            DELTA_OUT, apply_competitor_shift, {"cj": jK}, "jK_shift", log, bk1, blq,
            moved_competitors=[jK]
        )
        r_jK, p_jK, alpha_jK, _ = fit_arm(recs_jK, "total_delta_kappa")
        r_jK_kj1, _, _, _ = fit_arm(recs_jK, "delta_kappa_j1")
        log(f"  jK_shift: r(total_kappa, logit)={r_jK:.3f}, r(kappa_j1, logit)={r_jK_kj1:.3f}")

        # ARM top2_shift
        log(f"\n  [ARM top2_shift] j1+j2 both move outward (ci FIXED)")
        recs_top2 = run_sweep(
            X, y, centroids, sigma_W, d, ci,
            DELTA_OUT, apply_top_m_shift, {"top_m_list": [j1, j2]}, "top2_shift", log, bk1, blq,
            moved_competitors=[j1, j2]
        )
        r_top2, p_top2, alpha_top2, _ = fit_arm(recs_top2, "delta_kappa_j1")
        log(f"  top2_shift: r(kappa_j1, logit)={r_top2:.3f}, alpha_top2={alpha_top2:.4f}")

        # ARM top4_shift
        log(f"\n  [ARM top4_shift] top-4 competitors move outward (ci FIXED)")
        recs_top4 = run_sweep(
            X, y, centroids, sigma_W, d, ci,
            DELTA_OUT, apply_top_m_shift, {"top_m_list": j4_list}, "top4_shift", log, bk1, blq,
            moved_competitors=list(j4_list)
        )
        r_top4, p_top4, alpha_top4, _ = fit_arm(recs_top4, "delta_kappa_j1")
        log(f"  top4_shift: r(kappa_j1, logit)={r_top4:.3f}, alpha_top4={alpha_top4:.4f}")

        # ARM top_all_shift
        log(f"\n  [ARM top_all_shift] ALL {K_MINUS_1} competitors move outward (ci FIXED)")
        recs_all = run_sweep(
            X, y, centroids, sigma_W, d, ci,
            DELTA_OUT, apply_top_m_shift, {"top_m_list": all_competitors}, "top_all_shift", log, bk1, blq,
            moved_competitors=list(all_competitors)
        )
        r_all, p_all, alpha_topall, _ = fit_arm(recs_all, "delta_kappa_j1")
        log(f"  top_all_shift: r(kappa_j1, logit)={r_all:.3f}, alpha_topall={alpha_topall:.4f}")

        # Epsilon invariance (per j1/j2/jK arms):
        # epsilon = delta_logit - alpha_arch * delta_kappa (using the moved kappa as x)
        def mean_epsilon(recs, x_key="delta_kappa_j1"):
            eps = []
            for r in recs:
                x = r.get(x_key)
                y = r.get("delta_logit_q")
                if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                    eps.append(y - ALPHA_ARCH * x)
            return float(np.mean(eps)) if eps else None

        eps_j1 = mean_epsilon(recs_j1)
        eps_j2 = mean_epsilon(recs_j2, "total_delta_kappa")
        eps_jK = mean_epsilon(recs_jK, "total_delta_kappa")

        # Alpha ratio test
        alpha_ratio = (alpha_topall / alpha_j1) if abs(alpha_j1) > 0.01 else None

        log(f"\n  --- Class {ci} Summary ---")
        log(f"  alpha_j1={alpha_j1:.4f}, alpha_topall={alpha_topall:.4f}, "
            f"ratio={alpha_ratio:.2f}" if alpha_ratio else "  alpha_ratio=N/A")
        log(f"  Epsilon: j1={eps_j1:.3f}, j2={eps_j2:.3f}, jK={eps_jK:.3f}"
            if None not in [eps_j1, eps_j2, eps_jK] else "  Epsilon: partial")

        # Fixed-slope R2 pooled (j1+top2+top4+top_all) using total_delta_kappa as x
        pool_x = []
        pool_y = []
        for arm_recs in [recs_j1, recs_top2, recs_top4, recs_all]:
            for r in arm_recs:
                if "total_delta_kappa" in r and np.isfinite(r["delta_logit_q"]):
                    pool_x.append(r["total_delta_kappa"])
                    pool_y.append(r["delta_logit_q"])
        r2_fixed = fixed_slope_r2(pool_x, pool_y, ALPHA_ARCH)
        log(f"  Fixed-slope R2 (alpha_arch={ALPHA_ARCH}): {r2_fixed:.4f}")

        ci_result.update({
            "arm_j1": {"r": r_j1, "p": p_j1, "alpha": alpha_j1,
                       "n_pts": len(recs_j1)},
            "arm_j2": {"r": r_j2, "p": p_j2, "alpha": alpha_j2,
                       "r_kappa_j1": r_j2_kj1, "n_pts": len(recs_j2)},
            "arm_jK": {"r": r_jK, "p": p_jK, "alpha": alpha_jK,
                       "r_kappa_j1": r_jK_kj1, "n_pts": len(recs_jK)},
            "arm_top2": {"r": r_top2, "p": p_top2, "alpha": alpha_top2,
                         "n_pts": len(recs_top2)},
            "arm_top4": {"r": r_top4, "p": p_top4, "alpha": alpha_top4,
                         "n_pts": len(recs_top4)},
            "arm_top_all": {"r": r_all, "p": p_all, "alpha": alpha_topall,
                            "n_pts": len(recs_all)},
            "alpha_ratio_topall_j1": float(alpha_ratio) if alpha_ratio else None,
            "epsilon_j1": float(eps_j1) if eps_j1 is not None else None,
            "epsilon_j2": float(eps_j2) if eps_j2 is not None else None,
            "epsilon_jK": float(eps_jK) if eps_jK is not None else None,
            "fixed_slope_r2": float(r2_fixed),
        })
        all_results[str(ci)] = ci_result

    # ----------------------------------------------------------------
    # AGGREGATE
    # ----------------------------------------------------------------
    log("\n" + "="*70)
    log("AGGREGATE RESULTS (Fisher z-mean across focus classes)")
    log("="*70)

    def agg(key, sub):
        vals = [all_results[str(ci)][key][sub]
                for ci in classes if str(ci) in all_results
                and all_results[str(ci)][key].get(sub) is not None]
        return vals

    rs_j1   = agg("arm_j1", "r")
    rs_j2   = agg("arm_j2", "r")       # r(total_kappa_j2, logit)
    rs_jK   = agg("arm_jK", "r")
    rs_all  = agg("arm_top_all", "r")
    rs_j2_kj1 = agg("arm_j2", "r_kappa_j1")
    rs_jK_kj1 = agg("arm_jK", "r_kappa_j1")

    alphas_j1    = agg("arm_j1", "alpha")
    alphas_top2  = agg("arm_top2", "alpha")
    alphas_top4  = agg("arm_top4", "alpha")
    alphas_all   = agg("arm_top_all", "alpha")

    eps_j1_vals  = [all_results[str(ci)]["epsilon_j1"]
                    for ci in classes if str(ci) in all_results
                    and all_results[str(ci)]["epsilon_j1"] is not None]
    eps_j2_vals  = [all_results[str(ci)]["epsilon_j2"]
                    for ci in classes if str(ci) in all_results
                    and all_results[str(ci)]["epsilon_j2"] is not None]
    eps_jK_vals  = [all_results[str(ci)]["epsilon_jK"]
                    for ci in classes if str(ci) in all_results
                    and all_results[str(ci)]["epsilon_jK"] is not None]
    r2_vals      = [all_results[str(ci)]["fixed_slope_r2"]
                    for ci in classes if str(ci) in all_results]
    ratio_vals   = [all_results[str(ci)]["alpha_ratio_topall_j1"]
                    for ci in classes if str(ci) in all_results
                    and all_results[str(ci)]["alpha_ratio_topall_j1"] is not None]

    mean_r_j1   = fisher_z_mean(rs_j1)   if rs_j1   else 0.0
    mean_r_j2   = fisher_z_mean(rs_j2)   if rs_j2   else 0.0
    mean_r_jK   = fisher_z_mean(rs_jK)   if rs_jK   else 0.0
    mean_r_all  = fisher_z_mean(rs_all)  if rs_all  else 0.0
    mean_r_j2_kj1 = fisher_z_mean(rs_j2_kj1) if rs_j2_kj1 else 0.0
    mean_r_jK_kj1 = fisher_z_mean(rs_jK_kj1) if rs_jK_kj1 else 0.0

    mean_alpha_j1   = float(np.mean(alphas_j1))   if alphas_j1   else None
    mean_alpha_top2 = float(np.mean(alphas_top2)) if alphas_top2 else None
    mean_alpha_top4 = float(np.mean(alphas_top4)) if alphas_top4 else None
    mean_alpha_all  = float(np.mean(alphas_all))  if alphas_all  else None

    mean_eps_j1  = float(np.mean(eps_j1_vals))  if eps_j1_vals  else None
    mean_eps_j2  = float(np.mean(eps_j2_vals))  if eps_j2_vals  else None
    mean_eps_jK  = float(np.mean(eps_jK_vals))  if eps_jK_vals  else None
    mean_r2      = float(np.mean(r2_vals))       if r2_vals      else None
    mean_ratio   = float(np.mean(ratio_vals))    if ratio_vals   else None

    log(f"\nArm j1_shift (n={len(rs_j1)} classes):")
    log(f"  r(kappa_j1, logit_q) = {mean_r_j1:.3f}  [threshold > {THRESH_J1_R}]")
    log(f"  alpha_j1 = {mean_alpha_j1:.4f}" if mean_alpha_j1 is not None else "  alpha_j1 = N/A")

    log(f"\nArm j2_shift (orthogonal, n={len(rs_j2)} classes):")
    log(f"  r(total_kappa, logit_q) = {mean_r_j2:.3f}  [threshold < {THRESH_J2_R}]")
    log(f"  r(kappa_j1, logit_q) = {mean_r_j2_kj1:.3f}  [should be ~0: j1 unchanged]")

    log(f"\nArm jK_shift (neg control, n={len(rs_jK)} classes):")
    log(f"  r(total_kappa, logit_q) = {mean_r_jK:.3f}  [threshold < {THRESH_JK_R}]")
    log(f"  r(kappa_j1, logit_q) = {mean_r_jK_kj1:.3f}")

    log(f"\nArm top_all_shift (n={len(rs_all)} classes):")
    log(f"  r(kappa_j1, logit_q) = {mean_r_all:.3f}  [threshold > {THRESH_TOP_ALL_R}]")
    log(f"  alpha_topall = {mean_alpha_all:.4f}" if mean_alpha_all is not None else "  alpha_topall = N/A")

    log(f"\nAlpha scaling (equal-additive model):")
    log(f"  alpha(m=1) = {mean_alpha_j1:.4f}" if mean_alpha_j1 else "  N/A")
    log(f"  alpha(m=2) = {mean_alpha_top2:.4f}" if mean_alpha_top2 else "  N/A")
    log(f"  alpha(m=4) = {mean_alpha_top4:.4f}" if mean_alpha_top4 else "  N/A")
    log(f"  alpha(m=K-1) = {mean_alpha_all:.4f}" if mean_alpha_all else "  N/A")
    log(f"  ratio topall/j1 = {mean_ratio:.2f}  "
        f"[threshold [{ALPHA_RATIO_LO:.1f}, {ALPHA_RATIO_HI:.1f}]]"
        if mean_ratio is not None else "  ratio = N/A")

    log(f"\nFixed-slope R2 (alpha={ALPHA_ARCH}, x=total_delta_kappa):")
    log(f"  Mean R2 = {mean_r2:.4f}  [threshold >= {THRESH_R2_POOLED}]"
        if mean_r2 is not None else "  R2 = N/A")

    log(f"\nEpsilon invariance (mean residual):")
    log(f"  arm j1:  {mean_eps_j1:+.4f}  [threshold +/-{THRESH_EPSILON}]"
        if mean_eps_j1 is not None else "  arm j1: N/A")
    log(f"  arm j2:  {mean_eps_j2:+.4f}"
        if mean_eps_j2 is not None else "  arm j2: N/A")
    log(f"  arm jK:  {mean_eps_jK:+.4f}"
        if mean_eps_jK is not None else "  arm jK: N/A")

    # ----------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------
    pass_j1  = mean_r_j1  >= THRESH_J1_R
    pass_j2  = abs(mean_r_j2)  < THRESH_J2_R
    pass_jK  = abs(mean_r_jK)  < THRESH_JK_R
    pass_all = mean_r_all >= THRESH_TOP_ALL_R
    pass_r2  = (mean_r2 is not None) and (mean_r2 >= THRESH_R2_POOLED)
    pass_eps_j1 = (mean_eps_j1 is not None) and (abs(mean_eps_j1) < THRESH_EPSILON)
    pass_eps_j2 = (mean_eps_j2 is not None) and (abs(mean_eps_j2) < THRESH_EPSILON)
    pass_eps_jK = (mean_eps_jK is not None) and (abs(mean_eps_jK) < THRESH_EPSILON)
    pass_eps = pass_eps_j1 and pass_eps_j2 and pass_eps_jK
    pass_ratio = (mean_ratio is not None) and (ALPHA_RATIO_LO <= mean_ratio <= ALPHA_RATIO_HI)

    log("\n" + "="*70)
    log("VERDICT")
    log("="*70)
    log(f"  arm_j1_r:         {'PASS' if pass_j1  else 'FAIL'} (r={mean_r_j1:.3f})")
    log(f"  arm_j2_r:         {'PASS' if pass_j2  else 'FAIL'} (r={mean_r_j2:.3f})")
    log(f"  arm_jK_r:         {'PASS' if pass_jK  else 'FAIL'} (r={mean_r_jK:.3f})")
    log(f"  arm_top_all_r:    {'PASS' if pass_all else 'FAIL'} (r={mean_r_all:.3f})")
    log(f"  fixed_slope_R2:   {'PASS' if pass_r2  else 'FAIL'} (R2={mean_r2:.3f})")
    log(f"  epsilon_invariance: {'PASS' if pass_eps else 'FAIL'} "
        f"(j1={mean_eps_j1:.3f}, j2={mean_eps_j2:.3f}, jK={mean_eps_jK:.3f})"
        if None not in [mean_eps_j1, mean_eps_j2, mean_eps_jK] else
        "  epsilon_invariance: N/A")
    log(f"  alpha_ratio:      {'PASS' if pass_ratio else 'FAIL'} (ratio={mean_ratio:.2f})"
        if mean_ratio is not None else "  alpha_ratio: N/A")

    n_pass = sum([pass_j1, pass_j2, pass_jK, pass_all, pass_r2, pass_eps, pass_ratio])
    primary_pass = n_pass >= 5
    log(f"\n  OVERALL: {n_pass}/7 criteria pass. {'PRIMARY PASS' if primary_pass else 'PRIMARY FAIL'}")

    # ----------------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------------
    def np_clean(obj):
        if isinstance(obj, dict):
            return {k: np_clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [np_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return obj
        return obj

    out = {
        "experiment": "causal_sufficiency_rct_frozen_topm",
        "prereg_file": "results/cti_causal_sufficiency_rct_prereg.json",
        "model": "pythia-160m",
        "layer": 12,
        "dataset": "dbpedia14",
        "K": K,
        "alpha_arch": ALPHA_ARCH,
        "A_renorm": A_RENORM,
        "n_classes": len(all_results),
        "aggregate": {
            "arm_j1_r": mean_r_j1,
            "arm_j2_r": mean_r_j2,
            "arm_j2_r_kappa_j1_unchanged": mean_r_j2_kj1,
            "arm_jK_r": mean_r_jK,
            "arm_top_all_r": mean_r_all,
            "alpha_j1": mean_alpha_j1,
            "alpha_top2": mean_alpha_top2,
            "alpha_top4": mean_alpha_top4,
            "alpha_topall": mean_alpha_all,
            "alpha_ratio_topall_j1": mean_ratio,
            "fixed_slope_r2": mean_r2,
            "epsilon_j1": mean_eps_j1,
            "epsilon_j2": mean_eps_j2,
            "epsilon_jK": mean_eps_jK,
        },
        "verdict": {
            "arm_j1_r": bool(pass_j1),
            "arm_j2_r": bool(pass_j2),
            "arm_jK_r": bool(pass_jK),
            "arm_top_all_r": bool(pass_all),
            "fixed_slope_r2": bool(pass_r2),
            "epsilon_invariance": bool(pass_eps),
            "alpha_ratio": bool(pass_ratio),
            "n_pass": n_pass,
            "primary_pass": bool(primary_pass),
        },
        "per_class": all_results,
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(np_clean(out), f, indent=2)
    log(f"\nSaved to {OUT_JSON.name}")
    log(f"PRIMARY: {'PASS' if primary_pass else 'FAIL'} ({n_pass}/7)")


if __name__ == "__main__":
    main()
