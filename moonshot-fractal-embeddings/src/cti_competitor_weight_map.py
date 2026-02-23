#!/usr/bin/env python -u
"""
Single-Competitor Causal Weight Map: w_r for all K-1 competitors
================================================================
Session 38: Codex priority #2 after RCT.

DESIGN:
For each focus class ci and each competitor cj (r=1..K-1):
  Move ONLY cj outward from ci by delta (ci FIXED) -- clean unconfounded intervention
  Measure alpha_cj = slope of (delta_kappa_cj, delta_logit_q_ci)

Then:
  w_r = alpha_cj_ranked_r / alpha_j1
  (w_1 = 1 by definition; w_r gives empirical weight of r-th competitor)

INCLUDES: unconfounded j1_only arm (same design as j2..jK, but for the nearest competitor)
This corrects the RCT v1 confound where j1_shift moved both ci and j1.

KEY PREDICTIONS (from RCT v1 + theory):
  - alpha_j1_only / alpha_j1_pairshift ~ 0.5-0.6 (pairshift confounded by ci moving)
  - alpha_j1_only ~ 1.052 (A_renorm, from pre-registered multi-arch result)
  - w_r ~ exp(-A_local * delta_kappa_r * sqrt(d_eff)) where A_local ~ 1.75
  - sum w_r ~ 4-5 (from RCT: alpha_topall/alpha_j1_pure ~ 4.59)

OUTPUT: results/cti_competitor_weight_map.json
  - Per-class per-competitor: alpha_cj, r_cj, w_cj
  - Aggregate: Fisher z-mean r, mean w per rank
  - Test: w_r ~ exp(-A_local * delta_kappa * sqrt(d_eff))
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
OUT_JSON  = REPO_ROOT / "results" / "cti_competitor_weight_map.json"

K          = 14
ALPHA_J1_PURE = 1.052    # Pre-registered pure j1 effect (Session 37)

DELTA_OUT  = np.linspace(0.0, 5.0, 11)
N_CV_SPLITS = 5


# ================================================================
# GEOMETRY HELPERS (identical to cti_causal_sufficiency_rct.py)
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
    sigma_W = float(np.sqrt(np.mean(R**2)))
    return centroids, sigma_W


def get_competitor_ranking(centroids, sigma_W, d, ci):
    mu_i = centroids[ci]
    ranking = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa = dist / (sigma_W * np.sqrt(d) + 1e-10)
        ranking.append((kappa, cj, dist))
    ranking.sort(key=lambda x: x[0])
    return ranking


def compute_per_class_q(X, y, ci, n_splits=N_CV_SPLITS):
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


def apply_competitor_shift(X, y, centroids, ci, cj, delta):
    """Move ONLY cj outward from ci by delta (ci FIXED)."""
    mu_i, mu_j = centroids[ci], centroids[cj]
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] += delta * direction
    return X_new


def fit_slope_r(records, x_key, y_key="delta_logit_q"):
    """Fit r and free slope. Returns (r, p, alpha, C). r=0 if constant."""
    pairs = [(r[x_key], r[y_key]) for r in records
             if r.get(x_key) is not None and r.get(y_key) is not None]
    if len(pairs) < 4:
        return 0.0, 1.0, 0.0, 0.0
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 4 or np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
        return 0.0, 1.0, 0.0, float(np.mean(ys)) if len(ys) > 0 else 0.0
    r, p = stats.pearsonr(xs, ys)
    coeffs = np.polyfit(xs, ys, 1)
    return float(r), float(p), float(coeffs[0]), float(coeffs[1])


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
    log("SINGLE-COMPETITOR CAUSAL WEIGHT MAP")
    log("=" * 70)
    log(f"Embeddings: {EMBS_FILE.name}")
    log(f"K={K}, delta_out=[{DELTA_OUT[0]:.1f}, {DELTA_OUT[-1]:.1f}], {len(DELTA_OUT)} pts")
    log(f"Moving ONLY competitor (ci FIXED) -- unconfounded measurement")
    log(f"KEY: alpha_j1_only / ALPHA_J1_PURE ({ALPHA_J1_PURE}) should ~ 1.0")
    log("=" * 70)

    data = np.load(str(EMBS_FILE))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    log(f"\nLoaded: N={X.shape[0]}, d={d}, K={len(classes)}")

    centroids, sigma_W = compute_class_stats(X, y)
    log(f"sigma_W={sigma_W:.4f}")

    # Baseline q per class
    baseline_q = {}
    baseline_logit = {}
    for ci in classes:
        q = compute_per_class_q(X, y, ci)
        if q is None:
            continue
        baseline_q[ci] = q
        baseline_logit[ci] = safe_logit(q)

    all_results = {}

    for ci in classes:
        if ci not in baseline_q:
            continue
        ranking = get_competitor_ranking(centroids, sigma_W, d, ci)
        orig_kappa = {cj: k for k, cj, _ in ranking}

        log(f"\n{'='*60}")
        log(f"CLASS {ci} | baseline_q={baseline_q[ci]:.4f} | "
            f"j1={ranking[0][1]} kappa={ranking[0][0]:.3f}")
        log(f"{'='*60}")

        bk1 = float(ranking[0][0])
        blq = baseline_logit[ci]

        ci_result = {
            "class": int(ci),
            "baseline_kappa_j1": bk1,
            "baseline_logit_q": blq,
            "competitors": [],
        }

        for rank_r, (kappa_r, cj, dist_r) in enumerate(ranking):
            log(f"\n  [rank {rank_r+1} cj={cj}] kappa={kappa_r:.3f} (margin vs j1: {kappa_r/bk1:.2f}x)")

            recs = []
            for delta in DELTA_OUT:
                X_new = apply_competitor_shift(X, y, centroids, ci, cj, delta)
                new_cents, new_sw = compute_class_stats(X_new, y)
                new_rank = get_competitor_ranking(new_cents, new_sw, d, ci)
                new_kappa = {cj2: k for k, cj2, _ in new_rank}

                kappa_j1_new = float(new_rank[0][0])
                delta_kappa_cj = new_kappa.get(cj, 0.0) - orig_kappa.get(cj, 0.0)

                q_ci = compute_per_class_q(X_new, y, ci)
                if q_ci is None:
                    continue
                lq = safe_logit(q_ci)

                recs.append({
                    "delta": float(delta),
                    "kappa_j1_new": kappa_j1_new,
                    "delta_kappa_cj": delta_kappa_cj,
                    "delta_kappa_j1": kappa_j1_new - bk1,
                    "q_ci": float(q_ci),
                    "delta_logit_q": float(lq - blq),
                })

            r_cj, p_cj, alpha_cj, _ = fit_slope_r(recs, "delta_kappa_cj")
            r_j1_unchanged, _, _, _ = fit_slope_r(recs, "delta_kappa_j1")
            w_cj = alpha_cj / ALPHA_J1_PURE if abs(ALPHA_J1_PURE) > 0.01 else None

            log(f"  r={r_cj:.3f} p={p_cj:.4f}, alpha={alpha_cj:.4f}, "
                f"w={w_cj:.3f}" if w_cj is not None else
                f"  r={r_cj:.3f}, alpha={alpha_cj:.4f}")
            log(f"  r(kappa_j1_unchanged)={r_j1_unchanged:.3f}  [should be ~0]")

            ci_result["competitors"].append({
                "rank": rank_r + 1,
                "cj": int(cj),
                "kappa_r": float(kappa_r),
                "margin_vs_j1": float(kappa_r / bk1) if bk1 > 0 else None,
                "r": float(r_cj),
                "p": float(p_cj),
                "alpha": float(alpha_cj),
                "w": float(w_cj) if w_cj is not None else None,
                "r_j1_unchanged": float(r_j1_unchanged),
                "n_pts": len(recs),
            })

        # Summary: alpha vs rank
        alphas = [c["alpha"] for c in ci_result["competitors"]]
        ws = [c["w"] for c in ci_result["competitors"] if c["w"] is not None]
        log(f"\n  Class {ci} alpha by rank: " + " | ".join(f"r{i+1}:{a:.3f}" for i, a in enumerate(alphas)))
        log(f"  Sum w_r = {sum(ws):.3f} (number effective competitors)"
            if ws else "  Sum w_r = N/A")

        ci_result["alpha_j1_only"] = float(alphas[0]) if alphas else None
        ci_result["sum_w"] = float(sum(ws)) if ws else None
        all_results[str(ci)] = ci_result

    # ----------------------------------------------------------------
    # AGGREGATE
    # ----------------------------------------------------------------
    log("\n" + "="*70)
    log("AGGREGATE")
    log("="*70)

    # Per-rank aggregation
    max_rank = K - 1
    rank_alphas = {r: [] for r in range(1, max_rank + 1)}
    rank_rs = {r: [] for r in range(1, max_rank + 1)}
    rank_ws = {r: [] for r in range(1, max_rank + 1)}
    rank_margins = {r: [] for r in range(1, max_rank + 1)}

    for ci in classes:
        if str(ci) not in all_results:
            continue
        for comp in all_results[str(ci)]["competitors"]:
            r = comp["rank"]
            rank_alphas[r].append(comp["alpha"])
            rank_rs[r].append(comp["r"])
            if comp["w"] is not None:
                rank_ws[r].append(comp["w"])
            if comp["margin_vs_j1"] is not None:
                rank_margins[r].append(comp["margin_vs_j1"])

    log(f"\n{'rank':>5} {'mean_alpha':>12} {'mean_r':>8} {'mean_w':>8} {'mean_margin':>12}")
    log("-" * 55)

    aggregate_by_rank = []
    for r in range(1, max_rank + 1):
        ma = float(np.mean(rank_alphas[r])) if rank_alphas[r] else None
        mr = fisher_z_mean(rank_rs[r]) if rank_rs[r] else None
        mw = float(np.mean(rank_ws[r])) if rank_ws[r] else None
        mm = float(np.mean(rank_margins[r])) if rank_margins[r] else None
        log(f"  {r:>3}  {ma:>12.4f}  {mr:>8.3f}  {mw:>8.3f}  {mm:>12.3f}"
            if None not in [ma, mr, mw, mm] else f"  {r:>3}  N/A")
        aggregate_by_rank.append({
            "rank": r,
            "mean_alpha": ma,
            "mean_r": mr,
            "mean_w": mw,
            "mean_margin": mm,
        })

    # Key stats
    all_alpha_j1_only = [all_results[str(ci)]["alpha_j1_only"]
                         for ci in classes if str(ci) in all_results
                         and all_results[str(ci)]["alpha_j1_only"] is not None]
    all_sum_w = [all_results[str(ci)]["sum_w"]
                 for ci in classes if str(ci) in all_results
                 and all_results[str(ci)]["sum_w"] is not None]

    mean_alpha_j1_only = float(np.mean(all_alpha_j1_only)) if all_alpha_j1_only else None
    mean_sum_w = float(np.mean(all_sum_w)) if all_sum_w else None

    log(f"\nalpha_j1_only (unconfounded): mean={mean_alpha_j1_only:.4f}"
        if mean_alpha_j1_only else "  alpha_j1_only: N/A")
    log(f"  (pre-registered: should ~ {ALPHA_J1_PURE})")
    log(f"Sum w_r (effective competitors): mean={mean_sum_w:.2f}"
        if mean_sum_w else "  Sum w: N/A")

    # Fit w_r ~ exp(-A_local * delta_kappa * sqrt(d_eff)) model
    # Use rank-margin as proxy for delta_kappa (margin = kappa_r/kappa_j1)
    log(f"\nFitting w_r ~ exp(-A_local * log(margin_r)):")
    margins_fit = []
    ws_fit = []
    for rd in aggregate_by_rank:
        if rd["mean_w"] is not None and rd["mean_margin"] is not None:
            if rd["mean_margin"] > 1.0 and rd["mean_w"] > 0:
                margins_fit.append(np.log(rd["mean_margin"]))
                ws_fit.append(np.log(rd["mean_w"] + 1e-6))
    if len(margins_fit) >= 3:
        coeffs = np.polyfit(margins_fit, ws_fit, 1)
        A_local_fit = -float(coeffs[0])
        r_fit, p_fit = stats.pearsonr(margins_fit, ws_fit)
        log(f"  A_local = {A_local_fit:.3f}, r = {r_fit:.3f} p = {p_fit:.4f}")
        log(f"  (theory: A_local ~ 1.75)")
    else:
        A_local_fit = None
        r_fit = None
        log(f"  Insufficient data for fit")

    # Save
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
        return obj

    out = {
        "experiment": "single_competitor_weight_map",
        "session": 38,
        "model": "pythia-160m",
        "layer": 12,
        "dataset": "dbpedia14",
        "K": K,
        "alpha_j1_pure_prereg": ALPHA_J1_PURE,
        "aggregate_by_rank": aggregate_by_rank,
        "aggregate_summary": {
            "mean_alpha_j1_only": mean_alpha_j1_only,
            "mean_sum_w": mean_sum_w,
            "A_local_fit": A_local_fit,
            "r_log_log_fit": r_fit,
        },
        "per_class": all_results,
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(np_clean(out), f, indent=2)
    log(f"\nSaved to {OUT_JSON.name}")
    log(f"Mean alpha_j1_only = {mean_alpha_j1_only:.4f} (expected ~{ALPHA_J1_PURE})"
        if mean_alpha_j1_only else "")
    log(f"Mean sum_w = {mean_sum_w:.2f} effective competitors"
        if mean_sum_w else "")


if __name__ == "__main__":
    main()
