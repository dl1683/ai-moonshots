#!/usr/bin/env python -u
"""
Weight-Map Transfer Test: No-Refit Held-Out Prediction
=======================================================
Session 72 — Codex priority #1 after competitor weight map.

DESIGN (Codex recommendation: highest Nobel/Turing leverage):
  Source: pythia-160m / DBpedia / L12 (weight map already fitted)
  Held-out: pythia-410m / L3, bert-base-uncased / L10

For each held-out arch:
  1. Run j1_only arm (ci FIXED, move only j1 outward) -> alpha_j1_only_held
     This is the ONE calibration parameter.
  2. Run top2, top4, topall arms -> alpha_topm_held (ground truth)
  3. Predict: alpha_topm_pred = alpha_j1_only_held * sum_{r=1}^{m} w_r_true
     where w_r_true = alpha_r_source / alpha_j1_only_source (from pythia-160m weight map)
  4. Test H1: |alpha_topm_pred - alpha_topm_held| / alpha_topm_held < 0.30 for all m
  5. Test H2: Pearson r(alpha_topm_pred, alpha_topm_held) > 0.90 over {top2, top4, topall}

PRE-REGISTERED HYPOTHESES:
  H1: Relative prediction error < 30% for top2, top4, topall on BOTH held-out archs
      Rationale: if w_r shape is universal, calibrating by j1_only should transfer
  H2: Pearson r(predicted, actual) > 0.90 across 2 archs x 3 topm = 6 datapoints
  H3: alpha_j1_only_held is within [0.7, 1.9] for both archs
      Rationale: 1.272 +/- 50% band (universality check on j1_only itself)

NULL: All effects are equally weighted (uniform competition) predicts
      alpha_topm_pred_null = alpha_j1_only_held * m
      — i.e., no decay structure. We expect our transfer to outperform this null.

KEY PREDICTION FROM SOURCE w_r (normalized by alpha_j1_only_source=1.272):
  w_r_true = [1.000, 0.294, 0.214, 0.171, 0.088, 0.084, ...]
  sum_top2 = 1.294
  sum_top4 = 1.679
  sum_topall = 2.005

OUTPUT: results/cti_weight_map_transfer.json
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

HELD_OUT_MODELS = [
    {
        "name": "pythia-410m",
        "npz": "dointerv_multi_pythia-410m_l3.npz",
    },
    {
        "name": "bert-base-uncased",
        "npz": "dointerv_multi_bert-base-uncased_l10.npz",
    },
]

SOURCE_WEIGHT_MAP_JSON = REPO_ROOT / "results" / "cti_competitor_weight_map.json"
OUT_JSON = REPO_ROOT / "results" / "cti_weight_map_transfer.json"

K = 14
DELTA_OUT = np.linspace(0.0, 5.0, 11)
N_CV_SPLITS = 5

# Pre-registered thresholds
H1_REL_ERROR_MAX = 0.30
H2_R_MIN = 0.90
H3_J1_RANGE = (0.70, 1.90)


# ================================================================
# LOAD SOURCE WEIGHT MAP
# ================================================================
def load_source_weights():
    with open(str(SOURCE_WEIGHT_MAP_JSON)) as f:
        wm = json.load(f)
    alpha_j1_only_source = wm["aggregate_summary"]["mean_alpha_j1_only"]
    ranks = wm["aggregate_by_rank"]
    # w_r_true = alpha_r_source / alpha_j1_only_source
    w_r_true = [r["mean_alpha"] / alpha_j1_only_source for r in ranks]
    return {
        "alpha_j1_only_source": alpha_j1_only_source,
        "w_r_true": w_r_true,
        "sum_top2": sum(w_r_true[:2]),
        "sum_top4": sum(w_r_true[:4]),
        "sum_topall": sum(w_r_true),
        "K_minus_1": len(w_r_true),
    }


# ================================================================
# GEOMETRY + Q HELPERS (same as cti_competitor_weight_map.py)
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
    return float((q_raw - 1.0 / K_local) / (1.0 - 1.0 / K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1 - 1e-5))
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


def apply_top_m_shift(X, y, centroids, ci, top_m_list, delta):
    """Move top-m competitors outward from ci by delta each (ci FIXED)."""
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


def fit_slope_r(records, x_key, y_key="delta_logit_q"):
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


# ================================================================
# RUN ARMS FOR ONE ARCH
# ================================================================
def run_arch(npz_path, arch_name, source_weights, log):
    data = np.load(str(npz_path))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    log(f"\nLoaded {arch_name}: N={X.shape[0]}, d={d}, K={len(classes)}")

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

    per_class_results = {}
    for ci in classes:
        if ci not in baseline_q:
            continue
        ranking = get_competitor_ranking(centroids, sigma_W, d, ci)
        j1 = ranking[0][1]
        j2 = ranking[1][1]
        j4_list = [ranking[r][1] for r in range(min(4, len(ranking)))]
        all_comps = [ranking[r][1] for r in range(len(ranking))]

        bk1 = float(ranking[0][0])
        blq = baseline_logit[ci]

        log(f"\n  CLASS {ci} | q={baseline_q[ci]:.4f}, kappa_j1={bk1:.3f}")

        # --- ARM 1: j1_only (ci FIXED, move only j1) ---
        recs_j1 = []
        for delta in DELTA_OUT:
            X_new = apply_competitor_shift(X, y, centroids, ci, j1, delta)
            new_cents, new_sw = compute_class_stats(X_new, y)
            new_rank = get_competitor_ranking(new_cents, new_sw, d, ci)
            kappa_j1_new = float(new_rank[0][0])
            q_ci = compute_per_class_q(X_new, y, ci)
            if q_ci is None:
                continue
            lq = safe_logit(q_ci)
            recs_j1.append({
                "delta": float(delta),
                "delta_kappa_j1": kappa_j1_new - bk1,
                "delta_logit_q": float(lq - blq),
            })
        r_j1, p_j1, alpha_j1_only, _ = fit_slope_r(recs_j1, "delta_kappa_j1")
        log(f"    j1_only: r={r_j1:.3f}, alpha={alpha_j1_only:.4f}")

        # --- ARM 2: top2_shift ---
        recs_t2 = []
        for delta in DELTA_OUT:
            X_new = apply_top_m_shift(X, y, centroids, ci, [j1, j2], delta)
            new_cents, new_sw = compute_class_stats(X_new, y)
            new_rank = get_competitor_ranking(new_cents, new_sw, d, ci)
            kappa_j1_new = float(new_rank[0][0])
            q_ci = compute_per_class_q(X_new, y, ci)
            if q_ci is None:
                continue
            lq = safe_logit(q_ci)
            recs_t2.append({
                "delta": float(delta),
                "delta_kappa_j1": kappa_j1_new - bk1,
                "delta_logit_q": float(lq - blq),
            })
        _, _, alpha_top2, _ = fit_slope_r(recs_t2, "delta_kappa_j1")

        # --- ARM 3: top4_shift ---
        recs_t4 = []
        for delta in DELTA_OUT:
            X_new = apply_top_m_shift(X, y, centroids, ci, j4_list, delta)
            new_cents, new_sw = compute_class_stats(X_new, y)
            new_rank = get_competitor_ranking(new_cents, new_sw, d, ci)
            kappa_j1_new = float(new_rank[0][0])
            q_ci = compute_per_class_q(X_new, y, ci)
            if q_ci is None:
                continue
            lq = safe_logit(q_ci)
            recs_t4.append({
                "delta": float(delta),
                "delta_kappa_j1": kappa_j1_new - bk1,
                "delta_logit_q": float(lq - blq),
            })
        _, _, alpha_top4, _ = fit_slope_r(recs_t4, "delta_kappa_j1")

        # --- ARM 4: topall_shift ---
        recs_ta = []
        for delta in DELTA_OUT:
            X_new = apply_top_m_shift(X, y, centroids, ci, all_comps, delta)
            new_cents, new_sw = compute_class_stats(X_new, y)
            new_rank = get_competitor_ranking(new_cents, new_sw, d, ci)
            kappa_j1_new = float(new_rank[0][0])
            q_ci = compute_per_class_q(X_new, y, ci)
            if q_ci is None:
                continue
            lq = safe_logit(q_ci)
            recs_ta.append({
                "delta": float(delta),
                "delta_kappa_j1": kappa_j1_new - bk1,
                "delta_logit_q": float(lq - blq),
            })
        _, _, alpha_topall, _ = fit_slope_r(recs_ta, "delta_kappa_j1")

        log(f"    top2={alpha_top2:.4f}, top4={alpha_top4:.4f}, topall={alpha_topall:.4f}")

        per_class_results[str(ci)] = {
            "class": int(ci),
            "baseline_kappa_j1": bk1,
            "baseline_logit_q": blq,
            "alpha_j1_only": float(alpha_j1_only),
            "alpha_top2": float(alpha_top2),
            "alpha_top4": float(alpha_top4),
            "alpha_topall": float(alpha_topall),
        }

    # --- AGGREGATE ---
    valid_classes = [k for k, v in per_class_results.items() if v["alpha_j1_only"] is not None]
    alpha_j1_only_vals = [per_class_results[k]["alpha_j1_only"] for k in valid_classes]
    alpha_top2_vals = [per_class_results[k]["alpha_top2"] for k in valid_classes]
    alpha_top4_vals = [per_class_results[k]["alpha_top4"] for k in valid_classes]
    alpha_topall_vals = [per_class_results[k]["alpha_topall"] for k in valid_classes]

    mean_alpha_j1_only = float(np.mean(alpha_j1_only_vals))
    mean_alpha_top2 = float(np.mean(alpha_top2_vals))
    mean_alpha_top4 = float(np.mean(alpha_top4_vals))
    mean_alpha_topall = float(np.mean(alpha_topall_vals))

    # --- PREDICTIONS using source w_r ---
    w_r = source_weights["w_r_true"]
    pred_top2 = mean_alpha_j1_only * sum(w_r[:2])
    pred_top4 = mean_alpha_j1_only * sum(w_r[:4])
    pred_topall = mean_alpha_j1_only * sum(w_r)

    # Null: uniform competition (all weights = 1)
    pred_top2_null = mean_alpha_j1_only * 2
    pred_top4_null = mean_alpha_j1_only * 4
    pred_topall_null = mean_alpha_j1_only * len(w_r)

    rel_err_top2 = abs(pred_top2 - mean_alpha_top2) / (abs(mean_alpha_top2) + 1e-8)
    rel_err_top4 = abs(pred_top4 - mean_alpha_top4) / (abs(mean_alpha_top4) + 1e-8)
    rel_err_topall = abs(pred_topall - mean_alpha_topall) / (abs(mean_alpha_topall) + 1e-8)

    rel_err_null_top2 = abs(pred_top2_null - mean_alpha_top2) / (abs(mean_alpha_top2) + 1e-8)
    rel_err_null_top4 = abs(pred_top4_null - mean_alpha_top4) / (abs(mean_alpha_top4) + 1e-8)
    rel_err_null_topall = abs(pred_topall_null - mean_alpha_topall) / (abs(mean_alpha_topall) + 1e-8)

    h1_pass_top2 = bool(rel_err_top2 < H1_REL_ERROR_MAX)
    h1_pass_top4 = bool(rel_err_top4 < H1_REL_ERROR_MAX)
    h1_pass_topall = bool(rel_err_topall < H1_REL_ERROR_MAX)
    h1_pass = h1_pass_top2 and h1_pass_top4 and h1_pass_topall

    h3_pass = bool(H3_J1_RANGE[0] <= mean_alpha_j1_only <= H3_J1_RANGE[1])

    log(f"\n  --- AGGREGATE for {arch_name} ---")
    log(f"  alpha_j1_only: {mean_alpha_j1_only:.4f} (H3 range: {H3_J1_RANGE}): {'PASS' if h3_pass else 'FAIL'}")
    log(f"  Actual:  top2={mean_alpha_top2:.4f}, top4={mean_alpha_top4:.4f}, topall={mean_alpha_topall:.4f}")
    log(f"  Pred:    top2={pred_top2:.4f}, top4={pred_top4:.4f}, topall={pred_topall:.4f}")
    log(f"  RelErr:  top2={rel_err_top2:.3f}, top4={rel_err_top4:.3f}, topall={rel_err_topall:.3f}")
    log(f"  H1 (each <{H1_REL_ERROR_MAX}): top2={'PASS' if h1_pass_top2 else 'FAIL'}, top4={'PASS' if h1_pass_top4 else 'FAIL'}, topall={'PASS' if h1_pass_topall else 'FAIL'}")
    log(f"  H1 overall: {'PASS' if h1_pass else 'FAIL'}")
    log(f"  Null RelErr: top2={rel_err_null_top2:.3f}, top4={rel_err_null_top4:.3f}, topall={rel_err_null_topall:.3f}")

    return {
        "arch_name": arch_name,
        "mean_alpha_j1_only": mean_alpha_j1_only,
        "mean_alpha_top2": mean_alpha_top2,
        "mean_alpha_top4": mean_alpha_top4,
        "mean_alpha_topall": mean_alpha_topall,
        "pred_top2": float(pred_top2),
        "pred_top4": float(pred_top4),
        "pred_topall": float(pred_topall),
        "rel_err_top2": float(rel_err_top2),
        "rel_err_top4": float(rel_err_top4),
        "rel_err_topall": float(rel_err_topall),
        "pred_null_top2": float(pred_top2_null),
        "pred_null_top4": float(pred_top4_null),
        "pred_null_topall": float(pred_topall_null),
        "rel_err_null_top2": float(rel_err_null_top2),
        "rel_err_null_top4": float(rel_err_null_top4),
        "rel_err_null_topall": float(rel_err_null_topall),
        "h1_pass_top2": h1_pass_top2,
        "h1_pass_top4": h1_pass_top4,
        "h1_pass_topall": h1_pass_topall,
        "h1_pass": h1_pass,
        "h3_pass": h3_pass,
        "per_class": per_class_results,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs(str(REPO_ROOT / "results"), exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    log("=" * 70)
    log("WEIGHT-MAP TRANSFER TEST (No-Refit Held-Out Prediction)")
    log("=" * 70)

    source_weights = load_source_weights()
    log(f"\nSource: pythia-160m / DBpedia / L12")
    log(f"  alpha_j1_only_source = {source_weights['alpha_j1_only_source']:.4f}")
    log(f"  w_r_true (top 6): {[round(w, 4) for w in source_weights['w_r_true'][:6]]}")
    log(f"  sum_top2={source_weights['sum_top2']:.4f}, sum_top4={source_weights['sum_top4']:.4f}, sum_topall={source_weights['sum_topall']:.4f}")

    arch_results = []
    for arch_config in HELD_OUT_MODELS:
        npz_path = REPO_ROOT / "results" / arch_config["npz"]
        if not npz_path.exists():
            log(f"\nSKIPPING {arch_config['name']}: {npz_path} not found")
            continue
        log(f"\n{'='*70}")
        log(f"HELD-OUT ARCH: {arch_config['name']}")
        log(f"{'='*70}")
        result = run_arch(npz_path, arch_config["name"], source_weights, log)
        arch_results.append(result)

    # --- GLOBAL H2: Pearson r over all 2*3 = 6 datapoints ---
    all_pred = []
    all_actual = []
    for ar in arch_results:
        all_pred += [ar["pred_top2"], ar["pred_top4"], ar["pred_topall"]]
        all_actual += [ar["mean_alpha_top2"], ar["mean_alpha_top4"], ar["mean_alpha_topall"]]

    if len(all_pred) >= 3:
        r_global, p_global = stats.pearsonr(all_pred, all_actual)
        h2_pass = bool(r_global >= H2_R_MIN)
    else:
        r_global, p_global, h2_pass = 0.0, 1.0, False

    # Per-arch H1 summary
    h1_global = all([ar["h1_pass"] for ar in arch_results]) if arch_results else False

    log(f"\n{'='*70}")
    log("FINAL VERDICT")
    log(f"{'='*70}")
    log(f"H1 (rel_err<30% all archs+topm): {'PASS' if h1_global else 'FAIL'}")
    log(f"H2 (global Pearson r>{H2_R_MIN}): r={r_global:.3f}, p={p_global:.4f} -> {'PASS' if h2_pass else 'FAIL'}")
    for ar in arch_results:
        log(f"  {ar['arch_name']}: H1={'PASS' if ar['h1_pass'] else 'FAIL'}, H3={'PASS' if ar['h3_pass'] else 'FAIL'}")

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
        "experiment": "weight_map_transfer",
        "session": 72,
        "description": "No-refit transfer of pythia-160m w_r profile to held-out archs",
        "source_model": "pythia-160m",
        "source_alpha_j1_only": source_weights["alpha_j1_only_source"],
        "source_w_r_true": source_weights["w_r_true"],
        "source_sums": {
            "sum_top2": source_weights["sum_top2"],
            "sum_top4": source_weights["sum_top4"],
            "sum_topall": source_weights["sum_topall"],
        },
        "prereg_thresholds": {
            "H1_rel_error_max": H1_REL_ERROR_MAX,
            "H2_r_min": H2_R_MIN,
            "H3_j1_range": list(H3_J1_RANGE),
        },
        "arch_results": arch_results,
        "global": {
            "r_pearson_pred_vs_actual": float(r_global),
            "p_pearson": float(p_global),
            "h1_global_pass": h1_global,
            "h2_pass": h2_pass,
        },
        "verdict": {
            "h1_pass": h1_global,
            "h2_pass": h2_pass,
            "n_pass": sum([h1_global, h2_pass]),
            "primary_pass": h1_global and h2_pass,
        },
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(np_clean(out), f, indent=2)
    log(f"\nSaved to {OUT_JSON.name}")


if __name__ == "__main__":
    main()
