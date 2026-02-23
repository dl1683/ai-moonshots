#!/usr/bin/env python -u
"""
Held-Out w_r Transfer Test (Codex recommendation, Session 38)
==============================================================
PRE-REGISTERED TEST:
  - Train: w_r weights from pythia-160m + DBpedia14 (cti_competitor_weight_map.json)
  - Held-out: pythia-410m, electra-small, rwkv-4-169m, bert-base (same dataset)
  - Prediction: effective_kappa_i = sum_r w_r * kappa_jr should predict logit(q_i)
    BETTER than kappa_nearest alone (no refitting of w_r)

PRE-REGISTERED CRITERION:
  - Pooled R2(effective_kappa) - R2(kappa_nearest) > 0.05 (absolute)
  - Equivalently, R2 improvement > 10% relative
  - This must hold on HELD-OUT architectures (w_r frozen from train)

WHY THIS MATTERS:
  If w_r weights (learned causally from one architecture) transfer to predict
  accuracy in other architectures, this is the first cross-arch evidence that
  competition structure is universal.

DESIGN:
  1. Load w_r from cti_competitor_weight_map.json (aggregate_by_rank)
  2. For each held-out model's cached embeddings:
     - Compute kappa_jr for all classes and all ranks
     - Compute effective_kappa_i = sum_r w_r * kappa_jr (frozen weights)
     - Measure logit(q_ci) via 5-fold CV (per-class)
     - Regress logit(q_ci) ~ effective_kappa_i and ~ kappa_nearest separately
  3. Compare R2 values across all held-out points (pooled)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import linregress, pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
WEIGHT_MAP_JSON = RESULTS_DIR / "cti_competitor_weight_map.json"
OUT_JSON = RESULTS_DIR / "cti_kappa_eff_held_out.json"

# Training source: pythia-160m (w_r already measured)
TRAIN_MODEL = "pythia-160m"

# Held-out architectures (different from train, same DBpedia14 dataset)
HELD_OUT_EMBS = {
    "pythia-410m":   "results/dointerv_multi_pythia-410m_l3.npz",
    "electra-small": "results/dointerv_multi_electra-small_l3.npz",
    "rwkv-4-169m":   "results/dointerv_multi_rwkv-4-169m_l12.npz",
    "bert-base":     "results/dointerv_multi_bert-base-uncased_l10.npz",
}

# Pre-registered thresholds
R2_ABS_IMPROVEMENT_THRESHOLD = 0.05   # effective_kappa R2 > kappa_nearest R2 + 0.05
R2_REL_IMPROVEMENT_THRESHOLD = 0.10   # 10% relative improvement

ALPHA_J1_PURE = 1.052   # pre-registered (re-used from Session 37)
N_SPLITS_CV = 5


def load_w_r_weights(json_path):
    """Load mean w_r by rank from the weight map JSON."""
    with open(json_path) as f:
        data = json.load(f)
    agg = data["aggregate_by_rank"]  # list of {rank, mean_w, ...}
    # Build dict: rank -> mean_w (rank is 1-indexed)
    w_by_rank = {}
    for entry in agg:
        r = int(entry["rank"])
        w = float(entry["mean_w"])
        w_by_rank[r] = w
    print(f"Loaded w_r for ranks {min(w_by_rank)}-{max(w_by_rank)}")
    print(f"  w_r profile: " + ", ".join(f"r{r}={w_by_rank[r]:.3f}" for r in sorted(w_by_rank)))
    return w_by_rank


def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(0)
        centroids[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R**2)))
    return centroids, sigma_W


def compute_all_kappas_sorted(centroids, sigma_W, d, ci):
    """Return kappas sorted ascending (nearest first)."""
    mu_i = centroids[ci]
    kappas = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappas.append(dist / (sigma_W * np.sqrt(d) + 1e-10))
    kappas.sort()
    return kappas


def compute_per_class_q(X, y, ci, n_splits=N_SPLITS_CV):
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
        mask = y_te == ci
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


def fit_r2(xs, ys):
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 4 or np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
        return 0.0, 0.0, 0.0
    r, _ = pearsonr(xs, ys)
    slope, intercept, _, _, _ = linregress(xs, ys)
    y_pred = slope * xs + intercept
    ss_res = float(np.sum((ys - y_pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(r), float(r2), float(slope)


def main():
    print("=" * 70)
    print("HELD-OUT w_r TRANSFER TEST")
    print(f"Train: {TRAIN_MODEL} + DBpedia14 (w_r frozen)")
    print(f"Held-out: {list(HELD_OUT_EMBS.keys())}")
    print(f"Pre-registered criterion: R2(eff_kappa) - R2(kappa_nearest) > {R2_ABS_IMPROVEMENT_THRESHOLD}")
    print("=" * 70)

    # Load frozen weights
    w_by_rank = load_w_r_weights(WEIGHT_MAP_JSON)
    sum_w = sum(w_by_rank.values())
    print(f"\n  Sum w_r = {sum_w:.3f} effective competitors")
    print(f"  (w[1]={w_by_rank.get(1,0):.3f} is always included as kappa_nearest)")

    all_kappa_nearest = []
    all_kappa_eff = []
    all_logit_q = []

    per_model_results = {}

    for model_name, emb_path in HELD_OUT_EMBS.items():
        path = REPO_ROOT / emb_path
        if not path.exists():
            print(f"\n  MISSING: {emb_path}")
            continue

        data = np.load(str(path))
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64)
        d = X.shape[1]
        classes = sorted(np.unique(y).tolist())
        K = len(classes)

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} (d={d}, K={K}, N={len(X)})")
        print("=" * 60)

        centroids, sigma_W = compute_class_stats(X, y)

        knn_list, keff_list, lq_list = [], [], []

        for ci in classes:
            kappas = compute_all_kappas_sorted(centroids, sigma_W, d, ci)
            if not kappas:
                continue

            # kappa_nearest = rank-1 competitor
            kappa_nearest = kappas[0]

            # effective_kappa = sum_r w_r * kappa_jr (frozen w from train)
            kappa_eff = 0.0
            for rank_1idx, kappa_r in enumerate(kappas):
                r = rank_1idx + 1  # 1-indexed
                w = w_by_rank.get(r, 0.0)
                kappa_eff += w * kappa_r

            # Per-class accuracy via CV
            q = compute_per_class_q(X, y, ci)
            if q is None:
                continue
            lq = safe_logit(q)

            knn_list.append(kappa_nearest)
            keff_list.append(kappa_eff)
            lq_list.append(lq)

            print(f"  ci={ci}: kappa_nearest={kappa_nearest:.4f}, "
                  f"kappa_eff={kappa_eff:.4f}, q={q:.4f}, logit={lq:.4f}")

        if len(knn_list) < 4:
            print(f"  Too few valid classes for {model_name}")
            continue

        r_knn, r2_knn, slope_knn = fit_r2(knn_list, lq_list)
        r_eff, r2_eff, slope_eff = fit_r2(keff_list, lq_list)
        r2_improvement = r2_eff - r2_knn
        r2_rel_improvement = r2_improvement / abs(r2_knn) if abs(r2_knn) > 1e-6 else 0.0

        print(f"\n  kappa_nearest:    r={r_knn:.4f}, R2={r2_knn:.4f}, slope={slope_knn:.4f}")
        print(f"  kappa_eff(frozen): r={r_eff:.4f}, R2={r2_eff:.4f}, slope={slope_eff:.4f}")
        print(f"  R2 improvement: {r2_improvement:+.4f} (rel: {r2_rel_improvement*100:+.1f}%)")

        per_model_results[model_name] = {
            "n_classes": len(knn_list),
            "r2_kappa_nearest": r2_knn,
            "r_kappa_nearest": r_knn,
            "r2_kappa_eff_frozen": r2_eff,
            "r_kappa_eff_frozen": r_eff,
            "r2_abs_improvement": r2_improvement,
            "r2_rel_improvement_pct": r2_rel_improvement * 100,
            "improvement_pass": r2_improvement > R2_ABS_IMPROVEMENT_THRESHOLD,
        }

        all_kappa_nearest.extend(knn_list)
        all_kappa_eff.extend(keff_list)
        all_logit_q.extend(lq_list)

    print(f"\n{'='*70}")
    print("POOLED HELD-OUT RESULTS")
    print("=" * 70)

    if len(all_kappa_nearest) < 4:
        print("[ERROR] Too few pooled data points")
        return

    r_pool_knn, r2_pool_knn, slope_pool_knn = fit_r2(all_kappa_nearest, all_logit_q)
    r_pool_eff, r2_pool_eff, slope_pool_eff = fit_r2(all_kappa_eff, all_logit_q)

    r2_pool_improvement = r2_pool_eff - r2_pool_knn
    r2_pool_rel = r2_pool_improvement / abs(r2_pool_knn) if abs(r2_pool_knn) > 1e-6 else 0.0

    pooled_pass = r2_pool_improvement > R2_ABS_IMPROVEMENT_THRESHOLD

    print(f"\nPooled (n={len(all_logit_q)} class-model pairs):")
    print(f"  kappa_nearest:     r={r_pool_knn:.4f}, R2={r2_pool_knn:.4f}")
    print(f"  kappa_eff(frozen): r={r_pool_eff:.4f}, R2={r2_pool_eff:.4f}")
    print(f"  R2 improvement: {r2_pool_improvement:+.4f} (rel: {r2_pool_rel*100:+.1f}%)")
    print(f"\nPRE-REGISTERED CRITERION (R2 improvement > {R2_ABS_IMPROVEMENT_THRESHOLD}):")
    print(f"  POOLED: {'PASS' if pooled_pass else 'FAIL'}")

    per_model_pass = {m: v["improvement_pass"] for m, v in per_model_results.items()}
    n_pass = sum(per_model_pass.values())
    print(f"\nPer-model: {n_pass}/{len(per_model_results)} pass:")
    for m, v in per_model_results.items():
        status = "PASS" if v["improvement_pass"] else "FAIL"
        print(f"  {m}: [{status}] R2_knn={v['r2_kappa_nearest']:.4f} -> R2_eff={v['r2_kappa_eff_frozen']:.4f} "
              f"(+{v['r2_abs_improvement']:.4f})")

    result = {
        "experiment": "kappa_eff_held_out_transfer",
        "session": 38,
        "train_source": TRAIN_MODEL,
        "train_source_file": str(WEIGHT_MAP_JSON),
        "w_by_rank": {str(r): float(w) for r, w in sorted(w_by_rank.items())},
        "sum_w": float(sum_w),
        "held_out_models": list(HELD_OUT_EMBS.keys()),
        "n_pooled_points": len(all_logit_q),
        "pooled": {
            "r2_kappa_nearest": float(r2_pool_knn),
            "r_kappa_nearest": float(r_pool_knn),
            "r2_kappa_eff_frozen": float(r2_pool_eff),
            "r_kappa_eff_frozen": float(r_pool_eff),
            "r2_abs_improvement": float(r2_pool_improvement),
            "r2_rel_improvement_pct": float(r2_pool_rel * 100),
            "pooled_pass": pooled_pass,
        },
        "per_model": per_model_results,
        "pre_registered_criterion": {
            "r2_abs_improvement_threshold": R2_ABS_IMPROVEMENT_THRESHOLD,
            "r2_rel_improvement_threshold_pct": R2_REL_IMPROVEMENT_THRESHOLD * 100,
        },
        "summary_pass": pooled_pass,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
