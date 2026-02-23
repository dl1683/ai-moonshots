#!/usr/bin/env python -u
"""
Cross-Dataset Top-K Universality Test (Session 43)
===================================================
Codex design: test whether k*=5 is absolute or k/(K-1)~0.385 fraction.

DBpedia14 (K=14): k*=5 in 5/5 LOAO folds (Session 42, established).
  k/(K-1) = 5/13 = 0.385

20newsgroups (K=20): NEW test. 4 architectures.

PRE-REGISTERED HYPOTHESES:
  H_absolute: k*=5 on K=20 wins (>= 3/4 LOAO folds) AND
              R2(k=5) > R2(k=1) on >= 3/4 folds (vs random null >= 50th pct)
  H_fraction: k* in {6,7,8} on K=20 wins (>= 3/4 LOAO folds)
              (fraction 5/13*(K-1) = 5/13*19 = 7.3 -> k=7)
  H_nearest:  k*=1 wins (null baseline)

SECONDARY:
  - Fixed k=5 vs fixed k=7: compare delta_R2 on K=20 directly
  - delta_R2(k=5) at >= 50th pct vs random null on K=20

Files (20newsgroups K=20):
  deberta-base:  causal_v2_embs_deberta-base_20newsgroups.npz
  olmo-1b:       causal_v2_embs_olmo-1b_20newsgroups.npz
  qwen3-0.6b:    causal_v2_embs_qwen3-0.6b_20newsgroups.npz
  mamba-130m:    do_int_embs_mamba-130m_20newsgroups.npz

DBpedia14 (K=14) reference:
  Results from kernel showdown: k*=5 in ALL 5 LOAO folds.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_JSON = RESULTS_DIR / "cti_crossdataset_topk.json"

# Pre-registered from Session 42 (DBpedia14 k*)
K_STAR_DBPEDIA = 5
K_DBPEDIA = 14
FRACTION_DBPEDIA = K_STAR_DBPEDIA / (K_DBPEDIA - 1)  # 5/13 = 0.385

# 20newsgroups
K_20NG = 20
K_FRACTION_PRED = round(FRACTION_DBPEDIA * (K_20NG - 1))  # round(0.385 * 19) = 7

TOPK_GRID_20NG = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 19]
N_CV_Q = 5
BOOTSTRAP_N = 500
N_RANDOM_NULL = 500
RNG_SEED = 42

ARCH_CACHES_20NG = {
    "deberta-base": "causal_v2_embs_deberta-base_20newsgroups.npz",
    "olmo-1b":      "causal_v2_embs_olmo-1b_20newsgroups.npz",
    "qwen3-0.6b":   "causal_v2_embs_qwen3-0.6b_20newsgroups.npz",
    "mamba-130m":   "do_int_embs_mamba-130m_20newsgroups.npz",
}


def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        centroids[c] = Xc.mean(0)
        resids.append(Xc - centroids[c])
    sigma_W = float(np.sqrt(np.mean(np.vstack(resids)**2)))
    return centroids, sigma_W


def compute_per_class_q(X, y, ci):
    K_local = len(np.unique(y))
    skf = StratifiedKFold(n_splits=N_CV_Q, shuffle=True, random_state=42)
    recalls = []
    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if (y_tr == ci).sum() < 1:
            continue
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        knn.fit(X_tr, y_tr)
        mask = y_te == ci
        if mask.sum() == 0:
            continue
        recalls.append(float((knn.predict(X_te[mask]) == ci).mean()))
    if not recalls:
        return None
    q_raw = float(np.mean(recalls))
    return float((q_raw - 1.0 / K_local) / (1.0 - 1.0 / K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1 - 1e-5))
    return float(np.log(q / (1.0 - q)))


def load_arch(cache_path):
    data = np.load(str(cache_path))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    centroids, sigma_W = compute_class_stats(X, y)

    rows = []
    for ci in classes:
        mu_i = centroids[ci]
        kappas = []
        for cj, mu_j in centroids.items():
            if cj == ci:
                continue
            dist = float(np.linalg.norm(mu_i - mu_j))
            k = dist / (sigma_W * np.sqrt(d) + 1e-10)
            kappas.append((k, cj))
        kappas.sort()
        if len(kappas) < 2:
            continue
        q = compute_per_class_q(X, y, ci)
        if q is None:
            continue
        rows.append({
            "ci": int(ci),
            "kappa_nearest": float(kappas[0][0]),
            "kappas": np.array([kp for kp, _ in kappas]),
            "logit_q": safe_logit(q),
        })

    print(f"  {cache_path.name}: N={len(X)}, K={len(classes)}, d={d}, valid={len(rows)}")
    return rows


def feat_topk(rows, k):
    return np.array([float(np.mean(r["kappas"][:k])) for r in rows])


def r2_within(feat, lq):
    x = np.array(feat).reshape(-1, 1)
    y = np.array(lq)
    if len(x) < 4 or float(np.std(x)) < 1e-10:
        return 0.0
    lr = LinearRegression().fit(x, y)
    ss_res = float(np.sum((y - lr.predict(x))**2))
    ss_tot = float(np.sum((y - float(np.mean(y)))**2))
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def find_best_topk(tr_rows_list, grid):
    """Find k that maximizes mean within-arch R2 on training archs."""
    best_k, bv = grid[0], -1e9
    for k in grid:
        r2s = [r2_within(feat_topk(rows, k), [r["logit_q"] for r in rows])
               for rows in tr_rows_list]
        v = float(np.mean(r2s))
        if v > bv:
            bv, best_k = v, k
    return best_k


def json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    print("=" * 70)
    print("CROSS-DATASET TOP-K UNIVERSALITY TEST")
    print(f"DBpedia14: K={K_DBPEDIA}, k*=5 (established), fraction={FRACTION_DBPEDIA:.3f}")
    print(f"20newsgroups: K={K_20NG}, k*_fraction_pred={K_FRACTION_PRED}")
    print(f"Grid: {TOPK_GRID_20NG}")
    print("=" * 70)

    # --- Load 20newsgroups ---
    arch_rows = {}
    for name, cname in ARCH_CACHES_20NG.items():
        cp = RESULTS_DIR / cname
        if not cp.exists():
            print(f"  MISSING: {cname}")
            continue
        print(f"\nLoading {name}...")
        arch_rows[name] = load_arch(cp)

    arch_names = list(arch_rows.keys())
    N_ARCH = len(arch_names)
    print(f"\n{N_ARCH} architectures loaded: {arch_names}")

    K_minus_1 = len(arch_rows[arch_names[0]][0]["kappas"])
    print(f"K-1 = {K_minus_1} competitors per class")

    # Clip grid to valid range (k <= K-1)
    valid_grid = [k for k in TOPK_GRID_20NG if k <= K_minus_1]
    print(f"Valid k grid: {valid_grid}")

    # --- LOAO ---
    print("\n" + "=" * 70)
    print("LOAO FOLDS (20newsgroups)")
    print("=" * 70)

    loao_results = []
    pointwise_by_k = {k: [] for k in valid_grid}

    for held_out in arch_names:
        tr_archs = [a for a in arch_names if a != held_out]
        tr_rows_list = [arch_rows[a] for a in tr_archs]
        te_rows = arch_rows[held_out]
        te_lq = np.array([r["logit_q"] for r in te_rows])

        # Find k* on training archs
        k_star = find_best_topk(tr_rows_list, valid_grid)
        print(f"\n  Held-out: {held_out}")
        print(f"    k*={k_star} (from training on {tr_archs})")

        # Evaluate ALL k on test arch
        r2_by_k = {}
        for k in valid_grid:
            feat = feat_topk(te_rows, k)
            r2_by_k[k] = r2_within(feat.tolist(), te_lq.tolist())

        r2_base = r2_by_k[1]  # k=1 = kappa_nearest
        print(f"    R2 by k: " + ", ".join(f"k={k}:{r2_by_k[k]:.3f}" for k in valid_grid))
        print(f"    R2(base=k1)={r2_base:.3f}, R2(k5)={r2_by_k.get(5,0):.3f}, "
              f"R2(k{K_FRACTION_PRED})={r2_by_k.get(K_FRACTION_PRED,0):.3f}")

        loao_results.append({
            "held_out": held_out,
            "k_star": int(k_star),
            "r2_by_k": {str(k): float(v) for k, v in r2_by_k.items()},
            "r2_base": float(r2_base),
            "dr2_k5": float(r2_by_k.get(5, r2_base) - r2_base),
            "dr2_kfrac": float(r2_by_k.get(K_FRACTION_PRED, r2_base) - r2_base),
            "dr2_kstar": float(r2_by_k[k_star] - r2_base),
        })

        # Pointwise for CI
        feat_base = feat_topk(te_rows, 1).reshape(-1, 1)
        lr_base = LinearRegression().fit(feat_base, te_lq)
        e_base = (te_lq - lr_base.predict(feat_base))**2

        for k in valid_grid:
            feat_k = feat_topk(te_rows, k).reshape(-1, 1)
            lr_k = LinearRegression().fit(feat_k, te_lq)
            e_k = (te_lq - lr_k.predict(feat_k))**2
            pointwise_by_k[k].extend((e_base - e_k).tolist())

    # --- Aggregate ---
    rng = np.random.default_rng(RNG_SEED)
    pooled_dr2_by_k = {}
    ci_by_k = {}
    for k in valid_grid:
        dr2s = [r["r2_by_k"][str(k)] - r["r2_base"] for r in loao_results]
        pooled_dr2_by_k[k] = float(np.mean(dr2s))
        pw = np.array(pointwise_by_k[k])
        boots = [float(np.mean(rng.choice(pw, size=len(pw), replace=True)))
                 for _ in range(BOOTSTRAP_N)]
        ci_by_k[k] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    print("\n" + "=" * 70)
    print("DR2 BY K (20newsgroups LOAO, vs k=1 baseline):")
    for k in valid_grid:
        ci = ci_by_k[k]
        print(f"  k={k:2d}: dR2={pooled_dr2_by_k[k]:+.4f}  CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")

    # Best k on 20newsgroups
    best_k_20ng = max(valid_grid, key=lambda k: pooled_dr2_by_k[k])
    print(f"\n  Best k on 20newsgroups: k*={best_k_20ng} (dR2={pooled_dr2_by_k[best_k_20ng]:+.4f})")
    print(f"  k=5 (absolute pred):    dR2={pooled_dr2_by_k.get(5, 0):+.4f}")
    print(f"  k={K_FRACTION_PRED} (fraction pred):  dR2={pooled_dr2_by_k.get(K_FRACTION_PRED, 0):+.4f}")

    k_star_counts = {}
    for r in loao_results:
        ks = r["k_star"]
        k_star_counts[ks] = k_star_counts.get(ks, 0) + 1
    print(f"\n  k* frequency in LOAO folds: {k_star_counts}")

    # --- Random Null ---
    print("\n" + "=" * 70)
    print("RANDOM NULL (500 monotone weight vectors)")
    print("=" * 70)
    rng2 = np.random.default_rng(RNG_SEED + 1)

    null_pooled = []
    for i_null in range(N_RANDOM_NULL):
        raw = rng2.exponential(1.0, size=K_minus_1)
        w_null = np.sort(raw)[::-1]
        w_null = w_null / w_null.sum()

        fold_deltas = []
        for held_out in arch_names:
            te_rows = arch_rows[held_out]
            te_lq = np.array([r["logit_q"] for r in te_rows])

            # Random weighted feature
            te_null = np.array([
                float(np.dot(w_null[:len(r["kappas"])], r["kappas"][:len(w_null)]))
                for r in te_rows
            ])
            te_kappa1 = np.array([r["kappa_nearest"] for r in te_rows])

            r2_n = r2_within(te_null.tolist(), te_lq.tolist())
            r2_b = r2_within(te_kappa1.tolist(), te_lq.tolist())
            fold_deltas.append(r2_n - r2_b)

        null_pooled.append(float(np.mean(fold_deltas)))
        if i_null % 100 == 0:
            print(f"  null {i_null}/{N_RANDOM_NULL}, mean={float(np.mean(null_pooled)):.4f}")

    null_arr = np.array(null_pooled)
    null_mean = float(null_arr.mean())
    null_p90 = float(np.percentile(null_arr, 90))
    print(f"Null: mean={null_mean:.4f}, p90={null_p90:.4f}")

    # Percentile for k=5, k=fraction, k=best
    pct_k5 = float(np.mean(pooled_dr2_by_k.get(5, 0) >= null_arr) * 100)
    pct_kfrac = float(np.mean(pooled_dr2_by_k.get(K_FRACTION_PRED, 0) >= null_arr) * 100)
    pct_best = float(np.mean(pooled_dr2_by_k[best_k_20ng] >= null_arr) * 100)

    # --- Hypothesis evaluation ---
    k_star_list = [r["k_star"] for r in loao_results]
    n_k5_wins = sum(1 for ks in k_star_list if ks == 5)
    n_kfrac_wins = sum(1 for ks in k_star_list if ks == K_FRACTION_PRED)
    n_near_kfrac = sum(1 for ks in k_star_list if abs(ks - K_FRACTION_PRED) <= 1)

    pass_H_abs = (n_k5_wins >= 3) and (pooled_dr2_by_k.get(5, 0) > 0) and (pct_k5 >= 50.0)
    pass_H_frac = (n_near_kfrac >= 3) and (pooled_dr2_by_k.get(K_FRACTION_PRED, 0) > 0)

    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)
    print(f"DBpedia14 (K=14): k*=5 in 5/5 folds (established)")
    print(f"Fraction: 5/(14-1) = {FRACTION_DBPEDIA:.4f}")
    print(f"Predicted k* for K=20: fraction -> k={K_FRACTION_PRED}, absolute -> k=5")
    print()
    print(f"H_absolute (k*=5): n_wins={n_k5_wins}/4, dR2={pooled_dr2_by_k.get(5,0):+.4f}, "
          f"null_pct={pct_k5:.1f}th -> {'PASS' if pass_H_abs else 'FAIL'}")
    print(f"H_fraction (k*={K_FRACTION_PRED}): n_near_wins={n_near_kfrac}/4, "
          f"dR2={pooled_dr2_by_k.get(K_FRACTION_PRED,0):+.4f}, "
          f"null_pct={pct_kfrac:.1f}th -> {'PASS' if pass_H_frac else 'FAIL'}")
    print(f"Best k overall: k*={best_k_20ng}, dR2={pooled_dr2_by_k[best_k_20ng]:+.4f}, "
          f"null_pct={pct_best:.1f}th")
    print(f"\nk* LOAO folds: {k_star_list}")
    print(f"k* frequency: {k_star_counts}")

    # --- Save ---
    output = {
        "experiment": "crossdataset_topk_universality",
        "session": 43,
        "preregistered": {
            "K_dbpedia": K_DBPEDIA,
            "k_star_dbpedia": K_STAR_DBPEDIA,
            "fraction_dbpedia": FRACTION_DBPEDIA,
            "K_20ng": K_20NG,
            "k_fraction_pred_20ng": K_FRACTION_PRED,
            "H_absolute_criteria": "k*=5 >= 3/4 folds AND dR2>0 AND null_pct>=50",
            "H_fraction_criteria": f"k* in {{{K_FRACTION_PRED-1},{K_FRACTION_PRED},{K_FRACTION_PRED+1}}} >= 3/4 folds AND dR2>0",
        },
        "architectures_20ng": arch_names,
        "n_arch_20ng": N_ARCH,
        "loao_results": loao_results,
        "pooled_dr2_by_k": {str(k): float(v) for k, v in pooled_dr2_by_k.items()},
        "ci_by_k": {str(k): v for k, v in ci_by_k.items()},
        "k_star_20ng_loao_folds": k_star_list,
        "k_star_20ng_counts": k_star_counts,
        "best_k_20ng": best_k_20ng,
        "null_mean": null_mean,
        "null_p90": null_p90,
        "null_pct_k5": pct_k5,
        "null_pct_kfrac": pct_kfrac,
        "null_pct_best": pct_best,
        "pass_H_absolute": bool(pass_H_abs),
        "pass_H_fraction": bool(pass_H_frac),
        "interpretation": (
            "absolute" if pass_H_abs and not pass_H_frac else
            "fraction" if pass_H_frac and not pass_H_abs else
            "both" if pass_H_abs and pass_H_frac else
            "neither"
        ),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
