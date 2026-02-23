#!/usr/bin/env python -u
"""
Frozen-Transfer Model Tournament + Random-Null Specificity Test
===============================================================
Codex priority experiment (Session 38, after held-out 4/4 PASS):

QUESTION: Is causal w_r BETTER than kappa_mean, phi(tau), or random weights?
If not, the held-out improvement is an artifact of including more kappas.

DESIGN:
  - Same 4 held-out architectures as cti_kappa_eff_held_out.py
  - Compare 4 predictors, all with A refitted per held-out model (free slope test):
    1. kappa_nearest (baseline)
    2. kappa_mean (equal weights, mean of all K-1 kappas)
    3. phi(tau=0.2) [best tau from phi_upgrade_pooled.json]
    4. kappa_eff_causal (w_r from pythia-160m causal weight map)
  - Also test: 1000 random monotone w_r vectors with matched sum_w
    => compute random-null percentile for causal w_r

PRE-REGISTERED CRITERION:
  - causal w_r R2 > kappa_mean R2 on pooled held-out (directional test)
  - causal w_r R2 above 90th percentile of random null

NOTE: All competitors use A refitted per model (free slope).
This tests the FUNCTIONAL FORM QUALITY independent of amplitude.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, linregress
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
WEIGHT_MAP_JSON = RESULTS_DIR / "cti_competitor_weight_map.json"
OUT_JSON = RESULTS_DIR / "cti_kappa_tournament.json"

TAU_STAR = 0.2   # best tau from phi_upgrade_pooled
N_RANDOM = 1000  # random weight vectors for null distribution
N_SPLITS_CV = 5
RANDOM_SEED = 42

HELD_OUT_EMBS = {
    "pythia-410m":   "results/dointerv_multi_pythia-410m_l3.npz",
    "electra-small": "results/dointerv_multi_electra-small_l3.npz",
    "rwkv-4-169m":   "results/dointerv_multi_rwkv-4-169m_l12.npz",
    "bert-base":     "results/dointerv_multi_bert-base-uncased_l10.npz",
}


def load_causal_w_r(json_path):
    with open(json_path) as f:
        data = json.load(f)
    agg = data["aggregate_by_rank"]
    w = {}
    for entry in agg:
        w[int(entry["rank"])] = float(entry["mean_w"])
    return w


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
    mu_i = centroids[ci]
    kappas = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappas.append(dist / (sigma_W * np.sqrt(d) + 1e-10))
    kappas.sort()
    return kappas


def phi_tau_fn(kappas, tau):
    kappas = np.array(kappas)
    z = -kappas / tau
    z_max = z.max()
    return float(-tau * (z_max + np.log(np.sum(np.exp(z - z_max)))))


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


def fit_r2_free(xs, ys):
    """R2 with free slope and intercept."""
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 4 or np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
        return 0.0, 0.0
    r, _ = pearsonr(xs, ys)
    slope, intercept, _, _, _ = linregress(xs, ys)
    y_pred = slope * xs + intercept
    ss_res = float(np.sum((ys - y_pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(r), float(r2)


def generate_random_weight_vectors(K_minus_1, n_random, sum_w_target, rng):
    """Generate n_random random monotone-decreasing non-negative weight vectors
    with sum approximately equal to sum_w_target."""
    vectors = []
    for _ in range(n_random):
        # Sample K-1 uniform values, sort descending (monotone), normalize to target sum
        raw = rng.exponential(1.0, size=K_minus_1)
        raw_sorted = np.sort(raw)[::-1]  # descending
        total = raw_sorted.sum()
        if total < 1e-10:
            vectors.append(np.zeros(K_minus_1))
        else:
            vectors.append(raw_sorted * sum_w_target / total)
    return vectors


def main():
    print("=" * 70)
    print("FROZEN TRANSFER TOURNAMENT + RANDOM-NULL SPECIFICITY TEST")
    print(f"Comparing: kappa_nearest, kappa_mean, phi(tau={TAU_STAR}), kappa_eff_causal")
    print(f"Random null: {N_RANDOM} random monotone weight vectors")
    print("=" * 70)

    # Load causal weights
    w_causal = load_causal_w_r(WEIGHT_MAP_JSON)
    sum_w = sum(w_causal.values())
    K_minus_1 = len(w_causal)

    print(f"\nCausal w_r: sum_w={sum_w:.3f}, K-1={K_minus_1}")
    print(f"  Profile: " + ", ".join(f"r{r}={w_causal[r]:.3f}" for r in sorted(w_causal)))

    # Pre-generate random weight vectors (same for all models)
    rng = np.random.default_rng(RANDOM_SEED)
    random_weights_list = generate_random_weight_vectors(K_minus_1, N_RANDOM, sum_w, rng)
    print(f"\nGenerated {N_RANDOM} random weight vectors (sum_w matched to {sum_w:.3f})")

    # Storage for pooled results
    all_lq = []
    all_knn = []
    all_kmean = []
    all_phi = []
    all_eff = []
    all_rand = [[] for _ in range(N_RANDOM)]

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
        print(f"MODEL: {model_name} (d={d}, K={K})")
        print("=" * 60)

        centroids, sigma_W = compute_class_stats(X, y)

        knn_vals, kmean_vals, phi_vals, eff_vals, lq_vals = [], [], [], [], []
        rand_vals = [[] for _ in range(N_RANDOM)]

        for ci in classes:
            kappas = compute_all_kappas_sorted(centroids, sigma_W, d, ci)
            if not kappas:
                continue
            kappas_arr = np.array(kappas)
            n_k = len(kappas)

            # kappa_nearest
            knn_v = kappas[0]

            # kappa_mean
            kmean_v = float(np.mean(kappas))

            # phi(tau=0.2)
            phi_v = phi_tau_fn(kappas, TAU_STAR)

            # kappa_eff_causal
            eff_v = sum(w_causal.get(r + 1, 0.0) * kappas[r] for r in range(n_k))

            # Per-class accuracy
            q = compute_per_class_q(X, y, ci)
            if q is None:
                continue
            lq = safe_logit(q)

            knn_vals.append(knn_v)
            kmean_vals.append(kmean_v)
            phi_vals.append(phi_v)
            eff_vals.append(eff_v)
            lq_vals.append(lq)

            # Random weights (need kappas of length K-1=13, pad if needed)
            k13 = np.zeros(K_minus_1)
            k13[:n_k] = kappas[:K_minus_1]
            for ri, w_vec in enumerate(random_weights_list):
                rand_v = float(np.dot(w_vec[:n_k], kappas_arr[:len(w_vec[:n_k])]))
                rand_vals[ri].append(rand_v)

        if len(lq_vals) < 4:
            continue

        # Fit R2 per predictor
        r_knn, r2_knn = fit_r2_free(knn_vals, lq_vals)
        r_kmean, r2_kmean = fit_r2_free(kmean_vals, lq_vals)
        r_phi, r2_phi = fit_r2_free(phi_vals, lq_vals)
        r_eff, r2_eff = fit_r2_free(eff_vals, lq_vals)

        # Random null R2 distribution
        rand_r2s = []
        for ri in range(N_RANDOM):
            if len(rand_vals[ri]) == len(lq_vals):
                _, r2_rand = fit_r2_free(rand_vals[ri], lq_vals)
                rand_r2s.append(r2_rand)

        pct_causal_vs_random = (np.array(rand_r2s) < r2_eff).mean() * 100 if rand_r2s else 0.0

        print(f"\n  n_classes={len(lq_vals)}")
        print(f"  kappa_nearest:  r={r_knn:.4f}, R2={r2_knn:.4f}")
        print(f"  kappa_mean:     r={r_kmean:.4f}, R2={r2_kmean:.4f}")
        print(f"  phi(tau=0.2):   r={r_phi:.4f}, R2={r2_phi:.4f}")
        print(f"  kappa_eff_causal: r={r_eff:.4f}, R2={r2_eff:.4f}")
        print(f"  Random null: causal > {pct_causal_vs_random:.1f}% of random (n={len(rand_r2s)})")

        per_model_results[model_name] = {
            "n_classes": len(lq_vals),
            "r2_kappa_nearest": r2_knn,
            "r2_kappa_mean": r2_kmean,
            "r2_phi": r2_phi,
            "r2_kappa_eff_causal": r2_eff,
            "random_pct": float(pct_causal_vs_random),
            "causal_beats_mean": r2_eff > r2_kmean,
            "causal_beats_phi": r2_eff > r2_phi,
            "causal_above_90pct_random": pct_causal_vs_random >= 90.0,
        }

        all_lq.extend(lq_vals)
        all_knn.extend(knn_vals)
        all_kmean.extend(kmean_vals)
        all_phi.extend(phi_vals)
        all_eff.extend(eff_vals)
        for ri in range(N_RANDOM):
            if len(rand_vals[ri]) == len(lq_vals):
                all_rand[ri].extend(rand_vals[ri])

    # Pooled results
    print(f"\n{'='*70}")
    print("POOLED TOURNAMENT RESULTS")
    print("=" * 70)

    n_pool = len(all_lq)
    print(f"\nPooled (n={n_pool} class-model pairs):")

    _, r2_pool_knn = fit_r2_free(all_knn, all_lq)
    _, r2_pool_kmean = fit_r2_free(all_kmean, all_lq)
    _, r2_pool_phi = fit_r2_free(all_phi, all_lq)
    _, r2_pool_eff = fit_r2_free(all_eff, all_lq)

    print(f"  kappa_nearest:    R2={r2_pool_knn:.4f}")
    print(f"  kappa_mean:       R2={r2_pool_kmean:.4f}")
    print(f"  phi(tau=0.2):     R2={r2_pool_phi:.4f}")
    print(f"  kappa_eff_causal: R2={r2_pool_eff:.4f}")

    # Pooled random null
    pool_rand_r2s = []
    for ri in range(N_RANDOM):
        if len(all_rand[ri]) == n_pool:
            _, r2r = fit_r2_free(all_rand[ri], all_lq)
            pool_rand_r2s.append(r2r)

    pct_pool_vs_random = (np.array(pool_rand_r2s) < r2_pool_eff).mean() * 100 if pool_rand_r2s else 0.0
    print(f"  Random null (pooled): causal > {pct_pool_vs_random:.1f}% of {len(pool_rand_r2s)} vectors")
    print(f"    random null R2: mean={np.mean(pool_rand_r2s):.4f}, p90={np.percentile(pool_rand_r2s, 90):.4f}, p99={np.percentile(pool_rand_r2s, 99):.4f}")

    # Pre-registered criteria
    causal_beats_mean = r2_pool_eff > r2_pool_kmean
    causal_above_90pct = pct_pool_vs_random >= 90.0
    print(f"\nPRE-REGISTERED CRITERIA:")
    print(f"  causal > kappa_mean (pooled): {'PASS' if causal_beats_mean else 'FAIL'} "
          f"({r2_pool_eff:.4f} vs {r2_pool_kmean:.4f})")
    print(f"  causal > 90th pct random: {'PASS' if causal_above_90pct else 'FAIL'} "
          f"({pct_pool_vs_random:.1f}%)")

    result = {
        "experiment": "kappa_tournament_frozen_transfer",
        "session": 38,
        "tau_star": TAU_STAR,
        "n_random_null": N_RANDOM,
        "w_causal": {str(r): float(w) for r, w in sorted(w_causal.items())},
        "sum_w": float(sum_w),
        "n_pooled": n_pool,
        "pooled": {
            "r2_kappa_nearest": float(r2_pool_knn),
            "r2_kappa_mean": float(r2_pool_kmean),
            "r2_phi_tau020": float(r2_pool_phi),
            "r2_kappa_eff_causal": float(r2_pool_eff),
            "random_null_r2_mean": float(np.mean(pool_rand_r2s)) if pool_rand_r2s else None,
            "random_null_r2_p90": float(np.percentile(pool_rand_r2s, 90)) if pool_rand_r2s else None,
            "random_null_r2_p99": float(np.percentile(pool_rand_r2s, 99)) if pool_rand_r2s else None,
            "pct_causal_beats_random": float(pct_pool_vs_random),
            "causal_beats_mean": causal_beats_mean,
            "causal_above_90pct_random": causal_above_90pct,
        },
        "per_model": per_model_results,
        "pre_registered": {
            "criterion_1": "causal R2 > kappa_mean R2 (pooled)",
            "criterion_2": "causal R2 above 90th percentile of random null",
            "overall_pass": causal_beats_mean and causal_above_90pct,
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
