#!/usr/bin/env python -u
"""
RECONCILIATION BRIDGE: Unified law spanning log(K) -> sqrt(K) regimes.

Codex design (Feb 20, 2026): 10/10 Nobel-track.

THE DISCREPANCY (critical finding):
  - Theory (Gumbel race) predicts: logit(q) = A*kappa - B*log(K) + C
  - Synthetic 1-NN/nearest centroid: log(K) wins (R^2=0.962)
  - Real data 5-NN cosine: sqrt(K) wins (R^2=0.785)

THE RESOLUTION (Codex diagnosis):
  1. Different statistics: 1-NN (Gumbel EVT) vs 5-NN cosine (different statistic)
  2. Cross-dataset pooling confound (random intercepts per dataset)
  3. Wrong order parameter (kappa_spec vs kappa_nearest)

THE BRIDGE MODEL:
  logit(q) = (a*kappa - b*log(K-1) + c) / sqrt(1 + d*K^psi)

  Limits:
  - d=0: additive log-law (Gumbel, 1-NN regime)
  - large d*K^psi: divisive power-law ~ kappa / K^(psi/2) (kNN majority vote regime)

EXPERIMENTS:
1. Statistic ablation on synthetic: 1nn_l2, 1nn_cos, 5nn_l2, 5nn_cos, centroid_l2
   - Compare additive-log vs divisive-sqrt vs bridge model for each
   - KEY PREDICTION: 1nn_l2/centroid -> log(K); 5nn -> different scaling

2. Real data mixed-effects fit:
   - Pool all real data with dataset random intercepts
   - Test kappa_spec vs kappa_nearest proxy
   - After controlling for dataset intercept, does log(K) work?

3. Unifying fit: bridge model on synthetic + real simultaneously
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logit as sp_logit
from scipy.special import expit as sigmoid

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
np.random.seed(42)


# ============================================================
# SYNTHETIC GAUSSIAN: K-class classification with different statistics
# ============================================================

def generate_gaussian_data(K, r, kappa_near, d=100, n_per=100, sigma_w2=1.0, seed=42):
    """Generate K-class Gaussian data with given kappa_nearest."""
    rng = np.random.RandomState(seed)

    # Compute sigma_b2 from kappa_nearest
    h = h_rK(r, K)
    kappa_spec = kappa_near / max(h, 1e-12)
    sigma_b2 = kappa_spec * (K / max(K - 1, 1)) * d / r

    # Draw class means in R^r embedded in R^d
    means_r = rng.randn(K, r) * np.sqrt(sigma_b2)
    means = np.zeros((K, d))
    means[:, :r] = means_r

    # Draw training and test data
    X_train, y_train = [], []
    X_test, y_test = [], []
    n_test = max(n_per, 50)

    for k in range(K):
        Xk_train = means[k] + rng.randn(n_per, d) * np.sqrt(sigma_w2)
        Xk_test = means[k] + rng.randn(n_test, d) * np.sqrt(sigma_w2)
        X_train.append(Xk_train)
        X_test.append(Xk_test)
        y_train.extend([k] * n_per)
        y_test.extend([k] * n_test)

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test, means, float(kappa_spec)


def compute_knn_quality(X_train, y_train, X_test, y_test, K, k_neighbors, metric="l2"):
    """Compute kNN quality q with given k and metric."""
    from sklearn.neighbors import KNeighborsClassifier

    if metric == "cosine":
        # Normalize for cosine
        X_train_n = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-12)
        X_test_n = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-12)
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean")
        knn.fit(X_train_n, y_train)
        pred = knn.predict(X_test_n)
    else:
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean")
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)

    acc = np.mean(pred == y_test)
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    return float(max(q, 0.0))


def compute_centroid_quality(X_train, y_train, X_test, y_test, K):
    """Nearest centroid classification quality."""
    centroids = np.array([X_train[y_train == k].mean(0) for k in range(K)])
    # Distances to centroids
    dists = np.sum((X_test[:, np.newaxis] - centroids[np.newaxis, :]) ** 2, axis=2)
    pred = np.argmin(dists, axis=1)
    acc = np.mean(pred == y_test)
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    return float(max(q, 0.0))


def compute_kappa_nearest_empirical(X_train, y_train, K):
    """Compute kappa_nearest from training data."""
    d = X_train.shape[1]
    centroids = np.array([X_train[y_train == k].mean(0) for k in range(K)])
    min_dists = []
    for i in range(K):
        dists = [np.sum((centroids[i] - centroids[j]) ** 2) for j in range(K) if j != i]
        min_dists.append(min(dists))
    return float(np.mean(min_dists)) / d


# ============================================================
# H(r, K) FUNCTION
# ============================================================

_h_cache = {}

def h_rK(r, K):
    """h(r, K) = 2 * E[chi^2(r)_min_(K-1)] / r via numerical integration."""
    key = (r, K)
    if key in _h_cache:
        return _h_cache[key]
    m = K - 1
    if m < 1:
        val = 2.0
    else:
        x_max = float(stats.chi2.ppf(1 - 1e-10, df=r))
        xs = np.linspace(0, x_max, 2000)
        dx = xs[1] - xs[0]
        survival = (1.0 - stats.chi2.cdf(xs, df=r)) ** m
        E_min = float(np.sum(survival) * dx)
        val = 2.0 * E_min / r
    _h_cache[key] = val
    return val


# ============================================================
# FIT UTILITIES
# ============================================================

def fit_additive_log(kappa, K, q):
    """Fit logit(q) = a*kappa - b*log(K-1) + c."""
    q_clip = np.clip(q, 0.001, 0.999)
    logit_q = sp_logit(q_clip)
    log_K = np.log(np.maximum(K - 1, 1))

    def loss(params):
        a, b, c = params
        pred = a * kappa - b * log_K + c
        return np.mean((logit_q - pred) ** 2)

    from scipy.optimize import minimize
    res = minimize(loss, [10.0, 1.0, 0.0], method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-8})
    a, b, c = res.x
    pred = a * kappa - b * log_K + c
    ss_res = np.sum((logit_q - pred) ** 2)
    ss_tot = np.sum((logit_q - logit_q.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    mae = np.mean(np.abs(sigmoid(pred) - q))
    return {"model": "additive_log", "r2": float(r2), "mae": float(mae),
            "a": float(a), "b": float(b), "c": float(c)}


def fit_divisive_power(kappa, K, q, gamma):
    """Fit logit(q) = a*kappa/K^gamma + c."""
    q_clip = np.clip(q, 0.001, 0.999)
    x = kappa / (K ** gamma + 1e-12)
    logit_q = sp_logit(q_clip)

    def loss(params):
        a, c = params
        pred = a * x + c
        return np.mean((logit_q - pred) ** 2)

    from scipy.optimize import minimize
    res = minimize(loss, [10.0, 0.0], method="Nelder-Mead",
                   options={"maxiter": 3000})
    a, c = res.x
    pred = a * x + c
    ss_res = np.sum((logit_q - pred) ** 2)
    ss_tot = np.sum((logit_q - logit_q.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    mae = np.mean(np.abs(sigmoid(pred) - q))
    return {"model": f"divisive_K^{gamma:.2f}", "r2": float(r2), "mae": float(mae),
            "a": float(a), "c": float(c), "gamma": float(gamma)}


def fit_bridge(kappa, K, q):
    """Fit logit(q) = (a*kappa - b*log(K-1) + c) / sqrt(1 + d*K^psi)."""
    q_clip = np.clip(q, 0.001, 0.999)
    logit_q = sp_logit(q_clip)
    log_K = np.log(np.maximum(K - 1, 1))

    def loss(params):
        a, b, c, d_param, psi = params
        denom = np.sqrt(1.0 + abs(d_param) * K ** abs(psi))
        pred = (a * kappa - b * log_K + c) / denom
        return np.mean((logit_q - pred) ** 2)

    from scipy.optimize import minimize
    res = minimize(loss, [10.0, 1.0, 0.0, 0.01, 1.0], method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-10})
    a, b, c, d_param, psi = res.x
    d_param, psi = abs(d_param), abs(psi)
    denom = np.sqrt(1.0 + d_param * K ** psi)
    pred = (a * kappa - b * log_K + c) / denom
    ss_res = np.sum((logit_q - pred) ** 2)
    ss_tot = np.sum((logit_q - logit_q.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    mae = np.mean(np.abs(sigmoid(pred) - q))
    return {"model": "bridge", "r2": float(r2), "mae": float(mae),
            "a": float(a), "b": float(b), "c": float(c),
            "d": float(d_param), "psi": float(psi),
            "regime": "EVT-dominant" if d_param * K.max() ** psi < 1 else "variance-dominant"}


# ============================================================
# PART 1: STATISTIC ABLATION ON SYNTHETIC DATA
# ============================================================

print("=" * 70)
print("PART 1: STATISTIC ABLATION (k-NN statistic vs K-normalization)")
print("=" * 70)

K_vals = [5, 10, 20, 50, 100, 200]
r_val = 20
d_val = 100
n_per = 80
kappa_near_grid = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]

statistics = [
    ("centroid_l2", "centroid", None, "l2"),
    ("1nn_l2", 1, None, "l2"),
    ("1nn_cos", 1, None, "cosine"),
    ("5nn_l2", 5, None, "l2"),
    ("5nn_cos", 5, None, "cosine"),
]

stat_results = {}

for stat_name, k_or_centroid, _, metric in statistics:
    print(f"\n--- Statistic: {stat_name} ---", flush=True)
    all_rows = []

    for K in K_vals:
        for kn_idx, kn in enumerate(kappa_near_grid):
            # Multiple seeds for robustness
            q_vals = []
            kn_emp_vals = []
            for seed_offset in range(3):
                seed = 42 + K * 100 + kn_idx * 7 + seed_offset * 31
                try:
                    X_tr, y_tr, X_te, y_te, means, kappa_spec = generate_gaussian_data(
                        K=K, r=r_val, kappa_near=kn, d=d_val, n_per=n_per, seed=seed
                    )
                    if k_or_centroid == "centroid":
                        q = compute_centroid_quality(X_tr, y_tr, X_te, y_te, K)
                    else:
                        q = compute_knn_quality(X_tr, y_tr, X_te, y_te, K,
                                                k_neighbors=k_or_centroid, metric=metric)
                    kn_emp = compute_kappa_nearest_empirical(X_tr, y_tr, K)
                    q_vals.append(q)
                    kn_emp_vals.append(kn_emp)
                except Exception as exc:
                    print(f"  Error K={K}, kn={kn}: {exc}", flush=True)
                    continue

            if len(q_vals) > 0:
                all_rows.append({
                    "K": float(K), "kappa_near": float(kn),
                    "q": float(np.mean(q_vals)),
                    "kappa_emp": float(np.mean(kn_emp_vals)),
                })
                sys.stdout.flush()

    if len(all_rows) < 10:
        print(f"  Not enough data points ({len(all_rows)})", flush=True)
        continue

    kappa_arr = np.array([r["kappa_near"] for r in all_rows])
    K_arr = np.array([r["K"] for r in all_rows])
    q_arr = np.array([r["q"] for r in all_rows])

    # Fit all models
    fit_add = fit_additive_log(kappa_arr, K_arr, q_arr)
    fit_sqrtK = fit_divisive_power(kappa_arr, K_arr, q_arr, gamma=0.5)
    fit_logK = fit_divisive_power(kappa_arr, K_arr, q_arr, gamma=0.0)  # divisive log(K) ~ K^0.001 approx

    # Find best gamma
    best_gamma, best_r2 = 0.5, -1.0
    for gamma in np.arange(0.0, 1.01, 0.05):
        fit = fit_divisive_power(kappa_arr, K_arr, q_arr, gamma=gamma)
        if fit["r2"] > best_r2:
            best_r2 = fit["r2"]
            best_gamma = gamma

    fit_free = fit_divisive_power(kappa_arr, K_arr, q_arr, gamma=best_gamma)
    fit_br = fit_bridge(kappa_arr, K_arr, q_arr)

    print(f"  additive-log: R^2={fit_add['r2']:.4f}, b={fit_add['b']:.3f}", flush=True)
    print(f"  divisive-sqrt: R^2={fit_sqrtK['r2']:.4f}", flush=True)
    print(f"  free-power(gamma={best_gamma:.2f}): R^2={fit_free['r2']:.4f}", flush=True)
    print(f"  bridge: R^2={fit_br['r2']:.4f}, psi={fit_br['psi']:.3f}, d={fit_br['d']:.4f}", flush=True)

    stat_results[stat_name] = {
        "n_points": len(all_rows),
        "data": all_rows,
        "additive_log": fit_add,
        "divisive_sqrt": fit_sqrtK,
        "free_power": fit_free,
        "bridge": fit_br,
        "best_gamma": float(best_gamma),
    }


# ============================================================
# PART 2: REAL DATA MIXED-EFFECTS FIT (with dataset random intercepts)
# ============================================================

print("\n" + "=" * 70)
print("PART 2: REAL DATA MIXED-EFFECTS FIT (dataset intercepts)")
print("=" * 70)

# Load real data
real_points = []
D_HIDDEN = {
    "EleutherAI/pythia-160m": 768,
    "EleutherAI/pythia-410m": 1024,
    "EleutherAI/pythia-1b": 2048,
    "EleutherAI/pythia-1.4b": 2048,
    "EleutherAI/pythia-2.8b": 2560,
    "HuggingFaceTB/SmolLM2-360M": 960,
    "Qwen/Qwen2-0.5B": 896,
    "Qwen/Qwen3-0.6B": 1024,
    "state-spaces/mamba-130m-hf": 768,
    "state-spaces/mamba-370m-hf": 1024,
    "state-spaces/mamba-790m-hf": 1536,
    "state-spaces/mamba-2.8b-hf": 2560,
    "tiiuae/falcon-h1-0.5b-base": 1024,
}

# Load from spectral collapse
try:
    sc = json.load(open(RESULTS_DIR / "cti_spectral_collapse.json"))
    for p in sc.get("all_points", []):
        K = p.get("K", 150)
        knn = p.get("knn", 0)
        kappa = p.get("kappa", 0)
        dataset = p.get("dataset", "unknown")
        q = (knn - 1.0 / K) / (1.0 - 1.0 / K)
        if kappa > 0 and q > 0:
            real_points.append({
                "kappa": float(kappa), "K": float(K), "q": float(q),
                "dataset": dataset, "source": "spectral_collapse",
            })
    print(f"Loaded {len(real_points)} points from spectral_collapse")
except Exception as e:
    print(f"spectral_collapse load failed: {e}")

# Load from prospective kappa
try:
    pk = json.load(open(RESULTS_DIR / "cti_prospective_kappa.json"))
    for p in pk.get("all_predictions", []):
        K = p.get("K", 150)
        knn = p.get("knn_q", 0)
        kappa = p.get("kappa", 0)
        dataset = p.get("dataset", "unknown")
        if kappa > 0 and knn > 0:
            real_points.append({
                "kappa": float(kappa), "K": float(K), "q": float(knn),
                "dataset": dataset, "source": "prospective_kappa",
            })
    print(f"Total after prospective_kappa: {len(real_points)} points")
except Exception as e:
    print(f"prospective_kappa load failed: {e}")

# Load from multi_obs_sweep
try:
    ms = json.load(open(RESULTS_DIR / "cti_multi_obs_sweep.json"))
    for p in ms.get("all_points", []):
        K = p.get("K", 150)
        knn = p.get("knn", 0)
        kappa = p.get("kappa", 0)
        dataset = p.get("dataset", "unknown")
        q = (knn - 1.0 / K) / (1.0 - 1.0 / K)
        if kappa > 0 and q > 0:
            real_points.append({
                "kappa": float(kappa), "K": float(K), "q": float(q),
                "dataset": dataset, "source": "multi_obs_sweep",
            })
    print(f"Total after multi_obs_sweep: {len(real_points)} points")
except Exception as e:
    print(f"multi_obs_sweep load failed: {e}")

print(f"\nTotal real points: {len(real_points)}")
datasets = list(set(p["dataset"] for p in real_points))
print(f"Datasets: {datasets}")

if len(real_points) >= 20:
    kappa_r = np.array([p["kappa"] for p in real_points])
    K_r = np.array([p["K"] for p in real_points])
    q_r = np.array([p["q"] for p in real_points])
    ds_r = [p["dataset"] for p in real_points]
    ds_idx = {ds: i for i, ds in enumerate(datasets)}
    ds_ids = np.array([ds_idx[d] for d in ds_r])

    # Pooled fits (no intercepts)
    print("\n--- Pooled fits (no dataset intercepts) ---")
    fit_p_add = fit_additive_log(kappa_r, K_r, q_r)
    fit_p_sqrt = fit_divisive_power(kappa_r, K_r, q_r, gamma=0.5)
    fit_p_br = fit_bridge(kappa_r, K_r, q_r)
    print(f"  additive-log: R^2={fit_p_add['r2']:.4f}")
    print(f"  divisive-sqrt: R^2={fit_p_sqrt['r2']:.4f}")
    print(f"  bridge: R^2={fit_p_br['r2']:.4f}, psi={fit_p_br['psi']:.3f}")

    # Mixed-effects: add dataset intercepts
    print("\n--- Mixed-effects fits (with dataset intercepts) ---")
    n_ds = len(datasets)

    def fit_mixed_additive_log(kappa, K, q, ds_ids, n_ds):
        """logit(q) = a*kappa - b*log(K-1) + alpha_ds."""
        q_clip = np.clip(q, 0.001, 0.999)
        logit_q = sp_logit(q_clip)
        log_K = np.log(np.maximum(K - 1, 1))

        def loss(params):
            a = params[0]
            b = params[1]
            alphas = params[2:]
            pred = a * kappa - b * log_K + alphas[ds_ids]
            return np.mean((logit_q - pred) ** 2)

        x0 = [10.0, 1.0] + [0.0] * n_ds
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
        params = res.x
        a, b = params[0], params[1]
        alphas = params[2:]
        pred = a * kappa - b * log_K + alphas[ds_ids]
        ss_res = np.sum((logit_q - pred) ** 2)
        ss_tot = np.sum((logit_q - logit_q.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        mae = np.mean(np.abs(sigmoid(pred) - q))
        return {"model": "mixed_additive_log", "r2": float(r2), "mae": float(mae),
                "a": float(a), "b": float(b), "dataset_intercepts": dict(zip(datasets, alphas.tolist()))}

    def fit_mixed_sqrt(kappa, K, q, ds_ids, n_ds):
        """logit(q) = a*kappa/sqrt(K) + alpha_ds."""
        q_clip = np.clip(q, 0.001, 0.999)
        x = kappa / np.sqrt(K)
        logit_q = sp_logit(q_clip)

        def loss(params):
            a = params[0]
            alphas = params[1:]
            pred = a * x + alphas[ds_ids]
            return np.mean((logit_q - pred) ** 2)

        x0 = [10.0] + [0.0] * n_ds
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 15000, "xatol": 1e-7})
        params = res.x
        a = params[0]
        alphas = params[1:]
        pred = a * x + alphas[ds_ids]
        ss_res = np.sum((logit_q - pred) ** 2)
        ss_tot = np.sum((logit_q - logit_q.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        mae = np.mean(np.abs(sigmoid(pred) - q))
        return {"model": "mixed_sqrt", "r2": float(r2), "mae": float(mae),
                "a": float(a), "dataset_intercepts": dict(zip(datasets, alphas.tolist()))}

    fit_m_add = fit_mixed_additive_log(kappa_r, K_r, q_r, ds_ids, n_ds)
    fit_m_sqrt = fit_mixed_sqrt(kappa_r, K_r, q_r, ds_ids, n_ds)
    print(f"  mixed additive-log: R^2={fit_m_add['r2']:.4f}, b={fit_m_add['b']:.3f}")
    print(f"  mixed sqrt(K): R^2={fit_m_sqrt['r2']:.4f}")
    print(f"  DELTA R^2 (bridge - additive): {fit_p_br['r2'] - fit_p_add['r2']:.4f}")
    print(f"  Mixed R^2 gain (additive-log): {fit_m_add['r2'] - fit_p_add['r2']:.4f}")

    real_results = {
        "n_points": len(real_points),
        "datasets": datasets,
        "pooled": {
            "additive_log": fit_p_add,
            "divisive_sqrt": fit_p_sqrt,
            "bridge": fit_p_br,
        },
        "mixed_effects": {
            "additive_log": fit_m_add,
            "divisive_sqrt": fit_m_sqrt,
        }
    }
else:
    print("Not enough real data points, skipping mixed-effects fit")
    real_results = {"n_points": len(real_points)}


# ============================================================
# VERDICT
# ============================================================

print("\n" + "=" * 70)
print("RECONCILIATION VERDICT")
print("=" * 70)

print("\nSynthetic statistic ablation (which model wins for each statistic?):")
for stat_name, res in stat_results.items():
    best_r2 = max(res["additive_log"]["r2"], res["divisive_sqrt"]["r2"], res["bridge"]["r2"])
    winner = "additive_log" if res["additive_log"]["r2"] == best_r2 else (
        "divisive_sqrt" if res["divisive_sqrt"]["r2"] == best_r2 else "bridge"
    )
    print(f"  {stat_name:>12}: winner={winner} (R^2={best_r2:.4f}), "
          f"add-log R^2={res['additive_log']['r2']:.4f} (b={res['additive_log']['b']:.2f}), "
          f"div-sqrt R^2={res['divisive_sqrt']['r2']:.4f}, "
          f"best_gamma={res['best_gamma']:.2f}")

if real_results.get("n_points", 0) >= 20:
    print(f"\nReal data pooled: add-log R^2={real_results['pooled']['additive_log']['r2']:.4f}, "
          f"div-sqrt R^2={real_results['pooled']['divisive_sqrt']['r2']:.4f}")
    print(f"Real data mixed: add-log R^2={real_results['mixed_effects']['additive_log']['r2']:.4f}, "
          f"div-sqrt R^2={real_results['mixed_effects']['divisive_sqrt']['r2']:.4f}")

# Key test: does 1-NN have b~1 and 5-NN have different b?
if "1nn_l2" in stat_results and "5nn_l2" in stat_results:
    b_1nn = stat_results["1nn_l2"]["additive_log"]["b"]
    b_5nn = stat_results["5nn_l2"]["additive_log"]["b"]
    print(f"\nKEY TEST: b for 1-NN = {b_1nn:.3f} (theory: 1.0)")
    print(f"          b for 5-NN = {b_5nn:.3f} (expect different)")
    print(f"          Confirms statistic-dependent K-scaling: {abs(b_1nn - b_5nn) > 0.3}")

# Save
out = {
    "statistic_ablation": stat_results,
    "real_data": real_results,
}
out_path = RESULTS_DIR / "cti_reconciliation_bridge.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved: {out_path.name}")
