#!/usr/bin/env python -u
"""
Kernel Showdown: What is the Correct Competition Field Form? (Session 40)
=========================================================================
Codex recommendation: pre-registered nested LOAO kernel showdown.

Tests 7 candidate kernels against the kappa_nearest baseline:
  M0: kappa_nearest (baseline)
  M1: phi_exp(tau)         -- current theory: -tau*log(sum exp(-kappa_j/tau))
  M2: kappa_mean           -- uniform average of all K-1 kappas
  M3: top2_mean            -- (kappa_j1 + kappa_j2) / 2
  M4: power_law(p)         -- sum_r (1/r^p) * kappa_jr, p tuned via LOAO
  M5: top_k(k)             -- mean of top-k nearest kappas, k in {2,3,5} via LOAO
  M6: stretched_exp(a,b)   -- sum_r exp(-a*(r-1)^b), a,b tuned via LOAO

Protocol (same as phi-LOAO v3):
  - Outer: LOAO by architecture
  - Inner: find best params on training archs (within-arch R2)
  - Eval: within test-arch R2 with per-arch slope+intercept
  - Random null: 1000 random monotone weights (same as before)

Result: which kernel wins? Is any uniquely privileged vs random monotone?
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_JSON = RESULTS_DIR / "cti_kernel_showdown.json"

BOOTSTRAP_N = 500
RNG_SEED = 42
N_RANDOM_NULL = 500  # smaller for speed

ARCH_CACHES = {
    "pythia-160m":   "dointerv_multi_pythia-160m_l12.npz",
    "pythia-410m":   "dointerv_multi_pythia-410m_l3.npz",
    "electra-small": "dointerv_multi_electra-small_l3.npz",
    "bert-base":     "dointerv_multi_bert-base-uncased_l10.npz",
    "rwkv-4-169m":   "dointerv_multi_rwkv-4-169m_l12.npz",
}

TAU_GRID = np.logspace(np.log10(0.02), np.log10(2.0), 41)
POWER_GRID = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
TOPK_GRID = [2, 3, 4, 5, 7, 10]
A_GRID = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0])
B_GRID = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0])

N_CV_Q = 5


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
            "kappa_nearest": float(kappas[0][0]),
            "kappas": np.array([k for k, _ in kappas]),
            "logit_q": safe_logit(q),
        })

    print(f"  {cache_path.name}: N={len(X)}, valid={len(rows)}")
    return rows


# ---- Feature functions ----
def feat_phi_exp(rows, tau):
    def f(kv):
        k0 = kv[0]
        return float(-tau * (-k0 / tau + np.log(np.sum(np.exp(-(kv - k0) / tau)))))
    return np.array([f(r["kappas"]) for r in rows])


def feat_mean(rows):
    return np.array([float(np.mean(r["kappas"])) for r in rows])


def feat_top2(rows):
    return np.array([float(np.mean(r["kappas"][:2])) for r in rows])


def feat_power(rows, p):
    def f(kv):
        rs = np.arange(1, len(kv) + 1, dtype=float)
        w = 1.0 / rs**p
        return float(np.dot(w, kv) / w.sum())
    return np.array([f(r["kappas"]) for r in rows])


def feat_topk(rows, k):
    return np.array([float(np.mean(r["kappas"][:k])) for r in rows])


def feat_stretched(rows, a, b):
    def f(kv):
        rs = np.arange(len(kv), dtype=float)
        w = np.exp(-a * rs**b)
        return float(np.dot(w, kv) / (w.sum() + 1e-10))
    return np.array([f(r["kappas"]) for r in rows])


# ---- R2 ----
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


def find_best_tau(tr_rows_list):
    best, bv = float(TAU_GRID[0]), -1e9
    for tau in TAU_GRID:
        r2s = [r2_within(feat_phi_exp(rows, tau), [r["logit_q"] for r in rows]) for rows in tr_rows_list]
        v = float(np.mean(r2s))
        if v > bv:
            bv, best = v, float(tau)
    return best


def find_best_power(tr_rows_list):
    best, bv = float(POWER_GRID[0]), -1e9
    for p in POWER_GRID:
        r2s = [r2_within(feat_power(rows, p), [r["logit_q"] for r in rows]) for rows in tr_rows_list]
        v = float(np.mean(r2s))
        if v > bv:
            bv, best = v, float(p)
    return best


def find_best_topk(tr_rows_list):
    best, bv = TOPK_GRID[0], -1e9
    for k in TOPK_GRID:
        r2s = [r2_within(feat_topk(rows, k), [r["logit_q"] for r in rows]) for rows in tr_rows_list]
        v = float(np.mean(r2s))
        if v > bv:
            bv, best = v, k
    return best


def find_best_stretched(tr_rows_list):
    best_a, best_b, bv = float(A_GRID[0]), float(B_GRID[0]), -1e9
    for a in A_GRID:
        for b in B_GRID:
            r2s = [r2_within(feat_stretched(rows, a, b), [r["logit_q"] for r in rows]) for rows in tr_rows_list]
            v = float(np.mean(r2s))
            if v > bv:
                bv, best_a, best_b = v, float(a), float(b)
    return best_a, best_b


def json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    print("=" * 70)
    print("KERNEL SHOWDOWN: FINDING THE CORRECT COMPETITION FIELD")
    print("=" * 70)

    arch_rows = {}
    for name, cname in ARCH_CACHES.items():
        cp = RESULTS_DIR / cname
        if not cp.exists():
            print(f"  MISSING: {cname}")
            continue
        arch_rows[name] = load_arch(cp)

    arch_names = list(arch_rows.keys())
    N_ARCH = len(arch_names)
    print(f"\n{N_ARCH} architectures: {arch_names}")

    # Store per-kernel Delta_R2 per arch
    kernel_names = ["kappa_nearest", "phi_exp", "kappa_mean", "top2",
                    "power_law", "top_k", "stretched_exp"]
    kernel_results = {k: {"per_arch": {}, "pooled_dr2": None, "ci": None, "null_pct": None}
                      for k in kernel_names}

    all_pointwise = {k: [] for k in kernel_names}

    print("\n" + "=" * 70)
    print("LOAO FOLDS")
    print("=" * 70)

    for held_out in arch_names:
        tr_archs = [a for a in arch_names if a != held_out]
        tr_rows_list = [arch_rows[a] for a in tr_archs]
        te_rows = arch_rows[held_out]

        print(f"\n  Held-out: {held_out}")

        # Find params on training archs
        tau_star = find_best_tau(tr_rows_list)
        power_star = find_best_power(tr_rows_list)
        k_star = find_best_topk(tr_rows_list)
        a_star, b_star = find_best_stretched(tr_rows_list)

        print(f"    tau*={tau_star:.3f}, power*={power_star:.2f}, k*={k_star}, a*={a_star:.2f}, b*={b_star:.2f}")

        te_lq = np.array([r["logit_q"] for r in te_rows])

        feats = {
            "kappa_nearest": np.array([r["kappa_nearest"] for r in te_rows]),
            "phi_exp":        feat_phi_exp(te_rows, tau_star),
            "kappa_mean":     feat_mean(te_rows),
            "top2":           feat_top2(te_rows),
            "power_law":      feat_power(te_rows, power_star),
            "top_k":          feat_topk(te_rows, k_star),
            "stretched_exp":  feat_stretched(te_rows, a_star, b_star),
        }

        r2s = {k: r2_within(v, te_lq) for k, v in feats.items()}
        r2_base = r2s["kappa_nearest"]

        for k in kernel_names:
            dr2 = float(r2s[k] - r2_base)
            kernel_results[k]["per_arch"][held_out] = {"r2": float(r2s[k]), "delta_r2": dr2}

            # Point-wise for CI
            x_base = feats["kappa_nearest"].reshape(-1, 1)
            x_k = feats[k].reshape(-1, 1)
            lr_base = LinearRegression().fit(x_base, te_lq)
            lr_k = LinearRegression().fit(x_k, te_lq)
            e_base = (te_lq - lr_base.predict(x_base))**2
            e_k = (te_lq - lr_k.predict(x_k))**2
            all_pointwise[k].extend((e_base - e_k).tolist())

        print(f"    R2: base={r2_base:.3f}", end="")
        for k in kernel_names[1:]:
            print(f", {k}={r2s[k]:.3f}", end="")
        print()

    # Pooled stats + bootstrap CI
    rng = np.random.default_rng(RNG_SEED)
    for k in kernel_names:
        dr2s = [v["delta_r2"] for v in kernel_results[k]["per_arch"].values()]
        pooled = float(np.mean(dr2s))
        pw = np.array(all_pointwise[k])
        boots = [float(np.mean(rng.choice(pw, size=len(pw), replace=True))) for _ in range(BOOTSTRAP_N)]
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))
        kernel_results[k]["pooled_dr2"] = pooled
        kernel_results[k]["ci"] = [ci_lo, ci_hi]
        kernel_results[k]["n_pos"] = int(sum(1 for d in dr2s if d > 0))

    # Random null
    print("\n" + "=" * 70)
    print("RANDOM NULL")
    print("=" * 70)
    rng2 = np.random.default_rng(RNG_SEED + 1)
    K_minus_1 = len(arch_rows[arch_names[0]][0]["kappas"])

    null_pooled = []
    for i_null in range(N_RANDOM_NULL):
        raw = rng2.exponential(1.0, size=K_minus_1)
        w_null = np.sort(raw)[::-1]
        w_null = w_null / w_null.sum()

        fold_deltas = []
        for held_out in arch_names:
            te_rows = arch_rows[held_out]
            te_lq = np.array([r["logit_q"] for r in te_rows])

            def null_f(rows):
                res = []
                for r in rows:
                    kv = r["kappas"]
                    n = min(len(kv), len(w_null))
                    res.append(float(np.dot(w_null[:n], kv[:n])))
                return np.array(res)

            te_null = null_f(te_rows)
            te_kappa = np.array([r["kappa_nearest"] for r in te_rows])
            r2_null = r2_within(te_null.tolist(), te_lq.tolist())
            r2_base = r2_within(te_kappa.tolist(), te_lq.tolist())
            fold_deltas.append(r2_null - r2_base)

        null_pooled.append(float(np.mean(fold_deltas)))
        if i_null % 100 == 0:
            print(f"  null {i_null}/{N_RANDOM_NULL}, mean={float(np.mean(null_pooled)):.4f}")

    null_arr = np.array(null_pooled)
    null_mean = float(null_arr.mean())
    null_p90 = float(np.percentile(null_arr, 90))
    print(f"Null: mean={null_mean:.4f}, p90={null_p90:.4f}")

    for k in kernel_names:
        pct = float(np.mean(kernel_results[k]["pooled_dr2"] >= null_arr) * 100)
        kernel_results[k]["null_pct"] = pct

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Random null: mean={null_mean:.4f}, p90={null_p90:.4f}")
    print(f"{'Kernel':<20} {'pooled_dR2':>12} {'CI_lo':>8} {'CI_hi':>8} {'n_pos':>6} {'null_pct':>10}")
    for k in kernel_names:
        kr = kernel_results[k]
        ci = kr["ci"]
        print(f"{k:<20} {kr['pooled_dr2']:>12.4f} {ci[0]:>8.4f} {ci[1]:>8.4f} {kr['n_pos']:>6} {kr['null_pct']:>10.1f}th")

    out = {
        "experiment": "kernel_showdown",
        "session": 40,
        "n_arch": N_ARCH,
        "architectures": arch_names,
        "null_mean": null_mean,
        "null_p90": null_p90,
        "kernels": {
            k: {
                "pooled_dr2": kernel_results[k]["pooled_dr2"],
                "ci_95": kernel_results[k]["ci"],
                "n_pos": kernel_results[k]["n_pos"],
                "null_pct": kernel_results[k]["null_pct"],
                "per_arch": kernel_results[k]["per_arch"],
            }
            for k in kernel_names
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
