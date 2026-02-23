#!/usr/bin/env python -u
"""
phi-LOAO v3: Within-Architecture R2, Tau LOAO (Session 40)
============================================================
Fix for v2: use within-arch R2 (not cross-arch linear transfer).

The ONLY quantity held out is tau*:
  - tau* is optimized on K-1 training architectures
  - Within each test architecture, fit per-arch linear model
    (slope + intercept are free, tau is frozen from training archs)
  - Compare R2(phi(tau*)) vs R2(kappa_nearest) within test arch

This isolates tau as the single held-out quantity and avoids scale mismatch.
This matches the cti_kappa_eff_held_out.py design (frozen w_r from calib arch).

Success criteria (pre-registered):
  PRIMARY: pooled Delta_R2 >= +0.05 with 95% bootstrap CI > 0
  ARCH: positive Delta_R2 in >= 3/5 architectures
  NULL: phi at >= 90th percentile of random monotone null
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_JSON = RESULTS_DIR / "cti_phi_loao_v3.json"

DELTA_R2_THRESHOLD = 0.05
MIN_ARCH_PASS = 3
BOOTSTRAP_N = 1000
RNG_SEED = 42
N_RANDOM_NULL = 1000

TAU_GRID = np.logspace(np.log10(0.02), np.log10(2.0), 41)

ARCH_CACHES = {
    "pythia-160m":   "dointerv_multi_pythia-160m_l12.npz",
    "pythia-410m":   "dointerv_multi_pythia-410m_l3.npz",
    "electra-small": "dointerv_multi_electra-small_l3.npz",
    "bert-base":     "dointerv_multi_bert-base-uncased_l10.npz",
    "rwkv-4-169m":   "dointerv_multi_rwkv-4-169m_l12.npz",
}

N_CV_Q = 5


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


def compute_all_kappas(centroids, sigma_W, d, ci):
    mu_i = centroids[ci]
    kappas = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        k = dist / (sigma_W * np.sqrt(d) + 1e-10)
        kappas.append((k, cj))
    kappas.sort()
    return kappas


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


def phi_tau_val(kappas_sorted, tau):
    kv = np.array([k for k, _ in kappas_sorted])
    if tau < 1e-10:
        return float(kv[0])
    k0 = kv[0]
    log_sum = float(-k0 / tau + np.log(np.sum(np.exp(-(kv - k0) / tau))))
    return float(-tau * log_sum)


def load_arch(cache_path):
    print(f"  Loading {cache_path.name}...")
    data = np.load(str(cache_path))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    K = len(classes)
    centroids, sigma_W = compute_class_stats(X, y)

    rows = []
    for ci in classes:
        kappas = compute_all_kappas(centroids, sigma_W, d, ci)
        if len(kappas) < 2:
            continue
        q = compute_per_class_q(X, y, ci)
        if q is None:
            continue
        rows.append({
            "ci": int(ci),
            "kappa_nearest": float(kappas[0][0]),
            "kappas_sorted": kappas,
            "logit_q": safe_logit(q),
        })

    print(f"    N={len(X)}, d={d}, K={K}, valid={len(rows)}")
    return rows


def make_phi_features(rows, tau):
    return np.array([phi_tau_val(r["kappas_sorted"], tau) for r in rows])


def r2_within_arch(feat, logit_qs):
    """Compute R2 with per-arch linear fit (slope + intercept free)."""
    x = np.array(feat).reshape(-1, 1)
    y = np.array(logit_qs)
    if len(x) < 4 or float(np.std(x)) < 1e-10:
        return 0.0
    lr = LinearRegression().fit(x, y)
    ss_res = float(np.sum((y - lr.predict(x))**2))
    ss_tot = float(np.sum((y - float(np.mean(y)))**2))
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def find_best_tau(train_rows_list):
    """Find tau that maximizes mean within-arch R2 across training architectures."""
    best_tau = float(TAU_GRID[0])
    best_r2 = -1e9
    for tau in TAU_GRID:
        r2s = []
        for rows in train_rows_list:
            phi = make_phi_features(rows, tau)
            lq = np.array([r["logit_q"] for r in rows])
            r2s.append(r2_within_arch(phi.tolist(), lq.tolist()))
        avg = float(np.mean(r2s))
        if avg > best_r2:
            best_r2 = avg
            best_tau = float(tau)
    return best_tau, best_r2


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
    print("phi-LOAO v3: WITHIN-ARCH R2, TAU LOAO")
    print("Frozen: tau (from training archs); Free: per-arch slope+intercept")
    print(f"Criteria: Delta_R2 >= {DELTA_R2_THRESHOLD}, CI>0, pos in >= {MIN_ARCH_PASS}/5")
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
    if N_ARCH < 3:
        print("ERROR: need >= 3")
        return

    print("\n" + "=" * 70)
    print("LOAO FOLDS")
    print("=" * 70)

    per_arch = {}
    all_pointwise = []  # per-class squared-error improvement

    for held_out in arch_names:
        tr_archs = [a for a in arch_names if a != held_out]
        tr_rows_list = [arch_rows[a] for a in tr_archs]
        te_rows = arch_rows[held_out]

        # Inner: find tau* on training archs
        tau_star, tau_train_r2 = find_best_tau(tr_rows_list)
        print(f"\n  Held-out: {held_out}")
        print(f"    tau* (train) = {tau_star:.4f}, train_R2={tau_train_r2:.4f}")

        # Evaluate WITHIN test arch
        te_kappa = np.array([r["kappa_nearest"] for r in te_rows])
        te_logit = np.array([r["logit_q"] for r in te_rows])
        te_phi = make_phi_features(te_rows, tau_star)

        r2_m0 = r2_within_arch(te_kappa.tolist(), te_logit.tolist())
        r2_m1 = r2_within_arch(te_phi.tolist(), te_logit.tolist())
        dr2 = float(r2_m1 - r2_m0)

        print(f"    R2(kappa_nearest) = {r2_m0:.4f}")
        print(f"    R2(phi_tau*) = {r2_m1:.4f}")
        print(f"    Delta_R2 = {dr2:+.4f}")

        # Point-wise squared error improvement
        lr0 = LinearRegression().fit(te_kappa.reshape(-1, 1), te_logit)
        lr1 = LinearRegression().fit(te_phi.reshape(-1, 1), te_logit)
        e0 = (te_logit - lr0.predict(te_kappa.reshape(-1, 1)))**2
        e1 = (te_logit - lr1.predict(te_phi.reshape(-1, 1)))**2
        all_pointwise.extend((e0 - e1).tolist())

        per_arch[held_out] = {
            "tau_star": float(tau_star),
            "r2_m0": float(r2_m0),
            "r2_m1": float(r2_m1),
            "delta_r2": float(dr2),
            "n_test": len(te_rows),
        }

    dr2_list = [v["delta_r2"] for v in per_arch.values()]
    pooled_m0 = float(np.mean([v["r2_m0"] for v in per_arch.values()]))
    pooled_m1 = float(np.mean([v["r2_m1"] for v in per_arch.values()]))
    pooled_dr2 = float(np.mean(dr2_list))
    n_pos = int(sum(1 for d in dr2_list if d > 0))
    tau_vals = [v["tau_star"] for v in per_arch.values()]
    tau_cv = float(np.std(tau_vals) / (float(np.mean(tau_vals)) + 1e-10))

    # Bootstrap CI
    rng = np.random.default_rng(RNG_SEED)
    pw = np.array(all_pointwise)
    boot_means = [float(np.mean(rng.choice(pw, size=len(pw), replace=True))) for _ in range(BOOTSTRAP_N)]
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    # Random null: same LOAO protocol but with random monotone weights
    print("\n" + "=" * 70)
    print("RANDOM MONOTONE NULL")
    print("=" * 70)
    rng2 = np.random.default_rng(RNG_SEED + 1)
    K_minus_1 = len(arch_rows[arch_names[0]][0]["kappas_sorted"])

    null_pooled = []
    for i_null in range(N_RANDOM_NULL):
        raw = rng2.exponential(1.0, size=K_minus_1)
        w_null = np.sort(raw)[::-1]
        w_null = w_null / w_null.sum()

        fold_deltas = []
        for held_out in arch_names:
            te_rows = arch_rows[held_out]

            def null_f(rows):
                res = []
                for r in rows:
                    kv = np.array([k for k, _ in r["kappas_sorted"]])
                    n = min(len(kv), len(w_null))
                    res.append(float(np.dot(w_null[:n], kv[:n])))
                return np.array(res)

            te_kappa_n = np.array([r["kappa_nearest"] for r in te_rows])
            te_logit_n = np.array([r["logit_q"] for r in te_rows])
            te_nf = null_f(te_rows)

            r2_null = r2_within_arch(te_nf.tolist(), te_logit_n.tolist())
            r2_base = r2_within_arch(te_kappa_n.tolist(), te_logit_n.tolist())
            fold_deltas.append(r2_null - r2_base)

        null_pooled.append(float(np.mean(fold_deltas)))
        if i_null % 200 == 0:
            print(f"  null {i_null}/{N_RANDOM_NULL}, mean={float(np.mean(null_pooled)):.4f}")

    null_arr = np.array(null_pooled)
    phi_pct = float(np.mean(pooled_dr2 >= null_arr) * 100)
    null_p90 = float(np.percentile(null_arr, 90))
    null_mean = float(null_arr.mean())

    print(f"\nNull: mean={null_mean:.4f}, p90={null_p90:.4f}")
    print(f"phi pct vs null: {phi_pct:.1f}th")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Pooled R2(M0) = {pooled_m0:.4f}")
    print(f"Pooled R2(M1) = {pooled_m1:.4f}")
    print(f"Pooled Delta_R2 = {pooled_dr2:+.4f} (need +{DELTA_R2_THRESHOLD})")
    print(f"95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"Positive in {n_pos}/{N_ARCH} archs (need {MIN_ARCH_PASS})")
    print(f"tau_cv = {tau_cv:.3f}, tau range: [{min(tau_vals):.3f}, {max(tau_vals):.3f}]")
    for a, v in per_arch.items():
        print(f"  {a}: tau*={v['tau_star']:.3f}, R2(M0)={v['r2_m0']:.4f}, R2(M1)={v['r2_m1']:.4f}, dR2={v['delta_r2']:+.4f}")

    pass_primary = bool(pooled_dr2 >= DELTA_R2_THRESHOLD and ci_lo > 0)
    pass_arch = bool(n_pos >= MIN_ARCH_PASS)
    pass_null = bool(phi_pct >= 90.0)
    print(f"\nPASS primary (dR2>={DELTA_R2_THRESHOLD}, CI>0): {'PASS' if pass_primary else 'FAIL'}")
    print(f"PASS arch (>={MIN_ARCH_PASS}/5 pos): {'PASS' if pass_arch else 'FAIL'}")
    print(f"PASS null (>=90th): {'PASS' if pass_null else 'FAIL'}")

    out = {
        "experiment": "phi_loao_v3",
        "session": 40,
        "design": "within_arch_R2_tau_LOAO",
        "n_arch": N_ARCH,
        "architectures": arch_names,
        "pooled_r2_m0": pooled_m0,
        "pooled_r2_m1": pooled_m1,
        "pooled_delta_r2": pooled_dr2,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "n_arch_positive_delta": n_pos,
        "tau_cv": tau_cv,
        "tau_range": [float(min(tau_vals)), float(max(tau_vals))],
        "pass_primary": pass_primary,
        "pass_arch": pass_arch,
        "phi_pct_vs_null": phi_pct,
        "null_mean": null_mean,
        "null_p90": null_p90,
        "pass_null": pass_null,
        "per_arch": per_arch,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
