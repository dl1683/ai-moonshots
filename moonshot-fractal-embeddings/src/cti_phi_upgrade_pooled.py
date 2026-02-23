#!/usr/bin/env python -u
"""
Phi Upgrade: Pooled Multi-Model Test
=====================================
Tests kappa_mean vs kappa_nearest across 5 models * 14 classes = 70 data points.
Much higher statistical power than single-model (14-point) test.

KEY QUESTION: Does phi(tau) / kappa_mean improve cross-class logit(q) prediction
ACROSS MULTIPLE ARCHITECTURES pooled?

DESIGN: For each (model, class) pair, compute:
- kappa_j1: nearest competitor distance (current law)
- kappa_mean: mean of all K-1 competitor distances (upgrade candidate)
- phi(tau*): soft-min of competitors (upgrade candidate)
- logit(q_ci): per-class accuracy via 5-fold CV

Regress logit(q) vs each predictor pooled across all models.

PRE-REG: kappa_mean has higher pooled R2 than kappa_nearest.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, linregress

EMBS = {
    "pythia-160m": "results/dointerv_multi_pythia-160m_l12.npz",
    "pythia-410m": "results/dointerv_multi_pythia-410m_l3.npz",
    "electra-small": "results/dointerv_multi_electra-small_l3.npz",
    "rwkv-4-169m": "results/dointerv_multi_rwkv-4-169m_l12.npz",
    "bert-base": "results/dointerv_multi_bert-base-uncased_l10.npz",
}
OUT_JSON = Path("results/cti_phi_upgrade_pooled.json")
K = 14
TAU_RANGE = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9]


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
        kappas.append(dist / (sigma_W * np.sqrt(d) + 1e-10))
    kappas.sort()
    return kappas  # sorted ascending: nearest first


def phi_tau(kappas, tau):
    kappas = np.array(kappas)
    if tau > 1e8:
        return float(np.mean(kappas))
    z = -kappas / tau
    z_max = z.max()
    return float(-tau * (z_max + np.log(np.sum(np.exp(z - z_max)))))


def compute_per_class_q(X, y, ci, n_splits=5):
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
    return float((q_raw - 1.0/K_local) / (1.0 - 1.0/K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1-1e-5))
    return float(np.log(q / (1.0 - q)))


def fit_r2(xs, ys):
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 10 or xs.std() < 1e-8 or ys.std() < 1e-8:
        return 0.0, 0.0, 0.0
    r, _ = pearsonr(xs, ys)
    slope, intercept, _, _, _ = linregress(xs, ys)
    y_pred = slope * xs + intercept
    ss_res = float(np.sum((ys - y_pred)**2))
    ss_tot = float(np.sum((ys - ys.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
    return float(r), float(r2), float(slope)


def main():
    print("=" * 70)
    print("PHI UPGRADE: POOLED 5-MODEL TEST (70 data points)")
    print("=" * 70)

    all_kj1, all_kmean, all_logit_q = [], [], []
    phi_by_tau = {tau: [] for tau in TAU_RANGE}
    records = []

    for model_name, emb_path in EMBS.items():
        path = Path(emb_path)
        if not path.exists():
            print(f"  MISSING: {emb_path}")
            continue
        data = np.load(str(path))
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64)
        d = X.shape[1]
        classes = sorted(np.unique(y).tolist())
        print(f"\nModel: {model_name} (d={d}, K={len(classes)})")

        centroids, sigma_W = compute_class_stats(X, y)

        for ci in classes:
            kappas = compute_all_kappas(centroids, sigma_W, d, ci)
            if not kappas:
                continue
            kj1 = kappas[0]
            kmean = float(np.mean(kappas))
            q = compute_per_class_q(X, y, ci)
            if q is None:
                continue
            lq = safe_logit(q)
            all_kj1.append(kj1)
            all_kmean.append(kmean)
            all_logit_q.append(lq)
            phi_dict = {}
            for tau in TAU_RANGE:
                pv = phi_tau(kappas, tau)
                phi_by_tau[tau].append(pv)
                phi_dict[str(tau)] = pv
            records.append({
                "model": model_name, "class": ci,
                "kappa_j1": kj1, "kappa_mean": kmean, "logit_q": lq
            })
            print(f"  ci={ci}: kj1={kj1:.4f}, kmean={kmean:.4f}, "
                  f"q={q:.4f}, logit={lq:.4f}")

    n = len(all_kj1)
    print(f"\nTotal data points: {n}")

    r_kj1, r2_kj1, s_kj1 = fit_r2(all_kj1, all_logit_q)
    r_kmean, r2_kmean, s_kmean = fit_r2(all_kmean, all_logit_q)

    print(f"\n{'='*70}")
    print(f"BASELINE kappa_nearest:  r={r_kj1:.4f}, R2={r2_kj1:.4f}, slope={s_kj1:.4f}")
    print(f"UPGRADE kappa_mean:      r={r_kmean:.4f}, R2={r2_kmean:.4f}, slope={s_kmean:.4f}")
    print(f"\nPHI(tau) SWEEP:")

    best_tau = None
    best_r2 = -999
    tau_results = {}
    for tau in TAU_RANGE:
        if len(phi_by_tau[tau]) < 10:
            continue
        r_phi, r2_phi, s_phi = fit_r2(phi_by_tau[tau], all_logit_q)
        tau_label = f"{tau:.2f}" if tau < 1e8 else "inf"
        print(f"  tau={tau_label:>8}: r={r_phi:.4f}, R2={r2_phi:.4f}, slope={s_phi:.4f}")
        tau_results[tau_label] = {"r": r_phi, "r2": r2_phi, "slope": s_phi}
        if r2_phi > best_r2:
            best_r2 = r2_phi
            best_tau = tau

    phi_upgrade = best_r2 > r2_kj1
    mean_upgrade = r2_kmean > r2_kj1
    print(f"\n{'='*70}")
    print(f"VERDICT (n={n} data points):")
    print(f"  kappa_mean upgrade: {'PASS' if mean_upgrade else 'FAIL'} "
          f"(R2: {r2_kj1:.4f} -> {r2_kmean:.4f})")
    print(f"  phi(tau*) upgrade:  {'PASS' if phi_upgrade else 'FAIL'} "
          f"(R2: {r2_kj1:.4f} -> {best_r2:.4f}, tau*={best_tau})")

    result = {
        "experiment": "phi_upgrade_pooled",
        "n_models": len(set(r["model"] for r in records)),
        "n_data_points": n,
        "baseline_kappa_nearest": {"r": r_kj1, "r2": r2_kj1, "slope": s_kj1},
        "upgrade_kappa_mean": {"r": r_kmean, "r2": r2_kmean, "slope": s_kmean,
                                "pass": mean_upgrade},
        "tau_sweep": tau_results,
        "best_tau": float(best_tau) if best_tau and best_tau < 1e8 else "inf",
        "best_r2_phi": float(best_r2),
        "phi_upgrade_pass": phi_upgrade,
        "r2_improvement_pct": float((best_r2 - r2_kj1) / max(abs(r2_kj1), 1e-8) * 100),
    }
    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
