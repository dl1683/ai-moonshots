#!/usr/bin/env python -u
"""
Phi-LOAO Test: Does phi(tau*=0.2) have lower CV across architectures?
=======================================================================
HYPOTHESIS: The upgraded law logit(q) = A_phi * phi(tau=0.2) + C
has A_phi with LOWER CV across 5 architectures than alpha_kappa_nearest.

Current: alpha_kappa CV=2.3% (12 models), first confirmed on 5 models.

If CV(A_phi) < CV(alpha_kappa_5models), the upgraded law is MORE UNIVERSAL.

Also tests: A_phi = sqrt(4/pi) = 1.128 (renormalized universal constant)?

DESIGN:
- 5 model architectures with cached embeddings
- Per-class phi(tau=0.2) and logit(q_ci) computed
- Per-model free-slope regression: logit(q_ci) = A_phi * phi_i + C_model
- CV = std(A_phi)/mean(A_phi) across 5 models
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
OUT_JSON = Path("results/cti_phi_loao.json")
K = 14
TAU_STAR = 0.2  # best tau from phi_upgrade_pooled
A_RENORM = 1.0535  # pre-registered universal constant


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
    return kappas


def phi_tau(kappas, tau):
    kappas = np.array(kappas)
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


def per_model_fit(kj1_list, phi_list, lq_list):
    """Fit per-model slope for kappa_nearest and phi."""
    kj1 = np.array(kj1_list, dtype=float)
    phi = np.array(phi_list, dtype=float)
    lq = np.array(lq_list, dtype=float)
    valid = np.isfinite(kj1) & np.isfinite(phi) & np.isfinite(lq)
    kj1, phi, lq = kj1[valid], phi[valid], lq[valid]

    if len(kj1) < 4:
        return None, None, None, None

    def linfit(xs, ys):
        if xs.std() < 1e-8 or ys.std() < 1e-8:
            return None, None, None
        slope, intercept, r, _, _ = linregress(xs, ys)
        return float(slope), float(intercept), float(r)

    slope_kj1, _, r_kj1 = linfit(kj1, lq)
    slope_phi, _, r_phi = linfit(phi, lq)
    return slope_kj1, r_kj1, slope_phi, r_phi


def main():
    print("=" * 70)
    print("PHI-LOAO: Does phi(tau=0.2) have lower CV across architectures?")
    print(f"tau* = {TAU_STAR}, A_RENORM = {A_RENORM}")
    print("=" * 70)

    per_model_results = {}
    alphas_kj1 = []
    alphas_phi = []

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

        centroids, sigma_W = compute_class_stats(X, y)

        kj1_list, phi_list, lq_list = [], [], []
        for ci in classes:
            kappas = compute_all_kappas(centroids, sigma_W, d, ci)
            if not kappas:
                continue
            kj1 = kappas[0]
            phi_v = phi_tau(kappas, TAU_STAR)
            q = compute_per_class_q(X, y, ci)
            if q is None:
                continue
            lq = safe_logit(q)
            kj1_list.append(kj1)
            phi_list.append(phi_v)
            lq_list.append(lq)

        slope_kj1, r_kj1, slope_phi, r_phi = per_model_fit(kj1_list, phi_list, lq_list)

        if slope_kj1 is not None:
            alphas_kj1.append(slope_kj1)
            alphas_phi.append(slope_phi)
            print(f"\n{model_name}:")
            print(f"  kappa_nearest: alpha={slope_kj1:.4f}, r={r_kj1:.4f}")
            print(f"  phi(tau=0.2):  alpha={slope_phi:.4f}, r={r_phi:.4f}")
            per_model_results[model_name] = {
                "alpha_kappa": slope_kj1, "r_kappa": r_kj1,
                "alpha_phi": slope_phi, "r_phi": r_phi,
            }

    print(f"\n{'='*70}")
    print("LOAO UNIVERSALITY COMPARISON (5 models)")

    alphas_kj1 = np.array(alphas_kj1)
    alphas_phi = np.array(alphas_phi)

    mean_kj1 = float(alphas_kj1.mean())
    std_kj1 = float(alphas_kj1.std())
    cv_kj1 = float(std_kj1 / abs(mean_kj1)) if abs(mean_kj1) > 1e-8 else None

    mean_phi = float(alphas_phi.mean())
    std_phi = float(alphas_phi.std())
    cv_phi = float(std_phi / abs(mean_phi)) if abs(mean_phi) > 1e-8 else None

    print(f"\nkappa_nearest alphas: {alphas_kj1.round(4).tolist()}")
    print(f"  mean={mean_kj1:.4f}, std={std_kj1:.4f}, CV={cv_kj1:.4f}")

    print(f"\nphi(tau=0.2) alphas: {alphas_phi.round(4).tolist()}")
    print(f"  mean={mean_phi:.4f}, std={std_phi:.4f}, CV={cv_phi:.4f}")

    phi_lower_cv = (cv_phi is not None and cv_kj1 is not None and cv_phi < cv_kj1)
    print(f"\nPhi has LOWER CV: {'PASS' if phi_lower_cv else 'FAIL'}")
    print(f"  CV improvement: {cv_kj1:.4f} -> {cv_phi:.4f}")

    # Test A_phi = A_RENORM?
    a_phi_test = abs(mean_phi - A_RENORM) / A_RENORM
    print(f"\nA_phi = A_RENORM test: A_phi={mean_phi:.4f}, A_RENORM={A_RENORM}")
    print(f"  Error: {a_phi_test*100:.1f}% (pass if <10%)")

    result = {
        "experiment": "phi_loao",
        "tau_star": TAU_STAR,
        "A_renorm": A_RENORM,
        "n_models": len(per_model_results),
        "per_model": per_model_results,
        "loao_kappa_nearest": {
            "alphas": alphas_kj1.tolist(),
            "mean": mean_kj1, "std": std_kj1, "cv": cv_kj1,
        },
        "loao_phi": {
            "alphas": alphas_phi.tolist(),
            "mean": mean_phi, "std": std_phi, "cv": cv_phi,
        },
        "phi_lower_cv_pass": phi_lower_cv,
        "a_phi_test_pass": a_phi_test < 0.10,
        "a_phi_error_pct": float(a_phi_test * 100),
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
