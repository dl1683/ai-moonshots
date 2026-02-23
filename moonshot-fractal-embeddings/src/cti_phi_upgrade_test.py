#!/usr/bin/env python -u
"""
Phi (Soft-Minimum) Upgrade Test: kappa_effective = LSE Competition Metric
==========================================================================
HYPOTHESIS: The correct law is NOT logit(q) = A * kappa_nearest + C
but rather: logit(q) = A * kappa_eff + C
where kappa_eff = -tau * log(sum_j exp(-kappa_j/tau))  [soft-minimum over competitors]

This is the Gumbel-race consistent competition metric. In limits:
- tau->0: kappa_eff -> kappa_nearest (nearest-only, recovered single-pair slope)
- tau->inf: kappa_eff -> mean(kappa) (all competitors equally weighted)
- tau = tau*: kappa_eff captures m_eff=1.4 active competitors

CODEX (Session 38): upgrade to multivariate law explains:
- j2 causal (r=0.811 in RCT): kappa_j2 is the second active competitor
- m_eff=1.40: softmax weight w_j2 ≈ 0.40 (tau calibrated to near-tie kappas)
- Correct formula: kappa_eff = kappa_j1 + 0.40*kappa_j2 for DBpedia

TEST:
1. Compute per-class kappa_ij for K=14 DBpedia frozen embeddings
2. Compute phi(tau) = -tau*log(sum_j exp(-kappa_ij/tau)) for tau in {0.01, 0.1, 0.2, 0.5, 1.0, inf}
3. Per-class 5-fold CV recall -> logit(q)
4. Compare R2: phi(tau*) vs kappa_nearest vs kappa_eff=kappa_j1+0.40*kappa_j2
5. Find tau* that maximizes R2
6. Compute w_j = softmax(-kappa_j/tau*) and verify w_j2 ≈ 0.40

PRE-REGISTERED: phi(tau*) has HIGHER R2 than kappa_nearest (upgrade test)
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, linregress
from scipy.special import logit as logit_fn

# ================================================================
# CONFIG
# ================================================================
EMBS_FILE = Path("results/dointerv_multi_pythia-160m_l12.npz")
OUT_JSON = Path("results/cti_phi_upgrade_test.json")
K = 14
N_CV_SPLITS = 5
TAU_RANGE = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9]  # last=inf approx
W_J2_PRED = 0.40  # Codex predicted weight for j2 given near-tie DBpedia kappas


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


def compute_per_class_kappa(centroids, sigma_W, d, ci):
    """Return sorted list of (kappa_j, class_j) ascending."""
    mu_i = centroids[ci]
    ranking = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa = dist / (sigma_W * np.sqrt(d) + 1e-10)
        ranking.append((kappa, cj))
    ranking.sort()
    return ranking  # sorted ascending: nearest first


def phi_tau(kappas, tau):
    """Soft-minimum: -tau * log(sum_j exp(-kappa_j/tau))."""
    kappas = np.array(kappas)
    if tau > 1e8:  # inf limit = mean
        return float(np.mean(kappas))
    # Numerical stability: factor out min
    z = -kappas / tau
    z_max = z.max()
    val = -tau * (z_max + np.log(np.sum(np.exp(z - z_max))))
    return float(val)


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
    """Pearson r and free-slope R2."""
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 4 or xs.std() < 1e-8 or ys.std() < 1e-8:
        return 0.0, 0.0, 0.0, 0.0
    r, _ = pearsonr(xs, ys)
    slope, intercept, _, _, _ = linregress(xs, ys)
    y_pred = slope * xs + intercept
    ss_res = float(np.sum((ys - y_pred)**2))
    ss_tot = float(np.sum((ys - ys.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(r), float(r2), float(slope), float(intercept)


def main():
    print("=" * 70)
    print("PHI UPGRADE TEST: kappa_eff = soft-min vs kappa_nearest")
    print("=" * 70)

    # Load frozen embeddings
    data = np.load(str(EMBS_FILE))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    print(f"Loaded: N={X.shape[0]}, d={d}, K={len(classes)}")

    centroids, sigma_W = compute_class_stats(X, y)

    # Compute per-class kappas and logit(q)
    per_class = {}
    for ci in classes:
        ranking = compute_per_class_kappa(centroids, sigma_W, d, ci)
        kappas = [k for k, _ in ranking]
        kappa_j1 = kappas[0]
        kappa_j2 = kappas[1] if len(kappas) > 1 else kappas[0]

        # phi for each tau
        phi_vals = {tau: phi_tau(kappas, tau) for tau in TAU_RANGE}

        # kappa_eff linear: kappa_j1 + 0.40 * kappa_j2
        kappa_eff_linear = kappa_j1 + W_J2_PRED * kappa_j2

        q = compute_per_class_q(X, y, ci)
        if q is None:
            continue
        lq = safe_logit(q)
        print(f"  class {ci}: kappa_j1={kappa_j1:.4f}, kappa_j2={kappa_j2:.4f}, "
              f"kappa_eff={kappa_eff_linear:.4f}, q={q:.4f}, logit={lq:.4f}")

        per_class[ci] = {
            "kappa_j1": kappa_j1,
            "kappa_j2": kappa_j2,
            "kappa_eff_linear": kappa_eff_linear,
            "phi": phi_vals,
            "kappas_sorted": kappas,
            "logit_q": lq,
        }

    if len(per_class) < 5:
        print("ERROR: too few classes with valid q")
        sys.exit(1)

    # Collect data for regression
    lq_vals = [per_class[ci]["logit_q"] for ci in per_class]
    kj1_vals = [per_class[ci]["kappa_j1"] for ci in per_class]
    keff_lin_vals = [per_class[ci]["kappa_eff_linear"] for ci in per_class]

    # Baseline: kappa_nearest only
    r_kj1, r2_kj1, slope_kj1, _ = fit_r2(kj1_vals, lq_vals)

    # kappa_eff linear (Codex formula)
    r_eff, r2_eff, slope_eff, _ = fit_r2(keff_lin_vals, lq_vals)

    print(f"\n{'='*70}")
    print("BASELINE: kappa_nearest")
    print(f"  r = {r_kj1:.4f}, R2 = {r2_kj1:.4f}, slope = {slope_kj1:.4f}")
    print(f"\nCODEX UPGRADE: kappa_eff = kappa_j1 + 0.40*kappa_j2")
    print(f"  r = {r_eff:.4f}, R2 = {r2_eff:.4f}, slope = {slope_eff:.4f}")

    # Phi(tau) sweep
    print(f"\nPHI(tau) SWEEP:")
    tau_results = {}
    best_tau = None
    best_r2 = -999
    for tau in TAU_RANGE:
        phi_vals_list = [per_class[ci]["phi"][tau] for ci in per_class]
        r_phi, r2_phi, slope_phi, _ = fit_r2(phi_vals_list, lq_vals)
        tau_label = f"{tau:.2f}" if tau < 1e8 else "inf"
        print(f"  tau={tau_label:>8}: r={r_phi:.4f}, R2={r2_phi:.4f}, "
              f"slope={slope_phi:.4f}")
        tau_results[tau_label] = {"r": r_phi, "r2": r2_phi, "slope": slope_phi}
        if r2_phi > best_r2:
            best_r2 = r2_phi
            best_tau = tau

    # Softmax weights at best_tau
    print(f"\nBEST TAU: {best_tau}")
    if best_tau < 1e8:
        example_ci = list(per_class.keys())[0]
        kappas_ex = per_class[example_ci]["kappas_sorted"]
        neg_k_over_tau = np.array([-k/best_tau for k in kappas_ex])
        weights = np.exp(neg_k_over_tau - neg_k_over_tau.max())
        weights = weights / weights.sum()
        print(f"  softmax(-kappa/tau*) for class {example_ci}:")
        for rank, (kappa, w) in enumerate(zip(kappas_ex[:5], weights[:5])):
            print(f"    rank {rank+1}: kappa={kappa:.4f}, w={w:.4f}")
        w_j2_measured = float(weights[1])
        print(f"\n  w_j2 (tau*={best_tau}): {w_j2_measured:.4f}  [predicted: {W_J2_PRED}]")
    else:
        w_j2_measured = None

    # Summary
    print(f"\n{'='*70}")
    print("VERDICT: Does phi upgrade kappa_nearest?")
    phi_upgrade_pass = best_r2 > r2_kj1
    eff_upgrade_pass = r2_eff > r2_kj1
    print(f"  phi(tau*)  R2 = {best_r2:.4f}  vs  kappa_nearest R2 = {r2_kj1:.4f}")
    print(f"  phi upgrade: {'PASS' if phi_upgrade_pass else 'FAIL'}")
    print(f"  kappa_eff_linear upgrade: {'PASS' if eff_upgrade_pass else 'FAIL'}")

    result = {
        "experiment": "phi_upgrade_test",
        "model": "pythia-160m",
        "dataset": "dbpedia14",
        "K": K,
        "n_classes": len(per_class),
        "baseline_kappa_nearest": {"r": r_kj1, "r2": r2_kj1, "slope": slope_kj1},
        "upgrade_kappa_eff_linear": {"r": r_eff, "r2": r2_eff, "slope": slope_eff,
                                      "w_j2_used": W_J2_PRED},
        "tau_sweep": tau_results,
        "best_tau": float(best_tau) if best_tau < 1e8 else None,
        "best_r2_phi": float(best_r2),
        "phi_upgrade_pass": phi_upgrade_pass,
        "eff_upgrade_pass": eff_upgrade_pass,
        "w_j2_measured_at_best_tau": w_j2_measured,
        "w_j2_predicted": W_J2_PRED,
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
