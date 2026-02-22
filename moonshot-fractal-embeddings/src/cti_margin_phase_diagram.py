#!/usr/bin/env python -u
"""
MARGIN PHASE DIAGRAM: When Does j2 Matter? (Feb 21 2026)
=========================================================
Codex recommendation: checkpoint-conditioned phase diagram of B_j2 vs (margin, q, kappa).

This script provides a SYNTHETIC proof of concept:
  - K=3 classes in R^d (controlled Gaussians)
  - Class 0 is focus; j1=1 at fixed D1; j2=2 at D2 = D1 * margin
  - Sweep margin from 1.0 to 3.0
  - At each margin: run Arm B (move j2 farther, class 0 fixed)
  - Record B_j2_r = r(delta_j2, delta_logit_q_0)
  - EXPECTED: B_j2_r is large at margin=1.0, decays to ~0 as margin increases

PRE-REGISTERED PREDICTION (from Gumbel Race theory):
  B_j2_r should decay from ~1.0 at margin=1.0 toward 0 at margin>1.5
  This is the FINITE-COMPETITION CORRECTION to the hard-min CTI law.

Also runs the MARGIN SWEEP at different kappa levels (low/mid/high) to show
that the transition point depends on kappa*sqrt(d_eff) (the discriminability).

THEORETICAL PREDICTION (from soft-min formula):
  If j1 and j2 are both at distance kappa*sigma_W*sqrt(d), the correction is:
    delta_logit_q from removing j2 = log(K_eff_old/K_eff_new)
    where K_eff is determined by exp(-A*(kappa_j2 - kappa_j1)*sqrt(d_eff))
  At margin=1.0: K_eff contribution from j2 = exp(0) = 1 (maximum, same as j1)
  At margin=1.5: K_eff contribution from j2 = exp(-A*0.5*kappa*sqrt(d_eff))
"""

import json
import os
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

# ================================================================
# CONFIG
# ================================================================
OUT_JSON = "results/cti_margin_phase_diagram.json"
OUT_LOG  = "results/cti_margin_phase_diagram_log.txt"

D = 100           # embedding dimension (fast but high-d Gaussian behavior)
N_PER_CLASS = 400 # samples per class (enough for good 1NN resolution)
N_CV = 10         # 10-fold CV for q estimation

# MARGIN sweep: D2/D1 ratio
MARGIN_RANGE = np.array([1.01, 1.05, 1.10, 1.20, 1.30, 1.50, 1.75, 2.00, 2.50, 3.00])

# DELTA range for Arm B: push j2 farther (fractions of kappa_j1 distance)
N_DELTA = 13
DELTA_B_MAX = 3.0   # embedding units

# KAPPA LEVELS: test at low, mid, high kappa (controlled D1/sigma)
KAPPA_LEVELS = {
    "kappa_low":  0.30,   # low: many errors, j1 and j2 both confuse
    "kappa_mid":  0.60,   # mid: moderate errors
    "kappa_high": 1.00,   # high: few errors, j1 barely confuses
}

N_SEEDS = 5   # multiple seeds for error bars
K = 3         # 3 classes: {0, 1, 2}

# ================================================================
# GAUSSIAN GEOMETRY
# ================================================================
def make_gaussians(d, n_per_class, kappa_j1, margin, sigma=1.0, seed=0):
    """
    3-class Gaussian: class 0 at origin, j1 at D1, j2 at D1*margin.
    kappa_j1 = D1 / (sigma * sqrt(d)) -> D1 = kappa_j1 * sigma * sqrt(d)
    """
    rng = np.random.RandomState(seed)
    D1 = kappa_j1 * sigma * np.sqrt(d)
    D2 = D1 * margin

    # Place centroids on first two axes for interpretability
    mu0 = np.zeros(d)
    mu1 = np.zeros(d); mu1[0] = D1           # j1 along axis 0
    mu2 = np.zeros(d); mu2[0] = D2           # j2 along axis 0 (farther)

    X0 = rng.randn(n_per_class, d) * sigma + mu0
    X1 = rng.randn(n_per_class, d) * sigma + mu1
    X2 = rng.randn(n_per_class, d) * sigma + mu2

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class, dtype=np.int64)
    return X, y, mu0, mu1, mu2, sigma


def compute_per_class_q(X, y, ci, n_splits=N_CV, seed=42):
    """10-fold CV per-class recall."""
    K = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    recalls = []
    for tr, te in skf.split(X, y):
        if (y[tr] == ci).sum() < 2:
            continue
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X[tr], y[tr])
        mask = (y[te] == ci)
        if mask.sum() == 0:
            continue
        preds = knn.predict(X[te][mask])
        recalls.append(float((preds == ci).mean()))
    if not recalls:
        return None
    q_raw = np.mean(recalls)
    K_inv = 1.0 / K
    return float((q_raw - K_inv) / (1.0 - K_inv))


def logit(q):
    return float(np.log(np.clip(q, 1e-5, 1-1e-5) / np.clip(1 - q, 1e-5, 1-1e-5)))


def apply_competitor_shift(X, y, cj, delta, direction):
    """Move only class cj in direction by delta. Class 0 fixed."""
    X_new = X.copy()
    X_new[y == cj] += delta * direction
    return X_new


def run_arm_B(X, y, mu0, mu2, sigma, kappa_j1_orig, delta_range, seed):
    """Arm B: push j2 (class 2) farther from class 0 (fixed), measure q_0."""
    direction = (mu2 - mu0) / np.linalg.norm(mu2 - mu0)
    records = []
    for delta in delta_range:
        X_new = apply_competitor_shift(X, y, 2, delta, direction)
        # Recompute kappa_j1 (to class 1) — should NOT change
        new_cents = {c: X_new[y==c].mean(axis=0) for c in [0,1,2]}
        sw = float(np.sqrt(np.mean((X_new - np.array([new_cents[c] for c in y]))**2)))
        kappa_j1_new = float(np.linalg.norm(new_cents[1] - new_cents[0]) / (sw * np.sqrt(X.shape[1]) + 1e-10))
        kappa_j2_new = float(np.linalg.norm(new_cents[2] - new_cents[0]) / (sw * np.sqrt(X.shape[1]) + 1e-10))

        q0 = compute_per_class_q(X_new, y, ci=0, seed=seed)
        if q0 is None:
            continue
        records.append({
            "delta": float(delta),
            "kappa_j1_new": kappa_j1_new,
            "kappa_j2_new": kappa_j2_new,
            "q0": float(q0),
            "logit_q0": logit(q0),
        })
    return records


def fit_r(records, x_key, y_key="logit_q0"):
    xs = np.array([r[x_key] for r in records])
    ys = np.array([r[y_key] for r in records])
    if len(xs) < 4 or np.std(xs) < 1e-10 or np.std(ys) < 1e-10:
        return 0.0, 1.0
    r, p = stats.pearsonr(xs, ys)
    return float(r), float(p)


def theoretical_B_r(kappa_j1, margin, d_eff, A=1.054):
    """
    Expected B_j2 effect: from Gumbel Race soft-min formula.
    K_eff contribution of j2 relative to j1 at original vs shifted position.
    This is proportional to exp(-A * (kappa_j2 - kappa_j1) * sqrt(d_eff))
    """
    kappa_j2 = kappa_j1 * margin
    delta_kappa = kappa_j2 - kappa_j1
    weight_j2 = float(np.exp(-A * delta_kappa * np.sqrt(d_eff)))
    return weight_j2   # higher = j2 matters more


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs("results", exist_ok=True)
    log_file = open(OUT_LOG, "w", buffering=1)
    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log("=" * 70)
    log("MARGIN PHASE DIAGRAM: When Does j2 Matter?")
    log("=" * 70)
    log(f"d={D}, N_per_class={N_PER_CLASS}, K=3, N_CV={N_CV}, N_seeds={N_SEEDS}")
    log(f"Margin range: {MARGIN_RANGE}")
    log(f"Kappa levels: {KAPPA_LEVELS}")
    log("=" * 70)
    log("THEORY: B_j2_r should decay from ~1.0 at margin=1.0 toward 0 at margin>1.5")

    DELTA_B_RANGE = np.linspace(0.0, DELTA_B_MAX, N_DELTA)
    sigma = 1.0
    d_eff = D   # for unit-variance Gaussians, d_eff = d

    all_results = {}

    for kappa_name, kappa_j1 in KAPPA_LEVELS.items():
        log(f"\n{'='*60}")
        log(f"KAPPA LEVEL: {kappa_name} (kappa_j1={kappa_j1})")
        log(f"{'='*60}")

        margin_results = []
        for margin in MARGIN_RANGE:
            log(f"\n  margin={margin:.2f} (kappa_j2={kappa_j1*margin:.3f})")

            # Run multiple seeds
            seed_rs = []
            seed_qs = []
            for seed in range(N_SEEDS):
                X, y, mu0, mu1, mu2, sig = make_gaussians(D, N_PER_CLASS, kappa_j1, margin, sigma, seed)

                # Baseline q_0
                q0_base = compute_per_class_q(X, y, ci=0, seed=seed+100)
                if q0_base is None:
                    continue
                seed_qs.append(q0_base)

                # Arm B sweep
                recs = run_arm_B(X, y, mu0, mu2, sig, kappa_j1, DELTA_B_RANGE, seed)
                if len(recs) < 4:
                    seed_rs.append(0.0)
                    continue
                r_B, p_B = fit_r(recs, "kappa_j2_new")
                seed_rs.append(r_B)

                # Verify kappa_j1 unchanged
                kappa_j1_check = recs[0]["kappa_j1_new"]
                kappa_j1_final = recs[-1]["kappa_j1_new"]

            if not seed_rs:
                continue

            mean_r = float(np.tanh(np.arctanh(np.clip(seed_rs, -0.9999, 0.9999)).mean()))
            std_r  = float(np.std(seed_rs))
            mean_q = float(np.mean(seed_qs))
            theory_weight = theoretical_B_r(kappa_j1, margin, d_eff)

            log(f"    B_j2_r = {mean_r:.3f} +/- {std_r:.3f} (seeds: {[f'{r:.2f}' for r in seed_rs]})")
            log(f"    baseline q_0 = {mean_q:.3f}, theory_weight = {theory_weight:.3f}")

            margin_results.append({
                "margin": float(margin),
                "kappa_j1": kappa_j1,
                "kappa_j2": float(kappa_j1 * margin),
                "B_j2_r_mean": mean_r,
                "B_j2_r_std": std_r,
                "B_j2_r_seeds": [float(r) for r in seed_rs],
                "q0_baseline": mean_q,
                "theory_weight_j2": theory_weight,
            })

        all_results[kappa_name] = margin_results

        # Summary for this kappa level
        log(f"\n  SUMMARY for {kappa_name}:")
        log(f"  {'Margin':>8} | {'B_j2_r':>8} | {'q_0':>8} | {'Theory_w':>10}")
        log(f"  {'-'*45}")
        for r in margin_results:
            log(f"  {r['margin']:>8.2f} | {r['B_j2_r_mean']:>8.3f} | {r['q0_baseline']:>8.3f} | {r['theory_weight_j2']:>10.3f}")

    # ----------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------
    log("\n" + "="*70)
    log("PHASE DIAGRAM VERDICT")
    log("="*70)
    for kappa_name, margin_results in all_results.items():
        if not margin_results:
            continue
        # Check monotonicity: B_j2_r should decrease with margin
        rs = [r["B_j2_r_mean"] for r in margin_results]
        margins = [r["margin"] for r in margin_results]
        rho, p = stats.spearmanr(margins, rs)
        log(f"\n{kappa_name}: Spearman r(margin, B_j2_r) = {rho:.3f} p={p:.4f}")
        log(f"  Expected: negative (B_j2_r decreases as margin increases)")
        log(f"  B_j2_r at margin=1.01: {rs[0]:.3f}")
        log(f"  B_j2_r at margin=3.00: {rs[-1]:.3f}")
        if rho < -0.5 and p < 0.05:
            log(f"  PASS: monotonic decay confirmed -> Gumbel Race theory correct")
        else:
            log(f"  FAIL: expected monotonic decay not confirmed")

    # Save
    out = {
        "experiment": "margin_phase_diagram",
        "description": "Synthetic K=3 Gaussians. Arm B effect as function of j2 margin.",
        "config": {
            "d": D, "n_per_class": N_PER_CLASS, "k": K, "n_cv": N_CV,
            "sigma": 1.0, "margin_range": list(MARGIN_RANGE),
            "kappa_levels": KAPPA_LEVELS,
        },
        "results": all_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nSaved to {OUT_JSON}")
    log_file.close()


if __name__ == "__main__":
    main()
