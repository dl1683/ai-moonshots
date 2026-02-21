#!/usr/bin/env python -u
"""
TRAINING DYNAMICS VALIDATION (Feb 21 2026)
==========================================
Does the law logit(q) = alpha * kappa_nearest + C hold
WITHIN A SINGLE TRAINING RUN (across training epochs)?

This is a stronger test than cross-architecture:
  - If alpha (within-run slope) matches LOAO alpha = 1.54, then:
    the law is not just a coincidence across architectures
    it holds DYNAMICALLY as representations evolve
  - If alpha differs: different dynamics than steady-state law

Data sources:
1. CIFAR triplet arm: per-epoch (kappa, q) for CE+triplet (kappa DECREASED)
2. Cross-modal arm: per-epoch (kappa, q) for pure CE (kappa increases during training)
3. Anti-triplet arm: per-epoch (kappa, q) for CE+anti-triplet

Key theoretical prediction:
  Within a training run, logit(q) ~ alpha_dyn * kappa_nearest + C_run
  If alpha_dyn == 1.54 (LOAO value): law is universal across BOTH architectures AND time
  If alpha_dyn != 1.54: different regime (non-stationary dynamics vs stationary law)

Nobel relevance:
  "Universal constant alpha appears not just across architectures but across training time"
  This would strengthen the claim that alpha is a FUNDAMENTAL constant.
"""

import json
import os
import numpy as np
from scipy import stats

LOAO_ALPHA = 1.549  # From 7-architecture LOAO (per-task-intercept version)
K_CIFAR    = 20     # CIFAR-100 coarse classes


def logit(q, K=K_CIFAR):
    """Compute logit(q_normalized) where q_normalized = (q_raw - 1/K)/(1-1/K)."""
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def load_triplet_dynamics():
    """Load per-epoch (kappa, q) from triplet arm."""
    path = "results/cti_cifar_triplet.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)

    runs = []
    for seed_str, epochs in d.items():
        if not isinstance(epochs, list):
            continue
        run = []
        for ep in epochs:
            q_raw = float(ep["q"])
            kappa = float(ep["kappa"])
            dr = float(ep.get("dr", 0))
            run.append({
                "epoch": ep["epoch"],
                "q": q_raw,
                "kappa": kappa,
                "dr": dr,
                "logit_q": logit(q_raw),
                "seed": int(seed_str),
                "arm": "triplet",
            })
        if run:
            runs.append(run)
    return runs


def load_crossmodal_dynamics():
    """Load per-epoch (kappa, q) from cross-modal arm (pure CE)."""
    path = "results/cti_cifar_crossmodal.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)

    runs = []
    # Structure: {seed: [{epoch, q_actual, kappa, ...}, ...], ...}
    for key in d:
        try:
            seed = int(key)
        except (ValueError, TypeError):
            continue
        epochs = d[key]
        if not isinstance(epochs, list):
            continue
        run = []
        for ep in epochs:
            q_raw = float(ep.get("q_actual", ep.get("q", 0)))
            kappa = float(ep.get("kappa", ep.get("kappa_nearest", 0)))
            run.append({
                "epoch": ep.get("epoch", 0),
                "q": q_raw,
                "kappa": kappa,
                "logit_q": logit(q_raw),
                "seed": seed,
                "arm": "crossmodal_ce",
            })
        if run:
            runs.append(run)
    return runs


def load_antitriplet_dynamics():
    """Load per-epoch (kappa, q) from anti-triplet arm."""
    path = "results/cti_cifar_antitriplet.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)

    runs = []
    for seed_str, epochs in d.items():
        if not isinstance(epochs, list):
            continue
        run = []
        for ep in epochs:
            q_raw = float(ep.get("q", ep.get("knn_q", 0)))
            kappa = float(ep.get("kappa", ep.get("kappa_nearest", 0)))
            run.append({
                "epoch": ep["epoch"],
                "q": q_raw,
                "kappa": kappa,
                "logit_q": logit(q_raw),
                "seed": int(seed_str),
                "arm": "antitriplet",
            })
        if run:
            runs.append(run)
    return runs


def analyze_within_run_dynamics(runs, arm_name):
    """
    For each run: fit logit(q) = alpha * kappa + C
    Report:
    - alpha_within (per-run regression slope)
    - Pearson r (within-run)
    - Comparison to LOAO_ALPHA = 1.54
    """
    if not runs:
        print(f"  {arm_name}: No data")
        return

    print(f"\n{arm_name.upper()}: Within-run dynamics (law holds across training time?)")
    print(f"  Law prediction: alpha_within == {LOAO_ALPHA} (LOAO from 7 archs)")
    print()

    per_run_alphas = []
    per_run_rs = []

    for run in runs:
        kappas = np.array([p["kappa"] for p in run])
        logit_qs = np.array([p["logit_q"] for p in run])
        seed = run[0]["seed"]

        if len(kappas) < 3 or np.std(kappas) < 1e-6 or np.std(logit_qs) < 1e-6:
            print(f"  Seed {seed}: insufficient variation (kappa_std={np.std(kappas):.4f})", flush=True)
            continue

        # OLS regression: logit_q = alpha * kappa + C
        A = np.vstack([kappas, np.ones(len(kappas))]).T
        alpha, C = np.linalg.lstsq(A, logit_qs, rcond=None)[0]
        r = np.corrcoef(kappas, logit_qs)[0, 1]

        per_run_alphas.append(alpha)
        per_run_rs.append(r)

        q_range = max(p["q"] for p in run) - min(p["q"] for p in run)
        k_range = max(p["kappa"] for p in run) - min(p["kappa"] for p in run)
        print(f"  Seed {seed}: alpha={alpha:.4f}, r={r:.4f}, "
              f"C={C:.4f}, kappa_range={k_range:.4f}, q_range={q_range:.4f}")

    if not per_run_alphas:
        print("  No valid runs")
        return

    mean_alpha = np.mean(per_run_alphas)
    std_alpha  = np.std(per_run_alphas)
    mean_r     = np.mean(per_run_rs)
    cv_alpha   = std_alpha / abs(mean_alpha) if abs(mean_alpha) > 1e-6 else float("inf")

    print(f"\n  Mean within-run alpha = {mean_alpha:.4f} +/- {std_alpha:.4f} (CV={cv_alpha:.3f})")
    print(f"  LOAO alpha = {LOAO_ALPHA:.4f}")
    deviation = abs(mean_alpha - LOAO_ALPHA) / LOAO_ALPHA
    print(f"  Deviation from LOAO: {deviation:.1%}")
    print(f"  Mean r = {mean_r:.4f}")

    if abs(deviation) < 0.2:
        print(f"  *** ALPHA CONSISTENT with LOAO (< 20% deviation) ***")
        print(f"  INTERPRETATION: Law holds dynamically, not just across architectures")
    else:
        print(f"  ALPHA INCONSISTENT with LOAO (>{deviation:.0%} deviation)")
        print(f"  INTERPRETATION: Training dynamics have different alpha than steady-state")

    return {"mean_alpha": float(mean_alpha), "std_alpha": float(std_alpha),
            "cv_alpha": float(cv_alpha), "mean_r": float(mean_r),
            "deviation_from_loao": float(deviation)}


def analyze_all_runs_pooled(all_run_data, arm_name):
    """Pool all epochs from all runs and fit a single regression."""
    all_kappas   = []
    all_logit_qs = []
    all_seeds    = []

    for run in all_run_data:
        # Demean per run (within-run variation)
        kappas   = np.array([p["kappa"]   for p in run])
        logit_qs = np.array([p["logit_q"] for p in run])
        seed     = run[0]["seed"]

        # Demean (within-run)
        all_kappas.extend(kappas - kappas.mean())
        all_logit_qs.extend(logit_qs - logit_qs.mean())
        all_seeds.extend([seed] * len(run))

    kappas_arr  = np.array(all_kappas)
    logit_q_arr = np.array(all_logit_qs)

    if len(kappas_arr) < 4 or np.std(kappas_arr) < 1e-6:
        print(f"  {arm_name}: Insufficient data for pooled fit")
        return

    A = np.vstack([kappas_arr, np.ones(len(kappas_arr))]).T
    alpha_pool, C_pool = np.linalg.lstsq(A, logit_q_arr, rcond=None)[0]
    r_pool = np.corrcoef(kappas_arr, logit_q_arr)[0, 1]

    ss_res = np.sum((logit_q_arr - (alpha_pool * kappas_arr + C_pool))**2)
    ss_tot = np.sum((logit_q_arr - logit_q_arr.mean())**2)
    r2_pool = 1 - ss_res / (ss_tot + 1e-10)

    print(f"\n  {arm_name.upper()} POOLED WITHIN-RUN FIT:")
    print(f"    alpha = {alpha_pool:.4f} (LOAO = {LOAO_ALPHA})")
    print(f"    r = {r_pool:.4f}, R2 = {r2_pool:.4f}")
    deviation = abs(alpha_pool - LOAO_ALPHA) / LOAO_ALPHA
    print(f"    Deviation from LOAO alpha: {deviation:.1%}")

    return {"alpha_pooled": float(alpha_pool), "r_pooled": float(r_pool),
            "r2_pooled": float(r2_pool), "n_points": int(len(kappas_arr))}


def main():
    print("=" * 70)
    print("TRAINING DYNAMICS: Does logit(q) = alpha * kappa hold across time?")
    print("=" * 70)
    print()

    results = {}

    # Load all data
    triplet_runs    = load_triplet_dynamics()
    crossmodal_runs = load_crossmodal_dynamics()
    antitriplet_runs = load_antitriplet_dynamics()

    arms = [
        (triplet_runs,    "CE+Triplet (kappa decreased)"),
        (crossmodal_runs, "Pure CE (kappa increases during training)"),
        (antitriplet_runs,"CE+Anti-Triplet"),
    ]

    for runs, name in arms:
        if runs:
            print(f"\n--- {name} (n_runs={len(runs)}) ---")
            per_run_result = analyze_within_run_dynamics(runs, name)
            pool_result    = analyze_all_runs_pooled(runs, name)
            results[name] = {
                "per_run": per_run_result,
                "pooled":  pool_result,
            }
        else:
            print(f"\n--- {name}: No data available yet ---")

    # Combined analysis: pool all arms together (if multiple available)
    all_available = [r for r, _ in arms if r]
    if len(all_available) >= 2:
        combined = []
        for runs, name in arms:
            if runs:
                combined.extend(runs)

        print(f"\n\n{'='*70}")
        print("COMBINED ANALYSIS (all arms pooled)")
        pool_result = analyze_all_runs_pooled(combined, "All arms")
        results["combined_all_arms"] = pool_result

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print()
    print(f"  LOAO alpha (7 architectures): {LOAO_ALPHA:.4f}")
    print()
    for arm_name, r in results.items():
        if r.get("pooled"):
            p = r["pooled"]
            print(f"  {arm_name}:")
            print(f"    Within-run alpha = {p['alpha_pooled']:.4f}, R2 = {p['r2_pooled']:.4f}")
            print(f"    Deviation from LOAO: {p.get('deviation_from_loao', abs(p['alpha_pooled']-LOAO_ALPHA)/LOAO_ALPHA):.1%}")

    # Save
    out = {
        "experiment": "training_dynamics_alpha_validation",
        "loao_alpha": LOAO_ALPHA,
        "results": {k: v for k, v in results.items() if v},
    }
    with open("results/cti_training_dynamics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/cti_training_dynamics.json")


if __name__ == "__main__":
    main()
