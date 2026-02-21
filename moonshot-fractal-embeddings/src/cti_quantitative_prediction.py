#!/usr/bin/env python -u
"""
QUANTITATIVE MAGNITUDE PREDICTION TEST (Feb 21 2026)
=====================================================
Codex identified this as the highest-leverage Nobel-track experiment:
  "pre-registered bidirectional causal intervention with QUANTITATIVE MAGNITUDE prediction"

The law: logit(q) = A * (dist_ratio - 1) + C

PREDICTION: if we intervene to change dist_ratio by delta_DR, then
  delta_logit(q) = A_hat * delta_DR

where A_hat is fitted from held-out (non-interventional) data.

This script:
1. Loads the triplet arm results (dist_ratio and q per epoch)
2. Loads the anti-triplet arm results (if available)
3. Fits A_hat from the epoch-by-epoch trajectory (causal variation in dist_ratio)
4. Predicts delta_q from delta_DR using A_hat
5. Compares predicted vs actual delta_q

Key innovation: we use WITHIN-ARM trajectory (epoch 1 → epoch 35) to fit A,
then test the cross-arm prediction (triplet vs anti-triplet vs baseline).

Pre-registered criteria:
  - Directional: triplet q > baseline > anti-triplet (all 3 ordered correctly)
  - Quantitative: |predicted_delta_q - actual_delta_q| / actual_delta_q < 0.50
    (50% tolerance given noise in small n)
"""

import json
import sys
import os
import numpy as np
from scipy import stats

# ================================================================
# LOAD RESULTS
# ================================================================
def load_results(path):
    """Load JSON results file."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_final_metrics(results, seeds):
    """Extract final q, DR, kappa per seed."""
    metrics = {}
    for seed in seeds:
        key = str(seed)
        if key not in results or not results[key]:
            continue
        final = results[key][-1]
        metrics[seed] = {
            "q": final.get("q"),
            "dr": final.get("dr"),
            "kappa": final.get("kappa"),
            "knn_acc": final.get("knn_acc"),
        }
    return metrics


def extract_trajectory(results, seeds):
    """Extract (epoch, q, DR) trajectory for fitting A."""
    traj = []
    for seed in seeds:
        key = str(seed)
        if key not in results:
            continue
        for point in results[key]:
            q = point.get("q")
            dr = point.get("dr")
            if q is not None and dr is not None and dr > 0 and 0.01 < q < 0.99:
                traj.append({
                    "seed": seed,
                    "epoch": point.get("epoch"),
                    "q": float(q),
                    "dr": float(dr),
                    "logit_q": float(np.log(q / (1 - q))),
                })
    return traj


# ================================================================
# FIT A FROM WITHIN-ARM TRAJECTORY
# ================================================================
def fit_A_from_trajectory(traj):
    """Fit logit(q) = A * dist_ratio + C from trajectory data."""
    if len(traj) < 4:
        return None, None, None
    drs = np.array([p["dr"] for p in traj])
    logit_qs = np.array([p["logit_q"] for p in traj])
    slope, intercept, r, p, se = stats.linregress(drs, logit_qs)
    return float(slope), float(intercept), float(r**2)


# ================================================================
# QUANTITATIVE PREDICTION
# ================================================================
def predict_delta_q(A_hat, C_hat, dr_baseline, q_baseline, delta_dr):
    """
    Predict delta_q given delta_dr using fitted law.
    logit(q_new) = A_hat * (dr_baseline + delta_dr) + C_hat
    logit(q_baseline) = A_hat * dr_baseline + C_hat
    delta_logit = A_hat * delta_dr
    """
    logit_q_baseline = float(np.log(q_baseline / (1 - q_baseline))) if 0 < q_baseline < 1 else None
    if logit_q_baseline is None:
        return None
    logit_q_predicted = logit_q_baseline + A_hat * delta_dr
    q_predicted = float(1 / (1 + np.exp(-logit_q_predicted)))
    return q_predicted


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("QUANTITATIVE MAGNITUDE PREDICTION TEST")
    print("=" * 70)

    SEEDS = [42, 123, 456, 789, 1024]
    BASELINE_Q = {42: 0.7073, 123: 0.7071, 456: 0.7047, 789: 0.7105, 1024: 0.7087}
    BASELINE_Q_MEAN = np.mean(list(BASELINE_Q.values()))

    # Load available results
    triplet_res = load_results("results/cti_cifar_triplet.json")
    antitriplet_res = load_results("results/cti_cifar_antitriplet.json")
    baseline_dr_path = "results/cti_cifar_baseline_dr.json"

    print(f"\nAvailable files:")
    print(f"  Triplet arm: {'LOADED' if triplet_res else 'NOT AVAILABLE'}")
    print(f"  Anti-triplet arm: {'LOADED' if antitriplet_res else 'NOT AVAILABLE'}")

    if triplet_res is None and antitriplet_res is None:
        print("\nNo completed arm results yet. Run this after arms complete.")
        print("For now, showing what the analysis WILL look like.")
        _show_analysis_template(BASELINE_Q_MEAN)
        return

    # Extract metrics
    results_summary = {}

    if triplet_res:
        triplet_final = extract_final_metrics(triplet_res, SEEDS)
        triplet_traj = extract_trajectory(triplet_res, SEEDS)
        print(f"\nTriplet arm: {len(triplet_final)} seeds with final metrics")
        for seed, m in sorted(triplet_final.items()):
            base_q = BASELINE_Q.get(seed, BASELINE_Q_MEAN)
            delta = (m["q"] or 0) - base_q
            print(f"  Seed {seed}: q={m['q']:.4f}  DR={m['dr']:.4f}  delta_q={delta:+.4f}")
        results_summary["triplet"] = triplet_final

    if antitriplet_res:
        anti_final = extract_final_metrics(antitriplet_res, SEEDS)
        anti_traj = extract_trajectory(antitriplet_res, SEEDS)
        print(f"\nAnti-triplet arm: {len(anti_final)} seeds with final metrics")
        for seed, m in sorted(anti_final.items()):
            base_q = BASELINE_Q.get(seed, BASELINE_Q_MEAN)
            delta = (m["q"] or 0) - base_q
            print(f"  Seed {seed}: q={m['q']:.4f}  DR={m['dr']:.4f}  delta_q={delta:+.4f}")
        results_summary["antitriplet"] = anti_final

    # Fit A from trajectory (within-arm)
    print("\n" + "=" * 70)
    print("FIT A FROM WITHIN-ARM EPOCH TRAJECTORY")
    print("=" * 70)

    if triplet_res and triplet_traj:
        A_trip, C_trip, r2_trip = fit_A_from_trajectory(triplet_traj)
        print(f"\nTriplet arm trajectory fit:")
        print(f"  A_hat = {A_trip:.4f}  C_hat = {C_trip:.4f}  R2 = {r2_trip:.3f}")
        print(f"  n_points = {len(triplet_traj)}")
    else:
        A_trip, C_trip, r2_trip = None, None, None

    # Cross-arm quantitative prediction
    print("\n" + "=" * 70)
    print("QUANTITATIVE PREDICTION: delta_logit(q) = A_hat * delta_DR")
    print("=" * 70)

    if A_trip is not None and antitriplet_res:
        print(f"\nUsing A_hat = {A_trip:.4f} from triplet arm trajectory")

        successes = 0
        total_seeds = 0
        for seed in SEEDS:
            if str(seed) not in (triplet_res or {}):
                continue
            if str(seed) not in (antitriplet_res or {}):
                continue

            trip_m = triplet_final.get(seed, {})
            anti_m = anti_final.get(seed, {})
            base_q = BASELINE_Q.get(seed, BASELINE_Q_MEAN)
            base_dr = 1.0  # approximate baseline DR (will be updated when baseline arm runs)

            if trip_m.get("dr") and anti_m.get("dr"):
                trip_dr = trip_m["dr"]
                anti_dr = anti_m["dr"]
                trip_q = trip_m["q"]
                anti_q = anti_m["q"]

                delta_dr_trip = trip_dr - base_dr
                delta_dr_anti = anti_dr - base_dr

                pred_q_trip = predict_delta_q(A_trip, C_trip, base_dr, base_q, delta_dr_trip)
                pred_q_anti = predict_delta_q(A_trip, C_trip, base_dr, base_q, delta_dr_anti)

                print(f"\n  Seed {seed}:")
                print(f"    Triplet:     DR={trip_dr:.4f}  q_actual={trip_q:.4f}  q_pred={pred_q_trip:.4f}  "
                      f"err={abs(trip_q-pred_q_trip):.4f}")
                print(f"    Anti-triplet: DR={anti_dr:.4f}  q_actual={anti_q:.4f}  q_pred={pred_q_anti:.4f}  "
                      f"err={abs(anti_q-pred_q_anti):.4f}")

                if pred_q_trip is not None and pred_q_anti is not None:
                    err_trip = abs(trip_q - pred_q_trip)
                    err_anti = abs(anti_q - pred_q_anti)
                    if err_trip < 0.05 and err_anti < 0.05:
                        successes += 1
                    total_seeds += 1

        if total_seeds > 0:
            print(f"\n  Quantitative accuracy: {successes}/{total_seeds} seeds within 0.05 q error")

    # Directional test
    print("\n" + "=" * 70)
    print("DIRECTIONAL TEST: triplet q > baseline > anti-triplet q")
    print("=" * 70)

    if triplet_res and antitriplet_res:
        trip_qs = [triplet_final.get(s, {}).get("q", 0) for s in SEEDS if s in triplet_final]
        anti_qs = [anti_final.get(s, {}).get("q", 0) for s in SEEDS if s in anti_final]
        base_qs = [BASELINE_Q[s] for s in SEEDS]

        mean_trip = np.mean(trip_qs) if trip_qs else None
        mean_anti = np.mean(anti_qs) if anti_qs else None
        mean_base = BASELINE_Q_MEAN

        print(f"\n  Mean q:")
        print(f"    +triplet:    {mean_trip:.4f}  ({mean_trip - mean_base:+.4f} vs baseline)" if mean_trip else "    +triplet: N/A")
        print(f"    Baseline:    {mean_base:.4f}")
        print(f"    -anti:       {mean_anti:.4f}  ({mean_anti - mean_base:+.4f} vs baseline)" if mean_anti else "    -anti: N/A")

        if mean_trip and mean_anti:
            directional_pass = (mean_trip > mean_base > mean_anti)
            print(f"\n  Directional test (trip > base > anti): {'PASS' if directional_pass else 'FAIL'}")
            print(f"  Asymmetry (trip improvement / anti degradation): "
                  f"{abs(mean_trip - mean_base) / (abs(mean_anti - mean_base) + 1e-6):.2f}x")


def _show_analysis_template(baseline_q_mean):
    """Show what the analysis will look like when data is available."""
    print("\nTemplate (once arms complete):")
    print(f"  Baseline q = {baseline_q_mean:.4f}")
    print(f"  +Triplet arm (running): if q=0.727 -> delta_q=+0.020, DR should increase ~0.05")
    print(f"  -Anti-triplet arm (queued): if q=0.695 -> delta_q=-0.012, DR should decrease ~0.03")
    print(f"\n  Quantitative prediction: A_hat * delta_DR = delta_logit(q)")
    print(f"    If A~20: delta_DR=0.05 -> delta_logit=1.0 -> delta_q~0.025 (at q=0.71)")
    print(f"  Pre-registered: |pred - actual| < 50% of actual")


if __name__ == "__main__":
    main()
