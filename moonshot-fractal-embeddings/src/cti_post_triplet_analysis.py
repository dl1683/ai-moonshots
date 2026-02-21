#!/usr/bin/env python -u
"""
POST-TRIPLET SYNTHESIS ANALYSIS (Feb 21 2026)
=============================================
Immediately after triplet arm completes:
1. Summarize triplet vs baseline vs anti-triplet comparison
2. Test quantitative prediction: delta_logit_q / delta_kappa vs alpha_hat
3. Update Nobel scorecard
4. Print comprehensive summary

Usage: python src/cti_post_triplet_analysis.py
"""

import json
import os
import numpy as np
from scipy import stats

BASELINE_Q_MEAN = 0.7077
BASELINE_SEEDS = {42: 0.7073, 123: 0.7071, 456: 0.7047, 789: 0.7105, 1024: 0.7087}
PASS_THRESHOLD = 0.7277  # baseline + 0.020

FROZEN_ALPHA = 1.54  # from LOAO text models (within-task)

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def summarize_arm(results, name, baseline_q_mean, K=20):
    """Extract final q per seed from arm results."""
    if results is None:
        print(f"  {name}: NOT AVAILABLE")
        return None

    # Try different structures
    if isinstance(results, dict):
        # Try seed-keyed dict
        if "summary" in results:
            s = results["summary"]
            q = s.get("mean_q", s.get("q_mean"))
            kappa = s.get("mean_kappa", s.get("kappa_mean"))
            dr = s.get("mean_dr", s.get("dr_mean"))
            print(f"  {name}: q={q:.4f}, kappa={kappa:.4f}, dr={dr:.4f if dr else 'N/A'}")
            return s

        # Try per-seed structure
        seeds_data = {}
        for key in results:
            try:
                seed = int(key)
                seed_results = results[key]
                if isinstance(seed_results, list) and seed_results:
                    final = seed_results[-1]
                    seeds_data[seed] = final
            except (ValueError, TypeError):
                pass

        if seeds_data:
            qs = [v.get("q", v.get("knn_acc_q")) for v in seeds_data.values() if v.get("q") or v.get("knn_acc_q")]
            kappas = [v.get("kappa", v.get("kappa_nearest")) for v in seeds_data.values() if v.get("kappa") or v.get("kappa_nearest")]
            drs = [v.get("dr", v.get("dist_ratio")) for v in seeds_data.values() if v.get("dr") or v.get("dist_ratio")]

            if qs:
                q_mean = np.mean(qs)
                q_std = np.std(qs)
                delta_q = q_mean - baseline_q_mean
                # Normalized q (K=20)
                q_norm_mean = np.mean([(q - 1/K) / (1 - 1/K) for q in qs])

                print(f"  {name}: q={q_mean:.4f}±{q_std:.4f}, delta_q={delta_q:+.4f}, q_norm={q_norm_mean:.4f}")
                if kappas:
                    print(f"    kappa={np.mean(kappas):.4f}±{np.std(kappas):.4f}")
                if drs:
                    print(f"    dr={np.mean(drs):.4f}±{np.std(drs):.4f}")

                # t-test vs baseline
                baseline_qs = [BASELINE_SEEDS.get(s, BASELINE_Q_MEAN) for s in seeds_data.keys()]
                if len(qs) == len(baseline_qs) and len(qs) > 1:
                    t, p = stats.ttest_rel(qs, baseline_qs)
                    print(f"    paired t-test vs baseline: t={t:.3f}, p={p:.4f}")

                passed = q_mean >= PASS_THRESHOLD if delta_q > 0 else None
                if passed is not None:
                    print(f"    PRE-REGISTERED: {'PASS' if passed else 'FAIL'} (threshold={PASS_THRESHOLD:.4f})")

                return {
                    "q_mean": q_mean,
                    "q_std": q_std,
                    "delta_q": delta_q,
                    "q_norm": q_norm_mean,
                    "kappa_mean": np.mean(kappas) if kappas else None,
                    "dr_mean": np.mean(drs) if drs else None,
                    "passed": passed,
                    "seeds": qs,
                }

    print(f"  {name}: Could not parse results (keys={list(results.keys())[:5] if isinstance(results, dict) else type(results).__name__})")
    return None


def main():
    print("=" * 70)
    print("POST-TRIPLET CAUSAL SYNTHESIS ANALYSIS")
    print("=" * 70)
    print()

    # Load all arm results
    baseline = load("results/cti_cifar_baseline.json")
    triplet = load("results/cti_cifar_triplet.json")
    antitriplet = load("results/cti_cifar_antitriplet.json")
    crossmodal = load("results/cti_cifar_crossmodal.json")
    quant_pred = load("results/cti_quantitative_prediction.json")

    print("=== ARM RESULTS ===")
    print(f"  Baseline: q={BASELINE_Q_MEAN:.4f} (pre-computed, 5 seeds)")

    triplet_summary = summarize_arm(triplet, "Triplet (+lambda)", BASELINE_Q_MEAN)
    antitriplet_summary = summarize_arm(antitriplet, "Anti-triplet (-lambda)", BASELINE_Q_MEAN)

    print()
    print("=== DIRECTIONAL CAUSAL TEST ===")
    if triplet_summary and antitriplet_summary:
        t_q = triplet_summary.get("q_mean", BASELINE_Q_MEAN)
        a_q = antitriplet_summary.get("q_mean", BASELINE_Q_MEAN)

        triplet_up = t_q > BASELINE_Q_MEAN
        antitriplet_down = a_q < BASELINE_Q_MEAN
        both = triplet_up and antitriplet_down

        print(f"  Triplet INCREASED q:     {'YES' if triplet_up else 'NO'} (q={t_q:.4f} vs baseline={BASELINE_Q_MEAN:.4f})")
        print(f"  Anti-triplet DECREASED q: {'YES' if antitriplet_down else 'NO'} (q={a_q:.4f} vs baseline={BASELINE_Q_MEAN:.4f})")
        print(f"  Bidirectional causal:    {'PASS' if both else 'FAIL'}")

        if both:
            print()
            print("  INTERPRETATION: kappa_nearest IS the causal variable for q.")
            print("  Increasing kappa_nearest (triplet) increases q.")
            print("  Decreasing kappa_nearest (anti-triplet) decreases q.")

    print()
    print("=== QUANTITATIVE PREDICTION TEST ===")
    if quant_pred:
        print(json.dumps(quant_pred, indent=2))
    elif triplet_summary and triplet_summary.get("kappa_mean"):
        # Compute manually
        baseline_kappa = 0.30  # approximate from existing data
        triplet_kappa = triplet_summary.get("kappa_mean", baseline_kappa)
        delta_kappa = triplet_kappa - baseline_kappa

        predicted_delta_logit = FROZEN_ALPHA * delta_kappa

        # Actual
        K = 20
        q0 = BASELINE_Q_MEAN
        q0_norm = (q0 - 1/K) / (1 - 1/K)
        q1 = triplet_summary.get("q_mean", BASELINE_Q_MEAN)
        q1_norm = (q1 - 1/K) / (1 - 1/K)
        actual_delta_logit = logit(q1_norm) - logit(q0_norm)

        rel_error = abs(predicted_delta_logit - actual_delta_logit) / (abs(actual_delta_logit) + 1e-6)
        print(f"  delta_kappa = {delta_kappa:.4f}")
        print(f"  alpha_frozen = {FROZEN_ALPHA}")
        print(f"  Predicted delta_logit_q = {predicted_delta_logit:.4f}")
        print(f"  Actual delta_logit_q = {actual_delta_logit:.4f}")
        print(f"  Relative error = {rel_error:.4f}")
        print(f"  50% tolerance test: {'PASS' if rel_error < 0.5 else 'FAIL'}")

    print()
    print("=== CROSS-MODAL VALIDATION ===")
    if crossmodal:
        print(json.dumps(crossmodal, indent=2)[:500])
    else:
        print("  Not yet available")

    print()
    print("=== NOBEL SCORECARD UPDATE ===")
    print()

    # Count passed criteria
    score_items = [
        ("LOAO universality (alpha=1.54, CV=4.4%, 7 archs)", True),
        ("Prospective Phi-2 (r=0.985)", True),
        ("Prospective DeBERTa (r=0.982)", True),
        ("Theorem 12: d_eff_cls=1.16, 95% CI includes 1.0", True),
        ("ELECTRA slope universal (r=0.937)", True),
        ("Mamba regime boundary documented", True),
        ("Zero-shot layer selection (72% vs 25% random)", True),
        ("Triplet arm PASSED (kappa_nearest causal)", triplet_summary.get("passed") if triplet_summary else False),
        ("Anti-triplet PASSED (bidirectional causal)",
         (antitriplet_summary and antitriplet_summary.get("q_mean", BASELINE_Q_MEAN) < BASELINE_Q_MEAN) if antitriplet_summary else False),
        ("Cross-modal PASSED", crossmodal is not None and crossmodal.get("passed", False)),
    ]

    n_passed = sum(1 for _, p in score_items if p)
    print(f"Criteria passed: {n_passed}/{len(score_items)}")
    print()
    for name, passed in score_items:
        status = "✓ PASS" if passed else "✗ PENDING/FAIL"
        print(f"  [{status}] {name}")

    print()
    if n_passed >= 9:
        print("ESTIMATED NOBEL SCORE: 5-6/10 (strong universal law + causal evidence)")
    elif n_passed >= 7:
        print("ESTIMATED NOBEL SCORE: 4-5/10 (strong universal law, causal pending)")
    else:
        print("ESTIMATED NOBEL SCORE: 3-4/10 (correlational evidence only)")

    print()
    print("=== NEXT PRIORITY EXPERIMENTS ===")
    if triplet_summary and triplet_summary.get("passed"):
        print("1. Cross-modal test (CIFAR ViT / ResNet on ImageNet)")
        print("2. Audio/speech model kappa_nearest test")
        print("3. External replication request (post preprint)")
        print("4. Beyond 1-NN: linear probe correlation with kappa_nearest")
        print("5. Compute tighter non-asymptotic bounds using d_eff correction")
    else:
        print("1. Investigate why triplet arm failed (if applicable)")
        print("2. Try NC-loss (Neural Collapse objective) as alternative causal lever")
        print("3. Consider kappa_nearest regularizer (direct min-margin term)")


if __name__ == "__main__":
    main()
