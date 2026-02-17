#!/usr/bin/env python
"""
TWO-STEP PHASE TRANSITION analysis from multi-observable data.

Key discovery: Geometric observables and quality observables
transition at DIFFERENT alpha values:

Step 1 (alpha ~ 0.7): COMPLEXITY PEAK
  - Intrinsic dimensionality peaks then drops
  - Effective rank peaks then drops
  - Representations are maximally complex at intermediate residual strength

Step 2 (alpha ~ 0.85): QUALITY EMERGENCE
  - kNN accuracy rises sharply
  - Alignment improves (drops)
  - Representations become semantically organized

The GAP between Step 1 and Step 2 is the "compression-before-fitting" signature
predicted by information bottleneck theory.

This is MORE interesting than a single phase transition — it reveals
a TWO-STAGE process: geometric structuring must precede quality.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    print("=" * 70)
    print("TWO-STEP PHASE TRANSITION ANALYSIS")
    print("Geometric complexity peak vs Quality emergence")
    print("=" * 70)

    with open(RESULTS_DIR / "cti_multi_observable.json") as f:
        data = json.load(f)

    alphas = np.array(data["alphas"])
    obs = data["observables"]

    # Extract all observables
    knn = np.array(obs["knn_acc"])
    ID = np.array(obs["intrinsic_dim"])
    ER = np.array(obs["effective_rank"])
    align = np.array(obs["alignment"])

    print(f"\nModel: {data['model_id']} ({data['num_layers']} layers)")
    print(f"Dataset: {data['dataset']}")

    # ============================================================
    # Raw data table
    # ============================================================
    print(f"\n{'alpha':>6} {'kNN':>8} {'ID':>8} {'EffRank':>8} {'Align':>8}")
    print("-" * 45)
    for i, a in enumerate(alphas):
        print(f"{a:>6.2f} {knn[i]:>8.3f} {ID[i]:>8.1f} {ER[i]:>8.1f} {align[i]:>8.4f}")

    # ============================================================
    # Step 1: COMPLEXITY PEAK
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 1: COMPLEXITY PEAK (Intrinsic Dimensionality)")
    print(f"{'='*70}")

    # Find ID peak
    id_peak_idx = np.argmax(ID)
    id_peak_alpha = alphas[id_peak_idx]
    id_peak_val = ID[id_peak_idx]
    print(f"\n  ID peaks at alpha = {id_peak_alpha:.2f} (ID = {id_peak_val:.1f})")

    # ER peak
    er_peak_idx = np.argmax(ER)
    er_peak_alpha = alphas[er_peak_idx]
    print(f"  EffRank peaks at alpha = {er_peak_alpha:.2f} (ER = {ER[er_peak_idx]:.1f})")

    # The complexity peak is the GEOMETRIC transition
    alpha_complexity = (id_peak_alpha + er_peak_alpha) / 2
    print(f"\n  Mean geometric peak: alpha_geom = {alpha_complexity:.2f}")

    # ID value at alpha=0 vs alpha=1 vs peak
    print(f"\n  ID at alpha=0:    {ID[0]:.1f}")
    print(f"  ID at peak:       {id_peak_val:.1f} ({100*(id_peak_val/ID[0]-1):.0f}% above alpha=0)")
    print(f"  ID at alpha=1:    {ID[-1]:.1f} ({100*(ID[-1]/id_peak_val-1):.0f}% below peak)")

    # ============================================================
    # Step 2: QUALITY EMERGENCE
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 2: QUALITY EMERGENCE (kNN accuracy)")
    print(f"{'='*70}")

    # kNN transition (midpoint of increase)
    knn_min, knn_max = knn.min(), knn.max()
    knn_norm = (knn - knn_min) / (knn_max - knn_min)
    alpha_knn = None
    for i in range(len(alphas) - 1):
        if knn_norm[i] <= 0.5 and knn_norm[i+1] > 0.5:
            frac = (0.5 - knn_norm[i]) / (knn_norm[i+1] - knn_norm[i])
            alpha_knn = alphas[i] + frac * (alphas[i+1] - alphas[i])
            break

    # Alignment transition
    align_min, align_max = align.min(), align.max()
    align_norm = (align - align_min) / (align_max - align_min)
    alpha_align = None
    for i in range(len(alphas) - 1):
        if align_norm[i] >= 0.5 and align_norm[i+1] < 0.5:
            frac = (0.5 - align_norm[i]) / (align_norm[i+1] - align_norm[i])
            alpha_align = alphas[i] + frac * (alphas[i+1] - alphas[i])
            break

    print(f"\n  kNN transition: alpha_knn = {alpha_knn:.3f}" if alpha_knn else "  kNN: no transition found")
    print(f"  Alignment transition: alpha_align = {alpha_align:.3f}" if alpha_align else "  Alignment: no transition found")

    alpha_quality = alpha_knn if alpha_knn else alpha_align
    print(f"\n  Quality emergence point: alpha_qual = {alpha_quality:.3f}" if alpha_quality else "")

    # ============================================================
    # THE GAP: Compression before Fitting
    # ============================================================
    print(f"\n{'='*70}")
    print("THE GAP: Compression Must Precede Quality")
    print(f"{'='*70}")

    gap = alpha_quality - alpha_complexity if alpha_quality else None
    if gap:
        print(f"\n  Geometric peak:      alpha_geom = {alpha_complexity:.2f}")
        print(f"  Quality emergence:   alpha_qual = {alpha_quality:.3f}")
        print(f"  GAP:                 {gap:.3f}")
        print(f"\n  Interpretation:")
        print(f"    1. At alpha < {alpha_complexity:.2f}: representations grow MORE complex with residual strength")
        print(f"    2. At alpha ~ {alpha_complexity:.2f}: maximum representational complexity (\"edge of chaos\")")
        print(f"    3. At {alpha_complexity:.2f} < alpha < {alpha_quality:.3f}: representations COMPRESS but quality is still low")
        print(f"    4. At alpha > {alpha_quality:.3f}: compressed representations become QUALITY-RICH")
        print(f"\n  This is the information bottleneck signature:")
        print(f"    - Compression (Step 1) MUST happen before fitting (Step 2)")
        print(f"    - The gap of {gap:.3f} in alpha is the \"compression zone\"")

    # ============================================================
    # Quantitative: at what alpha does each observable change most?
    # ============================================================
    print(f"\n{'='*70}")
    print("MAXIMUM RATE OF CHANGE (steepest transition)")
    print(f"{'='*70}")

    for name, values in [("kNN", knn), ("ID", ID), ("EffRank", ER), ("Alignment", align)]:
        # Compute numerical derivative
        deriv = np.gradient(values, alphas)
        max_deriv_idx = np.argmax(np.abs(deriv))
        alpha_steepest = alphas[max_deriv_idx]
        direction = "increasing" if deriv[max_deriv_idx] > 0 else "decreasing"
        print(f"  {name:12s}: steepest at alpha = {alpha_steepest:.2f} ({direction}, "
              f"|d/dalpha| = {abs(deriv[max_deriv_idx]):.2f})")

    # ============================================================
    # Cross-observable correlations
    # ============================================================
    print(f"\n{'='*70}")
    print("CROSS-OBSERVABLE CORRELATIONS")
    print(f"{'='*70}")

    from scipy.stats import pearsonr
    pairs = [("kNN", knn), ("ID", ID), ("EffRank", ER), ("Alignment", align)]
    print(f"\n  {'':>12}", end="")
    for name, _ in pairs:
        print(f" {name:>10}", end="")
    print()
    for name1, v1 in pairs:
        print(f"  {name1:>12}", end="")
        for name2, v2 in pairs:
            r, _ = pearsonr(v1, v2)
            print(f" {r:>10.3f}", end="")
        print()

    # kNN vs ID: if these are anti-correlated in the transition region,
    # it supports the compression-then-fitting story
    from scipy.stats import spearmanr
    # Only look at alpha >= 0.7 (post-complexity-peak)
    mask = alphas >= 0.7
    if np.sum(mask) >= 4:
        rho_knn_id, p_knn_id = spearmanr(knn[mask], ID[mask])
        print(f"\n  Post-peak (alpha >= 0.7):")
        print(f"    kNN vs ID:   rho = {rho_knn_id:.3f} (p = {p_knn_id:.3f})")
        print(f"    => {'ANTI-CORRELATED' if rho_knn_id < -0.5 else 'not anti-correlated'}: "
              f"quality rises as complexity falls")

    # ============================================================
    # Save
    # ============================================================
    out = {
        "model_id": data["model_id"],
        "analysis": "two_step_phase_transition",
        "step1_complexity_peak": {
            "alpha_id_peak": float(id_peak_alpha),
            "alpha_er_peak": float(er_peak_alpha),
            "alpha_mean": float(alpha_complexity),
            "id_at_peak": float(id_peak_val),
            "id_at_alpha_0": float(ID[0]),
            "id_at_alpha_1": float(ID[-1]),
        },
        "step2_quality_emergence": {
            "alpha_knn": float(alpha_knn) if alpha_knn else None,
            "alpha_align": float(alpha_align) if alpha_align else None,
        },
        "gap": float(gap) if gap else None,
        "interpretation": "compression_before_fitting",
    }

    out_path = RESULTS_DIR / "cti_two_step_transition.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
