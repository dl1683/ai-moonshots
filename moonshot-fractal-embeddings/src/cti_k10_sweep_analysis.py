"""
Analyze K=10 kappa sweep results once JSON is written.
Evaluates PR1-PR5 from the pre-registration.

Pre-registered in results/cti_k10_kappa_sweep.json (committed with src/cti_k10_kappa_sweep.py).
"""
import json
import numpy as np
from scipy.stats import pearsonr
import sys
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
JSON_PATH = os.path.join(RESULTS_DIR, "cti_k10_kappa_sweep.json")
PRIOR_C0 = 0.816  # from multi-K sweep (K=7 result)


def main():
    if not os.path.exists(JSON_PATH):
        print("ERROR: cti_k10_kappa_sweep.json not yet written. Sweep still running.")
        sys.exit(1)

    with open(JSON_PATH) as f:
        data = json.load(f)

    print("=" * 60)
    print("K=10 KAPPA SWEEP ANALYSIS")
    print("=" * 60)

    # Extract aggregate results per kappa
    aggregates = data.get("aggregates", {})
    kappas = sorted([float(k) for k in aggregates.keys()])

    B_obs_list = []
    B_pred_list = []
    C_0_list = []

    print("\nPer-kappa summary:")
    print(f"{'kappa':>8} {'B_obs':>8} {'B_pred':>8} {'rel_err':>8} {'C_0_obs':>8} {'PR1':>6}")
    for kappa in kappas:
        agg = aggregates[str(kappa)]
        B_obs = agg["B_obs_mean"]
        B_pred = agg["B_pred"]
        rel_err = (B_obs - B_pred) / B_pred
        C_0 = agg["C_0_obs_mean"]
        pr1 = "PASS" if abs(rel_err) < 0.30 else "FAIL"
        print(f"{kappa:>8.2f} {B_obs:>8.4f} {B_pred:>8.4f} {rel_err:>8.3f} {C_0:>8.4f} {pr1:>6}")
        B_obs_list.append(B_obs)
        B_pred_list.append(B_pred)
        C_0_list.append(C_0)

    n = len(kappas)
    print(f"\nN kappa values: {n}")

    # PR1: All within 30%
    rel_errs = [(B_obs_list[i] - B_pred_list[i]) / B_pred_list[i] for i in range(n)]
    pr1 = all(abs(e) < 0.30 for e in rel_errs)
    pr1_max = max(abs(e) for e in rel_errs)
    print(f"\nPR1 (all |rel_err| < 30%): {'PASS' if pr1 else 'FAIL'} (max={pr1_max:.3f})")

    # PR2: Log-log slope in [-0.90, -0.55]
    log_k = np.log(kappas)
    log_B = np.log(B_obs_list)
    slope, intercept = np.polyfit(log_k, log_B, 1)
    pr2 = -0.90 <= slope <= -0.55
    print(f"PR2 (log-log slope in [-0.90,-0.55]): {'PASS' if pr2 else 'FAIL'} (slope={slope:.4f})")

    # PR3: Mean C_0 within ±20% of prior
    mean_C0 = np.mean(C_0_list)
    ratio = mean_C0 / PRIOR_C0
    pr3 = 0.80 <= ratio <= 1.20
    print(f"PR3 (mean C_0 within +/-20% of {PRIOR_C0}): {'PASS' if pr3 else 'FAIL'} "
          f"(mean={mean_C0:.4f}, ratio={ratio:.3f})")

    # PR4: C_0 CV < 25%
    cv_C0 = np.std(C_0_list) / mean_C0
    pr4 = cv_C0 < 0.25
    print(f"PR4 (C_0 CV < 25%): {'PASS' if pr4 else 'FAIL'} (CV={cv_C0:.3f})")

    # PR5: Pearson p < 0.05
    r, p = pearsonr(log_k, log_B)
    pr5 = p < 0.05
    print(f"PR5 (Pearson r B vs kappa, p < 0.05): {'PASS' if pr5 else 'FAIL'} "
          f"(r={r:.4f}, p={p:.4f})")

    # Overall verdict
    prs = [pr1, pr2, pr3, pr4, pr5]
    n_pass = sum(prs)
    print(f"\n{'='*60}")
    print(f"VERDICT: {n_pass}/5 PRs PASS")
    if n_pass == 5:
        print("STRONG_PASS: All pre-registered criteria satisfied.")
    elif n_pass >= 3:
        print("PARTIAL_PASS: Majority of criteria satisfied.")
    else:
        print("FAIL: Fewer than 3 criteria satisfied.")

    # C_0 interpretation
    print(f"\nC_0 interpretation:")
    print(f"  Prior C_0 (K=7): {PRIOR_C0}")
    print(f"  K=10 mean C_0:   {mean_C0:.4f}")
    print(f"  Ratio K10/K7:    {ratio:.3f}")
    if ratio < 0.95:
        print(f"  NOTE: C_0 is {100*(1-ratio):.1f}% below prior — consistent with easy-class")
        print(f"  dilution at K=10 (8 easy classes pull q_base toward 1, diluting B signal).")

    # Save analysis
    analysis = {
        "n_kappa": n,
        "kappas": kappas,
        "B_obs": B_obs_list,
        "B_pred": B_pred_list,
        "C_0_obs": C_0_list,
        "log_log_slope": float(slope),
        "log_log_intercept": float(intercept),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mean_C0": float(mean_C0),
        "prior_C0": PRIOR_C0,
        "C0_ratio": float(ratio),
        "C0_cv": float(cv_C0),
        "pr1_pass": pr1,
        "pr2_pass": pr2,
        "pr3_pass": pr3,
        "pr4_pass": pr4,
        "pr5_pass": pr5,
        "n_pass": n_pass,
        "verdict": "STRONG_PASS" if n_pass == 5 else ("PARTIAL_PASS" if n_pass >= 3 else "FAIL"),
    }
    out_path = os.path.join(RESULTS_DIR, "cti_k10_sweep_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    main()
