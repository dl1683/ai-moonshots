"""
Pair Coupling Analysis Post-Processor
Analyzes results from cti_pair_coupling.py when complete.
Tests: K_eff_pred(i) = rank_eff(V_i) = tr(V_i)^2 / tr(V_i^2)
"""
import json
import numpy as np
from scipy import stats
import sys

RESULTS_PATH = "results/cti_pair_coupling.json"
LOG_PATH = "results/cti_pair_coupling_log.txt"

def analyze_pair_coupling():
    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Results file not found yet. Experiment still running.")
        return

    print("=" * 70)
    print("PAIR COUPLING ANALYSIS")
    print("=" * 70)
    print(f"Pre-registered: R2 > 0.5 AND Spearman rho > 0.5 for K_eff_obs vs K_eff_pred")
    print()

    # Collect all (K_eff_obs, K_eff_pred, c_obs, f_sub) pairs per r value
    records_r5 = []
    records_r10 = []
    all_c_r5 = []
    all_c_r10 = []

    for seed_data in data.get("seeds", []):
        seed = seed_data["seed"]
        for cls_data in seed_data.get("classes", []):
            cls_idx = cls_data["target_class"]
            K_eff_pred = cls_data["K_eff_pred"]
            f_sub = cls_data["f_sub"]
            kappa_eff = cls_data.get("kappa_eff", None)

            for r_data in cls_data.get("r_results", []):
                r = r_data["r"]
                K_eff_obs = r_data["K_eff_obs"]
                c_obs = r_data["c_obs"]
                if abs(r - 5.0) < 0.1:
                    records_r5.append({
                        "seed": seed, "cls": cls_idx,
                        "K_eff_obs": K_eff_obs, "K_eff_pred": K_eff_pred,
                        "c_obs": c_obs, "f_sub": f_sub
                    })
                    all_c_r5.append(c_obs)
                elif abs(r - 10.0) < 0.1:
                    records_r10.append({
                        "seed": seed, "cls": cls_idx,
                        "K_eff_obs": K_eff_obs, "K_eff_pred": K_eff_pred,
                        "c_obs": c_obs, "f_sub": f_sub
                    })
                    all_c_r10.append(c_obs)

    # Combine r5 and r10 for joint analysis
    all_K_eff_obs = [r["K_eff_obs"] for r in records_r5] + [r["K_eff_obs"] for r in records_r10]
    all_K_eff_pred = [r["K_eff_pred"] for r in records_r5] + [r["K_eff_pred"] for r in records_r10]
    all_f_sub = [r["f_sub"] for r in records_r5] + [r["f_sub"] for r in records_r10]

    n_total = len(all_K_eff_obs)
    print(f"Total data points: {n_total} ({len(records_r5)} at r=5, {len(records_r10)} at r=10)")
    print()

    if n_total < 5:
        print("Not enough data points for analysis.")
        return

    # R2 and Spearman for K_eff_obs vs K_eff_pred
    K_obs_arr = np.array(all_K_eff_obs)
    K_pred_arr = np.array(all_K_eff_pred)
    f_sub_arr = np.array(all_f_sub)

    # R2 for K_eff_pred
    ss_res = np.sum((K_obs_arr - K_pred_arr)**2)
    ss_tot = np.sum((K_obs_arr - K_obs_arr.mean())**2)
    r2_K_eff = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    spearman_K_eff, p_spearman_K_eff = stats.spearmanr(K_obs_arr, K_pred_arr)
    pearson_K_eff, p_pearson_K_eff = stats.pearsonr(K_obs_arr, K_pred_arr)

    # R2 for f_sub as predictor of c_obs (c_obs = K_eff_obs / d_eff)
    # For c_obs vs f_sub analysis
    c_obs_r5 = np.array([r["c_obs"] for r in records_r5])
    f_sub_r5 = np.array([r["f_sub"] for r in records_r5])
    if len(c_obs_r5) > 2:
        ss_res_c = np.sum((c_obs_r5 - f_sub_r5)**2)
        ss_tot_c = np.sum((c_obs_r5 - c_obs_r5.mean())**2)
        r2_c_fsub = 1.0 - ss_res_c / ss_tot_c if ss_tot_c > 0 else 0.0
        spearman_c, p_spearman_c = stats.spearmanr(c_obs_r5, f_sub_r5)
    else:
        r2_c_fsub = float('nan')
        spearman_c, p_spearman_c = float('nan'), float('nan')

    print("=" * 70)
    print("PRIMARY TEST: K_eff_obs vs K_eff_pred = rank_eff(V_i)")
    print("=" * 70)
    print(f"  R2(K_eff_obs, K_eff_pred):    {r2_K_eff:.4f}  {'PASS' if r2_K_eff > 0.5 else 'FAIL'} (threshold: 0.5)")
    print(f"  Spearman rho:                  {spearman_K_eff:.4f}  {'PASS' if spearman_K_eff > 0.5 else 'FAIL'} (threshold: 0.5)")
    print(f"  Pearson r:                     {pearson_K_eff:.4f}  (p={p_pearson_K_eff:.4f})")
    print()
    print(f"  OVERALL: {'PASS' if r2_K_eff > 0.5 and spearman_K_eff > 0.5 else 'FAIL'}")
    print()

    print("=" * 70)
    print("SECONDARY TEST: c_obs vs f_sub = tr(V_i)/tr(W) at r=5")
    print("=" * 70)
    print(f"  R2(c_obs, f_sub):              {r2_c_fsub:.4f}")
    print(f"  Spearman rho:                  {spearman_c:.4f}  (p={p_spearman_c:.4f})")
    print()

    # r-invariance test: Pearson(c_r5, c_r10) > 0.8
    print("=" * 70)
    print("R-INVARIANCE: Pearson(c_r5, c_r10) per class/seed")
    print("=" * 70)
    # Match records_r5 and records_r10 by (seed, cls)
    r5_dict = {(r["seed"], r["cls"]): r["c_obs"] for r in records_r5}
    r10_dict = {(r["seed"], r["cls"]): r["c_obs"] for r in records_r10}
    common_keys = sorted(set(r5_dict.keys()) & set(r10_dict.keys()))
    if len(common_keys) > 2:
        c5_matched = [r5_dict[k] for k in common_keys]
        c10_matched = [r10_dict[k] for k in common_keys]
        pearson_rinv, p_rinv = stats.pearsonr(c5_matched, c10_matched)
        print(f"  Matched pairs: {len(common_keys)}")
        print(f"  Pearson(c_r5, c_r10):         {pearson_rinv:.4f}  {'PASS' if pearson_rinv > 0.8 else 'FAIL'} (threshold: 0.8)")
        print(f"  Mean c_r5: {np.mean(c5_matched):.3f}, Mean c_r10: {np.mean(c10_matched):.3f}")
    else:
        print("  Not enough matched pairs yet.")
        pearson_rinv = float('nan')

    print()

    # Descriptive statistics
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(f"  K_eff_obs range:  [{K_obs_arr.min():.2f}, {K_obs_arr.max():.2f}]  mean={K_obs_arr.mean():.2f}  std={K_obs_arr.std():.2f}")
    print(f"  K_eff_pred range: [{K_pred_arr.min():.2f}, {K_pred_arr.max():.2f}]  mean={K_pred_arr.mean():.2f}  std={K_pred_arr.std():.2f}")
    print(f"  f_sub range:      [{f_sub_arr.min():.3f}, {f_sub_arr.max():.3f}]  mean={f_sub_arr.mean():.3f}")

    # Per-seed breakdown
    print()
    print("=" * 70)
    print("PER-SEED BREAKDOWN (r=5)")
    print("=" * 70)
    for seed in sorted(set(r["seed"] for r in records_r5)):
        seed_recs = [r for r in records_r5 if r["seed"] == seed]
        if len(seed_recs) < 3:
            continue
        K_obs_s = [r["K_eff_obs"] for r in seed_recs]
        K_pred_s = [r["K_eff_pred"] for r in seed_recs]
        ss_res_s = sum((o-p)**2 for o,p in zip(K_obs_s, K_pred_s))
        ss_tot_s = sum((o-np.mean(K_obs_s))**2 for o in K_obs_s)
        r2_s = 1.0 - ss_res_s/ss_tot_s if ss_tot_s > 0 else 0.0
        rho_s, _ = stats.spearmanr(K_obs_s, K_pred_s)
        print(f"  Seed {seed}: n={len(seed_recs)}, R2={r2_s:.3f}, Spearman={rho_s:.3f}")
        # Show top-5 by K_eff_obs
        sorted_recs = sorted(seed_recs, key=lambda r: r["K_eff_obs"], reverse=True)
        print(f"    Top-5 by K_eff_obs:")
        for rec in sorted_recs[:5]:
            print(f"      cls={rec['cls']:2d}: K_eff_obs={rec['K_eff_obs']:.2f}, K_eff_pred={rec['K_eff_pred']:.2f}, c_obs={rec['c_obs']:.3f}, f_sub={rec['f_sub']:.3f}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    overall_pass = r2_K_eff > 0.5 and spearman_K_eff > 0.5
    print(f"  PRIMARY (K_eff_pred): {'PASS' if overall_pass else 'FAIL'}")
    print(f"  R2={r2_K_eff:.4f}, Spearman={spearman_K_eff:.4f}")
    if not np.isnan(pearson_rinv):
        print(f"  R-INVARIANCE: {'PASS' if pearson_rinv > 0.8 else 'FAIL'} (r={pearson_rinv:.4f})")
    print()
    print(f"INTERPRETATION:")
    if overall_pass:
        print("  K_eff_pred = rank_eff(V_i) is a VALID zero-parameter geometric predictor.")
        print("  The effective number of competitors is encoded in the projected covariance V_i.")
        print("  This suggests K_eff reflects the anisotropy of within-class variance in centroid space.")
    else:
        print("  K_eff_pred = rank_eff(V_i) FAILS to predict K_eff_obs.")
        if r2_K_eff > 0.0:
            print(f"  Some correlation exists (R2={r2_K_eff:.3f}) but below threshold.")
            print("  Alternative: K_eff_obs may be better predicted by tr(V_i)/V_i[0,0] (first competitor dominance)")
        else:
            print("  No correlation — K_eff_obs is not driven by centroid subspace geometry.")

if __name__ == "__main__":
    analyze_pair_coupling()
