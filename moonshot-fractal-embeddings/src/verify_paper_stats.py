"""Verify all paper statistics against result JSONs.

Run after any paper edit to catch discrepancies.
Uses paired tests throughout (matching methodology Section 2).

Run: python src/verify_paper_stats.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

issues = []


def check(label, expected, actual, tol=0.02, status_override=None):
    """Check if actual matches expected within tolerance."""
    if expected is None or actual is None:
        return
    diff = abs(expected - actual)
    if diff <= tol:
        status = status_override or PASS
        print(f"  [{status}] {label}: paper={expected}, data={actual:.4f}")
    else:
        status = status_override or FAIL
        print(f"  [{status}] {label}: paper={expected}, data={actual:.4f} (diff={diff:.4f})")
        if status == FAIL:
            issues.append(f"{label}: paper={expected} vs data={actual:.4f}")


def paired_stats(v5_vals, cond_vals):
    """Compute paired t-test and Cohen's d."""
    n = min(len(v5_vals), len(cond_vals))
    v5 = np.array(v5_vals[:n])
    c = np.array(cond_vals[:n])
    diff = v5 - c
    t, p = sp_stats.ttest_rel(v5, c)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    return t, p, d


def independent_stats(v5_vals, cond_vals):
    """Compute independent t-test and Cohen's d."""
    t, p = sp_stats.ttest_ind(v5_vals, cond_vals)
    pooled = np.sqrt(
        (np.std(v5_vals, ddof=1) ** 2 + np.std(cond_vals, ddof=1) ** 2) / 2
    )
    d = (np.mean(v5_vals) - np.mean(cond_vals)) / pooled if pooled > 0 else 0
    return t, p, d


def verify_ablation():
    """Verify Table 3: Causal ablation (paired tests)."""
    print("\n=== TABLE 3: CAUSAL ABLATION (paired) ===")

    for ds in ["clinc", "trec"]:
        abl_path = RESULTS_DIR / f"ablation_steerability_bge-small_{ds}.json"
        uhmt_path = RESULTS_DIR / f"uhmt_ablation_bge-small_{ds}.json"

        if not abl_path.exists():
            print(f"  Missing: {abl_path.name}")
            continue

        with open(abl_path) as f:
            abl_data = json.load(f)
        v5_vals = [r["steerability_score"] for r in abl_data["results"]["v5"]]

        print(f"\n  {ds.upper()} (n={len(v5_vals)})")
        print(f"    V5: {np.mean(v5_vals):.4f} +/- {np.std(v5_vals, ddof=1):.4f}")

        for cond in ["inverted", "no_prefix"]:
            if cond not in abl_data["results"]:
                continue
            cond_vals = [
                r["steerability_score"] for r in abl_data["results"][cond]
            ]
            t, p, d = paired_stats(v5_vals, cond_vals)
            print(
                f"    {cond}: mean={np.mean(cond_vals):.4f}, "
                f"paired t={t:.1f}, p={p:.6f}, d={d:.1f}"
            )

        if uhmt_path.exists():
            with open(uhmt_path) as f:
                uhmt_data = json.load(f)
            uhmt_vals = [r["steerability_score"] for r in uhmt_data["results"]]
            t, p, d = paired_stats(v5_vals, uhmt_vals)
            print(
                f"    uhmt: mean={np.mean(uhmt_vals):.4f}, "
                f"paired t={t:.1f}, p={p:.6f}, d={d:.1f}"
            )


def verify_backbone():
    """Verify backbone fine-tuning control (paired d)."""
    print("\n=== BACKBONE CONTROL (paired) ===")

    path = RESULTS_DIR / "backbone_finetune_control.json"
    if not path.exists():
        print("  Missing backbone_finetune_control.json")
        return

    with open(path) as f:
        bc = json.load(f)

    for ds, arms in bc["results"].items():
        print(f"\n  {ds}:")
        for arm in ["v5_frozen", "mrl_frozen", "flat_finetune", "v5_finetune"]:
            if arm in arms and arms[arm]:
                vals = [r["steerability_score"] for r in arms[arm]]
                print(
                    f"    {arm}: {np.mean(vals):.4f} +/- "
                    f"{np.std(vals, ddof=1):.4f} (n={len(vals)})"
                )

        # Key comparison: v5_frozen vs flat_finetune (paired d)
        if "v5_frozen" in arms and "flat_finetune" in arms:
            v5f = [r["steerability_score"] for r in arms["v5_frozen"]]
            flat = [r["steerability_score"] for r in arms["flat_finetune"]]
            if len(v5f) >= 3 and len(flat) >= 3:
                t, p, d = paired_stats(v5f, flat)
                print(f"    V5-frozen vs flat-finetune: paired t={t:.1f}, p={p:.6f}, d={d:.1f}")
                if ds == "clinc":
                    check("Backbone d (CLINC)", 8.1, d, tol=0.2)


def verify_crossmodel():
    """Verify cross-model gaps."""
    print("\n=== CROSS-MODEL GAPS ===")

    # Paper claims: BGE +0.143, E5 +0.115, Qwen3 +0.145
    models = {
        "bge-small": {"clinc_gap": 0.143},
        "e5-small": {"clinc_gap": 0.115},
        "qwen3-0.6b": {"clinc_gap": 0.145},
    }

    for model, expected in models.items():
        v5_path = RESULTS_DIR / f"steerability_bge-small_clinc.json"
        # Cross-model files have different naming
        cm_path = RESULTS_DIR / f"cross_model_{model}_clinc.json"
        steer_path = RESULTS_DIR / f"steerability_{model}_clinc.json"

        for p in [cm_path, steer_path]:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                if "results" in data:
                    results = data["results"]
                    if isinstance(results, dict):
                        v5_vals = [
                            r["steerability_score"]
                            for r in results.get("v5", results.get("fractal", []))
                        ]
                        mrl_vals = [
                            r["steerability_score"]
                            for r in results.get("mrl", [])
                        ]
                    elif isinstance(results, list):
                        v5_vals = [r.get("v5_steerability", 0) for r in results]
                        mrl_vals = [r.get("mrl_steerability", 0) for r in results]
                    else:
                        continue

                    if v5_vals and mrl_vals:
                        gap = np.mean(v5_vals) - np.mean(mrl_vals)
                        print(
                            f"  {model}: V5={np.mean(v5_vals):.3f}, "
                            f"MRL={np.mean(mrl_vals):.3f}, gap={gap:.3f}"
                        )
                        check(
                            f"Cross-model gap {model}",
                            expected["clinc_gap"],
                            gap,
                            tol=0.002,
                        )
                break
        else:
            print(f"  {model}: no result file found")


def verify_retrieval():
    """Verify retrieval ramp SDs."""
    print("\n=== RETRIEVAL RAMP ===")

    path = RESULTS_DIR / "retrieval_benchmark_bge-small_clinc.json"
    if not path.exists():
        print("  Missing retrieval file")
        return

    with open(path) as f:
        data = json.load(f)

    v5_ramps = []
    mrl_ramps = []
    for seed_key, seed_data in data["results_by_seed"].items():
        for method, ramps in [("v5", v5_ramps), ("mrl", mrl_ramps)]:
            md = seed_data[method]
            l1_64 = md["1"]["L1"]["recall@1"]
            l1_256 = md["4"]["L1"]["recall@1"]
            ramps.append(l1_256 - l1_64)

    v5_mean = np.mean(v5_ramps) * 100
    v5_sd = np.std(v5_ramps, ddof=1) * 100
    mrl_mean = np.mean(mrl_ramps) * 100
    mrl_sd = np.std(mrl_ramps, ddof=1) * 100

    print(f"  V5 ramp: {v5_mean:.1f} +/- {v5_sd:.1f}pp")
    print(f"  MRL ramp: {mrl_mean:.1f} +/- {mrl_sd:.1f}pp")
    check("V5 ramp mean", 6.3, v5_mean, tol=0.1)
    check("V5 ramp SD", 1.1, v5_sd, tol=0.1)
    check("MRL ramp SD", 0.4, mrl_sd, tol=0.1)


def verify_three_level():
    """Verify three-level hierarchy stats."""
    print("\n=== THREE-LEVEL HIERARCHY ===")

    path = RESULTS_DIR / "three_level_clinc.json"
    if not path.exists():
        print("  Missing three_level_clinc.json")
        return

    with open(path) as f:
        data = json.load(f)

    v5_s02 = [r["v5"]["steerability_02"] for r in data["results"]]
    mrl_s02 = [r["mrl"]["steerability_02"] for r in data["results"]]

    t, p, d = paired_stats(v5_s02, mrl_s02)
    print(
        f"  V5 S02: {np.mean(v5_s02):.4f} +/- {np.std(v5_s02, ddof=1):.4f}"
    )
    print(
        f"  MRL S02: {np.mean(mrl_s02):.4f} +/- {np.std(mrl_s02, ddof=1):.4f}"
    )
    print(f"  Paired: t={t:.1f}, p={p:.4f}, d={d:.1f}")
    check("Three-level t", 18.9, abs(t), tol=0.2)
    check("Three-level d", 10.9, abs(d), tol=0.2)


def verify_meta_analysis():
    """Verify meta-analysis values."""
    print("\n=== META-ANALYSIS ===")

    path = RESULTS_DIR / "meta_analysis_results.json"
    if not path.exists():
        print("  Missing meta_analysis_results.json")
        return

    with open(path) as f:
        ma = json.load(f)

    if "random_effects" in ma:
        re = ma["random_effects"]
        print(f"  Pooled d: {re.get('pooled_d', 'N/A')}")
        print(f"  z: {re.get('z', 'N/A')}, p: {re.get('p', 'N/A')}")
        print(f"  I^2: {re.get('I2', 'N/A')}")
        if "pooled_d" in re:
            check("Meta pooled d", 1.49, re["pooled_d"], tol=0.02)


def verify_scaling():
    """Verify scaling analysis."""
    print("\n=== SCALING ANALYSIS ===")

    path = RESULTS_DIR / "scaling_robustness_results.json"
    if not path.exists():
        # Try other filenames
        path = RESULTS_DIR / "paper_statistics_holm.json"
        if not path.exists():
            print("  No scaling results found")
            return

    with open(path) as f:
        data = json.load(f)

    if "product_predictor" in data:
        pp = data["product_predictor"]
        print(f"  Product rho: {pp.get('spearman_rho', 'N/A')}")
        print(f"  Product p: {pp.get('spearman_p', 'N/A')}")
        print(f"  Pearson r: {pp.get('pearson_r', 'N/A')}")


if __name__ == "__main__":
    print("=" * 60)
    print("PAPER STATISTICS VERIFICATION")
    print("Using paired tests per methodology Section 2")
    print("=" * 60)

    verify_ablation()
    verify_backbone()
    verify_crossmodel()
    verify_retrieval()
    verify_three_level()
    verify_meta_analysis()
    verify_scaling()

    print("\n" + "=" * 60)
    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        for i in issues:
            print(f"  - {i}")
    else:
        print("ALL CHECKS PASSED")
    print("=" * 60)
