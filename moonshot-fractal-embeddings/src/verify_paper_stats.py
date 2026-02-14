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


def compute_steerability_from_prefix(prefix_acc):
    """Compute steerability from prefix_accuracy dict."""
    j1_l0 = prefix_acc.get("j1_l0", 0)
    j4_l0 = prefix_acc.get("j4_l0", 0)
    j1_l1 = prefix_acc.get("j1_l1", 0)
    j4_l1 = prefix_acc.get("j4_l1", 0)
    return (j1_l0 - j4_l0) + (j4_l1 - j1_l1)


def verify_crossmodel():
    """Verify cross-model gaps."""
    print("\n=== CROSS-MODEL GAPS ===")

    # Paper claims: BGE +0.143, E5 +0.115, Qwen3 +0.145
    # Actual file names differ per model
    models = {
        "bge-small": {
            "clinc_gap": 0.143,
            "v5_file": "benchmark_bge-small_clinc.json",
        },
        "e5-small": {
            "clinc_gap": 0.115,
            "v5_file": "benchmark_e5-small_clinc.json",
        },
        "qwen3-0.6b": {
            "clinc_gap": 0.145,
            "v5_file": "crossmodel_qwen3-0.6b_clinc.json",
        },
    }

    for model, expected in models.items():
        path = RESULTS_DIR / expected["v5_file"]
        if not path.exists():
            print(f"  {model}: missing {expected['v5_file']}")
            continue

        with open(path) as f:
            data = json.load(f)

        v5_steers = []
        mrl_steers = []

        for seed_key, seed_data in data.get("v5", {}).items():
            pa = seed_data.get("prefix_accuracy", {})
            if pa:
                v5_steers.append(compute_steerability_from_prefix(pa))

        for seed_key, seed_data in data.get("mrl", {}).items():
            pa = seed_data.get("prefix_accuracy", {})
            if pa:
                mrl_steers.append(compute_steerability_from_prefix(pa))

        if v5_steers and mrl_steers:
            gap = np.mean(v5_steers) - np.mean(mrl_steers)
            print(
                f"  {model}: V5={np.mean(v5_steers):.3f}, "
                f"MRL={np.mean(mrl_steers):.3f}, gap={gap:.3f}"
            )
            check(
                f"Cross-model gap {model}",
                expected["clinc_gap"],
                gap,
                tol=0.002,
            )
        else:
            print(f"  {model}: could not compute steerability from {path.name}")


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

    for fname in ["meta_analysis.json", "meta_analysis_results.json"]:
        path = RESULTS_DIR / fname
        if path.exists():
            break
    else:
        print("  Missing meta-analysis results")
        return

    with open(path) as f:
        ma = json.load(f)

    # Handle both formats: {random_effects: {...}} and {cohens_d: {...}}
    re = ma.get("random_effects", ma.get("cohens_d", {}))
    if re:
        pooled = re.get("pooled_d", re.get("pooled"))
        z_val = re.get("z")
        p_val = re.get("p")
        i2 = re.get("I2")
        print(f"  Pooled d: {pooled}")
        print(f"  z: {z_val}, p: {p_val}")
        print(f"  I^2: {i2}")
        if pooled is not None:
            check("Meta pooled d", 1.87, pooled, tol=0.02)
        if z_val is not None:
            check("Meta z", 4.99, z_val, tol=0.02)
        if p_val is not None:
            check("Meta p", 0.000001, p_val, tol=0.00001)


def verify_scaling():
    """Verify scaling analysis: raw H and product predictor correlations."""
    print("\n=== SCALING ANALYSIS ===")

    # Raw H correlation from scaling_robustness.json
    sr_path = RESULTS_DIR / "scaling_robustness.json"
    if sr_path.exists():
        with open(sr_path) as f:
            sr = json.load(f)
        fc = sr.get("full_correlation", {})
        rho_raw = fc.get("spearman_rho")
        if rho_raw is not None:
            print(f"  Raw H rho: {rho_raw:.4f} (p={fc.get('spearman_p', 'N/A'):.4f})")
            check("Raw H Spearman rho", 0.61, rho_raw, tol=0.02)
        bs = sr.get("bootstrap_rho", {})
        if bs:
            print(f"  Bootstrap frac positive: {bs.get('frac_positive', 'N/A')}")
    else:
        print("  Missing scaling_robustness.json")

    # Product predictor: compute from hierarchy_profiles + benchmark files
    # Replicates paper_figures.py fig5_scaling_law() logic
    profiles_path = RESULTS_DIR / "hierarchy_profiles.json"
    DS_H_FALLBACK = {"dbpedia_classes": 3.17, "wos": 5.05}
    ALL_DS = ["yahoo", "goemotions", "newsgroups", "trec", "arxiv",
              "clinc", "dbpedia_classes", "wos"]

    if not profiles_path.exists():
        print("  Missing hierarchy_profiles.json")
        return

    with open(profiles_path) as f:
        profiles = json.load(f)

    h_vals, prod_vals, s_vals = [], [], []
    for ds in ALL_DS:
        h = profiles.get(ds, {}).get("h_l1_given_l0", DS_H_FALLBACK.get(ds, 0))
        if h == 0:
            continue

        # Get steerability values from benchmark/steerability files
        bench_file = RESULTS_DIR / f"benchmark_bge-small_{ds}.json"
        steer_file = RESULTS_DIR / f"steerability_bge-small_{ds}.json"
        v5_steers = []
        best_l1 = 0

        for bf in [bench_file, steer_file]:
            if bf.exists():
                with open(bf) as f:
                    bd = json.load(f)
                for seed_key, seed_data in bd.get("v5", {}).items():
                    pa = seed_data.get("prefix_accuracy", {})
                    if pa:
                        s = compute_steerability_from_prefix(pa)
                        v5_steers.append(s)
                    if best_l1 == 0:
                        best_l1 = seed_data.get("baseline", {}).get("l1_accuracy", 0)
                if v5_steers:
                    break

        if v5_steers and best_l1 > 0:
            h_vals.append(h)
            prod_vals.append(h * best_l1)
            s_vals.append(np.mean(v5_steers))

    if len(h_vals) >= 5:
        rho_p, p_p = sp_stats.spearmanr(prod_vals, s_vals)
        r_p, pr_p = sp_stats.pearsonr(prod_vals, s_vals)
        print(f"  Product predictor ({len(h_vals)} datasets): "
              f"rho={rho_p:.2f} (p={p_p:.4f}), r={r_p:.2f} (p={pr_p:.4f})")
        check("Product Spearman rho", 0.90, rho_p, tol=0.02)
        check("Product Pearson r", 0.97, r_p, tol=0.02)
    else:
        print(f"  Only {len(h_vals)} datasets with product data, need >= 5")


def verify_dataset_table():
    """Verify Table 1: dataset H(L1|L0) against profiles."""
    print("\n=== TABLE 1: DATASET PROFILES ===")

    profiles_path = RESULTS_DIR / "hierarchy_profiles.json"
    if not profiles_path.exists():
        print("  Missing hierarchy_profiles.json")
        return

    with open(profiles_path) as f:
        profiles = json.load(f)

    # Paper Table 1 values: H(L1|L0) is what matters for claims
    paper_h = {
        "yahoo": 1.23, "goemotions": 1.88, "newsgroups": 1.88,
        "trec": 2.21, "arxiv": 2.62, "dbpedia_classes": 3.17,
        "clinc": 3.90, "wos": 5.05,
    }

    for ds, expected_h in paper_h.items():
        actual_h = profiles.get(ds, {}).get("h_l1_given_l0")
        if actual_h is not None:
            check(f"H(L1|L0) {ds}", expected_h, actual_h, tol=0.02)
        else:
            print(f"  {ds}: missing from profiles")


def verify_accuracy_table():
    """Verify Table 2: k-NN accuracy at j=4 (prefix_accuracy source)."""
    print("\n=== TABLE 2: k-NN ACCURACY (j=4, 256d) ===")

    # Paper Table 2 values (V5 and MRL from prefix_accuracy['j4'])
    paper_vals = {
        "yahoo":      {"v5_l0": 0.699, "v5_l1": 0.629, "mrl_l0": 0.698, "mrl_l1": 0.635},
        "goemotions": {"v5_l0": 0.600, "v5_l1": 0.429, "mrl_l0": 0.578, "mrl_l1": 0.411},
        "newsgroups": {"v5_l0": 0.802, "v5_l1": 0.639, "mrl_l0": 0.800, "mrl_l1": 0.650},
        "trec":       {"v5_l0": 0.934, "v5_l1": 0.794, "mrl_l0": 0.932, "mrl_l1": 0.790},
        "arxiv":      {"v5_l0": 0.703, "v5_l1": 0.401, "mrl_l0": 0.692, "mrl_l1": 0.381},
        "clinc":      {"v5_l0": 0.954, "v5_l1": 0.676, "mrl_l0": 0.910, "mrl_l1": 0.704},
        "dbpedia_classes": {"v5_l0": 0.945, "v5_l1": 0.789, "mrl_l0": 0.935, "mrl_l1": 0.802},
        "wos":        {"v5_l0": 0.601, "v5_l1": 0.111, "mrl_l0": 0.599, "mrl_l1": 0.115},
    }

    for ds, expected in paper_vals.items():
        bench_file = RESULTS_DIR / f"benchmark_bge-small_{ds}.json"
        if not bench_file.exists():
            print(f"  {ds}: missing benchmark file")
            continue

        with open(bench_file) as f:
            bd = json.load(f)

        for method, sub_key in [("v5", "v5"), ("mrl", "mrl")]:
            seeds = sorted(bd[method].keys())
            j4_l0 = np.mean([bd[method][s]["prefix_accuracy"]["j4_l0"] for s in seeds])
            j4_l1 = np.mean([bd[method][s]["prefix_accuracy"]["j4_l1"] for s in seeds])

            check(f"{ds} {method} L0", expected[f"{method}_l0"], j4_l0, tol=0.002)
            check(f"{ds} {method} L1", expected[f"{method}_l1"], j4_l1, tol=0.002)


if __name__ == "__main__":
    print("=" * 60)
    print("PAPER STATISTICS VERIFICATION")
    print("Using paired tests per methodology Section 2")
    print("=" * 60)

    verify_dataset_table()
    verify_accuracy_table()
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
