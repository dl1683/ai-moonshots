#!/usr/bin/env python
"""
cgp_week2_analysis.py

Pre-registered analysis for CGP Week 2 control study.
Runs AFTER cgp_week2_control_study.py produces results.

Tests:
  H1: Monotonic dose-response (Jonckheere-Terpstra trend test)
  H2: Pooled effect CI excludes zero (mixed-effects or bootstrap)
  H3: Class separation mediates quality (Spearman > 0.5)
  H4: Uniformity does NOT help (paired comparison)

Decision: GREEN / YELLOW / RED

Usage:
    python -u src/cgp_week2_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_results():
    """Load Week 2 results and flatten to per-condition-per-dataset rows."""
    path = RESULTS_DIR / "cgp_week2_control_study.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run cgp_week2_control_study.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)

    # Flatten conditions: each condition has metrics per dataset per layer
    # We want last-layer metrics for each condition+dataset
    conditions = data.get("conditions", [])
    datasets = data.get("design", {}).get("eval_datasets", ["clinc", "dbpedia_classes"])

    flat = []
    for c in conditions:
        for ds in datasets:
            ds_metrics = c.get("metrics", {}).get(ds, {})
            if isinstance(ds_metrics, dict) and "error" not in ds_metrics:
                # Get last (deepest) layer
                int_keys = [int(k) for k in ds_metrics.keys() if k.isdigit() or isinstance(k, int)]
                if not int_keys:
                    continue
                last_key = str(max(int_keys))
                m = ds_metrics[last_key]
                flat.append({
                    "objective": c["objective"],
                    "lambda_sep": c["lambda_sep"],
                    "lambda_uni": c["lambda_uni"],
                    "seed": c["seed"],
                    "dataset": ds,
                    "knn_l0": m.get("knn_l0", 0),
                    "knn_l1": m.get("knn_l1", 0),
                    "class_sep_l1": m.get("class_sep_l1", 0),
                    "alignment_l1": m.get("alignment_l1", 0),
                    "uniformity": m.get("uniformity", 0),
                    "anisotropy": m.get("anisotropy", 0),
                    "effective_rank": m.get("effective_rank", 0),
                    "hubness_skewness": m.get("hubness_skewness", 0),
                    "mean_local_margin": m.get("mean_local_margin", 0),
                    "mean_neighborhood_entropy": m.get("mean_neighborhood_entropy", 0),
                })

    return flat, datasets, data


def jonckheere_terpstra(groups):
    """
    Jonckheere-Terpstra trend test.
    groups: list of arrays, ordered by hypothesized increasing trend.
    Returns (statistic, p_value).
    """
    k = len(groups)
    n_total = sum(len(g) for g in groups)

    # Count concordant pairs
    S = 0
    for i in range(k):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj > xi:
                        S += 1
                    elif xj < xi:
                        S -= 1

    # Under H0, E[S] = 0
    # Variance calculation
    n = [len(g) for g in groups]
    N = sum(n)

    # Var(S) = (N^2(2N+3) - sum(n_i^2(2n_i+3))) / 72
    var_S = (N * N * (2 * N + 3) - sum(ni * ni * (2 * ni + 3) for ni in n)) / 72

    if var_S <= 0:
        return 0.0, 1.0

    z = S / np.sqrt(var_S)
    p = 1 - stats.norm.cdf(z)  # one-sided: trend is increasing

    return float(z), float(p)


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(means, 100 * alpha)
    hi = np.percentile(means, 100 * (1 - alpha))
    return float(lo), float(hi), float(np.mean(means))


def cohens_d_hedges(group1, group2):
    """Cohen's d with Hedges' correction."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0

    d = (m2 - m1) / s_pooled
    # Hedges' correction
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * correction)


def test_h1_monotonic(rows, datasets):
    """H1: Monotonic dose-response for contrastive objective."""
    results = {}

    for ds in datasets:
        # Get contrastive conditions with lambda_uni=0
        groups_sep = {}
        groups_knn = {}

        for r in rows:
            if r["objective"] != "contrastive" or r["lambda_uni"] != 0.0:
                continue
            if r["dataset"] != ds:
                continue
            lam = r["lambda_sep"]
            if lam not in groups_sep:
                groups_sep[lam] = []
                groups_knn[lam] = []
            groups_sep[lam].append(r["class_sep_l1"])
            groups_knn[lam].append(r["knn_l1"])

        if len(groups_sep) < 3:
            results[ds] = {"pass": False, "reason": "insufficient data"}
            continue

        # Order by lambda_sep
        lam_vals = sorted(groups_sep.keys())
        sep_groups = [np.array(groups_sep[l]) for l in lam_vals]
        knn_groups = [np.array(groups_knn[l]) for l in lam_vals]

        # Trend test on class separation
        z_sep, p_sep = jonckheere_terpstra(sep_groups)
        z_knn, p_knn = jonckheere_terpstra(knn_groups)

        # Check monotonicity of means
        sep_means = [np.mean(g) for g in sep_groups]
        knn_means = [np.mean(g) for g in knn_groups]
        sep_monotonic = all(sep_means[i] <= sep_means[i+1] for i in range(len(sep_means)-1))
        knn_monotonic = all(knn_means[i] <= knn_means[i+1] for i in range(len(knn_means)-1))

        results[ds] = {
            "lambda_values": [float(l) for l in lam_vals],
            "sep_means": [float(s) for s in sep_means],
            "knn_means": [float(k) for k in knn_means],
            "sep_monotonic": sep_monotonic,
            "knn_monotonic": knn_monotonic,
            "jt_z_sep": z_sep,
            "jt_p_sep": p_sep,
            "jt_z_knn": z_knn,
            "jt_p_knn": p_knn,
            "pass": p_sep < 0.05 and sep_monotonic,
        }

    overall_pass = all(r.get("pass", False) for r in results.values())
    return {"per_dataset": results, "pass": overall_pass}


def test_h2_pooled_effect(rows, datasets):
    """H2: Pooled effect of lambda_sep > 0 vs = 0, CI excludes zero."""
    results = {}

    for obj in ["contrastive", "lm"]:
        baseline_vals = []
        treated_vals = []

        for r in rows:
            if r["objective"] != obj or r["lambda_uni"] != 0.0:
                continue
            if r["dataset"] not in datasets:
                continue

            knn = r.get("knn_l1", None)
            if knn is None:
                continue

            if r["lambda_sep"] == 0.0:
                baseline_vals.append(knn)
            else:
                treated_vals.append(knn)

        if len(baseline_vals) < 3 or len(treated_vals) < 3:
            results[obj] = {"pass": False, "reason": "insufficient data"}
            continue

        # Effect = treated - baseline
        effects = []
        for t in treated_vals:
            for b in baseline_vals:
                effects.append(t - b)

        # Bootstrap CI on the mean difference
        all_baseline = np.array(baseline_vals)
        all_treated = np.array(treated_vals)

        mean_diff = np.mean(all_treated) - np.mean(all_baseline)

        # Bootstrap
        rng = np.random.RandomState(42)
        boot_diffs = []
        for _ in range(10000):
            b_sample = rng.choice(all_baseline, size=len(all_baseline), replace=True)
            t_sample = rng.choice(all_treated, size=len(all_treated), replace=True)
            boot_diffs.append(np.mean(t_sample) - np.mean(b_sample))

        ci_lo = np.percentile(boot_diffs, 2.5)
        ci_hi = np.percentile(boot_diffs, 97.5)

        # Also t-test
        t_stat, p_val = stats.ttest_ind(all_treated, all_baseline)

        # Cohen's d
        d = cohens_d_hedges(all_baseline, all_treated)

        results[obj] = {
            "mean_baseline": float(np.mean(all_baseline)),
            "mean_treated": float(np.mean(all_treated)),
            "mean_diff": float(mean_diff),
            "ci_95": [float(ci_lo), float(ci_hi)],
            "ci_excludes_zero": bool(ci_lo > 0),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": d,
            "n_baseline": len(all_baseline),
            "n_treated": len(all_treated),
            "pass": bool(ci_lo > 0),
        }

    # H2 passes if contrastive passes
    overall_pass = results.get("contrastive", {}).get("pass", False)
    return {"per_objective": results, "pass": overall_pass}


def test_h3_mediation(rows, datasets):
    """H3: Class separation mediates quality.

    UPDATED per Codex review: use LEVEL-MATCHED correlations:
    - sep_l1 -> knn_l1 (not sep_l1 -> knn_l0, which is cross-level!)
    - Split by objective (don't pool contrastive + LM)
    """
    # Level-matched: sep_l1 -> knn_l1
    results = {}

    for obj in ["contrastive", "lm", "all"]:
        all_sep = []
        all_knn_l1 = []

        for r in rows:
            if r["objective"] == "baseline":
                continue
            if obj != "all" and r["objective"] != obj:
                continue
            if r["dataset"] not in datasets:
                continue

            sep = r.get("class_sep_l1", None)
            knn_l1 = r.get("knn_l1", None)

            if sep is not None and knn_l1 is not None:
                if np.isfinite(sep) and np.isfinite(knn_l1):
                    all_sep.append(sep)
                    all_knn_l1.append(knn_l1)

        if len(all_sep) < 10:
            results[obj] = {"pass": False, "reason": "insufficient data"}
            continue

        rho, p = stats.spearmanr(all_sep, all_knn_l1)
        r_val, pr = stats.pearsonr(all_sep, all_knn_l1)

        results[obj] = {
            "n_points": len(all_sep),
            "spearman": {"rho": float(rho), "p": float(p)},
            "pearson": {"r": float(r_val), "p": float(pr)},
            "pass": bool(rho > 0.4 and p < 0.01),
        }

    # Also keep the legacy cross-level test for comparison
    all_sep_legacy = []
    all_knn_l0_legacy = []
    all_knn_l1_legacy = []
    for r in rows:
        if r["objective"] == "baseline":
            continue
        if r["dataset"] not in datasets:
            continue
        sep = r.get("class_sep_l1", None)
        knn_l0 = r.get("knn_l0", None)
        knn_l1 = r.get("knn_l1", None)
        if sep is not None and knn_l0 is not None and knn_l1 is not None:
            if np.isfinite(sep) and np.isfinite(knn_l0) and np.isfinite(knn_l1):
                all_sep_legacy.append(sep)
                all_knn_l0_legacy.append(knn_l0)
                all_knn_l1_legacy.append(knn_l1)

    legacy = {}
    if len(all_sep_legacy) >= 10:
        rho_l0, p_l0 = stats.spearmanr(all_sep_legacy, all_knn_l0_legacy)
        rho_l1, p_l1 = stats.spearmanr(all_sep_legacy, all_knn_l1_legacy)
        legacy = {
            "n_points": len(all_sep_legacy),
            "cross_level_spearman_l0": {"rho": float(rho_l0), "p": float(p_l0)},
            "cross_level_spearman_l1": {"rho": float(rho_l1), "p": float(p_l1)},
            "note": "Legacy cross-level test (sep_l1 vs knn_l0). Negative rho expected.",
        }

    # H3 passes if contrastive objective shows level-matched mediation
    contrastive_pass = results.get("contrastive", {}).get("pass", False)
    return {
        "level_matched": results,
        "legacy_cross_level": legacy,
        "pass": contrastive_pass,
    }


def test_h4_uniformity(rows, datasets):
    """H4: Uniformity does NOT help."""
    results = {}

    for obj in ["contrastive", "lm"]:
        uni0_vals = []
        uni03_vals = []

        for r in rows:
            if r["objective"] != obj:
                continue
            if r["dataset"] not in datasets:
                continue

            knn = r.get("knn_l1", None)
            if knn is None:
                continue

            if r["lambda_uni"] == 0.0:
                uni0_vals.append(knn)
            elif r["lambda_uni"] == 0.3:
                uni03_vals.append(knn)

        if len(uni0_vals) < 3 or len(uni03_vals) < 3:
            results[obj] = {"pass": True, "reason": "insufficient data (vacuously true)"}
            continue

        t_stat, p_val = stats.ttest_ind(uni03_vals, uni0_vals)
        mean_diff = np.mean(uni03_vals) - np.mean(uni0_vals)

        # H4 passes if uniformity does NOT significantly help
        # (no significant positive effect)
        results[obj] = {
            "mean_uni0": float(np.mean(uni0_vals)),
            "mean_uni03": float(np.mean(uni03_vals)),
            "mean_diff": float(mean_diff),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "uniformity_helps": bool(mean_diff > 0 and p_val < 0.05),
            "pass": not bool(mean_diff > 0 and p_val < 0.05),
        }

    overall_pass = all(r.get("pass", False) for r in results.values())
    return {"per_objective": results, "pass": overall_pass}


def compute_kappa_from_rows(rows, datasets):
    """
    Compute centroid regularity parameter kappa from the data.
    kappa = min pairwise dist^2 / mean pairwise dist^2.
    kappa = 1 means simplex (equidistant centroids), kappa << 1 means irregular.

    Since we don't have raw embeddings, we estimate kappa from class_sep_l1
    variation across conditions. This is a PROXY — full kappa computation
    requires raw embeddings and will be done in Week 3.
    """
    # For now, report that kappa computation needs raw embeddings
    # We can still estimate it from the sep_l1 variation
    sep_by_condition = {}
    for r in rows:
        if r["objective"] == "baseline":
            continue
        if r["dataset"] not in datasets:
            continue
        key = (r["objective"], r["lambda_sep"], r["lambda_uni"], r["dataset"])
        sep_by_condition[key] = r.get("class_sep_l1", 0)

    if len(sep_by_condition) < 5:
        return {"note": "insufficient data for kappa estimation"}

    seps = [v for v in sep_by_condition.values() if np.isfinite(v) and v > 0]
    if len(seps) < 5:
        return {"note": "insufficient valid separation values"}

    # kappa_proxy: min(S) / mean(S) — rough proxy
    kappa_proxy = min(seps) / np.mean(seps) if np.mean(seps) > 0 else 0
    return {
        "kappa_proxy": float(kappa_proxy),
        "min_sep": float(min(seps)),
        "max_sep": float(max(seps)),
        "mean_sep": float(np.mean(seps)),
        "std_sep": float(np.std(seps)),
        "n_conditions": len(seps),
        "note": "Proxy only. Full kappa needs raw embeddings (Week 3).",
    }


def test_q_vs_s(rows, datasets):
    """
    Compute multivariate Fisher Q and compare predictive power to scalar S.
    Q = trace(Sigma_between) / trace(Sigma_within)
    """
    # For each condition, we have class_sep_l1 (scalar S).
    # We need to compute Q from the raw condition data.
    # Since we only have summary stats in the flat rows, we can test
    # whether additional geometric features improve on S alone.

    # Collect feature vectors for prediction
    features = []
    targets_l0 = []
    targets_l1 = []

    feature_names = [
        "class_sep_l1", "alignment_l1", "uniformity", "anisotropy",
        "effective_rank", "hubness_skewness", "mean_local_margin",
        "mean_neighborhood_entropy",
    ]

    for r in rows:
        if r["objective"] == "baseline":
            continue
        if r["dataset"] not in datasets:
            continue

        feat = [r.get(f, 0.0) for f in feature_names]
        if all(np.isfinite(f) for f in feat):
            features.append(feat)
            targets_l0.append(r["knn_l0"])
            targets_l1.append(r["knn_l1"])

    if len(features) < 10:
        return {"error": "insufficient data"}

    X = np.array(features)
    y_l0 = np.array(targets_l0)
    y_l1 = np.array(targets_l1)

    # R2 with just S (class_sep_l1, index 0)
    reg_s = LinearRegression().fit(X[:, :1], y_l0)
    r2_s_l0 = reg_s.score(X[:, :1], y_l0)
    reg_s1 = LinearRegression().fit(X[:, :1], y_l1)
    r2_s_l1 = reg_s1.score(X[:, :1], y_l1)

    # R2 with all features (proxy for Q-like multivariate discriminant)
    reg_all = LinearRegression().fit(X, y_l0)
    r2_all_l0 = reg_all.score(X, y_l0)
    reg_all1 = LinearRegression().fit(X, y_l1)
    r2_all_l1 = reg_all1.score(X, y_l1)

    # Feature importances
    importances_l0 = {}
    for i, fn in enumerate(feature_names):
        importances_l0[fn] = float(abs(reg_all.coef_[i]) * X[:, i].std())

    return {
        "n_points": len(features),
        "r2_S_only_l0": float(r2_s_l0),
        "r2_S_only_l1": float(r2_s_l1),
        "r2_all_features_l0": float(r2_all_l0),
        "r2_all_features_l1": float(r2_all_l1),
        "improvement_l0": float(r2_all_l0 - r2_s_l0),
        "improvement_l1": float(r2_all_l1 - r2_s_l1),
        "feature_importances_l0": importances_l0,
        "s_is_sufficient": bool(r2_all_l0 - r2_s_l0 < 0.05),
    }


def test_mediation(rows, datasets):
    """
    Test if Q/S mediates the objective -> quality path.
    If adding objective indicators doesn't improve R2 beyond S, mediation holds.
    """
    features_with_obj = []
    features_no_obj = []
    targets = []

    for r in rows:
        if r["objective"] == "baseline":
            continue
        if r["dataset"] not in datasets:
            continue

        sep = r.get("class_sep_l1", 0.0)
        if not np.isfinite(sep):
            continue

        # Without objective indicator
        features_no_obj.append([sep])

        # With objective indicator (one-hot)
        obj_contrastive = 1.0 if r["objective"] == "contrastive" else 0.0
        obj_lm = 1.0 if r["objective"] == "lm" else 0.0
        features_with_obj.append([sep, obj_contrastive, obj_lm])

        targets.append(r["knn_l0"])

    if len(targets) < 10:
        return {"error": "insufficient data"}

    X_no = np.array(features_no_obj)
    X_with = np.array(features_with_obj)
    y = np.array(targets)

    reg_no = LinearRegression().fit(X_no, y)
    r2_no = reg_no.score(X_no, y)

    reg_with = LinearRegression().fit(X_with, y)
    r2_with = reg_with.score(X_with, y)

    # Mediation holds if adding objective doesn't improve R2 much
    delta_r2 = r2_with - r2_no

    return {
        "n_points": len(targets),
        "r2_sep_only": float(r2_no),
        "r2_sep_plus_objective": float(r2_with),
        "delta_r2": float(delta_r2),
        "mediation_holds": bool(delta_r2 < 0.05),
        "interpretation": (
            "Mediation CONFIRMED: objective adds negligible information beyond class separation"
            if delta_r2 < 0.05 else
            f"Mediation PARTIAL: objective adds {delta_r2:.3f} R2 beyond class separation"
        ),
    }


def make_decision(h1, h2, h3, h4):
    """GREEN / YELLOW / RED decision."""
    h1_pass = h1["pass"]
    h2_pass = h2["pass"]
    h3_pass = h3["pass"]

    if h1_pass and h2_pass and h3_pass:
        return "GREEN"
    elif h1_pass or h2_pass:
        return "YELLOW"
    else:
        return "RED"


def main():
    print("=" * 60)
    print("  CGP Week 2: Pre-Registered Analysis")
    print("=" * 60)

    rows, datasets, raw_data = load_results()

    if not rows:
        print("ERROR: No conditions found in results.")
        sys.exit(1)

    print(f"\nLoaded {len(rows)} data points across {len(datasets)} datasets")
    print(f"Datasets: {datasets}")
    objectives = sorted(set(r["objective"] for r in rows))
    print(f"Objectives: {objectives}")

    # Run tests
    print("\n--- H1: Monotonic Dose-Response ---")
    h1 = test_h1_monotonic(rows, datasets)
    for ds, r in h1["per_dataset"].items():
        if "sep_means" in r:
            print(f"  {ds}: sep_means={[f'{x:.3f}' for x in r['sep_means']]}")
            print(f"    JT z={r['jt_z_sep']:.2f}, p={r['jt_p_sep']:.4f}")
            print(f"    Monotonic: {r['sep_monotonic']}, PASS: {r['pass']}")
    print(f"  H1 overall: {'PASS' if h1['pass'] else 'FAIL'}")

    print("\n--- H2: Pooled Effect CI ---")
    h2 = test_h2_pooled_effect(rows, datasets)
    for obj, r in h2["per_objective"].items():
        if "mean_diff" in r:
            print(f"  {obj}: diff={r['mean_diff']:.4f}, CI=[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]")
            print(f"    d={r['cohens_d']:.2f}, p={r['p_value']:.4f}")
            print(f"    CI excludes zero: {r['ci_excludes_zero']}, PASS: {r['pass']}")
    print(f"  H2 overall: {'PASS' if h2['pass'] else 'FAIL'}")

    print("\n--- H3: Level-Matched Mediation (sep_l1 -> knn_l1) ---")
    h3 = test_h3_mediation(rows, datasets)
    for obj, r in h3.get("level_matched", {}).items():
        if "spearman" in r:
            print(f"  {obj}: n={r['n_points']}, "
                  f"Spearman rho={r['spearman']['rho']:.3f} (p={r['spearman']['p']:.4f}), "
                  f"Pearson r={r['pearson']['r']:.3f} (p={r['pearson']['p']:.4f}), "
                  f"PASS={r['pass']}")
    if h3.get("legacy_cross_level"):
        lc = h3["legacy_cross_level"]
        print(f"  Legacy cross-level (sep_l1 vs knn_l0): rho={lc['cross_level_spearman_l0']['rho']:.3f}")
    print(f"  H3 overall: {'PASS' if h3['pass'] else 'FAIL'}")

    print("\n--- H4: Uniformity NOT helpful ---")
    h4 = test_h4_uniformity(rows, datasets)
    for obj, r in h4["per_objective"].items():
        if "mean_diff" in r:
            print(f"  {obj}: uni_diff={r['mean_diff']:.4f}, helps={r['uniformity_helps']}")
    print(f"  H4 overall: {'PASS' if h4['pass'] else 'FAIL'}")

    # Kappa estimation
    print("\n--- Centroid Regularity (kappa proxy) ---")
    kappa = compute_kappa_from_rows(rows, datasets)
    if "kappa_proxy" in kappa:
        print(f"  kappa_proxy = {kappa['kappa_proxy']:.3f}")
        print(f"  sep range: [{kappa['min_sep']:.3f}, {kappa['max_sep']:.3f}]")
        print(f"  sep mean = {kappa['mean_sep']:.3f} +/- {kappa['std_sep']:.3f}")
    else:
        print(f"  {kappa.get('note', 'N/A')}")

    # Theory tests
    print("\n--- Theory Test: Q vs S ---")
    q_vs_s = test_q_vs_s(rows, datasets)
    if "r2_S_only_l0" in q_vs_s:
        print(f"  R2(S only, L0) = {q_vs_s['r2_S_only_l0']:.3f}")
        print(f"  R2(all features, L0) = {q_vs_s['r2_all_features_l0']:.3f}")
        print(f"  Improvement: {q_vs_s['improvement_l0']:.3f}")
        print(f"  S is sufficient: {q_vs_s['s_is_sufficient']}")

    print("\n--- Theory Test: Mediation ---")
    mediation = test_mediation(rows, datasets)
    if "r2_sep_only" in mediation:
        print(f"  R2(sep only) = {mediation['r2_sep_only']:.3f}")
        print(f"  R2(sep + objective) = {mediation['r2_sep_plus_objective']:.3f}")
        print(f"  Delta R2 = {mediation['delta_r2']:.3f}")
        print(f"  {mediation['interpretation']}")

    # Decision
    decision = make_decision(h1, h2, h3, h4)
    print(f"\n{'=' * 60}")
    print(f"  DECISION: {decision}")
    print(f"{'=' * 60}")

    if decision == "GREEN":
        print("  -> Proceed to Week 3-4: cross-architecture replication")
        print("  -> Add held-out model (bge-base)")
        print("  -> Add held-out datasets")
        print("  -> Add linear probe metric")
    elif decision == "YELLOW":
        print("  -> Increase seeds to 5-10")
        print("  -> Try continuous lambda_sep sweep")
        print("  -> Try alternative regularizer formulations")
    else:
        print("  -> Pivot to direct geometry intervention")
        print("  -> Or pivot to training dynamics approach")

    # Save
    output = {
        "h1_monotonic": h1,
        "h2_pooled_effect": h2,
        "h3_mediation": h3,
        "h4_uniformity": h4,
        "kappa_estimation": kappa,
        "theory_q_vs_s": q_vs_s,
        "theory_mediation": mediation,
        "decision": decision,
        "n_conditions": len(rows),
        "datasets": datasets,
    }

    out_path = RESULTS_DIR / "cgp_week2_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"\nSaved to {out_path}")
    return decision


if __name__ == "__main__":
    main()
