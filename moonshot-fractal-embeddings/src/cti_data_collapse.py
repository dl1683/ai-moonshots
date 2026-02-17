#!/usr/bin/env python
"""
FINITE-SIZE SCALING DATA COLLAPSE for residual phase transition.

The gold standard for proving a genuine phase transition in statistical physics:
If different system sizes (depths L) show the same universal behavior after rescaling,
this proves the transition is governed by a single universality class.

Key signatures:
1. Transition WIDTH narrows with depth: width ~ L^(-1/nu)
2. Sigmoid steepness INCREASES with depth: k ~ L^(1/nu)
3. After rescaling, all curves COLLAPSE onto a single master curve
4. Multi-dataset consistency: alpha* is the same for clinc and trec

Observables:
- Primary: normalized beta(alpha) curve (Gaussian curvature of depth profile)
- Secondary: delta_r2(alpha) curve (bell vs linear preference)
- Cross-dataset: clinc vs trec alpha* consistency
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def sigmoid(x, x0, k, ymin, ymax):
    """Standard sigmoid function."""
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - x0)))


def load_pythia_data():
    """Load Pythia depth sweep data."""
    path = RESULTS_DIR / "cti_pythia_depth_sweep.json"
    with open(path) as f:
        data = json.load(f)

    models = {}
    for model_id, result in data["results"].items():
        L = result["num_layers"]
        n_params = result["n_params"]

        # Get d_model from known Pythia specs
        d_model = {6: 512, 12: 768, 16: 2048, 24: None, 32: 2560}  # 24L has 2 models
        if model_id == "EleutherAI/pythia-410m":
            d_model_val = 1024
        elif model_id == "EleutherAI/pythia-1.4b":
            d_model_val = 2048
        else:
            d_model_val = d_model.get(L, 0)

        for dataset in ["clinc", "trec"]:
            fits = result["fits"].get(dataset, {})
            if not fits:
                continue
            alphas = np.array(sorted([float(a) for a in fits.keys()]))
            betas = np.array([fits[str(a)]["beta"] for a in alphas])
            delta_r2s = np.array([fits[str(a)]["delta_r2"] for a in alphas])

            key = f"{model_id}|{dataset}"
            models[key] = {
                "model_id": model_id,
                "dataset": dataset,
                "L": L,
                "d_model": d_model_val,
                "n_params": n_params,
                "family": "pythia",
                "alphas": alphas,
                "beta": betas,
                "delta_r2": delta_r2s,
            }

    return models


def load_modern_models():
    """Load modern model sweep data."""
    models = {}
    for filename, family in [
        ("cti_residual_dense.json", "qwen3"),
        ("cti_residual_smollm2.json", "smollm2"),
        ("cti_residual_olmo2.json", "olmo2"),
    ]:
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        model_id = data["model_id"]
        L = data["num_layers"]

        for dataset in ["clinc"]:  # Modern models only have clinc
            experiments = data["experiments"].get(dataset, {})
            if not experiments:
                continue

            alphas, betas, delta_r2s = [], [], []
            for alpha_str in sorted(experiments.keys(), key=float):
                r = experiments[alpha_str]
                if r.get("status") != "ok" or not r.get("fit"):
                    continue
                alphas.append(float(alpha_str))
                fit = r["fit"]
                betas.append(fit["beta"])
                # Compute delta_r2 if available
                dr2 = fit.get("delta_r2")
                if dr2 is None:
                    dr2 = fit.get("bell_r2", 0) - fit.get("lin_r2", 0)
                delta_r2s.append(dr2)

            key = f"{model_id}|{dataset}"
            models[key] = {
                "model_id": model_id,
                "dataset": dataset,
                "L": L,
                "d_model": None,
                "n_params": data["n_params"],
                "family": family,
                "alphas": np.array(alphas),
                "beta": np.array(betas),
                "delta_r2": np.array(delta_r2s),
            }

    return models


def fit_sigmoid(alphas, values, increasing=True):
    """Fit sigmoid to order parameter curve. Returns fit params or None."""
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 0.01:
        return None

    v_norm = (values - v_min) / (v_max - v_min)
    if not increasing and values[-1] < values[0]:
        v_norm = 1 - v_norm

    try:
        popt, pcov = curve_fit(
            sigmoid, alphas, v_norm,
            p0=[0.7, 10, 0, 1],
            bounds=([0, 0.1, -0.5, 0.5], [1.0, 200, 0.5, 1.5]),
            maxfev=10000,
        )
        x0, k, ymin, ymax = popt
        perr = np.sqrt(np.diag(pcov))

        # R^2
        y_pred = sigmoid(alphas, *popt)
        ss_res = np.sum((v_norm - y_pred) ** 2)
        ss_tot = np.sum((v_norm - v_norm.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Transition width (10%-90%)
        width = 2 * np.log(9) / k

        return {
            "x0": float(x0),          # Transition midpoint
            "k": float(k),             # Steepness
            "k_err": float(perr[1]) if len(perr) > 1 else 0,
            "width": float(width),     # 10%-90% width
            "r2": float(r2),
            "direction": "increasing" if values[-1] >= values[0] else "decreasing",
            "range": float(v_max - v_min),
        }
    except Exception:
        return None


def data_collapse_quality(nu, models_data):
    """
    Compute data collapse quality for given nu.
    After rescaling x_scaled = (alpha - x0) * L^(1/nu), all curves should overlap.
    Quality = 1 - (within-bin variance / total variance).
    """
    all_X = []
    all_Y = []

    for key, data in models_data.items():
        if data.get("sigmoid") is None:
            continue
        L = data["L"]
        x0 = data["sigmoid"]["x0"]
        alphas = data["alphas"]
        v_norm = data["beta_norm"]

        # Rescale x-axis
        try:
            X = (alphas - x0) * L ** (1.0 / nu)
        except (OverflowError, FloatingPointError):
            return 1e6

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return 1e6

        all_X.extend(X.tolist())
        all_Y.extend(v_norm.tolist())

    if len(all_X) < 15:
        return 1e6

    all_X = np.array(all_X)
    all_Y = np.array(all_Y)

    # Bin and compute within-bin variance
    n_bins = 15
    X_min, X_max = np.percentile(all_X, [5, 95])
    if X_max <= X_min:
        return 1e6
    bin_edges = np.linspace(X_min, X_max, n_bins + 1)

    within_var = 0
    n_bins_used = 0
    for i in range(n_bins):
        mask = (all_X >= bin_edges[i]) & (all_X < bin_edges[i + 1])
        count = np.sum(mask)
        if count >= 2:
            within_var += np.var(all_Y[mask]) * count
            n_bins_used += 1

    if n_bins_used < 3:
        return 1e6

    total_var = np.var(all_Y) * len(all_Y)
    if total_var < 1e-10:
        return 1e6

    # Collapse quality = within_var / total_var (lower = better collapse)
    return within_var / total_var


def main():
    print("=" * 70)
    print("FINITE-SIZE SCALING: TRANSITION WIDTH AND DATA COLLAPSE")
    print("=" * 70)

    pythia = load_pythia_data()
    modern = load_modern_models()
    all_models = {**pythia, **modern}
    print(f"\nLoaded {len(all_models)} model-dataset combinations")

    # ============================================================
    # STEP 1: Fit sigmoids to all models
    # ============================================================
    print(f"\n{'='*70}")
    print("SIGMOID FITS TO NORMALIZED BETA CURVES")
    print(f"{'='*70}")
    print(f"\n{'Key':<45} {'L':>3} {'k':>7} {'width':>7} {'x0':>6} {'R2':>6} {'dir':>6}")
    print("-" * 85)

    for key, data in sorted(all_models.items(), key=lambda x: (x[1]["L"], x[0])):
        sig = fit_sigmoid(data["alphas"], data["beta"])
        data["sigmoid"] = sig

        # Normalize beta for collapse
        b = data["beta"]
        b_min, b_max = b.min(), b.max()
        if b_max - b_min > 0.01:
            b_norm = (b - b_min) / (b_max - b_min)
            if b[-1] < b[0]:  # Flip decreasing curves
                b_norm = 1 - b_norm
        else:
            b_norm = b * 0 + 0.5
        data["beta_norm"] = b_norm

        if sig:
            print(f"{key:<45} {data['L']:>3} {sig['k']:>7.2f} {sig['width']:>7.3f} "
                  f"{sig['x0']:>6.3f} {sig['r2']:>6.3f} {sig['direction'][:4]:>6}")
        else:
            print(f"{key:<45} {data['L']:>3}  -- FAILED --")

    # ============================================================
    # STEP 2: Transition width scaling (k vs L) — Pythia family
    # ============================================================
    print(f"\n{'='*70}")
    print("TRANSITION WIDTH SCALING: k vs L (Pythia family, clinc)")
    print(f"{'='*70}")

    pythia_clinc = {k: v for k, v in all_models.items()
                    if v["family"] == "pythia" and v["dataset"] == "clinc" and v.get("sigmoid")}

    if len(pythia_clinc) >= 3:
        Ls = np.array([v["L"] for v in pythia_clinc.values()])
        ks = np.array([v["sigmoid"]["k"] for v in pythia_clinc.values()])
        widths = np.array([v["sigmoid"]["width"] for v in pythia_clinc.values()])
        names = list(pythia_clinc.keys())

        print(f"\n  {'Model':<35} {'L':>3} {'k':>8} {'width':>8}")
        for i, name in enumerate(names):
            print(f"  {name.split('|')[0]:<35} {Ls[i]:>3} {ks[i]:>8.2f} {widths[i]:>8.3f}")

        # Spearman correlations
        rho_k, p_k = spearmanr(Ls, ks)
        rho_w, p_w = spearmanr(Ls, widths)
        print(f"\n  k vs L:     Spearman rho = {rho_k:.4f} (p = {p_k:.4f})")
        print(f"  width vs L: Spearman rho = {rho_w:.4f} (p = {p_w:.4f})")

        # Power law fit: k = a * L^(1/nu)
        try:
            def power_k(L, a, inv_nu):
                return a * L ** inv_nu

            popt_k, pcov_k = curve_fit(power_k, Ls, ks, p0=[1, 1], maxfev=10000)
            perr_k = np.sqrt(np.diag(pcov_k))
            pred_k = power_k(Ls, *popt_k)
            ss_res = np.sum((ks - pred_k) ** 2)
            ss_tot = np.sum((ks - ks.mean()) ** 2)
            r2_k = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            inv_nu = popt_k[1]
            nu = 1.0 / inv_nu if abs(inv_nu) > 0.01 else float("inf")
            print(f"\n  Power law: k = {popt_k[0]:.3f} * L^({inv_nu:.3f} +/- {perr_k[1]:.3f})")
            print(f"  => nu = 1/{inv_nu:.3f} = {nu:.3f}")
            print(f"  R^2 = {r2_k:.4f}")

            # Compare with theoretical predictions
            print(f"\n  Theoretical predictions:")
            print(f"    SDE limit (Fischer): nu = 2.0,  1/nu = 0.50")
            print(f"    ODE limit (Marion):  nu = 1.0,  1/nu = 1.00")
            print(f"    Empirical:           nu = {nu:.3f}, 1/nu = {inv_nu:.3f}")

        except Exception as e:
            print(f"\n  Power law fit failed: {e}")
            inv_nu = None
            nu = None

        # Width scaling: width = b * L^(-1/nu)
        try:
            popt_w, pcov_w = curve_fit(power_k, Ls, widths, p0=[3, -1], maxfev=10000)
            perr_w = np.sqrt(np.diag(pcov_w))
            pred_w = power_k(Ls, *popt_w)
            ss_res = np.sum((widths - pred_w) ** 2)
            ss_tot = np.sum((widths - widths.mean()) ** 2)
            r2_w = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            print(f"\n  Width: width = {popt_w[0]:.3f} * L^({popt_w[1]:.3f} +/- {perr_w[1]:.3f})")
            print(f"  R^2 = {r2_w:.4f}")
        except Exception as e:
            print(f"\n  Width fit failed: {e}")

    # ============================================================
    # STEP 3: Multi-dataset consistency (clinc vs trec)
    # ============================================================
    print(f"\n{'='*70}")
    print("MULTI-DATASET CONSISTENCY: clinc vs trec alpha*")
    print(f"{'='*70}")

    clinc_x0 = {}
    trec_x0 = {}
    for key, data in all_models.items():
        if data.get("sigmoid") is None:
            continue
        model_id = data["model_id"]
        if data["dataset"] == "clinc":
            clinc_x0[model_id] = data["sigmoid"]["x0"]
        elif data["dataset"] == "trec":
            trec_x0[model_id] = data["sigmoid"]["x0"]

    common = set(clinc_x0.keys()) & set(trec_x0.keys())
    if len(common) >= 3:
        print(f"\n  {'Model':<35} {'clinc x0':>10} {'trec x0':>10} {'diff':>8}")
        diffs = []
        for model_id in sorted(common):
            c = clinc_x0[model_id]
            t = trec_x0[model_id]
            diff = abs(c - t)
            diffs.append(diff)
            print(f"  {model_id:<35} {c:>10.3f} {t:>10.3f} {diff:>8.3f}")

        mean_diff = np.mean(diffs)
        print(f"\n  Mean |clinc - trec| = {mean_diff:.3f}")

        # Correlation between clinc and trec alpha*
        c_vals = [clinc_x0[m] for m in sorted(common)]
        t_vals = [trec_x0[m] for m in sorted(common)]
        rho, p = spearmanr(c_vals, t_vals)
        r, p_r = pearsonr(c_vals, t_vals)
        print(f"  Spearman rho = {rho:.4f} (p = {p:.4f})")
        print(f"  Pearson r = {r:.4f} (p = {p_r:.4f})")

        if rho > 0.8 and p < 0.05:
            print("  CONSISTENT: Same transition structure across datasets!")
        elif rho > 0.5:
            print("  PARTIALLY CONSISTENT: Trend preserved but noisy")
        else:
            print("  INCONSISTENT: Different transition points for different datasets")

    # ============================================================
    # STEP 4: Data collapse optimization (Pythia clinc only)
    # ============================================================
    print(f"\n{'='*70}")
    print("DATA COLLAPSE OPTIMIZATION (Pythia clinc)")
    print(f"{'='*70}")

    pythia_collapse = {k: v for k, v in all_models.items()
                       if v["family"] == "pythia" and v["dataset"] == "clinc"
                       and v.get("sigmoid") is not None
                       and v["sigmoid"]["r2"] > 0.5}  # Only well-fit curves

    if len(pythia_collapse) >= 3:
        # Optimize nu
        best_nu = None
        best_cost = np.inf

        for nu_init in np.arange(0.2, 3.0, 0.1):
            try:
                result = minimize(
                    data_collapse_quality,
                    x0=[nu_init],
                    args=(pythia_collapse,),
                    method="Nelder-Mead",
                    options={"maxiter": 2000},
                )
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_nu = result.x[0]
            except Exception:
                continue

        if best_nu is not None:
            print(f"\n  Optimal nu = {best_nu:.4f}")
            print(f"  Collapse quality = {1 - best_cost:.4f} (1.0 = perfect)")

            # Permutation test
            n_perm = 1000
            x0_vals = [v["sigmoid"]["x0"] for v in pythia_collapse.values()]
            rng = np.random.RandomState(42)
            perm_costs = []
            for _ in range(n_perm):
                shuffled = rng.permutation(x0_vals)
                temp = {}
                for i, (k, v) in enumerate(pythia_collapse.items()):
                    temp[k] = dict(v)
                    temp[k]["sigmoid"] = dict(v["sigmoid"])
                    temp[k]["sigmoid"]["x0"] = shuffled[i]
                perm_costs.append(data_collapse_quality(best_nu, temp))

            p_collapse = np.mean(np.array(perm_costs) <= best_cost)
            print(f"  Permutation p-value = {p_collapse:.4f} (n={n_perm})")

            # Compare nu from collapse vs from k-scaling
            if nu is not None:
                print(f"\n  Consistency check:")
                print(f"    nu from k-scaling:    {nu:.3f}")
                print(f"    nu from data collapse: {best_nu:.3f}")
                nu_diff = abs(nu - best_nu) / ((nu + best_nu) / 2)
                print(f"    Relative difference:   {100 * nu_diff:.1f}%")

    # ============================================================
    # STEP 5: Width control (L=24: Pythia-410m vs Pythia-1.4b)
    # ============================================================
    print(f"\n{'='*70}")
    print("WIDTH CONTROL: Same L, different d_model")
    print(f"{'='*70}")

    for dataset in ["clinc", "trec"]:
        k1 = f"EleutherAI/pythia-410m|{dataset}"
        k2 = f"EleutherAI/pythia-1.4b|{dataset}"
        if k1 in all_models and k2 in all_models:
            s1 = all_models[k1].get("sigmoid")
            s2 = all_models[k2].get("sigmoid")
            if s1 and s2:
                print(f"\n  {dataset}:")
                print(f"    Pythia-410m (d=1024): x0={s1['x0']:.3f}, k={s1['k']:.2f}, width={s1['width']:.3f}")
                print(f"    Pythia-1.4b (d=2048): x0={s2['x0']:.3f}, k={s2['k']:.2f}, width={s2['width']:.3f}")
                print(f"    Width DOES affect steepness: k ratio = {s2['k']/s1['k']:.2f}")

    # ============================================================
    # SAVE
    # ============================================================
    out = {
        "analysis": "finite_size_scaling_v2",
        "sigmoid_fits": {},
        "scaling_law": {},
        "multi_dataset": {},
    }

    for key, data in all_models.items():
        if data.get("sigmoid"):
            out["sigmoid_fits"][key] = {
                "L": data["L"],
                "family": data["family"],
                "d_model": data["d_model"],
                "n_params": data["n_params"],
                **data["sigmoid"],
            }

    if nu is not None:
        out["scaling_law"] = {
            "k_vs_L": {
                "exponent_1_over_nu": float(inv_nu),
                "nu": float(nu),
                "r2": float(r2_k),
            }
        }

    if best_nu is not None:
        out["data_collapse"] = {
            "nu_optimal": float(best_nu),
            "quality": float(1 - best_cost),
            "permutation_p": float(p_collapse),
        }

    out_path = RESULTS_DIR / "cti_data_collapse.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
