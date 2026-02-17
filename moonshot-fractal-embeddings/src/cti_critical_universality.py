#!/usr/bin/env python
"""
CRITICAL UNIVERSALITY: Is kappa a true phase transition order parameter?

Tests Codex's highest-upside hypothesis:
  Q = F((kappa - kappa_c) * L^(1/nu), h)
where L = system size (number of layers), nu = critical exponent,
h = task "external field".

If the SAME exponents hold across transformers AND Mamba/SSM,
this proves intelligence quality is a universal geometric phase transition.

Protocol:
  1. Residual surgery alpha sweeps on Mamba size ladder (130M/370M/790M)
  2. Pool with existing transformer data (4 models from spectral collapse)
  3. For each model: fit sigmoid(kappa) -> find kappa_c and transition width
  4. Test finite-size scaling: width ~ L^(-1/nu) with SHARED nu
  5. Data collapse: Q vs (kappa - kappa_c) * L^(1/nu) for all models

Pre-registered criteria:
  - Shared nu across architectures (bootstrap CI overlap)
  - Data collapse R^2 > 0.95
  - Mamba exponents within 30% of transformer exponents

All models from MODEL_DIRECTORY.md.
"""

import json
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset


def extract_all_layer_reps(model, tokenizer, texts, alpha, device="cuda", batch_size=32):
    """Extract all layer representations with residual scaling."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask",
                           torch.ones(enc["input_ids"].shape, device=device))
            for idx, hs in enumerate(out.hidden_states):
                hs_f = hs.float()
                m = mask.unsqueeze(-1).float()
                pooled = (hs_f * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if idx not in all_hidden:
                    all_hidden[idx] = []
                all_hidden[idx].append(pooled.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


def compute_kappa(X, labels):
    """Compute kappa = trace(S_B)/trace(S_W)."""
    try:
        if np.isnan(X).any():
            return 0.0
        unique_labels = np.unique(labels)
        grand_mean = X.mean(0)
        trace_sb, trace_sw = 0.0, 0.0
        for lbl in unique_labels:
            mask = labels == lbl
            n_k = mask.sum()
            if n_k < 2:
                continue
            X_k = X[mask]
            mu_k = X_k.mean(0)
            trace_sb += n_k * np.sum((mu_k - grand_mean) ** 2)
            trace_sw += np.sum((X_k - mu_k) ** 2)
        if trace_sw < 1e-12:
            return 100.0 if trace_sb > 0 else 0.0
        return float(min(trace_sb / trace_sw, 100.0))
    except Exception:
        return 0.0


def compute_knn(X, labels, n_train_frac=0.7):
    """kNN accuracy."""
    n = len(labels)
    n_train = int(n_train_frac * n)
    if n_train < 5 or n - n_train < 5:
        return 0.0
    try:
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X[:n_train], labels[:n_train])
        return float(knn.score(X[n_train:], labels[n_train:]))
    except Exception:
        return 0.0


def compute_mean_obs(reps, labels):
    """Compute mean kNN and kappa across layers."""
    knn_vals, kappa_vals = [], []
    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue
        knn_vals.append(compute_knn(X, labels))
        kv = compute_kappa(X, labels)
        if np.isfinite(kv):
            kappa_vals.append(kv)
    return {
        "knn": float(np.mean(knn_vals)) if knn_vals else 0,
        "kappa": float(np.mean(kappa_vals)) if kappa_vals else 0,
    }


def sigmoid(x, a, b, c, d):
    """Generalized sigmoid."""
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def fit_transition(kappas, knns):
    """Fit sigmoid kNN = sigmoid(kappa) and extract transition point + width."""
    try:
        popt, pcov = curve_fit(sigmoid, kappas, knns,
                               p0=[0.5, 10, 0.3, 0.1], maxfev=10000)
        # kappa_c = midpoint c, width = 1/b
        kappa_c = popt[2]
        width = 1.0 / max(abs(popt[1]), 0.01)
        pred = sigmoid(kappas, *popt)
        ss_res = np.sum((knns - pred) ** 2)
        ss_tot = np.sum((knns - knns.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {
            "kappa_c": float(kappa_c),
            "width": float(width),
            "steepness": float(popt[1]),
            "amplitude": float(popt[0] - popt[3]),
            "r2": float(r2),
            "params": [float(x) for x in popt],
        }
    except Exception as e:
        return {"error": str(e)}


def data_collapse_r2(all_model_data, nu):
    """Compute R^2 of data collapse with given nu.

    Rescaled variable: z = (kappa - kappa_c) * L^(1/nu)
    All models should collapse onto one curve Q = F(z).
    """
    z_all, q_all = [], []
    for md in all_model_data:
        if "error" in md["fit"]:
            continue
        L = md["n_layers"]
        kappa_c = md["fit"]["kappa_c"]
        for p in md["points"]:
            z = (p["kappa"] - kappa_c) * (L ** (1.0 / nu))
            q_all.append(p["knn"])
            z_all.append(z)

    if len(z_all) < 10:
        return 0.0

    z_arr = np.array(z_all)
    q_arr = np.array(q_all)

    # Fit sigmoid to collapsed data
    try:
        popt, _ = curve_fit(sigmoid, z_arr, q_arr,
                            p0=[0.5, 1, 0, 0.1], maxfev=10000)
        pred = sigmoid(z_arr, *popt)
        ss_res = np.sum((q_arr - pred) ** 2)
        ss_tot = np.sum((q_arr - q_arr.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except Exception:
        return 0.0


def run_sweep(model_id, dataset_name, alphas, device):
    """Run alpha sweep, return points and model info."""
    ds = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])

    model, tokenizer, n_layers, n_params = load_model(model_id, device)

    points = []
    for alpha in alphas:
        print(f"    alpha={alpha:.2f}", end="", flush=True)
        t0 = time.time()
        reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
        obs = compute_mean_obs(reps, labels)
        elapsed = time.time() - t0
        print(f"  kNN={obs['knn']:.3f}  kappa={obs['kappa']:.4f}  ({elapsed:.1f}s)")
        points.append({
            "alpha": alpha,
            "kappa": obs["kappa"],
            "knn": obs["knn"],
        })
        sys.stdout.flush()

    info = {
        "model": model_id,
        "n_layers": n_layers,
        "n_params": n_params,
        "paradigm": "ssm" if "mamba" in model_id.lower() else "transformer",
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return info, points


def main():
    print("=" * 70)
    print("CRITICAL UNIVERSALITY: Phase transition across architectures")
    print("Is kappa a universal order parameter with shared exponents?")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    # ============================================================
    # STEP 1: Load existing transformer data
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 1: LOAD EXISTING TRANSFORMER DATA")
    print(f"{'='*70}")

    all_model_data = []

    sc_path = RESULTS_DIR / "cti_spectral_collapse.json"
    if sc_path.exists():
        with open(sc_path) as f:
            sc = json.load(f)

        # Group by model
        from collections import defaultdict
        model_pts = defaultdict(list)
        for p in sc["all_points"]:
            model_pts[p["model"]].append(p)

        transformer_layers = {
            "Qwen/Qwen2-0.5B": 24,
            "HuggingFaceTB/SmolLM2-360M": 32,
            "EleutherAI/pythia-410m": 24,
            "Qwen/Qwen3-0.6B": 28,
        }
        transformer_params = {
            "Qwen/Qwen2-0.5B": 494e6,
            "HuggingFaceTB/SmolLM2-360M": 362e6,
            "EleutherAI/pythia-410m": 354e6,
            "Qwen/Qwen3-0.6B": 596e6,
        }

        for model_id, pts in model_pts.items():
            points = [{"alpha": p["alpha"], "kappa": p["kappa"], "knn": p["knn"]}
                      for p in pts]
            points.sort(key=lambda x: x["alpha"])
            kappas = np.array([p["kappa"] for p in points])
            knns = np.array([p["knn"] for p in points])
            fit = fit_transition(kappas, knns)
            L = transformer_layers.get(model_id, 24)
            md = {
                "model": model_id,
                "n_layers": L,
                "n_params": transformer_params.get(model_id, 400e6),
                "paradigm": "transformer",
                "points": points,
                "fit": fit,
            }
            all_model_data.append(md)
            short = model_id.split("/")[-1]
            if "error" not in fit:
                print(f"  {short:>25} (L={L:>2}): kappa_c={fit['kappa_c']:.4f}, "
                      f"width={fit['width']:.4f}, R2={fit['r2']:.4f}")
            else:
                print(f"  {short:>25} (L={L:>2}): FIT FAILED")

    # ============================================================
    # STEP 2: Run Mamba size ladder
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 2: MAMBA/SSM SIZE LADDER")
    print(f"{'='*70}")

    # All models from MODEL_DIRECTORY.md
    mamba_models = [
        "state-spaces/mamba-130m-hf",
        "state-spaces/mamba-370m-hf",
        "state-spaces/mamba-790m-hf",
    ]

    for model_id in mamba_models:
        short = model_id.split("/")[-1]
        print(f"\n  --- {short} ---")
        try:
            info, points = run_sweep(model_id, "clinc", alphas, device)
            kappas = np.array([p["kappa"] for p in points])
            knns = np.array([p["knn"] for p in points])
            fit = fit_transition(kappas, knns)

            md = {
                "model": model_id,
                "n_layers": info["n_layers"],
                "n_params": info["n_params"],
                "paradigm": "ssm",
                "points": points,
                "fit": fit,
            }
            all_model_data.append(md)

            if "error" not in fit:
                print(f"  kappa_c={fit['kappa_c']:.4f}, width={fit['width']:.4f}, "
                      f"R2={fit['r2']:.4f}")
            else:
                print(f"  FIT FAILED: {fit['error']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # STEP 3: Finite-size scaling analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 3: FINITE-SIZE SCALING")
    print(f"{'='*70}")

    valid_models = [md for md in all_model_data if "error" not in md["fit"]]

    if len(valid_models) < 3:
        print("  Insufficient valid models for scaling analysis")
    else:
        # Extract L and width for each model
        print(f"\n  {'Model':>30} {'Paradigm':>12} {'L':>4} {'kappa_c':>10} "
              f"{'Width':>10} {'Steepness':>10}")
        print(f"  {'-'*80}")

        for md in valid_models:
            short = md["model"].split("/")[-1]
            print(f"  {short:>30} {md['paradigm']:>12} {md['n_layers']:>4} "
                  f"{md['fit']['kappa_c']:>10.4f} {md['fit']['width']:>10.4f} "
                  f"{md['fit']['steepness']:>10.2f}")

        # Separate by paradigm
        transformers = [md for md in valid_models if md["paradigm"] == "transformer"]
        ssms = [md for md in valid_models if md["paradigm"] == "ssm"]

        # Test width ~ L^(-1/nu) for transformers
        if len(transformers) >= 3:
            L_t = np.array([md["n_layers"] for md in transformers])
            w_t = np.array([md["fit"]["width"] for md in transformers])
            # Fit: log(width) = -1/nu * log(L) + const
            try:
                from numpy.polynomial.polynomial import polyfit
                log_L = np.log(L_t)
                log_w = np.log(np.clip(w_t, 1e-6, None))
                slope, intercept = np.polyfit(log_L, log_w, 1)
                nu_transformer = -1.0 / slope if abs(slope) > 0.01 else float("inf")
                rho_t, p_t = spearmanr(log_L, log_w)
                print(f"\n  TRANSFORMER width ~ L^(-1/nu):")
                print(f"    Slope = {slope:.4f}")
                print(f"    nu (transformer) = {nu_transformer:.4f}")
                print(f"    Spearman rho = {rho_t:.4f} (p={p_t:.4f})")
            except Exception as e:
                print(f"  Transformer scaling fit failed: {e}")
                nu_transformer = None
        else:
            print(f"\n  Only {len(transformers)} transformers - need >=3 for scaling")
            nu_transformer = None

        # Test width ~ L^(-1/nu) for SSMs
        if len(ssms) >= 2:
            L_s = np.array([md["n_layers"] for md in ssms])
            w_s = np.array([md["fit"]["width"] for md in ssms])
            try:
                log_L = np.log(L_s)
                log_w = np.log(np.clip(w_s, 1e-6, None))
                if len(ssms) >= 3:
                    slope, intercept = np.polyfit(log_L, log_w, 1)
                    nu_ssm = -1.0 / slope if abs(slope) > 0.01 else float("inf")
                else:
                    slope = (log_w[1] - log_w[0]) / (log_L[1] - log_L[0]) if log_L[1] != log_L[0] else 0
                    nu_ssm = -1.0 / slope if abs(slope) > 0.01 else float("inf")
                print(f"\n  SSM width ~ L^(-1/nu):")
                print(f"    Slope = {slope:.4f}")
                print(f"    nu (SSM) = {nu_ssm:.4f}")
            except Exception as e:
                print(f"  SSM scaling fit failed: {e}")
                nu_ssm = None
        else:
            print(f"\n  Only {len(ssms)} SSMs - need >=2 for scaling")
            nu_ssm = None

    # ============================================================
    # STEP 4: Data collapse optimization
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 4: DATA COLLAPSE OPTIMIZATION")
    print(f"{'='*70}")

    # Find optimal nu for ALL models
    def neg_collapse_r2(nu):
        if nu < 0.1 or nu > 10:
            return 0
        return -data_collapse_r2(valid_models, nu)

    result = minimize_scalar(neg_collapse_r2, bounds=(0.1, 10.0), method="bounded")
    nu_optimal = result.x
    r2_optimal = -result.fun

    print(f"\n  OPTIMAL nu (all models) = {nu_optimal:.4f}")
    print(f"  Collapse R^2 = {r2_optimal:.4f}")
    print(f"  Pre-registered: R^2 > 0.95")
    if r2_optimal > 0.95:
        print(f"  UNIVERSAL COLLAPSE: YES")
    elif r2_optimal > 0.90:
        print(f"  NEAR-UNIVERSAL (R^2 > 0.90)")
    else:
        print(f"  NOT UNIVERSAL (R^2 < 0.90)")

    # Separate collapse for each paradigm
    if len(transformers) >= 2:
        def neg_r2_t(nu):
            return -data_collapse_r2(transformers, nu) if 0.1 < nu < 10 else 0
        res_t = minimize_scalar(neg_r2_t, bounds=(0.1, 10.0), method="bounded")
        r2_t = -res_t.fun
        nu_t = res_t.x
        print(f"\n  Transformer-only: nu={nu_t:.4f}, R^2={r2_t:.4f}")
    else:
        nu_t, r2_t = None, None

    if len(ssms) >= 2:
        def neg_r2_s(nu):
            return -data_collapse_r2(ssms, nu) if 0.1 < nu < 10 else 0
        res_s = minimize_scalar(neg_r2_s, bounds=(0.1, 10.0), method="bounded")
        r2_s = -res_s.fun
        nu_s = res_s.x
        print(f"  SSM-only: nu={nu_s:.4f}, R^2={r2_s:.4f}")
    else:
        nu_s, r2_s = None, None

    # ============================================================
    # STEP 5: Cross-architecture exponent comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 5: CROSS-ARCHITECTURE UNIVERSALITY")
    print(f"{'='*70}")

    if nu_t is not None and nu_s is not None:
        ratio = nu_s / nu_t if nu_t > 0 else float("inf")
        print(f"\n  nu (transformer) = {nu_t:.4f}")
        print(f"  nu (SSM)         = {nu_s:.4f}")
        print(f"  Ratio SSM/Transformer = {ratio:.4f}")
        print(f"  Pre-registered: ratio within [0.7, 1.3]")
        if 0.7 <= ratio <= 1.3:
            print(f"  SHARED EXPONENTS: YES (ratio={ratio:.2f})")
            shared = True
        else:
            print(f"  SHARED EXPONENTS: NO (ratio={ratio:.2f})")
            shared = False
    else:
        print(f"  Cannot compare: insufficient data for one paradigm")
        shared = None

    # ============================================================
    # STEP 6: Bootstrap CI on nu
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 6: BOOTSTRAP CI ON nu")
    print(f"{'='*70}")

    n_boot = 1000
    nu_boot = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        boot_idx = rng.choice(len(valid_models), size=len(valid_models), replace=True)
        boot_models = [valid_models[i] for i in boot_idx]
        def neg_r2_boot(nu, bm=boot_models):
            return -data_collapse_r2(bm, nu) if 0.1 < nu < 10 else 0
        try:
            res = minimize_scalar(neg_r2_boot, bounds=(0.1, 10.0), method="bounded")
            nu_boot.append(res.x)
        except Exception:
            pass

    if nu_boot:
        nu_boot = np.array(nu_boot)
        ci_lo = np.percentile(nu_boot, 2.5)
        ci_hi = np.percentile(nu_boot, 97.5)
        print(f"\n  nu = {nu_optimal:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (95% CI)")
        print(f"  CI width = {ci_hi - ci_lo:.4f}")
    else:
        ci_lo, ci_hi = None, None

    # ============================================================
    # STEP 7: Global kappa-kNN correlation across architectures
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 7: GLOBAL kappa-kNN CORRELATION")
    print(f"{'='*70}")

    all_kappas = []
    all_knns = []
    all_paradigms = []
    for md in valid_models:
        for p in md["points"]:
            all_kappas.append(p["kappa"])
            all_knns.append(p["knn"])
            all_paradigms.append(md["paradigm"])

    kappas_arr = np.array(all_kappas)
    knns_arr = np.array(all_knns)

    rho_all, p_all = spearmanr(kappas_arr, knns_arr)
    r_all, pr_all = pearsonr(kappas_arr, knns_arr)
    print(f"\n  ALL models (N={len(kappas_arr)}):")
    print(f"    Spearman rho = {rho_all:.4f} (p={p_all:.6e})")
    print(f"    Pearson r = {r_all:.4f} (p={pr_all:.6e})")

    # Per-paradigm
    for paradigm in ["transformer", "ssm"]:
        mask = np.array([p == paradigm for p in all_paradigms])
        if mask.sum() > 5:
            rho_p, p_p = spearmanr(kappas_arr[mask], knns_arr[mask])
            r_p, pr_p = pearsonr(kappas_arr[mask], knns_arr[mask])
            print(f"\n  {paradigm:>12} (N={mask.sum()}):")
            print(f"    Spearman rho = {rho_p:.4f} (p={p_p:.6e})")
            print(f"    Pearson r = {r_p:.4f} (p={pr_p:.6e})")

    # ============================================================
    # SAVE
    # ============================================================
    out = {
        "experiment": "critical_universality",
        "hypothesis": "Q = F((kappa - kappa_c) * L^(1/nu)) with shared nu across architectures",
        "preregistered": {
            "collapse_r2_threshold": 0.95,
            "exponent_ratio_range": [0.7, 1.3],
        },
        "models": [{
            "model": md["model"],
            "paradigm": md["paradigm"],
            "n_layers": md["n_layers"],
            "n_params": md["n_params"],
            "fit": md["fit"],
            "n_points": len(md["points"]),
        } for md in all_model_data],
        "scaling": {
            "nu_optimal": float(nu_optimal),
            "collapse_r2": float(r2_optimal),
            "nu_transformer": float(nu_t) if nu_t else None,
            "r2_transformer": float(r2_t) if r2_t else None,
            "nu_ssm": float(nu_s) if nu_s else None,
            "r2_ssm": float(r2_s) if r2_s else None,
        },
        "bootstrap_ci": {
            "nu_lo": float(ci_lo) if ci_lo else None,
            "nu_hi": float(ci_hi) if ci_hi else None,
        },
        "cross_architecture": {
            "shared_exponents": shared,
            "exponent_ratio": float(nu_s / nu_t) if (nu_s and nu_t and nu_t > 0) else None,
        },
        "global_correlation": {
            "rho": float(rho_all),
            "r": float(r_all),
            "n_points": len(kappas_arr),
        },
    }

    out_path = RESULTS_DIR / "cti_critical_universality.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
