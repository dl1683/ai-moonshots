#!/usr/bin/env python -u
"""
CTI Renormalization Theory Validation
======================================
Tests whether the observed alpha_modality variation (NLP=1.477, ViT=0.630) is
fully explained by d_eff_formula = tr(Sigma_W) / sigma_centroid_dir^2.

Theory predicts:
  logit(q) = A_renorm * kappa_renorm + C
  where kappa_renorm = delta_min / sigma_centroid_dir  [no sqrt(d) factor!]
  and   kappa_raw    = delta_min / (sigma_W_global * sqrt(d))
  so    alpha_raw    = A_renorm * sqrt(d_eff_formula)
  where sqrt(d_eff_formula) = sigma_W_global * sqrt(d) / sigma_centroid_dir

If A_renorm is truly universal, then:
  alpha_NLP / sqrt(d_eff_NLP) == alpha_ViT / sqrt(d_eff_ViT) == A_renorm

Data sources:
  - Embedding caches: dose_response_embs_*.npz, vit_loao_embs_*.npz
  - Alpha estimates: from dose-response delta_logit vs delta_kappa regression
  - Global alpha references: alpha_NLP=1.4773 (LOAO-12), alpha_ViT=0.630 (ViT LOAO)
"""

import json, os, sys
import numpy as np
from scipy.special import logit as sp_logit
from scipy.stats import pearsonr

# ================================================================
# CONFIG
# ================================================================
ALPHA_NLP_GLOBAL = 1.4773   # from LOAO-12 per-dataset fit
ALPHA_VIT_GLOBAL = 0.630    # from ViT LOAO (layer 12, CIFAR-10)

LOG_FILE = "results/cti_renorm_theory.log"
OUT_JSON = "results/cti_renorm_theory.json"

# ================================================================
# GEOMETRY HELPERS
# ================================================================

def compute_geometry(X, y):
    """
    Compute all geometry needed for renormalization theory.

    Returns dict with:
      - kappa_nearest: delta_min / (sigma_W_global * sqrt(d))
      - d_eff_formula: tr(Sigma_W) / sigma_centroid_dir^2
      - kappa_renorm:  delta_min / sigma_centroid_dir
      - sigma_W_global: global within-class std
      - sigma_centroid_dir: within-class std in nearest centroid pair direction
      - tr_Sigma_W: trace of within-class covariance
      - delta_min: minimum centroid pair distance
      - nearest_pair: (j, k) indices
    """
    X = X.astype(np.float64)
    classes = np.unique(y)
    N, D = X.shape

    # --- Centroids ---
    centroids = {}
    for c in classes:
        Xc = X[y == c]
        centroids[c] = Xc.mean(0)

    # --- tr(Sigma_W) = average total within-class variance ---
    # = (1/K) * sum_k (1/n_k) * sum_{i in k} ||x_i - mu_k||^2
    tr_list = []
    for c in classes:
        Xc = X[y == c]
        dev = Xc - centroids[c]
        tr_list.append(np.mean(np.sum(dev**2, axis=1)))  # mean sq norm
    tr_Sigma_W = float(np.mean(tr_list))  # average across classes

    # --- sigma_W_global = sqrt(tr_Sigma_W / D) ---
    sigma_W_global = float(np.sqrt(tr_Sigma_W / D))

    # --- Nearest centroid pair ---
    min_dist = np.inf
    nearest_pair = (classes[0], classes[1])
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            ci, cj = classes[i], classes[j]
            d = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            if d < min_dist:
                min_dist = d
                nearest_pair = (ci, cj)

    j, k = nearest_pair
    diff = centroids[k] - centroids[j]
    direction = diff / np.linalg.norm(diff)

    # --- sigma_centroid_dir = pooled within-class std in that direction ---
    proj_vars = []
    for c in classes:
        Xc = X[y == c]
        proj = (Xc - centroids[c]) @ direction   # shape (n_c,)
        proj_vars.append(float(np.var(proj)))
    sigma_centroid_dir = float(np.sqrt(np.mean(proj_vars)))

    # --- Derived quantities ---
    d_eff_formula = tr_Sigma_W / (sigma_centroid_dir**2) if sigma_centroid_dir > 0 else np.nan
    kappa_nearest = min_dist / (sigma_W_global * np.sqrt(D)) if sigma_W_global > 0 else np.nan
    kappa_renorm  = min_dist / sigma_centroid_dir if sigma_centroid_dir > 0 else np.nan

    return {
        "kappa_nearest":      float(kappa_nearest),
        "kappa_renorm":       float(kappa_renorm),
        "d_eff_formula":      float(d_eff_formula),
        "sigma_W_global":     float(sigma_W_global),
        "sigma_centroid_dir": float(sigma_centroid_dir),
        "tr_Sigma_W":         float(tr_Sigma_W),
        "delta_min":          float(min_dist),
        "nearest_pair":       (int(nearest_pair[0]), int(nearest_pair[1])),
        "D":                  int(D),
        "K":                  int(len(classes)),
        "N":                  int(N),
    }


def fit_alpha_local(dose_results, kappa_base, q_base):
    """
    Estimate local alpha from dose-response data.
    Fit: delta_logit(q_obs) = alpha_local * delta_kappa   [no intercept]
    Uses all seeds x non-zero scales.

    Returns (alpha_local, r, n_points) or (nan, nan, 0) if insufficient data.
    """
    logit_base = float(sp_logit(q_base))
    delta_kappas = []
    delta_logits = []

    for seed_key, scale_list in dose_results.items():
        for entry in scale_list:
            if entry["scale"] == 0.0:
                continue
            q_obs = entry.get("q_obs")
            dk = entry.get("delta_kappa")
            kb = entry.get("kappa_base")
            if q_obs is None or dk is None:
                continue
            # guard against degenerate q
            q_obs = float(np.clip(q_obs, 1e-6, 1 - 1e-6))
            logit_q_obs = float(sp_logit(q_obs))
            logit_q_base_seed = float(sp_logit(float(np.clip(kb / kappa_base * q_base, 1e-6, 1 - 1e-6))))
            # Use per-seed baseline
            q_base_seed = entry.get("kappa_base") / kappa_base * q_base  # not ideal
            # Better: recompute from stored q_base
            # Use the actual q_base stored per entry if available
            q_bs = entry.get("q_base", q_base)
            q_bs = float(np.clip(q_bs, 1e-6, 1 - 1e-6))
            logit_q_bs = float(sp_logit(q_bs))
            delta_logits.append(logit_q_obs - logit_q_bs)
            delta_kappas.append(float(dk))

    if len(delta_kappas) < 4:
        return float('nan'), float('nan'), 0

    dk = np.array(delta_kappas)
    dl = np.array(delta_logits)

    # OLS no-intercept: alpha = sum(dk * dl) / sum(dk^2)
    alpha_local = float(np.dot(dk, dl) / np.dot(dk, dk))

    # Pearson r as quality check
    if len(dk) > 2 and np.std(dk) > 0 and np.std(dl) > 0:
        r, _ = pearsonr(dk, dl)
    else:
        r = float('nan')

    return alpha_local, float(r), len(dk)


# ================================================================
# LOAD EMBEDDINGS
# ================================================================

def load_text_embs(arch_key, seed=11):
    """Load text embedding NPZ for dose-response archs."""
    fname = f"results/dose_response_embs_{arch_key}_dbpedia.npz"
    if not os.path.exists(fname):
        return None, None
    npz = np.load(fname)
    X = npz["X"].astype(np.float32)
    y = npz["y"]
    npz.close()
    # Subsample to N_SAMPLE per seed (match dose-response script)
    N_SAMPLE = 5000
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    K = len(classes)
    n_per = N_SAMPLE // K
    idx = []
    for c in classes:
        cidx = np.where(y == c)[0]
        chosen = rng.choice(cidx, min(n_per, len(cidx)), replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    return X[idx], y[idx]


def load_vision_embs(arch_key, layer_key, seed=11):
    """Load vision embedding NPZ for dose-response archs."""
    if arch_key == "resnet50":
        fname = "results/dose_response_embs_resnet50_cifar10.npz"
    elif arch_key == "vit-base":
        fname = "results/vit_loao_embs_vit-base-patch16-224_cifar10.npz"
    elif arch_key == "vit-large":
        fname = "results/vit_loao_embs_vit-large-patch16-224_cifar10.npz"
    else:
        return None, None

    if not os.path.exists(fname):
        return None, None

    npz = np.load(fname)
    y_all = npz["y"]

    if arch_key == "resnet50":
        X_all = npz["X"].astype(np.float32)
    else:
        X_all = npz[layer_key].astype(np.float32)
    npz.close()

    # Subsample N_SAMPLE per seed
    N_SAMPLE = 5000
    rng = np.random.default_rng(seed)
    classes = np.unique(y_all)
    K = len(classes)
    n_per = N_SAMPLE // K
    idx = []
    for c in classes:
        cidx = np.where(y_all == c)[0]
        chosen = rng.choice(cidx, min(n_per, len(cidx)), replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    return X_all[idx], y_all[idx]


# ================================================================
# MAIN
# ================================================================

def main():
    log = open(LOG_FILE, "w", buffering=1)
    def pr(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    pr("=" * 70)
    pr("CTI Renormalization Theory Validation")
    pr("=" * 70)
    pr(f"Theory: alpha_modality = A_renorm * sqrt(d_eff_formula)")
    pr(f"Test: alpha / sqrt(d_eff) is constant across NLP and ViT")
    pr(f"Expected: A_renorm ~ sqrt(4/pi) ~ 1.128")
    pr("")

    # Load dose-response results
    with open("results/cti_dose_response.json") as f:
        dr = json.load(f)

    # Architecture configs (matching dose-response script)
    arch_configs = {
        "gemma-2-2b":  {"modality": "text", "layer_key": None},
        "phi-2":       {"modality": "text", "layer_key": None},
        "mamba-130m":  {"modality": "text", "layer_key": None},
        "vit-base":    {"modality": "vision", "layer_key": "8"},
        "vit-large":   {"modality": "vision", "layer_key": "12"},
        "resnet50":    {"modality": "vision", "layer_key": None},
    }

    results = {}
    A_renorm_per_arch = {}

    seeds = [11, 23, 47]

    for arch_key, cfg in arch_configs.items():
        pr(f"\n--- {arch_key} ({cfg['modality']}) ---")

        geom_per_seed = []
        for seed in seeds:
            # Load embeddings
            if cfg["modality"] == "text":
                X, y = load_text_embs(arch_key, seed)
            else:
                X, y = load_vision_embs(arch_key, cfg["layer_key"], seed)

            if X is None:
                pr(f"  [SKIP] No embedding cache found")
                break

            g = compute_geometry(X, y)
            geom_per_seed.append(g)
            pr(f"  seed={seed}: D={g['D']}, K={g['K']}, N={g['N']}")
            pr(f"    kappa_nearest={g['kappa_nearest']:.4f}, kappa_renorm={g['kappa_renorm']:.4f}")
            pr(f"    d_eff_formula={g['d_eff_formula']:.4f}, sigma_W={g['sigma_W_global']:.4f}")
            pr(f"    sigma_centroid_dir={g['sigma_centroid_dir']:.4f}, delta_min={g['delta_min']:.4f}")
            pr(f"    sqrt(d_eff)={np.sqrt(g['d_eff_formula']):.4f}")

        if not geom_per_seed:
            continue

        # Average geometry across seeds
        avg_geom = {k: float(np.mean([g[k] for g in geom_per_seed]))
                    for k in geom_per_seed[0] if isinstance(geom_per_seed[0][k], (int, float))}
        pr(f"  MEAN: d_eff_formula={avg_geom['d_eff_formula']:.4f}, "
           f"sqrt(d_eff)={np.sqrt(avg_geom['d_eff_formula']):.4f}, "
           f"kappa_nearest={avg_geom['kappa_nearest']:.4f}")

        # Fit alpha_local from dose-response data
        arch_dr = dr.get(arch_key, {})
        q_base_mean = np.mean([arch_dr.get(f"q_base_seed{s}", np.nan) for s in seeds])
        kappa_base_mean = np.mean([arch_dr.get(f"kappa_base_seed{s}", np.nan) for s in seeds])
        alpha_local, r_fit, n_pts = fit_alpha_local(
            arch_dr.get("seed_results", {}), kappa_base_mean, q_base_mean
        )
        pr(f"  Dose-response fit: alpha_local={alpha_local:.4f}, r={r_fit:.3f}, n={n_pts}")

        # Use global reference alpha (more reliable for NLP in ceiling)
        if cfg["modality"] == "text":
            alpha_ref = ALPHA_NLP_GLOBAL
            alpha_source = "LOAO-12 global"
        else:
            alpha_ref = ALPHA_VIT_GLOBAL
            alpha_source = "ViT-LOAO global"

        # Compute A_renorm from both sources
        d_eff = avg_geom["d_eff_formula"]
        sqrt_d_eff = np.sqrt(d_eff)

        A_from_global = alpha_ref / sqrt_d_eff
        A_from_local  = alpha_local / sqrt_d_eff if not np.isnan(alpha_local) else np.nan

        pr(f"  A_renorm (global alpha / sqrt(d_eff)): {A_from_global:.4f} [from {alpha_source} alpha={alpha_ref:.4f}]")
        pr(f"  A_renorm (local alpha / sqrt(d_eff)):  {A_from_local:.4f} [from dose-response fit]")
        pr(f"  Theory prediction: sqrt(4/pi) = {np.sqrt(4/np.pi):.4f}")

        results[arch_key] = {
            "modality": cfg["modality"],
            "layer_key": cfg["layer_key"],
            "geometry": avg_geom,
            "alpha_ref": alpha_ref,
            "alpha_ref_source": alpha_source,
            "alpha_local": alpha_local,
            "alpha_local_r": r_fit,
            "alpha_local_n": n_pts,
            "A_renorm_global": A_from_global,
            "A_renorm_local":  A_from_local,
            "sqrt_d_eff": float(sqrt_d_eff),
        }
        A_renorm_per_arch[arch_key] = A_from_global

    # ================================================================
    # Summary analysis
    # ================================================================
    pr("\n" + "=" * 70)
    pr("SUMMARY: A_renorm = alpha / sqrt(d_eff_formula)")
    pr("=" * 70)
    pr(f"{'Arch':<16} {'Modality':<10} {'alpha':<8} {'d_eff':<8} {'sqrt(d_eff)':<12} {'A_renorm':<10}")
    pr("-" * 70)

    nlp_As = []
    vit_As = []
    all_As = []

    for arch_key, res in results.items():
        if not res:
            continue
        alpha = res["alpha_ref"]
        d_eff = res["geometry"]["d_eff_formula"]
        sqrt_d = res["sqrt_d_eff"]
        A_r = res["A_renorm_global"]
        pr(f"  {arch_key:<14} {res['modality']:<10} {alpha:<8.4f} {d_eff:<8.4f} {sqrt_d:<12.4f} {A_r:<10.4f}")
        if not np.isnan(A_r):
            all_As.append(A_r)
            if res["modality"] == "text":
                nlp_As.append(A_r)
            else:
                vit_As.append(A_r)

    pr("")
    pr(f"NLP A_renorm: mean={np.mean(nlp_As):.4f} +/- {np.std(nlp_As):.4f} (n={len(nlp_As)})")
    pr(f"ViT A_renorm: mean={np.mean(vit_As):.4f} +/- {np.std(vit_As):.4f} (n={len(vit_As)})")
    pr(f"All A_renorm: mean={np.mean(all_As):.4f} +/- {np.std(all_As):.4f} (n={len(all_As)})")
    pr(f"Theory:       sqrt(4/pi) = {np.sqrt(4/np.pi):.4f}")
    pr("")
    pr(f"NLP vs ViT ratio (A_NLP/A_ViT): {np.mean(nlp_As)/np.mean(vit_As):.4f}")
    pr(f"Predicted ratio (alpha_NLP/alpha_ViT): {ALPHA_NLP_GLOBAL / ALPHA_VIT_GLOBAL:.4f}")
    pr(f"Ratio explained by d_eff: sqrt(d_eff_NLP/d_eff_ViT) = ?")

    # Compute d_eff ratio
    nlp_d_effs = [results[k]["geometry"]["d_eff_formula"]
                  for k in ["gemma-2-2b", "phi-2", "mamba-130m"] if k in results]
    vit_d_effs  = [results[k]["geometry"]["d_eff_formula"]
                   for k in ["vit-base", "vit-large", "resnet50"] if k in results]

    mean_nlp_d = np.mean(nlp_d_effs) if nlp_d_effs else np.nan
    mean_vit_d = np.mean(vit_d_effs) if vit_d_effs else np.nan

    pr(f"  mean d_eff NLP = {mean_nlp_d:.4f}")
    pr(f"  mean d_eff ViT = {mean_vit_d:.4f}")
    pr(f"  d_eff ratio NLP/ViT = {mean_nlp_d / mean_vit_d:.4f}")
    pr(f"  sqrt(d_eff ratio)   = {np.sqrt(mean_nlp_d / mean_vit_d):.4f}")
    pr(f"  alpha ratio         = {ALPHA_NLP_GLOBAL / ALPHA_VIT_GLOBAL:.4f}")

    pr("")
    cv_all = np.std(all_As) / np.mean(all_As) if np.mean(all_As) > 0 else np.nan
    pr(f"UNIVERSALITY TEST: CV(A_renorm) across all 6 archs = {cv_all:.4f}")
    pr(f"  PASS (CV < 0.25)? {'YES' if cv_all < 0.25 else 'NO'}")
    pr(f"  STRONG (CV < 0.05)? {'YES' if cv_all < 0.05 else 'NO'}")

    # ================================================================
    # Also test kappa_renorm as a unified predictor
    # ================================================================
    pr("\n" + "=" * 70)
    pr("BONUS: kappa_renorm = delta_min / sigma_centroid_dir")
    pr("Predicted: logit(q) = A_renorm * kappa_renorm + C with UNIVERSAL A_renorm")
    pr("=" * 70)
    pr(f"{'Arch':<16} {'kappa_raw':<12} {'kappa_renorm':<14} {'ratio':<10}")
    pr("-" * 55)
    for arch_key, res in results.items():
        if not res:
            continue
        g = res["geometry"]
        ratio = g["kappa_renorm"] / g["kappa_nearest"] if g["kappa_nearest"] > 0 else np.nan
        pr(f"  {arch_key:<14} {g['kappa_nearest']:<12.4f} {g['kappa_renorm']:<14.4f} {ratio:<10.4f}")
    pr("(ratio = sqrt(d_eff_formula))")

    # Save results
    out = {
        "experiment": "renormalization_theory_validation",
        "theory": "alpha = A_renorm * sqrt(d_eff_formula)",
        "alpha_NLP_global": ALPHA_NLP_GLOBAL,
        "alpha_ViT_global": ALPHA_VIT_GLOBAL,
        "sqrt_4_pi": float(np.sqrt(4/np.pi)),
        "per_arch": results,
        "summary": {
            "A_renorm_NLP_mean": float(np.mean(nlp_As)) if nlp_As else None,
            "A_renorm_NLP_std":  float(np.std(nlp_As)) if nlp_As else None,
            "A_renorm_ViT_mean": float(np.mean(vit_As)) if vit_As else None,
            "A_renorm_ViT_std":  float(np.std(vit_As)) if vit_As else None,
            "A_renorm_all_mean": float(np.mean(all_As)) if all_As else None,
            "A_renorm_all_cv":   float(cv_all) if not np.isnan(cv_all) else None,
            "d_eff_NLP_mean":    float(mean_nlp_d) if not np.isnan(mean_nlp_d) else None,
            "d_eff_ViT_mean":    float(mean_vit_d) if not np.isnan(mean_vit_d) else None,
            "alpha_ratio":       float(ALPHA_NLP_GLOBAL / ALPHA_VIT_GLOBAL),
            "d_eff_ratio":       float(mean_nlp_d / mean_vit_d) if not np.isnan(mean_nlp_d) else None,
            "universality_pass": bool(cv_all < 0.25) if not np.isnan(cv_all) else False,
        }
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    pr(f"\nResults saved to {OUT_JSON}")
    log.close()


if __name__ == "__main__":
    main()
