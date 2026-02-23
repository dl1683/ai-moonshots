#!/usr/bin/env python -u
"""
EQUICORRELATION d_eff TEST
==========================

PRE-REGISTERED EXPERIMENT (commit before running).

THEORETICAL BACKGROUND (Codex Session 68):
  The CTI constant alpha = A_renorm * sqrt(d_eff_comp) where:
    A_renorm = sqrt(4/pi) = 1.128  (Theorem 15, zero free parameters)
    d_eff_comp = 1/(1-rho)         (competition equicorrelation formula)
    rho = avg cosine similarity of Sigma_W-whitened centroid differences

  Derivation: For x in class c, the K-1 competition margin variables are:
    G_j = ||x - mu_j||^2 - ||x - mu_c||^2 = 2*delta_j^T*z + ||delta_j||^2
    where z = x - mu_c ~ N(0, Sigma_W), delta_j = mu_j - mu_c.

  Cov(G_j, G_k) = 4 * delta_j^T Sigma_W delta_k
  rho_{jk|c} = (delta_j^T Sigma_W delta_k) / sqrt((delta_j^T Sigma_W delta_j)(delta_k^T Sigma_W delta_k))
             = cosine_similarity(delta_j, delta_k) in the Sigma_W metric (whitened space)

  d_eff_comp = 1/(1-rho) is the effective competition dimensionality.

QUANTITATIVE PREDICTION:
  alpha = 1.477 (measured, CV=2.3%, 12 NLP decoders)
  A_renorm = sqrt(4/pi) = 1.128 (theoretical)
  => d_eff_comp = (alpha/A_renorm)^2 = (1.477/1.128)^2 = 1.716
  => rho = 1 - 1/d_eff_comp = 1 - 1/1.716 = 0.417

PRE-REGISTERED HYPOTHESES:
  H1: |d_eff_comp - 1.716| / 1.716 < 0.25 for >= 3/5 architectures
      (d_eff_comp in [1.29, 2.15] for majority of archs)
  H2: CV(d_eff_comp across 5 archs) < 0.30
      (universality: d_eff_comp relatively constant across architectures)
  H3: rho > 0 for all 5 architectures
      (centroid differences positively correlated in whitened space)

SIGNIFICANCE:
  If H1+H2+H3 all PASS: rho universal explains alpha universal.
    The universality of alpha is a GEOMETRIC consequence of the universal
    way class centroids arrange in embedding space.
  If H1/H2 FAIL but H3 PASS: rho varies across archs, so the explanation
    for alpha universality lies elsewhere.

MODELS: pythia-160m, gpt-neo-125m, pythia-410m, pythia-1b, OLMo-1B
DATASET: DBpedia K=14, N=2000
SEED: 42 throughout

Usage:
    python -u src/cti_equicorrelation_deff.py [--fast]
    --fast: N=500 for quick validation

Pre-registration commit: TBD (commit this file BEFORE running)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PRE-REGISTERED CONSTANTS (DO NOT MODIFY)
# ============================================================
ALPHA_OBS_MEAN = 1.477          # observed alpha, 12-arch LOAO CV=2.3%
THEORY_A_RENORM = float(np.sqrt(4.0 / np.pi))  # 1.1284
D_EFF_COMP_THEORY = (ALPHA_OBS_MEAN / THEORY_A_RENORM) ** 2  # = 1.716
RHO_THEORY = 1.0 - 1.0 / D_EFF_COMP_THEORY  # = 0.417

# Pre-registered per-architecture alpha values (from cti_extended_family_loao.json)
# Used only for EXPLORATORY check (not pre-registered H1/H2/H3)
ALPHA_OBS_PER_ARCH = {
    "pythia-160m":  1.7199847353492173,
    "gpt-neo-125m": 2.0173342229551126,
    "pythia-410m":  0.8644965588847874,
    "pythia-1b":    1.0020929266994805,
    "OLMo-1B-hf":   1.3262106570325984,
}

# Pre-registered success criteria
H1_FRAC_PASS = 3.0 / 5.0    # >= 3/5 archs within 25% of d_eff_comp = 1.716
H1_MAX_REL_ERR = 0.25
H2_MAX_CV = 0.30
H3_RHO_MIN = 0.0              # rho > 0 for all archs

MODELS = ["pythia-160m", "gpt-neo-125m", "pythia-410m", "pythia-1b", "OLMo-1B-hf"]
MODEL_IDS = {
    "pythia-160m":  "EleutherAI/pythia-160m",
    "pythia-410m":  "EleutherAI/pythia-410m",
    "pythia-1b":    "EleutherAI/pythia-1b",
    "gpt-neo-125m": "EleutherAI/gpt-neo-125m",
    "OLMo-1B-hf":   "allenai/OLMo-1B-hf",
}

DATASET_NAME = "dbpedia_14"
N_SAMPLES = 2000
N_SAMPLES_FAST = 500
N_PCA_COMPONENTS = 256   # use top-256 PCs of Sigma_W to approximate whitening


def load_model_and_tokenizer(model_id, device):
    """Load a causal LM and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  {model_id}: {n_layers} layers, d={d}", flush=True)
    return model, tok, n_layers, d


def load_dbpedia(n_samples):
    """Load DBpedia texts and labels."""
    from datasets import load_dataset
    import random
    ds = load_dataset("dbpedia_14", split="train")
    random.seed(42)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:n_samples]
    texts = [ds["content"][i] for i in indices]
    labels = np.array([ds["label"][i] for i in indices])
    return texts, labels


@torch.no_grad()
def extract_embeddings_at_layer(model, tokenizer, texts, layer_idx, device,
                                batch_size=32):
    """Extract mean-pooled embeddings at a specific layer."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        outputs = model(**enc, output_hidden_states=True)
        hs = outputs.hidden_states[layer_idx + 1]  # +1: embedding layer is index 0
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
        all_embeds.append(pooled.cpu().numpy())
    return np.concatenate(all_embeds, axis=0)


def compute_kappa_nearest(X, labels):
    """Compute kappa_nearest = delta_min / sqrt(trace_SW)."""
    from sklearn.metrics import pairwise_distances
    classes = np.unique(labels)
    K = len(classes)
    n, d = X.shape
    centroids = {c: X[labels == c].mean(0) for c in classes if (labels == c).sum() >= 2}
    if len(centroids) < K:
        return None

    centroid_list = np.array([centroids[c] for c in sorted(centroids.keys())])
    Xc = np.concatenate([X[labels == c] - centroids[c]
                         for c in classes if (labels == c).sum() >= 2])
    trace_SW = float(np.sum(Xc ** 2)) / len(Xc)

    cent_dists = pairwise_distances(centroid_list, metric="euclidean")
    np.fill_diagonal(cent_dists, np.inf)
    delta_min = cent_dists.min()
    kappa = delta_min / (np.sqrt(trace_SW) + 1e-12)
    return float(kappa)


def compute_equicorrelation(X, labels, n_pca=256):
    """
    Compute the equicorrelation rho and d_eff_comp = 1/(1-rho).

    rho is the average off-diagonal entry of the correlation matrix
    of G_j = ||x - mu_j||^2 - ||x - mu_c||^2 for x in class c.

    Since Cov(G_j, G_k) = 4 * delta_j^T Sigma_W delta_k, we compute
    rho_{jk|c} = cosine_similarity(delta_j, delta_k) in Sigma_W metric,
    i.e., the cosine similarity of Sigma_W^{1/2}-whitened centroid differences.

    Uses top-n_pca PCs of within-class covariance for efficiency.

    Returns:
        rho: mean off-diagonal equicorrelation
        d_eff_comp: 1/(1-rho)
        rho_per_class: per-class equicorrelation values
        rho_matrix: mean K-1 x K-1 correlation matrix (averaged over classes)
    """
    from sklearn.decomposition import TruncatedSVD

    classes = np.unique(labels)
    K = len(classes)
    n, d = X.shape

    # Compute class centroids
    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = X[mask].mean(0).astype(np.float64)

    if len(centroids) < K:
        return None, None, None, None

    centroid_array = np.array([centroids[c] for c in sorted(centroids.keys())])  # K x d

    # Pool within-class centered data for PCA of Sigma_W
    Xc_list = []
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            Xc_list.append((X[mask] - centroids[c]).astype(np.float64))
    Z = np.concatenate(Xc_list, axis=0)  # shape (N, d)
    N_total = len(Z)

    # PCA of within-class covariance (TruncatedSVD on centered data)
    # Sigma_W = (1/N) * Z^T Z  (pooled)
    # Top-n_pca eigenvectors V and eigenvalues Lambda
    n_components = min(n_pca, d, N_total - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(Z)
    V = svd.components_.T  # shape (d, n_components)
    # eigenvalues of Sigma_W: lambda_i = (singular_value / sqrt(N))^2
    Lambda = (svd.singular_values_ ** 2) / N_total  # shape (n_components,)
    sqrt_Lambda = np.sqrt(Lambda + 1e-12)  # regularized

    # For each class c, compute the whitened centroid differences
    # delta_j = mu_j - mu_c for j != c
    # whitened_delta_j = diag(sqrt(Lambda)) * V^T * delta_j  (Sigma_W^{1/2} * delta_j)
    # Sigma_W-whitened cosine sim of delta_j and delta_k:
    #   = (whitened_delta_j . whitened_delta_k) / (||whitened_delta_j|| ||whitened_delta_k||)

    classes_sorted = sorted(centroids.keys())
    rho_per_class = []
    all_cos_sims = []  # collect all off-diagonal values for aggregate stats

    for c_idx, c in enumerate(classes_sorted):
        # delta_j = mu_j - mu_c for all j != c
        other_indices = [i for i in range(K) if i != c_idx]
        deltas = centroid_array[other_indices] - centroids[c]  # shape (K-1, d)

        # Project to PCA space and scale
        proj = deltas @ V  # shape (K-1, n_components)
        whitened = proj * sqrt_Lambda[None, :]  # shape (K-1, n_components)

        # Compute cosine similarity matrix
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)  # (K-1, 1)
        norms = np.maximum(norms, 1e-12)
        whitened_norm = whitened / norms  # (K-1, n_components)
        cos_matrix = whitened_norm @ whitened_norm.T  # (K-1, K-1)

        # Off-diagonal entries only
        n_c = K - 1
        off_diag_mask = ~np.eye(n_c, dtype=bool)
        off_diag_vals = cos_matrix[off_diag_mask]

        rho_c = float(off_diag_vals.mean())
        rho_per_class.append(rho_c)
        all_cos_sims.extend(off_diag_vals.tolist())

    rho = float(np.mean(rho_per_class))
    rho_std = float(np.std(rho_per_class))

    if rho >= 1.0:
        d_eff_comp = float("inf")
    elif rho <= -1.0:
        d_eff_comp = 0.0
    else:
        d_eff_comp = 1.0 / (1.0 - rho)

    return rho, d_eff_comp, rho_per_class, rho_std


def find_best_layer_by_kappa(model, tokenizer, texts, labels, n_layers, device,
                              n_sample=6):
    """Quick layer search: find layer with highest kappa_nearest."""
    layer_indices = list(np.linspace(0, n_layers - 1, n_sample, dtype=int))
    best_kappa = -1.0
    best_layer = 0
    for layer_idx in layer_indices:
        X = extract_embeddings_at_layer(model, tokenizer, texts, layer_idx, device)
        kappa = compute_kappa_nearest(X, labels)
        if kappa is not None and kappa > best_kappa:
            best_kappa = kappa
            best_layer = layer_idx
    return best_layer, best_kappa


def process_architecture(arch_name, model_id, texts, labels, device):
    """Run full pipeline for one architecture."""
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"ARCH: {arch_name} ({model_id})", flush=True)

    model, tok, n_layers, d = load_model_and_tokenizer(model_id, device)

    # Find best layer
    print(f"  Finding best layer (sampling {min(6, n_layers)} layers)...", flush=True)
    best_layer, best_kappa = find_best_layer_by_kappa(
        model, tok, texts, labels, n_layers, device, n_sample=min(6, n_layers)
    )
    print(f"  Best layer: {best_layer} (kappa={best_kappa:.4f})", flush=True)

    # Extract full embeddings at best layer
    print(f"  Extracting embeddings at layer {best_layer}...", flush=True)
    X = extract_embeddings_at_layer(model, tok, texts, best_layer, device)
    X = X.astype(np.float64)

    # Free GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Computing equicorrelation...", flush=True)
    rho, d_eff_comp, rho_per_class, rho_std = compute_equicorrelation(
        X, labels, n_pca=N_PCA_COMPONENTS
    )

    elapsed = time.time() - t0

    if rho is None:
        print(f"  ERROR: could not compute equicorrelation", flush=True)
        return None

    # Individual-arch alpha prediction from d_eff_comp
    alpha_pred_from_deff = THEORY_A_RENORM * np.sqrt(d_eff_comp)
    alpha_obs = ALPHA_OBS_PER_ARCH[arch_name]

    # Check H1 for this arch
    h1_arch = abs(d_eff_comp - D_EFF_COMP_THEORY) / D_EFF_COMP_THEORY < H1_MAX_REL_ERR
    # Check H3 for this arch
    h3_arch = rho > H3_RHO_MIN

    print(f"  rho = {rho:.4f} +/- {rho_std:.4f}", flush=True)
    print(f"  d_eff_comp = 1/(1-rho) = {d_eff_comp:.4f}", flush=True)
    print(f"  target d_eff_comp = {D_EFF_COMP_THEORY:.4f} (rho_target={RHO_THEORY:.4f})", flush=True)
    print(f"  rel_error vs theory = {abs(d_eff_comp - D_EFF_COMP_THEORY)/D_EFF_COMP_THEORY:.4f}", flush=True)
    print(f"  alpha_obs={alpha_obs:.4f}, alpha_pred_from_deff={alpha_pred_from_deff:.4f}", flush=True)
    print(f"  H1 (|d_eff_comp - 1.716|/1.716 < 0.25): {h1_arch}", flush=True)
    print(f"  H3 (rho > 0): {h3_arch}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    return {
        "arch": arch_name,
        "model_id": model_id,
        "best_layer": int(best_layer),
        "n_layers": int(n_layers),
        "d": int(d),
        "kappa_nearest": float(best_kappa),
        "rho": float(rho),
        "rho_std_over_classes": float(rho_std),
        "d_eff_comp": float(d_eff_comp),
        "d_eff_comp_target": float(D_EFF_COMP_THEORY),
        "rel_error_d_eff_comp": float(abs(d_eff_comp - D_EFF_COMP_THEORY) / D_EFF_COMP_THEORY),
        "rho_target": float(RHO_THEORY),
        "alpha_obs": float(alpha_obs),
        "alpha_pred_from_deff_comp": float(alpha_pred_from_deff),
        "alpha_rel_error": float(abs(alpha_pred_from_deff - alpha_obs) / alpha_obs),
        "h1_pass": bool(h1_arch),
        "h3_pass": bool(h3_arch),
        "elapsed_s": float(elapsed),
    }


def main():
    fast = "--fast" in sys.argv
    n_samples = N_SAMPLES_FAST if fast else N_SAMPLES

    print("EQUICORRELATION d_eff TEST", flush=True)
    print(f"Pre-registered constants:", flush=True)
    print(f"  alpha_obs = {ALPHA_OBS_MEAN:.4f} (12-arch LOAO)", flush=True)
    print(f"  A_renorm  = {THEORY_A_RENORM:.4f} (sqrt(4/pi))", flush=True)
    print(f"  d_eff_comp_theory = {D_EFF_COMP_THEORY:.4f}", flush=True)
    print(f"  rho_theory = {RHO_THEORY:.4f}", flush=True)
    print(f"  n_samples = {n_samples}, n_pca = {N_PCA_COMPONENTS}", flush=True)
    print(f"  device = {DEVICE}", flush=True)

    # Load dataset once
    print(f"\nLoading {DATASET_NAME}...", flush=True)
    texts, labels = load_dbpedia(n_samples)
    K = len(np.unique(labels))
    print(f"  Loaded {len(texts)} samples, K={K} classes", flush=True)

    results = {}
    for arch in MODELS:
        model_id = MODEL_IDS[arch]
        try:
            res = process_architecture(arch, model_id, texts, labels, DEVICE)
            if res is not None:
                results[arch] = res
        except Exception as e:
            print(f"ERROR processing {arch}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # ============================================================
    # SCORECARD
    # ============================================================
    print("\n" + "="*60, flush=True)
    print("SCORECARD", flush=True)
    print("="*60, flush=True)

    all_d_eff = [results[a]["d_eff_comp"] for a in MODELS if a in results]
    all_rho = [results[a]["rho"] for a in MODELS if a in results]

    # H1: >= 3/5 architectures have |d_eff_comp - 1.716| / 1.716 < 0.25
    h1_n_pass = sum(1 for a in MODELS if a in results and results[a]["h1_pass"])
    h1_pass = h1_n_pass >= (H1_FRAC_PASS * len(MODELS))
    print(f"H1 ({H1_MAX_REL_ERR*100:.0f}% of d_eff_comp=1.716): {h1_n_pass}/{len(MODELS)} pass => H1={'PASS' if h1_pass else 'FAIL'}", flush=True)

    # H2: CV(d_eff_comp) < 0.30
    if len(all_d_eff) >= 2:
        cv_deff = float(np.std(all_d_eff) / (np.mean(all_d_eff) + 1e-12))
        h2_pass = cv_deff < H2_MAX_CV
        print(f"H2 (CV(d_eff_comp) < {H2_MAX_CV:.2f}): CV={cv_deff:.4f} => H2={'PASS' if h2_pass else 'FAIL'}", flush=True)
    else:
        cv_deff = None
        h2_pass = False

    # H3: rho > 0 for ALL architectures
    h3_all_pos = all(r > H3_RHO_MIN for r in all_rho)
    print(f"H3 (rho > 0 for all archs): {h3_all_pos} => H3={'PASS' if h3_all_pos else 'FAIL'}", flush=True)

    # EXPLORATORY: correlation between d_eff_comp and alpha_obs
    if len(all_d_eff) >= 3:
        all_alpha_obs = [results[a]["alpha_obs"] for a in MODELS if a in results]
        r_deff_alpha, p_r = pearsonr(all_d_eff, all_alpha_obs)
        print(f"Exploratory r(d_eff_comp, alpha_obs) = {r_deff_alpha:.4f} (p={p_r:.4f})", flush=True)
        # Test if alpha_pred = A_renorm*sqrt(d_eff_comp) matches alpha_obs
        all_alpha_pred = [results[a]["alpha_pred_from_deff_comp"] for a in MODELS if a in results]
        r_pred, p_pred = pearsonr(all_alpha_pred, all_alpha_obs)
        print(f"Exploratory r(alpha_pred, alpha_obs) = {r_pred:.4f} (p={p_pred:.4f})", flush=True)
    else:
        r_deff_alpha = None
        r_pred = None
        p_r = None
        p_pred = None

    overall_pass = h1_pass and h2_pass and h3_all_pos

    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)
    print(f"  mean rho = {np.mean(all_rho):.4f} (target={RHO_THEORY:.4f})", flush=True)
    print(f"  mean d_eff_comp = {np.mean(all_d_eff):.4f} (target={D_EFF_COMP_THEORY:.4f})", flush=True)

    # Save results
    out = {
        "experiment": "cti_equicorrelation_deff",
        "preregistered": True,
        "theory": {
            "alpha_obs_mean": ALPHA_OBS_MEAN,
            "A_renorm": THEORY_A_RENORM,
            "d_eff_comp_predicted": D_EFF_COMP_THEORY,
            "rho_predicted": RHO_THEORY,
            "interpretation": "rho = avg Sigma_W-whitened cosine sim of centroid differences; d_eff_comp = 1/(1-rho)"
        },
        "pre_registered_criteria": {
            "H1_frac_pass": H1_FRAC_PASS,
            "H1_max_rel_error": H1_MAX_REL_ERR,
            "H2_max_CV": H2_MAX_CV,
            "H3_rho_min": H3_RHO_MIN,
        },
        "dataset": DATASET_NAME,
        "n_samples": n_samples,
        "n_pca_components": N_PCA_COMPONENTS,
        "models": MODELS,
        "results": results,
        "scorecard": {
            "H1_n_pass": int(h1_n_pass),
            "H1_n_total": len(MODELS),
            "H1_pass": bool(h1_pass),
            "H2_cv_deff": float(cv_deff) if cv_deff is not None else None,
            "H2_pass": bool(h2_pass),
            "H3_all_rho_positive": bool(h3_all_pos),
            "H3_pass": bool(h3_all_pos),
            "overall_pass": bool(overall_pass),
        },
        "aggregate": {
            "mean_rho": float(np.mean(all_rho)) if all_rho else None,
            "std_rho": float(np.std(all_rho)) if all_rho else None,
            "mean_d_eff_comp": float(np.mean(all_d_eff)) if all_d_eff else None,
            "std_d_eff_comp": float(np.std(all_d_eff)) if all_d_eff else None,
            "cv_d_eff_comp": float(cv_deff) if cv_deff is not None else None,
            "all_rho": all_rho,
            "all_d_eff_comp": all_d_eff,
            "r_deff_alpha_exploratory": float(r_deff_alpha) if r_deff_alpha is not None else None,
            "r_pred_alpha_exploratory": float(r_pred) if r_pred is not None else None,
        },
    }

    out_path = RESULTS_DIR / "cti_equicorrelation_deff.json"
    with open(out_path, "w", encoding="ascii") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
