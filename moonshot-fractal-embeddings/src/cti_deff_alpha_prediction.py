#!/usr/bin/env python -u
"""
PRE-REGISTERED PROSPECTIVE ALPHA PREDICTION CHALLENGE
=====================================================

Nobel-track experiment: test whether alpha (CTI slope) is predictable
from d_eff_formula = tr(Sigma_W) / sigma_centroid_dir^2.

Theory (Theorem 14/15): A_renorm = alpha / sqrt(d_eff_formula) = sqrt(4/pi) = 1.128
universally across architecture families.

This script DIRECTLY MEASURES d_eff_formula from fresh embeddings and
predicts alpha WITHOUT ever fitting logit(q) ~ kappa on those architectures.

Pre-registered design (commit this file BEFORE running):
  TRAINING: pythia-160m (alpha=1.720), gpt-neo-125m (alpha=2.017)
    - Measure d_eff_formula from DBpedia embeddings
    - Compute A_renorm_i = alpha_i / sqrt(d_eff_i) for each training arch
    - A_renorm_preregistered = mean(A_renorm_training)
    - THEORY PREDICTION: A_renorm_preregistered ~ 1.128

  TEST (prospective, unseen):
    - pythia-1b   (alpha_obs=1.002  from extended LOAO)  -- LOWER alpha
    - pythia-410m (alpha_obs=0.864  from extended LOAO)  -- LOWEST alpha
    - OLMo-1B-hf  (alpha_obs=1.326  from extended LOAO)  -- MIDDLE alpha

  For each TEST arch:
    - Measure d_eff_formula (FIRST, before comparing to alpha_obs)
    - Predict alpha_pred = A_renorm_preregistered * sqrt(d_eff)
    - Compare: relative error = |alpha_pred - alpha_obs| / alpha_obs

  Pre-registered success criteria (H1, H2):
    H1: |alpha_pred - alpha_obs| / alpha_obs < 0.25 for ALL 3 test archs
    H2: Pearson r(alpha_pred, alpha_obs) > 0.85 across ALL 5 archs (train+test)

  Note: alpha_obs values come from results/cti_extended_family_loao.json
  Note: These were computed on a DIFFERENT DATASET (multi-dataset LOAO) than
        what d_eff is measured on (DBpedia). This is a TRUE prospective test.

Usage:
    python -u src/cti_deff_alpha_prediction.py [--fast]
    --fast: use smaller sample for quick validation
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.special import logit as scipy_logit
from scipy.stats import pearsonr
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PRE-REGISTERED ALPHA VALUES (from cti_extended_family_loao.json)
# DO NOT MODIFY -- these are the observed alpha values we test against
# ============================================================
ALPHA_OBS = {
    "pythia-160m":  1.7199847353492173,   # training
    "gpt-neo-125m": 2.0173342229551126,   # training
    "pythia-1b":    1.0020929266994805,   # test: LOWER alpha
    "pythia-410m":  0.8644965588847874,   # test: LOWEST alpha
    "OLMo-1B-hf":   1.3262106570325984,  # test: MIDDLE alpha
}

# ============================================================
# PRE-REGISTERED DESIGN
# ============================================================
TRAINING_MODELS = ["pythia-160m", "gpt-neo-125m"]
TEST_MODELS = ["pythia-1b", "pythia-410m", "OLMo-1B-hf"]

THEORY_A_RENORM = float(np.sqrt(4.0 / np.pi))  # sqrt(4/pi) = 1.1284

# Success criteria
H1_MAX_REL_ERROR = 0.25   # < 25% relative error for each test arch
H2_MIN_R = 0.85           # r(alpha_pred, alpha_obs) > 0.85 across all 5

# Model IDs on HuggingFace
MODEL_IDS = {
    "pythia-160m":  "EleutherAI/pythia-160m",
    "pythia-410m":  "EleutherAI/pythia-410m",
    "pythia-1b":    "EleutherAI/pythia-1b",
    "gpt-neo-125m": "EleutherAI/gpt-neo-125m",
    "OLMo-1B-hf":   "allenai/OLMo-1B-hf",
}

# Dataset config: DBpedia K=14 (moderate K, good statistics)
DATASET_NAME = "dbpedia_14"
N_SAMPLES = 2000
N_SAMPLES_FAST = 500


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


def compute_deff_formula(X, labels):
    """
    Compute d_eff_formula = tr(Sigma_W) / sigma_centroid_dir^2

    sigma_centroid_dir^2 is measured along the nearest-centroid direction
    for the nearest class pair (same as in kappa_nearest computation).

    Returns: d_eff_formula, kappa_nearest, q (normalized 1-NN accuracy)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    classes = np.unique(labels)
    K = len(classes)
    n, d = X.shape

    # Compute centroids
    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = X[mask].mean(0)

    if len(centroids) < K:
        return None, None, None

    # tr(Sigma_W): pooled within-class variance = sum of within-class deviations
    Xc = []
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            Xc.append(X[mask] - centroids[c])
    if not Xc:
        return None, None, None
    Z = np.concatenate(Xc, axis=0).astype(np.float64)
    # tr(Sigma_W) = sum of squared deviations / N (pooled)
    N = len(Z)
    trace_SW = float(np.sum(Z ** 2)) / N  # per-dimension within-class variance

    # Compute within-class covariance Sigma_W (d x d matrix)
    # For large d, use matrix-free computation: sigma_centroid_dir^2 = ||Sigma_W^{1/2} e||^2
    # = e^T Sigma_W e = (1/N) * sum_i (z_i^T e)^2 where z_i are centered class samples

    # Find nearest class pair (for kappa_nearest)
    centroid_list = np.array([centroids[c] for c in sorted(centroids.keys())])
    labels_list = sorted(centroids.keys())

    # Pairwise centroid distances
    from sklearn.metrics import pairwise_distances
    cent_dists = pairwise_distances(centroid_list, metric="euclidean")
    np.fill_diagonal(cent_dists, np.inf)

    # Global Sigma_W^{1/2} e via projection: for direction e, sigma_cdir = (Z @ e)^2 mean
    # Find nearest pair
    i_min, j_min = np.unravel_index(np.argmin(cent_dists), cent_dists.shape)
    delta_min = cent_dists[i_min, j_min]

    c_i = centroids[labels_list[i_min]]
    c_j = centroids[labels_list[j_min]]
    e_ij = (c_j - c_i) / (delta_min + 1e-10)  # unit vector to nearest centroid

    # sigma_centroid_dir^2 = e^T Sigma_W e
    # = (1/N) * sum over all within-class samples: (z_k^T e)^2
    projections = Z @ e_ij  # shape: (N_total,)
    sigma_cdir_sq = float(np.mean(projections ** 2))

    if sigma_cdir_sq < 1e-12:
        return None, None, None

    d_eff = trace_SW / sigma_cdir_sq

    # sigma_W_global = sqrt(trace_SW / d)  -- global within-class std
    sigma_W_global = float(np.sqrt(trace_SW / d))

    # kappa_nearest = delta_min / (sigma_W_global * sqrt(d)) = delta_min / sqrt(trace_SW)
    kappa_nearest = delta_min / float(np.sqrt(trace_SW * d)) if trace_SW > 1e-12 else 0.0

    # Normalized 1-NN accuracy q
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.25,
                                                   random_state=42, stratify=labels)
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X_tr, y_tr)
        acc = knn.score(X_te, y_te)
        q = (acc - 1.0/K) / (1.0 - 1.0/K)
        q = max(min(q, 0.999), 0.001)
    except Exception:
        q = None

    return float(d_eff), float(kappa_nearest), q


def find_best_layer(model, tokenizer, texts, labels, n_layers, device,
                    n_layers_sample=6):
    """Find the layer with highest kappa_nearest (best geometric signal)."""
    # Sample layers evenly
    layer_indices = list(np.linspace(0, n_layers-1, n_layers_sample, dtype=int))

    best_layer = 0
    best_kappa = -1.0
    results = {}

    for li in layer_indices:
        t0 = time.time()
        X = extract_embeddings_at_layer(model, tokenizer, texts, li, device)
        d_eff, kappa, q = compute_deff_formula(X, labels)
        elapsed = time.time() - t0
        if kappa is not None:
            results[li] = {"d_eff": d_eff, "kappa": kappa, "q": q}
            print(f"    L{li:2d}: kappa={kappa:.4f}  d_eff={d_eff:.4f}  q={q:.4f}  ({elapsed:.1f}s)", flush=True)
            if kappa > best_kappa:
                best_kappa = kappa
                best_layer = li
        else:
            print(f"    L{li:2d}: computation failed", flush=True)

    return best_layer, results


def process_architecture(arch_name, model_id, texts, labels, device):
    """Load model, find best layer, compute d_eff + kappa."""
    try:
        model, tokenizer, n_layers, d = load_model_and_tokenizer(model_id, device)
    except Exception as e:
        print(f"  [ERROR loading {arch_name}]: {e}", flush=True)
        return None

    print(f"  Finding best layer for {arch_name} (sampling {min(6, n_layers)} layers)...", flush=True)
    best_layer, layer_results = find_best_layer(model, tokenizer, texts, labels,
                                                n_layers, device)
    print(f"  -> Best layer: {best_layer}", flush=True)

    # Use best-layer results
    result = layer_results.get(best_layer, {})
    result["arch"] = arch_name
    result["model_id"] = model_id
    result["best_layer"] = best_layer
    result["n_layers"] = n_layers
    result["d"] = d
    result["alpha_obs"] = ALPHA_OBS.get(arch_name)
    result["all_layer_results"] = layer_results

    # Clean up GPU
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use fewer samples for quick test")
    args = parser.parse_args()

    n_samples = N_SAMPLES_FAST if args.fast else N_SAMPLES

    print("=" * 70)
    print("PRE-REGISTERED PROSPECTIVE ALPHA PREDICTION CHALLENGE")
    print(f"Theory: alpha = A_renorm * sqrt(d_eff), A_renorm_theory = {THEORY_A_RENORM:.4f}")
    print("=" * 70)
    print(f"\nDataset: {DATASET_NAME}  n_samples={n_samples}")
    print(f"Training archs: {TRAINING_MODELS}")
    print(f"Test archs:     {TEST_MODELS}")
    print(f"\nPre-registered alpha_obs values:")
    for arch, alpha in ALPHA_OBS.items():
        tag = "[TRAIN]" if arch in TRAINING_MODELS else "[TEST] "
        print(f"  {tag} {arch}: alpha_obs = {alpha:.4f}")
    print()

    # Load dataset once
    print("Loading dataset...", flush=True)
    texts, labels = load_dbpedia(n_samples)
    K = len(np.unique(labels))
    print(f"  Loaded {len(texts)} samples, K={K} classes", flush=True)

    # ============================================================
    # STEP 1: Process training architectures
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Training Architectures (measuring d_eff)")
    print("=" * 70)

    training_results = {}
    for arch_name in TRAINING_MODELS:
        print(f"\n--- {arch_name} ---", flush=True)
        model_id = MODEL_IDS[arch_name]
        result = process_architecture(arch_name, model_id, texts, labels, DEVICE)
        if result is not None:
            training_results[arch_name] = result
            d_eff = result["d_eff"]
            kappa = result["kappa"]
            q = result["q"]
            alpha_obs = result["alpha_obs"]
            A_renorm_i = alpha_obs / np.sqrt(d_eff) if d_eff > 0 else None
            print(f"\n  RESULTS for {arch_name}:")
            print(f"    d_eff_formula = {d_eff:.4f}")
            print(f"    kappa_nearest = {kappa:.4f}")
            print(f"    q             = {q:.4f}")
            print(f"    alpha_obs     = {alpha_obs:.4f}")
            print(f"    A_renorm      = alpha_obs/sqrt(d_eff) = {A_renorm_i:.4f}")
            print(f"    Theory A_renorm = {THEORY_A_RENORM:.4f}")
            result["A_renorm_observed"] = A_renorm_i

    if len(training_results) < 2:
        print("[ERROR] Too few training architectures succeeded!")
        return

    # ============================================================
    # STEP 2: Lock in A_renorm (pre-registered prediction)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Lock in A_renorm (before looking at test architectures)")
    print("=" * 70)

    A_renorm_values = [r["A_renorm_observed"] for r in training_results.values()
                       if r["A_renorm_observed"] is not None]
    A_renorm_measured = float(np.mean(A_renorm_values))
    A_renorm_std = float(np.std(A_renorm_values))

    print(f"\n  Training A_renorm values: {[f'{v:.4f}' for v in A_renorm_values]}")
    print(f"  A_renorm_measured = {A_renorm_measured:.4f} +/- {A_renorm_std:.4f}")
    print(f"  Theory A_renorm   = {THEORY_A_RENORM:.4f} (sqrt(4/pi))")
    print(f"  Ratio measured/theory = {A_renorm_measured/THEORY_A_RENORM:.4f}")
    print(f"\n  >>> LOCKED IN: A_renorm_preregistered = {A_renorm_measured:.4f} <<<")

    # ============================================================
    # STEP 3: Process test architectures (prospective prediction)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: Test Architectures (prospective predictions)")
    print("=" * 70)

    test_results = {}
    for arch_name in TEST_MODELS:
        print(f"\n--- {arch_name} ---", flush=True)
        model_id = MODEL_IDS[arch_name]

        # FIRST: measure d_eff (BEFORE comparing to alpha_obs!)
        result = process_architecture(arch_name, model_id, texts, labels, DEVICE)
        if result is None:
            continue

        d_eff = result["d_eff"]

        # PREDICT alpha from d_eff (prospective)
        alpha_pred = A_renorm_measured * np.sqrt(d_eff)
        alpha_obs = result["alpha_obs"]

        rel_error = abs(alpha_pred - alpha_obs) / abs(alpha_obs)
        h1_pass = rel_error < H1_MAX_REL_ERROR

        print(f"\n  PROSPECTIVE RESULTS for {arch_name}:")
        print(f"    d_eff_formula = {d_eff:.4f}")
        print(f"    alpha_pred    = A_renorm * sqrt(d_eff) = {A_renorm_measured:.4f} * sqrt({d_eff:.4f}) = {alpha_pred:.4f}")
        print(f"    alpha_obs     = {alpha_obs:.4f}")
        print(f"    rel_error     = {rel_error*100:.1f}%  {'PASS H1' if h1_pass else 'FAIL H1'} (<{H1_MAX_REL_ERROR*100:.0f}%)")

        result["alpha_pred"] = float(alpha_pred)
        result["rel_error"] = float(rel_error)
        result["h1_pass"] = h1_pass
        result["A_renorm_would_be"] = float(alpha_obs / np.sqrt(d_eff)) if d_eff > 0 else None
        test_results[arch_name] = result

    # ============================================================
    # STEP 4: Final scorecard
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL SCORECARD")
    print("=" * 70)

    all_alpha_obs = []
    all_alpha_pred = []

    print("\n  TRAINING ARCHITECTURES (used to set A_renorm):")
    print(f"  {'arch':>30} {'d_eff':>8} {'alpha_obs':>10} {'A_renorm':>10}")
    print("  " + "-" * 65)
    for arch, r in training_results.items():
        all_alpha_obs.append(r["alpha_obs"])
        pred = A_renorm_measured * np.sqrt(r["d_eff"])
        all_alpha_pred.append(pred)
        print(f"  {arch:>30} {r['d_eff']:>8.4f} {r['alpha_obs']:>10.4f} {r['A_renorm_observed']:>10.4f}")

    print("\n  TEST ARCHITECTURES (prospective predictions):")
    print(f"  {'arch':>30} {'d_eff':>8} {'alpha_obs':>10} {'alpha_pred':>11} {'rel_err%':>9} {'H1':>5}")
    print("  " + "-" * 80)
    h1_passes = 0
    for arch, r in test_results.items():
        all_alpha_obs.append(r["alpha_obs"])
        all_alpha_pred.append(r["alpha_pred"])
        h1 = "PASS" if r["h1_pass"] else "FAIL"
        if r["h1_pass"]:
            h1_passes += 1
        print(f"  {arch:>30} {r['d_eff']:>8.4f} {r['alpha_obs']:>10.4f} "
              f"{r['alpha_pred']:>11.4f} {r['rel_error']*100:>8.1f}% {h1:>5}")

    print()

    # H1: all test archs within 25% rel error
    h1_result = h1_passes == len(test_results) and len(test_results) > 0
    print(f"  H1 (rel_error < {H1_MAX_REL_ERROR*100:.0f}% for all test archs): "
          f"{h1_passes}/{len(test_results)} PASS -> {'PASS' if h1_result else 'FAIL'}")

    # H2: Pearson r(alpha_pred, alpha_obs) > 0.85 across all 5
    if len(all_alpha_obs) >= 3:
        try:
            r_pearson, p_pearson = pearsonr(all_alpha_pred, all_alpha_obs)
        except Exception:
            r_pearson, p_pearson = 0.0, 1.0
        h2_result = r_pearson > H2_MIN_R
        print(f"  H2 (r(alpha_pred, alpha_obs) > {H2_MIN_R}): "
              f"r={r_pearson:.4f} p={p_pearson:.4f} -> {'PASS' if h2_result else 'FAIL'}")
    else:
        r_pearson, p_pearson = None, None
        h2_result = False
        print(f"  H2: INSUFFICIENT DATA (n={len(all_alpha_obs)})")

    print()
    print(f"  Theory A_renorm check (training): {A_renorm_measured:.4f} vs theory {THEORY_A_RENORM:.4f}")
    print(f"    -> ratio = {A_renorm_measured/THEORY_A_RENORM:.4f} "
          f"({'CLOSE' if abs(A_renorm_measured/THEORY_A_RENORM - 1) < 0.15 else 'FAR'})")

    # ============================================================
    # Save results
    # ============================================================
    out = {
        "experiment": "cti_deff_alpha_prediction",
        "preregistered": True,
        "pre_registered_criteria": {
            "H1_max_rel_error": H1_MAX_REL_ERROR,
            "H2_min_r": H2_MIN_R,
        },
        "theory_A_renorm": THEORY_A_RENORM,
        "dataset": DATASET_NAME,
        "n_samples": n_samples,
        "training_models": TRAINING_MODELS,
        "test_models": TEST_MODELS,
        "A_renorm_preregistered": A_renorm_measured,
        "A_renorm_std": A_renorm_std,
        "A_renorm_theory": THEORY_A_RENORM,
        "A_renorm_ratio_measured_over_theory": float(A_renorm_measured / THEORY_A_RENORM)
            if THEORY_A_RENORM > 0 else None,
        "training_results": {k: {kk: float(vv) if isinstance(vv, float) else vv
                                  for kk, vv in v.items() if kk != "all_layer_results"}
                             for k, v in training_results.items()},
        "test_results": {k: {kk: float(vv) if isinstance(vv, float) else vv
                              for kk, vv in v.items() if kk != "all_layer_results"}
                         for k, v in test_results.items()},
        "scorecard": {
            "H1_n_pass": h1_passes,
            "H1_n_total": len(test_results),
            "H1_pass": h1_result,
            "H2_pearson_r": float(r_pearson) if r_pearson is not None else None,
            "H2_pearson_p": float(p_pearson) if p_pearson is not None else None,
            "H2_pass": h2_result,
            "overall_pass": h1_result and h2_result,
        },
        "all_alpha_obs": all_alpha_obs,
        "all_alpha_pred": all_alpha_pred,
    }

    out_path = RESULTS_DIR / "cti_deff_alpha_prediction.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
