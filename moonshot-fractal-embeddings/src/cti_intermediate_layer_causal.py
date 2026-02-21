#!/usr/bin/env python -u
"""
INTERMEDIATE LAYER CAUSAL SWEEP (Feb 21 2026)
=============================================
Tests whether alpha from the kappa_nearest law is LAYER-INDEPENDENT.

HYPOTHESIS:
  The causal mechanism logit(q) = alpha * kappa_nearest has alpha ≈ 1.54
  at ALL intermediate layers of a model, not just the final layer.

  Evidence from LOAO: alpha holds across architectures at their best layer.
  Now we test: does it hold WITHIN one model at different representation quality levels?

PROTOCOL (frozen-embedding do-intervention):
  Model: pythia-160m (12 transformer blocks, layers 0-12)
  Dataset: dbpedia (K=14, margin_ratio=4.11x - ISOLATED PAIRS)
  Layers tested: 4, 6, 8, 10, 12 (final already confirmed alpha=1.601)

  For each layer:
    1. Extract frozen embeddings (mean-pooled hidden states)
    2. Compute baseline kappa_nearest and q
    3. Run dose-response sweep (push nearest centroid pair, measure kappa and q)
    4. Fit alpha_layer from dose-response
    5. Test: does alpha_layer ≈ 1.54?

EXPECTED:
  - Earlier layers: lower kappa (0.15-0.35), lower q (0.3-0.6), more dose-response room
  - alpha_layer ≈ 1.54 across ALL layers (if causal mechanism is layer-independent)
  - SAME margin_ratio across layers (class geometry of dbpedia doesn't change)

PRE-REGISTERED CRITERIA:
  For each layer (excluding ceiling, q > 0.92):
    C1: r(delta_kappa, delta_logit_q) > 0.90
    C2: |alpha_layer - 1.54| / 1.54 < 0.25 (25% tolerance; wider than cross-arch)
    C3: farthest pair control: max|delta_kappa| < 0.01 (geometry check)

SUCCESS: >= 3 layers pass all criteria, with consistent alpha across layers.

WHY THIS MATTERS:
  If alpha is the same at layers 4, 6, 8, 10, 12, it means:
  - The causal effect of kappa_nearest on q is universal across representation quality
  - It's not an artifact of the final layer's geometry
  - Alpha is a PROPERTY OF THE CLASSIFICATION TASK, not the specific representation level
  This strengthens the universal law claim significantly.
"""

import json
import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# CONFIG
# ================================================================
MODEL_NAME   = "EleutherAI/pythia-160m"
DATASET_NAME = "fancyzhx/dbpedia_14"
TEXT_COL     = "content"
LABEL_COL    = "label"
K            = 14

# Test layers: 4, 6, 8, 10, 12 (pythia-160m has 12 transformer blocks = 13 hidden states 0-12)
TEST_LAYERS  = [4, 6, 8, 10, 12]

N_SAMPLE    = 5000
BATCH_SIZE  = 64
DELTA_RANGE = np.linspace(-3.0, 3.0, 21)

# PRE-REGISTERED
LOAO_ALPHA        = 1.549
LAYER12_ALPHA     = 1.601   # measured in previous experiment (our reference)
PRE_REG_R         = 0.90
PRE_REG_ALPHA_TOL = 0.25    # 25% tolerance (slightly looser than cross-arch 20%)
PRE_REG_CONTROL   = 0.01    # max|delta_kappa| for farthest pair
CEILING_Q         = 0.92


# ================================================================
# DATA LOADING
# ================================================================
def load_dataset_once():
    """Load dbpedia and return texts, labels."""
    import random
    random.seed(42)

    try:
        ds = load_dataset(DATASET_NAME, split="test")
    except Exception:
        ds = load_dataset(DATASET_NAME, split="train")

    n = min(N_SAMPLE, len(ds))
    indices = random.sample(range(len(ds)), n)
    texts   = [ds[TEXT_COL][i] for i in indices]
    labels  = [ds[LABEL_COL][i] for i in indices]

    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"  Loaded {len(texts)} samples, {K} classes", flush=True)
    return texts, y


def extract_embeddings_layer(texts, layer_idx, tokenizer=None, model=None):
    """Extract mean-pooled embeddings from a specific hidden state layer."""
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc   = tokenizer(batch, return_tensors="pt", truncation=True,
                              max_length=128, padding=True).to(DEVICE)
            out   = model(**enc, output_hidden_states=True)
            hs    = out.hidden_states[layer_idx]
            mask  = enc["attention_mask"].unsqueeze(-1).float()
            emb   = (hs * mask).sum(1) / mask.sum(1)
            e     = emb.cpu().float().numpy()
            e     = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            all_embs.append(e)
    return np.vstack(all_embs)


def clean_embeddings(X, y):
    """Remove NaN, inf, and zero-vector rows."""
    finite_mask = np.all(np.isfinite(X), axis=1)
    X, y = X[finite_mask], y[finite_mask]
    norms = np.linalg.norm(X, axis=1)
    valid = norms > 1e-3
    return X[valid], y[valid]


# ================================================================
# GEOMETRY
# ================================================================
def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids, within_vars = {}, []
    for c in classes:
        Xc = X[y == c]
        valid = np.all(np.isfinite(Xc), axis=1)
        Xc   = Xc[valid] if valid.sum() > 0 else Xc
        centroids[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - centroids[c])**2, axis=1)))
    sigma_W = float(np.sqrt(np.mean(within_vars) / X.shape[1]))
    return centroids, sigma_W


def compute_kappa_nearest(centroids, sigma_W, d):
    classes = list(centroids.keys())
    min_dist, max_dist = np.inf, -np.inf
    nearest_pair, farthest_pair = (classes[0], classes[1]), (classes[0], classes[1])
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            if dist < min_dist: min_dist = dist; nearest_pair  = (ci, cj)
            if dist > max_dist: max_dist = dist; farthest_pair = (ci, cj)
    kappa = float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))
    return kappa, nearest_pair, farthest_pair, min_dist, max_dist


def compute_q(X, y):
    valid = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X, y  = X[valid], y[valid]
    if len(X) < 2 * K:
        return None
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    return float((acc - 1.0/K) / (1.0 - 1.0/K))


def logit_q(q):
    return float(np.log(np.clip(q, 1e-6, 1-1e-6) / (1 - np.clip(q, 1e-6, 1-1e-6))))


# ================================================================
# DO-INTERVENTION
# ================================================================
def apply_centroid_shift(X, y, centroids, cj, ck, delta):
    mu_j, mu_k = centroids[cj].copy(), centroids[ck].copy()
    if not (np.all(np.isfinite(mu_j)) and np.all(np.isfinite(mu_k))):
        return X.copy()
    diff = mu_k - mu_j
    dist = float(np.linalg.norm(diff))
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] -= (delta / 2) * direction
    X_new[y == ck] += (delta / 2) * direction
    return X_new


def run_dose_response(X, y, pair_mode, centroids=None, sigma_W=None):
    d = X.shape[1]
    if centroids is None:
        centroids, sigma_W = compute_class_stats(X, y)
    kappa0, nearest_pair, farthest_pair, min_d, max_d = \
        compute_kappa_nearest(centroids, sigma_W, d)
    margin_ratio = max_d / min_d

    target = nearest_pair if pair_mode == "nearest" else farthest_pair
    print(f"\n    [{pair_mode}] pair={target}, kappa={kappa0:.4f}, margin_ratio={margin_ratio:.2f}x", flush=True)

    points, delta_kappas = [], []
    for delta in DELTA_RANGE:
        X_new = apply_centroid_shift(X, y, centroids, target[0], target[1], delta)
        c_new, sw_new = compute_class_stats(X_new, y)
        kn, _, _, _, _ = compute_kappa_nearest(c_new, sw_new, d)
        q = compute_q(X_new, y)
        if q is None:
            continue
        dk = kn - kappa0
        delta_kappas.append(dk)
        points.append({
            "delta": float(delta), "kappa_nearest": float(kn),
            "delta_kappa": float(dk), "q": float(q), "logit_q": logit_q(q),
        })
        print(f"      delta={delta:+.2f}  kappa={kn:.4f} ({dk:+.5f})  q={q:.4f}", flush=True)

    max_dk_control = max(abs(dk) for dk in delta_kappas) if delta_kappas else 0.0
    return points, kappa0, margin_ratio, max_dk_control


def fit_dose_response(points, label):
    if len(points) < 4:
        return {}
    kappas = np.array([p["kappa_nearest"] for p in points])
    logits = np.array([p["logit_q"] for p in points])
    dk = kappas - kappas.mean()
    dl = logits - logits.mean()
    r  = float(np.corrcoef(dk, dl)[0, 1]) if np.std(dk) > 1e-6 else float("nan")
    A  = np.vstack([kappas, np.ones(len(kappas))]).T
    (alpha_hat, C), _, _, _ = np.linalg.lstsq(A, logits, rcond=None)
    ss_res = np.sum((logits - (float(alpha_hat) * kappas + float(C)))**2)
    ss_tot = np.sum((logits - logits.mean())**2)
    r2     = float(1 - ss_res / (ss_tot + 1e-10))
    dev    = abs(float(alpha_hat) - LOAO_ALPHA) / LOAO_ALPHA
    print(f"    [{label}] alpha={alpha_hat:.4f}  dev={dev:.1%}  r={r:.4f}  R2={r2:.4f}", flush=True)
    return {
        "alpha": float(alpha_hat), "C": float(C), "r": float(r) if not np.isnan(r) else None,
        "r2": r2, "deviation_from_loao": dev,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70, flush=True)
    print("INTERMEDIATE LAYER CAUSAL SWEEP", flush=True)
    print(f"Model: {MODEL_NAME}  Dataset: dbpedia (K={K})", flush=True)
    print(f"Layers: {TEST_LAYERS}", flush=True)
    print(f"LOAO_ALPHA={LOAO_ALPHA:.4f}  LAYER12_ALPHA={LAYER12_ALPHA:.4f}", flush=True)
    print("=" * 70, flush=True)

    # Load data once
    print("\nLoading dataset...", flush=True)
    texts, y = load_dataset_once()

    # Load model once (for all layers)
    print(f"\nLoading model {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_obj = AutoModel.from_pretrained(
        MODEL_NAME, output_hidden_states=True, torch_dtype=torch.float16
    ).to(DEVICE)
    model_obj.eval()
    print(f"  Model loaded. Hidden layers: {model_obj.config.num_hidden_layers}", flush=True)

    all_results = {}
    layer_alphas = []
    layer_rs     = []

    for layer_idx in TEST_LAYERS:
        print(f"\n{'='*60}", flush=True)
        print(f"LAYER {layer_idx}", flush=True)
        print(f"{'='*60}", flush=True)

        # Load from cache if available
        cache_path = f"results/do_int_layer_pythia160m_dbpedia_L{layer_idx}.npz"
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            X, y_arr = data["X"], data["y"]
            print(f"  Loaded cache: {X.shape}", flush=True)
        else:
            print(f"  Extracting layer {layer_idx} embeddings...", flush=True)
            t0 = time.time()
            X  = extract_embeddings_layer(texts, layer_idx, tokenizer, model_obj)
            print(f"  Done in {time.time()-t0:.0f}s. Shape: {X.shape}", flush=True)
            np.savez(cache_path, X=X, y=y)
            y_arr = y

        X, y_arr = clean_embeddings(X, y_arr)
        d = X.shape[1]

        # Baseline geometry
        centroids, sigma_W = compute_class_stats(X, y_arr)
        kappa0, nearest_pair, farthest_pair, min_d, max_d = \
            compute_kappa_nearest(centroids, sigma_W, d)
        q0 = compute_q(X, y_arr)
        margin_ratio = max_d / min_d

        print(f"  Baseline: kappa={kappa0:.4f}, q={q0:.4f}, margin_ratio={margin_ratio:.2f}x", flush=True)

        is_ceiling = q0 > CEILING_Q
        if is_ceiling:
            print(f"  [SKIP] Ceiling: q={q0:.4f} > {CEILING_Q}", flush=True)
            all_results[f"layer_{layer_idx}"] = {
                "layer": layer_idx, "baseline": {"kappa": kappa0, "q": q0},
                "is_ceiling": True, "margin_ratio": margin_ratio,
            }
            continue

        # Dose-response
        pts_nearest, _, _, _   = run_dose_response(X, y_arr, "nearest", centroids, sigma_W)
        pts_farthest, _, _, max_dk_far = run_dose_response(X, y_arr, "farthest", centroids, sigma_W)

        an_nearest  = fit_dose_response(pts_nearest,  f"L{layer_idx}/nearest")
        an_farthest = fit_dose_response(pts_farthest, f"L{layer_idx}/farthest")

        # Control check
        control_ok = max_dk_far < PRE_REG_CONTROL
        print(f"  Control: max|delta_kappa(farthest)| = {max_dk_far:.6f} "
              f"[{'PASS' if control_ok else 'FAIL'}]", flush=True)

        # Pass/fail
        r_n   = an_nearest.get("r") or 0.0
        dev_n = an_nearest.get("deviation_from_loao", 1.0)
        c1    = float(r_n) > PRE_REG_R
        c2    = float(dev_n) < PRE_REG_ALPHA_TOL
        c3    = control_ok
        layer_pass = c1 and c2 and c3

        print(f"\n  Layer {layer_idx}: {'PASS' if layer_pass else 'FAIL'}", flush=True)
        print(f"    C1 r={r_n:.4f} > {PRE_REG_R}: {'PASS' if c1 else 'FAIL'}", flush=True)
        print(f"    C2 dev={dev_n:.1%} < {PRE_REG_ALPHA_TOL:.0%}: {'PASS' if c2 else 'FAIL'}", flush=True)
        print(f"    C3 ctrl: {'PASS' if c3 else 'FAIL'}", flush=True)

        layer_res = {
            "layer": layer_idx,
            "baseline": {"kappa": kappa0, "q": q0},
            "margin_ratio": margin_ratio,
            "is_ceiling": False,
            "nearest": {"points": pts_nearest, "analysis": an_nearest},
            "farthest": {"points": pts_farthest, "analysis": an_farthest,
                         "max_delta_kappa": max_dk_far},
            "layer_pass": layer_pass,
        }
        all_results[f"layer_{layer_idx}"] = layer_res

        if not is_ceiling:
            alpha_val = an_nearest.get("alpha")
            r_val     = an_nearest.get("r")
            if alpha_val is not None and not np.isnan(alpha_val):
                layer_alphas.append((layer_idx, float(alpha_val)))
            if r_val is not None and not np.isnan(r_val):
                layer_rs.append((layer_idx, float(r_val)))

        # Partial save
        with open("results/cti_intermediate_layer_causal.json", "w") as f:
            json.dump(all_results, f, indent=2, default=lambda x: None)

    # Clean up model to free VRAM
    del model_obj
    torch.cuda.empty_cache()

    # ======================================================
    # SUMMARY
    # ======================================================
    print(f"\n\n{'='*70}", flush=True)
    print("INTERMEDIATE LAYER CAUSAL SWEEP SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  LOAO alpha = {LOAO_ALPHA:.4f}", flush=True)
    print(f"  Layer 12 reference alpha = {LAYER12_ALPHA:.4f}", flush=True)
    print(f"\n  Layer-by-layer results:", flush=True)

    valid_alphas = [a for _, a in layer_alphas if not np.isnan(a)]
    n_pass = sum(1 for k, v in all_results.items()
                 if not v.get("is_ceiling", True) and v.get("layer_pass", False))

    for layer_idx in TEST_LAYERS:
        key = f"layer_{layer_idx}"
        v = all_results.get(key, {})
        if v.get("is_ceiling"):
            print(f"  L{layer_idx}: CEILING (q > {CEILING_Q})", flush=True)
        else:
            an = v.get("nearest", {}).get("analysis", {})
            a  = an.get("alpha", float("nan"))
            r  = an.get("r", float("nan")) or float("nan")
            mr = v.get("margin_ratio", float("nan"))
            kappa = v.get("baseline", {}).get("kappa", float("nan"))
            q     = v.get("baseline", {}).get("q", float("nan"))
            passed = v.get("layer_pass", False)
            print(f"  L{layer_idx}: alpha={a:.4f}  r={r:.4f}  margin_ratio={mr:.2f}x  "
                  f"kappa={kappa:.4f}  q={q:.4f}  {'PASS' if passed else 'FAIL'}", flush=True)

    if valid_alphas:
        alpha_mean = float(np.mean(valid_alphas))
        alpha_std  = float(np.std(valid_alphas))
        alpha_cv   = float(alpha_std / abs(alpha_mean)) if abs(alpha_mean) > 1e-6 else float("inf")
        print(f"\n  alpha across valid layers: {alpha_mean:.4f} +/- {alpha_std:.4f}  CV={alpha_cv:.3f}", flush=True)
        print(f"  LOAO alpha: {LOAO_ALPHA:.4f}  deviation: {abs(alpha_mean-LOAO_ALPHA)/LOAO_ALPHA:.1%}", flush=True)

    print(f"\n  Layers passing all criteria: {n_pass}/{len(TEST_LAYERS)}", flush=True)
    overall = n_pass >= 3
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'} (need >= 3 layers passing)", flush=True)

    all_results["summary"] = {
        "loao_alpha": LOAO_ALPHA,
        "layer12_alpha": LAYER12_ALPHA,
        "n_layers_pass": n_pass,
        "overall_pass": overall,
        "layer_alphas": dict(layer_alphas),
        "layer_rs":     dict(layer_rs),
        "alpha_mean":   float(np.mean(valid_alphas)) if valid_alphas else None,
        "alpha_std":    float(np.std(valid_alphas))  if valid_alphas else None,
        "alpha_cv":     float(np.std(valid_alphas)/abs(np.mean(valid_alphas))) if valid_alphas else None,
    }

    with open("results/cti_intermediate_layer_causal.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None)
    print(f"\nSaved: results/cti_intermediate_layer_causal.json", flush=True)


if __name__ == "__main__":
    main()
