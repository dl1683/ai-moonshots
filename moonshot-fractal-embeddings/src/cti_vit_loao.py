#!/usr/bin/env python -u
"""
VISION CROSS-MODAL LOAO TEST (Feb 21 2026)
==========================================
Tests whether the universal law logit(q) = alpha * kappa_nearest + C holds
for VISION models (ViT), and whether alpha_vision ≈ alpha_text ≈ 1.54.

This is the CROSS-MODAL UNIVERSALITY test: the law was established for
text models (7 architecture families, alpha=1.54, CV=4.4%). Here we test:
  - Does the same law hold for image classification representations?
  - Is alpha_vision consistent across ViT models (vision LOAO)?
  - Is alpha_vision ≈ 1.54 (cross-modal universality)?

MODELS TESTED (from MODEL_DIRECTORY.md):
  - google/vit-base-patch16-224 (86M, d=768, 12 layers, ImageNet-21k pretrained)
  - google/vit-large-patch16-224 (307M, d=1024, 24 layers, ImageNet-21k pretrained)

DATASET:
  CIFAR-10 (K=10, N=10000): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  Used with ViT (images are 224x224 resized from 32x32)

PROTOCOL:
  For each model:
    1. Load CIFAR-10 test split (1000 samples/class = 10000 total)
    2. Extract CLS token embeddings from each transformer layer
    3. For each layer: compute kappa_nearest and q (1-NN accuracy)
    4. Fit: logit(q) = alpha_layer * kappa_nearest + C
    5. Compute within-model r(kappa, logit_q) across layers
    6. Fit global: logit(q) = alpha_model * kappa_nearest + C across all layers

  Vision LOAO: alpha consistency across ViT models

PRE-REGISTERED CRITERIA:
  C1: r(kappa_nearest, logit_q) > 0.90 within each model (law holds for vision)
  C2: alpha_ViT-Base deviation from alpha_text (1.549) < 0.30 (30% tolerance)
  C3: alpha_vision consistent across models: |alpha_Base - alpha_Large| / mean < 0.20

WHY THIS MATTERS (Nobel-track):
  If alpha_vision ≈ 1.54 = alpha_text:
  - The law is UNIVERSAL across modalities (text AND vision)
  - It's not a property of language models - it's a property of neural classification
  - A single constant predicts kNN accuracy from representation geometry across ALL modalities
  - This is the "universal constant of intelligence" claim

  If alpha_vision ≠ 1.54 but is still CONSISTENT across ViT models:
  - Modality-specific alpha (text: 1.54, vision: X)
  - Still interesting: alpha is a property of the model family/training paradigm
"""

import json
import os
import time
import numpy as np
import torch
from transformers import ViTModel, ViTImageProcessor
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# CONFIG
# ================================================================
VISION_MODELS = {
    "google/vit-base-patch16-224":  {"layers": list(range(1, 13)), "d": 768},   # all 12 layers
    "google/vit-large-patch16-224": {"layers": list(range(1, 25)), "d": 1024},  # all 24 layers
}

K         = 10    # CIFAR-10
N_SAMPLE  = 1000  # per class = 10000 total
BATCH_SIZE = 32

# PRE-REGISTERED
LOAO_ALPHA_TEXT  = 1.549   # from text LOAO (7 architectures)
PRE_REG_R        = 0.90    # within-model correlation
PRE_REG_ALPHA_TOL = 0.30   # 30% deviation from text alpha
PRE_REG_LOAO_CV   = 0.20   # cross-model alpha CV


# ================================================================
# DATA LOADING
# ================================================================
def load_cifar10(n_per_class=N_SAMPLE):
    """Load CIFAR-10 test split, return numpy arrays."""
    print(f"Loading CIFAR-10 test split ({n_per_class} per class = {n_per_class*K} total)...", flush=True)
    ds = load_dataset("cifar10", split="test")

    # Collect n_per_class examples per class
    from collections import defaultdict
    import random
    random.seed(42)

    class_indices = defaultdict(list)
    for i, label in enumerate(ds["label"]):
        class_indices[label].append(i)

    selected = []
    for c in range(K):
        idx = random.sample(class_indices[c], min(n_per_class, len(class_indices[c])))
        selected.extend([(i, c) for i in idx])

    random.shuffle(selected)
    indices, labels = zip(*selected)

    images = [ds[i]["img"] for i in indices]
    y      = np.array(labels)

    print(f"  Loaded {len(images)} images, {K} classes", flush=True)
    return images, y


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================
def extract_vit_embeddings_all_layers(images, model_name, cfg):
    """Extract CLS token embeddings from ALL layers of a ViT model."""
    processor = ViTImageProcessor.from_pretrained(model_name)
    model     = ViTModel.from_pretrained(
        model_name, add_pooling_layer=False, output_hidden_states=True
    ).to(DEVICE)
    model.eval()

    n_layers = len(cfg["layers"])
    d        = cfg["d"]
    all_layer_embs = {l: [] for l in cfg["layers"]}

    print(f"  Extracting {n_layers} layers from {model_name.split('/')[-1]}...", flush=True)
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch_imgs = images[i:i + BATCH_SIZE]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
            out    = model(**inputs, output_hidden_states=True)

            for layer_idx in cfg["layers"]:
                # CLS token (index 0) from the requested hidden state
                cls_emb = out.hidden_states[layer_idx][:, 0, :].cpu().float().numpy()
                cls_emb = np.nan_to_num(cls_emb, nan=0.0)
                all_layer_embs[layer_idx].append(cls_emb)

            if (i // BATCH_SIZE) % 20 == 0:
                print(f"    Batch {i//BATCH_SIZE + 1}/{(len(images)+BATCH_SIZE-1)//BATCH_SIZE} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()

    # Stack each layer's embeddings
    return {l: np.vstack(all_layer_embs[l]) for l in cfg["layers"]}


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
    nearest_pair, farthest_pair = None, None
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            if dist < min_dist:
                min_dist = dist; nearest_pair  = (ci, cj)
            if dist > max_dist:
                max_dist = dist; farthest_pair = (ci, cj)
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
# LOAO FIT
# ================================================================
def fit_loao(layer_results, model_key):
    """
    Fit logit(q) = alpha * kappa + C across all layers for a single model.
    Returns alpha and r.
    """
    kappas = np.array([r["kappa_nearest"] for r in layer_results])
    logits = np.array([r["logit_q"] for r in layer_results])

    # Filter NaN
    valid = np.isfinite(kappas) & np.isfinite(logits)
    kappas, logits = kappas[valid], logits[valid]
    if len(kappas) < 3:
        return {"alpha": None, "r": None, "r2": None}

    # Correlation
    dk = kappas - kappas.mean()
    dl = logits - logits.mean()
    r  = float(np.corrcoef(dk, dl)[0, 1]) if np.std(dk) > 1e-6 else float("nan")

    # Linear fit
    A = np.vstack([kappas, np.ones(len(kappas))]).T
    (alpha_hat, C_hat), _, _, _ = np.linalg.lstsq(A, logits, rcond=None)
    ss_res = float(np.sum((logits - (float(alpha_hat) * kappas + float(C_hat)))**2))
    ss_tot = float(np.sum((logits - logits.mean())**2))
    r2     = float(1 - ss_res / (ss_tot + 1e-10))

    dev = abs(float(alpha_hat) - LOAO_ALPHA_TEXT) / LOAO_ALPHA_TEXT

    print(f"\n  [{model_key}] LOAO fit across {len(kappas)} layers:", flush=True)
    print(f"    alpha_vision    = {alpha_hat:.4f}", flush=True)
    print(f"    alpha_text      = {LOAO_ALPHA_TEXT:.4f}", flush=True)
    print(f"    deviation       = {dev:.1%}", flush=True)
    print(f"    r               = {r:.4f}", flush=True)
    print(f"    R2              = {r2:.4f}", flush=True)

    c1 = not np.isnan(r) and r > PRE_REG_R
    c2 = dev < PRE_REG_ALPHA_TOL
    print(f"    [{'PASS' if c1 else 'FAIL'}] r > {PRE_REG_R}", flush=True)
    print(f"    [{'PASS' if c2 else 'FAIL'}] alpha deviation < {PRE_REG_ALPHA_TOL:.0%}", flush=True)

    return {
        "alpha": float(alpha_hat), "C": float(C_hat),
        "r": float(r) if not np.isnan(r) else None,
        "r2": r2, "deviation_from_text_alpha": dev,
        "n_layers": len(kappas),
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70, flush=True)
    print("VISION CROSS-MODAL LOAO TEST", flush=True)
    print(f"Testing: ViT-Base, ViT-Large  Dataset: CIFAR-10 (K={K})", flush=True)
    print(f"Text LOAO alpha = {LOAO_ALPHA_TEXT:.4f}  (7 arch families)", flush=True)
    print("=" * 70, flush=True)

    # Load dataset once
    images, y = load_cifar10()

    all_results = {}
    model_alphas = []

    for model_name, cfg in VISION_MODELS.items():
        model_key = model_name.split("/")[-1]
        print(f"\n{'='*60}", flush=True)
        print(f"MODEL: {model_key}  (d={cfg['d']}, {len(cfg['layers'])} layers)", flush=True)
        print(f"{'='*60}", flush=True)

        # Cache path for this model's embeddings
        cache_path = f"results/vit_loao_embs_{model_key}_cifar10.npz"

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            layer_embs = {int(k): data[k] for k in data.files if k != "y"}
            y_cached   = data["y"]
            print(f"  Loaded cached embeddings for {len(layer_embs)} layers", flush=True)
        else:
            print(f"  Extracting embeddings (this may take several minutes)...", flush=True)
            layer_embs = extract_vit_embeddings_all_layers(images, model_name, cfg)
            y_cached   = y
            # Save all layers to npz
            save_dict = {"y": y}
            for l, emb in layer_embs.items():
                save_dict[str(l)] = emb
            np.savez(cache_path, **save_dict)
            print(f"  Saved to {cache_path}", flush=True)

        # Compute kappa and q for each layer
        layer_results = []
        print(f"\n  Layer-by-layer geometry:", flush=True)
        for layer_idx in cfg["layers"]:
            X = layer_embs[layer_idx]
            # Clean
            norms = np.linalg.norm(X, axis=1)
            valid = norms > 1e-3
            X_clean = X[valid]
            y_clean = y_cached[valid]

            d = X_clean.shape[1]
            centroids, sigma_W = compute_class_stats(X_clean, y_clean)
            kappa, nearest_pair, farthest_pair, min_d, max_d = \
                compute_kappa_nearest(centroids, sigma_W, d)
            margin_ratio = max_d / min_d
            q = compute_q(X_clean, y_clean)
            if q is None:
                continue

            lq = logit_q(q)
            print(f"    L{layer_idx:2d}: kappa={kappa:.4f}  q={q:.4f}  logit(q)={lq:.4f}  "
                  f"margin_ratio={margin_ratio:.2f}x", flush=True)

            layer_results.append({
                "layer": layer_idx,
                "kappa_nearest": kappa,
                "q": q,
                "logit_q": lq,
                "margin_ratio": margin_ratio,
            })

        # Fit LOAO for this model
        fit = fit_loao(layer_results, model_key)

        model_res = {
            "model": model_key,
            "d": cfg["d"],
            "n_layers": len(cfg["layers"]),
            "layer_results": layer_results,
            "loao_fit": fit,
        }
        all_results[model_key] = model_res

        if fit.get("alpha") is not None and not np.isnan(fit["alpha"]):
            model_alphas.append((model_key, fit["alpha"]))

        # Partial save
        with open("results/cti_vit_loao.json", "w") as f:
            json.dump(all_results, f, indent=2, default=lambda x: None)

    # ======================================================
    # SUMMARY
    # ======================================================
    print(f"\n\n{'='*70}", flush=True)
    print("VISION CROSS-MODAL LOAO SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Text LOAO alpha = {LOAO_ALPHA_TEXT:.4f}  CV = 0.044", flush=True)
    print(f"\n  Vision models:", flush=True)

    valid_alphas = [a for _, a in model_alphas if not np.isnan(a)]
    for model_key, alpha in model_alphas:
        fit = all_results[model_key].get("loao_fit", {})
        r   = fit.get("r") or float("nan")
        dev = fit.get("deviation_from_text_alpha", float("nan"))
        n   = fit.get("n_layers", 0)
        print(f"  {model_key}: alpha={alpha:.4f}  r={r:.4f}  dev={dev:.1%}  n_layers={n}", flush=True)

    if valid_alphas:
        alpha_mean = float(np.mean(valid_alphas))
        alpha_std  = float(np.std(valid_alphas))
        vision_cv  = float(alpha_std / abs(alpha_mean)) if abs(alpha_mean) > 1e-6 else float("inf")
        dev_text   = abs(alpha_mean - LOAO_ALPHA_TEXT) / LOAO_ALPHA_TEXT

        print(f"\n  Vision alpha mean = {alpha_mean:.4f} +/- {alpha_std:.4f}  CV={vision_cv:.3f}", flush=True)
        print(f"  Text alpha        = {LOAO_ALPHA_TEXT:.4f}  deviation = {dev_text:.1%}", flush=True)

        # Pass/fail
        all_r_pass = all(
            (all_results[k].get("loao_fit", {}).get("r") or 0.0) > PRE_REG_R
            for k, _ in model_alphas
        )
        c1 = all_r_pass
        c2 = dev_text < PRE_REG_ALPHA_TOL
        c3 = vision_cv < PRE_REG_LOAO_CV

        overall = c1 and c2 and c3
        print(f"\n  OVERALL VISION LOAO: {'PASS' if overall else 'FAIL'}", flush=True)
        print(f"    [{'PASS' if c1 else 'FAIL'}] All models r > {PRE_REG_R}", flush=True)
        print(f"    [{'PASS' if c2 else 'FAIL'}] deviation from text alpha < {PRE_REG_ALPHA_TOL:.0%}: {dev_text:.1%}", flush=True)
        print(f"    [{'PASS' if c3 else 'FAIL'}] Vision LOAO CV < {PRE_REG_LOAO_CV:.0%}: {vision_cv:.3f}", flush=True)

        all_results["summary"] = {
            "text_loao_alpha": LOAO_ALPHA_TEXT,
            "vision_alpha_mean": alpha_mean,
            "vision_alpha_std": alpha_std,
            "vision_alpha_cv": vision_cv,
            "deviation_from_text": dev_text,
            "model_alphas": dict(model_alphas),
            "overall_pass": overall,
        }

    with open("results/cti_vit_loao.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None)
    print(f"\nSaved: results/cti_vit_loao.json", flush=True)


if __name__ == "__main__":
    main()
