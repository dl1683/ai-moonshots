#!/usr/bin/env python -u
"""
CTI Pre-Registered Dose-Response Causal Experiment
====================================================
Pre-registration: results/dose_response_preregistration.json
Design hash: 7bb675204eed1da5943ee519dd89fad8afa0970082f2aa6b9408013d7003fe54

Frozen do-surgery at 7 scales on 6 UNSEEN architectures (text + vision).
Frozen alpha* = 1.4773 (from LOAO-12 per-dataset fit, no refit allowed).

Text archs (dbpedia14, K=14): gemma-2-2b, phi-2, mamba-130m-hf
Vision archs (CIFAR-10, K=10): vit-base-patch16-224, vit-large-patch16-224, resnet50

Surgery: scale_s in [-0.40, -0.25, -0.10, 0.0, 0.10, 0.25, 0.40]
  negative = push nearest pair closer (decrease kappa)
  positive = push nearest pair apart (increase kappa)
  0.0      = no surgery (neutral control)

Predictions (frozen):
  delta_kappa = kappa_new - kappa_baseline  (measured after surgery)
  delta_z_pred = FROZEN_ALPHA * delta_kappa
  q_pred = sigmoid(logit(q_baseline) + delta_z_pred)
  delta_q_pred = q_pred - q_baseline

PASS CRITERIA (all must pass):
  C1. Directional: >=5/6 archs: mean_delta_q(s<0)<0 AND mean_delta_q(s>0)>0
  C2. Pooled r(delta_q_pred, delta_q_obs) >= 0.90
  C3. Pooled MAE(delta_q_pred, delta_q_obs) <= 0.03
  C4. Text-modality pooled r >= 0.85
  C5. Vision-modality pooled r >= 0.85
  C6. |mean delta_q(s=0)| <= 0.01 (neutral control)
"""

import json, os, sys, time, hashlib
import numpy as np
import torch
from scipy.special import logit as sp_logit, expit as sp_sigmoid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import pearsonr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
print(f"CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}", flush=True)

# ================================================================
# CONFIG (matches preregistration exactly)
# ================================================================
FROZEN_ALPHA   = 1.4773
SURGERY_SCALES = [-0.40, -0.25, -0.10, 0.0, 0.10, 0.25, 0.40]
SEEDS          = [11, 23, 47]
N_SAMPLE       = 5000
DESIGN_HASH    = "7bb675204eed1da5943ee519dd89fad8afa0970082f2aa6b9408013d7003fe54"

# Pass thresholds
PASS_DIRECTIONAL = 5        # out of 6 architectures
PASS_R_POOLED    = 0.90
PASS_MAE_POOLED  = 0.03
PASS_R_TEXT      = 0.85
PASS_R_VISION    = 0.85
PASS_NEUTRAL     = 0.01

# Text architecture config (all OUTSIDE the 12-arch LOAO training set)
TEXT_ARCHS = {
    "gemma-2-2b": {
        "hf_path": "google/gemma-2-2b",
        "layer": -1,          # last hidden layer
        "pooling": "mean",    # mean pool over tokens
        "trust_remote_code": False,
        "dtype": torch.float16,
        "batch_size": 16,
    },
    "phi-2": {
        "hf_path": "microsoft/phi-2",
        "layer": -1,          # last hidden layer (32nd)
        "pooling": "mean",
        "trust_remote_code": True,
        "dtype": torch.float16,
        "batch_size": 16,
    },
    "mamba-130m": {
        "hf_path": "state-spaces/mamba-130m-hf",
        "layer": 23,          # penultimate layer (as in cross_arch_causal_v2)
        "pooling": "last",    # last token
        "trust_remote_code": True,
        "dtype": torch.float16,
        "batch_size": 32,
    },
}

# Vision architecture config
VISION_ARCHS = {
    "vit-base": {
        "emb_cache": "results/vit_loao_embs_vit-base-patch16-224_cifar10.npz",
        "layer_key": "8",     # layer 8 / 12 total (middle layer, not ceiling)
        "hf_path": "google/vit-base-patch16-224",
        "K": 10,
    },
    "vit-large": {
        "emb_cache": "results/vit_loao_embs_vit-large-patch16-224_cifar10.npz",
        "layer_key": "12",    # layer 12 / 24 total (q~0.678 from orthogonal factorial)
        "hf_path": "google/vit-large-patch16-224",
        "K": 10,
    },
    "resnet50": {
        "emb_cache": "results/dose_response_embs_resnet50_cifar10.npz",
        "layer_key": "avgpool",
        "K": 10,
    },
}

OUTPUT_JSON = "results/cti_dose_response.json"


# ================================================================
# GEOMETRY HELPERS
# ================================================================
def compute_class_stats(X, y):
    """Compute centroids and within-class std (pooled, per dimension)."""
    classes = np.unique(y)
    centroids = {}
    within_vars = []
    for c in classes:
        Xc = X[y == c]
        valid = np.all(np.isfinite(Xc), axis=1)
        Xc = Xc[valid] if valid.sum() > 0 else Xc
        centroids[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - centroids[c])**2, axis=1)))
    sigma_W = float(np.sqrt(np.mean(within_vars) / X.shape[1]))
    return centroids, sigma_W


def compute_kappa_nearest(centroids, sigma_W, d):
    """Compute kappa_nearest = min_dist / (sigma_W * sqrt(d))."""
    classes = list(centroids.keys())
    min_dist = np.inf
    nearest_pair = (classes[0], classes[1])
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            if dist < min_dist:
                min_dist = dist
                nearest_pair = (ci, cj)
    kappa = float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))
    return kappa, nearest_pair, float(min_dist)


def compute_q(X, y, K):
    """1-NN accuracy normalized: q = (acc - 1/K) / (1 - 1/K)."""
    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]
    if len(X) < 2 * K:
        return None
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean",
                               algorithm="ball_tree", n_jobs=1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = float((acc - 1.0 / K) / (1.0 - 1.0 / K))
    return float(np.clip(q, 0.0, 1.0))


def logit_q(q):
    return float(sp_logit(np.clip(q, 1e-6, 1 - 1e-6)))


def apply_surgery(X, y, scale, nearest_pair):
    """Apply centroid-shift surgery: move nearest pair by scale * current_distance."""
    centroids, _ = compute_class_stats(X, y)
    j, k = nearest_pair
    diff = centroids[k] - centroids[j]
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    shift = scale * dist / 2.0   # half-shift on each centroid
    X_new = X.copy()
    X_new[y == j] -= shift * direction
    X_new[y == k] += shift * direction
    return X_new


# ================================================================
# TEXT EMBEDDING EXTRACTION
# ================================================================
def extract_text_embeddings(arch_key, cfg, texts, labels):
    """Extract embeddings for text architecture using AutoModel."""
    from transformers import AutoTokenizer, AutoModel
    import warnings
    warnings.filterwarnings("ignore")

    print(f"  Loading {cfg['hf_path']}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["hf_path"], trust_remote_code=cfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        cfg["hf_path"],
        output_hidden_states=True,
        torch_dtype=cfg["dtype"],
        trust_remote_code=cfg["trust_remote_code"],
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()

    all_embs = []
    batch_size = cfg["batch_size"]
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=128, padding=True).to(DEVICE)
            out = model(**enc, output_hidden_states=True)

            hs = out.hidden_states  # tuple of (n_layers+1) tensors
            layer_idx = cfg["layer"]
            h = hs[layer_idx]  # [batch, seq, d]

            if cfg["pooling"] == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (h * mask).sum(1) / mask.sum(1)
            else:  # last token
                lengths = enc["attention_mask"].sum(1) - 1
                emb = h[torch.arange(h.size(0)), lengths]

            e = emb.cpu().float().numpy()
            e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            all_embs.append(e)

            if (i // batch_size) % 20 == 0:
                print(f"    [{arch_key}] batch {i//batch_size+1} / "
                      f"{(len(texts)+batch_size-1)//batch_size} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    X = np.vstack(all_embs)
    # filter zero/inf rows
    valid = np.all(np.isfinite(X), axis=1) & (np.linalg.norm(X, axis=1) > 1e-3)
    return X[valid], np.array(labels)[valid]


def load_dbpedia_texts(n_sample, seed):
    """Sample n_sample texts from dbpedia14 (K=14)."""
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder
    import random

    ds = load_dataset("fancyzhx/dbpedia_14", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_sample, len(ds)))
    texts = [ds["content"][i] for i in indices]
    raw_labels = [ds["label"][i] for i in indices]
    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    return texts, labels, 14


# ================================================================
# VISION EMBEDDING EXTRACTION (ResNet50)
# ================================================================
def extract_resnet50_embeddings(cache_path):
    """Extract ResNet50 avgpool features on CIFAR-10."""
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10

    print("  Loading ResNet50 (ImageNet pretrained)...", flush=True)
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
    # remove fc layer, keep avgpool
    model.fc = torch.nn.Identity()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_set = CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=False
    )

    all_feats = []
    all_labels = []
    t0 = time.time()

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            all_feats.append(feats.cpu().float().numpy())
            all_labels.extend(lbls.numpy().tolist())
            if i % 20 == 0:
                print(f"    ResNet50: batch {i+1}/{len(loader)} ({time.time()-t0:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    X = np.vstack(all_feats)
    y = np.array(all_labels)
    np.savez(cache_path, X=X, y=y)
    print(f"  Saved ResNet50 cache: {X.shape}", flush=True)
    return X, y


def load_vision_embeddings(arch_key, cfg, seed):
    """Load vision embeddings from cache, subsample per seed."""
    if arch_key == "resnet50":
        if not os.path.exists(cfg["emb_cache"]):
            X_full, y_full = extract_resnet50_embeddings(cfg["emb_cache"])
        else:
            d = np.load(cfg["emb_cache"])
            X_full, y_full = d["X"], d["y"]
    else:
        # Load from ViT LOAO cache
        d = np.load(cfg["emb_cache"])
        X_full = d[cfg["layer_key"]]
        y_full = d["y"]

    # Subsample 500/class = 5000 total per seed
    rng = np.random.default_rng(seed)
    classes = np.unique(y_full)
    n_per_class = N_SAMPLE // len(classes)
    sel_idx = []
    for c in classes:
        idx = np.where(y_full == c)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        sel_idx.extend(chosen.tolist())
    sel_idx = np.array(sel_idx)
    rng.shuffle(sel_idx)
    return X_full[sel_idx], y_full[sel_idx]


# ================================================================
# DOSE-RESPONSE SWEEP (one architecture, one seed)
# ================================================================
def run_dose_response_sweep(X, y, K, arch_key, seed_idx):
    """Apply all 7 surgery scales. Return predictions and observations."""
    d = X.shape[1]

    # Baseline
    centroids, sigma_W = compute_class_stats(X, y)
    kappa_base, nearest_pair, min_dist_base = compute_kappa_nearest(centroids, sigma_W, d)
    q_base = compute_q(X, y, K)
    if q_base is None:
        print(f"  WARNING: q_base is None for {arch_key} seed={seed_idx}", flush=True)
        return None

    z_base = logit_q(q_base)
    print(f"  Baseline: kappa={kappa_base:.4f}, q={q_base:.4f}, "
          f"nearest_pair={nearest_pair}", flush=True)

    results = []
    for scale in SURGERY_SCALES:
        X_new = apply_surgery(X, y, scale, nearest_pair)

        # Recompute geometry
        new_centroids, new_sigma_W = compute_class_stats(X_new, y)
        kappa_new, _, _ = compute_kappa_nearest(new_centroids, new_sigma_W, d)
        delta_kappa = float(kappa_new - kappa_base)

        # Frozen prediction
        delta_z_pred = FROZEN_ALPHA * delta_kappa
        q_pred = float(sp_sigmoid(z_base + delta_z_pred))
        delta_q_pred = q_pred - q_base

        # Observation
        q_obs = compute_q(X_new, y, K)
        if q_obs is None:
            print(f"  WARNING: q_obs=None at scale={scale}", flush=True)
            continue
        delta_q_obs = float(q_obs - q_base)

        row = {
            "scale": float(scale),
            "kappa_base": float(kappa_base),
            "kappa_new": float(kappa_new),
            "delta_kappa": delta_kappa,
            "q_base": float(q_base),
            "q_obs": float(q_obs),
            "q_pred": float(q_pred),
            "delta_q_obs": delta_q_obs,
            "delta_q_pred": float(delta_q_pred),
        }
        results.append(row)
        print(f"    scale={scale:+.2f}: dk={delta_kappa:+.4f}, "
              f"q_obs={q_obs:.4f}, q_pred={q_pred:.4f}, "
              f"dq_obs={delta_q_obs:+.4f}, dq_pred={delta_q_pred:+.4f}", flush=True)

    return results, float(kappa_base), float(q_base)


# ================================================================
# PASS CRITERIA EVALUATION
# ================================================================
def evaluate_pass_criteria(all_arch_results):
    """Evaluate all pre-registered pass criteria."""
    print("\n\n" + "=" * 70, flush=True)
    print("PASS CRITERIA EVALUATION", flush=True)
    print("=" * 70, flush=True)

    # Collect pooled prediction/observation pairs across all archs, seeds, nonzero scales
    pooled_pred = []
    pooled_obs  = []
    text_pred, text_obs = [], []
    vision_pred, vision_obs = [], []
    neutral_deltas = []

    directional_pass = 0
    directional_total = 0

    for arch_key, arch_data in all_arch_results.items():
        is_text = arch_key in TEXT_ARCHS
        arch_seed_results = arch_data.get("seed_results", {})

        # Collect all non-s=0 pairs for directional test (per arch, mean over seeds)
        neg_dq_obs = []
        pos_dq_obs = []

        for seed, seed_rows in arch_seed_results.items():
            for row in seed_rows:
                s = row["scale"]
                if abs(s) < 1e-6:  # neutral control
                    neutral_deltas.append(abs(row["delta_q_obs"]))
                    continue
                pooled_pred.append(row["delta_q_pred"])
                pooled_obs.append(row["delta_q_obs"])
                if is_text:
                    text_pred.append(row["delta_q_pred"])
                    text_obs.append(row["delta_q_obs"])
                else:
                    vision_pred.append(row["delta_q_pred"])
                    vision_obs.append(row["delta_q_obs"])

                if s < 0:
                    neg_dq_obs.append(row["delta_q_obs"])
                else:
                    pos_dq_obs.append(row["delta_q_obs"])

        # Directional: mean_delta_q(s<0) < 0 AND mean_delta_q(s>0) > 0
        directional_total += 1
        mean_neg = float(np.mean(neg_dq_obs)) if neg_dq_obs else 0.0
        mean_pos = float(np.mean(pos_dq_obs)) if pos_dq_obs else 0.0
        arch_dir_pass = (mean_neg < 0) and (mean_pos > 0)
        if arch_dir_pass:
            directional_pass += 1
        print(f"  [{arch_key}] directional: neg={mean_neg:+.4f}, pos={mean_pos:+.4f} "
              f"-> {'PASS' if arch_dir_pass else 'FAIL'}", flush=True)

    # Compute correlation and MAE
    pooled_pred = np.array(pooled_pred)
    pooled_obs  = np.array(pooled_obs)
    text_pred   = np.array(text_pred)
    text_obs    = np.array(text_obs)
    vision_pred = np.array(vision_pred)
    vision_obs  = np.array(vision_obs)

    r_pooled = float(pearsonr(pooled_pred, pooled_obs)[0]) if len(pooled_pred) >= 4 else float("nan")
    mae_pooled = float(np.mean(np.abs(pooled_pred - pooled_obs)))
    r_text   = float(pearsonr(text_pred, text_obs)[0]) if len(text_pred) >= 4 else float("nan")
    r_vision = float(pearsonr(vision_pred, vision_obs)[0]) if len(vision_pred) >= 4 else float("nan")
    mean_neutral = float(np.mean(neutral_deltas)) if neutral_deltas else float("nan")

    c1 = directional_pass >= PASS_DIRECTIONAL
    c2 = not np.isnan(r_pooled) and r_pooled >= PASS_R_POOLED
    c3 = mae_pooled <= PASS_MAE_POOLED
    c4 = not np.isnan(r_text)   and r_text   >= PASS_R_TEXT
    c5 = not np.isnan(r_vision) and r_vision >= PASS_R_VISION
    c6 = not np.isnan(mean_neutral) and mean_neutral <= PASS_NEUTRAL

    print(f"\n  C1. Directional: {directional_pass}/{directional_total} >= {PASS_DIRECTIONAL} -> "
          f"{'PASS' if c1 else 'FAIL'}", flush=True)
    print(f"  C2. Pooled r: {r_pooled:.4f} >= {PASS_R_POOLED} -> {'PASS' if c2 else 'FAIL'}", flush=True)
    print(f"  C3. Pooled MAE: {mae_pooled:.4f} <= {PASS_MAE_POOLED} -> {'PASS' if c3 else 'FAIL'}", flush=True)
    print(f"  C4. Text r: {r_text:.4f} >= {PASS_R_TEXT} -> {'PASS' if c4 else 'FAIL'}", flush=True)
    print(f"  C5. Vision r: {r_vision:.4f} >= {PASS_R_VISION} -> {'PASS' if c5 else 'FAIL'}", flush=True)
    print(f"  C6. Neutral |dq|: {mean_neutral:.4f} <= {PASS_NEUTRAL} -> {'PASS' if c6 else 'FAIL'}", flush=True)

    overall_pass = c1 and c2 and c3 and c4 and c5 and c6
    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)
    if not overall_pass:
        failed = []
        if not c1: failed.append("C1-directional")
        if not c2: failed.append("C2-r_pooled")
        if not c3: failed.append("C3-mae")
        if not c4: failed.append("C4-text_r")
        if not c5: failed.append("C5-vision_r")
        if not c6: failed.append("C6-neutral")
        print(f"  Failed criteria: {failed}", flush=True)

    return {
        "overall_pass": overall_pass,
        "directional_pass": directional_pass,
        "directional_total": directional_total,
        "r_pooled": float(r_pooled) if not np.isnan(r_pooled) else None,
        "mae_pooled": float(mae_pooled),
        "r_text": float(r_text) if not np.isnan(r_text) else None,
        "r_vision": float(r_vision) if not np.isnan(r_vision) else None,
        "mean_neutral_delta_q": float(mean_neutral) if not np.isnan(mean_neutral) else None,
        "n_pooled_pairs": int(len(pooled_pred)),
        "criteria": {
            "C1_directional": bool(c1),
            "C2_r_pooled": bool(c2),
            "C3_mae": bool(c3),
            "C4_text_r": bool(c4),
            "C5_vision_r": bool(c5),
            "C6_neutral": bool(c6),
        }
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70, flush=True)
    print("CTI PRE-REGISTERED DOSE-RESPONSE CAUSAL EXPERIMENT", flush=True)
    print("=" * 70, flush=True)
    print(f"Design hash: {DESIGN_HASH}", flush=True)
    print(f"Frozen alpha*: {FROZEN_ALPHA}", flush=True)
    print(f"Surgery scales: {SURGERY_SCALES}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(flush=True)

    all_results = {
        "experiment": "cti_dose_response_preregistered",
        "design_hash": DESIGN_HASH,
        "frozen_alpha": FROZEN_ALPHA,
        "surgery_scales": SURGERY_SCALES,
        "seeds": SEEDS,
        "pass_criteria": {
            "directional": f">={PASS_DIRECTIONAL}/6 archs",
            "r_pooled": f">={PASS_R_POOLED}",
            "mae_pooled": f"<={PASS_MAE_POOLED}",
            "r_text": f">={PASS_R_TEXT}",
            "r_vision": f">={PASS_R_VISION}",
            "neutral": f"<={PASS_NEUTRAL}",
        }
    }

    # --------------------------------------------------------
    # BLOCK 1: TEXT ARCHITECTURES
    # --------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("BLOCK 1: TEXT ARCHITECTURES (dbpedia14, K=14)", flush=True)
    print("=" * 70, flush=True)

    for arch_key, cfg in TEXT_ARCHS.items():
        print(f"\n--- {arch_key.upper()} ---", flush=True)
        arch_result = {"modality": "text", "arch": arch_key, "dataset": "dbpedia14", "K": 14, "seed_results": {}}

        # Extract full embeddings once per arch (seed only controls subsample)
        emb_cache = f"results/dose_response_embs_{arch_key}_dbpedia.npz"

        if os.path.exists(emb_cache):
            d = np.load(emb_cache)
            X_full, y_full = d["X"], d["y"]
            print(f"  Loaded embedding cache: {X_full.shape}", flush=True)
        else:
            print(f"  Extracting embeddings for {arch_key} on dbpedia14...", flush=True)
            # Use seed 0 to get all data; subsample per actual seed below
            texts, y_full, K = load_dbpedia_texts(n_sample=min(15000, 560000), seed=0)
            X_full, y_full = extract_text_embeddings(arch_key, cfg, texts, y_full)
            np.savez(emb_cache, X=X_full, y=y_full)
            print(f"  Saved embedding cache: {X_full.shape}", flush=True)

        for seed in SEEDS:
            print(f"\n  Seed {seed}:", flush=True)
            rng = np.random.default_rng(seed)
            classes = np.unique(y_full)
            K = 14
            n_per_class = N_SAMPLE // K

            sel_idx = []
            for c in classes:
                idx = np.where(y_full == c)[0]
                chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
                sel_idx.extend(chosen.tolist())
            sel_idx = np.array(sel_idx)
            rng.shuffle(sel_idx)
            X_seed, y_seed = X_full[sel_idx], y_full[sel_idx]

            t0 = time.time()
            out = run_dose_response_sweep(X_seed, y_seed, K, arch_key, seed)
            if out is not None:
                rows, kappa_base, q_base = out
                arch_result["seed_results"][str(seed)] = rows
                arch_result[f"kappa_base_seed{seed}"] = kappa_base
                arch_result[f"q_base_seed{seed}"] = q_base
            print(f"  Seed {seed} done in {time.time()-t0:.1f}s", flush=True)

        all_results[arch_key] = arch_result
        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved partial results -> {OUTPUT_JSON}", flush=True)

    # --------------------------------------------------------
    # BLOCK 2: VISION ARCHITECTURES
    # --------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("BLOCK 2: VISION ARCHITECTURES (CIFAR-10, K=10)", flush=True)
    print("=" * 70, flush=True)

    for arch_key, cfg in VISION_ARCHS.items():
        print(f"\n--- {arch_key.upper()} ---", flush=True)
        arch_result = {"modality": "vision", "arch": arch_key, "dataset": "cifar10", "K": 10, "seed_results": {}}

        # For ResNet50: extract if not cached
        if arch_key == "resnet50" and not os.path.exists(cfg["emb_cache"]):
            print("  Extracting ResNet50 features...", flush=True)
            extract_resnet50_embeddings(cfg["emb_cache"])

        K = cfg["K"]
        for seed in SEEDS:
            print(f"\n  Seed {seed}:", flush=True)
            X_seed, y_seed = load_vision_embeddings(arch_key, cfg, seed)

            t0 = time.time()
            out = run_dose_response_sweep(X_seed, y_seed, K, arch_key, seed)
            if out is not None:
                rows, kappa_base, q_base = out
                arch_result["seed_results"][str(seed)] = rows
                arch_result[f"kappa_base_seed{seed}"] = kappa_base
                arch_result[f"q_base_seed{seed}"] = q_base
            print(f"  Seed {seed} done in {time.time()-t0:.1f}s", flush=True)

        all_results[arch_key] = arch_result
        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved partial results -> {OUTPUT_JSON}", flush=True)

    # --------------------------------------------------------
    # PASS CRITERIA EVALUATION
    # --------------------------------------------------------
    arch_only = {k: v for k, v in all_results.items()
                 if k in list(TEXT_ARCHS.keys()) + list(VISION_ARCHS.keys())}
    summary = evaluate_pass_criteria(arch_only)
    all_results["summary"] = summary

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved -> {OUTPUT_JSON}", flush=True)

    # Final summary
    print("\n" + "=" * 70, flush=True)
    overall = summary.get("overall_pass", False)
    print(f"DOSE-RESPONSE EXPERIMENT: {'PASS' if overall else 'FAIL'}", flush=True)
    print(f"  r_pooled = {summary.get('r_pooled')}", flush=True)
    print(f"  mae_pooled = {summary.get('mae_pooled')}", flush=True)
    print(f"  r_text = {summary.get('r_text')}", flush=True)
    print(f"  r_vision = {summary.get('r_vision')}", flush=True)
    print(f"  neutral = {summary.get('mean_neutral_delta_q')}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
