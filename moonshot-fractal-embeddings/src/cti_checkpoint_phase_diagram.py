#!/usr/bin/env python -u
"""
CHECKPOINT PHASE DIAGRAM: B_j2 Effect vs Training Progress (Feb 21 2026)
=========================================================================
Codex recommendation (highest-leverage): run orthogonal Arm B across
TRAINING CHECKPOINTS of Pythia-160m on DBpedia.

DESIGN:
  For each checkpoint step (5 steps: 512, 2000, 16000, 48000, 143000):
  1. Extract DBpedia embeddings from Pythia-160m at that checkpoint (layer 12)
  2. Compute geometry: kappa_j1, margin (kappa_j2/kappa_j1), q_ci for each class
  3. Run Arm A (standard centroid shift on nearest competitor j1)
  4. Run Arm B (ORTHOGONAL: move ONLY j2, kappa_j1 unchanged)
  5. Record B_j2_r per focus class

PRE-REGISTERED PREDICTION (from Gumbel Race + margin phase diagram):
  As training progresses (kappa increases, margin may widen):
  - Early checkpoints (low kappa, classes overlap): B_j2_r is HIGH (j2 also confuses)
  - Late checkpoints (high kappa, ETF approach): B_j2_r is LOW (j1 dominates)

  Spearman r(step, mean_B_j2_r) < 0   [B_j2_r DECREASES as training progresses]
  This would confirm that the 1-layer CTI law becomes increasingly accurate with training.

CHECKPOINT STEPS:
  step-512    (very early: ~3M tokens seen, kappa very low ~0.1-0.2)
  step-2000   (early: ~12M tokens, kappa ~0.2-0.3)
  step-16000  (mid: ~97M tokens, kappa ~0.3-0.5)
  step-48000  (late-mid: ~290M tokens, kappa ~0.5-0.6)
  step-143000 (final: ~859M tokens, kappa ~0.5-0.8 best layer)
"""

import json
import os
import sys
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# CONFIG
# ================================================================
MODEL_NAME  = "EleutherAI/pythia-160m"
DATASET     = {"hf_name": "fancyzhx/dbpedia_14", "text_col": "content", "label_col": "label", "K": 14}
N_SAMPLE    = 1000
BATCH_SIZE  = 64
LAYER       = 12    # best layer for Pythia-160m on DBpedia (from previous experiments)

CHECKPOINT_STEPS = ["step512", "step2000", "step16000", "step48000", "step143000"]

OUT_JSON = "results/cti_checkpoint_phase_diagram.json"
OUT_LOG  = "results/cti_checkpoint_phase_diagram_log.txt"

# Orthogonal Arm B config
N_DELTA_A  = 9
N_DELTA_B  = 9
DELTA_A    = np.linspace(-2.0, 2.0, N_DELTA_A)
DELTA_B    = np.linspace(0.0, 4.0, N_DELTA_B)

# Only run on N_FOCUS focus classes (to keep runtime reasonable)
N_FOCUS    = 6      # use first 6 classes
N_CV       = 5      # 5-fold CV (faster than 10-fold)

# PRE-REGISTERED: Spearman r(step, mean_B_j2_r) should be < 0
PRE_REG_SPEARMAN = 0.0   # direction: should be negative


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================
def extract_embeddings(model_name, step_revision, dataset_cfg, n_sample, batch_size, layer, device):
    """Extract embeddings from a specific checkpoint."""
    print(f"  [EXTRACT] Loading {model_name} @ {step_revision}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=step_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name, revision=step_revision,
        torch_dtype=torch.float32, output_hidden_states=True
    ).to(device).eval()

    print(f"  [EXTRACT] Loading dataset...", flush=True)
    raw = load_dataset(dataset_cfg["hf_name"], split="test", trust_remote_code=True)
    le = LabelEncoder()
    labels = le.fit_transform(raw[dataset_cfg["label_col"]])
    texts = raw[dataset_cfg["text_col"]]
    K = dataset_cfg["K"]

    # Stratified subsample
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_sample, random_state=42)
    _, idx = next(sss.split(np.zeros(len(labels)), labels))
    texts_sub = [texts[i] for i in idx]
    labels_sub = labels[idx]

    print(f"  [EXTRACT] Encoding {n_sample} samples at layer {layer}...", flush=True)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts_sub), batch_size):
            batch = texts_sub[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # Use hidden state at specified layer (mean-pool)
            h = out.hidden_states[layer]   # [B, seq_len, d]
            attn_mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * attn_mask).sum(1) / attn_mask.sum(1)
            all_embs.append(pooled.cpu().numpy())

    X = np.vstack(all_embs).astype(np.float32)
    y = labels_sub.astype(np.int64)

    # Free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return X, y


# ================================================================
# GEOMETRY
# ================================================================
def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        centroids[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R**2)))
    return centroids, sigma_W


def get_ranking(centroids, sigma_W, d, ci):
    mu_i = centroids[ci]
    ranking = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa_ij = dist / (sigma_W * np.sqrt(d) + 1e-10)
        ranking.append((kappa_ij, cj, dist))
    ranking.sort(key=lambda x: x[0])
    return ranking


def compute_per_class_q(X, y, ci, n_splits=N_CV, seed=42):
    K = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    recalls = []
    for tr, te in skf.split(X, y):
        if (y[tr] == ci).sum() < 2:
            continue
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X[tr], y[tr])
        mask = (y[te] == ci)
        if mask.sum() == 0:
            continue
        recalls.append(float((knn.predict(X[te][mask]) == ci).mean()))
    if not recalls:
        return None
    q = float(np.mean(recalls))
    K_inv = 1.0 / K
    return float((q - K_inv) / (1.0 - K_inv))


def logit(q):
    return float(np.log(np.clip(q, 1e-5, 1-1e-5) / np.clip(1-q, 1e-5, 1-1e-5)))


# ================================================================
# SURGERY ARMS
# ================================================================
def apply_centroid_shift(X, y, centroids, cj, ck, delta):
    mu_j, mu_k = centroids[cj].copy(), centroids[ck].copy()
    diff = mu_k - mu_j
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] -= (delta / 2) * direction
    X_new[y == ck] += (delta / 2) * direction
    return X_new


def apply_competitor_shift(X, y, centroids, ci, cj, delta):
    """Orthogonal: move ONLY cj away from ci. ci unchanged."""
    mu_i = centroids[ci]
    mu_j = centroids[cj].copy()
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] += delta * direction
    return X_new


def fit_r(recs, xk, yk="logit_q0"):
    xs = np.array([r[xk] for r in recs if r.get(xk) is not None])
    ys = np.array([r[yk] for r in recs if r.get(xk) is not None])
    if len(xs) < 4 or np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
        return 0.0, 1.0
    r, p = stats.pearsonr(xs, ys)
    return float(r), float(p)


# ================================================================
# CHECKPOINT LOOP
# ================================================================
def run_checkpoint(X, y, d, K, focus_classes, centroids, sigma_W, log_fn):
    """Run Arm A and Arm B on one checkpoint's embeddings."""
    results = {}
    for ci in focus_classes:
        ranking = get_ranking(centroids, sigma_W, d, ci)
        j1 = ranking[0][1]; kappa_j1 = ranking[0][0]
        j2 = ranking[1][1]; kappa_j2 = ranking[1][0]
        margin = kappa_j2 / (kappa_j1 + 1e-10)

        q0_base = compute_per_class_q(X, y, ci)
        if q0_base is None:
            continue

        log_fn(f"    ci={ci}: kappa_j1={kappa_j1:.3f} j2={j2} kappa_j2={kappa_j2:.3f} "
               f"margin={margin:.2f}x q_ci={q0_base:.4f}")

        # Arm A
        recs_A = []
        for delta in DELTA_A:
            X_new = apply_centroid_shift(X, y, centroids, ci, j1, delta)
            c_new, sw_new = compute_class_stats(X_new, y)
            rank_new = get_ranking(c_new, sw_new, d, ci)
            kappa_j1_new = rank_new[0][0]
            q_ci = compute_per_class_q(X_new, y, ci)
            if q_ci is None:
                continue
            recs_A.append({"delta": float(delta), "kappa_j1_new": kappa_j1_new, "logit_q0": logit(q_ci)})
        r_A, p_A = fit_r(recs_A, "kappa_j1_new")

        # Arm B (orthogonal)
        recs_B = []
        for delta in DELTA_B:
            X_new = apply_competitor_shift(X, y, centroids, ci, j2, delta)
            c_new, sw_new = compute_class_stats(X_new, y)
            rank_new = get_ranking(c_new, sw_new, d, ci)
            kappa_j1_new = rank_new[0][0]   # should be unchanged
            kappa_j2_new = None
            for k, j, dist_ in rank_new:
                if j == j2:
                    kappa_j2_new = k
                    break
            q_ci = compute_per_class_q(X_new, y, ci)
            if q_ci is None:
                continue
            recs_B.append({
                "delta": float(delta),
                "kappa_j1_new": kappa_j1_new,
                "kappa_j2_new": float(kappa_j2_new) if kappa_j2_new is not None else None,
                "logit_q0": logit(q_ci),
            })
        r_B_kappa, _ = fit_r(recs_B, "kappa_j1_new")
        r_B_j2, p_B_j2 = fit_r(recs_B, "kappa_j2_new")

        log_fn(f"      Arm A: r(kappa_j1, logit)={r_A:.3f} p={p_A:.4f}")
        log_fn(f"      Arm B: r(kappa_j1_unchanged={r_B_kappa:.3f}) r(kappa_j2={r_B_j2:.3f} p={p_B_j2:.4f})")

        results[int(ci)] = {
            "ci": int(ci), "j1": int(j1), "j2": int(j2),
            "kappa_j1": float(kappa_j1), "kappa_j2": float(kappa_j2), "margin": float(margin),
            "q_ci_baseline": float(q0_base),
            "arm_A_r": float(r_A), "arm_A_p": float(p_A),
            "arm_B_r_kappa_j1_unchanged": float(r_B_kappa),
            "arm_B_r_kappa_j2": float(r_B_j2), "arm_B_p": float(p_B_j2),
        }
    return results


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs("results", exist_ok=True)
    log_file = open(OUT_LOG, "w", buffering=1)
    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log("=" * 70)
    log("CHECKPOINT PHASE DIAGRAM: B_j2 vs Training Progress")
    log("=" * 70)
    log(f"Model: {MODEL_NAME}")
    log(f"Checkpoints: {CHECKPOINT_STEPS}")
    log(f"Dataset: DBpedia K=14, N={N_SAMPLE}")
    log(f"Layer: {LAYER}, Focus classes: 0..{N_FOCUS-1}")
    log("PRE-REGISTERED: Spearman r(step, mean_B_j2_r) < 0")
    log("=" * 70)

    focus_classes = list(range(N_FOCUS))
    all_checkpoint_results = {}

    for step_rev in CHECKPOINT_STEPS:
        log(f"\n{'='*60}")
        log(f"CHECKPOINT: {step_rev}")
        log(f"{'='*60}")

        # Extract embeddings
        try:
            X, y = extract_embeddings(
                MODEL_NAME, step_rev, DATASET, N_SAMPLE, BATCH_SIZE, LAYER, DEVICE
            )
        except Exception as e:
            log(f"  ERROR extracting embeddings: {e}")
            continue

        d = X.shape[1]
        K = len(np.unique(y))
        centroids, sigma_W = compute_class_stats(X, y)
        log(f"  Embedded: shape={X.shape}, K={K}, sigma_W={sigma_W:.4f}")

        # Overall kappa_nearest (global, not per-class)
        all_kappas = []
        for ci in focus_classes:
            ranking = get_ranking(centroids, sigma_W, d, ci)
            if ranking:
                all_kappas.append(ranking[0][0])
        mean_kappa = float(np.mean(all_kappas)) if all_kappas else 0.0
        log(f"  Mean kappa_nearest (focus classes): {mean_kappa:.4f}")

        # Run arms
        ckpt_results = run_checkpoint(X, y, d, K, focus_classes, centroids, sigma_W, log)

        # Aggregate B_j2_r across focus classes
        rs_B_j2 = [r["arm_B_r_kappa_j2"] for r in ckpt_results.values() if "arm_B_r_kappa_j2" in r]
        rs_A = [r["arm_A_r"] for r in ckpt_results.values() if "arm_A_r" in r]
        mean_B_j2_r = float(np.tanh(np.arctanh(np.clip(rs_B_j2, -0.9999, 0.9999)).mean())) if rs_B_j2 else 0.0
        mean_A_r = float(np.tanh(np.arctanh(np.clip(rs_A, -0.9999, 0.9999)).mean())) if rs_A else 0.0

        log(f"\n  CHECKPOINT SUMMARY ({step_rev}):")
        log(f"    mean kappa_j1 = {mean_kappa:.4f}")
        log(f"    mean Arm A r(kappa, logit) = {mean_A_r:.3f}")
        log(f"    mean Arm B r(kappa_j2, logit) = {mean_B_j2_r:.3f}")
        log(f"    per-class B_j2_r: {[f'{r:.2f}' for r in rs_B_j2]}")

        all_checkpoint_results[step_rev] = {
            "step_rev": step_rev,
            "mean_kappa_j1": mean_kappa,
            "mean_arm_A_r": mean_A_r,
            "mean_arm_B_j2_r": mean_B_j2_r,
            "per_class": ckpt_results,
        }

        del X, y, centroids
        gc.collect()

    # ----------------------------------------------------------------
    # PHASE DIAGRAM VERDICT
    # ----------------------------------------------------------------
    log("\n" + "="*70)
    log("PHASE DIAGRAM VERDICT")
    log("="*70)

    steps_done = [k for k in CHECKPOINT_STEPS if k in all_checkpoint_results]
    if len(steps_done) >= 3:
        step_nums = list(range(len(steps_done)))
        mean_Bs = [all_checkpoint_results[s]["mean_arm_B_j2_r"] for s in steps_done]
        mean_kappas = [all_checkpoint_results[s]["mean_kappa_j1"] for s in steps_done]

        rho_step, p_step = stats.spearmanr(step_nums, mean_Bs)
        rho_kappa, p_kappa = stats.spearmanr(mean_kappas, mean_Bs)

        log(f"\nPhase diagram table:")
        log(f"  {'Step':>12} | {'kappa_j1':>10} | {'Arm A r':>9} | {'Arm B r':>9}")
        log(f"  {'-'*50}")
        for s in steps_done:
            r = all_checkpoint_results[s]
            log(f"  {s:>12} | {r['mean_kappa_j1']:>10.4f} | {r['mean_arm_A_r']:>9.3f} | {r['mean_arm_B_j2_r']:>9.3f}")

        log(f"\nSpearman r(step, B_j2_r)  = {rho_step:.3f} p={p_step:.4f}")
        log(f"Spearman r(kappa, B_j2_r) = {rho_kappa:.3f} p={p_kappa:.4f}")
        log(f"Pre-reg: both should be NEGATIVE")

        if rho_step < 0 and p_step < 0.10:
            log(f"\nPASS: B_j2_r decreases with training -> 1-layer law increasingly accurate")
        else:
            log(f"\nFAIL: B_j2_r does not decrease with training step")

        if rho_kappa < 0 and p_kappa < 0.10:
            log(f"PASS: B_j2_r decreases with kappa -> phase transition confirmed")
        else:
            log(f"FAIL: B_j2_r does not decrease with kappa")

    # Save
    out = {
        "experiment": "checkpoint_phase_diagram",
        "description": "Arm B effect as training progresses. Tests 1-layer law increasingly accurate.",
        "config": {
            "model": MODEL_NAME, "layer": LAYER, "n_sample": N_SAMPLE,
            "checkpoints": CHECKPOINT_STEPS, "focus_classes": N_FOCUS,
        },
        "results": all_checkpoint_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nSaved to {OUT_JSON}")
    log_file.close()


if __name__ == "__main__":
    main()
