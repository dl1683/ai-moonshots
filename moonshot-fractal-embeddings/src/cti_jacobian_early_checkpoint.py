#!/usr/bin/env python -u
"""
Real-Embedding Jacobian Test: Pythia-160m Step=512 (Session 40)
==================================================================
Codex priority #1 (Session 39 end): Real-embedding Jacobian in non-ceiling regime.

USE CASE: Step=512 has B_j2_r=0.831 (strong j2 competition) AND low q (non-ceiling).
This is the ONLY checkpoint where we can measure per-class w_j2 reliably.

PRE-REGISTERED (tau*=0.20 FIXED a priori from phi_upgrade_pooled + synthetic Jacobian):
  PREDICTION: w_j2 = exp(-(kappa_j2 - kappa_j1) / tau*)
  TEST: r(log(w_j2), -(kappa_j2 - kappa_j1) / tau*) > 0.85

This is an OUT-OF-SAMPLE test: tau* was fixed from earlier experiments (NOT fitted here).

DESIGN:
  1. Extract DBpedia embeddings at step=512, layer=12 (load checkpoint from HuggingFace)
  2. For each class ci:
     a. Compute kappa_j1, kappa_j2, gap = kappa_j2 - kappa_j1
     b. Move j2 outward from ci (ci FIXED) over DELTA_OUT range
     c. Fit slope alpha_j2 = d(logit_q)/d(delta_kappa_j2)
     d. Compute w_j2 = alpha_j2 / alpha_j1 (normalized by rank-1 slope)
  3. Test: r(log(w_j2), -gap/tau*) > 0.85 [PRE-REGISTERED]
  4. Secondary: fit tau from data (should = 0.20)

Also runs the FULL rank sweep (all K-1 competitors) to compare with final checkpoint.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_NPZ = RESULTS_DIR / "checkpoint_embs_pythia-160m_step512.npz"
OUT_JSON = RESULTS_DIR / "cti_jacobian_early_checkpoint.json"

# Pre-registered
TAU_STAR = 0.20   # FIXED from phi_upgrade_pooled and synthetic Jacobian
R_THRESHOLD = 0.85  # pre-registered pass threshold

# Embedding extraction
MODEL_NAME = "EleutherAI/pythia-160m"
STEP_REVISION = "step512"
LAYER = 12
DATASET_NAME = "fancyzhx/dbpedia_14"
N_SAMPLE = 7000   # match existing dointerv_multi files for apples-to-apples comparison
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SPLITS_CV = 5
DELTA_OUT = np.linspace(0.0, 5.0, 11)


def extract_embeddings():
    """Extract step=512 embeddings from pythia-160m checkpoint."""
    print(f"Loading model {MODEL_NAME} at revision {STEP_REVISION}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=STEP_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_NAME, revision=STEP_REVISION,
        torch_dtype=torch.float16, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    print(f"Loading dataset {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="test", trust_remote_code=True)
    all_texts = list(ds["content"])
    all_labels = np.array(ds["label"], dtype=np.int64)
    # Stratified subsample to get balanced class coverage
    sss = StratifiedShuffleSplit(n_splits=1, test_size=N_SAMPLE, random_state=42)
    _, idx = next(sss.split(np.zeros(len(all_labels)), all_labels))
    texts = [all_texts[i] for i in idx]
    labels = all_labels[idx]
    print(f"Stratified sample: N={len(texts)}, classes={np.unique(labels).tolist()}")

    print(f"Extracting embeddings at layer {LAYER} (N={len(texts)})...")
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[LAYER]  # 0=embedding, 1..N=transformer layers
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        all_embs.append(pooled.cpu().float().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  batch {i // BATCH_SIZE + 1}/{len(texts) // BATCH_SIZE + 1}", flush=True)

    X = np.concatenate(all_embs, axis=0).astype(np.float32)
    y = labels
    print(f"Extracted X.shape={X.shape}, y={y.min()}-{y.max()}")

    np.savez_compressed(str(CACHE_NPZ), X=X, y=y)
    print(f"Saved to {CACHE_NPZ}")

    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X, y


def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(0)
        centroids[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R**2)))
    return centroids, sigma_W


def compute_all_kappas_sorted(centroids, sigma_W, d, ci):
    mu_i = centroids[ci]
    kappas_with_class = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        k = dist / (sigma_W * np.sqrt(d) + 1e-10)
        kappas_with_class.append((k, cj))
    kappas_with_class.sort(key=lambda x: x[0])
    return kappas_with_class  # list of (kappa, class_id) sorted ascending


def compute_per_class_q(X, y, ci, n_splits=N_SPLITS_CV):
    K_local = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    recalls = []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if (y_tr == ci).sum() < 1:
            continue
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X_tr, y_tr)
        mask = y_te == ci
        if mask.sum() == 0:
            continue
        preds = knn.predict(X_te[mask])
        recalls.append(float((preds == ci).mean()))
    if not recalls:
        return None
    q_raw = float(np.mean(recalls))
    return float((q_raw - 1.0 / K_local) / (1.0 - 1.0 / K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1 - 1e-5))
    return float(np.log(q / (1.0 - q)))


def apply_competitor_shift(X, y, centroids, ci, cj, delta):
    """Move ONLY cj outward from ci by delta (ci FIXED)."""
    mu_i, mu_j = centroids[ci], centroids[cj]
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] += delta * direction
    return X_new


def fit_slope(deltas, logit_qs):
    """Fit slope of (delta_kappa_cj, delta_logit_q)."""
    if len(deltas) < 4 or np.std(deltas) < 1e-8 or np.std(logit_qs) < 1e-8:
        return 0.0, 0.0, 1.0
    r, p = pearsonr(deltas, logit_qs)
    slope, _, _, _, _ = linregress(deltas, logit_qs)
    return float(slope), float(r), float(p)


def main():
    print("=" * 70)
    print("REAL-EMBEDDING JACOBIAN TEST: PYTHIA-160M STEP=512")
    print(f"PRE-REGISTERED: tau*={TAU_STAR} FIXED, r threshold={R_THRESHOLD}")
    print(f"Test: r(log(w_j2), -gap/tau*) > {R_THRESHOLD}")
    print("=" * 70)

    # Load or extract embeddings
    if CACHE_NPZ.exists():
        print(f"Loading cached embeddings from {CACHE_NPZ}")
        data = np.load(str(CACHE_NPZ))
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64)
    else:
        print(f"Cache not found, extracting from checkpoint...")
        X, y = extract_embeddings()
        X = X.astype(np.float64)

    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    K = len(classes)
    print(f"\nData: N={len(X)}, d={d}, K={K}")

    centroids, sigma_W = compute_class_stats(X, y)

    # Compute baselines
    print("\nComputing per-class geometry...")
    per_class_geom = {}
    for ci in classes:
        kappas_list = compute_all_kappas_sorted(centroids, sigma_W, d, ci)
        if len(kappas_list) < 2:
            continue
        kappa_j1, j1 = kappas_list[0]
        kappa_j2, j2 = kappas_list[1]
        gap = kappa_j2 - kappa_j1
        q = compute_per_class_q(X, y, ci)
        if q is None:
            continue
        lq = safe_logit(q)
        per_class_geom[ci] = {
            "j1": j1, "j2": j2,
            "kappa_j1": kappa_j1, "kappa_j2": kappa_j2,
            "gap": gap, "q": q, "logit_q": lq,
            "all_kappas": kappas_list,
        }
        print(f"  ci={ci}: kappa_j1={kappa_j1:.4f}, kappa_j2={kappa_j2:.4f}, gap={gap:.4f}, q={q:.4f}")

    # Single-competitor Jacobian: move j2 (ci fixed), measure slope
    print("\n" + "=" * 70)
    print("RUNNING j2 JACOBIAN TEST")
    print("=" * 70)

    baseline_logit_q_ci = {}
    for ci in per_class_geom:
        baseline_logit_q_ci[ci] = per_class_geom[ci]["logit_q"]

    jacobian_results = []
    alpha_j1_list = []
    alpha_j2_list = []
    gap_list = []

    for ci in sorted(per_class_geom.keys()):
        geom = per_class_geom[ci]
        j1 = geom["j1"]
        j2 = geom["j2"]
        kappa_j1 = geom["kappa_j1"]
        kappa_j2 = geom["kappa_j2"]
        gap = geom["gap"]
        blq = geom["logit_q"]

        print(f"\nClass {ci} (j1={j1}, j2={j2}, gap={gap:.4f}, kappa_j1={kappa_j1:.4f}, kappa_j2={kappa_j2:.4f})")

        # Measure alpha_j1 (rank-1 competitor, ci fixed)
        delta_kappa_j1_list = []
        delta_logit_j1_list = []
        for delta in DELTA_OUT[1:]:  # skip delta=0
            X_mod = apply_competitor_shift(X, y, centroids, ci, j1, delta)
            centroids_mod, sigma_W_mod = compute_class_stats(X_mod, y)
            kappas_mod = compute_all_kappas_sorted(centroids_mod, sigma_W_mod, d, ci)
            if not kappas_mod:
                continue
            kappa_j1_mod = kappas_mod[0][0]
            q_mod = compute_per_class_q(X_mod, y, ci)
            if q_mod is None:
                continue
            delta_kappa_j1_list.append(kappa_j1_mod - kappa_j1)
            delta_logit_j1_list.append(safe_logit(q_mod) - blq)

        alpha_j1, r_j1, p_j1 = fit_slope(delta_kappa_j1_list, delta_logit_j1_list)
        print(f"  alpha_j1={alpha_j1:.4f}, r={r_j1:.4f}, p={p_j1:.4f}")

        # Measure alpha_j2 (rank-2 competitor, ci fixed)
        delta_kappa_j2_list = []
        delta_logit_j2_list = []
        for delta in DELTA_OUT[1:]:
            X_mod = apply_competitor_shift(X, y, centroids, ci, j2, delta)
            centroids_mod, sigma_W_mod = compute_class_stats(X_mod, y)
            kappas_mod = compute_all_kappas_sorted(centroids_mod, sigma_W_mod, d, ci)
            if not kappas_mod:
                continue
            # Find kappa of j2 in modified embeddings
            j2_kappas = [k for k, c in kappas_mod if c == j2]
            if not j2_kappas:
                continue
            kappa_j2_mod = j2_kappas[0]
            # Check j1 rank unchanged
            j1_kappas = [k for k, c in kappas_mod if c == j1]
            if not j1_kappas:
                continue
            kappa_j1_mod = j1_kappas[0]
            j1_rank_unchanged = abs(kappa_j1_mod - kappa_j1) < 0.01

            q_mod = compute_per_class_q(X_mod, y, ci)
            if q_mod is None:
                continue
            delta_kappa = kappa_j2_mod - kappa_j2
            delta_logit = safe_logit(q_mod) - blq
            delta_kappa_j2_list.append(delta_kappa)
            delta_logit_j2_list.append(delta_logit)

        alpha_j2, r_j2, p_j2 = fit_slope(delta_kappa_j2_list, delta_logit_j2_list)
        print(f"  alpha_j2={alpha_j2:.4f}, r={r_j2:.4f}, p={p_j2:.4f}")

        # Compute w_j2 = alpha_j2 / alpha_j1 (if alpha_j1 > 0)
        if alpha_j1 > 0.01 and alpha_j2 is not None:
            w_j2 = alpha_j2 / alpha_j1
            log_w_j2 = np.log(max(w_j2, 1e-6))
            theory_log_w = -gap / TAU_STAR
            print(f"  w_j2={w_j2:.4f}, log(w)={log_w_j2:.4f}, theory=-gap/tau*={theory_log_w:.4f}")

            alpha_j1_list.append(alpha_j1)
            alpha_j2_list.append(alpha_j2)
            gap_list.append(gap)

            jacobian_results.append({
                "ci": ci,
                "j1": j1, "j2": j2,
                "kappa_j1": float(kappa_j1),
                "kappa_j2": float(kappa_j2),
                "gap": float(gap),
                "alpha_j1": float(alpha_j1),
                "r_j1": float(r_j1),
                "alpha_j2": float(alpha_j2),
                "r_j2": float(r_j2),
                "w_j2": float(w_j2),
                "log_w_j2": float(log_w_j2),
                "theory_log_w": float(theory_log_w),
            })

    # Aggregate Jacobian test
    print("\n" + "=" * 70)
    print("JACOBIAN AGGREGATION")
    print("=" * 70)

    if len(jacobian_results) < 4:
        print("Too few valid Jacobian points")
        result = {"error": "too_few_points", "n": len(jacobian_results)}
        with open(OUT_JSON, "w") as f:
            json.dump(result, f, indent=2)
        return

    log_ws = np.array([r["log_w_j2"] for r in jacobian_results])
    gaps = np.array([r["gap"] for r in jacobian_results])
    theory_log_ws = -gaps / TAU_STAR

    r_pearson, p_pearson = pearsonr(log_ws, theory_log_ws)
    rho_spearman, p_spearman = spearmanr(log_ws, theory_log_ws)

    # Also fit tau from data
    if np.std(gaps) > 1e-8:
        slope_fit, intercept_fit, _, _, _ = linregress(-gaps, log_ws)
        tau_fit = 1.0 / slope_fit if abs(slope_fit) > 1e-8 else None
    else:
        tau_fit = None

    print(f"\nN={len(jacobian_results)} valid classes")
    print(f"Gaps: {gaps.round(4).tolist()}")
    print(f"log(w_j2): {log_ws.round(4).tolist()}")
    print(f"Theory log(w) = -gap/{TAU_STAR}: {theory_log_ws.round(4).tolist()}")
    print(f"\nPearson r(log_w, -gap/tau*) = {r_pearson:.4f}, p={p_pearson:.4f}")
    print(f"Spearman rho = {rho_spearman:.4f}, p={p_spearman:.4f}")
    if tau_fit is not None:
        print(f"Fitted tau* from data = {tau_fit:.4f} (pre-registered: {TAU_STAR})")

    pass_criterion = r_pearson > R_THRESHOLD
    print(f"\nPRE-REGISTERED PASS (r > {R_THRESHOLD}): {'PASS' if pass_criterion else 'FAIL'}")
    print(f"  r = {r_pearson:.4f} (threshold = {R_THRESHOLD})")

    result = {
        "experiment": "jacobian_early_checkpoint",
        "session": 40,
        "checkpoint": STEP_REVISION,
        "tau_star_prereg": TAU_STAR,
        "r_threshold_prereg": R_THRESHOLD,
        "n_valid": len(jacobian_results),
        "r_pearson": float(r_pearson),
        "p_pearson": float(p_pearson),
        "rho_spearman": float(rho_spearman),
        "p_spearman": float(p_spearman),
        "tau_fit_from_data": float(tau_fit) if tau_fit is not None else None,
        "tau_error_pct": float(abs(tau_fit - TAU_STAR) / TAU_STAR * 100) if tau_fit is not None else None,
        "pass_criterion": bool(pass_criterion),
        "per_class": jacobian_results,
    }

    def json_default(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
