#!/usr/bin/env python -u
"""
CROSS-TASK CAUSAL TRANSPORT TEST (Feb 21 2026)
==============================================
Codex-recommended experiment: test whether alpha from the kappa_nearest law
is universal ACROSS TASKS (not just across architectures).

TRAINING CONDITION (already measured):
  pythia-160m / dbpedia (K=14): alpha=1.601, r=0.974, control_r=0.000 (PERFECT)

TRANSPORT CONDITIONS (new, pre-registered):
  pythia-160m  / yahoo_answers_topics (K=10)
  gpt-neo-125m / yahoo_answers_topics (K=10)

PROTOCOL (frozen-embedding do-intervention):
  1. Extract embeddings from last layer (no gradient, no training)
  2. For each delta in [-3, +3], shift nearest centroid pair by delta
  3. Measure kappa_nearest and q at each point
  4. Fit alpha_local to the dose-response curve

PRE-REGISTERED CRITERIA:
  Transport PASS requires for EACH valid transport condition:
    C1: r(delta_kappa, delta_logit_q) > 0.90
    C2: |alpha_local - TRAINING_ALPHA| / TRAINING_ALPHA < 0.20  (20% tolerance)
    C3: farthest pair r < 0.30 (specificity control)
    C4: baseline q < 0.92 (not at ceiling)

  TRANSPORT_ALPHA = 1.601  (from pythia-160m/dbpedia, measured Feb 21)
  LOAO_ALPHA     = 1.549   (from 7-architecture LOAO, measured Feb 21)
  Both serve as reference: TRANSPORT_ALPHA is the tighter, pre-registered value.

WHY YAHOO_ANSWERS (K=10):
  - K=10 -> 45 centroid pairs. Perturbing the farthest pair CANNOT affect
    kappa_nearest (which depends only on the minimum pair). Control is rigorous.
  - 10 semantically distinct topics (politics, science, sports, ...) -> large
    nearest/farthest ratio -> no crowded-pair problem that killed 20newsgroups
  - Expected baseline q: 0.4-0.7 for these small models (not at ceiling)

WHAT TRANSPORT PROVES:
  If alpha from dbpedia (encyclopedic, K=14) matches alpha from yahoo_answers
  (Q&A topics, K=10) for the SAME model, it proves the slope of the
  kappa_nearest law is task-invariant (not just architecture-invariant).
  Combined with LOAO (architecture-invariant), this would establish:
    logit(q) = 1.55 * kappa_nearest + C(task)
  where ONLY the intercept C depends on task, not the slope.
"""

import json
import os
import sys
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
MODELS_LAYERS = {
    "EleutherAI/pythia-160m":  12,   # final layer
    "EleutherAI/gpt-neo-125m": 12,   # final layer
}

TRANSPORT_DATASETS = {
    # Validation condition (same as training -- should reproduce alpha=1.601)
    "dbpedia": {
        "hf_name": "fancyzhx/dbpedia_14",
        "text_col": "content",
        "label_col": "label",
        "K": 14,
        "is_validation": True,   # already measured; tests pipeline consistency
    },
    # Transport condition (new task -- pre-registered to match alpha=1.601)
    "yahoo_answers_topics": {
        "hf_name": "yahoo_answers_topics",
        "text_col": "question_content",
        "label_col": "topic",
        "K": 10,
        "is_validation": False,
    },
}

N_SAMPLE   = 5000
BATCH_SIZE = 64
DELTA_RANGE = np.linspace(-3.0, 3.0, 21)

# PRE-REGISTERED VALUES
TRAINING_ALPHA      = 1.601   # from pythia-160m/dbpedia do-intervention
LOAO_ALPHA          = 1.549   # from 7-arch LOAO
PRE_REG_R           = 0.90
PRE_REG_ALPHA_TOL   = 0.20    # 20% tolerance on alpha
PRE_REG_CONTROL_R   = 0.30
CEILING_Q           = 0.92


# ================================================================
# DATA LOADING
# ================================================================
def get_texts_labels(hf_name, text_col, label_col, n_samples=N_SAMPLE):
    """Load and subsample a HuggingFace dataset."""
    import random
    random.seed(42)

    # Try test split, then train, then validation
    for split in ["test", "train", "validation"]:
        try:
            ds = load_dataset(hf_name, split=split)
            break
        except Exception:
            continue
    else:
        print(f"  Failed to load {hf_name}", flush=True)
        return None, None

    n = min(n_samples, len(ds))
    indices = random.sample(range(len(ds)), n)

    # Handle different ways to access dataset fields
    try:
        texts  = [str(ds[text_col][i])  for i in indices]
        labels = [ds[label_col][i]       for i in indices]
    except TypeError:
        texts  = [str(ds[i][text_col])  for i in indices]
        labels = [ds[i][label_col]       for i in indices]

    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"  Loaded {len(texts)} samples, {len(le.classes_)} classes: {list(le.classes_[:5])}...", flush=True)
    return texts, y


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================
def extract_embeddings(hf_name, texts, layer_idx, batch_size=BATCH_SIZE):
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        hf_name, output_hidden_states=True, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=128, padding=True).to(DEVICE)
            out = model(**enc, output_hidden_states=True)
            hs  = out.hidden_states[layer_idx]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (hs * mask).sum(1) / mask.sum(1)
            e = emb.cpu().float().numpy()
            e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            all_embs.append(e)

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embs)


def load_or_extract(cache_path, v2_cache_path, hf_name, layer_idx, ds_cfg):
    """Load embeddings from cache, or extract fresh."""
    for path in [cache_path, v2_cache_path]:
        if path and os.path.exists(path):
            data = np.load(path)
            X, y = data["X"], data["y"]
            print(f"  Loaded cache: {X.shape} from {path}", flush=True)
            return X, y

    texts, y = get_texts_labels(ds_cfg["hf_name"], ds_cfg["text_col"], ds_cfg["label_col"])
    if texts is None:
        return None, None

    print(f"  Extracting embeddings (layer {layer_idx})...", flush=True)
    t0 = time.time()
    X  = extract_embeddings(hf_name, texts, layer_idx)
    print(f"  Done in {time.time()-t0:.0f}s. Shape: {X.shape}", flush=True)
    np.savez(cache_path, X=X, y=y)
    return X, y


def clean_embeddings(X, y, label=""):
    """Remove NaN, inf, and zero-vector rows."""
    finite_mask = np.all(np.isfinite(X), axis=1)
    X, y = X[finite_mask], y[finite_mask]
    norms = np.linalg.norm(X, axis=1)
    valid_mask = norms > 1e-3
    n_removed = len(X) - valid_mask.sum()
    if n_removed > 0:
        print(f"  {label}: Removed {n_removed} zero/NaN rows", flush=True)
    return X[valid_mask], y[valid_mask]


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
            if dist < min_dist:
                min_dist = dist; nearest_pair  = (ci, cj)
            if dist > max_dist:
                max_dist = dist; farthest_pair = (ci, cj)
    kappa = float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))
    return kappa, nearest_pair, farthest_pair, min_dist, max_dist


def compute_q(X, y, K):
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
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return float(np.log(q / (1 - q)))


# ================================================================
# DO-INTERVENTION
# ================================================================
def apply_centroid_shift(X, y, centroids, cj, ck, delta):
    mu_j, mu_k = centroids[cj].copy(), centroids[ck].copy()
    if not (np.all(np.isfinite(mu_j)) and np.all(np.isfinite(mu_k))):
        return X.copy()
    diff = mu_k - mu_j
    dist = float(np.linalg.norm(diff))
    if not np.isfinite(dist) or dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] -= (delta / 2) * direction
    X_new[y == ck] += (delta / 2) * direction
    return X_new


def run_dose_response(X, y, K, pair_mode="nearest"):
    d = X.shape[1]
    centroids, sigma_W = compute_class_stats(X, y)
    kappa0, nearest_pair, farthest_pair, min_dist, max_dist = \
        compute_kappa_nearest(centroids, sigma_W, d)

    target = nearest_pair if pair_mode == "nearest" else farthest_pair
    print(f"\n  [{pair_mode}] target pair: {target}", flush=True)
    print(f"  kappa_baseline = {kappa0:.4f}  margin_ratio = {max_dist/min_dist:.2f}x", flush=True)

    points, delta_kappas = [], []
    for delta in DELTA_RANGE:
        X_new = apply_centroid_shift(X, y, centroids, target[0], target[1], delta)
        c_new, sw_new = compute_class_stats(X_new, y)
        kappa_new, _, _, _, _ = compute_kappa_nearest(c_new, sw_new, d)
        q = compute_q(X_new, y, K)
        if q is None:
            continue
        dk = kappa_new - kappa0
        delta_kappas.append(dk)
        points.append({
            "delta": float(delta),
            "kappa_nearest": float(kappa_new),
            "delta_kappa":   float(dk),
            "q":             float(q),
            "logit_q":       logit_q(q),
        })
        print(f"    delta={delta:+.2f}  kappa={kappa_new:.4f} ({dk:+.5f})  q={q:.4f}", flush=True)

    if pair_mode == "farthest":
        max_dk = max(abs(dk) for dk in delta_kappas) if delta_kappas else 0.0
        print(f"  [{'PASS' if max_dk < 0.01 else 'FAIL'}] Control: max|delta_kappa| = {max_dk:.6f} < 0.01", flush=True)

    return points, float(kappa0), (max_dist / min_dist)


def fit_dose_response(points, reference_alpha, cond_label):
    """Fit alpha from dose-response curve and test against reference."""
    if len(points) < 4:
        return {}

    kappas = np.array([p["kappa_nearest"] for p in points])
    logits = np.array([p["logit_q"]       for p in points])

    # Correlation
    dk = kappas - kappas.mean()
    dl = logits - logits.mean()
    r  = float(np.corrcoef(dk, dl)[0, 1]) if np.std(dk) > 1e-6 else float("nan")

    # Linear fit: logit(q) = alpha * kappa + C
    A = np.vstack([kappas, np.ones(len(kappas))]).T
    (alpha_hat, C_hat), _, _, _ = np.linalg.lstsq(A, logits, rcond=None)
    alpha_hat, C_hat = float(alpha_hat), float(C_hat)

    ss_res = float(np.sum((logits - (alpha_hat * kappas + C_hat))**2))
    ss_tot = float(np.sum((logits - logits.mean())**2))
    r2     = float(1 - ss_res / (ss_tot + 1e-10))

    deviation = abs(alpha_hat - reference_alpha) / (abs(reference_alpha) + 1e-10)

    print(f"\n  [{cond_label}] Dose-response fit:", flush=True)
    print(f"    alpha_local     = {alpha_hat:.4f}", flush=True)
    print(f"    reference alpha = {reference_alpha:.4f}", flush=True)
    print(f"    deviation       = {deviation:.1%}", flush=True)
    print(f"    r               = {r:.4f}", flush=True)
    print(f"    R2              = {r2:.4f}", flush=True)

    c1 = not np.isnan(r) and r > PRE_REG_R
    c2 = deviation < PRE_REG_ALPHA_TOL
    print(f"    [{'PASS' if c1 else 'FAIL'}] r > {PRE_REG_R}", flush=True)
    print(f"    [{'PASS' if c2 else 'FAIL'}] deviation < {PRE_REG_ALPHA_TOL:.0%}", flush=True)

    return {
        "alpha_intervention": alpha_hat,
        "C": C_hat,
        "r": r if not np.isnan(r) else None,
        "r2": r2,
        "deviation_from_reference": deviation,
        "reference_alpha": reference_alpha,
        "n_points": len(points),
    }


# ================================================================
# CROSS-TASK TRANSPORT PREDICTION
# ================================================================
def transport_prediction_test(points_transport, training_alpha, cond_label):
    """
    Key transport test: use training_alpha (from dbpedia) to PREDICT the
    logit(q) values for the transport condition, by fitting ONLY the intercept
    (C_task is allowed to vary). Tests if slope transports across tasks.

    Method:
      1. Use alpha_fixed = training_alpha (NO re-fitting of slope)
      2. Fit only intercept C via: C = mean(logit_q) - alpha_fixed * mean(kappa)
      3. Predicted: logit_q_pred = alpha_fixed * kappa + C
      4. Test: R2(predicted, actual) and slope of actual vs predicted
    """
    if len(points_transport) < 4:
        return {}

    kappas = np.array([p["kappa_nearest"] for p in points_transport])
    logits = np.array([p["logit_q"]       for p in points_transport])

    # Fit intercept only (slope = training_alpha)
    C_fixed = float(np.mean(logits) - training_alpha * np.mean(kappas))
    logits_pred = training_alpha * kappas + C_fixed

    ss_res = float(np.sum((logits - logits_pred)**2))
    ss_tot = float(np.sum((logits - logits.mean())**2))
    r2_transport = float(1 - ss_res / (ss_tot + 1e-10))

    # How well does fixed-slope model fit vs free slope?
    A = np.vstack([kappas, np.ones(len(kappas))]).T
    (alpha_free, C_free), _, _, _ = np.linalg.lstsq(A, logits, rcond=None)
    ss_res_free = float(np.sum((logits - (float(alpha_free) * kappas + float(C_free)))**2))
    r2_free = float(1 - ss_res_free / (ss_tot + 1e-10))

    # RMSE of residuals from transported model
    rmse = float(np.sqrt(np.mean((logits - logits_pred)**2)))

    print(f"\n  [{cond_label}] TRANSPORT PREDICTION (alpha fixed = {training_alpha:.4f}):", flush=True)
    print(f"    Intercept C_task (fitted)  = {C_fixed:.4f}", flush=True)
    print(f"    R2 (transported model)     = {r2_transport:.4f}", flush=True)
    print(f"    R2 (free-slope model)      = {r2_free:.4f}", flush=True)
    print(f"    RMSE (logit units)         = {rmse:.4f}", flush=True)
    print(f"    Slope overhead = R2_free - R2_fixed = {r2_free - r2_transport:.4f}", flush=True)
    transport_pass = r2_transport > 0.80
    print(f"    [{'PASS' if transport_pass else 'FAIL'}] R2 (transported) > 0.80", flush=True)

    return {
        "training_alpha_used": training_alpha,
        "C_task_fitted": C_fixed,
        "r2_transported": r2_transport,
        "r2_free_slope": r2_free,
        "rmse_logit": rmse,
        "slope_overhead": float(r2_free - r2_transport),
        "transport_pass": transport_pass,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70, flush=True)
    print("CROSS-TASK CAUSAL TRANSPORT TEST", flush=True)
    print("Training alpha from pythia-160m/dbpedia; transport to yahoo_answers", flush=True)
    print("=" * 70, flush=True)
    print(f"TRAINING_ALPHA  = {TRAINING_ALPHA:.4f}  (pythia-160m/dbpedia do-interv)", flush=True)
    print(f"LOAO_ALPHA      = {LOAO_ALPHA:.4f}  (7-arch LOAO)", flush=True)
    print(f"PRE-REG CRITERIA: r > {PRE_REG_R}, deviation < {PRE_REG_ALPHA_TOL:.0%}, "
          f"control_r < {PRE_REG_CONTROL_R}", flush=True)
    print(flush=True)

    all_results = {}
    transport_pass_list = []

    for hf_name, layer_idx in MODELS_LAYERS.items():
        model_key = hf_name.split("/")[-1]

        for ds_name, ds_cfg in TRANSPORT_DATASETS.items():
            K   = ds_cfg["K"]
            key = f"{model_key}_{ds_name}"
            print(f"\n{'='*60}", flush=True)
            print(f"CONDITION: {key}  (K={K})", flush=True)
            print(f"{'='*60}", flush=True)

            # Cache paths (reuse v2 for dbpedia; create new for yahoo)
            cache_v2 = f"results/do_int_embs_{model_key}_{ds_name}.npz"
            cache_new = f"results/do_int_transport_{model_key}_{ds_name}.npz"

            X, y = load_or_extract(cache_new, cache_v2, hf_name, layer_idx, ds_cfg)
            if X is None:
                print(f"  SKIP: could not load {ds_name}", flush=True)
                continue

            X, y = clean_embeddings(X, y, key)
            d = X.shape[1]

            # Baseline
            centroids, sigma_W = compute_class_stats(X, y)
            kappa0, nearest_pair, farthest_pair, min_d, max_d = \
                compute_kappa_nearest(centroids, sigma_W, d)
            q0 = compute_q(X, y, K)
            margin_ratio = max_d / min_d

            print(f"  Baseline: kappa={kappa0:.4f}, q={q0:.4f}", flush=True)
            print(f"  Nearest pair:  {nearest_pair}  (dist={min_d:.3f})", flush=True)
            print(f"  Farthest pair: {farthest_pair} (dist={max_d:.3f})", flush=True)
            print(f"  K={K}, margin ratio: {margin_ratio:.2f}x", flush=True)

            is_ceiling = q0 > CEILING_Q
            if is_ceiling:
                print(f"  [SKIP] Ceiling: q={q0:.3f} > {CEILING_Q}", flush=True)
                all_results[key] = {"baseline": {"kappa": kappa0, "q": q0}, "is_ceiling": True}
                continue

            cond_result = {
                "baseline": {"kappa": kappa0, "q": q0},
                "K": K,
                "margin_ratio": margin_ratio,
                "nearest_pair": list(nearest_pair),
                "farthest_pair": list(farthest_pair),
                "is_ceiling": False,
            }

            # Nearest pair dose-response
            pts_nearest, _, _ = run_dose_response(X, y, K, "nearest")
            analysis_nearest  = fit_dose_response(pts_nearest, TRAINING_ALPHA, f"{key}/nearest")
            transport_test    = transport_prediction_test(pts_nearest, TRAINING_ALPHA, f"{key}/transport")

            # Farthest pair control
            pts_farthest, _, _ = run_dose_response(X, y, K, "farthest")
            # For farthest, reference_alpha doesn't matter (we're testing r < 0.30)
            analysis_farthest  = fit_dose_response(pts_farthest, TRAINING_ALPHA, f"{key}/farthest")

            # Check control specificity
            dk_farthest = [p["delta_kappa"] for p in pts_farthest]
            max_dk_far  = max(abs(dk) for dk in dk_farthest) if dk_farthest else 0.0
            control_geometry_ok = max_dk_far < 0.01
            print(f"\n  Control geometry check: max|delta_kappa(farthest)| = {max_dk_far:.6f} "
                  f"[{'PASS' if control_geometry_ok else 'FAIL'}]", flush=True)

            # OVERALL PASS/FAIL for this condition
            r_nearest = analysis_nearest.get("r") or 0.0
            dev_nearest = analysis_nearest.get("deviation_from_reference", 1.0)
            r_farthest  = analysis_farthest.get("r") or 1.0
            if r_farthest is None:
                r_farthest = 1.0

            c1 = r_nearest > PRE_REG_R
            c2 = dev_nearest < PRE_REG_ALPHA_TOL
            c3 = r_farthest < PRE_REG_CONTROL_R or not np.isfinite(r_farthest)
            cond_pass = c1 and c2 and c3

            print(f"\n  CONDITION {key}: {'PASS' if cond_pass else 'FAIL'}", flush=True)
            print(f"    C1 r={r_nearest:.4f} > {PRE_REG_R}: {'PASS' if c1 else 'FAIL'}", flush=True)
            print(f"    C2 dev={dev_nearest:.1%} < {PRE_REG_ALPHA_TOL:.0%}: {'PASS' if c2 else 'FAIL'}", flush=True)
            print(f"    C3 ctrl_r={r_farthest:.4f} < {PRE_REG_CONTROL_R}: {'PASS' if c3 else 'FAIL'}", flush=True)

            cond_result["nearest"]          = {"points": pts_nearest, "analysis": analysis_nearest}
            cond_result["farthest"]         = {"points": pts_farthest, "analysis": analysis_farthest}
            cond_result["transport_test"]   = transport_test
            cond_result["condition_pass"]   = cond_pass
            cond_result["control_geom_ok"]  = control_geometry_ok

            all_results[key] = cond_result
            transport_pass_list.append(cond_pass)

            # Partial save
            with open("results/cti_cross_task_transport.json", "w") as f:
                json.dump(all_results, f, indent=2, default=lambda x: None)

    # ======================================================
    # SUMMARY
    # ======================================================
    print(f"\n\n{'='*70}", flush=True)
    print("CROSS-TASK TRANSPORT SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    valid = {k: v for k, v in all_results.items() if not v.get("is_ceiling", True)}
    n_pass = sum(1 for v in valid.values() if v.get("condition_pass", False))
    n_total = len(valid)

    print(f"  Conditions tested:  {n_total}", flush=True)
    print(f"  Conditions passing: {n_pass}", flush=True)
    print(f"  Training alpha:     {TRAINING_ALPHA:.4f} (pythia-160m/dbpedia)", flush=True)

    for key, v in valid.items():
        an = v.get("nearest", {}).get("analysis", {})
        tt = v.get("transport_test", {})
        print(f"  {key}: alpha={an.get('alpha_intervention', 'N/A'):.4f}, "
              f"r={an.get('r', 'N/A'):.4f}, "
              f"transport_R2={tt.get('r2_transported', 'N/A'):.4f}, "
              f"{'PASS' if v.get('condition_pass') else 'FAIL'}", flush=True)

    overall = n_pass == n_total and n_total >= 1
    print(f"\n  OVERALL TRANSPORT: {'PASS' if overall else 'FAIL'}", flush=True)
    print(f"  (Pass requires: all conditions pass, n >= 1)", flush=True)

    all_results["summary_transport"] = {
        "training_alpha": TRAINING_ALPHA,
        "loao_alpha": LOAO_ALPHA,
        "n_valid_conditions": n_total,
        "n_pass": n_pass,
        "overall_pass": overall,
    }

    with open("results/cti_cross_task_transport.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None)
    print(f"\nSaved: results/cti_cross_task_transport.json", flush=True)


if __name__ == "__main__":
    main()
