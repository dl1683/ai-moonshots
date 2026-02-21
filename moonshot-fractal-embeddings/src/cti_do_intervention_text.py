#!/usr/bin/env python -u
"""
FROZEN-EMBEDDING DO-INTERVENTION ON TEXT MODELS (Feb 21 2026)
=============================================================
Codex recommendation: strongest causal test is a do-intervention on frozen embeddings.
This tests the law IN ITS HOME DOMAIN (text models where LOAO alpha=1.54 was measured).

DESIGN:
  1. Extract embeddings from pre-trained text models (Pythia-160m, GPT-Neo-125m)
     at best layer (highest kappa_nearest for the dataset)
  2. Apply do-intervention: move nearest class centroid pair apart/together by delta
     while keeping within-class residuals FIXED
  3. Measure q dose-response
  4. Fit alpha_intervention and compare to LOAO alpha = 1.54

KEY ADVANTAGE over triplet arm:
  - NO TRAINING required (no optimizer/gradient-conflict confounds)
  - EXACT control of kappa_nearest (we set it directly)
  - WITHIN the text domain where the law was measured
  - Specific negative control: perturbing FARTHEST pair (not nearest) should have weak effect

PRE-REGISTERED CRITERIA:
  1. Dose-response r(delta_kappa, delta_logit_q) > 0.90 (nearest pair)
  2. alpha_intervention consistent with LOAO: |alpha - 1.54| / 1.54 < 0.30 (30% tolerance)
  3. Control: farthest pair r < 0.30 (specificity)

MODELS:
  - Pythia-160m (12 layers, layer 12 has kappa=0.529 on AGNews)
  - GPT-Neo-125m (12 layers, best layer on AGNews)

DATASETS:
  - AGNews (K=4, 4 classes: World, Sports, Business, Tech) -> simplest case
  - DBpedia (K=14, encyclopedic) -> different task type

Nobel significance:
  - If alpha_intervention ~ 1.54: proves kappa_nearest CAUSES q (not just correlates)
  - This is the STRONGEST possible causal evidence short of external replication
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
    "EleutherAI/pythia-160m":  [12],   # best layer for AGNews
    "EleutherAI/gpt-neo-125m": [12],   # best layer for AGNews
}
DATASETS = {
    "agnews":  {"hf_name": "fancyzhx/ag_news", "text_col": "text",    "label_col": "label",  "K": 4},
    "dbpedia": {"hf_name": "fancyzhx/dbpedia_14", "text_col": "content", "label_col": "label", "K": 14},
}
N_SAMPLE  = 1000  # samples per dataset
BATCH_SIZE = 64

DELTA_RANGE = np.linspace(-0.4, 0.4, 21)  # broader range for text (kappa~0.3-0.7)

# PRE-REGISTERED CRITERIA
LOAO_ALPHA             = 1.549
PRE_REG_R_THRESHOLD    = 0.90
PRE_REG_ALPHA_TOLERANCE = 0.30
PRE_REG_CONTROL_R      = 0.30


# ================================================================
# EMBEDDINGS
# ================================================================
def get_texts_labels(hf_name, text_col, label_col, n_samples=N_SAMPLE):
    """Load dataset texts and labels."""
    try:
        ds = load_dataset(hf_name, split="test")
    except Exception:
        try:
            ds = load_dataset(hf_name, split="train")
        except Exception:
            return None, None

    import random
    random.seed(42)
    n = min(n_samples, len(ds))
    indices = random.sample(range(len(ds)), n)

    texts = [ds[text_col][i] for i in indices]
    raw_labels = [ds[label_col][i] for i in indices]

    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    return texts, labels


def extract_embeddings_layer(hf_name, texts, layer_idx, batch_size=BATCH_SIZE):
    """Extract mean-pooled embeddings from layer layer_idx."""
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
            hs = out.hidden_states[layer_idx]  # (B, seq, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hs * mask).sum(1) / mask.sum(1)
            all_embs.append(emb.cpu().float().numpy())

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embs)


# ================================================================
# GEOMETRY
# ================================================================
def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids, within_vars = {}, []
    for c in classes:
        Xc = X[y == c]
        centroids[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - centroids[c])**2, axis=1)))
    sigma_W = np.sqrt(np.mean(within_vars) / X.shape[1])
    return centroids, float(sigma_W)


def compute_kappa_nearest(centroids, sigma_W, d):
    classes = list(centroids.keys())
    min_dist, nearest_pair = np.inf, (classes[0], classes[1])
    max_dist, farthest_pair = -np.inf, (classes[0], classes[1])
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            dist = np.linalg.norm(centroids[ci] - centroids[cj])
            if dist < min_dist:
                min_dist = dist
                nearest_pair = (ci, cj)
            if dist > max_dist:
                max_dist = dist
                farthest_pair = (ci, cj)
    kappa = float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))
    return kappa, nearest_pair, farthest_pair


def compute_q(X, y, K):
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
    q = np.clip(q, 1e-6, 1-1e-6)
    return float(np.log(q / (1-q)))


# ================================================================
# DO-INTERVENTION
# ================================================================
def apply_centroid_shift(X, y, centroids, cj, ck, delta):
    """
    Translate samples of class cj and ck to move their centroids apart/together by delta.
    delta > 0: push apart; delta < 0: push together
    """
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


def do_intervention_sweep(X, y, K, delta_range, pair_mode="nearest"):
    """Sweep delta and record dose-response."""
    d = X.shape[1]
    centroids, sigma_W = compute_class_stats(X, y)
    kappa_orig, nearest_pair, farthest_pair = compute_kappa_nearest(centroids, sigma_W, d)

    if pair_mode == "nearest":
        target = nearest_pair
    elif pair_mode == "farthest":
        target = farthest_pair
    else:
        raise ValueError(pair_mode)

    results = []
    for delta in delta_range:
        X_new = apply_centroid_shift(X, y, centroids, target[0], target[1], delta)
        new_centroids, new_sigma_W = compute_class_stats(X_new, y)
        new_kappa, _, _ = compute_kappa_nearest(new_centroids, new_sigma_W, d)
        q = compute_q(X_new, y, K)
        if q is None:
            continue
        results.append({
            "delta": float(delta),
            "kappa_nearest": float(new_kappa),
            "delta_kappa": float(new_kappa - kappa_orig),
            "q": float(q),
            "logit_q": logit_q(q),
        })
        print(f"    [{pair_mode}] delta={delta:+.3f}: kappa={new_kappa:.4f} ({new_kappa-kappa_orig:+.4f}), "
              f"q={q:.4f}", flush=True)

    return results, float(kappa_orig)


def analyze_dose_response(results, label):
    """Fit alpha from dose-response and check pre-registered criteria."""
    if len(results) < 4:
        return {}

    kappas  = np.array([r["kappa_nearest"] for r in results])
    logits  = np.array([r["logit_q"] for r in results])
    deltas  = np.array([r["delta"] for r in results])

    # r(delta_kappa, delta_logit_q) using demeaned values
    dk = kappas - kappas.mean()
    dl = logits  - logits.mean()
    r = float(np.corrcoef(dk, dl)[0,1]) if np.std(dk) > 1e-6 else 0.0

    # OLS
    A = np.vstack([kappas, np.ones(len(kappas))]).T
    (alpha_hat, C), _, _, _ = np.linalg.lstsq(A, logits, rcond=None)
    ss_res = np.sum((logits - (alpha_hat * kappas + C))**2)
    ss_tot = np.sum((logits - logits.mean())**2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    deviation = abs(alpha_hat - LOAO_ALPHA) / LOAO_ALPHA

    print(f"\n  {label}:")
    print(f"    alpha_intervention = {alpha_hat:.4f}  (LOAO = {LOAO_ALPHA:.4f})")
    print(f"    deviation from LOAO = {deviation:.1%}")
    print(f"    r(delta_kappa, delta_logit_q) = {r:.4f}")
    print(f"    R2 = {r2:.4f}")

    if "nearest" in label.lower():
        c1 = r > PRE_REG_R_THRESHOLD
        c2 = deviation < PRE_REG_ALPHA_TOLERANCE
        print(f"    [{'PASS' if c1 else 'FAIL'}] r > {PRE_REG_R_THRESHOLD}")
        print(f"    [{'PASS' if c2 else 'FAIL'}] deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}")
    else:
        c3 = r < PRE_REG_CONTROL_R
        print(f"    [{'PASS' if c3 else 'FAIL'}] control r < {PRE_REG_CONTROL_R} (specificity)")

    return {
        "alpha_intervention": float(alpha_hat), "C": float(C),
        "r": float(r), "r2": float(r2),
        "deviation_from_loao": float(deviation), "n_points": len(results),
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70, flush=True)
    print("TEXT FROZEN-EMBEDDING DO-INTERVENTION ON kappa_nearest", flush=True)
    print("=" * 70, flush=True)
    print(f"LOAO alpha = {LOAO_ALPHA:.4f}", flush=True)
    print(f"Pre-registered: r > {PRE_REG_R_THRESHOLD}, deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}", flush=True)
    print(flush=True)

    all_results = {}
    all_nearest_alphas = []
    all_nearest_rs = []
    all_farthest_rs = []

    for hf_name, layers in MODELS_LAYERS.items():
        model_key = hf_name.split("/")[-1]

        for ds_name, ds_cfg in DATASETS.items():
            K = ds_cfg["K"]
            key = f"{model_key}_{ds_name}"

            # Check cache
            emb_cache = f"results/do_int_embs_{model_key}_{ds_name}.npz"
            if os.path.exists(emb_cache):
                data = np.load(emb_cache)
                X, y = data["X"], data["y"]
                print(f"\n{key}: Loaded cached embeddings {X.shape}", flush=True)
            else:
                print(f"\n{key}: Extracting embeddings...", flush=True)
                texts, y = get_texts_labels(
                    ds_cfg["hf_name"], ds_cfg["text_col"], ds_cfg["label_col"]
                )
                if texts is None:
                    print(f"  Failed to load dataset {ds_name}", flush=True)
                    continue

                best_layer = layers[0]  # pre-selected best layer
                t0 = time.time()
                X = extract_embeddings_layer(hf_name, texts, best_layer)
                elapsed = time.time() - t0
                np.savez(emb_cache, X=X, y=y)
                print(f"  Extracted {X.shape} in {elapsed:.0f}s, saved to {emb_cache}", flush=True)

            # Baseline stats
            d = X.shape[1]
            centroids, sigma_W = compute_class_stats(X, y)
            kappa_orig, nearest_pair, farthest_pair = compute_kappa_nearest(centroids, sigma_W, d)
            q_orig = compute_q(X, y, K)
            print(f"  Baseline: kappa={kappa_orig:.4f}, q={q_orig:.4f}", flush=True)
            print(f"  Nearest pair: {nearest_pair}, Farthest pair: {farthest_pair}", flush=True)

            results = {"baseline": {"kappa": kappa_orig, "q": q_orig}}

            for mode in ["nearest", "farthest"]:
                print(f"\n  --- {mode.upper()} PAIR ---", flush=True)
                sweep, _ = do_intervention_sweep(X, y, K, DELTA_RANGE, pair_mode=mode)
                analysis = analyze_dose_response(sweep, f"{key} - {mode}")
                results[mode] = {"sweep": sweep, "analysis": analysis}

                if mode == "nearest":
                    all_nearest_alphas.append(analysis.get("alpha_intervention", float("nan")))
                    all_nearest_rs.append(analysis.get("r", float("nan")))
                else:
                    all_farthest_rs.append(analysis.get("r", float("nan")))

            all_results[key] = results

            # Save partial
            with open("results/cti_do_intervention_text.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Summary
    print("\n\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    if all_nearest_alphas:
        ma = float(np.nanmean(all_nearest_alphas))
        sa = float(np.nanstd(all_nearest_alphas))
        mr = float(np.nanmean(all_nearest_rs))
        fcr = float(np.nanmean(all_farthest_rs)) if all_farthest_rs else float("nan")
        dev = abs(ma - LOAO_ALPHA) / LOAO_ALPHA

        print(f"  NEAREST: alpha = {ma:.4f} +/- {sa:.4f}")
        print(f"    LOAO alpha = {LOAO_ALPHA:.4f}, deviation = {dev:.1%}")
        print(f"    mean r = {mr:.4f}")
        print(f"  FARTHEST (control): mean r = {fcr:.4f}")

        c1 = mr > PRE_REG_R_THRESHOLD
        c2 = dev < PRE_REG_ALPHA_TOLERANCE
        c3 = (fcr < PRE_REG_CONTROL_R) if not np.isnan(fcr) else None
        passed = c1 and c2 and (c3 if c3 is not None else True)
        print(f"\n  OVERALL: {'PASS' if passed else 'FAIL'}")
        print(f"    [{' PASS' if c1 else ' FAIL'}] r > {PRE_REG_R_THRESHOLD}: {mr:.4f}")
        print(f"    [{' PASS' if c2 else ' FAIL'}] deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}: {dev:.1%}")
        if c3 is not None:
            print(f"    [{' PASS' if c3 else ' FAIL'}] control r < {PRE_REG_CONTROL_R}: {fcr:.4f}")

        all_results["summary"] = {
            "loao_alpha": LOAO_ALPHA,
            "mean_nearest_alpha": ma, "std_nearest_alpha": sa,
            "mean_nearest_r": mr, "mean_farthest_r": fcr,
            "overall_pass": bool(passed),
        }

    with open("results/cti_do_intervention_text.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/cti_do_intervention_text.json", flush=True)


if __name__ == "__main__":
    main()
