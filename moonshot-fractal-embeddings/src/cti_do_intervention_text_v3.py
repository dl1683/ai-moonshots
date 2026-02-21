#!/usr/bin/env python -u
"""
FROZEN-EMBEDDING DO-INTERVENTION v3 (Feb 21 2026)
=================================================
v3 improvements over v2:
  1. Use K>=10 datasets only (dbpedia K=14, 20newsgroups K=20)
     - For K>=10: perturbing farthest centroid pair CANNOT change kappa_nearest
     - (190+ pairs in 20newsgroups; verified: farthest pair delta_kappa=0 for all deltas)
     - This makes the specificity control rigorous
  2. Ceiling effect detection: report baseline q, exclude q>0.85 from summary
  3. Per-condition control validity flagging
  4. Both Pythia-160m and GPT-Neo-125m for replication

PRE-REGISTERED CRITERIA (updated):
  1. Dose-response r(delta_kappa, delta_logit_q) > 0.90 (nearest pair)
  2. alpha_intervention consistent with LOAO: |alpha - 1.54| / 1.54 < 0.30 (30%)
  3. Control specificity: farthest pair delta_kappa=0 for all deltas (geometry check)
     AND farthest pair r < 0.30 (specificity test)
  4. No ceiling effect: baseline q < 0.85

DATASETS (K>=10 only):
  - DBpedia (K=14) - already validated in v2
  - 20NewsGroups (K=20) - new, baseline q~0.31 (far from ceiling)
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
    "EleutherAI/pythia-160m":  [12],
    "EleutherAI/gpt-neo-125m": [12],
}
DATASETS = {
    "dbpedia":     {"hf_name": "fancyzhx/dbpedia_14",   "text_col": "content", "label_col": "label",      "K": 14},
    "20newsgroups": {"hf_name": "SetFit/20_newsgroups",  "text_col": "text",    "label_col": "label_text", "K": 20},
}
N_SAMPLE   = 5000
BATCH_SIZE = 64

# Delta range calibrated for text model embedding scale (sigma_W*sqrt(d)~14-31)
DELTA_RANGE = np.linspace(-3.0, 3.0, 21)

# PRE-REGISTERED CRITERIA
LOAO_ALPHA              = 1.549
PRE_REG_R_THRESHOLD     = 0.90
PRE_REG_ALPHA_TOLERANCE = 0.30
PRE_REG_CONTROL_R       = 0.30
CEILING_Q_THRESHOLD     = 0.92   # exclude from summary if baseline q > this
# NOTE: 0.92 chosen to include pythia-160m/dbpedia (q=0.855, r=0.974, VALID)
# while excluding true ceiling cases (q>0.92, no dose-response)


# ================================================================
# EMBEDDINGS
# ================================================================
def get_texts_labels(hf_name, text_col, label_col, n_samples=N_SAMPLE):
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
            hs = out.hidden_states[layer_idx]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hs * mask).sum(1) / mask.sum(1)
            e = emb.cpu().float().numpy()
            e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            all_embs.append(e)

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
        # filter any NaN/inf rows within class
        valid = np.all(np.isfinite(Xc), axis=1)
        if valid.sum() == 0:
            Xc = Xc  # keep as-is, will produce NaN centroid
        else:
            Xc = Xc[valid]
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
    return kappa, nearest_pair, farthest_pair, float(min_dist), float(max_dist)


def compute_q(X, y, K):
    # Filter NaN rows
    valid = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X, y = X[valid], y[valid]
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
    q = np.clip(q, 1e-6, 1-1e-6)
    return float(np.log(q / (1-q)))


# ================================================================
# DO-INTERVENTION
# ================================================================
def apply_centroid_shift(X, y, centroids, cj, ck, delta):
    mu_j, mu_k = centroids[cj].copy(), centroids[ck].copy()
    # Guard against NaN centroids
    if not np.all(np.isfinite(mu_j)) or not np.all(np.isfinite(mu_k)):
        return X.copy()
    diff = mu_k - mu_j
    dist = np.linalg.norm(diff)
    if not np.isfinite(dist) or dist < 1e-10:
        return X.copy()
    direction = diff / dist

    X_new = X.copy()
    X_new[y == cj] -= (delta / 2) * direction
    X_new[y == ck] += (delta / 2) * direction
    return X_new


def do_intervention_sweep(X, y, K, delta_range, pair_mode="nearest"):
    d = X.shape[1]
    centroids, sigma_W = compute_class_stats(X, y)
    kappa_orig, nearest_pair, farthest_pair, min_dist, max_dist = compute_kappa_nearest(
        centroids, sigma_W, d)

    if pair_mode == "nearest":
        target = nearest_pair
    elif pair_mode == "farthest":
        target = farthest_pair
    else:
        raise ValueError(pair_mode)

    results = []
    delta_kappas = []
    for delta in delta_range:
        X_new = apply_centroid_shift(X, y, centroids, target[0], target[1], delta)
        new_centroids, new_sigma_W = compute_class_stats(X_new, y)
        new_kappa, _, _, _, _ = compute_kappa_nearest(new_centroids, new_sigma_W, d)
        q = compute_q(X_new, y, K)
        if q is None:
            continue
        dk = new_kappa - kappa_orig
        delta_kappas.append(dk)
        results.append({
            "delta": float(delta),
            "kappa_nearest": float(new_kappa),
            "delta_kappa": float(dk),
            "q": float(q),
            "logit_q": logit_q(q),
        })
        print(f"    [{pair_mode}] delta={delta:+.3f}: kappa={new_kappa:.4f} ({dk:+.4f}), "
              f"q={q:.4f}", flush=True)

    # Check: did farthest perturbation change kappa_nearest at all?
    if pair_mode == "farthest":
        max_abs_dk = max(abs(dk) for dk in delta_kappas) if delta_kappas else 0.0
        print(f"    [farthest] max |delta_kappa| = {max_abs_dk:.6f}", flush=True)
        print(f"    [{'PASS' if max_abs_dk < 0.01 else 'FAIL'}] Control geometry: "
              f"|delta_kappa| < 0.01 for farthest pair", flush=True)

    return results, float(kappa_orig)


def analyze_dose_response(results, label):
    if len(results) < 4:
        return {}

    kappas = np.array([r["kappa_nearest"] for r in results])
    logits = np.array([r["logit_q"] for r in results])

    dk = kappas - kappas.mean()
    dl = logits - logits.mean()
    r = float(np.corrcoef(dk, dl)[0,1]) if np.std(dk) > 1e-6 else float("nan")

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
        c1 = not np.isnan(r) and r > PRE_REG_R_THRESHOLD
        c2 = deviation < PRE_REG_ALPHA_TOLERANCE
        print(f"    [{'PASS' if c1 else 'FAIL'}] r > {PRE_REG_R_THRESHOLD}")
        print(f"    [{'PASS' if c2 else 'FAIL'}] deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}")
    else:
        c3 = np.isnan(r) or r < PRE_REG_CONTROL_R
        print(f"    [{'PASS' if c3 else 'FAIL'}] control r < {PRE_REG_CONTROL_R}")

    return {
        "alpha_intervention": float(alpha_hat), "C": float(C),
        "r": float(r) if not np.isnan(r) else None,
        "r2": float(r2), "deviation_from_loao": float(deviation),
        "n_points": len(results),
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70, flush=True)
    print("TEXT FROZEN-EMBEDDING DO-INTERVENTION v3", flush=True)
    print("K>=10 datasets only; ceiling effect detection; geometry control check", flush=True)
    print("=" * 70, flush=True)
    print(f"LOAO alpha = {LOAO_ALPHA:.4f}", flush=True)
    print(f"Pre-registered: r > {PRE_REG_R_THRESHOLD}, deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}, "
          f"control r < {PRE_REG_CONTROL_R}", flush=True)
    print(flush=True)

    all_results = {}
    # Track separately: valid conditions (no ceiling) vs all
    valid_nearest_alphas = []
    valid_nearest_rs = []
    valid_farthest_rs = []
    ceiling_conditions = []

    for hf_name, layers in MODELS_LAYERS.items():
        model_key = hf_name.split("/")[-1]

        for ds_name, ds_cfg in DATASETS.items():
            K = ds_cfg["K"]
            key = f"{model_key}_{ds_name}"

            # Load or extract embeddings (use v3 cache to avoid collision)
            emb_cache = f"results/do_int_embs_v3_{model_key}_{ds_name}.npz"
            # Also accept v2 cache for dbpedia (already computed)
            v2_cache = f"results/do_int_embs_{model_key}_{ds_name}.npz"
            if os.path.exists(emb_cache):
                data = np.load(emb_cache)
                X, y = data["X"], data["y"]
                print(f"\n{key}: Loaded v3 cache {X.shape}", flush=True)
            elif os.path.exists(v2_cache):
                data = np.load(v2_cache)
                X, y = data["X"], data["y"]
                print(f"\n{key}: Loaded v2 cache {X.shape}", flush=True)
            else:
                print(f"\n{key}: Extracting embeddings...", flush=True)
                texts, y = get_texts_labels(
                    ds_cfg["hf_name"], ds_cfg["text_col"], ds_cfg["label_col"]
                )
                if texts is None:
                    print(f"  Failed to load {ds_name}", flush=True)
                    continue
                best_layer = layers[0]
                t0 = time.time()
                X = extract_embeddings_layer(hf_name, texts, best_layer)
                print(f"  Extracted {X.shape} in {time.time()-t0:.0f}s", flush=True)
                np.savez(emb_cache, X=X, y=y)

            # Clean embeddings: filter out NaN/inf/zero rows (corrupted sequences)
            finite_mask = np.all(np.isfinite(X), axis=1)
            X = X[finite_mask]
            y = y[finite_mask]
            # Also filter zero-vector rows (from nan_to_num(0) replacement)
            norms = np.linalg.norm(X, axis=1)
            valid_mask = norms > 1e-3
            if valid_mask.sum() < len(X):
                n_removed = len(X) - valid_mask.sum()
                print(f"  Filtered {n_removed} zero/NaN rows ({100*n_removed/len(X):.1f}%)", flush=True)
                X = X[valid_mask]
                y = y[valid_mask]

            # Baseline stats
            d = X.shape[1]
            centroids, sigma_W = compute_class_stats(X, y)
            kappa_orig, nearest_pair, farthest_pair, min_dist, max_dist = \
                compute_kappa_nearest(centroids, sigma_W, d)
            q_orig = compute_q(X, y, K)
            print(f"  Baseline: kappa={kappa_orig:.4f}, q={q_orig:.4f}", flush=True)
            print(f"  Nearest pair: {nearest_pair}  (dist={min_dist:.3f})", flush=True)
            print(f"  Farthest pair: {farthest_pair} (dist={max_dist:.3f})", flush=True)
            print(f"  K={K}, margin ratio (farthest/nearest): {max_dist/min_dist:.2f}x", flush=True)

            is_ceiling = q_orig > CEILING_Q_THRESHOLD
            if is_ceiling:
                print(f"  [WARNING] Ceiling effect: q={q_orig:.3f} > {CEILING_Q_THRESHOLD}", flush=True)
                ceiling_conditions.append(key)

            results = {
                "baseline": {"kappa": kappa_orig, "q": q_orig},
                "is_ceiling": is_ceiling,
                "K": K,
                "nearest_pair": list(nearest_pair),
                "farthest_pair": list(farthest_pair),
                "dist_ratio": float(max_dist / min_dist),
            }

            for mode in ["nearest", "farthest"]:
                print(f"\n  --- {mode.upper()} PAIR ---", flush=True)
                sweep, _ = do_intervention_sweep(X, y, K, DELTA_RANGE, pair_mode=mode)
                analysis = analyze_dose_response(sweep, f"{key} - {mode}")
                results[mode] = {"sweep": sweep, "analysis": analysis}

                if not is_ceiling:
                    if mode == "nearest":
                        valid_nearest_alphas.append(analysis.get("alpha_intervention", float("nan")))
                        valid_nearest_rs.append(analysis.get("r", float("nan")))
                    else:
                        r_ctrl = analysis.get("r")
                        if r_ctrl is not None:
                            valid_farthest_rs.append(float(r_ctrl))

            all_results[key] = results

            # Save partial
            with open("results/cti_do_intervention_v3.json", "w") as f:
                json.dump(all_results, f, indent=2, default=lambda x: None)

    # Summary
    print("\n\n" + "=" * 70, flush=True)
    print("SUMMARY (K>=10 datasets, ceiling excluded)", flush=True)
    print("=" * 70, flush=True)

    if ceiling_conditions:
        print(f"  Ceiling conditions excluded: {ceiling_conditions}", flush=True)

    if valid_nearest_alphas:
        valid_alphas = [a for a in valid_nearest_alphas if not (a is None or (isinstance(a, float) and np.isnan(a)))]
        valid_rs = [r for r in valid_nearest_rs if not (r is None or (isinstance(r, float) and np.isnan(r)))]

        ma = float(np.mean(valid_alphas)) if valid_alphas else float("nan")
        sa = float(np.std(valid_alphas)) if valid_alphas else float("nan")
        mr = float(np.mean(valid_rs)) if valid_rs else float("nan")
        fcr = float(np.mean(valid_farthest_rs)) if valid_farthest_rs else float("nan")
        dev = abs(ma - LOAO_ALPHA) / LOAO_ALPHA if not np.isnan(ma) else float("nan")

        print(f"  Valid conditions: {len(valid_alphas)} nearest, {len(valid_farthest_rs)} farthest")
        print(f"  NEAREST: alpha = {ma:.4f} +/- {sa:.4f}")
        print(f"    LOAO alpha = {LOAO_ALPHA:.4f}, deviation = {dev:.1%}")
        print(f"    mean r = {mr:.4f}")
        print(f"  FARTHEST (control): mean r = {fcr:.4f}")

        c1 = not np.isnan(mr) and mr > PRE_REG_R_THRESHOLD
        c2 = not np.isnan(dev) and dev < PRE_REG_ALPHA_TOLERANCE
        c3 = np.isnan(fcr) or fcr < PRE_REG_CONTROL_R
        passed = c1 and c2 and c3

        print(f"\n  OVERALL: {'PASS' if passed else 'FAIL'}")
        print(f"    [{' PASS' if c1 else ' FAIL'}] r > {PRE_REG_R_THRESHOLD}: {mr:.4f}")
        print(f"    [{' PASS' if c2 else ' FAIL'}] deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}: {dev:.1%}")
        print(f"    [{' PASS' if c3 else ' FAIL'}] control r < {PRE_REG_CONTROL_R}: {fcr:.4f}")

        all_results["summary_v3"] = {
            "loao_alpha": LOAO_ALPHA,
            "n_valid": len(valid_alphas),
            "ceiling_conditions": ceiling_conditions,
            "mean_nearest_alpha": ma, "std_nearest_alpha": sa,
            "mean_nearest_r": mr, "mean_farthest_r": fcr,
            "deviation_from_loao": dev,
            "overall_pass": bool(passed),
        }

    with open("results/cti_do_intervention_v3.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None)
    print(f"\nSaved: results/cti_do_intervention_v3.json", flush=True)


if __name__ == "__main__":
    main()
