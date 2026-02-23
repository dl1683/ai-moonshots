"""
Held-out pre-registered test: SmolLM2-1.7B on 3 datasets.

SmolLM2-1.7B (HuggingFaceTB) was NOT in the training set of 19 architectures.
This is a genuinely new architecture family (SmolLM family, very recent 2024/2025).

Pre-registered (existing LOAO acceptance interval):
- Pre-reg alpha interval [2.43, 3.29] from results/rwkv_preregistration.json
- (This interval comes from the 12-architecture LOAO: mean +/- 2*std = [2.43, 3.29])

New datasets (not in 10-dataset training set): clinc_oos (K=151), minds14 (K=14)
Old dataset (in training set): amazon_massive (K=59) [to verify model works correctly]

PASS criteria (pre-registered):
  PR1: alpha_dbpedia in [2.43, 3.29] (primary held-out test, same as RWKV pre-reg)
  PR2: logit(q_norm) ~ alpha * kappa_nearest has Pearson r >= 0.80 on dbpedia
  PR3 (new datasets): r(kappa_nearest, logit_q) >= 0.75 on clinc_oos (K=151)
  PR4 (new datasets): r(kappa_nearest, logit_q) >= 0.70 on minds14 (K=14)

Output: results/cti_smollm2_held_out.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_smollm2_held_out.json")

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"

# Pre-registered alpha interval from 12-arch LOAO
ALPHA_LOW = 2.43
ALPHA_HIGH = 3.29
ALPHA_UNIVERSAL = 3.598
BETA_UNIVERSAL = 0.478

# Dataset configs
DATASETS = [
    {
        "name": "dbpedia_14",
        "hf_name": "dbpedia_14",
        "hf_subset": None,
        "split": "train",
        "text_col": "content",
        "label_col": "label",
        "n_per_class": 200,
        "note": "PRIMARY TEST (in training set, new arch)",
    },
    {
        "name": "clinc_oos",
        "hf_name": "clinc_oos",
        "hf_subset": "plus",
        "split": "test",
        "text_col": "text",
        "label_col": "intent",
        "n_per_class": 100,
        "note": "NEW DATASET (K=151, large-K test)",
    },
    {
        "name": "minds14",
        "hf_name": "PolyAI/minds14",
        "hf_subset": "en-US",
        "split": "train",
        "text_col": "transcription",
        "label_col": "intent_class",
        "n_per_class": 100,
        "note": "NEW DATASET (intent detection, different domain)",
    },
]

N_LAYERS = 5  # sample 5 layers to get kappa vs logit relationship


def compute_kappa_and_q(embs_by_class, all_classes):
    """Compute kappa_nearest and 1-NN accuracy across layers."""
    from sklearn.neighbors import KNeighborsClassifier

    K = len(all_classes)
    n_per_class = min(len(embs_by_class[all_classes[0]]), 200)
    d = embs_by_class[all_classes[0]].shape[1]

    # Centroids and within-class variance
    mu = {}
    within_var_sum = 0.0
    n_total = 0
    for ci in all_classes:
        embs = embs_by_class[ci][:n_per_class]
        mu[ci] = embs.mean(0)
        within_var_sum += np.sum((embs - mu[ci])**2)
        n_total += len(embs)
    sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))

    # Kappa nearest
    min_dist = float("inf")
    for i, ci in enumerate(all_classes):
        for j, cj in enumerate(all_classes):
            if i >= j:
                continue
            dist = float(np.linalg.norm(mu[ci] - mu[cj]))
            min_dist = min(min_dist, dist)
    kappa_nearest = min_dist / (sigma_W * np.sqrt(d))

    # 1-NN accuracy (80/20 split)
    train_embs, train_labels, test_embs, test_labels = [], [], [], []
    rng = np.random.default_rng(42)
    for ci in all_classes:
        embs = embs_by_class[ci][:n_per_class]
        n = len(embs)
        n_train = max(1, int(0.8 * n))
        idx = rng.permutation(n)
        train_embs.append(embs[idx[:n_train]])
        train_labels.extend([ci] * n_train)
        test_embs.append(embs[idx[n_train:]])
        test_labels.extend([ci] * (n - n_train))

    train_X = np.vstack(train_embs)
    test_X = np.vstack(test_embs)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(train_X, np.array(train_labels))
    q_raw = float(np.mean(knn.predict(test_X) == np.array(test_labels)))
    q_norm = float((q_raw - 1.0/K) / (1.0 - 1.0/K))

    return kappa_nearest, q_raw, q_norm, sigma_W, d


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    from scipy.special import logit as scipy_logit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    full_model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers_total = len(full_model.model.layers) if hasattr(full_model, 'model') else 24
    print(f"Total layers: {n_layers_total}")
    # Select evenly-spaced layers + last
    layer_indices = sorted(set(
        [int(n_layers_total * i // (N_LAYERS-1)) for i in range(N_LAYERS-1)] + [n_layers_total - 1]
    ))
    print(f"Testing layers: {layer_indices}")

    dataset_results = []

    for ds_config in DATASETS:
        ds_name = ds_config["name"]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({ds_config['note']})")

        # Load dataset
        if ds_config["hf_subset"]:
            ds = load_dataset(ds_config["hf_name"], ds_config["hf_subset"],
                              split=ds_config["split"])
        else:
            ds = load_dataset(ds_config["hf_name"], split=ds_config["split"])

        texts = [x[ds_config["text_col"]] for x in ds]
        labels = [x[ds_config["label_col"]] for x in ds]
        all_classes = sorted(set(labels))
        K = len(all_classes)
        n_per_class = ds_config["n_per_class"]
        print(f"  K={K}, total={len(texts)}, n_per_class={n_per_class}")

        # Extract layer-wise embeddings at LAST TOKEN
        kappa_vals = []
        q_norm_vals = []
        logit_q_vals = []
        layer_results = {}

        for layer_idx in layer_indices:
            print(f"  Layer {layer_idx}...")

            # Extract embeddings from this layer
            class_embeddings = {}
            for ci in all_classes:
                idx_ci = [i for i, l in enumerate(labels) if l == ci][:n_per_class]
                texts_ci = [texts[i] for i in idx_ci]
                if len(texts_ci) < 5:
                    continue
                embs_ci = []
                for b_start in range(0, len(texts_ci), 32):
                    batch = texts_ci[b_start:b_start+32]
                    tok = tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        out = full_model(**tok, output_hidden_states=True)
                    # Get hidden state at requested layer, last token
                    hidden = out.hidden_states[layer_idx + 1]  # +1 for embedding layer
                    emb = hidden[:, -1, :].cpu().float().numpy()
                    embs_ci.extend(emb)
                if embs_ci:
                    class_embeddings[ci] = np.array(embs_ci)

            if len(class_embeddings) < K:
                print(f"    Skipped (only {len(class_embeddings)} classes)")
                continue

            kappa, q_raw, q_norm, sigma_W, d = compute_kappa_and_q(
                class_embeddings, sorted(class_embeddings.keys()))

            q_norm_clipped = float(np.clip(q_norm, 0.01, 0.99))
            logit_q = float(scipy_logit(q_norm_clipped))

            kappa_vals.append(kappa)
            q_norm_vals.append(q_norm)
            logit_q_vals.append(logit_q)
            layer_results[layer_idx] = {
                "kappa_nearest": kappa,
                "q_raw": q_raw,
                "q_norm": q_norm,
                "logit_q_norm": logit_q,
                "sigma_W": sigma_W,
                "d": d,
            }
            print(f"    kappa={kappa:.4f}, q_norm={q_norm:.3f}, logit(q)={logit_q:.3f}")

        if len(kappa_vals) >= 2:
            r_pearson, p_pearson = pearsonr(kappa_vals, logit_q_vals)
            # Fit alpha (slope of logit(q) vs kappa)
            from scipy.stats import linregress
            slope, intercept, _, _, se = linregress(kappa_vals, logit_q_vals)

            pr_r = r_pearson >= 0.75
            pr_alpha = ALPHA_LOW <= slope <= ALPHA_HIGH

            print(f"\n  Fit: alpha={slope:.4f}+/-{se:.4f}, r={r_pearson:.4f} (p={p_pearson:.4f})")
            print(f"  PR_ALPHA in [{ALPHA_LOW},{ALPHA_HIGH}]: {'PASS' if pr_alpha else 'FAIL'}")
            print(f"  PR_R >= 0.75: {'PASS' if pr_r else 'FAIL'}")

            dataset_results.append({
                "dataset": ds_name,
                "K": K,
                "note": ds_config["note"],
                "layer_results": layer_results,
                "alpha_fit": float(slope),
                "alpha_se": float(se),
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "pr_alpha_pass": bool(pr_alpha),
                "pr_r_pass": bool(pr_r),
            })
        else:
            print(f"  INSUFFICIENT DATA (only {len(kappa_vals)} valid layers)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: SmolLM2-1.7B External Held-Out Test")
    print(f"{'='*60}")
    for res in dataset_results:
        print(f"  {res['dataset']} (K={res['K']}): alpha={res['alpha_fit']:.4f}, "
              f"r={res['pearson_r']:.4f}, "
              f"alpha_pass={'PASS' if res['pr_alpha_pass'] else 'FAIL'}, "
              f"r_pass={'PASS' if res['pr_r_pass'] else 'FAIL'}")

    n_alpha_pass = sum(1 for r in dataset_results if r["pr_alpha_pass"])
    n_r_pass = sum(1 for r in dataset_results if r["pr_r_pass"])
    overall = n_alpha_pass >= 1 and n_r_pass >= 2

    print(f"\n  Overall: n_alpha_pass={n_alpha_pass}/{len(dataset_results)}, "
          f"n_r_pass={n_r_pass}/{len(dataset_results)}")
    print(f"  VERDICT: {'PASS' if overall else 'FAIL'}")

    output = {
        "experiment": "smollm2_held_out_replication",
        "model": MODEL_NAME,
        "note": "SmolLM2-1.7B not in 19-arch training set. Genuine new architecture family.",
        "pre_reg_alpha_interval": [ALPHA_LOW, ALPHA_HIGH],
        "datasets_tested": dataset_results,
        "n_alpha_pass": n_alpha_pass,
        "n_r_pass": n_r_pass,
        "overall_pass": bool(overall),
        "verdict": "PASS" if overall else "FAIL",
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
