"""
SmolLM2-1.7B fast cross-dataset replication.

DESIGN: 5 fixed layers (0, 6, 12, 18, 23), pick best layer per dataset.
Uses pre-computed data for dbpedia_14 and clinc_oos from the original run
(results already logged in cti_smollm2_held_out.json).
Runs 3 NEW datasets: banking77 (K=77), ag_news (K=4), 20newsgroups (K=20).

Pre-registered constants (frozen from pre-reg):
  alpha_low=2.43, alpha_high=3.29

Overall pass: alpha_fit in [2.43, 3.29] AND pearson_r >= 0.80.

Output: results/cti_smollm2_fast_replication.json
"""

import json
import os
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.special import logit as scipy_logit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_smollm2_fast_replication.json")

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
ALPHA_LOW = 2.43
ALPHA_HIGH = 3.29

# Pre-computed from cti_smollm2_held_out.json (best layer per dataset, layers 0,6,12,18,23)
# dbpedia: best layer=18 (q_norm=0.681)
# clinc_oos: best layer=0 (q_norm=0.560)
PRECOMPUTED = [
    {
        "dataset": "dbpedia_14",
        "K": 14,
        "best_layer": 18,
        "kappa_nearest": 0.0307,
        "q_raw": None,
        "q_norm": 0.681,
        "logit_q_norm": 0.757,
        "note": "In training set (K=14) -- from original smollm2_held_out run",
    },
    {
        "dataset": "clinc_oos",
        "K": 151,
        "best_layer": 0,
        "kappa_nearest": 0.1216,
        "q_raw": None,
        "q_norm": 0.560,
        "logit_q_norm": 0.241,
        "note": "OUT-OF-DISTRIBUTION (K=151, new domain) -- from original run",
    },
]

# New datasets to run
NEW_DATASETS = [
    {
        "name": "banking77",
        "hf_name": "banking77",
        "hf_subset": None,
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "n_per_class": 200,
        "note": "In training set (K=77, large-K)",
    },
    {
        "name": "ag_news",
        "hf_name": "ag_news",
        "hf_subset": None,
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "n_per_class": 200,
        "note": "In training set (K=4, small-K)",
    },
    {
        "name": "20newsgroups",
        "hf_name": "SetFit/20_newsgroups",
        "hf_subset": None,
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "n_per_class": 200,
        "note": "In training set (K=20)",
    },
]

# Fixed 5 layers (same as original run)
LAYER_INDICES = [0, 6, 12, 18, 23]
N_LAYERS_TOTAL = 24


def compute_kappa_and_q(embs_by_class, all_classes):
    from sklearn.neighbors import KNeighborsClassifier
    K = len(all_classes)
    n_per_class = min(len(embs_by_class[all_classes[0]]), 200)
    d = embs_by_class[all_classes[0]].shape[1]

    mu = {}
    within_var_sum = 0.0
    n_total = 0
    for ci in all_classes:
        embs = embs_by_class[ci][:n_per_class]
        mu[ci] = embs.mean(0)
        within_var_sum += np.sum((embs - mu[ci]) ** 2)
        n_total += len(embs)
    sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))

    min_dist = float("inf")
    for i, ci in enumerate(all_classes):
        for j, cj in enumerate(all_classes):
            if i >= j:
                continue
            dist = float(np.linalg.norm(mu[ci] - mu[cj]))
            min_dist = min(min_dist, dist)
    kappa_nearest = min_dist / (sigma_W * np.sqrt(d))

    rng = np.random.default_rng(42)
    train_embs, train_labels, test_embs, test_labels = [], [], [], []
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
    q_norm = float((q_raw - 1.0 / K) / (1.0 - 1.0 / K))
    return kappa_nearest, q_raw, q_norm, sigma_W, d


def process_dataset(model, tokenizer, device, ds_config):
    import torch
    from datasets import load_dataset

    print(f"  Loading {ds_config['name']}...")
    try:
        if ds_config["hf_subset"]:
            ds = load_dataset(ds_config["hf_name"], ds_config["hf_subset"],
                              split=ds_config["split"], trust_remote_code=False)
        else:
            ds = load_dataset(ds_config["hf_name"], split=ds_config["split"],
                              trust_remote_code=False)
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        return None

    texts = [x[ds_config["text_col"]] for x in ds]
    labels = [x[ds_config["label_col"]] for x in ds]
    all_classes = sorted(set(labels))
    K = len(all_classes)
    n_per_class = ds_config["n_per_class"]
    print(f"  K={K}, n_per_class={n_per_class}")

    best_layer = -1
    best_q_raw = -1.0
    best_kappa = None
    best_q_norm = None
    best_logit_q = None

    for layer_idx in LAYER_INDICES:
        print(f"  Layer {layer_idx}...")
        class_embeddings = {}
        for ci in all_classes:
            idx_ci = [i for i, l in enumerate(labels) if l == ci][:n_per_class]
            texts_ci = [texts[i] for i in idx_ci]
            if len(texts_ci) < 5:
                continue
            embs_ci = []
            for b_start in range(0, len(texts_ci), 32):
                batch = texts_ci[b_start:b_start + 32]
                tok = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    out = model(**tok, output_hidden_states=True)
                hidden = out.hidden_states[layer_idx + 1]
                emb = hidden[:, -1, :].cpu().float().numpy()
                embs_ci.extend(emb)
            if embs_ci:
                class_embeddings[ci] = np.array(embs_ci)

        if len(class_embeddings) < K:
            continue

        kappa, q_raw, q_norm, sigma_W, d = compute_kappa_and_q(
            class_embeddings, sorted(class_embeddings.keys()))
        q_norm_clipped = float(np.clip(q_norm, 0.01, 0.99))
        logit_q = float(scipy_logit(q_norm_clipped))
        print(f"    kappa={kappa:.4f}, q_norm={q_norm:.3f}, logit(q)={logit_q:.3f}")

        if q_raw > best_q_raw:
            best_q_raw = q_raw
            best_layer = layer_idx
            best_kappa = kappa
            best_q_norm = q_norm
            best_logit_q = logit_q

    return {
        "dataset": ds_config["name"],
        "K": K,
        "note": ds_config["note"],
        "best_layer": best_layer,
        "best_q_raw": best_q_raw,
        "kappa_nearest": best_kappa,
        "q_norm": best_q_norm,
        "logit_q_norm": best_logit_q,
    }


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print("Strategy: 5-layer scan (0,6,12,18,23), pick best layer per dataset")
    print("Pre-computed: dbpedia (best_layer=18) and clinc_oos (best_layer=0) from prior run\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_results = list(PRECOMPUTED)  # start with pre-computed data

    for ds_config in NEW_DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_config['name']} ({ds_config['note']})")
        result = process_dataset(model, tokenizer, device, ds_config)
        if result is not None:
            dataset_results.append(result)
            print(f"  BEST: layer={result['best_layer']}, kappa={result['kappa_nearest']:.4f}, "
                  f"q_norm={result['q_norm']:.3f}, logit={result['logit_q_norm']:.3f}")

    # Fit alpha across datasets
    print(f"\n{'='*60}")
    print(f"CROSS-DATASET FIT FOR SmolLM2-1.7B")
    print(f"{'='*60}")

    kappa_all = [r["kappa_nearest"] for r in dataset_results if r.get("kappa_nearest") is not None]
    logit_q_all = [r["logit_q_norm"] for r in dataset_results if r.get("logit_q_norm") is not None]

    for r in dataset_results:
        print(f"  {r['dataset']} (K={r['K']}): kappa={r.get('kappa_nearest', 'N/A'):.4f}, "
              f"logit={r.get('logit_q_norm', 'N/A'):.3f}")

    if len(kappa_all) >= 3:
        r_pearson, p_pearson = pearsonr(kappa_all, logit_q_all)
        slope, intercept, _, _, se = linregress(kappa_all, logit_q_all)

        pr1 = ALPHA_LOW <= slope <= ALPHA_HIGH
        pr2 = r_pearson >= 0.80
        overall = pr1 and pr2

        print(f"\n  alpha_fit = {slope:.4f} +/- {se:.4f}")
        print(f"  Pearson r = {r_pearson:.4f} (p={p_pearson:.4f})")
        print(f"\n  PR1 (alpha in [{ALPHA_LOW},{ALPHA_HIGH}]): {'PASS' if pr1 else 'FAIL'}")
        print(f"  PR2 (r >= 0.80): {'PASS' if pr2 else 'FAIL'}")
        print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

        output = {
            "experiment": "smollm2_fast_replication",
            "model": MODEL_NAME,
            "design": "cross-dataset at best-of-5-layers (0,6,12,18,23). Matches LODO protocol.",
            "pre_reg_alpha_interval": [ALPHA_LOW, ALPHA_HIGH],
            "dataset_results": dataset_results,
            "kappa_values": kappa_all,
            "logit_q_values": logit_q_all,
            "alpha_fit": float(slope),
            "alpha_se": float(se),
            "intercept": float(intercept),
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "pr1_alpha_pass": bool(pr1),
            "pr2_r_pass": bool(pr2),
            "overall_pass": bool(overall),
            "verdict": "PASS" if overall else "FAIL",
        }
    else:
        output = {
            "experiment": "smollm2_fast_replication",
            "model": MODEL_NAME,
            "dataset_results": dataset_results,
            "error": f"Only {len(kappa_all)} datasets valid",
            "overall_pass": False,
            "verdict": "FAIL",
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
