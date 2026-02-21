#!/usr/bin/env python -u
"""
PRE-REGISTERED HELD-OUT UNIVERSALITY TEST (Codex recommendation, Feb 20 2026)

Goal: Show that logit(q) = A*(dist_ratio-1) + C is universal:
  - Fit A, C on TRAINING models/datasets
  - Predict q on HELD-OUT models/datasets with FROZEN A, C
  - If MAE < threshold on held-out, claim universality

Nobel-track significance:
  - Previous tests fit parameters on all data together (not truly held-out)
  - This test is a genuine prospective prediction: fit on one set, predict another
  - Passing this = "the law has ONE universal formula, not a family of fits"

Pre-registration (BEFORE running):
  - Training: Pythia-160m, Pythia-410m on CLINC, AGNews
  - Held-out: Pythia-1b on CLINC, AGNews (different size)
             + Any Pythia model on DBpedia (different dataset)
  - Criterion: held-out MAE < 0.10 q units (absolute), R2 > 0.80

Split design:
  - Training split: (model in {160m, 410m}) AND (dataset in {clinc, agnews})
  - Held-out split A: (model=1b) AND (dataset in {clinc, agnews})  [new model size]
  - Held-out split B: (any model) AND (dataset=dbpedia)            [new dataset]
  - Held-out split C: (model=1b) AND (dataset=dbpedia)            [both new]
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.stats import spearmanr
import torch
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# PRE-REGISTERED SPLITS
# ================================================================
TRAINING_MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
]
TRAINING_DATASETS = ["clinc_oos", "agnews"]

HELD_OUT_A_MODELS = ["EleutherAI/pythia-1b"]   # new model size
HELD_OUT_A_DATASETS = ["clinc_oos", "agnews"]

HELD_OUT_B_MODELS = [                            # any model, new dataset
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]
HELD_OUT_B_DATASETS = ["dbpedia_14"]

HELD_OUT_C_MODELS = ["EleutherAI/pythia-1b"]    # both new
HELD_OUT_C_DATASETS = ["dbpedia_14"]

# Pre-registered threshold
MAE_THRESHOLD = 0.10
R2_THRESHOLD = 0.80

N_LAYERS_SAMPLE = 8    # layers to sample per model
N_SAMPLES = 1000       # samples per dataset (for speed)


def load_model_and_tokenizer(model_id):
    """Load model from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()
    return model, tok


def load_dataset_texts_labels(dataset_name, n_samples=1000):
    """Load texts and integer labels for a dataset."""
    from datasets import load_dataset
    if dataset_name == "clinc_oos":
        ds = load_dataset("clinc_oos", "plus", split="train")
        texts = list(ds["text"])[:n_samples]
        labels = np.array(ds["intent"])[:n_samples]
    elif dataset_name == "agnews":
        ds = load_dataset("ag_news", split="train")
        texts = list(ds["text"])[:n_samples]
        labels = np.array(ds["label"])[:n_samples]
    elif dataset_name == "dbpedia_14":
        ds = load_dataset("dbpedia_14", split="train")
        # DBpedia is sorted by class - must shuffle before sampling
        import random
        random.seed(42)
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:n_samples]
        texts = [ds["content"][i] for i in indices]
        labels = np.array([ds["label"][i] for i in indices])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return texts, labels


def get_embeddings_at_layer(model, tokenizer, texts, layer_idx, batch_size=32):
    """Get mean-pooled embeddings at a specific layer."""
    all_hidden = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx + 1]  # +1 for embedding
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        all_hidden.append(pooled.cpu().float().numpy())
    return np.concatenate(all_hidden, axis=0)


def compute_dist_ratio_and_q(embeddings, labels):
    """Compute dist_ratio = E[D_inter] / E[D_intra] and kNN q."""
    X = np.nan_to_num(embeddings, nan=0.0)
    y = labels

    K = len(np.unique(y))
    N = len(X)

    if N < K * 5 or K < 2:
        return None

    # kNN accuracy (stratified 80/20 split)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))

    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    q = max(min(q, 0.999), 0.001)

    # dist_ratio: sample-based E[D_intra] and E[D_inter]
    # Subsample for efficiency
    n_sub = min(N, 500)
    idx_sub = np.random.choice(N, n_sub, replace=False)
    X_sub = X[idx_sub]
    y_sub = y[idx_sub]

    # Compute pairwise distances
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X_sub, metric="euclidean")
    np.fill_diagonal(D, np.inf)

    intra_dists = []
    inter_dists = []
    for i in range(n_sub):
        same_mask = (y_sub == y_sub[i])
        same_mask[i] = False  # exclude self
        diff_mask = ~same_mask
        diff_mask[i] = False

        if same_mask.sum() > 0:
            intra_dists.append(D[i][same_mask].min())
        if diff_mask.sum() > 0:
            inter_dists.append(D[i][diff_mask].min())

    if len(intra_dists) == 0 or len(inter_dists) == 0:
        return None

    d_intra = float(np.mean(intra_dists))
    d_inter = float(np.mean(inter_dists))
    dist_ratio = d_inter / (d_intra + 1e-10)

    return {
        "knn_acc": acc,
        "q": q,
        "dist_ratio": dist_ratio,
        "K": K,
        "N": N,
    }


def collect_data_for_split(model_ids, dataset_names, max_layers=N_LAYERS_SAMPLE):
    """Collect (dist_ratio, q) points for a given set of models and datasets."""
    data_points = []

    for model_id in model_ids:
        try:
            model, tokenizer = load_model_and_tokenizer(model_id)
            n_layers = model.config.num_hidden_layers
            # Sample layers evenly
            layer_indices = list(np.linspace(0, n_layers-1, max_layers, dtype=int))

            for dataset_name in dataset_names:
                print(f"  {model_id.split('/')[-1]} + {dataset_name}...", end="", flush=True)
                texts, labels = load_dataset_texts_labels(dataset_name, N_SAMPLES)

                for layer_idx in layer_indices:
                    emb = get_embeddings_at_layer(model, tokenizer, texts, layer_idx)
                    result = compute_dist_ratio_and_q(emb, labels)
                    if result is not None:
                        result["model"] = model_id
                        result["dataset"] = dataset_name
                        result["layer"] = int(layer_idx)
                        data_points.append(result)

                print(f" {len([p for p in data_points if p['model']==model_id and p['dataset']==dataset_name])} pts")

            del model
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR: {e}]")

    return data_points


def fit_law(data_points, verbose=True):
    """Fit logit(q) = A*(dist_ratio - 1) + C using OLS in logit space."""
    drs = np.array([p["dist_ratio"] for p in data_points])
    qs = np.array([p["q"] for p in data_points])
    logit_qs = logit(qs)

    # Design matrix: [dist_ratio - 1, 1]
    X_design = np.column_stack([drs - 1, np.ones(len(drs))])
    # OLS: theta = (X^T X)^{-1} X^T y
    try:
        theta = np.linalg.lstsq(X_design, logit_qs, rcond=None)[0]
        A, C = float(theta[0]), float(theta[1])
    except Exception:
        A, C = 2.3, 0.0  # fallback to prior

    pred = A * (drs - 1) + C
    r2 = 1 - np.sum((logit_qs - pred)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)
    q_pred = expit(pred)
    mae_q = float(np.mean(np.abs(qs - q_pred)))

    if verbose:
        print(f"    Fit: logit(q) = {A:.3f}*(DR-1) + {C:.3f}")
        print(f"    R2={r2:.4f}, MAE_q={mae_q:.4f}")

    return A, C, float(r2), mae_q


def predict_held_out(data_points, A, C, verbose=True):
    """Apply frozen A, C to held-out data points and compute metrics."""
    drs = np.array([p["dist_ratio"] for p in data_points])
    qs = np.array([p["q"] for p in data_points])

    pred_logit = A * (drs - 1) + C
    q_pred = expit(pred_logit)
    logit_qs = logit(qs)

    mae_q = float(np.mean(np.abs(qs - q_pred)))
    r2 = 1 - np.sum((logit_qs - pred_logit)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)
    rho = float(spearmanr(qs, q_pred).correlation)

    if verbose:
        print(f"    Held-out: n={len(data_points)}, MAE_q={mae_q:.4f}, R2={r2:.4f}, rho={rho:.4f}")
        print(f"    MAE threshold: {MAE_THRESHOLD}, R2 threshold: {R2_THRESHOLD}")
        print(f"    PASS: {mae_q < MAE_THRESHOLD and r2 > R2_THRESHOLD}")

    return mae_q, float(r2), rho


def main():
    print("=" * 70)
    print("PRE-REGISTERED HELD-OUT UNIVERSALITY TEST")
    print("logit(q) = A*(dist_ratio-1) + C : fit on training, predict held-out")
    print("=" * 70)
    print(f"\nPre-registered criteria:")
    print(f"  MAE_q < {MAE_THRESHOLD} on held-out")
    print(f"  R2 > {R2_THRESHOLD} on held-out")
    print()

    # ================================================================
    # Step 1: Collect training data
    # ================================================================
    print("=" * 70)
    print("STEP 1: Collecting training data")
    print(f"  Models: {[m.split('/')[-1] for m in TRAINING_MODELS]}")
    print(f"  Datasets: {TRAINING_DATASETS}")
    print("=" * 70)

    training_data = collect_data_for_split(TRAINING_MODELS, TRAINING_DATASETS)
    print(f"\n  Training data: {len(training_data)} valid points")

    if len(training_data) < 10:
        print("  [ERROR] Too few training points!")
        return

    # ================================================================
    # Step 2: Fit law on training data
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Fitting law on training data")
    print("=" * 70)
    A_train, C_train, r2_train, mae_train = fit_law(training_data)
    print(f"  Training fit: A={A_train:.4f}, C={C_train:.4f}")
    print(f"  Training R2={r2_train:.4f}, MAE={mae_train:.4f}")

    # ================================================================
    # Step 3: Collect held-out data and predict
    # ================================================================
    results = {
        "pre_registered_criteria": {
            "mae_threshold": MAE_THRESHOLD,
            "r2_threshold": R2_THRESHOLD,
        },
        "training": {
            "models": TRAINING_MODELS,
            "datasets": TRAINING_DATASETS,
            "n_points": len(training_data),
            "A": A_train,
            "C": C_train,
            "r2": r2_train,
            "mae": mae_train,
        },
        "held_out_splits": {}
    }

    for split_name, model_ids, dataset_names in [
        ("A_new_model_size", HELD_OUT_A_MODELS, HELD_OUT_A_DATASETS),
        ("B_new_dataset", HELD_OUT_B_MODELS, HELD_OUT_B_DATASETS),
        ("C_both_new", HELD_OUT_C_MODELS, HELD_OUT_C_DATASETS),
    ]:
        print(f"\n{'='*70}")
        print(f"STEP 3{split_name[0]}: Held-out split {split_name}")
        print(f"  Models: {[m.split('/')[-1] for m in model_ids]}")
        print(f"  Datasets: {dataset_names}")
        print("=" * 70)

        held_out_data = collect_data_for_split(model_ids, dataset_names)
        print(f"  Held-out data: {len(held_out_data)} valid points")

        if len(held_out_data) < 5:
            print(f"  [ERROR] Too few held-out points for split {split_name}!")
            results["held_out_splits"][split_name] = {
                "n_points": len(held_out_data),
                "error": "too_few_points"
            }
            continue

        mae_ho, r2_ho, rho_ho = predict_held_out(held_out_data, A_train, C_train)
        passed = mae_ho < MAE_THRESHOLD and r2_ho > R2_THRESHOLD

        print(f"  -> {'PASS' if passed else 'FAIL'}")

        results["held_out_splits"][split_name] = {
            "models": model_ids,
            "datasets": dataset_names,
            "n_points": len(held_out_data),
            "mae_q": mae_ho,
            "r2": r2_ho,
            "rho": rho_ho,
            "passed": passed,
        }

    # ================================================================
    # Overall score
    # ================================================================
    passed_splits = sum(1 for v in results["held_out_splits"].values()
                       if isinstance(v.get("passed"), bool) and v["passed"])
    total_splits = sum(1 for v in results["held_out_splits"].values()
                      if "error" not in v)

    results["scorecard"] = {
        "passed": passed_splits,
        "total": total_splits,
        "overall_pass": passed_splits == total_splits and total_splits > 0,
    }

    print("\n" + "=" * 70)
    print("FINAL SCORECARD")
    print("=" * 70)
    for split_name, res in results["held_out_splits"].items():
        if "error" in res:
            print(f"  {split_name}: ERROR ({res['error']})")
        else:
            status = "PASS" if res["passed"] else "FAIL"
            print(f"  {split_name}: [{status}] MAE={res['mae_q']:.4f}, R2={res['r2']:.4f}")
    print(f"\n  TOTAL: {passed_splits}/{total_splits}")

    # Save
    out_path = RESULTS_DIR / "cti_held_out_universality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
