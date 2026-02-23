"""
Two-part fix for SmolLM2 LOAO replication:

1. Convert existing SmolLM2 3-dataset caches from q_raw -> q_norm convention
   (The original LOAO used q_norm, not q_raw)

2. Generate go_emotions cache for SmolLM2 using q_norm convention
   (The original 12-arch LOAO used 4 datasets: dbpedia, agnews, 20newsgroups, go_emotions)

3. Run corrected LOAO analysis on original 12 archs + SmolLM2, 4 datasets
   -> check if SmolLM2 alpha in [2.43, 3.29]

This corrects a design error: the pre-reg said 'logit(q_raw)' but the original
calibration used logit(q_norm). Using q_norm is correct for comparison with [2.43, 3.29].
"""
import json
import os
import glob
import numpy as np
from scipy.stats import pearsonr
from scipy.special import logit as scipy_logit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_SHORT = "SmolLM2-1.7B"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
ALPHA_LOW = 2.43
ALPHA_HIGH = 3.29
LAYER_INDICES = [6, 12, 18, 23]

# The original 12 architectures that calibrated [2.43, 3.29]
ORIGINAL_12 = {
    'pythia-160m', 'pythia-410m', 'pythia-1b', 'gpt-neo-125m',
    'Qwen2.5-0.5B', 'OLMo-1B-hf',
    'TinyLlama-1.1B-intermediate-step-1431k-3T',
    'Qwen3-0.6B', 'Qwen3-1.7B', 'Mistral-7B-v0.3',
    'Falcon-H1-0.5B-Base', 'rwkv-4-169m-pile'
}

ALL_4_DATASETS = ['dbpedia', 'agnews', '20newsgroups', 'go_emotions']


# ============================================================
# STEP 1: Fix existing SmolLM2 caches (q_raw -> q_norm)
# ============================================================
def fix_smollm2_caches():
    """Convert stored q_raw and logit(q_raw) to q_norm and logit(q_norm)."""
    for ds in ['dbpedia', 'agnews', '20newsgroups']:
        path = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds}_{MODEL_SHORT}.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            pts = json.load(f)
        for pt in pts:
            K = pt['K']
            q_raw = pt['q']  # stored as raw accuracy
            q_norm = float((q_raw - 1.0/K) / (1.0 - 1.0/K))
            q_norm = float(np.clip(q_norm, 0.001, 0.999))
            pt['q'] = q_norm  # overwrite with q_norm
            pt['logit_q'] = float(scipy_logit(q_norm))  # recompute logit
        with open(path, 'w') as f:
            json.dump(pts, f, indent=2)
        print(f"  Fixed {path}: {len(pts)} points converted to q_norm")


# ============================================================
# STEP 2: Generate go_emotions for SmolLM2
# ============================================================
def extract_mean_pool_at_layer(model, tokenizer, device, texts, layer_idx):
    import torch
    all_embs = []
    for b_start in range(0, len(texts), 32):
        batch = texts[b_start:b_start + 32]
        tok = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**tok, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx + 1].float()
        mask = tok["attention_mask"].unsqueeze(-1).float()
        mask_sum = mask.sum(1).clamp(min=1e-9)
        mean_emb = (hidden * mask).sum(1) / mask_sum
        emb_np = mean_emb.cpu().numpy()
        valid = ~np.isnan(emb_np).any(axis=1)
        all_embs.append(emb_np[valid])
    return np.vstack(all_embs)


def compute_kappa_nearest(embs, labels, K, subsample=500):
    unique_classes = sorted(set(labels))
    d = embs.shape[1]
    class_embs = {}
    for ci in unique_classes:
        idx = np.where(labels == ci)[0][:subsample]
        class_embs[ci] = embs[idx]
    centroids = {}
    within_var_sum = 0.0
    n_total = 0
    for ci in unique_classes:
        e = class_embs[ci]
        centroids[ci] = e.mean(0)
        within_var_sum += np.sum((e - centroids[ci]) ** 2)
        n_total += len(e)
    sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))
    kappa_list = []
    for ci in unique_classes:
        min_dist = float("inf")
        for cj in unique_classes:
            if ci == cj:
                continue
            dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            min_dist = min(min_dist, dist)
        kappa_list.append(min_dist / (sigma_W * np.sqrt(d)))
    return float(np.mean(kappa_list)), float(np.min(kappa_list))


def compute_knn_q_norm(embs, labels, K, subsample=500):
    """Returns q_norm = (acc - 1/K)/(1-1/K)."""
    from sklearn.neighbors import KNeighborsClassifier
    unique_classes = sorted(set(labels))
    rng = np.random.default_rng(42)
    train_embs, train_labels, test_embs, test_labels = [], [], [], []
    for ci in unique_classes:
        idx = np.where(labels == ci)[0][:subsample]
        n = len(idx)
        n_train = max(1, int(0.8 * n))
        perm = rng.permutation(n)
        train_embs.append(embs[idx[perm[:n_train]]])
        train_labels.extend([ci] * n_train)
        test_embs.append(embs[idx[perm[n_train:]]])
        test_labels.extend([ci] * (n - n_train))
    train_X = np.vstack(train_embs)
    test_X = np.vstack(test_embs)
    if np.isnan(train_X).any() or np.isnan(test_X).any():
        return None
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(train_X, np.array(train_labels))
    acc = float(np.mean(knn.predict(test_X) == np.array(test_labels)))
    K_eff = len(unique_classes)
    q_norm = float((acc - 1.0/K_eff) / (1.0 - 1.0/K_eff))
    return float(np.clip(q_norm, 0.001, 0.999))


def generate_go_emotions(model, tokenizer, device):
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder

    ds_name = "go_emotions"
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name} (K=28)")

    ds = load_dataset("google-research-datasets/go_emotions", "simplified",
                      split="test", trust_remote_code=False)
    texts_raw = [x["text"] for x in ds]
    labels_raw = [x["labels"][0] if isinstance(x["labels"], list) and x["labels"] else 0
                  for x in ds]
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    K = len(le.classes_)
    print(f"  K={K}, total={len(texts_raw)}")

    texts_sub = []
    labels_sub = []
    for ci in range(K):
        idx_ci = [i for i, l in enumerate(labels) if l == ci][:500]
        texts_sub.extend([texts_raw[i] for i in idx_ci])
        labels_sub.extend([ci] * len(idx_ci))
    labels_arr = np.array(labels_sub)
    logKm1 = float(np.log(K - 1))

    cache_points = []
    for layer_idx in LAYER_INDICES:
        print(f"  Layer {layer_idx}...", flush=True)
        embs = extract_mean_pool_at_layer(model, tokenizer, device, texts_sub, layer_idx)
        print(f"    {len(embs)} valid embeddings")

        kappa_near, kappa_min = compute_kappa_nearest(embs, labels_arr[:len(embs)], K)
        q_norm_raw = compute_knn_q_norm(embs, labels_arr[:len(embs)], K)
        if q_norm_raw is None:
            print(f"    SKIP: KNN failed")
            continue
        logit_q = float(scipy_logit(q_norm_raw))

        pt = {
            "model": MODEL_SHORT, "dataset": ds_name, "layer": layer_idx,
            "K": K, "q": q_norm_raw, "kappa_nearest": kappa_near,
            "kappa_min": kappa_min, "logit_q": logit_q, "logKm1": logKm1,
        }
        cache_points.append(pt)
        print(f"    kappa={kappa_near:.4f}, q_norm={q_norm_raw:.3f}, logit={logit_q:.3f}")

    cache_path = os.path.join(RESULTS_DIR, f"kappa_near_cache_go_emotions_{MODEL_SHORT}.json")
    with open(cache_path, 'w') as f:
        json.dump(cache_points, f, indent=2)
    print(f"  Saved: {cache_path}")
    return cache_points


# ============================================================
# STEP 3: LOAO Analysis (12 original archs + SmolLM2, 4 datasets)
# ============================================================
def run_loao_analysis():
    all_points = []
    for ds in ALL_4_DATASETS:
        pattern = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds}_*.json")
        for cache_file in glob.glob(pattern):
            model_short = os.path.basename(cache_file).replace(
                f"kappa_near_cache_{ds}_", "").replace(".json", "")
            # Only include original 12 architectures OR SmolLM2
            if model_short not in ORIGINAL_12 and model_short != MODEL_SHORT:
                continue
            with open(cache_file) as f:
                pts = json.load(f)
            for pt in pts:
                if 'logKm1' not in pt:
                    pt['logKm1'] = float(np.log(pt['K'] - 1))
            all_points.extend(pts)

    print(f"\n{'='*60}")
    print("CORRECTED LOAO ANALYSIS (12 original archs + SmolLM2, 4 datasets)")
    print(f"{'='*60}")
    print(f"Total points: {len(all_points)}")
    models = sorted(set(p['model'] for p in all_points))
    print(f"Architectures ({len(models)}): {models}")

    train_pts = [p for p in all_points if p['model'] != MODEL_SHORT]
    test_pts = [p for p in all_points if p['model'] == MODEL_SHORT]
    print(f"\nTrain: {len(train_pts)} pts ({len(set(p['model'] for p in train_pts))} archs)")
    print(f"Test (SmolLM2): {len(test_pts)} pts")

    # Fit on train (12 original archs)
    kappa_tr = np.array([p['kappa_nearest'] for p in train_pts])
    logKm1_tr = np.array([p['logKm1'] for p in train_pts])
    logit_tr = np.array([p['logit_q'] for p in train_pts])
    X_tr = np.column_stack([kappa_tr, logKm1_tr, np.ones(len(train_pts))])
    coeffs, _, _, _ = np.linalg.lstsq(X_tr, logit_tr, rcond=None)
    alpha_loao, beta_loao, C0_loao = coeffs
    print(f"\nTrain fit (12 archs): alpha={alpha_loao:.4f}, beta={beta_loao:.4f}, C={C0_loao:.4f}")

    # Fit SmolLM2 independently
    kappa_te = np.array([p['kappa_nearest'] for p in test_pts])
    logKm1_te = np.array([p['logKm1'] for p in test_pts])
    logit_te = np.array([p['logit_q'] for p in test_pts])
    X_te = np.column_stack([kappa_te, logKm1_te, np.ones(len(test_pts))])
    coeffs_smol, _, _, _ = np.linalg.lstsq(X_te, logit_te, rcond=None)
    alpha_smol, beta_smol, C_smol = coeffs_smol
    print(f"\nSmolLM2 fit: alpha={alpha_smol:.4f}, beta={beta_smol:.4f}, C={C_smol:.4f}")

    # Prediction r
    logit_pred_frozen = X_te @ coeffs
    r_pred, p_pred = pearsonr(logit_te, logit_pred_frozen)
    mae = float(np.mean(np.abs(logit_te - logit_pred_frozen)))

    pr1 = ALPHA_LOW <= alpha_smol <= ALPHA_HIGH
    pr2 = r_pred >= 0.80

    print(f"\nPR1 (SmolLM2 alpha in [{ALPHA_LOW},{ALPHA_HIGH}]): {'PASS' if pr1 else 'FAIL'}")
    print(f"     alpha_smol={alpha_smol:.4f}")
    print(f"PR2 (r(pred,obs) >= 0.80): {'PASS' if pr2 else 'FAIL'}")
    print(f"     r={r_pred:.4f}, MAE={mae:.4f}")
    print(f"\nOVERALL: {'PASS' if (pr1 and pr2) else 'FAIL'}")

    print(f"\nSmolLM2 data points (q_norm convention):")
    for p in sorted(test_pts, key=lambda x: (x['dataset'], x['layer'])):
        pred = coeffs[0]*p['kappa_nearest'] + coeffs[1]*p['logKm1'] + coeffs[2]
        print(f"  {p['dataset']} L{p['layer']}: kappa={p['kappa_nearest']:.4f}, "
              f"q_norm={p['q']:.3f}, logit={p['logit_q']:.3f}, pred={pred:.3f}")

    output = {
        "experiment": "smollm2_loao_corrected",
        "model": MODEL_NAME,
        "design": "LOAO using q_norm convention (matches original calibration), 4 datasets, 12 original archs + SmolLM2",
        "convention": "logit(q_norm) -- matches original 12-arch LOAO that produced [2.43,3.29]",
        "pre_reg_alpha_interval": [ALPHA_LOW, ALPHA_HIGH],
        "datasets_used": ALL_4_DATASETS,
        "train_architectures": sorted(ORIGINAL_12),
        "n_train_points": len(train_pts),
        "n_smollm2_points": len(test_pts),
        "smollm2_points": test_pts,
        "train_fit": {"alpha": float(alpha_loao), "beta": float(beta_loao), "C0": float(C0_loao)},
        "smollm2_fit": {"alpha": float(alpha_smol), "beta": float(beta_smol), "C": float(C_smol)},
        "evaluation": {"mae_frozen": mae, "pearson_r_pred": float(r_pred), "pearson_p_pred": float(p_pred)},
        "pr1_alpha_pass": bool(pr1),
        "pr2_r_pass": bool(pr2),
        "overall_pass": bool(pr1 and pr2),
        "verdict": "PASS" if (pr1 and pr2) else "FAIL",
    }

    out_path = os.path.join(RESULTS_DIR, "cti_smollm2_loao_corrected.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    return pr1, pr2


def main():
    print("STEP 1: Fix SmolLM2 caches (q_raw -> q_norm)")
    fix_smollm2_caches()

    print("\nSTEP 2: Generate go_emotions for SmolLM2")
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    generate_go_emotions(model, tokenizer, device)
    del model
    torch.cuda.empty_cache()

    print("\nSTEP 3: Corrected LOAO Analysis")
    run_loao_analysis()


if __name__ == "__main__":
    main()
