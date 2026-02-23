"""
Strictly blind OOD prediction test for CTI law.

Design:
  - alpha, beta, C_d_main (for dbpedia/agnews/20news/go_emotions) from 12-arch LOAO training
  - C_banking77, C_amazon_massive estimated from 6 partial architectures (q_raw->q_norm fixed)
  - SmolLM2 banking77 and amazon_massive caches generated fresh (zero calibration from SmolLM2)
  - Predict SmolLM2 logit on OOD datasets using ONLY training-derived parameters
  - Evaluation: Pearson r(pred, obs) across 8 OOD data points (2 datasets x 4 layers)

Pre-registered: r >= 0.80 for blind OOD prediction to pass

This is a STRONGER test than LOAO because:
  1. SmolLM2 is a NEW architecture (not in training)
  2. banking77 and amazon_massive are NEW datasets (not in main LOAO training)
  3. C_d for new datasets estimated from different architectures (6 partial archs)
  4. NO SmolLM2 data used anywhere for calibration
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
LAYER_INDICES = [6, 12, 18, 23]

# Training parameters from 12-arch LOAO (per-dataset-intercept model)
# alpha=1.477, beta=-0.309 (from main LOAO per-dataset fit)
# alpha=2.866, beta=-0.746 (from main LOAO global fit)
# Use PER-DATASET model since it has better predictive power
ALPHA_DS = 1.477
BETA_DS = -0.309

OOD_DATASETS = [
    {
        "name": "banking77",
        "hf_name": "banking77",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "K": 77,
        "n_per_class": 500,
        "partial_archs": 6,  # 6 partial cache files exist
    },
    {
        "name": "amazon_massive",
        "hf_name": "AmazonScience/massive",
        "hf_subset": "en-US",
        "split": "train",
        "text_col": "utt",
        "label_col": "intent",
        "K": 59,
        "n_per_class": 500,
        "partial_archs": 6,
    },
]


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
        mean_emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb_np = mean_emb.cpu().numpy()
        valid = ~np.isnan(emb_np).any(axis=1)
        all_embs.append(emb_np[valid])
    return np.vstack(all_embs)


def compute_kappa(embs, labels, K, subsample=500):
    unique_classes = sorted(set(labels))
    d = embs.shape[1]
    class_embs = {ci: embs[np.where(labels == ci)[0][:subsample]] for ci in unique_classes}
    centroids = {}
    wv, nt = 0.0, 0
    for ci in unique_classes:
        e = class_embs[ci]
        centroids[ci] = e.mean(0)
        wv += np.sum((e - centroids[ci]) ** 2)
        nt += len(e)
    sigma_W = float(np.sqrt(wv / (nt * d)))
    kappa_list = []
    for ci in unique_classes:
        min_dist = min(np.linalg.norm(centroids[ci] - centroids[cj])
                       for cj in unique_classes if cj != ci)
        kappa_list.append(float(min_dist) / (sigma_W * np.sqrt(d)))
    return float(np.mean(kappa_list)), float(np.min(kappa_list)), sigma_W


def compute_q_norm(embs, labels, K, subsample=500):
    from sklearn.neighbors import KNeighborsClassifier
    unique_classes = sorted(set(labels))
    rng = np.random.default_rng(42)
    tr_e, tr_l, te_e, te_l = [], [], [], []
    for ci in unique_classes:
        idx = np.where(labels == ci)[0][:subsample]
        n = len(idx); n_tr = max(1, int(0.8*n)); perm = rng.permutation(n)
        tr_e.append(embs[idx[perm[:n_tr]]]); tr_l += [ci]*n_tr
        te_e.append(embs[idx[perm[n_tr:]]]); te_l += [ci]*(n-n_tr)
    train_X, test_X = np.vstack(tr_e), np.vstack(te_e)
    if np.isnan(train_X).any() or np.isnan(test_X).any():
        return None
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(train_X, np.array(tr_l))
    acc = float(np.mean(knn.predict(test_X) == np.array(te_l)))
    K_eff = len(unique_classes)
    return float(np.clip((acc - 1.0/K_eff) / (1.0 - 1.0/K_eff), 0.001, 0.999))


def load_partial_caches_and_estimate_Cd(ds_name, alpha, beta):
    """Load existing 6-arch partial caches, convert q_raw->q_norm, estimate C_d."""
    pattern = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name}_*.json")
    all_pts = []
    for f in glob.glob(pattern):
        if MODEL_SHORT in f:
            continue
        with open(f) as fp:
            pts = json.load(fp)
        for pt in pts:
            K = pt['K']
            q_raw = pt['q']  # stored as q_raw in partial caches
            q_norm = float(np.clip((q_raw - 1.0/K) / (1.0 - 1.0/K), 0.001, 0.999))
            pt['q_norm'] = q_norm
            pt['logit_q_norm'] = float(scipy_logit(q_norm))
            if 'logKm1' not in pt:
                pt['logKm1'] = float(np.log(K - 1))
        all_pts.extend(pts)
    print(f"  {ds_name}: loaded {len(all_pts)} pts from {len(glob.glob(pattern))-1} partial archs")

    # Estimate C_d = mean(logit_q_norm - alpha*kappa - beta*logKm1)
    residuals = [p['logit_q_norm'] - alpha*p['kappa_nearest'] - beta*p['logKm1']
                 for p in all_pts]
    C_d = float(np.mean(residuals))
    C_d_std = float(np.std(residuals))
    print(f"  C_{ds_name} = {C_d:.4f} ± {C_d_std:.4f} ({len(residuals)} residuals)")
    return C_d, all_pts


def generate_smollm2_cache(model, tokenizer, device, ds_config):
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder

    ds_name = ds_config["name"]
    print(f"\n{'='*60}")
    print(f"Generating SmolLM2 cache: {ds_name} (K={ds_config['K']})")

    try:
        if ds_config.get("hf_subset"):
            ds = load_dataset(ds_config["hf_name"], ds_config["hf_subset"],
                              split=ds_config["split"], trust_remote_code=False)
        else:
            ds = load_dataset(ds_config["hf_name"], split=ds_config["split"],
                              trust_remote_code=False)
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        return []

    texts_raw = [x[ds_config["text_col"]] for x in ds]
    labels_raw = [x[ds_config["label_col"]] for x in ds]
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    K = len(le.classes_)
    print(f"  K={K}, n={len(texts_raw)}")

    texts_sub, labels_sub = [], []
    for ci in range(K):
        idx_ci = [i for i, l in enumerate(labels) if l == ci][:ds_config["n_per_class"]]
        texts_sub += [texts_raw[i] for i in idx_ci]
        labels_sub += [ci] * len(idx_ci)
    labels_arr = np.array(labels_sub)
    logKm1 = float(np.log(K - 1))

    cache_points = []
    for layer_idx in LAYER_INDICES:
        print(f"  Layer {layer_idx}...", flush=True)
        embs = extract_mean_pool_at_layer(model, tokenizer, device, texts_sub, layer_idx)
        kappa, kappa_min, _ = compute_kappa(embs, labels_arr[:len(embs)], K)
        q_norm = compute_q_norm(embs, labels_arr[:len(embs)], K)
        if q_norm is None:
            print("    SKIP: NaN in KNN")
            continue
        logit_q = float(scipy_logit(q_norm))
        pt = {"model": MODEL_SHORT, "dataset": ds_name, "layer": layer_idx,
              "K": K, "q": q_norm, "kappa_nearest": kappa, "kappa_min": kappa_min,
              "logit_q": logit_q, "logKm1": logKm1}
        cache_points.append(pt)
        print(f"    kappa={kappa:.4f}, q_norm={q_norm:.3f}, logit={logit_q:.3f}")

    cache_path = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name}_{MODEL_SHORT}.json")
    with open(cache_path, 'w') as f:
        json.dump(cache_points, f, indent=2)
    print(f"  Saved: {cache_path}")
    return cache_points


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("="*60)
    print("STRICTLY BLIND OOD PREDICTION TEST")
    print(f"Training params: alpha={ALPHA_DS:.4f}, beta={BETA_DS:.4f} (per-dataset-intercept)")
    print("New datasets: banking77 (K=77) and amazon_massive (K=59)")
    print("C_d estimated from 6 partial architectures (NOT including SmolLM2)")
    print("SmolLM2 caches generated from scratch (zero SmolLM2 calibration)")
    print("="*60)

    # Step 1: Estimate C_d for OOD datasets from partial archs
    print("\nSTEP 1: Estimate C_d from partial architectures")
    C_d_values = {}
    for ds_config in OOD_DATASETS:
        C_d, _ = load_partial_caches_and_estimate_Cd(ds_config["name"], ALPHA_DS, BETA_DS)
        C_d_values[ds_config["name"]] = C_d

    # Step 2: Generate SmolLM2 caches for OOD datasets
    print("\nSTEP 2: Generate SmolLM2 caches for OOD datasets")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    smollm2_ood_points = []
    for ds_config in OOD_DATASETS:
        pts = generate_smollm2_cache(model, tokenizer, device, ds_config)
        smollm2_ood_points.extend(pts)

    del model
    torch.cuda.empty_cache()

    # Step 3: Blind prediction
    print("\nSTEP 3: Blind OOD prediction")
    obs_logits = []
    pred_logits = []
    print(f"\n{'Dataset':<20} {'Layer':<6} {'kappa':<8} {'q_norm':<8} {'obs_logit':<12} {'pred_logit':<12} {'error':<8}")
    for pt in sorted(smollm2_ood_points, key=lambda x: (x['dataset'], x['layer'])):
        C_d = C_d_values.get(pt['dataset'], 0.0)
        pred = ALPHA_DS * pt['kappa_nearest'] + BETA_DS * pt['logKm1'] + C_d
        obs = pt['logit_q']
        obs_logits.append(obs)
        pred_logits.append(pred)
        print(f"  {pt['dataset']:<20} L{pt['layer']:<5} {pt['kappa_nearest']:<8.4f} "
              f"{pt['q']:<8.3f} {obs:<12.3f} {pred:<12.3f} {obs-pred:<8.3f}")

    if len(obs_logits) >= 3:
        r, pv = pearsonr(obs_logits, pred_logits)
        mae = float(np.mean(np.abs(np.array(obs_logits) - np.array(pred_logits))))
        print(f"\nBlind OOD prediction: r={r:.4f} (p={pv:.4f}), MAE={mae:.4f}")
        print(f"n={len(obs_logits)} data points (2 datasets × 4 layers)")
        threshold = 0.80
        status = "PASS" if r >= threshold else "FAIL"
        print(f"Pre-registered threshold r>={threshold}: {status}")
    else:
        r, pv, mae = 0.0, 1.0, float('inf')
        print("Insufficient data points for correlation")

    output = {
        "experiment": "smollm2_ood_blind_prediction",
        "model": MODEL_NAME,
        "design": "Blind OOD: alpha/beta from 12-arch LOAO, C_d from 6 partial archs, zero SmolLM2 calibration",
        "training_params": {"alpha": ALPHA_DS, "beta": BETA_DS},
        "C_d_values": C_d_values,
        "ood_datasets": ["banking77 (K=77)", "amazon_massive (K=59)"],
        "smollm2_ood_points": smollm2_ood_points,
        "blind_prediction": {
            "pearson_r": float(r),
            "pearson_p": float(pv),
            "mae": float(mae),
            "n_points": len(obs_logits),
            "pass_threshold_0.80": bool(r >= 0.80),
        },
        "verdict": "PASS" if r >= 0.80 else "FAIL",
    }
    out_path = os.path.join(RESULTS_DIR, "cti_smollm2_ood_prediction.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
