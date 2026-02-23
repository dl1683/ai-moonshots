"""
Fix for OOD prediction: use mteb/amazon_massive_intent instead of
AmazonScience/massive (which uses deprecated loading scripts).

Steps:
1. Generate SmolLM2 cache for amazon_massive using mteb version (K=60, English only)
2. Load existing banking77 cache (already done)
3. Recompute blind prediction with full n=8 points
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

# From 12-arch LOAO per-dataset-intercept model
ALPHA_DS = 1.477
BETA_DS = -0.309


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
    return float(np.mean(kappa_list)), float(np.min(kappa_list))


def compute_q_norm(embs, labels, K, subsample=500):
    from sklearn.neighbors import KNeighborsClassifier
    unique_classes = sorted(set(labels))
    rng = np.random.default_rng(42)
    tr_e, tr_l, te_e, te_l = [], [], [], []
    for ci in unique_classes:
        idx = np.where(labels == ci)[0][:subsample]
        n = len(idx)
        n_tr = max(1, int(0.8 * n))
        perm = rng.permutation(n)
        tr_e.append(embs[idx[perm[:n_tr]]])
        tr_l += [ci] * n_tr
        te_e.append(embs[idx[perm[n_tr:]]])
        te_l += [ci] * (n - n_tr)
    train_X, test_X = np.vstack(tr_e), np.vstack(te_e)
    if np.isnan(train_X).any() or np.isnan(test_X).any():
        return None
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(train_X, np.array(tr_l))
    acc = float(np.mean(knn.predict(test_X) == np.array(te_l)))
    K_eff = len(unique_classes)
    return float(np.clip((acc - 1.0 / K_eff) / (1.0 - 1.0 / K_eff), 0.001, 0.999))


def load_partial_caches_and_estimate_Cd(ds_name, alpha, beta):
    """Load partial caches, convert q_raw->q_norm, estimate C_d."""
    pattern = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name}_*.json")
    all_pts = []
    for f in glob.glob(pattern):
        if MODEL_SHORT in f:
            continue
        with open(f) as fp:
            pts = json.load(fp)
        for pt in pts:
            K = pt['K']
            q_raw = pt['q']
            q_norm = float(np.clip((q_raw - 1.0 / K) / (1.0 - 1.0 / K), 0.001, 0.999))
            pt['q_norm'] = q_norm
            pt['logit_q_norm'] = float(scipy_logit(q_norm))
            if 'logKm1' not in pt:
                pt['logKm1'] = float(np.log(K - 1))
        all_pts.extend(pts)
    print(f"  {ds_name}: loaded {len(all_pts)} pts from partial archs")
    residuals = [p['logit_q_norm'] - alpha * p['kappa_nearest'] - beta * p['logKm1']
                 for p in all_pts]
    C_d = float(np.mean(residuals))
    C_d_std = float(np.std(residuals))
    print(f"  C_{ds_name} = {C_d:.4f} +/- {C_d_std:.4f} ({len(residuals)} residuals)")
    return C_d


def generate_amazon_massive_smollm2(model, tokenizer, device):
    """Generate SmolLM2 cache for amazon_massive using mteb version (English)."""
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder

    ds_name = "amazon_massive"
    print(f"\n{'=' * 60}")
    print(f"Generating SmolLM2 cache: {ds_name} (MTEB version, English)")

    ds = load_dataset("mteb/amazon_massive_intent", split="train")
    # Filter English only
    ds_en = ds.filter(lambda x: x["lang"] == "en")
    print(f"  English samples: {len(ds_en)}")

    texts_raw = [x["text"] for x in ds_en]
    labels_raw = [x["label"] for x in ds_en]
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    K = len(le.classes_)
    print(f"  K={K}, n={len(texts_raw)}")

    texts_sub, labels_sub = [], []
    for ci in range(K):
        idx_ci = [i for i, l in enumerate(labels) if l == ci][:500]
        texts_sub += [texts_raw[i] for i in idx_ci]
        labels_sub += [ci] * len(idx_ci)
    labels_arr = np.array(labels_sub)
    logKm1 = float(np.log(K - 1))

    cache_points = []
    for layer_idx in LAYER_INDICES:
        print(f"  Layer {layer_idx}...", flush=True)
        embs = extract_mean_pool_at_layer(model, tokenizer, device, texts_sub, layer_idx)
        kappa, kappa_min = compute_kappa(embs, labels_arr[:len(embs)], K)
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

    print("=" * 60)
    print("OOD BLIND PREDICTION FIX: amazon_massive via MTEB")
    print(f"alpha={ALPHA_DS:.4f}, beta={BETA_DS:.4f} (per-dataset-intercept)")
    print("=" * 60)

    # Step 1: Re-estimate C_d values (same as before)
    print("\nEstimating C_d from partial archs...")
    C_banking77 = load_partial_caches_and_estimate_Cd("banking77", ALPHA_DS, BETA_DS)
    C_amazon = load_partial_caches_and_estimate_Cd("amazon_massive", ALPHA_DS, BETA_DS)

    # Step 2: Generate amazon_massive cache using MTEB
    print("\nLoading SmolLM2-1.7B...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    amazon_pts = generate_amazon_massive_smollm2(model, tokenizer, device)
    del model
    torch.cuda.empty_cache()

    # Step 3: Load banking77 cache (already generated)
    banking77_path = os.path.join(RESULTS_DIR, f"kappa_near_cache_banking77_{MODEL_SHORT}.json")
    with open(banking77_path) as f:
        banking77_pts = json.load(f)
    print(f"\nLoaded banking77: {len(banking77_pts)} pts")

    # Step 4: Blind prediction on all 8 points
    print("\n" + "=" * 60)
    print("BLIND OOD PREDICTION (8 points: banking77 + amazon_massive)")
    print("=" * 60)

    C_d_values = {"banking77": C_banking77, "amazon_massive": C_amazon}
    all_ood_pts = banking77_pts + amazon_pts

    obs_logits = []
    pred_logits = []
    print(f"\n{'Dataset':<22} {'Layer':<6} {'kappa':<8} {'q_norm':<8} "
          f"{'obs_logit':<12} {'pred_logit':<12} {'error':<8}")
    for pt in sorted(all_ood_pts, key=lambda x: (x['dataset'], x['layer'])):
        C_d = C_d_values.get(pt['dataset'], 0.0)
        pred = ALPHA_DS * pt['kappa_nearest'] + BETA_DS * pt['logKm1'] + C_d
        obs = pt['logit_q']
        obs_logits.append(obs)
        pred_logits.append(pred)
        print(f"  {pt['dataset']:<22} L{pt['layer']:<5} {pt['kappa_nearest']:<8.4f} "
              f"{pt['q']:<8.3f} {obs:<12.3f} {pred:<12.3f} {obs - pred:<8.3f}")

    r, pv = pearsonr(obs_logits, pred_logits)
    mae = float(np.mean(np.abs(np.array(obs_logits) - np.array(pred_logits))))
    print(f"\nBlind OOD prediction: r={r:.4f} (p={pv:.4f}), MAE={mae:.4f}")
    print(f"n={len(obs_logits)} data points (2 datasets x 4 layers)")
    threshold = 0.80
    status = "PASS" if r >= threshold else "FAIL"
    print(f"Pre-registered threshold r>={threshold}: {status}")

    output = {
        "experiment": "smollm2_ood_blind_prediction",
        "model": MODEL_NAME,
        "design": ("Blind OOD: alpha/beta from 12-arch LOAO, C_d from partial archs, "
                   "zero SmolLM2 calibration. amazon_massive via mteb/amazon_massive_intent (en)"),
        "training_params": {"alpha": ALPHA_DS, "beta": BETA_DS},
        "C_d_values": C_d_values,
        "ood_datasets": ["banking77 (K=77)", "amazon_massive (K=60, MTEB English)"],
        "smollm2_ood_points": all_ood_pts,
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
