"""
Expanded blind OOD prediction: new architecture families on new datasets.

Protocol:
- alpha=1.477, beta=-0.309 from 12-arch LOAO (fixed, NOT refitted)
- C_d estimated from 6 partial archs on banking77 / amazon_massive (already cached)
- Test architectures: all must be OUTSIDE original 12-arch calibration set
  Original 12: pythia-160m, pythia-410m, pythia-1b, gpt-neo-125m,
               Qwen2.5-0.5B, OLMo-1B-hf, TinyLlama, Qwen3-0.6B, Qwen3-1.7B,
               Mistral-7B-v0.3, Falcon-H1-0.5B-Base, rwkv-4-169m-pile
- Datasets: banking77 (K=77) and amazon_massive (K=60, MTEB English)
  -- both are new (not in any 12-arch training)

Pre-registered: r >= 0.80 across all test points (architecture x dataset x layer)
"""
import json
import os
import glob
import numpy as np
from scipy.stats import pearsonr
from scipy.special import logit as scipy_logit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LAYER_INDICES = [6, 12, 18, 23]  # proportional layers for 24-layer models; adjusted per model

# Fixed training parameters (from 12-arch LOAO per-dataset-intercept)
ALPHA_DS = 1.477
BETA_DS = -0.309

# New test architectures (NOT in original 12)
# Original 12: pythia-{160m,410m,1b}, gpt-neo-125m, Qwen2.5-0.5B, OLMo-1B-hf,
#              TinyLlama, Qwen3-{0.6B,1.7B}, Mistral-7B-v0.3,
#              Falcon-H1-0.5B-Base, rwkv-4-169m-pile
TEST_MODELS = [
    {
        "short": "phi-2",
        "hf_name": "microsoft/phi-2",
        "n_layers": 32,  # 32 transformer layers (Phi architecture, Microsoft)
        "layer_frac": [0.25, 0.5, 0.75, 1.0],
    },
    {
        "short": "gemma-3-1b",
        "hf_name": "google/gemma-3-1b-it",
        "n_layers": 26,  # 26 transformer layers (Gemma-3, Google)
        "layer_frac": [0.25, 0.5, 0.75, 1.0],
    },
]

# OOD datasets
OOD_DATASETS = [
    {
        "name": "banking77",
        "K": 77,
        "load_fn": "load_banking77",
    },
    {
        "name": "amazon_massive",
        "K": 60,
        "load_fn": "load_amazon_massive",
    },
]


def load_banking77():
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder
    ds = load_dataset("banking77", split="train", trust_remote_code=False)
    texts = [x["text"] for x in ds]
    labels_raw = [x["label"] for x in ds]
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    return texts, labels, len(le.classes_)


def load_amazon_massive():
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder
    ds = load_dataset("mteb/amazon_massive_intent", split="train")
    ds_en = ds.filter(lambda x: x["lang"] == "en")
    texts = [x["text"] for x in ds_en]
    labels_raw = [x["label"] for x in ds_en]
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    return texts, labels, len(le.classes_)


def subsample_per_class(texts, labels, K, n_per_class=500):
    texts_sub, labels_sub = [], []
    for ci in range(K):
        idx_ci = [i for i, l in enumerate(labels) if l == ci][:n_per_class]
        texts_sub += [texts[i] for i in idx_ci]
        labels_sub += [ci] * len(idx_ci)
    return texts_sub, np.array(labels_sub)


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
    centroids, wv, nt = {}, 0.0, 0
    for ci in unique_classes:
        e = class_embs[ci]
        centroids[ci] = e.mean(0)
        wv += np.sum((e - centroids[ci]) ** 2)
        nt += len(e)
    sigma_W = float(np.sqrt(wv / (nt * d)))
    kappa_list = [
        float(min(np.linalg.norm(centroids[ci] - centroids[cj])
                  for cj in unique_classes if cj != ci)) / (sigma_W * np.sqrt(d))
        for ci in unique_classes
    ]
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


def estimate_Cd(ds_name, alpha, beta):
    """Estimate C_d from partial architecture caches (excluding any test models)."""
    test_shorts = {m["short"] for m in TEST_MODELS}
    test_shorts.add("SmolLM2-1.7B")  # already done, exclude too
    pattern = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name}_*.json")
    all_pts = []
    for f in glob.glob(pattern):
        model_id = os.path.basename(f).replace(f"kappa_near_cache_{ds_name}_", "").replace(".json", "")
        if any(ts in model_id for ts in test_shorts):
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
    if not all_pts:
        return 0.0
    residuals = [p['logit_q_norm'] - alpha * p['kappa_nearest'] - beta * p['logKm1']
                 for p in all_pts]
    C_d = float(np.mean(residuals))
    print(f"  C_{ds_name} = {C_d:.4f} +/- {float(np.std(residuals)):.4f} ({len(residuals)} residuals)")
    return C_d


def get_layer_indices(n_layers, fracs):
    """Convert layer fractions to 0-indexed layer indices, capped at n_layers-1."""
    return [min(int(round(f * n_layers)) - 1, n_layers - 1) for f in fracs]


def process_model(model_cfg, ds_texts, ds_labels, K, ds_name):
    """Generate kappa/q_norm cache for one model on one dataset."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    short = model_cfg["short"]
    cache_path = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name}_{short}.json")
    if os.path.exists(cache_path):
        print(f"  Cache exists: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {model_cfg['hf_name']} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_name"])
    model = AutoModel.from_pretrained(model_cfg["hf_name"],
                                      torch_dtype=torch.float16).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model_cfg["n_layers"]
    layers = get_layer_indices(n_layers, model_cfg["layer_frac"])
    print(f"  Layer indices: {layers} (of {n_layers} layers)")

    logKm1 = float(np.log(K - 1))
    cache_points = []
    texts_sub, labels_arr = subsample_per_class(ds_texts, ds_labels, K)

    for layer_idx in layers:
        print(f"    Layer {layer_idx}...", flush=True)
        embs = extract_mean_pool_at_layer(model, tokenizer, device, texts_sub, layer_idx)
        kappa, kappa_min = compute_kappa(embs, labels_arr[:len(embs)], K)
        q_norm = compute_q_norm(embs, labels_arr[:len(embs)], K)
        if q_norm is None:
            print("      SKIP: NaN")
            continue
        logit_q = float(scipy_logit(q_norm))
        pt = {"model": short, "dataset": ds_name, "layer": layer_idx,
              "K": K, "q": q_norm, "kappa_nearest": kappa, "kappa_min": kappa_min,
              "logit_q": logit_q, "logKm1": logKm1}
        cache_points.append(pt)
        print(f"      kappa={kappa:.4f}, q_norm={q_norm:.3f}, logit={logit_q:.3f}")

    del model
    torch.cuda.empty_cache()

    with open(cache_path, 'w') as f:
        json.dump(cache_points, f, indent=2)
    print(f"  Saved: {cache_path}")
    return cache_points


def main():
    print("=" * 70)
    print("EXPANDED BLIND OOD PREDICTION")
    print(f"alpha={ALPHA_DS:.4f}, beta={BETA_DS:.4f} (from 12-arch LOAO, frozen)")
    print(f"Test models: {[m['short'] for m in TEST_MODELS]}")
    print("=" * 70)

    # Step 1: Estimate C_d from partial caches
    print("\nEstimating C_d from partial archs (excluding test models)...")
    C_d_values = {
        "banking77": estimate_Cd("banking77", ALPHA_DS, BETA_DS),
        "amazon_massive": estimate_Cd("amazon_massive", ALPHA_DS, BETA_DS),
    }

    # Step 2: Load datasets once
    print("\nLoading datasets...")
    load_fns = {"banking77": load_banking77, "amazon_massive": load_amazon_massive}
    ds_data = {}
    for ds_name in ["banking77", "amazon_massive"]:
        texts, labels, K = load_fns[ds_name]()
        ds_data[ds_name] = (texts, labels, K)
        print(f"  {ds_name}: K={K}, n={len(texts)}")

    # Step 3: Process each model x dataset
    all_ood_pts = []
    for model_cfg in TEST_MODELS:
        print(f"\n{'=' * 50}")
        print(f"Model: {model_cfg['short']}")
        for ds_name in ["banking77", "amazon_massive"]:
            texts, labels, K = ds_data[ds_name]
            pts = process_model(model_cfg, texts, labels, K, ds_name)
            all_ood_pts.extend(pts)

    # Step 4: Include SmolLM2 results (already cached)
    print("\nLoading SmolLM2-1.7B (already cached)...")
    for ds_name in ["banking77", "amazon_massive"]:
        cache_path = os.path.join(RESULTS_DIR,
                                  f"kappa_near_cache_{ds_name}_SmolLM2-1.7B.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                pts = json.load(f)
            all_ood_pts.extend(pts)
            print(f"  {ds_name}: {len(pts)} pts")

    # Step 5: Blind prediction on all points
    print("\n" + "=" * 70)
    print(f"COMBINED BLIND OOD PREDICTION ({len(all_ood_pts)} points total)")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Dataset':<15} {'Layer':<6} {'kappa':<8} "
          f"{'q_norm':<7} {'obs':<8} {'pred':<8} {'err':<6}")

    obs_logits, pred_logits = [], []
    per_model_results = {}
    for pt in sorted(all_ood_pts, key=lambda x: (x['model'], x['dataset'], x['layer'])):
        C_d = C_d_values.get(pt['dataset'], 0.0)
        pred = ALPHA_DS * pt['kappa_nearest'] + BETA_DS * pt['logKm1'] + C_d
        obs = pt['logit_q']
        obs_logits.append(obs)
        pred_logits.append(pred)
        m = pt['model']
        if m not in per_model_results:
            per_model_results[m] = {'obs': [], 'pred': []}
        per_model_results[m]['obs'].append(obs)
        per_model_results[m]['pred'].append(pred)
        print(f"  {pt['model']:<20} {pt['dataset']:<15} L{pt['layer']:<5} "
              f"{pt['kappa_nearest']:<8.4f} {pt['q']:<7.3f} "
              f"{obs:<8.3f} {pred:<8.3f} {obs - pred:<6.3f}")

    print("\nPer-model summary:")
    for m, d in per_model_results.items():
        if len(d['obs']) >= 3:
            r_m, _ = pearsonr(d['obs'], d['pred'])
            mae_m = float(np.mean(np.abs(np.array(d['obs']) - np.array(d['pred']))))
            print(f"  {m}: r={r_m:.4f}, MAE={mae_m:.4f}, n={len(d['obs'])}")

    r, pv = pearsonr(obs_logits, pred_logits)
    mae = float(np.mean(np.abs(np.array(obs_logits) - np.array(pred_logits))))
    n = len(obs_logits)
    print(f"\nCombined: r={r:.4f} (p={pv:.4f}), MAE={mae:.4f}, n={n}")
    threshold = 0.80
    status = "PASS" if r >= threshold else "FAIL"
    print(f"Pre-registered threshold r>={threshold}: {status}")

    output = {
        "experiment": "expanded_blind_ood_prediction",
        "design": ("Blind OOD: alpha/beta from 12-arch LOAO, C_d from partial archs. "
                   "Test models: Llama-3.2-1B, gemma-3-1b + SmolLM2-1.7B (already done). "
                   "Datasets: banking77 (K=77), amazon_massive (K=60)."),
        "training_params": {"alpha": ALPHA_DS, "beta": BETA_DS},
        "C_d_values": C_d_values,
        "test_models": [m["short"] for m in TEST_MODELS] + ["SmolLM2-1.7B"],
        "ood_datasets": ["banking77 (K=77)", "amazon_massive (K=60)"],
        "all_ood_points": all_ood_pts,
        "per_model_results": {m: {"r": float(pearsonr(d['obs'], d['pred'])[0])
                                  if len(d['obs']) >= 3 else None,
                                  "mae": float(np.mean(np.abs(
                                      np.array(d['obs']) - np.array(d['pred']))))}
                              for m, d in per_model_results.items()},
        "combined": {
            "pearson_r": float(r),
            "pearson_p": float(pv),
            "mae": float(mae),
            "n_points": n,
            "pass_threshold_0.80": bool(r >= 0.80),
        },
        "verdict": "PASS" if r >= 0.80 else "FAIL",
    }
    out_path = os.path.join(RESULTS_DIR, "cti_expanded_blind_ood.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
