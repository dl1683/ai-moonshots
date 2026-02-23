"""
Encoder OOD test (pre-registered in PREREGISTRATION_encoder_ood.md):
- New encoder: FacebookAI/roberta-base (NOT in BERT/DeBERTa/BGE calibration set)
- New dataset: banking77 (K=77) -- never used in encoder calibration
- Uses mean-pool at proportional layers [3, 6, 9, 12]

Pre-registered success criteria:
1. alpha_roberta in [5.0, 10.0] (encoder family interval)
2. within-RoBERTa r(kappa, logit_q) on banking77 >= 0.90
"""
import json
import os
import glob
import numpy as np
from scipy.stats import pearsonr
from scipy.special import logit as scipy_logit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_SHORT = "roberta-base"
MODEL_NAME = "FacebookAI/roberta-base"
LAYER_INDICES = [3, 6, 9, 12]  # proportional for 12-layer RoBERTa

# Pre-registered encoder interval
ALPHA_ENC_LOW = 5.0
ALPHA_ENC_HIGH = 10.0
R_WITHIN_THRESHOLD = 0.90


def extract_mean_pool_at_layer(model, tokenizer, device, texts, layer_idx):
    import torch
    all_embs = []
    for b_start in range(0, len(texts), 32):
        batch = texts[b_start:b_start + 32]
        tok = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**tok, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx].float()  # RoBERTa: 0=embed, 1-12=layers
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


def get_encoder_loao_alpha():
    """Compute alpha from 3-arch encoder LOAO (bert, deberta, bge) on 3 datasets."""
    encoder_models = ['bert-base-uncased', 'deberta-base', 'bge-base-v1.5']
    datasets = ['dbpedia', 'agnews', '20newsgroups']
    all_pts = []
    for m in encoder_models:
        for ds in datasets:
            p = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds}_{m}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                pts = json.load(f)
            for pt in pts:
                K = pt['K']
                q_raw = pt['q']
                q_norm = float(np.clip((q_raw - 1.0/K)/(1.0-1.0/K), 0.001, 0.999))
                pt['q_norm'] = q_norm
                pt['logit_q_norm'] = float(scipy_logit(q_norm))
                if 'logKm1' not in pt:
                    pt['logKm1'] = float(np.log(K-1))
            all_pts.extend(pts)
    kappa = np.array([p['kappa_nearest'] for p in all_pts])
    logKm1 = np.array([p['logKm1'] for p in all_pts])
    logit_q = np.array([p['logit_q_norm'] for p in all_pts])
    X = np.column_stack([kappa, logKm1, np.ones(len(all_pts))])
    coeffs, _, _, _ = np.linalg.lstsq(X, logit_q, rcond=None)
    alpha_enc, beta_enc, C_enc = coeffs
    print(f"Encoder LOAO params: alpha={alpha_enc:.4f}, beta={beta_enc:.4f}, C={C_enc:.4f}")
    print(f"  n={len(all_pts)} pts from {len(encoder_models)} encoders x {len(datasets)} datasets")
    return float(alpha_enc), float(beta_enc), float(C_enc)


def get_banking77_Cd_decoder():
    """Get C_d for banking77 from decoder partial caches (same as in OOD test)."""
    return 0.0167  # from cti_smollm2_ood_prediction.json


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder

    print("=" * 60)
    print("ENCODER OOD TEST (PRE-REGISTERED)")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: banking77 (K=77)")
    print(f"Success: alpha in [{ALPHA_ENC_LOW}, {ALPHA_ENC_HIGH}], within-r >= {R_WITHIN_THRESHOLD}")
    print("=" * 60)

    # Step 1: Compute encoder LOAO parameters
    print("\nStep 1: Encoder LOAO parameters")
    alpha_enc, beta_enc, C_enc = get_encoder_loao_alpha()

    # Step 2: Load banking77
    print("\nStep 2: Loading banking77...")
    ds = load_dataset("banking77", split="train", trust_remote_code=False)
    texts_raw = [x["text"] for x in ds]
    labels_raw = [x["label"] for x in ds]
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

    # Step 3: Generate RoBERTa mean-pool embeddings
    print("\nStep 3: Generating RoBERTa-base cache...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    cache_points = []
    for layer_idx in LAYER_INDICES:
        print(f"  Layer {layer_idx}...", flush=True)
        embs = extract_mean_pool_at_layer(model, tokenizer, device, texts_sub, layer_idx)
        kappa, kappa_min = compute_kappa(embs, labels_arr[:len(embs)], K)
        q_norm = compute_q_norm(embs, labels_arr[:len(embs)], K)
        if q_norm is None:
            print("    SKIP: NaN")
            continue
        logit_q = float(scipy_logit(q_norm))
        pt = {"model": MODEL_SHORT, "dataset": "banking77", "layer": layer_idx,
              "K": K, "q": q_norm, "kappa_nearest": kappa, "kappa_min": kappa_min,
              "logit_q": logit_q, "logKm1": logKm1}
        cache_points.append(pt)
        print(f"    kappa={kappa:.4f}, q_norm={q_norm:.3f}, logit={logit_q:.3f}")

    del model
    torch.cuda.empty_cache()

    # Save cache
    cache_path = os.path.join(RESULTS_DIR, f"kappa_near_cache_banking77_{MODEL_SHORT}.json")
    with open(cache_path, 'w') as f:
        json.dump(cache_points, f, indent=2)
    print(f"  Saved: {cache_path}")

    # Step 4: Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    kappa_arr = np.array([p['kappa_nearest'] for p in cache_points])
    logKm1_arr = np.array([p['logKm1'] for p in cache_points])
    logit_arr = np.array([p['logit_q'] for p in cache_points])

    # Check 1: Independently fitted alpha
    X = np.column_stack([kappa_arr, logKm1_arr, np.ones(len(cache_points))])
    coeffs, _, _, _ = np.linalg.lstsq(X, logit_arr, rcond=None)
    alpha_rob, beta_rob, C_rob = coeffs
    pr1 = ALPHA_ENC_LOW <= alpha_rob <= ALPHA_ENC_HIGH
    print(f"\nPR1 (alpha in [{ALPHA_ENC_LOW}, {ALPHA_ENC_HIGH}]): {'PASS' if pr1 else 'FAIL'}")
    print(f"  alpha_roberta={alpha_rob:.4f}, beta={beta_rob:.4f}, C={C_rob:.4f}")

    # Check 2: Within-model functional form
    if len(cache_points) >= 3:
        r_within, _ = pearsonr(kappa_arr, logit_arr)
    else:
        r_within = 0.0
    pr2 = r_within >= R_WITHIN_THRESHOLD
    print(f"\nPR2 (within-r >= {R_WITHIN_THRESHOLD}): {'PASS' if pr2 else 'FAIL'}")
    print(f"  r(kappa, logit_q) = {r_within:.4f}")

    # Check 3: Blind prediction with encoder LOAO params + decoder C_d
    C_d_banking77 = get_banking77_Cd_decoder()
    pred_blind = alpha_enc * kappa_arr + beta_enc * logKm1_arr + C_d_banking77
    r_blind = float(pearsonr(logit_arr, pred_blind)[0]) if len(cache_points) >= 3 else 0.0
    mae_blind = float(np.mean(np.abs(logit_arr - pred_blind)))
    print(f"\nSecondary blind prediction (encoder alpha + decoder C_d):")
    print(f"  r={r_blind:.4f}, MAE={mae_blind:.4f}")

    # Show data points
    print(f"\n{'Layer':<8} {'kappa':<8} {'q_norm':<8} {'logit_obs':<12} {'logit_pred_blind':<16}")
    for pt, pred in zip(cache_points, pred_blind):
        print(f"  L{pt['layer']:<6} {pt['kappa_nearest']:<8.4f} {pt['q']:<8.3f} "
              f"{pt['logit_q']:<12.3f} {pred:<12.3f}")

    overall_pass = pr1 and pr2
    print(f"\nOVERALL PRE-REGISTERED: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  PR1 (alpha in interval): {'PASS' if pr1 else 'FAIL'}")
    print(f"  PR2 (within-r >= 0.90): {'PASS' if pr2 else 'FAIL'}")

    output = {
        "experiment": "encoder_ood_roberta_banking77",
        "preregistration": "src/PREREGISTRATION_encoder_ood.md",
        "model": MODEL_NAME,
        "dataset": "banking77",
        "encoder_loao_params": {"alpha": alpha_enc, "beta": beta_enc, "C": C_enc},
        "roberta_fit": {"alpha": float(alpha_rob), "beta": float(beta_rob), "C": float(C_rob)},
        "roberta_points": cache_points,
        "evaluation": {
            "r_within": float(r_within),
            "r_blind": float(r_blind),
            "mae_blind": float(mae_blind),
            "pr1_alpha_pass": bool(pr1),
            "pr2_r_pass": bool(pr2),
            "overall_pass": bool(overall_pass),
        },
        "verdict": "PASS" if overall_pass else "FAIL",
    }
    out_path = os.path.join(RESULTS_DIR, "cti_encoder_ood_roberta.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
