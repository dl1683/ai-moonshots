#!/usr/bin/env python -u
"""
FROZEN-BACKBONE NC-LOSS CAUSAL TEST (DBpedia K=14)
==================================================

Fastest clean causal test per Codex guidance (task bc21248).

DESIGN FIXES vs. cti_nc_loss_quick.py:
  FIX 1: class means DIFFERENTIABLE (batch-scatter, no torch.no_grad EMA)
          -> L_ETF and L_margin propagate gradients to proj_head
  FIX 2: q/kappa measured in PROJ HEAD space (same space NC is optimized)
          -> No measurement/optimization space mismatch
  FIX 3: DBpedia K=14 (not CIFAR) -> no coarse-label confound, no crowding

PROTOCOL:
  - Pre-extract DBpedia embeddings from frozen pythia-160m layer 11
  - Train: proj_head (Linear 768->256) + ce_head (Linear 256->14)
  - 4 arms: CE, CE+L_within, CE+full_NC, shuffled_NC
  - 5 seeds, 50 epochs each
  - Report: delta_q and delta_kappa (NC - CE) per seed

PRE-REGISTERED HYPOTHESES:
  H1 (direction): mean delta_q(full_NC) > mean delta_q(CE) and
                  delta_q(NC) > 0 for >= 3/5 seeds
  H2 (control):   mean delta_q(shuffled_NC) <= mean delta_q(CE)
                  (shuffling breaks the causal effect)
  H3 (specificity): delta_kappa(NC) > delta_kappa(CE) for >= 4/5 seeds

Pre-registered: commit before running (after this write).
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCH = "pythia-160m"
MODEL_ID = "EleutherAI/pythia-160m"
BEST_LAYER = 11      # layer 11 = best for DBpedia K=14 per prior results
DATASET = "dbpedia_14"
K = 14
N_TRAIN = 3000       # 3000 total training samples (~214 per class)
N_TEST = 1400        # 1400 total test samples (100 per class)
MAX_LENGTH = 128

PROJ_DIM = 256
N_EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
N_SEEDS = 5
LAMBDA_NC = 0.1      # weight on NC loss term
MARGIN = 1.0         # margin for L_margin term

RESULT_PATH = RESULTS_DIR / "cti_frozen_backbone_nc.json"


# ────────────────────────────────────────────────────────
# 1. Embedding extraction (done once, cached in memory)
# ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model_id, best_layer, dataset_name, n_train, n_test, max_length):
    """Return (Z_train, y_train, Z_test, y_test) as numpy arrays."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import random

    print(f"Loading {model_id} for embedding extraction...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    lm.eval()

    print(f"Loading {dataset_name}...", flush=True)
    ds_train = load_dataset(dataset_name, split="train")
    ds_test  = load_dataset(dataset_name, split="test")

    random.seed(42)
    np.random.seed(42)

    # Stratified sample N_TRAIN from train, N_TEST from test
    def stratified_sample(ds, n_total, label_col="label", text_col="content"):
        labels = np.array(ds[label_col])
        classes = np.unique(labels)
        n_per_class = n_total // len(classes)
        idxs = []
        for c in classes:
            c_idxs = np.where(labels == c)[0]
            chosen = np.random.choice(c_idxs, min(n_per_class, len(c_idxs)), replace=False)
            idxs.extend(chosen.tolist())
        random.shuffle(idxs)
        texts = [ds[text_col][i] for i in idxs]
        lbls  = [labels[i] for i in idxs]
        return texts, np.array(lbls)

    train_texts, train_labels = stratified_sample(ds_train, n_train)
    test_texts,  test_labels  = stratified_sample(ds_test, n_test)

    def embed(texts, layer_idx, batch_size=32):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(DEVICE)
            out = lm(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx + 1]  # 0=embed, 1..n=layers
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
            all_embs.append(pooled.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    print(f"Extracting train embeddings (layer {best_layer})...", flush=True)
    Z_train = embed(train_texts, best_layer)
    print(f"Extracting test embeddings...", flush=True)
    Z_test  = embed(test_texts,  best_layer)

    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Z_train={Z_train.shape}, Z_test={Z_test.shape}", flush=True)
    return Z_train, train_labels, Z_test, test_labels


# ────────────────────────────────────────────────────────
# 2. Head architecture
# ────────────────────────────────────────────────────────

def make_heads(d_model, proj_dim, k):
    """proj_head: d_model -> proj_dim (BN), ce_head: proj_dim -> K."""
    proj_head = nn.Sequential(
        nn.Linear(d_model, proj_dim, bias=False),
        nn.BatchNorm1d(proj_dim),
    ).to(DEVICE)
    ce_head = nn.Linear(proj_dim, k).to(DEVICE)
    return proj_head, ce_head


# ────────────────────────────────────────────────────────
# 3. NC loss (DIFFERENTIABLE class means)
# ────────────────────────────────────────────────────────

def differentiable_class_means(z, y, k):
    """
    Compute class means from current batch in a differentiable way.
    z: (N, D) float32
    y: (N,) int64
    Returns mu: (K, D)
    """
    D = z.shape[1]
    mu = torch.zeros(k, D, device=z.device, dtype=z.dtype)
    count = torch.zeros(k, device=z.device, dtype=z.dtype)
    for c in range(k):
        mask = (y == c)
        if mask.sum() > 0:
            mu[c] = z[mask].mean(0)
            count[c] = 1.0
    # For empty classes, use zeros — won't contribute meaningfully
    return mu, count


def nc_loss(z, y, k, arm, class_perm=None):
    """
    FIX: class means computed differentiably from batch — gradients flow to proj_head.

    z:   (N, D) unit-norm features from proj_head
    y:   (N,) labels
    arm: 'within_only' | 'full_nc' | 'shuffled_nc'
    class_perm: permutation for shuffled_nc arm
    """
    y_nc = class_perm[y] if (arm == "shuffled_nc" and class_perm is not None) else y

    # Differentiable class means
    mu, count = differentiable_class_means(z, y_nc, k)

    # L_within: pull samples toward their class mean
    mu_yi = mu[y_nc]
    L_within = ((z - mu_yi) ** 2).mean()

    if arm == "within_only" or arm == "shuffled_nc":
        return L_within

    # L_ETF: push means toward equiangular tight frame
    mu_n = F.normalize(mu, dim=1)        # (K, D)
    G = mu_n @ mu_n.t()                  # (K, K) Gram matrix
    # Target: G_ETF_{ij} = delta_ij + (1-delta_ij)*(-1/(K-1))
    # Equivalently: K/(K-1)*I - 1/(K-1)*ones
    target_off = -1.0 / (k - 1)
    G_etf = torch.eye(k, device=z.device) * (1.0 - target_off) + target_off
    L_ETF = ((G - G_etf) ** 2).mean()

    # L_margin: push means apart (minimum pairwise distance >= MARGIN)
    # Only active classes
    active = (count > 0).nonzero(as_tuple=False).squeeze(1)
    if len(active) >= 2:
        mu_active = mu[active]
        dists = torch.cdist(mu_active.unsqueeze(0), mu_active.unsqueeze(0)).squeeze(0)
        inf_diag = torch.eye(len(active), device=z.device) * 1e6
        L_margin = F.softplus(MARGIN - (dists + inf_diag).min())
    else:
        L_margin = torch.tensor(0.0, device=z.device)

    return L_within + 0.5 * L_ETF + 0.5 * L_margin


# ────────────────────────────────────────────────────────
# 4. Evaluation: q and kappa in PROJ HEAD space
# ────────────────────────────────────────────────────────

def compute_q_and_kappa_head(proj_head, Z_train_t, y_train, Z_test_t, y_test, K):
    """
    FIX: measure in proj_head output space, same space NC is optimized.
    FIX2: fit 1-NN on TRAIN set, evaluate on TEST set (not same set!)
    """
    proj_head.eval()
    with torch.no_grad():
        train_chunks = torch.split(Z_train_t.to(DEVICE), 512)
        Z_train_proj = torch.cat([proj_head(c) for c in train_chunks], dim=0).cpu().numpy()
        test_chunks = torch.split(Z_test_t.to(DEVICE), 512)
        Z_test_proj = torch.cat([proj_head(c) for c in test_chunks], dim=0).cpu().numpy()
    proj_head.train()

    from sklearn.neighbors import KNeighborsClassifier

    # kappa_nearest in proj_head space (on TRAIN set)
    classes = np.unique(y_train)
    d = Z_train_proj.shape[1]
    means = {c: Z_train_proj[y_train == c].mean(0) for c in classes}
    within_var = np.mean([np.mean(np.sum((Z_train_proj[y_train == c] - means[c])**2, axis=1))
                          for c in classes])
    sigma_W = np.sqrt(within_var / d)
    pairwise_dists = [np.linalg.norm(means[classes[i]] - means[classes[j]])
                      for i in range(len(classes)) for j in range(i+1, len(classes))]
    delta_min = min(pairwise_dists)
    kappa = float(delta_min / (sigma_W * np.sqrt(d) + 1e-10))

    # 1-NN: fit on TRAIN, score on TEST
    knn = KNeighborsClassifier(1, metric="euclidean", n_jobs=-1)
    knn.fit(Z_train_proj, y_train)
    acc = float(knn.score(Z_test_proj, y_test))
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)

    return float(q), float(kappa)


# ────────────────────────────────────────────────────────
# 5. Training loop
# ────────────────────────────────────────────────────────

def train_arm(seed, arm, Z_train, y_train, Z_test, y_test, d_model):
    """Train one arm (CE / within_only / full_nc / shuffled_nc) for N_EPOCHS."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    proj_head, ce_head = make_heads(d_model, PROJ_DIM, K)
    params = list(proj_head.parameters()) + list(ce_head.parameters())
    optimizer = optim.Adam(params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()

    # Class permutation for shuffled control
    rng = np.random.RandomState(seed + 9999)
    class_perm = torch.tensor(rng.permutation(K), dtype=torch.long, device=DEVICE)

    # Build data tensors
    Z_train_t = torch.tensor(Z_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    Z_test_t  = torch.tensor(Z_test,  dtype=torch.float32)

    N = len(Z_train_t)
    idx_all = np.arange(N)

    checkpoints = []

    for epoch in range(1, N_EPOCHS + 1):
        proj_head.train(); ce_head.train()
        np.random.shuffle(idx_all)
        e_ce = e_nc = n_b = 0

        for start in range(0, N, BATCH_SIZE):
            bidx = idx_all[start:start+BATCH_SIZE]
            zb = Z_train_t[bidx].to(DEVICE)
            yb = y_train_t[bidx].to(DEVICE)

            # Forward
            zp = proj_head(zb)               # (B, proj_dim)
            zp_n = F.normalize(zp, dim=1)    # unit-norm features for NC
            logits = ce_head(zp)             # CE uses raw (not norm) features
            loss_ce = ce_loss_fn(logits, yb)

            if arm in ("within_only", "full_nc", "shuffled_nc"):
                loss_nc = nc_loss(zp_n, yb, K, arm, class_perm)
                loss = loss_ce + LAMBDA_NC * loss_nc
            else:
                loss = loss_ce
                loss_nc = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            e_ce += loss_ce.item()
            e_nc += loss_nc.item() if hasattr(loss_nc, "item") else 0.0
            n_b  += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == N_EPOCHS or epoch == 1:
            q_val, kappa_val = compute_q_and_kappa_head(proj_head, Z_train_t, y_train, Z_test_t, y_test, K)
            print(f"  [seed={seed} arm={arm} ep={epoch:3d}] "
                  f"q={q_val:.4f} kappa={kappa_val:.4f} "
                  f"L_ce={e_ce/n_b:.4f} L_nc={e_nc/n_b:.4f}", flush=True)
            checkpoints.append({
                "epoch": epoch,
                "q": float(q_val),
                "kappa": float(kappa_val),
                "loss_ce": float(e_ce / n_b),
                "loss_nc": float(e_nc / n_b),
            })

    final = checkpoints[-1]
    return {
        "seed": int(seed),
        "arm": arm,
        "final_q": float(final["q"]),
        "final_kappa": float(final["kappa"]),
        "checkpoints": checkpoints,
    }


# ────────────────────────────────────────────────────────
# 6. Main
# ────────────────────────────────────────────────────────

def main():
    print("FROZEN-BACKBONE NC-LOSS CAUSAL TEST", flush=True)
    print(f"Dataset: {DATASET} K={K}, Arch: {ARCH} layer {BEST_LAYER}", flush=True)
    print(f"Arms: CE, within_only, full_nc, shuffled_nc | Seeds: {N_SEEDS}", flush=True)
    print(f"Pre-registration: BEFORE running (commit then execute)", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    t0 = time.time()

    # ── Extract embeddings ──
    Z_train, y_train, Z_test, y_test = extract_embeddings(
        MODEL_ID, BEST_LAYER, DATASET, N_TRAIN, N_TEST, MAX_LENGTH
    )
    d_model = Z_train.shape[1]
    print(f"Embeddings ready: d={d_model}, N_train={len(Z_train)}, N_test={len(Z_test)}", flush=True)

    ARMS = ["ce", "within_only", "full_nc", "shuffled_nc"]
    results_by_arm = {arm: [] for arm in ARMS}

    for seed in range(N_SEEDS):
        print(f"\n{'='*60}", flush=True)
        print(f"SEED {seed}", flush=True)
        for arm in ARMS:
            print(f"\n  --- {arm} ---", flush=True)
            res = train_arm(seed, arm, Z_train, y_train, Z_test, y_test, d_model)
            results_by_arm[arm].append(res)

    # ── Summary ──
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY (final q and kappa per arm)", flush=True)
    print(f"{'='*60}", flush=True)

    arm_means = {}
    for arm in ARMS:
        qs     = [r["final_q"]     for r in results_by_arm[arm]]
        kappas = [r["final_kappa"] for r in results_by_arm[arm]]
        arm_means[arm] = {
            "mean_q":     float(np.mean(qs)),
            "std_q":      float(np.std(qs)),
            "mean_kappa": float(np.mean(kappas)),
            "std_kappa":  float(np.std(kappas)),
            "all_q":      [float(x) for x in qs],
            "all_kappa":  [float(x) for x in kappas],
        }
        print(f"  {arm:15s}: q={np.mean(qs):.4f}±{np.std(qs):.4f}, "
              f"kappa={np.mean(kappas):.4f}±{np.std(kappas):.4f}", flush=True)

    # ── Hypothesis tests ──
    delta_q_nc     = [results_by_arm["full_nc"][s]["final_q"]     - results_by_arm["ce"][s]["final_q"]     for s in range(N_SEEDS)]
    delta_q_shuf   = [results_by_arm["shuffled_nc"][s]["final_q"] - results_by_arm["ce"][s]["final_q"]     for s in range(N_SEEDS)]
    delta_k_nc     = [results_by_arm["full_nc"][s]["final_kappa"] - results_by_arm["ce"][s]["final_kappa"] for s in range(N_SEEDS)]
    delta_k_within = [results_by_arm["within_only"][s]["final_kappa"] - results_by_arm["ce"][s]["final_kappa"] for s in range(N_SEEDS)]

    h1_pass = (np.mean(delta_q_nc) > np.mean([r["final_q"] for r in results_by_arm["ce"]]) - np.mean([r["final_q"] for r in results_by_arm["ce"]])) and sum(d > 0 for d in delta_q_nc) >= 3
    # Simpler H1: mean delta_q(NC) > 0 AND >= 3/5 positive
    h1_pass = (np.mean(delta_q_nc) > 0) and (sum(d > 0 for d in delta_q_nc) >= 3)
    h2_pass = np.mean(delta_q_shuf) <= 0.0  # shuffled control shows no improvement
    h3_pass = sum(d > 0 for d in delta_k_nc) >= 4

    print(f"\nHYPOTHESES:", flush=True)
    print(f"  H1 (NC improves q, >=3/5 seeds): {'PASS' if h1_pass else 'FAIL'}", flush=True)
    print(f"    mean delta_q(NC)={np.mean(delta_q_nc):.4f}, n_pos={sum(d>0 for d in delta_q_nc)}/5", flush=True)
    print(f"  H2 (shuffled control <= 0):       {'PASS' if h2_pass else 'FAIL'}", flush=True)
    print(f"    mean delta_q(shuf)={np.mean(delta_q_shuf):.4f}", flush=True)
    print(f"  H3 (NC improves kappa, >=4/5):    {'PASS' if h3_pass else 'FAIL'}", flush=True)
    print(f"    mean delta_kappa(NC)={np.mean(delta_k_nc):.4f}, n_pos={sum(d>0 for d in delta_k_nc)}/5", flush=True)

    elapsed = time.time() - t0

    out = {
        "experiment": "cti_frozen_backbone_nc",
        "arch": ARCH,
        "dataset": DATASET,
        "K": K,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "proj_dim": PROJ_DIM,
        "n_epochs": N_EPOCHS,
        "n_seeds": N_SEEDS,
        "lambda_nc": LAMBDA_NC,
        "design": {
            "fix1": "differentiable class means (batch scatter, no detach)",
            "fix2": "q/kappa measured in proj_head space (same as NC optimization)",
            "fix3": "DBpedia K=14 (no CIFAR coarse-label confound)",
        },
        "pre_registered_hypotheses": {
            "H1": "mean delta_q(full_NC) > 0 AND >= 3/5 seeds positive",
            "H2": "mean delta_q(shuffled_NC) <= 0",
            "H3": "delta_kappa(NC) > 0 for >= 4/5 seeds",
        },
        "arm_summary": arm_means,
        "deltas": {
            "delta_q_nc":     [float(x) for x in delta_q_nc],
            "delta_q_shuf":   [float(x) for x in delta_q_shuf],
            "delta_kappa_nc": [float(x) for x in delta_k_nc],
            "delta_kappa_within": [float(x) for x in delta_k_within],
        },
        "scorecard": {
            "H1_pass": bool(h1_pass),
            "H2_pass": bool(h2_pass),
            "H3_pass": bool(h3_pass),
            "all_pass": bool(h1_pass and h2_pass and h3_pass),
        },
        "results_by_arm": results_by_arm,
        "elapsed_s": float(elapsed),
    }

    with open(RESULT_PATH, "w", encoding="ascii") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULT_PATH}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
