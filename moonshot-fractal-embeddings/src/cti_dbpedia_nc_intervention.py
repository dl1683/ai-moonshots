#!/usr/bin/env python -u
"""
DBpedia NC-Loss Head Intervention (Clean Causal Test)
======================================================
Pre-registered: results/cti_dbpedia_nc_prereg.json (2026-02-22T22:44:44Z)

Design:
  - Frozen Pythia backbone (layer 12): embeddings extracted once, cached
  - Trainable projection head: Linear(d, 256) -> LayerNorm -> ReLU -> Linear(256, 256)
  - CE classifier: Linear(256, K=14) on proj_head outputs
  - Full-batch training (7000 samples) for stable class mean estimates

Arms (3 seeds each):
  A. ce:           CE only (baseline)
  B. nc_full:      CE + L_NC (L_within + L_ETF + L_margin), DIFFERENTIABLE class means
  C. nc_within:    CE + L_within only (collapse test)
  D. shuffled_nc:  CE + NC-loss with shuffled class labels (control)

Critical fixes vs previous NC-loss experiments:
  1. q and kappa measured on proj_head OUTPUTS (same space as optimization)
  2. class_means computed DIFFERENTIABLY (no detach/no_grad for ETF)
  3. Non-crowded regime: DBpedia K=14 (kappa~0.45-0.61, q~0.76-0.83)
  4. Full-batch gradient for stable class means

Pre-registered prediction: alpha=1.477
  - nc_full: delta_q > 0.02, delta_kappa > 0
  - ratio delta_logit_q / delta_kappa in [1.034, 1.920] (alpha +/- 30%)
  - shuffled_nc: weaker effect than nc_full
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import pearsonr
from scipy.special import logit as sp_logit

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-registered constants (locked before data loading)
ALPHA_PREREG = 1.477
K = 14
N_PER_CLASS = 500  # 500 per class = 7000 total
PROJ_DIM = 256
N_SEEDS = 3
N_EPOCHS = 300
LR = 3e-3
LAMBDA_NC = 0.5
MARGIN = 0.5
ARMS = ["ce", "nc_full", "nc_within", "shuffled_nc"]
MODELS = [
    ("EleutherAI/pythia-160m", 12),   # (model_name, layer_idx)
    ("EleutherAI/pythia-410m", 16),
]


# -------------------------------------------------------
# Data extraction
# -------------------------------------------------------

def extract_frozen_embeddings(model_name, layer_idx, n_per_class, K):
    """Extract frozen backbone embeddings from Pythia, DBpedia K=14."""
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset

    print(f"  Extracting {model_name} layer {layer_idx} embeddings...")
    ds = load_dataset("fancyzhx/dbpedia_14", split="train")

    # Balance: n_per_class per class
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, item in enumerate(ds):
        if len(class_indices[item["label"]]) < n_per_class:
            class_indices[item["label"]].append(i)
    indices = []
    for c in sorted(class_indices.keys()):
        indices.extend(class_indices[c][:n_per_class])
    ds_sub = ds.select(indices)

    labels = np.array([ds_sub[i]["label"] for i in range(len(ds_sub))])
    texts = [ds_sub[i]["content"][:512] for i in range(len(ds_sub))]

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True
    ).to(DEVICE).eval()

    batch_size = 32
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=128, padding=True).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states  # (n_layers+1, B, T, d)
        h = hs[layer_idx]  # (B, T, d)
        # Mean pool
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1)  # (B, d)
        all_embs.append(emb.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()

    X = np.vstack(all_embs)  # (N, d)
    print(f"    X.shape={X.shape}, K={len(np.unique(labels))}")
    return X, labels


# -------------------------------------------------------
# Projection head
# -------------------------------------------------------

def make_proj_head(d_in, d_out=PROJ_DIM):
    return nn.Sequential(
        nn.Linear(d_in, d_out),
        nn.LayerNorm(d_out),
        nn.ReLU(),
        nn.Linear(d_out, d_out),
    ).to(DEVICE)


# -------------------------------------------------------
# NC loss (differentiable class means)
# -------------------------------------------------------

def compute_class_means_diff(z, y, K):
    """Differentiable class means (no detach)."""
    means = []
    for c in range(K):
        mask = (y == c)
        if mask.sum() > 0:
            means.append(z[mask].mean(0))
        else:
            means.append(z.new_zeros(z.shape[1]))
    return torch.stack(means)  # (K, d)


def nc_loss_full(z, y, K, class_perm=None):
    """Full NC loss with differentiable ETF and margin."""
    y_nc = class_perm[y] if class_perm is not None else y
    M = compute_class_means_diff(z, y_nc, K)  # (K, d)
    M_norm = F.normalize(M, dim=1)  # normalize for ETF computation

    # L_within: push samples to class mean
    mu_yi = M[y_nc]  # (N, d)
    L_within = ((z - mu_yi.detach()) ** 2).mean()
    # Keep detach here only for within (to prevent gradient explosion with z)
    # But for ETF we want real gradients:

    # L_ETF: push normalized centroids toward ETF
    G = M_norm @ M_norm.T  # (K, K)
    G_etf = (1.0 + 1.0/(K-1)) * torch.eye(K, device=z.device) \
            - (1.0/(K-1)) * torch.ones(K, K, device=z.device)
    L_ETF = ((G - G_etf) ** 2).sum() / (K ** 2)

    # L_margin: push minimum centroid distance above MARGIN
    dists = torch.cdist(M.unsqueeze(0), M.unsqueeze(0)).squeeze(0)
    dists = dists + torch.eye(K, device=z.device) * 1e6  # mask diagonal
    L_margin = F.softplus(MARGIN - dists.min())

    return L_within + 0.5 * L_ETF + 0.5 * L_margin


def nc_loss_within_only(z, y, K):
    """Only L_within (no ETF, no margin)."""
    M = compute_class_means_diff(z, y, K)
    mu_yi = M[y]
    return ((z - mu_yi.detach()) ** 2).mean()


# -------------------------------------------------------
# Metrics (kappa_nearest and q_norm) on proj outputs
# -------------------------------------------------------

def compute_kappa_q(X, y, K):
    """Compute kappa_nearest and q_norm from raw numpy arrays."""
    from sklearn.neighbors import KNeighborsClassifier
    classes = np.unique(y)
    if len(classes) < 2:
        return None, None

    mu = {c: X[y == c].mean(0) for c in classes if (y == c).sum() >= 2}
    if len(mu) < 2:
        return None, None

    # Pooled within-class std
    within_var = sum(np.sum((X[y == c] - mu[c])**2) for c in mu)
    n_total = sum((y == c).sum() for c in mu)
    sigma_W = float(np.sqrt(within_var / (n_total * X.shape[1])))
    if sigma_W < 1e-10:
        return None, None

    # kappa_nearest
    kappas = []
    for c in mu:
        dists = [np.linalg.norm(mu[c] - mu[j]) for j in mu if j != c]
        kappas.append(min(dists) / (sigma_W * np.sqrt(X.shape[1])))
    kappa = float(np.mean(kappas))

    # q_norm via 1-NN
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    if len(np.unique(y[tr])) < 2:
        return kappa, None
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
    knn.fit(X[tr], y[tr])
    acc = float(knn.score(X[te], y[te]))
    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    return kappa, q


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------

def train_arm(X_np, y_np, arm, seed, K, n_epochs=N_EPOCHS):
    """Train projection head + classifier for one arm and seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = X_np.shape[1]
    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_np, dtype=torch.long, device=DEVICE)

    proj = make_proj_head(d, PROJ_DIM)
    clf = nn.Linear(PROJ_DIM, K).to(DEVICE)

    opt = torch.optim.Adam(
        list(proj.parameters()) + list(clf.parameters()), lr=LR
    )

    # Shuffled labels for control arm
    if arm == "shuffled_nc":
        rng = np.random.default_rng(seed * 7 + 13)
        perm_arr = rng.permutation(K)
        class_perm = torch.tensor(perm_arr, dtype=torch.long, device=DEVICE)
    else:
        class_perm = None

    checkpoints = []
    log_epochs = [0, 50, 100, 150, 200, 250, 300]

    for epoch in range(n_epochs + 1):
        proj.train()
        clf.train()

        z = proj(X)  # (N, PROJ_DIM) — in the space we measure!
        logits = clf(z)
        L_ce = F.cross_entropy(logits, y)

        if arm == "ce":
            loss = L_ce
        elif arm == "nc_full":
            L_nc = nc_loss_full(z, y, K)
            loss = L_ce + LAMBDA_NC * L_nc
        elif arm == "nc_within":
            L_nc = nc_loss_within_only(z, y, K)
            loss = L_ce + LAMBDA_NC * L_nc
        elif arm == "shuffled_nc":
            L_nc = nc_loss_full(z, y, K, class_perm=class_perm)
            loss = L_ce + LAMBDA_NC * L_nc

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch in log_epochs:
            proj.eval()
            with torch.no_grad():
                z_eval = proj(X).cpu().numpy()
            kappa, q = compute_kappa_q(z_eval, y_np, K)
            if kappa is not None and q is not None:
                logit_q = float(np.clip(sp_logit(float(np.clip(q, 0.01, 0.99))), -10, 10))
                checkpoints.append({
                    "epoch": epoch,
                    "kappa": kappa,
                    "q": float(q),
                    "logit_q": logit_q,
                    "loss_ce": float(L_ce.item()),
                })
                if epoch % 50 == 0 or epoch == 0:
                    print(f"    [{arm} seed={seed} ep={epoch:3d}] "
                          f"q={q:.4f} kappa={kappa:.4f}")

    # Final measurement
    proj.eval()
    with torch.no_grad():
        z_final = proj(X).cpu().numpy()
    kappa_f, q_f = compute_kappa_q(z_final, y_np, K)
    return {
        "arm": arm,
        "seed": seed,
        "final_kappa": kappa_f,
        "final_q": float(q_f) if q_f is not None else None,
        "checkpoints": checkpoints,
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    print("=" * 65)
    print("DBpedia NC-Loss Head Intervention (Pre-registered)")
    print("=" * 65)
    print(f"alpha_prereg = {ALPHA_PREREG}")
    print(f"K={K}, n_per_class={N_PER_CLASS}, epochs={N_EPOCHS}, arms={ARMS}")
    print(f"Device: {DEVICE}")
    print()

    all_results = {}

    for model_name, layer_idx in MODELS:
        model_short = model_name.split("/")[-1]
        print(f"\n{'='*55}")
        print(f"Model: {model_short} (layer {layer_idx})")
        print(f"{'='*55}")

        # Cache path
        cache_path = RESULTS_DIR / f"dbpedia_nc_embs_{model_short}_l{layer_idx}.npz"
        if cache_path.exists():
            print(f"  Loading cached embeddings: {cache_path.name}")
            loaded = np.load(str(cache_path))
            X_np, y_np = loaded["X"], loaded["y"]
            print(f"  X.shape={X_np.shape}")
        else:
            X_np, y_np = extract_frozen_embeddings(model_name, layer_idx, N_PER_CLASS, K)
            np.savez(str(cache_path), X=X_np, y=y_np)
            print(f"  Saved embeddings to {cache_path.name}")

        # Baseline kappa/q on backbone
        kappa_backbone, q_backbone = compute_kappa_q(X_np, y_np, K)
        print(f"  Backbone: kappa={kappa_backbone:.4f}, q={q_backbone:.4f}")
        print()

        model_results = {"backbone_kappa": kappa_backbone, "backbone_q": float(q_backbone),
                         "arms": {}}

        for arm in ARMS:
            print(f"  --- ARM: {arm} ---")
            seed_results = []
            for seed in range(N_SEEDS):
                result = train_arm(X_np, y_np, arm, seed, K)
                seed_results.append(result)
            model_results["arms"][arm] = seed_results
            qs = [r["final_q"] for r in seed_results if r["final_q"] is not None]
            ks = [r["final_kappa"] for r in seed_results if r["final_kappa"] is not None]
            if qs:
                print(f"    DONE: mean_q={np.mean(qs):.4f}, mean_kappa={np.mean(ks):.4f}")
            print()

        all_results[model_short] = model_results

    # Analysis
    print("\n" + "=" * 65)
    print("RESULTS vs PRE-REGISTRATION")
    print("=" * 65)
    print(f"  alpha_prereg = {ALPHA_PREREG}")
    print()

    summary_rows = []

    for model_short, mres in all_results.items():
        arms = mres["arms"]
        print(f"Model: {model_short}")

        stats = {}
        for arm in ARMS:
            if arm not in arms:
                continue
            qs = [r["final_q"] for r in arms[arm] if r["final_q"] is not None]
            ks = [r["final_kappa"] for r in arms[arm] if r["final_kappa"] is not None]
            if qs:
                stats[arm] = {"mean_q": float(np.mean(qs)), "mean_kappa": float(np.mean(ks))}

        if "ce" not in stats:
            print("  MISSING CE baseline")
            continue

        ce_q = stats["ce"]["mean_q"]
        ce_k = stats["ce"]["mean_kappa"]
        ce_logit = float(sp_logit(float(np.clip(ce_q, 0.01, 0.99))))

        for arm in ["nc_full", "nc_within", "shuffled_nc"]:
            if arm not in stats:
                continue
            arm_q = stats[arm]["mean_q"]
            arm_k = stats[arm]["mean_kappa"]
            arm_logit = float(sp_logit(float(np.clip(arm_q, 0.01, 0.99))))

            dq = arm_q - ce_q
            dk = arm_k - ce_k
            dlogit = arm_logit - ce_logit
            ratio = dlogit / dk if abs(dk) > 1e-6 else float("nan")

            in_band = abs(ratio - ALPHA_PREREG) / ALPHA_PREREG < 0.30

            print(f"  {arm}: delta_q={dq:+.4f}, delta_kappa={dk:+.4f}, "
                  f"delta_logit={dlogit:+.4f}, ratio={ratio:.3f} "
                  f"{'PASS' if in_band else 'FAIL'}")

            summary_rows.append({
                "model": model_short,
                "arm": arm,
                "ce_q": ce_q,
                "ce_kappa": ce_k,
                "arm_q": arm_q,
                "arm_kappa": arm_k,
                "delta_q": dq,
                "delta_kappa": dk,
                "delta_logit_q": dlogit,
                "ratio_delta_logit_delta_kappa": ratio,
                "in_alpha_band": bool(in_band),
            })
        print()

    # Pass/fail verdict
    nc_rows = [r for r in summary_rows if r["arm"] == "nc_full"]
    n_pass_dq = sum(1 for r in nc_rows if r["delta_q"] > 0.02)
    n_pass_dk = sum(1 for r in nc_rows if r["delta_kappa"] > 0)
    n_pass_ratio = sum(1 for r in nc_rows if r["in_alpha_band"])

    print(f"NC_FULL pass count (out of {len(nc_rows)} models):")
    print(f"  delta_q > 0.02:   {n_pass_dq}/{len(nc_rows)}")
    print(f"  delta_kappa > 0:  {n_pass_dk}/{len(nc_rows)}")
    print(f"  ratio in band:    {n_pass_ratio}/{len(nc_rows)}")

    primary_pass = n_pass_dq == len(nc_rows) and n_pass_dk == len(nc_rows)
    print(f"\nPRIMARY PASS (all models): {'PASS' if primary_pass else 'FAIL'}")

    # Save
    out = {
        "experiment": "dbpedia_nc_head_intervention",
        "prereg_timestamp": "2026-02-22T22:44:44.960825+00:00",
        "alpha_prereg": ALPHA_PREREG,
        "K": K,
        "primary_pass": primary_pass,
        "summary_rows": summary_rows,
        "model_results": {k: {
            "backbone_kappa": v["backbone_kappa"],
            "backbone_q": v["backbone_q"],
            "arm_stats": {
                arm: {
                    "mean_q": float(np.mean([r["final_q"] for r in seeds if r["final_q"] is not None])),
                    "mean_kappa": float(np.mean([r["final_kappa"] for r in seeds if r["final_kappa"] is not None])),
                }
                for arm, seeds in v["arms"].items()
                if any(r["final_q"] is not None for r in seeds)
            }
        } for k, v in all_results.items()},
    }

    out_path = RESULTS_DIR / "cti_dbpedia_nc_intervention.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path.name}")


if __name__ == "__main__":
    main()
