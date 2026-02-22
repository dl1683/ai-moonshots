#!/usr/bin/env python -u
"""
cti_preregistered_causal_intervention.py

PREREGISTERED CAUSAL INTERVENTION — Feb 21, 2026

Question: Does DIRECTLY increasing kappa_nearest cause q to increase
          by the PREDICTED amount (Delta_logit_q = A * Delta_kappa)?

Design:
  - Frozen backbone: gpt-neo-125m (layer 12, d=768)
  - Dataset: 20newsgroups (K=20, ~600/class)
  - Projection head: Linear 768->64, trained from SAME random init
  - Arm CE:      Cross-entropy loss only
  - Arm Triplet: CE + hard-negative triplet loss (directly maximizes kappa)
  - Both arms: same random seeds, same training steps, same architecture

Preregistered prediction (LOCKED before seeing results):
  A_fit = 2.68 (fitted slope from gpt-neo-125m + 20newsgroups layer-wise data)
  Prediction: Delta_logit(q) = A_fit * Delta_kappa_nearest
  Pass criterion: Delta_q >= 0.02 (>= +2pp)
  Sign criterion: kappa_triplet > kappa_CE (triplet increased kappa as intended)

Pass criteria:
  1. delta_q >= 0.02 (nontrivial causal effect)
  2. kappa_triplet > kappa_CE (sign of kappa change correct)
  3. |predicted_delta_logit - actual_delta_logit| < 0.3 (quantitative prediction)
  4. Effect consistent across >= 3/5 seeds
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import logit as sp_logit
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# ============================================================
# LOCKED PREREGISTRATION (do NOT change after running)
# ============================================================
A_FIT = 2.68          # Fitted slope for gpt-neo-125m + 20newsgroups
PASS_DELTA_Q = 0.02   # Nontrivial causal effect threshold
PASS_PRED_TOL = 0.30  # Quantitative prediction tolerance (logit units)
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


# ========== Data loading ==========
def load_20newsgroups_embeddings():
    """Load precomputed gpt-neo-125m layer-12 embeddings if available,
    else extract from scratch."""
    emb_path = RESULTS_DIR / "do_int_embs_v3_gpt-neo-125m_20newsgroups.npz"
    if emb_path.exists():
        print(f"Loading precomputed embeddings from {emb_path.name}")
        npz = np.load(emb_path)
        X, y = npz["X"].astype(np.float32), npz["y"]
        print(f"  X={X.shape}, K={len(np.unique(y))}")
        return X, y
    else:
        print("Extracting embeddings from gpt-neo-125m...")
        return extract_embeddings()


def extract_embeddings():
    """Extract layer-12 embeddings from gpt-neo-125m on 20newsgroups."""
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset

    model_name = "EleutherAI/gpt-neo-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(DEVICE)

    dataset = load_dataset("SetFit/20_newsgroups", split="train")
    # subsample: 300 per class
    texts, labels = [], []
    from collections import defaultdict
    by_label = defaultdict(list)
    for item in dataset:
        by_label[item["label_text"]].append(item["text"])

    label_names = sorted(by_label.keys())
    label2idx = {l: i for i, l in enumerate(label_names)}
    rng = np.random.RandomState(42)
    for lname, ltexts in by_label.items():
        chosen = rng.choice(len(ltexts), min(300, len(ltexts)), replace=False)
        for idx in chosen:
            texts.append(ltexts[idx])
            labels.append(label2idx[lname])

    all_embs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=128, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[12]  # layer 12
        # Last token embedding
        emb = hidden[:, -1, :].cpu().numpy()
        all_embs.append(emb)

    X = np.vstack(all_embs).astype(np.float32)
    y = np.array(labels)
    print(f"  Extracted: X={X.shape}, K={len(np.unique(y))}")
    return X, y


# ========== Projection head ==========
class ProjectionHead(nn.Module):
    def __init__(self, d_in: int = 768, d_out: int = 64, K: int = 20):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=True)
        self.cls  = nn.Linear(d_out, K, bias=True)

    def forward(self, x):
        z = self.proj(x)  # projected embeddings
        logits = self.cls(z)
        return z, logits


# ========== Triplet mining ==========
def hard_negative_triplet_loss(z: torch.Tensor, y: torch.Tensor,
                                margin: float = 0.5) -> torch.Tensor:
    """Online hard-negative triplet loss: max(0, d_ap - d_an + margin)."""
    # Pairwise distances
    diff = z.unsqueeze(0) - z.unsqueeze(1)  # (n, n, d)
    dists = (diff ** 2).sum(-1).clamp(min=1e-12).sqrt()  # (n, n)

    same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    diff_mask = 1.0 - same
    same_mask = same - torch.eye(len(y), device=y.device)  # exclude self

    # For each anchor i:
    # positive: hardest positive (largest d_ap)
    # negative: hardest negative (smallest d_an)
    d_ap = (dists * same_mask).max(dim=1).values
    # Add large value to same-class for negative mining
    d_an = dists + 1e6 * same  # make same-class large
    d_an = d_an.min(dim=1).values

    loss = torch.clamp(d_ap - d_an + margin, min=0.0)
    return loss.mean()


# ========== Metrics ==========
def compute_kappa_nearest(X: np.ndarray, y: np.ndarray) -> float:
    """kappa_nearest = delta_min / (sigma_W * sqrt(d))."""
    classes = np.unique(y)
    d = X.shape[1]
    means = np.array([X[y == c].mean(0) for c in classes])
    dists = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dists.append(np.linalg.norm(means[i] - means[j]))
    delta_min = float(min(dists))
    n = len(y)
    X_c = np.vstack([X[y == c] - means[ci] for ci, c in enumerate(classes)])
    tr_W = float(np.trace(X_c.T @ X_c / n))
    sigma_W = float(np.sqrt(tr_W / d))
    if sigma_W < 1e-10:
        return 0.0
    return float(delta_min / (sigma_W * np.sqrt(d)))


def compute_q_nc(X: np.ndarray, y: np.ndarray) -> float:
    """Normalized nearest-centroid accuracy."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    classes = np.unique(y_tr)
    K = len(classes)
    means = np.array([X_tr[y_tr == c].mean(0) for c in classes])
    dists = np.array([[np.linalg.norm(x - means[j])
                       for j in range(K)] for x in X_te])
    pred = classes[np.argmin(dists, axis=1)]
    acc = float(np.mean(pred == y_te))
    return float((acc - 1.0 / K) / (1.0 - 1.0 / K))


# ========== Training ==========
def train_arm(X: np.ndarray, y: np.ndarray, arm: str, seed: int,
              n_epochs: int = 100, batch_size: int = 128,
              lr: float = 1e-3, triplet_lambda: float = 0.1,
              triplet_margin: float = 0.5) -> dict:
    """Train projection head with CE (arm='ce') or CE+triplet (arm='triplet')."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    K = len(np.unique(y))
    d_in = X.shape[1]
    model = ProjectionHead(d_in=d_in, d_out=64, K=K).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)

    n = len(X)
    checkpoints = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        idx = rng.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            b_idx = idx[i:i + batch_size]
            Xb = X_t[b_idx]
            yb = y_t[b_idx]
            optimizer.zero_grad()
            z, logits = model(Xb)
            loss = ce_loss(logits, yb)
            if arm == "triplet":
                t_loss = hard_negative_triplet_loss(z, yb, margin=triplet_margin)
                loss = loss + triplet_lambda * t_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                z_all, _ = model(X_t)
            z_np = z_all.cpu().numpy()
            kappa = compute_kappa_nearest(z_np, y)
            q_nc = compute_q_nc(z_np, y)
            logit_q = float(sp_logit(np.clip(q_nc, 1e-5, 1 - 1e-5)))
            checkpoints.append({
                "epoch": epoch,
                "loss": total_loss,
                "kappa": kappa,
                "q_nc": q_nc,
                "logit_q": logit_q,
            })
            print(f"  [arm={arm} seed={seed} epoch={epoch}/{n_epochs}] "
                  f"kappa={kappa:.4f} q_nc={q_nc:.4f} logit_q={logit_q:.4f} "
                  f"loss={total_loss:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        z_final, _ = model(X_t)
    z_np = z_final.cpu().numpy()
    final_kappa = compute_kappa_nearest(z_np, y)
    final_q = compute_q_nc(z_np, y)
    final_logit_q = float(sp_logit(np.clip(final_q, 1e-5, 1 - 1e-5)))

    return {
        "arm": arm,
        "seed": seed,
        "final_kappa": final_kappa,
        "final_q": final_q,
        "final_logit_q": final_logit_q,
        "checkpoints": checkpoints,
        "elapsed_sec": time.time() - t0,
    }


# ========== Main ==========
def main():
    print("=" * 70)
    print("PREREGISTERED CAUSAL INTERVENTION")
    print(f"A_FIT={A_FIT} (locked), pass_delta_q>={PASS_DELTA_Q}")
    print("=" * 70)

    # Load data
    X, y = load_20newsgroups_embeddings()
    K = len(np.unique(y))
    print(f"Data: n={len(X)}, d={X.shape[1]}, K={K}")

    # Baseline: raw embeddings (no training)
    kappa_raw = compute_kappa_nearest(X, y)
    q_raw = compute_q_nc(X, y)
    logit_q_raw = float(sp_logit(np.clip(q_raw, 1e-5, 1 - 1e-5)))
    print(f"\nBaseline (raw L12 embeddings):")
    print(f"  kappa={kappa_raw:.4f}, q_nc={q_raw:.4f}, logit_q={logit_q_raw:.4f}")

    # Train both arms with 5 seeds
    seeds = [42, 123, 456, 789, 1024]
    n_epochs = 100
    results = {"ce": [], "triplet": []}

    for arm in ["ce", "triplet"]:
        print(f"\n{'='*50}")
        print(f"TRAINING ARM: {arm}")
        print(f"{'='*50}")
        for seed in seeds:
            r = train_arm(X, y, arm=arm, seed=seed, n_epochs=n_epochs)
            results[arm].append(r)

    # Summary
    ce_results = results["ce"]
    triplet_results = results["triplet"]

    mean_q_ce = float(np.mean([r["final_q"] for r in ce_results]))
    mean_q_triplet = float(np.mean([r["final_q"] for r in triplet_results]))
    mean_kappa_ce = float(np.mean([r["final_kappa"] for r in ce_results]))
    mean_kappa_triplet = float(np.mean([r["final_kappa"] for r in triplet_results]))
    mean_logit_ce = float(np.mean([r["final_logit_q"] for r in ce_results]))
    mean_logit_triplet = float(np.mean([r["final_logit_q"] for r in triplet_results]))

    delta_q = mean_q_triplet - mean_q_ce
    delta_kappa = mean_kappa_triplet - mean_kappa_ce
    delta_logit = mean_logit_triplet - mean_logit_ce
    predicted_delta_logit = A_FIT * delta_kappa

    print(f"\n{'='*70}")
    print(f"INTERVENTION RESULTS:")
    print(f"  CE:      mean_q={mean_q_ce:.4f}, mean_kappa={mean_kappa_ce:.4f}, "
          f"mean_logit={mean_logit_ce:.4f}")
    print(f"  Triplet: mean_q={mean_q_triplet:.4f}, mean_kappa={mean_kappa_triplet:.4f}, "
          f"mean_logit={mean_logit_triplet:.4f}")
    print(f"  delta_q = {delta_q:+.4f}")
    print(f"  delta_kappa = {delta_kappa:+.4f}")
    print(f"  delta_logit_actual = {delta_logit:+.4f}")
    print(f"  delta_logit_predicted = A_FIT * delta_kappa = {predicted_delta_logit:+.4f}")
    print(f"  prediction_error = |actual - predicted| = {abs(delta_logit - predicted_delta_logit):.4f}")

    # Per-seed sign analysis
    signs_q = sum(t["final_q"] > c["final_q"] for t, c in zip(triplet_results, ce_results))
    signs_kappa = sum(t["final_kappa"] > c["final_kappa"]
                      for t, c in zip(triplet_results, ce_results))
    print(f"\n  Per-seed sign analysis:")
    print(f"  q_triplet > q_CE: {signs_q}/{len(seeds)} seeds")
    print(f"  kappa_triplet > kappa_CE: {signs_kappa}/{len(seeds)} seeds")

    # Criteria
    pass1 = delta_q >= PASS_DELTA_Q
    pass2 = delta_kappa > 0  # kappa increased as intended
    pass3 = abs(delta_logit - predicted_delta_logit) < PASS_PRED_TOL
    pass4 = signs_q >= 3  # consistent across seeds

    print(f"\n{'='*70}")
    print(f"PREREGISTERED PASS CRITERIA:")
    print(f"  1. delta_q >= {PASS_DELTA_Q}: {delta_q:+.4f} [{'PASS' if pass1 else 'FAIL'}]")
    print(f"  2. delta_kappa > 0: {delta_kappa:+.4f} [{'PASS' if pass2 else 'FAIL'}]")
    print(f"  3. |pred_err| < {PASS_PRED_TOL}: {abs(delta_logit - predicted_delta_logit):.4f} "
          f"[{'PASS' if pass3 else 'FAIL'}]")
    print(f"  4. sign consistency >= 3/5: {signs_q}/5 [{'PASS' if pass4 else 'FAIL'}]")

    n_pass = sum([pass1, pass2, pass3, pass4])
    verdict = "PASS" if n_pass >= 4 else ("PARTIAL" if n_pass >= 2 else "FAIL")
    print(f"\nVERDICT: {verdict} ({n_pass}/4 criteria)")

    # Save
    result_data = {
        "preregistered": {
            "A_FIT": A_FIT,
            "PASS_DELTA_Q": PASS_DELTA_Q,
            "PASS_PRED_TOL": PASS_PRED_TOL,
        },
        "baseline": {
            "kappa_raw": float(kappa_raw),
            "q_raw": float(q_raw),
            "logit_q_raw": float(logit_q_raw),
        },
        "summary": {
            "mean_q_ce": mean_q_ce,
            "mean_q_triplet": mean_q_triplet,
            "mean_kappa_ce": mean_kappa_ce,
            "mean_kappa_triplet": mean_kappa_triplet,
            "delta_q": delta_q,
            "delta_kappa": delta_kappa,
            "delta_logit_actual": delta_logit,
            "delta_logit_predicted": float(predicted_delta_logit),
            "prediction_error": float(abs(delta_logit - predicted_delta_logit)),
            "signs_q": int(signs_q),
            "signs_kappa": int(signs_kappa),
        },
        "criteria_pass": {
            "delta_q_nontrivial": pass1,
            "kappa_increased": pass2,
            "quantitative_prediction_accurate": pass3,
            "sign_consistent": pass4,
        },
        "verdict": verdict,
        "n_pass": n_pass,
        "results": results,
    }

    out_path = RESULTS_DIR / "cti_preregistered_causal_intervention.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
