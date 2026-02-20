#!/usr/bin/env python
"""
CAUSAL PAYOFF (CIFAR-100): Observable Order-Parameter as Training Objective

Codex-designed experiment (Feb 20, 2026):
- Dataset: CIFAR-100 coarse labels (K=20)
- Architecture: small CNN from scratch (fast, reproducible)
- Loss: CE + lambda * max(0, 1 - dist_ratio)  [THEOREM-DERIVED]
- 3 arms: baseline, dist_ratio, wrong_sign_control
- 5 seeds, 35 epochs, proper causal test

From theorem: logit(q) = A * (dist_ratio - 1) + C
So: maximizing dist_ratio should maximize q (kNN quality).

Causal test: wrong_sign_control inverts the regularizer (penalizes
dist_ratio > 1), which theory predicts should HURT quality. If the
ordering is: dist_ratio > baseline > wrong_sign, causality is confirmed.

Pre-registered criterion:
  - dist_ratio arm mean 5-NN gain >= +2.0 pp vs baseline
  - 95% bootstrap CI lower bound > 0
  - Ordering: dist_ratio > baseline > wrong_sign_control
"""

import json
import sys
import time
import gc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Experiment params
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 35
BATCH_SIZE = 256
LR_BASE = 0.1
WEIGHT_DECAY = 5e-4
LAMBDA = 0.2
LAMBDA_WARMUP_EPOCHS = 5
EMBED_DIM = 128


# ============================================================
# Model: Small CNN -> 128-dim embedding
# ============================================================

class SmallCNN(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_classes=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),                              # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),                              # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                         # 1x1
        )
        self.embed_head = nn.Linear(256, embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_emb=False):
        feat = self.encoder(x).squeeze(-1).squeeze(-1)   # [B, 256]
        emb = self.embed_head(feat)                       # [B, embed_dim]
        logits = self.cls_head(emb)                       # [B, num_classes]
        if return_emb:
            return logits, emb
        return logits


# ============================================================
# CIFAR-100 coarse-label loader
# ============================================================

def get_cifar100_coarse(train=True):
    """Load CIFAR-100 with coarse labels (K=20)."""
    import torchvision
    import torchvision.transforms as T

    if train:
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408),
                        (0.2675, 0.2565, 0.2761)),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408),
                        (0.2675, 0.2565, 0.2761)),
        ])

    ds = torchvision.datasets.CIFAR100(
        root=str(REPO_ROOT / "data" / "cifar100"),
        train=train, download=True, transform=transform,
    )
    # Standard CIFAR-100 coarse label map (fine index -> coarse index 0-19)
    fine_to_coarse = [
        4,  1,  14,  8,  0,  6,  7,  7,  18, 3,
        3,  14,  9,  18, 7, 11,  3,  9,  7, 11,
        6,  11,  5, 10,  7,  6, 13, 15,  3, 15,
        0,  11,  1, 10, 12, 14, 16,  9, 11,  5,
        5,  19,  8,  8, 15, 13, 14, 17, 18, 10,
        16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
        10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
        2,  10,  0,  1, 16, 12,  9, 13, 15, 13,
        16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
        18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
    ]
    ds.targets = [fine_to_coarse[t] for t in ds.targets]
    return ds


# ============================================================
# Dist-ratio loss (Codex-designed, efficient)
# ============================================================

def dist_ratio_reg(emb, labels, wrong_sign=False):
    """
    Batch-efficient dist_ratio regularizer.
    dist_ratio = E[NN_inter] / E[NN_intra] in L2 embedding space.

    If wrong_sign=True: maximize intra/inter (inverted causal control).
    Returns loss scalar.
    """
    z = F.normalize(emb, dim=1)
    # Squared L2 distances via cosine: ||a-b||^2 = 2 - 2*cos(a,b)
    D = (2 - 2 * (z @ z.T)).clamp_min(0)     # [B, B]

    same = labels.unsqueeze(1).eq(labels.unsqueeze(0))  # [B, B]
    same.fill_diagonal_(False)

    # Nearest intra-class and inter-class distances
    d_intra = D.masked_fill(~same, 1e9).min(1).values    # [B]
    d_inter = D.masked_fill(same, 1e9).min(1).values     # [B]

    valid = (d_intra < 1e8) & (d_inter < 1e8)
    if valid.sum() < 2:
        return torch.tensor(0.0, device=emb.device)

    dr = d_inter[valid].mean() / (d_intra[valid].mean() + 1e-8)

    if wrong_sign:
        # Invert: penalize dist_ratio > 1 (wrong direction per theory)
        return F.relu(dr - 1.0)
    else:
        # Normal: penalize dist_ratio < 1 (want inter > intra)
        return F.relu(1.0 - dr)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def get_embeddings(model, loader, device=DEVICE):
    model.eval()
    all_emb, all_labels = [], []
    for imgs, lbls in loader:
        _, emb = model(imgs.to(device), return_emb=True)
        all_emb.append(emb.cpu().numpy())
        all_labels.append(lbls.numpy())
    return np.concatenate(all_emb, 0), np.concatenate(all_labels, 0)


def evaluate(model, train_loader, test_loader, device=DEVICE):
    """Return kNN test accuracy and geometry stats."""
    X_tr, y_tr = get_embeddings(model, train_loader, device)
    X_te, y_te = get_embeddings(model, test_loader, device)
    K = int(y_tr.max()) + 1

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_tr, y_tr)
    knn_acc = float(knn.score(X_te, y_te))
    q = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)

    # kappa on test embeddings
    grand = X_te.mean(0)
    tr_sb, tr_sw = 0.0, 0.0
    for lbl in np.unique(y_te):
        m = y_te == lbl
        if m.sum() < 2:
            continue
        Xk = X_te[m]; muk = Xk.mean(0)
        tr_sb += float(m.sum()) * float(((muk - grand) ** 2).sum())
        tr_sw += float(((Xk - muk) ** 2).sum())
    kappa = tr_sb / (tr_sw + 1e-10)

    # dist_ratio on subset of test embeddings
    idx = np.random.choice(len(y_te), min(1000, len(y_te)), replace=False)
    Xs = X_te[idx]; ys = y_te[idx]
    Xs_n = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    sim = Xs_n @ Xs_n.T
    D_sq = (2 - 2 * sim).clip(0)
    intra_nn, inter_nn = [], []
    for i in range(len(ys)):
        same = (ys == ys[i]).copy()
        same[i] = False
        if same.sum() > 0:
            intra_nn.append(D_sq[i, same].min())
        if (~same).sum() > 1:
            inter_row = D_sq[i].copy()
            inter_row[ys == ys[i]] = np.inf
            inter_nn.append(inter_row.min())
    dr = (float(np.mean(inter_nn)) / (float(np.mean(intra_nn)) + 1e-10)
          if intra_nn and inter_nn else 0.0)

    return {"knn": knn_acc, "q": float(q), "kappa": float(kappa),
            "dist_ratio": float(dr), "K": K}


# ============================================================
# Training
# ============================================================

def train_one_seed(seed, arm, train_ds, test_ds):
    """Train one seed for a given arm and return trajectory."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SmallCNN(embed_dim=EMBED_DIM, num_classes=20).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    eval_train_loader = DataLoader(train_ds, batch_size=512, shuffle=False,
                                   num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=0, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR_BASE,
                                momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    trajectory = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Lambda warmup
        if arm == "baseline":
            lam = 0.0
        elif epoch <= LAMBDA_WARMUP_EPOCHS:
            lam = LAMBDA * epoch / LAMBDA_WARMUP_EPOCHS
        else:
            lam = LAMBDA

        ce_acc_sum, n_batches = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

            with torch.amp.autocast("cuda"):
                logits, emb = model(imgs, return_emb=True)
                ce_loss = F.cross_entropy(logits, lbls)

                if lam > 0 and arm == "dist_ratio":
                    reg = dist_ratio_reg(emb, lbls, wrong_sign=False)
                    loss = ce_loss + lam * reg
                elif lam > 0 and arm == "wrong_sign":
                    reg = dist_ratio_reg(emb, lbls, wrong_sign=True)
                    loss = ce_loss + lam * reg
                else:
                    loss = ce_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            ce_acc_sum += (logits.argmax(1) == lbls).float().mean().item()
            n_batches += 1

        scheduler.step()

        # Eval at certain epochs
        if epoch in (1, 5, 10, 15, 20, 25, 30, 35):
            ev = evaluate(model, eval_train_loader, test_loader)
            ev["epoch"] = epoch
            ev["ce_acc"] = ce_acc_sum / n_batches
            trajectory.append(ev)
            print(f"    Epoch {epoch:2d}: kNN={ev['knn']:.4f} q={ev['q']:.4f} "
                  f"DR={ev['dist_ratio']:.4f} kappa={ev['kappa']:.4f}",
                  flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return trajectory


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("CAUSAL PAYOFF: dist_ratio Regularizer on CIFAR-100 (Codex design)")
    print("Theorem: logit(q) = A*(dist_ratio - 1) + C")
    print("=" * 70)

    # Load data
    print("\nLoading CIFAR-100 coarse labels...", flush=True)
    train_ds = get_cifar100_coarse(train=True)
    test_ds = get_cifar100_coarse(train=False)
    K_actual = len(set(train_ds.targets))
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}, K={K_actual}",
          flush=True)

    arms = ["baseline", "dist_ratio", "wrong_sign"]
    all_results = {arm: [] for arm in arms}

    for arm in arms:
        print(f"\n{'='*70}\nARM: {arm}\n{'='*70}", flush=True)
        for seed in SEEDS:
            print(f"\n  Seed {seed}...", flush=True)
            t0 = time.time()
            traj = train_one_seed(seed, arm, train_ds, test_ds)
            all_results[arm].append({"seed": seed, "trajectory": traj})
            print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    # === ANALYSIS ===
    print(f"\n{'='*70}\nRESULTS\n{'='*70}")

    def final_q(arm):
        return [r["trajectory"][-1]["q"] for r in all_results[arm]]

    for arm in arms:
        qs = final_q(arm)
        print(f"  {arm:>15}: mean_q={np.mean(qs):.4f} +/- {np.std(qs):.4f}  "
              f"kNN={np.mean([r['trajectory'][-1]['knn'] for r in all_results[arm]]):.4f}  "
              f"DR={np.mean([r['trajectory'][-1]['dist_ratio'] for r in all_results[arm]]):.4f}")

    # Deltas vs baseline
    base_qs = np.array(final_q("baseline"))
    dr_qs = np.array(final_q("dist_ratio"))
    ws_qs = np.array(final_q("wrong_sign"))
    deltas = dr_qs - base_qs

    # Bootstrap CI for mean delta
    boot_means = []
    for _ in range(10000):
        idx = np.random.choice(len(deltas), len(deltas), replace=True)
        boot_means.append(deltas[idx].mean())
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    print(f"\n  dist_ratio vs baseline:")
    print(f"    Mean delta_q = {deltas.mean():.4f} pp ({deltas.mean()*100:.2f}%)")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Seed-wise deltas: {[f'{d:.4f}' for d in deltas]}")

    print(f"\n{'='*70}\nSCORECARD\n{'='*70}")
    checks = [
        ("dist_ratio mean gain >= +2.0 pp vs baseline",
         deltas.mean() * 100 >= 2.0,
         f"gain={deltas.mean()*100:.2f}pp"),
        ("95% CI lower bound > 0 (statistically significant)",
         ci_lo > 0,
         f"CI=[{ci_lo:.4f},{ci_hi:.4f}]"),
        ("All 5 seeds: dist_ratio > baseline",
         all(dr_qs > base_qs),
         f"wins={sum(dr_qs>base_qs)}/5"),
        ("Ordering: dist_ratio > baseline > wrong_sign",
         dr_qs.mean() > base_qs.mean() > ws_qs.mean(),
         f"dr={dr_qs.mean():.4f}>base={base_qs.mean():.4f}>ws={ws_qs.mean():.4f}"),
        ("wrong_sign arm is worse than baseline (causal direction confirmed)",
         ws_qs.mean() < base_qs.mean(),
         f"ws={ws_qs.mean():.4f} vs base={base_qs.mean():.4f}"),
    ]
    passes = sum(1 for _, p, _ in checks if p)
    for crit, passed, val in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {crit}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    out_path = RESULTS_DIR / "cti_dist_ratio_causal_cifar.json"
    with open(out_path, "w") as f:
        json.dump({
            "arms": all_results,
            "summary": {
                arm: {
                    "mean_q": float(np.mean(final_q(arm))),
                    "std_q": float(np.std(final_q(arm))),
                    "mean_knn": float(np.mean([r["trajectory"][-1]["knn"]
                                               for r in all_results[arm]])),
                    "mean_dist_ratio": float(np.mean([r["trajectory"][-1]["dist_ratio"]
                                                      for r in all_results[arm]])),
                }
                for arm in arms
            },
            "delta_dr_vs_base": {
                "mean": float(deltas.mean()),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "per_seed": deltas.tolist(),
            },
            "scorecard": {crit: passed for crit, passed, _ in checks},
            "passes": int(passes),
            "total": int(len(checks)),
        }, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
