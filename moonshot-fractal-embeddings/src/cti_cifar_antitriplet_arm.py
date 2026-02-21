#!/usr/bin/env python -u
"""
CIFAR-100 ANTI-TRIPLET ARM (Feb 21 2026)
=========================================
Bidirectional causal intervention:
  +lambda arm (cti_cifar_triplet_arm.py): CE + lambda*triplet -> INCREASE dist_ratio -> INCREASE q
  -lambda arm (this file):               CE + lambda*anti_triplet -> DECREASE dist_ratio -> DECREASE q

The anti-triplet loss explicitly REVERSES the triplet objective:
  d_pos = hardest positive (farthest same-class)
  d_neg = hardest negative (nearest different-class)
  anti_triplet_loss = max(0, margin - (d_neg - d_pos))
                    = max(0, d_pos - d_neg + margin)  [same form, but we MAXIMIZE dist confusion]

Wait: anti-triplet should REWARD confusion, not separation.
  regular triplet:     loss = max(0, d_pos - d_neg + margin) -> minimize -> d_neg > d_pos + margin
  anti-triplet (ours): loss = max(0, d_neg - d_pos + margin) -> minimize -> d_pos > d_neg + margin
                         i.e., push intra distances LARGER than inter distances -> confusion

Nobel-track PREDICTION:
  - Baseline: q = 0.7077, dist_ratio = DR_baseline (to be measured)
  - +triplet:  q INCREASES, DR INCREASES, Deltalq = A_hat * Delta_DR (from law)
  - -anti:     q DECREASES, DR DECREASES, Deltalq = A_hat * Delta_DR (same law, opposite direction)

Pre-registered criteria:
  1. Anti-triplet q < baseline q (any decrease counts as directional success)
  2. Quantitative: |Deltalq_anti| approx |Deltalq_triplet| (symmetric effect)
  3. Magnitude: Deltalq / Delta_DR is consistent with A_hat from per-dataset fits
"""

import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances as sk_pairwise_distances
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# CONFIG (matching triplet arm exactly)
# ================================================================
SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS = 35
BATCH_SIZE = 256
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
N_EVAL_SUBSAMPLE = 1000

TRIPLET_MARGIN = 0.3      # cosine margin on L2-normalized features
TRIPLET_LAMBDA = 0.1      # same lambda as +arm for symmetry; L2-norm prevents collapse

BASELINE_Q = {
    42: 0.7073, 123: 0.7071, 456: 0.7047, 789: 0.7105, 1024: 0.7087,
}
BASELINE_Q_MEAN = np.mean(list(BASELINE_Q.values()))


# ================================================================
# MODEL
# ================================================================
def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, 20)
    return model.to(DEVICE)


# ================================================================
# DATA
# ================================================================
def _coarse_label(x):
    """Convert CIFAR-100 fine label to coarse (20 classes)."""
    return x // 5


def get_cifar_coarse():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100(
        root="data", train=True, download=True, transform=transform_train,
        target_transform=_coarse_label
    )
    test_ds = torchvision.datasets.CIFAR100(
        root="data", train=False, download=False, transform=transform_test,
        target_transform=_coarse_label
    )
    return train_ds, test_ds


# ================================================================
# ANTI-TRIPLET LOSS: rewards confusion, not separation
# ================================================================
class AntiTripletLoss(nn.Module):
    """
    Anti-triplet: minimize max(0, d_neg - d_pos + margin)
    This pushes d_pos > d_neg + margin -> intra >> inter -> confusion.

    When minimized: intra-class distances EXCEED inter-class distances
    -> dist_ratio = E[inter_min] / E[intra_min] DECREASES
    -> kappa_nearest DECREASES
    -> q DECREASES
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # L2 normalize to prevent collapse (same as +triplet arm)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        B = embeddings.shape[0]
        dists = torch.cdist(embeddings, embeddings)  # (B, B) on unit sphere

        labels = labels.view(-1, 1)
        same_class = (labels == labels.t()).float()
        diff_class = (labels != labels.t()).float()

        eye = torch.eye(B, device=embeddings.device)
        same_class_no_self = same_class * (1 - eye)
        diff_class_no_self = diff_class * (1 - eye)

        losses = []
        for i in range(B):
            pos_mask = same_class_no_self[i].bool()
            neg_mask = diff_class_no_self[i].bool()

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # Anti-triplet: push d_pos > d_neg (confusion)
            # d_pos = nearest same-class (easiest positive — already close)
            # d_neg = farthest different-class (easiest negative — already far)
            # We push these to swap: make intra > inter
            d_pos = dists[i][pos_mask].min()   # NEAREST same-class (we want to push this FAR)
            d_neg = dists[i][neg_mask].max()   # FARTHEST diff-class (we want to pull this CLOSE)

            # Anti-triplet loss: reward when d_pos > d_neg + margin
            # -> penalize when d_neg > d_pos - margin
            loss_i = torch.relu(d_neg - d_pos + self.margin)
            losses.append(loss_i)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(losses).mean()


# ================================================================
# EVALUATION
# ================================================================
@torch.no_grad()
def get_embeddings(model, loader, n_max=N_EVAL_SUBSAMPLE):
    model.eval()
    features_list = []
    labels_list = []

    def hook_fn(module, input, output):
        features_list.append(output.squeeze().cpu().float())

    hook = model.avgpool.register_forward_hook(hook_fn)

    n_seen = 0
    for imgs, targets in loader:
        if n_seen >= n_max:
            break
        imgs = imgs.to(DEVICE)
        model(imgs)
        labels_list.append(targets.numpy())
        n_seen += imgs.shape[0]

    hook.remove()
    feats = torch.cat([f.view(f.shape[0], -1) if f.dim() > 1 else f.unsqueeze(0)
                       for f in features_list], dim=0).numpy()
    labels = np.concatenate(labels_list)
    return feats[:n_max], labels[:n_max]


def compute_q_and_dr(embeddings, labels, K=20):
    X, y = embeddings, labels

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None, None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)

    n_sub = min(len(X), 500)
    idx_sub = np.random.choice(len(X), n_sub, replace=False)
    X_sub = X[idx_sub]
    y_sub = y[idx_sub]

    D_mat = sk_pairwise_distances(X_sub, metric="euclidean")
    np.fill_diagonal(D_mat, np.inf)

    intra_mins, inter_mins = [], []
    for i in range(n_sub):
        same = (y_sub == y_sub[i]); same[i] = False
        diff = ~same; diff[i] = False
        if same.any(): intra_mins.append(D_mat[i][same].min())
        if diff.any(): inter_mins.append(D_mat[i][diff].min())

    if not intra_mins or not inter_mins:
        return float(q), None

    dr = float(np.mean(inter_mins)) / float(np.mean(intra_mins) + 1e-10)
    return float(q), dr


# ================================================================
# TRAINING
# ================================================================
def run_antitriplet_arm(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  Seed {seed}...", flush=True)

    train_ds, test_ds = get_cifar_coarse()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=True,
        num_workers=0
    )

    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 25, 32], gamma=0.1
    )

    ce_loss_fn = nn.CrossEntropyLoss()
    anti_loss_fn = AntiTripletLoss(margin=TRIPLET_MARGIN)

    seed_results = []
    t0 = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            feats = model.avgpool(model.layer4(model.layer3(model.layer2(
                model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(imgs)))))
            )))).squeeze(-1).squeeze(-1)

            logits = model.fc(feats)
            loss_ce = ce_loss_fn(logits, targets)
            loss_anti = anti_loss_fn(feats, targets)

            # Anti-triplet adds loss that rewards confusion
            loss = loss_ce + TRIPLET_LAMBDA * loss_anti

            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % 5 == 0 or epoch == N_EPOCHS:
            embs, lbls = get_embeddings(model, eval_loader, N_EVAL_SUBSAMPLE)
            q, dr = compute_q_and_dr(embs, lbls)

            mu_bar = embs.mean(0)
            S_B = sum(((embs[lbls == k].mean(0) - mu_bar) ** 2).sum() * (lbls == k).sum()
                      for k in range(20)) if len(embs) > 0 else 0
            S_W = sum(((embs[lbls == k] - embs[lbls == k].mean(0)) ** 2).sum()
                      for k in range(20)) if len(embs) > 0 else 1
            kappa = float(S_B / (S_W + 1e-10))

            knn_acc = q * (1 - 1/20) + 1/20
            print(f"    Epoch {epoch:2d}: kNN={knn_acc:.4f} q={q:.4f} DR={dr:.4f} kappa={kappa:.4f}",
                  flush=True)
            seed_results.append({
                "epoch": epoch, "q": q, "knn_acc": knn_acc, "dr": dr, "kappa": kappa
            })

    print(f"  Done in {int(time.time()-t0)}s", flush=True)
    return seed_results


def main():
    print("=" * 70)
    print("CIFAR-100 ARM: CE + Anti-Triplet Loss (bidirectional test, -lambda)")
    print(f"  margin={TRIPLET_MARGIN}, lambda={TRIPLET_LAMBDA}")
    print(f"  Baseline q (mean 5 seeds): {BASELINE_Q_MEAN:.4f}")
    print(f"  Prediction: q DECREASES (anti-triplet rewards confusion)")
    print("=" * 70)

    all_results = {}

    for seed in SEEDS:
        seed_results = run_antitriplet_arm(seed)
        all_results[str(seed)] = seed_results

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    final_qs = {}
    final_drs = {}
    for seed in SEEDS:
        results = all_results[str(seed)]
        if results:
            final_q = results[-1]["q"]
            final_dr = results[-1]["dr"]
            baseline_q = BASELINE_Q.get(seed, BASELINE_Q_MEAN)
            delta_q = final_q - baseline_q
            final_qs[seed] = final_q
            final_drs[seed] = final_dr
            print(f"  Seed {seed}: q={final_q:.4f}  DR={final_dr:.4f}  "
                  f"baseline={baseline_q:.4f}  delta={delta_q:+.4f}")

    if final_qs:
        mean_q = float(np.mean(list(final_qs.values())))
        delta_mean = mean_q - BASELINE_Q_MEAN
        directional_pass = delta_mean < 0  # anti-triplet should DECREASE q

        print(f"\n  Mean q: {mean_q:.4f}  ({delta_mean:+.4f} over baseline)")
        print(f"  Directional test (q < baseline): {'PASS' if directional_pass else 'FAIL'}")
        print(f"  Note: Compare DR and logit(q) changes against +triplet arm for quantitative test")

        all_results["summary"] = {
            "mean_q": mean_q,
            "baseline_q": BASELINE_Q_MEAN,
            "delta_q": delta_mean,
            "directional_pass": bool(directional_pass),
            "mean_dr": float(np.mean([v for v in final_drs.values() if v is not None]))
                       if any(v is not None for v in final_drs.values()) else None,
        }

    out_path = "results/cti_cifar_antitriplet.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
