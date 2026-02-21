#!/usr/bin/env python -u
"""
CIFAR-100 HARD-NEGATIVE TRIPLET ARM (Feb 21 2026)
==================================================
Theory prediction: kappa_nearest (min margin) is the CAUSAL driver of kNN quality.
dist_ratio regularizer FAILED (+0.003 vs +0.02 threshold) because it optimizes
MEAN distances, not MINIMUM margins.

This experiment tests the CORRECT causal lever: hard-negative triplet loss.
Triplet loss with hard negatives directly optimizes kappa_nearest by:
  - x_pos = hardest positive (nearest same-class sample -> within-class NN)
  - x_neg = hardest negative (nearest different-class sample -> between-class NN)
  - loss = max(0, d(x, x_neg) - d(x, x_pos) + margin)

Nobel-track PREDICTION:
  - Arm 1 (CE baseline): q ~ 0.707 (from completed baseline arm)
  - Arm 2 (CE + dist_ratio regularizer): q ~ 0.710 (FAIL, +0.003)
  - Arm 3 (CE + hard-neg triplet): q ~ 0.727+ (PREDICTION: +2pp over baseline)

If arm 3 gives +2pp: confirms kappa_nearest is the causal driver, dist_ratio is diagnostic.
If arm 3 also fails: our causal theory needs revision.

Pre-registered criterion: +2pp q gain over baseline (same as dist_ratio arm).
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
# CONFIG (matching baseline arm exactly)
# ================================================================
SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS = 35
BATCH_SIZE = 256
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
N_EVAL_SUBSAMPLE = 1000   # for kNN evaluation

# Triplet loss hyperparameter
TRIPLET_MARGIN = 0.5
TRIPLET_LAMBDA = 0.5      # weight of triplet loss relative to CE

PRE_REGISTERED_IMPROVEMENT = 0.02  # +2pp q gain over baseline


# ================================================================
# BASELINE REFERENCE (from completed baseline arm)
# ================================================================
BASELINE_Q = {
    42: 0.7073,
    123: 0.7071,
    456: 0.7047,
    789: 0.7105,
    1024: 0.7087,
}
BASELINE_Q_MEAN = np.mean(list(BASELINE_Q.values()))


# ================================================================
# MODEL: Simple ResNet-18 (same as baseline)
# ================================================================
def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, 20)  # CIFAR-100 coarse (20 classes)
    return model.to(DEVICE)


# ================================================================
# DATA
# ================================================================
def _coarse_label(x):
    """Convert CIFAR-100 fine label to coarse (20 classes)."""
    return x // 5


def get_cifar_coarse():
    """CIFAR-100 with coarse labels (20 classes)."""
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
        target_transform=_coarse_label  # coarse label (picklable for Windows multiprocessing)
    )
    test_ds = torchvision.datasets.CIFAR100(
        root="data", train=False, download=False, transform=transform_test,
        target_transform=_coarse_label
    )
    return train_ds, test_ds


# ================================================================
# HARD-NEGATIVE TRIPLET LOSS
# ================================================================
class HardNegativeTripletLoss(nn.Module):
    """
    Mine hardest positive and negative for each sample in the batch.
    Uses EMBEDDINGS (penultimate layer features), not logits.
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, d) float tensor
            labels: (B,) int tensor with class indices

        Returns:
            triplet_loss: scalar (mean over valid triplets)
        """
        B = embeddings.shape[0]
        # Pairwise distances
        dists = torch.cdist(embeddings, embeddings)  # (B, B)

        labels = labels.view(-1, 1)  # (B, 1)
        same_class = (labels == labels.t()).float()  # (B, B)
        diff_class = (labels != labels.t()).float()  # (B, B)

        # Mask self-distances
        eye = torch.eye(B, device=embeddings.device)
        same_class_no_self = same_class * (1 - eye)
        diff_class_no_self = diff_class * (1 - eye)

        # Hard positive: farthest same-class sample
        # For hard negatives: nearest different-class sample
        losses = []
        for i in range(B):
            pos_mask = same_class_no_self[i].bool()
            neg_mask = diff_class_no_self[i].bool()

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # Hard positive: most confusing same-class (farthest)
            d_pos = dists[i][pos_mask].max()
            # Hard negative: most confusing different-class (nearest)
            d_neg = dists[i][neg_mask].min()

            loss_i = torch.relu(d_pos - d_neg + self.margin)
            losses.append(loss_i)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(losses).mean()


# ================================================================
# EVALUATION
# ================================================================
@torch.no_grad()
def get_embeddings(model, loader, n_max=N_EVAL_SUBSAMPLE):
    """Extract penultimate-layer embeddings."""
    model.eval()
    # Hook to capture penultimate layer (avgpool output)
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
    """Compute q and dist_ratio from embeddings."""
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

    # dist_ratio (subsample)
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
# MAIN TRAINING LOOP
# ================================================================
def run_triplet_arm(seed):
    """Train ResNet-18 with CE + hard-negative triplet loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  Seed {seed}...", flush=True)

    train_ds, test_ds = get_cifar_coarse()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True  # num_workers=0 for Windows (avoids lambda pickling)
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
    triplet_loss_fn = HardNegativeTripletLoss(margin=TRIPLET_MARGIN)

    seed_results = []
    t0 = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            # Forward
            optimizer.zero_grad()

            # Get embeddings from penultimate layer
            feats = model.avgpool(model.layer4(model.layer3(model.layer2(
                model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(imgs)))))
            )))).squeeze(-1).squeeze(-1)  # (B, 512)

            logits = model.fc(feats)

            # CE loss
            loss_ce = ce_loss_fn(logits, targets)

            # Triplet loss on features
            loss_triplet = triplet_loss_fn(feats, targets)

            # Total loss
            loss = loss_ce + TRIPLET_LAMBDA * loss_triplet

            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % 5 == 0 or epoch == N_EPOCHS:
            embs, lbls = get_embeddings(model, eval_loader, N_EVAL_SUBSAMPLE)
            q, dr = compute_q_and_dr(embs, lbls)

            # kappa
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


# ================================================================
# RUN ALL SEEDS
# ================================================================
def main():
    print("=" * 70)
    print("CIFAR-100 ARM: CE + Hard-Negative Triplet Loss")
    print(f"Theory prediction: +{PRE_REGISTERED_IMPROVEMENT:.2f} q over baseline")
    print(f"  margin={TRIPLET_MARGIN}, lambda={TRIPLET_LAMBDA}")
    print(f"  Baseline q (mean 5 seeds): {BASELINE_Q_MEAN:.4f}")
    print("=" * 70)

    all_results = {}

    for seed in SEEDS:
        seed_results = run_triplet_arm(seed)
        all_results[str(seed)] = seed_results

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    final_qs = {}
    for seed in SEEDS:
        results = all_results[str(seed)]
        if results:
            final_q = results[-1]["q"]
            baseline_q = BASELINE_Q.get(seed, BASELINE_Q_MEAN)
            delta_q = final_q - baseline_q
            final_qs[seed] = final_q
            print(f"  Seed {seed}: q={final_q:.4f}  baseline={baseline_q:.4f}  delta={delta_q:+.4f}")

    if final_qs:
        mean_q = float(np.mean(list(final_qs.values())))
        delta_mean = mean_q - BASELINE_Q_MEAN
        passed = delta_mean >= PRE_REGISTERED_IMPROVEMENT

        print(f"\n  Mean q: {mean_q:.4f}  (+{delta_mean:.4f} over baseline)")
        print(f"  Pre-registered threshold: +{PRE_REGISTERED_IMPROVEMENT:.3f}")
        print(f"  RESULT: {'PASS - kappa_nearest IS causal (triplet works!)' if passed else 'FAIL - triplet does not help enough'}")

        all_results["summary"] = {
            "mean_q": mean_q,
            "baseline_q": BASELINE_Q_MEAN,
            "delta_q": delta_mean,
            "passed": bool(passed),
            "threshold": PRE_REGISTERED_IMPROVEMENT,
        }

    out_path = "results/cti_cifar_triplet.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
