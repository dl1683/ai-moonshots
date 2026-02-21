#!/usr/bin/env python -u
"""
CIFAR-100 TWO-STAGE TRIPLET ARM (Feb 21 2026)
=============================================
FIX for gradient conflict in Stage 1 (joint CE+triplet).

PROBLEM: Joint CE+triplet training conflicted:
  - CE trains unnormalized features for classification
  - Triplet normalizes features + mines hard negatives
  - Result: kappa collapsed from 0.35 -> 0.09, q 0.70 -> 0.22

TWO-STAGE FIX:
  Stage 1: Train pure CE for 35 epochs -> stable representations (kappa~0.35, q~0.70)
  Stage 2: Freeze FC head, fine-tune backbone with ONLY kappa_nearest-targeting triplet
           LR=1e-4 for 10 epochs -> should increase kappa without CE interference

THEORETICAL PREDICTION:
  logit(q) = alpha * kappa_nearest + C (alpha=1.54, within-task)
  delta_kappa = +0.10 (modest increase from fine-tuning) -> delta_logit = 0.154
  q_baseline = 0.707, logit(q_baseline) ≈ 0.879
  q_predicted = sigmoid(0.879 + 0.154) ≈ 0.726 -> delta_q ≈ +0.019
  So +2pp criterion (~+0.02) is barely achievable; +1pp is expected.

Pre-registered criterion: q_stage2 > q_stage1 (any improvement proves causal direction).
Stronger criterion (pre-registered): q_stage2 >= 0.7277 (baseline + 0.02).

NOTE: Uses centroid-based triplet in Stage 2:
  - For each batch, compute per-class centroids
  - Triplet: sample -> pull towards own centroid, push away from nearest other centroid
  - This DIRECTLY maximizes the min inter-centroid distance (kappa_nearest numerator)
  - Better than hard-negative mining (which can be unstable)
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
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# CONFIG
# ================================================================
SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS_CE   = 35    # Stage 1: CE-only training (matches baseline arm exactly)
N_EPOCHS_TRIP = 10    # Stage 2: centroid-triplet fine-tuning
BATCH_SIZE     = 256
LR_CE          = 0.1  # Stage 1 LR (same as baseline)
LR_TRIP        = 1e-4 # Stage 2 LR (small to preserve Stage 1 geometry)
MOMENTUM       = 0.9
WEIGHT_DECAY   = 1e-4
N_EVAL_SUBSAMPLE = 1000
TRIPLET_MARGIN   = 0.3   # Stage 2 triplet margin (cosine space on unit sphere)
K = 20  # CIFAR-100 coarse classes

PRE_REGISTERED_IMPROVEMENT = 0.02  # +2pp over baseline

# ================================================================
# BASELINE REFERENCE (from completed baseline arm)
# ================================================================
BASELINE_Q = {42: 0.7073, 123: 0.7071, 456: 0.7047, 789: 0.7105, 1024: 0.7087}
BASELINE_Q_MEAN = float(np.mean(list(BASELINE_Q.values())))


# ================================================================
# MODEL: ResNet-18 (same as baseline arm)
# ================================================================
def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, K)
    return model.to(DEVICE)


def coarse_label(x):
    return x // 5


# ================================================================
# DATA
# ================================================================
def get_cifar_coarse():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100(
        root="data", train=True,  download=True,
        transform=train_transform, target_transform=coarse_label
    )
    test_ds = torchvision.datasets.CIFAR100(
        root="data", train=False, download=False,
        transform=test_transform,  target_transform=coarse_label
    )
    return train_ds, test_ds


# ================================================================
# CENTROID-BASED TRIPLET LOSS (Stage 2)
# ================================================================
class CentroidTripletLoss(nn.Module):
    """
    For each sample in the batch, compute mini-batch centroids per class.
    Triplet:
      anchor:   sample x_i (normalized)
      positive: centroid of class y_i (normalized)
      negative: centroid of the NEAREST different class (normalized)

    This directly targets kappa_nearest = min_{j!=k} ||mu_k - mu_j|| / (sigma_W * sqrt(d))
    by pushing centroids apart (increasing numerator) and pulling within-class variance down.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: (B, d) float tensor — penultimate features (NOT normalized here)
        labels: (B,) int tensor

        Returns: scalar triplet loss
        """
        emb_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # (B, d)
        classes = torch.unique(labels)

        # Compute per-class centroids (in normalized space)
        centroids = {}
        for c in classes:
            mask = (labels == c)
            centroids[c.item()] = emb_norm[mask].mean(0)  # (d,)
        centroid_tensor = torch.stack(list(centroids.values()))  # (C, d)
        centroid_keys   = list(centroids.keys())

        # Normalize centroids
        centroid_tensor = torch.nn.functional.normalize(centroid_tensor, p=2, dim=1)

        # Pairwise centroid distances (for finding nearest negative centroid)
        # (C, C)
        centroid_dists = torch.cdist(centroid_tensor, centroid_tensor)

        losses = []
        for idx, c in enumerate(centroid_keys):
            # Positive: centroid of class c
            c_pos = centroid_tensor[idx]  # (d,)

            # Negative: nearest OTHER centroid
            neg_dists = centroid_dists[idx].clone()
            # Mask own centroid
            neg_dists[idx] = float("inf")
            nearest_neg_idx = neg_dists.argmin().item()
            c_neg = centroid_tensor[nearest_neg_idx]  # (d,)

            # All samples from class c
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            anchors = emb_norm[mask]  # (n_c, d)

            # d(anchor, positive_centroid) - pull together
            d_pos = 1 - (anchors * c_pos.unsqueeze(0)).sum(1)  # cosine distance = 1 - cos
            # d(anchor, negative_centroid) - push apart
            d_neg = 1 - (anchors * c_neg.unsqueeze(0)).sum(1)

            # Triplet loss: max(0, d_pos - d_neg + margin)
            loss_c = torch.relu(d_pos - d_neg + self.margin).mean()
            losses.append(loss_c)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return torch.stack(losses).mean()


# ================================================================
# EVALUATION
# ================================================================
@torch.no_grad()
def get_embeddings(model, loader, n_max=N_EVAL_SUBSAMPLE):
    """Extract penultimate-layer (avgpool) embeddings."""
    model.eval()
    features_list, labels_list = [], []

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


def compute_kappa_nearest(X, y, K=20):
    """
    kappa_nearest = min_{j!=k} ||mu_k - mu_j|| / (sigma_W * sqrt(d))
    sigma_W = sqrt(mean within-class variance / d)
    """
    classes = np.unique(y)
    d = X.shape[1]
    means, within_vars = {}, []
    for c in classes:
        Xc = X[y == c]
        means[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - means[c])**2, axis=1)))
    sigma_W = np.sqrt(np.mean(within_vars) / d)

    min_dist = np.inf
    cls_list = list(classes)
    for i in range(len(cls_list)):
        for j in range(i + 1, len(cls_list)):
            dist = np.linalg.norm(means[cls_list[i]] - means[cls_list[j]])
            if dist < min_dist:
                min_dist = dist

    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))


def compute_q_and_kappa(embeddings, labels, K=20):
    """Compute q (normalized 1-NN accuracy) and kappa_nearest."""
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
    kappa = compute_kappa_nearest(X, y, K=K)
    return float(q), kappa


# ================================================================
# STAGE 1: PURE CE TRAINING
# ================================================================
def run_stage1_ce(seed, train_ds, test_ds):
    """Train pure CE for N_EPOCHS_CE epochs. Returns model + stage1 metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = get_model()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_CE, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_CE)

    print(f"  [Seed {seed}] Stage 1: CE training for {N_EPOCHS_CE} epochs...", flush=True)
    t0 = time.time()
    for epoch in range(1, N_EPOCHS_CE + 1):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = ce_loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 5 == 0:
            elapsed = time.time() - t0
            embs, labels = get_embeddings(model, test_loader)
            q, kappa = compute_q_and_kappa(embs, labels, K=K)
            print(f"    Epoch {epoch:3d}: CE_loss={total_loss/len(train_loader):.4f}, "
                  f"q={q:.4f}, kappa={kappa:.4f} ({elapsed:.0f}s)", flush=True)

    # Final evaluation
    embs, labels = get_embeddings(model, test_loader)
    q, kappa = compute_q_and_kappa(embs, labels, K=K)
    elapsed = time.time() - t0
    print(f"  [Seed {seed}] Stage 1 complete: q={q:.4f}, kappa={kappa:.4f} ({elapsed:.0f}s)",
          flush=True)
    return model, float(q), float(kappa)


# ================================================================
# STAGE 2: CENTROID-TRIPLET FINE-TUNING
# ================================================================
def run_stage2_triplet(seed, model, train_ds, test_ds, stage1_q, stage1_kappa):
    """Fine-tune backbone with centroid-triplet loss, freeze FC head."""
    # Freeze FC head (classification head)
    for p in model.fc.parameters():
        p.requires_grad = False

    # Fine-tune backbone at small LR
    backbone_params = [p for name, p in model.named_parameters()
                       if "fc" not in name and p.requires_grad]
    optimizer = optim.Adam(backbone_params, lr=LR_TRIP)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    triplet_loss_fn = CentroidTripletLoss(margin=TRIPLET_MARGIN)

    per_epoch = []
    print(f"  [Seed {seed}] Stage 2: Centroid-triplet fine-tuning for {N_EPOCHS_TRIP} epochs...",
          flush=True)
    t0 = time.time()

    for epoch in range(1, N_EPOCHS_TRIP + 1):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            # Get penultimate features (bypass FC head)
            feats = model.avgpool(model.layer4(model.layer3(model.layer2(
                model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(imgs)))))
            )))).squeeze(-1).squeeze(-1)  # (B, 512)

            loss = triplet_loss_fn(feats, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        elapsed = time.time() - t0
        embs, labels = get_embeddings(model, test_loader)
        q, kappa = compute_q_and_kappa(embs, labels, K=K)
        delta_q = q - stage1_q
        delta_kappa = kappa - stage1_kappa
        print(f"    Epoch {epoch:2d}: triplet_loss={total_loss/len(train_loader):.4f}, "
              f"q={q:.4f} ({delta_q:+.4f}), kappa={kappa:.4f} ({delta_kappa:+.4f}) ({elapsed:.0f}s)",
              flush=True)

        per_epoch.append({
            "epoch": epoch,
            "q": float(q),
            "kappa": float(kappa),
            "delta_q": float(delta_q),
            "delta_kappa": float(delta_kappa),
            "triplet_loss": float(total_loss / len(train_loader)),
        })

    return per_epoch


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70, flush=True)
    print("TWO-STAGE TRIPLET ARM (CE first, then centroid-triplet fine-tune)", flush=True)
    print("=" * 70, flush=True)
    print(f"Stage 1: {N_EPOCHS_CE} epochs CE (LR={LR_CE})", flush=True)
    print(f"Stage 2: {N_EPOCHS_TRIP} epochs centroid-triplet fine-tuning (LR={LR_TRIP})", flush=True)
    print(f"Pre-registered criterion: q_stage2 >= {BASELINE_Q_MEAN + PRE_REGISTERED_IMPROVEMENT:.4f}", flush=True)
    print(flush=True)

    train_ds, test_ds = get_cifar_coarse()

    all_results = {}
    stage1_qs, stage2_qs, stage2_kappas = [], [], []

    for seed in SEEDS:
        print(f"\n{'='*50}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*50}", flush=True)

        # Stage 1: CE training
        model, s1_q, s1_kappa = run_stage1_ce(seed, train_ds, test_ds)
        stage1_qs.append(s1_q)

        # Stage 2: Centroid-triplet fine-tuning
        per_epoch = run_stage2_triplet(seed, model, train_ds, test_ds, s1_q, s1_kappa)

        if per_epoch:
            final = per_epoch[-1]
            stage2_qs.append(final["q"])
            stage2_kappas.append(final["kappa"])
        else:
            stage2_qs.append(s1_q)
            stage2_kappas.append(s1_kappa)

        all_results[str(seed)] = {
            "stage1_q": s1_q,
            "stage1_kappa": s1_kappa,
            "stage2_epochs": per_epoch,
        }

        # Save partial result
        with open("results/cti_cifar_twostage_triplet.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved partial result for seed {seed}", flush=True)

    # Summary
    mean_s1_q = float(np.mean(stage1_qs))
    mean_s2_q = float(np.mean(stage2_qs))
    delta_q   = mean_s2_q - mean_s1_q
    baseline_delta = mean_s2_q - BASELINE_Q_MEAN
    passed = mean_s2_q >= (BASELINE_Q_MEAN + PRE_REGISTERED_IMPROVEMENT)

    print("\n" + "=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"  Baseline (published):    q = {BASELINE_Q_MEAN:.4f}", flush=True)
    print(f"  Stage 1 (CE):            q = {mean_s1_q:.4f} ({mean_s1_q - BASELINE_Q_MEAN:+.4f})", flush=True)
    print(f"  Stage 2 (CE + triplet):  q = {mean_s2_q:.4f} ({delta_q:+.4f} over stage1, {baseline_delta:+.4f} over baseline)", flush=True)
    print(f"  Mean kappa (stage2):     kappa = {np.mean(stage2_kappas):.4f}", flush=True)
    print(f"  PRE-REGISTERED: {'PASS' if passed else 'FAIL'} (threshold = {BASELINE_Q_MEAN + PRE_REGISTERED_IMPROVEMENT:.4f})", flush=True)

    all_results["summary"] = {
        "baseline_q": BASELINE_Q_MEAN,
        "mean_stage1_q": mean_s1_q,
        "mean_stage2_q": mean_s2_q,
        "delta_q_over_baseline": float(baseline_delta),
        "delta_q_stage2_over_stage1": float(delta_q),
        "mean_stage2_kappa": float(np.mean(stage2_kappas)),
        "passed": bool(passed),
        "threshold": float(BASELINE_Q_MEAN + PRE_REGISTERED_IMPROVEMENT),
    }

    with open("results/cti_cifar_twostage_triplet.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved: results/cti_cifar_twostage_triplet.json", flush=True)


if __name__ == "__main__":
    main()
