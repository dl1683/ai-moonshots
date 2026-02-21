#!/usr/bin/env python -u
"""
CROSS-MODALITY FROZEN-ALPHA VALIDATION (Feb 21 2026)
=====================================================
KEY EXPERIMENT: Can the text-fitted kappa_nearest law predict
kNN quality in a VISION model without ANY refitting?

The law (fitted from text models): logit(q) = alpha * kappa_nearest + beta * log(K-1) + C
  alpha = 3.0 (from LOAO across 5 text models)
  beta  = -0.73 (from LOAO across 5 text models)

We freeze (alpha, beta) and test: does the law predict q
for ResNet-18 embeddings on CIFAR-100 WITHOUT any fine-tuning?

Pre-registered:
  - R2 > 0.5 for frozen-alpha prediction on CIFAR-100
  - OR: relative error |predicted_q - actual_q| / actual_q < 0.20

If this passes: the law is truly cross-modal (text + vision, same law).
If this fails: the law is architecture-universal but NOT modality-universal.

Both results are scientifically valuable.
"""

import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# FROZEN PARAMETERS FROM TEXT EXPERIMENTS
# ================================================================
FROZEN_ALPHA = 3.0    # from LOAO across 5 text model architectures, CV=0.061
FROZEN_BETA  = -0.73  # from LOAO across 5 text model architectures, CV=0.041
FROZEN_C0    = 0.70   # from global fit

PRE_REG_R2_THRESHOLD = 0.5
PRE_REG_REL_ERROR = 0.20  # 20% tolerance

# ================================================================
# CONFIG
# ================================================================
SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS_TO_TEST = [5, 15, 25, 35]  # test law at multiple training points
BATCH_SIZE = 256
K = 20  # CIFAR-100 coarse classes


# ================================================================
# DATA
# ================================================================
def _coarse_label(x):
    return x // 5


def get_cifar_coarse():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = torchvision.datasets.CIFAR100(
        root="/tmp/cifar100", train=False, download=True, transform=transform
    )
    testset.targets = [_coarse_label(t) for t in testset.targets]
    return testset


# ================================================================
# MODEL
# ================================================================
def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, 20)
    return model.to(DEVICE)


# ================================================================
# COMPUTE METRICS
# ================================================================
@torch.no_grad()
def get_embeddings(model, dataloader, n_max=2000):
    """Extract 512-dim penultimate features from ResNet-18."""
    model.eval()
    # Hook penultimate layer
    features_list = []
    labels_list = []

    def hook_fn(module, input, output):
        features_list.append(input[0].cpu().numpy())

    hook = model.fc.register_forward_hook(hook_fn)

    count = 0
    for x, y in dataloader:
        if count >= n_max:
            break
        x = x.to(DEVICE)
        _ = model(x)
        labels_list.extend(y.numpy())
        count += len(y)

    hook.remove()

    features = np.vstack(features_list)[:n_max]
    labels = np.array(labels_list[:n_max])
    return features, labels


def compute_kappa_nearest(X, y, K):
    """kappa_nearest = min_{j!=k} ||mu_k-mu_j|| / (sigma_W * sqrt(d))"""
    classes = np.unique(y)
    K_eff = len(classes)
    mu = {k: X[y == k].mean(0) for k in classes}

    total_var = sum(np.sum((X[y == k] - mu[k])**2) for k in classes)
    sigma_W = float(np.sqrt(total_var / (len(X) * X.shape[1])))
    if sigma_W < 1e-10:
        return None, None

    all_kappa = []
    for k in classes:
        min_dist = min(
            np.linalg.norm(mu[k] - mu[j])
            for j in classes if j != k
        )
        all_kappa.append(min_dist / (sigma_W * np.sqrt(X.shape[1])))

    return float(np.mean(all_kappa)), sigma_W


def compute_knn_q(X, y, K_eff):
    """1-NN q_norm."""
    if len(X) > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), 2000, replace=False)
        X, y = X[idx], y[idx]

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    return (acc - 1.0 / K_eff) / (1.0 - 1.0 / K_eff)


# ================================================================
# TRAIN AND EVALUATE LAW
# ================================================================
def train_and_eval_law(seed, testloader):
    """Train ResNet-18 CE for N_EPOCHS, measure kappa->q law at multiple checkpoints."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)

    # Training data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root="/tmp/cifar100", train=True, download=True, transform=transform_train
    )
    trainset.targets = [_coarse_label(t) for t in trainset.targets]
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    checkpoints = []
    max_epoch = max(N_EPOCHS_TO_TEST)

    for epoch in range(1, max_epoch + 1):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch in N_EPOCHS_TO_TEST:
            # Evaluate at this checkpoint
            X_emb, y_emb = get_embeddings(model, testloader)
            kappa, sigma_W = compute_kappa_nearest(X_emb, y_emb, K)
            q = compute_knn_q(X_emb, y_emb, K)

            if kappa is not None and q is not None and q > 0:
                logit_q_actual = float(np.log(max(q, 0.001) / max(1 - q, 0.001)))
                logKm1 = float(np.log(K - 1))

                # Frozen-parameter prediction
                logit_q_predicted = FROZEN_ALPHA * kappa + FROZEN_BETA * logKm1 + FROZEN_C0
                q_predicted = float(1.0 / (1.0 + np.exp(-logit_q_predicted)))

                pt = {
                    "seed": seed, "epoch": epoch,
                    "kappa_nearest": kappa, "q_actual": q,
                    "logit_q_actual": logit_q_actual,
                    "logit_q_predicted": float(logit_q_predicted),
                    "q_predicted": q_predicted,
                    "q_error": float(abs(q_predicted - q)),
                    "q_rel_error": float(abs(q_predicted - q) / max(q, 0.01)),
                    "logit_error": float(abs(logit_q_predicted - logit_q_actual)),
                }
                checkpoints.append(pt)

                print(f"  Epoch {epoch:2d}: kappa={kappa:.4f}  "
                      f"q_actual={q:.4f}  q_pred={q_predicted:.4f}  "
                      f"rel_err={pt['q_rel_error']:.3f}", flush=True)

    del model
    torch.cuda.empty_cache()
    return checkpoints


# ================================================================
# MAIN
# ================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("CROSS-MODALITY FROZEN-ALPHA VALIDATION")
    print(f"Frozen law: logit(q) = {FROZEN_ALPHA} * kappa + {FROZEN_BETA} * log(K-1) + {FROZEN_C0}")
    print(f"Parameters from: LOAO across 5 text model architectures")
    print(f"Test on: ResNet-18 + CIFAR-100 coarse (20 classes)")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    testset = get_cifar_coarse()
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    all_checkpoints = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        checkpoints = train_and_eval_law(seed, testloader)
        all_checkpoints.extend(checkpoints)

    # Analysis: Frozen-parameter prediction accuracy
    print("\n" + "=" * 70)
    print("CROSS-MODALITY LAW VALIDATION RESULTS")
    print("=" * 70)

    if not all_checkpoints:
        print("No valid checkpoints")
        return

    kappa_arr = np.array([p["kappa_nearest"] for p in all_checkpoints])
    logit_actual = np.array([p["logit_q_actual"] for p in all_checkpoints])
    logit_pred = np.array([p["logit_q_predicted"] for p in all_checkpoints])
    rel_errors = np.array([p["q_rel_error"] for p in all_checkpoints])

    # R2 of frozen prediction
    ss_tot = np.sum((logit_actual - logit_actual.mean())**2)
    ss_res = np.sum((logit_actual - logit_pred)**2)
    r2_frozen = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    # Also fit free alpha for comparison
    A = np.column_stack([kappa_arr, np.ones(len(kappa_arr))])
    c, _, _, _ = np.linalg.lstsq(A, logit_actual, rcond=None)
    logit_free = A @ c
    ss_res_free = np.sum((logit_actual - logit_free)**2)
    r2_free = float(1 - ss_res_free / ss_tot) if ss_tot > 1e-10 else 0.0

    mean_rel_error = float(np.mean(rel_errors))
    print(f"  N checkpoints: {len(all_checkpoints)}")
    print(f"  R2 (frozen alpha={FROZEN_ALPHA}): {r2_frozen:.3f}  "
          f"{'PASS' if r2_frozen > PRE_REG_R2_THRESHOLD else 'FAIL'}")
    print(f"  R2 (free alpha fit): {r2_free:.3f}  "
          f"fitted_alpha={c[0]:.3f}")
    print(f"  Mean rel error: {mean_rel_error:.3f}  "
          f"{'PASS' if mean_rel_error < PRE_REG_REL_ERROR else 'FAIL'}")
    print(f"  Cross-modality: alpha_text={FROZEN_ALPHA:.1f}  alpha_vision={c[0]:.3f}  "
          f"ratio={c[0]/FROZEN_ALPHA:.3f}")

    # By epoch
    print("\n  By training epoch:")
    for ep in N_EPOCHS_TO_TEST:
        ep_pts = [p for p in all_checkpoints if p["epoch"] == ep]
        if not ep_pts:
            continue
        mean_q = np.mean([p["q_actual"] for p in ep_pts])
        mean_pred = np.mean([p["q_predicted"] for p in ep_pts])
        mean_err = np.mean([p["q_rel_error"] for p in ep_pts])
        print(f"    Epoch {ep:2d}: mean_q={mean_q:.4f}  mean_pred={mean_pred:.4f}  "
              f"rel_err={mean_err:.3f}")

    # Save
    output = {
        "experiment": "cross_modality_frozen_alpha_validation",
        "frozen_params": {
            "alpha": FROZEN_ALPHA, "beta": FROZEN_BETA, "C0": FROZEN_C0,
            "source": "LOAO across 5 text model architectures, CV=0.061"
        },
        "pre_registered": {
            "r2_threshold": PRE_REG_R2_THRESHOLD,
            "rel_error_threshold": PRE_REG_REL_ERROR
        },
        "all_checkpoints": all_checkpoints,
        "results": {
            "r2_frozen": r2_frozen,
            "r2_free": r2_free,
            "free_alpha": float(c[0]),
            "mean_rel_error": mean_rel_error,
            "n": len(all_checkpoints),
            "cross_modal_alpha_ratio": float(c[0] / FROZEN_ALPHA),
        },
        "pass_r2": bool(r2_frozen > PRE_REG_R2_THRESHOLD),
        "pass_rel_error": bool(mean_rel_error < PRE_REG_REL_ERROR),
        "runtime_s": int(time.time() - t0)
    }

    out_path = "results/cti_cifar_crossmodal.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")
    print(f"Runtime: {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
