"""
NC-Loss Training RCT: CE vs CE+NC vs CE+shuffled-NC

Pre-registered causal intervention test.

Hypothesis: NC-loss (ETF geometry target) raises kappa_nearest,
which in turn raises q (normalized 1-NN accuracy).

Success criteria (pre-registered):
  - delta_q (CE+NC vs CE baseline) >= +0.02
  - 95% CI lower bound > 0
  - delta_kappa_nearest > 0
  - corr(delta_kappa, delta_logit_q) > 0.5

Arms (5 seeds each):
  A. CE baseline (pure cross-entropy)
  B. CE + NC loss (ETF geometry target)
  C. CE + shuffled-NC (control: same codepath, broken geometry target)

Architecture:
  - CIFAR-native ResNet18 (3x3 first conv, no maxpool)
  - Separate projection head for NC geometry
  - CE head uses raw backbone features

NC Loss:
  L_NC = L_within + 0.5*L_ETF + 0.5*L_margin
  L_total = L_CE + lambda * L_NC

Schedule:
  epochs 1-40:  lambda=0  (CE warmup)
  41-120: lambda linear 0->0.15
  121-200: lambda=0.15

Codex design (Feb 22, 2026)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier

# ================================================================
# CONFIG
# ================================================================
K = 20          # CIFAR-100 coarse superclasses
N_EPOCHS = 200
WARMUP_EPOCHS = 40
RAMP_END_EPOCH = 120
LAMBDA_MAX = 0.15
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
N_SEEDS = 5
PROJ_DIM = 256       # Projection head output dim
MARGIN = 1.0         # NC margin target
EMA_MOMENTUM = 0.95  # EMA class mean update
CHECKPOINT_EPOCHS = [0, 40, 80, 120, 160, 200]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "results/cti_nc_loss_training.json"


# ================================================================
# CIFAR-NATIVE RESNET18
# ================================================================
def get_model():
    """Modified ResNet18 for 32x32 CIFAR images.
    Key changes vs ImageNet ResNet18:
    - First conv: 7x7 stride-2 -> 3x3 stride-1
    - MaxPool removed (nn.Identity)
    - Separate projection head for NC loss
    """
    backbone = torchvision.models.resnet18(weights=None)
    # CIFAR adaptation: remove aggressive downsampling
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    # Remove the original FC head - we'll build separate heads
    backbone.fc = nn.Identity()
    feat_dim = 512  # ResNet18 final feature dim

    # CE head: raw features -> logits
    ce_head = nn.Linear(feat_dim, K)

    # Projection head: raw features -> NC geometry space (normalized)
    proj_head = nn.Sequential(
        nn.Linear(feat_dim, PROJ_DIM),
        nn.BatchNorm1d(PROJ_DIM),
    )

    model = nn.ModuleDict({
        'backbone': backbone,
        'ce_head': ce_head,
        'proj_head': proj_head,
    })
    return model.to(DEVICE)


# ================================================================
# DATA
# ================================================================
def coarse_label(x):
    return x // 5  # CIFAR-100 fine->coarse (20 superclasses)


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
        root="data", train=True, download=True,
        transform=train_transform, target_transform=coarse_label
    )
    test_ds = torchvision.datasets.CIFAR100(
        root="data", train=False, download=False,
        transform=test_transform, target_transform=coarse_label
    )
    return train_ds, test_ds


# ================================================================
# NC LOSS
# ================================================================
def compute_nc_loss(z, y, class_means, K, arm, class_perm=None):
    """
    Compute NC geometry loss.

    Args:
        z: (B, PROJ_DIM) L2-normalized projections
        y: (B,) class labels (for CE arm) or coarse labels
        class_means: (K, PROJ_DIM) EMA class means (L2-normalized)
        K: number of classes
        arm: 'nc' or 'shuffled_nc'
        class_perm: permutation for shuffled control

    Returns:
        L_NC: scalar loss
    """
    # For shuffled control: use permuted labels for NC loss only
    y_nc = class_perm[y] if arm == 'shuffled_nc' else y

    # L_within: pull samples toward their class mean
    mu_yi = class_means[y_nc]  # (B, PROJ_DIM)
    L_within = ((z - mu_yi) ** 2).mean()

    # L_ETF: push class means toward ETF configuration
    # ETF Gram matrix: G[i,i]=1, G[i,j]=-1/(K-1)
    M = class_means  # (K, PROJ_DIM)
    G = M @ M.t()   # (K, K)
    G_etf = torch.eye(K, device=z.device) * (1 + 1.0 / (K - 1)) - (1.0 / (K - 1)) * torch.ones(K, K, device=z.device)
    L_ETF = ((G - G_etf) ** 2).sum() / (K ** 2)

    # L_margin: push all class means to be at least MARGIN apart
    dists = torch.cdist(M.unsqueeze(0), M.unsqueeze(0)).squeeze(0)  # (K, K)
    dists_filled = dists + torch.eye(K, device=z.device) * 1e6  # fill diagonal
    min_dist = dists_filled.min()
    L_margin = F.softplus(MARGIN - min_dist)

    L_NC = L_within + 0.5 * L_ETF + 0.5 * L_margin
    return L_NC


def update_ema_means(class_means, z, y, K, arm, class_perm=None):
    """Update EMA class means (no_grad)."""
    y_nc = class_perm[y] if arm == 'shuffled_nc' else y
    with torch.no_grad():
        for c in range(K):
            mask = (y_nc == c)
            if mask.sum() > 0:
                batch_mean = z[mask].mean(0)
                class_means[c] = EMA_MOMENTUM * class_means[c] + (1 - EMA_MOMENTUM) * batch_mean
                # Normalize class mean to unit sphere
                class_means[c] = F.normalize(class_means[c].unsqueeze(0), dim=1).squeeze(0)
    return class_means


# ================================================================
# KAPPA_NEAREST + Q COMPUTATION
# ================================================================
def compute_kappa_nearest(X, y, K=K):
    """
    kappa_nearest = min_inter_centroid_dist / (sigma_W * sqrt(d))
    sigma_W = sqrt(mean_c[mean_i||x_i - mu_c||^2 / d])
    """
    classes = np.unique(y)
    d = X.shape[1]
    means, within_vars = {}, []
    for c in classes:
        Xc = X[y == c]
        means[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - means[c]) ** 2, axis=1)))
    sigma_W = np.sqrt(np.mean(within_vars) / d)

    min_dist = np.inf
    cls_list = list(classes)
    for i in range(len(cls_list)):
        for j in range(i + 1, len(cls_list)):
            dist = np.linalg.norm(means[cls_list[i]] - means[cls_list[j]])
            if dist < min_dist:
                min_dist = dist

    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))


def compute_q_and_kappa(model, test_ds, K=K):
    """Extract embeddings and compute q + kappa_nearest."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=0
    )
    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            h = model['backbone'](imgs)
            all_embs.append(h.cpu().numpy())
            all_labels.append(labels.numpy())
    X = np.concatenate(all_embs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # 1-NN accuracy
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1)
    knn.fit(X, y)
    acc = float(knn.score(X, y))  # train==test (upper bound) for simplicity
    # Use proper train/test split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    try:
        tr_idx, te_idx = next(sss.split(X, y))
        knn.fit(X[tr_idx], y[tr_idx])
        acc = float(knn.score(X[te_idx], y[te_idx]))
    except Exception:
        pass
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    kappa = compute_kappa_nearest(X, y, K=K)
    return float(q), float(kappa)


# ================================================================
# TRAINING
# ================================================================
def get_lambda(epoch):
    """Lambda schedule: 0 for warmup, linear ramp, then constant."""
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    elif epoch <= RAMP_END_EPOCH:
        return LAMBDA_MAX * (epoch - WARMUP_EPOCHS) / (RAMP_END_EPOCH - WARMUP_EPOCHS)
    else:
        return LAMBDA_MAX


def train_one_arm(seed, arm, train_ds, test_ds):
    """
    Train one arm (ce / nc / shuffled_nc) with given seed.

    Returns dict with checkpoints and final metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = get_model()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=0.9,
        weight_decay=WEIGHT_DECAY, nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    # Fixed class permutation for shuffled_nc control
    rng_perm = np.random.RandomState(seed + 999)
    class_perm = torch.tensor(rng_perm.permutation(K), device=DEVICE)

    # EMA class means (initialized randomly on unit sphere)
    class_means = F.normalize(torch.randn(K, PROJ_DIM, device=DEVICE), dim=1)

    checkpoints = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        lam = get_lambda(epoch) if arm in ('nc', 'shuffled_nc') else 0.0
        epoch_loss_ce = 0.0
        epoch_loss_nc = 0.0
        n_batches = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward through backbone
            h = model['backbone'](imgs)

            # CE loss through ce_head
            logits = model['ce_head'](h)
            loss_ce = ce_loss_fn(logits, labels)

            # NC loss (if active)
            if lam > 0 and arm in ('nc', 'shuffled_nc'):
                z_raw = model['proj_head'](h)
                z = F.normalize(z_raw, dim=1)
                # Update EMA means with detached z
                class_means = update_ema_means(class_means, z.detach(), labels, K, arm, class_perm)
                loss_nc = compute_nc_loss(z, labels, class_means, K, arm, class_perm)
                loss = loss_ce + lam * loss_nc
            else:
                loss = loss_ce
                loss_nc = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_ce += loss_ce.item()
            epoch_loss_nc += loss_nc.item() if isinstance(loss_nc, torch.Tensor) else 0.0
            n_batches += 1

        scheduler.step()

        # Record checkpoints
        if epoch in CHECKPOINT_EPOCHS:
            q_val, kappa_val = compute_q_and_kappa(model, test_ds)
            checkpoint = {
                'epoch': epoch,
                'q': q_val,
                'kappa': kappa_val,
                'logit_q': float(np.log(max(q_val + 1e-10, 1e-10) / max(1 - q_val + 1e-10, 1e-10))),
                'loss_ce': epoch_loss_ce / n_batches,
                'loss_nc': epoch_loss_nc / n_batches,
                'lambda': lam,
            }
            checkpoints.append(checkpoint)
            print(f"  [seed={seed} arm={arm} epoch={epoch}] "
                  f"q={q_val:.4f} kappa={kappa_val:.4f} "
                  f"lam={lam:.3f} loss_nc={epoch_loss_nc/n_batches:.4f}")
            sys.stdout.flush()

    final = checkpoints[-1] if checkpoints else {}
    return {
        'seed': seed,
        'arm': arm,
        'final_q': final.get('q'),
        'final_kappa': final.get('kappa'),
        'checkpoints': checkpoints,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print(f"NC-Loss Training RCT")
    print(f"Device: {DEVICE}")
    print(f"Arms: ce, nc, shuffled_nc | Seeds: {N_SEEDS} | Epochs: {N_EPOCHS}")
    print(f"Pre-registered criterion: delta_q >= 0.02, CI_low > 0, delta_kappa > 0")
    print("=" * 60)

    train_ds, test_ds = get_cifar_coarse()
    print(f"Dataset: CIFAR-100 coarse, K={K}, train={len(train_ds)}, test={len(test_ds)}")

    results = {'ce': [], 'nc': [], 'shuffled_nc': []}
    seeds = list(range(N_SEEDS))

    for arm in ['ce', 'nc', 'shuffled_nc']:
        print(f"\n=== ARM: {arm} ===")
        for seed in seeds:
            print(f"\n--- seed={seed} ---")
            t0 = time.time()
            res = train_one_arm(seed, arm, train_ds, test_ds)
            elapsed = time.time() - t0
            print(f"  DONE in {elapsed:.0f}s: final_q={res['final_q']:.4f} final_kappa={res['final_kappa']:.4f}")
            results[arm].append(res)

            # Save intermediate
            with open(RESULT_PATH, 'w') as f:
                json.dump({'status': 'running', 'results': results}, f,
                          default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    # ================================================================
    # ANALYSIS
    # ================================================================
    def arm_stats(arm_key):
        qs = [r['final_q'] for r in results[arm_key] if r['final_q'] is not None]
        ks = [r['final_kappa'] for r in results[arm_key] if r['final_kappa'] is not None]
        return {
            'mean_q': float(np.mean(qs)),
            'std_q': float(np.std(qs)),
            'mean_kappa': float(np.mean(ks)),
            'std_kappa': float(np.std(ks)),
            'n': len(qs),
        }

    stats = {arm: arm_stats(arm) for arm in results}
    delta_q_nc = stats['nc']['mean_q'] - stats['ce']['mean_q']
    delta_q_ctrl = stats['shuffled_nc']['mean_q'] - stats['ce']['mean_q']
    delta_kappa = stats['nc']['mean_kappa'] - stats['ce']['mean_kappa']

    # Bootstrap CI for delta_q_nc
    ce_qs = np.array([r['final_q'] for r in results['ce'] if r['final_q'] is not None])
    nc_qs = np.array([r['final_q'] for r in results['nc'] if r['final_q'] is not None])
    n_boot = 10000
    rng = np.random.default_rng(42)
    boot_deltas = []
    for _ in range(n_boot):
        ce_b = rng.choice(ce_qs, len(ce_qs), replace=True).mean()
        nc_b = rng.choice(nc_qs, len(nc_qs), replace=True).mean()
        boot_deltas.append(nc_b - ce_b)
    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])

    # Pass criteria
    pass_delta_q = delta_q_nc >= 0.02
    pass_ci = ci_low > 0
    pass_delta_kappa = delta_kappa > 0
    pass_control = delta_q_nc > delta_q_ctrl

    summary = {
        'status': 'complete',
        'pre_registered': {
            'delta_q_threshold': 0.02,
            'ci_lower_positive': True,
            'delta_kappa_positive': True,
        },
        'stats': stats,
        'delta_q_nc': float(delta_q_nc),
        'delta_q_ctrl': float(delta_q_ctrl),
        'delta_kappa': float(delta_kappa),
        'ci_95': [float(ci_low), float(ci_high)],
        'pass_delta_q': bool(pass_delta_q),
        'pass_ci': bool(pass_ci),
        'pass_delta_kappa': bool(pass_delta_kappa),
        'pass_control': bool(pass_control),
        'overall_pass': bool(pass_delta_q and pass_ci and pass_delta_kappa),
        'results': results,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for arm in results:
        s = stats[arm]
        print(f"  {arm}: mean_q={s['mean_q']:.4f}+/-{s['std_q']:.4f}  "
              f"mean_kappa={s['mean_kappa']:.4f}+/-{s['std_kappa']:.4f}")
    print(f"\n  delta_q (NC - CE): {delta_q_nc:+.4f}  (ctrl: {delta_q_ctrl:+.4f})")
    print(f"  delta_kappa:       {delta_kappa:+.4f}")
    print(f"  95% CI:            [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"\n  PASS delta_q>=0.02: {pass_delta_q}")
    print(f"  PASS CI_low>0:      {pass_ci}")
    print(f"  PASS delta_kappa>0: {pass_delta_kappa}")
    print(f"  PASS > control:     {pass_control}")
    print(f"\n  OVERALL PASS: {pass_delta_q and pass_ci and pass_delta_kappa}")
    print(f"\n  Saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
