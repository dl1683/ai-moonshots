"""
Quick NC-Loss Pilot: 60 epochs, 3 seeds, CE vs CE+NC only.

Same architecture as full RCT but faster for quick signal.
If this shows delta_q > 0, confirms direction before 200-epoch run.

Pre-registered threshold: delta_q > 0 (sign test, 3 seeds).
Full threshold (200-epoch): delta_q >= 0.02.
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
from sklearn.model_selection import StratifiedShuffleSplit

K = 20
N_EPOCHS = 60
WARMUP_EPOCHS = 25
RAMP_END_EPOCH = 50
LAMBDA_MAX = 0.15
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
N_SEEDS = 3
PROJ_DIM = 256
MARGIN = 1.0
EMA_MOMENTUM = 0.95
CHECKPOINT_EPOCHS = [0, 25, 40, 60]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "results/cti_nc_loss_quick.json"


def get_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    ce_head = nn.Linear(512, K)
    proj_head = nn.Sequential(nn.Linear(512, PROJ_DIM), nn.BatchNorm1d(PROJ_DIM))
    model = nn.ModuleDict({'backbone': backbone, 'ce_head': ce_head, 'proj_head': proj_head})
    return model.to(DEVICE)


def coarse_label(x): return x // 5


def get_cifar_coarse():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100('data', train=True, download=False,
        transform=train_transform, target_transform=coarse_label)
    test_ds = torchvision.datasets.CIFAR100('data', train=False, download=False,
        transform=test_transform, target_transform=coarse_label)
    return train_ds, test_ds


def compute_nc_loss(z, y, class_means, K, arm, class_perm=None):
    y_nc = class_perm[y] if arm == 'shuffled_nc' else y
    mu_yi = class_means[y_nc]
    L_within = ((z - mu_yi) ** 2).mean()
    M = class_means
    G = M @ M.t()
    G_etf = torch.eye(K, device=z.device) * (1 + 1.0/(K-1)) - (1.0/(K-1)) * torch.ones(K, K, device=z.device)
    L_ETF = ((G - G_etf) ** 2).sum() / (K ** 2)
    dists = torch.cdist(M.unsqueeze(0), M.unsqueeze(0)).squeeze(0) + torch.eye(K, device=z.device) * 1e6
    L_margin = F.softplus(MARGIN - dists.min())
    return L_within + 0.5 * L_ETF + 0.5 * L_margin


def update_ema_means(class_means, z, y, K, arm, class_perm=None):
    y_nc = class_perm[y] if arm == 'shuffled_nc' else y
    with torch.no_grad():
        for c in range(K):
            mask = (y_nc == c)
            if mask.sum() > 0:
                class_means[c] = EMA_MOMENTUM * class_means[c] + (1-EMA_MOMENTUM) * z[mask].mean(0)
                class_means[c] = F.normalize(class_means[c].unsqueeze(0), dim=1).squeeze(0)
    return class_means


def compute_kappa_nearest(X, y, K=K):
    classes = np.unique(y); d = X.shape[1]
    means, within_vars = {}, []
    for c in classes:
        Xc = X[y==c]; means[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - means[c])**2, axis=1)))
    sigma_W = np.sqrt(np.mean(within_vars) / d)
    min_dist = min(np.linalg.norm(means[classes[i]] - means[classes[j]])
                   for i in range(len(classes)) for j in range(i+1, len(classes)))
    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))


def compute_q_and_kappa(model, test_ds, K=K):
    model.eval()
    loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            embs.append(model['backbone'](imgs.to(DEVICE)).cpu().numpy())
            labels.append(lbs.numpy())
    X, y = np.concatenate(embs), np.concatenate(labels)
    sss = StratifiedShuffleSplit(1, test_size=0.3, random_state=42)
    tr, te = next(sss.split(X, y))
    knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
    knn.fit(X[tr], y[tr])
    acc = float(knn.score(X[te], y[te]))
    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    kappa = compute_kappa_nearest(X, y, K=K)
    return float(q), float(kappa)


def get_lambda(epoch):
    if epoch <= WARMUP_EPOCHS: return 0.0
    elif epoch <= RAMP_END_EPOCH: return LAMBDA_MAX * (epoch - WARMUP_EPOCHS) / (RAMP_END_EPOCH - WARMUP_EPOCHS)
    else: return LAMBDA_MAX


def train_one_arm(seed, arm, train_ds, test_ds):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    rng_perm = np.random.RandomState(seed + 999)
    class_perm = torch.tensor(rng_perm.permutation(K), device=DEVICE)
    class_means = F.normalize(torch.randn(K, PROJ_DIM, device=DEVICE), dim=1)
    checkpoints = []
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        lam = get_lambda(epoch) if arm == 'nc' else 0.0
        n_b = 0; e_ce = 0; e_nc = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            h = model['backbone'](imgs)
            logits = model['ce_head'](h)
            loss_ce = ce_loss_fn(logits, labels)
            if lam > 0 and arm == 'nc':
                z_raw = model['proj_head'](h)
                z = F.normalize(z_raw, dim=1)
                class_means = update_ema_means(class_means, z.detach(), labels, K, arm)
                loss_nc = compute_nc_loss(z, labels, class_means, K, arm)
                loss = loss_ce + lam * loss_nc
            else:
                loss = loss_ce; loss_nc = torch.tensor(0.0)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            e_ce += loss_ce.item(); e_nc += (loss_nc.item() if hasattr(loss_nc, 'item') else 0); n_b += 1
        scheduler.step()
        if epoch in CHECKPOINT_EPOCHS:
            q_val, kappa_val = compute_q_and_kappa(model, test_ds)
            checkpoints.append({'epoch': epoch, 'q': q_val, 'kappa': kappa_val,
                                 'loss_ce': e_ce/n_b, 'loss_nc': e_nc/n_b, 'lambda': lam})
            print(f"  [seed={seed} arm={arm} epoch={epoch}] q={q_val:.4f} kappa={kappa_val:.4f} lam={lam:.3f}")
            sys.stdout.flush()
    final = checkpoints[-1] if checkpoints else {}
    return {'seed': seed, 'arm': arm, 'final_q': final.get('q'), 'final_kappa': final.get('kappa'), 'checkpoints': checkpoints}


def main():
    print(f"NC-Loss Quick Pilot (60 epochs, 3 seeds)")
    print(f"Device: {DEVICE}")
    train_ds, test_ds = get_cifar_coarse()
    results = {'ce': [], 'nc': []}
    for arm in ['ce', 'nc']:
        print(f"\n=== ARM: {arm} ===")
        for seed in range(N_SEEDS):
            print(f"\n--- seed={seed} ---")
            res = train_one_arm(seed, arm, train_ds, test_ds)
            print(f"  DONE: final_q={res['final_q']:.4f} final_kappa={res['final_kappa']:.4f}")
            results[arm].append(res)
            with open(RESULT_PATH, 'w') as f:
                json.dump({'status': 'running', 'results': results}, f,
                          default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    ce_qs = [r['final_q'] for r in results['ce']]
    nc_qs = [r['final_q'] for r in results['nc']]
    ce_ks = [r['final_kappa'] for r in results['ce']]
    nc_ks = [r['final_kappa'] for r in results['nc']]
    delta_q = np.mean(nc_qs) - np.mean(ce_qs)
    delta_k = np.mean(nc_ks) - np.mean(ce_ks)
    summary = {
        'status': 'complete',
        'mean_q_ce': float(np.mean(ce_qs)), 'mean_q_nc': float(np.mean(nc_qs)),
        'delta_q': float(delta_q), 'delta_kappa': float(delta_k),
        'sign_pass': bool(delta_q > 0),
        'full_threshold_pass': bool(delta_q >= 0.02),
        'results': results,
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nRESULTS: CE mean_q={np.mean(ce_qs):.4f}, NC mean_q={np.mean(nc_qs):.4f}")
    print(f"delta_q={delta_q:+.4f}, delta_kappa={delta_k:+.4f}")
    print(f"Sign test PASS: {delta_q > 0}")


if __name__ == '__main__':
    main()
