"""
PROSPECTIVE ARCHITECTURE TEST: Universal Control Law on New Architecture

PRE-REGISTERED HYPOTHESIS (before running):
  For ANY CE-trained ResNet-family model on CIFAR-100 coarse (K=20):
    slope of logit(q) vs kappa_eff = A_renorm(K=20) = 1.0535 +/- 0.15

This tests architecture UNIVERSALITY of the control law.
The slope A_renorm is pre-registered from Theorem 15 (ZERO FREE PARAMETERS).
C (intercept) may vary with architecture/initialization but A_renorm should not.

DESIGN:
  - Train WideResNet-28-2 on CIFAR-100 coarse (K=20, same dataset as control law validation)
  - At checkpoints 25, 40, 60 epochs: measure kappa_nearest and d_eff DIRECTLY
  - Compute kappa_eff = sqrt(d_eff) * kappa_nearest
  - Fit slope of logit(q) vs kappa_eff (per seed)
  - Pre-registered criterion: slope in [0.85, 1.25] (20% tolerance around 1.0535)
  - Compare with ResNet-18 CE slope from control law validation (should match)

WHY THIS IS PROSPECTIVE:
  - WideResNet-28-2 has different architecture (wider, deeper, residual)
  - Different initialization, different feature maps
  - IF slope matches pre-registered A_renorm, law is architecture-universal
  - This uses DIRECT d_eff measurement (not inferred from alpha)

ARCHITECTURE DETAILS:
  WideResNet-28-2: depth=28, width_factor=2
  ~1.5M parameters (ResNet-18 has 11M) — smaller and different width
  Same CIFAR adaptations: 3x3 first conv, no maxpool
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# Pre-registered from Theorem 15
K = 20
A_RENORM_K20 = 1.0535
A_TOLERANCE = 0.15  # pre-registered: slope in [A_RENORM - tol, A_RENORM + tol]

N_EPOCHS = 60
WARMUP_EPOCHS = 25
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
N_SEEDS = 3
CHECKPOINT_EPOCHS = [25, 40, 60]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "results/cti_prospective_arch_test.json"


# ============================================================
# WideResNet-28-2 implementation
# ============================================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.is_in_equal_out = (in_planes == out_planes)
        self.shortcut = (nn.Sequential() if self.is_in_equal_out else
                         nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                   padding=0, bias=False))

    def forward(self, x):
        if not self.is_in_equal_out:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
        else:
            out = self.relu2(self.bn2(self.conv1(self.relu1(self.bn1(x)))))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + self.shortcut(x)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes if i == 0 else out_planes, out_planes,
                                stride if i == 0 else 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, width_factor=2, num_classes=20, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        nChannels = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], BasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def embed(self, x):
        """Return feature vector (before FC)."""
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        return out.view(-1, self.nChannels)

    def forward(self, x):
        return self.fc(self.embed(x))


def coarse_label(x):
    return x // 5


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
        'data', train=True, download=False,
        transform=train_transform, target_transform=coarse_label)
    test_ds = torchvision.datasets.CIFAR100(
        'data', train=False, download=False,
        transform=test_transform, target_transform=coarse_label)
    return train_ds, test_ds


def extract_all_embeddings(model, dataset):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0)
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            embs.append(model.embed(imgs.to(DEVICE)).cpu().numpy())
            labels.append(lbs.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def compute_d_eff(X, y):
    """d_eff = tr(W)^2 / tr(W^2) from within-class covariance (Gram approach)."""
    classes = np.unique(y)
    N = len(X)
    trW = 0.0
    trW2 = 0.0
    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        mu_c = Xc.mean(0)
        Xc_centered = Xc - mu_c
        trSigma_k = float(np.sum(Xc_centered ** 2)) / n_c
        trW += n_c * trSigma_k / N
        G = (Xc_centered @ Xc_centered.T) / n_c
        trSigma_k2 = float(np.sum(G ** 2))
        trW2 += (n_c / N) ** 2 * trSigma_k2
    d_eff = float(trW ** 2 / (trW2 + 1e-12))
    return d_eff


def compute_kappa_nearest(X, y):
    classes = np.unique(y)
    d = X.shape[1]
    means, within_vars = {}, []
    for c in classes:
        Xc = X[y == c]
        means[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - means[c]) ** 2, axis=1)))
    sigma_W = np.sqrt(np.mean(within_vars) / d)
    min_dist = min(
        np.linalg.norm(means[classes[i]] - means[classes[j]])
        for i in range(len(classes)) for j in range(i + 1, len(classes))
    )
    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))


def compute_q(X_test, y_test, random_state=42):
    sss = StratifiedShuffleSplit(1, test_size=0.3, random_state=random_state)
    tr, te = next(sss.split(X_test, y_test))
    knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
    knn.fit(X_test[tr], y_test[tr])
    acc = float(knn.score(X_test[te], y_test[te]))
    return (acc - 1.0 / K) / (1.0 - 1.0 / K)


def fit_control_law(checkpoints):
    """
    Fit slope of logit(q) vs kappa_eff.
    Returns slope (should be ~A_renorm = 1.0535), intercept, R2.
    """
    rows = [(ck['kappa_eff'], ck['logit_q'])
            for ck in checkpoints
            if 0 < ck.get('q', 0) < 1 and ck.get('kappa_eff', 0) > 0]
    if len(rows) < 2:
        return None, None, None
    kappa_effs = np.array([r[0] for r in rows])
    logit_qs = np.array([r[1] for r in rows])
    X_fit = np.column_stack([kappa_effs, np.ones(len(kappa_effs))])
    coeffs, _, _, _ = np.linalg.lstsq(X_fit, logit_qs, rcond=None)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    pred = coeffs[0] * kappa_effs + coeffs[1]
    ss_res = np.sum((logit_qs - pred) ** 2)
    ss_tot = np.sum((logit_qs - logit_qs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, float(r2)


def train_seed(seed, train_ds, test_ds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = WideResNet(depth=28, width_factor=2, num_classes=K).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    checkpoints = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for imgs, lbs in train_loader:
            imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbs)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            # Test embeddings for q, kappa
            X_test, y_test = extract_all_embeddings(model, test_ds)
            q_val = compute_q(X_test, y_test)
            kappa_val = compute_kappa_nearest(X_test, y_test)

            # Train embeddings for d_eff
            print(f"  [seed={seed} epoch={epoch}] q={q_val:.4f} kappa={kappa_val:.4f} "
                  f"computing d_eff...", end=' ', flush=True)
            X_train, y_train = extract_all_embeddings(model, train_ds)
            d_eff = compute_d_eff(X_train, y_train)
            kappa_eff = np.sqrt(d_eff) * kappa_val
            logit_q = float(np.log(max(q_val, 0.001) / max(1 - q_val, 0.001)))
            print(f"d_eff={d_eff:.3f} kappa_eff={kappa_eff:.4f}", flush=True)

            checkpoints.append({
                'epoch': epoch,
                'q': float(q_val),
                'kappa': float(kappa_val),
                'd_eff': float(d_eff),
                'kappa_eff': float(kappa_eff),
                'logit_q': float(logit_q),
            })

    slope, intercept, r2 = fit_control_law(checkpoints)
    print(f"  DONE: q={checkpoints[-1]['q']:.4f} slope={slope:.4f} "
          f"(target={A_RENORM_K20}) R2={r2:.4f}", flush=True)

    return {
        'seed': seed,
        'checkpoints': checkpoints,
        'final_q': checkpoints[-1]['q'] if checkpoints else None,
        'slope_control_law': slope,
        'intercept': intercept,
        'r2_control_law': r2,
        'slope_vs_target': float(abs(slope - A_RENORM_K20) / A_RENORM_K20) if slope else None,
        'slope_pass': bool(abs(slope - A_RENORM_K20) < A_TOLERANCE) if slope else False,
    }


def main():
    print("Prospective Architecture Test: WideResNet-28-2 on CIFAR-100 Coarse")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Architecture: WideResNet-28-2 (depth=28, width=2)")
    print(f"Pre-registered A_renorm(K=20) = {A_RENORM_K20}")
    print(f"Pre-registered tolerance = +-{A_TOLERANCE}")
    print(f"Pass criterion: |slope - {A_RENORM_K20}| < {A_TOLERANCE}")
    print()

    train_ds, test_ds = get_cifar_coarse()
    results = []

    for seed in range(N_SEEDS):
        print(f"\n--- seed={seed} ---")
        res = train_seed(seed, train_ds, test_ds)
        results.append(res)
        with open(RESULT_PATH, 'w') as f:
            json.dump({'status': 'running', 'results': results}, f,
                      default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    # Summary
    slopes = [r['slope_control_law'] for r in results if r['slope_control_law']]
    r2s = [r['r2_control_law'] for r in results if r['r2_control_law']]
    passes = [r['slope_pass'] for r in results]

    print("\n" + "=" * 70)
    print("SUMMARY: PROSPECTIVE ARCHITECTURE TEST")
    print("=" * 70)
    print(f"  Architecture: WideResNet-28-2")
    print(f"  Pre-registered A_renorm(K=20) = {A_RENORM_K20}")
    print()
    for res in results:
        print(f"  seed={res['seed']}: slope={res['slope_control_law']:.4f} "
              f"R2={res['r2_control_law']:.4f} PASS={res['slope_pass']}")
    print()
    if slopes:
        mean_slope = np.mean(slopes)
        print(f"  Mean slope = {mean_slope:.4f} (target = {A_RENORM_K20})")
        print(f"  Relative error = {abs(mean_slope - A_RENORM_K20) / A_RENORM_K20:.3f}")
        print(f"  Seeds passing: {sum(passes)}/{len(passes)}")
        print()
        p_overall = bool(sum(passes) >= 2 and abs(mean_slope - A_RENORM_K20) < A_TOLERANCE)
        print(f"  OVERALL PASS (2/3 seeds + mean within tolerance): {p_overall}")

    output = {
        'status': 'complete',
        'architecture': 'WideResNet-28-2',
        'K': K,
        'A_renorm_preregistered': A_RENORM_K20,
        'tolerance': A_TOLERANCE,
        'results': results,
        'summary': {
            'mean_slope': float(np.mean(slopes)) if slopes else None,
            'std_slope': float(np.std(slopes)) if slopes else None,
            'mean_r2': float(np.mean(r2s)) if r2s else None,
            'seeds_pass': int(sum(passes)),
            'OVERALL_PASS': bool(sum(passes) >= 2 and slopes and
                                  abs(np.mean(slopes) - A_RENORM_K20) < A_TOLERANCE),
        }
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == '__main__':
    main()
