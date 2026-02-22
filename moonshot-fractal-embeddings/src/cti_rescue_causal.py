"""
Orthogonal Intervention + Rescue Study (Codex Recommendation, Feb 23 2026)
===========================================================================

CODEX RECOMMENDATION: "The single highest-impact experiment for Nobel credibility."

THE KEY IDEA:
  The law logit(q) = A_renorm * sqrt(d_eff_sig) * kappa_nearest + C says that
  kappa and d_eff_sig are SUBSTITUTABLE: you can trade one for the other while
  holding q constant. The rescue arm TESTS this substitutability directly.

PRE-REGISTERED DESIGN:
  A_renorm = 1.0535 (Theorem 15, ZERO FREE PARAMETERS)
  C = intercept fit from CE arm ONLY (1 free parameter, shared across all arms)

  Prediction: logit(q_arm) = A_renorm * sqrt(d_eff_sig_arm) * kappa_arm + C_CE
  for ALL arms, with NO per-arm refitting.

  This is the quantitative, falsifiable test: not just "direction is right" but
  "the product A_renorm * sqrt(d_eff_sig) * kappa exactly predicts logit(q)
  across arms with different kappa and d_eff_sig combinations."

ARMS (6 total, 2 seeds each = 12 runs):
  ce:          Baseline CE. kappa~0.84, d_eff_sig~1.46, product~1.01
  margin_low:  CE + L_margin(lam=0.05). kappa UP, d_eff_sig ~same. product UP.
  margin_high: CE + L_margin(lam=0.15). kappa UP more. product UP more.
  etf_low:     CE + L_ETF(lam=0.05). d_eff_sig UP, kappa ~same. product UP.
  etf_high:    CE + L_ETF(lam=0.15). d_eff_sig UP more. product UP more.
  rescue:      CE + L_margin(lam=0.15) + L_within(lam=0.10).
               kappa UP (margin) AND d_eff_sig DOWN (within-class collapse).
               NET: product ~= CE baseline (substitutability test).

RESCUE ARM PREDICTION (pre-registered):
  logit(q_rescue) ~= logit(q_CE)  [despite different kappa AND d_eff_sig]

ISO-PRODUCT INVARIANCE:
  For any two conditions (kappa1, d_eff_sig1) and (kappa2, d_eff_sig2) with
  sqrt(d_eff_sig1)*kappa1 ~= sqrt(d_eff_sig2)*kappa2,
  we predict logit(q1) ~= logit(q2).

WHAT CONFIRMS THE LAW:
  Pass: R2 > 0.85 for ALL arms combined, slope fixed at A_renorm=1.0535, C_CE only
  Rescue pass: |logit(q_rescue) - logit(q_CE)| < 0.1 logit units
  If both pass: Nobel score jumps from 5.5 to ~7.5/10 (Codex estimate)

WHAT REFUTES:
  Fail: R2 < 0.5 OR rescue arm q differs significantly from CE
  -> Would require revising the product-form law or d_eff_sig definition
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

# Hyperparameters
K = 20
N_EPOCHS = 60
WARMUP_EPOCHS = 20
RAMP_END_EPOCH = 45
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
N_SEEDS = 2  # 2 seeds x 6 arms = 12 runs
PROJ_DIM = 256
MARGIN = 1.0
EMA_MOMENTUM = 0.95
CHECKPOINT_EPOCHS = [25, 40, 60]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "results/cti_rescue_causal.json"
LOG_PATH = "results/cti_rescue_causal_log.txt"
A_RENORM_K20 = 1.0535  # Theorem 15 pre-registered constant

# ARM CONFIGURATIONS: (lambda_margin, lambda_etf, lambda_within)
# lambda_margin: L_margin = softplus(MARGIN - min_dist), increases kappa
# lambda_etf:    L_ETF = ||G - G_etf||^2 / K^2, restructures toward simplex (d_eff_sig UP)
# lambda_within: L_within = ||z - mu_y||^2, pulls toward centroid (d_eff_sig DOWN, kappa DOWN)
ARM_CONFIGS = {
    'ce':           (0.00, 0.00, 0.00),  # baseline
    'margin_low':   (0.05, 0.00, 0.00),  # kappa lever low
    'margin_high':  (0.15, 0.00, 0.00),  # kappa lever high
    'etf_low':      (0.00, 0.05, 0.00),  # d_eff_sig lever low
    'etf_high':     (0.00, 0.15, 0.00),  # d_eff_sig lever high
    'rescue':       (0.15, 0.00, 0.10),  # kappa UP + d_eff_sig DOWN -> product ~= CE
}


def log(msg, path=LOG_PATH):
    print(msg, flush=True)
    with open(path, 'a') as f:
        f.write(msg + '\n')


def get_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    ce_head = nn.Linear(512, K)
    proj_head = nn.Sequential(nn.Linear(512, PROJ_DIM), nn.BatchNorm1d(PROJ_DIM))
    model = nn.ModuleDict({'backbone': backbone, 'ce_head': ce_head, 'proj_head': proj_head})
    return model.to(DEVICE)


def coarse_label(x):
    return x // 5


def get_cifar_coarse():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
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
            embs.append(model['backbone'](imgs.to(DEVICE)).cpu().numpy())
            labels.append(lbs.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def compute_d_eff_sig(X, y):
    """Signal-subspace participation ratio (Theorem 16, CORRECTED)."""
    classes = np.unique(y)
    K_actual = len(classes)
    N = len(X)
    d = X.shape[1]
    grand_mean = X.mean(0)
    centroids = np.stack([X[y == c].mean(0) for c in classes])
    centroids_centered = centroids - grand_mean
    U, S, Vt = np.linalg.svd(centroids_centered, full_matrices=False)
    n_sig = max(1, min(K_actual - 1, d, len(S)))
    P_B = Vt[:n_sig, :]  # (n_sig, d)
    W_sig = np.zeros((n_sig, n_sig), dtype=np.float64)
    trW_sig = 0.0
    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        Xc_centered = (Xc - Xc.mean(0)).astype(np.float64)
        Xc_proj = Xc_centered @ P_B.T
        Sigma_c_sig = (Xc_proj.T @ Xc_proj) / n_c
        W_sig += (n_c / N) * Sigma_c_sig
        trW_sig += (n_c / N) * float(np.trace(Sigma_c_sig))
    trW2_sig = float(np.sum(W_sig ** 2))
    return float(trW_sig ** 2 / (trW2_sig + 1e-12)), float(trW_sig), int(n_sig)


def compute_d_eff_gram(X, y):
    """Global participation ratio (includes all d dimensions, for comparison)."""
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
        trW += n_c * trSigma_k / len(X)
        G = (Xc_centered @ Xc_centered.T) / n_c
        trSigma_k2 = float(np.sum(G ** 2))
        trW2 += (n_c / N) ** 2 * trSigma_k2
    return float(trW ** 2 / (trW2 + 1e-12)), float(trW)


def compute_kappa_nearest(X, y):
    classes = np.unique(y)
    d = X.shape[1]
    means = {c: X[y == c].mean(0) for c in classes}
    within_vars = [np.mean(np.sum((X[y == c] - means[c]) ** 2, axis=1)) for c in classes]
    sigma_W = np.sqrt(np.mean(within_vars) / d)
    min_dist = min(
        np.linalg.norm(means[classes[i]] - means[classes[j]])
        for i in range(len(classes)) for j in range(i + 1, len(classes))
    )
    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))


def compute_q(X, y, random_state=42):
    sss = StratifiedShuffleSplit(1, test_size=0.3, random_state=random_state)
    tr, te = next(sss.split(X, y))
    knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
    knn.fit(X[tr], y[tr])
    acc = float(knn.score(X[te], y[te]))
    return (acc - 1.0 / K) / (1.0 - 1.0 / K)


def compute_aux_loss(arm, z, y, class_means):
    lam_margin, lam_etf, lam_within = ARM_CONFIGS[arm]
    total_loss = None

    if lam_within > 0:
        # Pull-to-centroid loss: reduces within-class variance AND concentrates signal
        mu_yi = class_means[y]
        L_within = ((z - mu_yi) ** 2).mean()
        total_loss = (total_loss or 0) + lam_within * L_within

    if lam_margin > 0 or lam_etf > 0:
        M = F.normalize(class_means, dim=1)  # (K, PROJ_DIM)
        dists = torch.cdist(M.unsqueeze(0), M.unsqueeze(0)).squeeze(0)
        mask = torch.eye(K, device=z.device).bool()
        dists_off = dists.masked_fill(mask, float('inf'))

        if lam_margin > 0:
            # Margin loss: pushes minimum centroid distance up -> kappa UP
            L_margin = F.softplus(MARGIN - dists_off.min())
            total_loss = (total_loss or 0) + lam_margin * L_margin

        if lam_etf > 0:
            # ETF loss: restructures centroids toward equidistant simplex -> d_eff_sig UP
            G = M @ M.t()
            G_etf = (torch.eye(K, device=z.device) * (1 + 1.0 / (K - 1))
                     - (1.0 / (K - 1)) * torch.ones(K, K, device=z.device))
            L_ETF = ((G - G_etf) ** 2).sum() / (K ** 2)
            total_loss = (total_loss or 0) + lam_etf * L_ETF

    return total_loss


def update_ema_means(class_means, z, y):
    with torch.no_grad():
        for c in range(K):
            mask = (y == c)
            if mask.sum() > 0:
                class_means[c] = (EMA_MOMENTUM * class_means[c]
                                  + (1 - EMA_MOMENTUM) * z[mask].mean(0))
                class_means[c] = F.normalize(class_means[c].unsqueeze(0), dim=1).squeeze(0)
    return class_means


def get_lambda_scale(epoch):
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    elif epoch <= RAMP_END_EPOCH:
        return (epoch - WARMUP_EPOCHS) / (RAMP_END_EPOCH - WARMUP_EPOCHS)
    else:
        return 1.0


def train_one_arm(seed, arm, train_ds, test_ds):
    lam_margin, lam_etf, lam_within = ARM_CONFIGS[arm]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = get_model()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=0.9,
        weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    class_means = F.normalize(torch.randn(K, PROJ_DIM, device=DEVICE), dim=1)
    checkpoints = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        lam_scale = get_lambda_scale(epoch)
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                emb = model['backbone'](imgs)
            logits = model['ce_head'](emb)
            z = F.normalize(model['proj_head'](emb), dim=1)
            loss = ce_loss_fn(logits, labels)
            if arm != 'ce' and lam_scale > 0:
                # Temporarily scale ARM_CONFIGS by lam_scale
                old_cfg = ARM_CONFIGS[arm]
                ARM_CONFIGS[arm] = tuple(v * lam_scale for v in old_cfg)
                aux = compute_aux_loss(arm, z, labels, class_means)
                ARM_CONFIGS[arm] = old_cfg
                if aux is not None:
                    loss = loss + aux
            class_means = update_ema_means(class_means, z, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            model.eval()
            log(f"  [seed={seed} arm={arm} epoch={epoch}] extracting train embs...")
            X_tr, y_tr = extract_all_embeddings(model, train_ds)
            X_te, y_te = extract_all_embeddings(model, test_ds)

            q_val = compute_q(X_te, y_te)
            kappa_val = compute_kappa_nearest(X_tr, y_tr)
            d_eff_sig_val, trW_sig, n_sig = compute_d_eff_sig(X_tr, y_tr)
            d_eff_gram_val, _ = compute_d_eff_gram(X_tr, y_tr)
            kappa_eff_sig = np.sqrt(d_eff_sig_val) * kappa_val
            logit_q = float(np.log(q_val / (1 - q_val + 1e-10) + 1e-10)
                            if 0 < q_val < 1 else (4.0 if q_val >= 1 else -4.0))
            pred_logit = A_RENORM_K20 * kappa_eff_sig  # prediction (no C, for comparison)

            log(f"  [seed={seed} arm={arm} epoch={epoch}] "
                f"q={q_val:.4f} kappa={kappa_val:.4f} "
                f"d_eff_sig={d_eff_sig_val:.3f} d_eff_gram={d_eff_gram_val:.1f} "
                f"kappa_eff_sig={kappa_eff_sig:.4f} "
                f"logit_q={logit_q:.4f} pred_logit(noC)={pred_logit:.4f}")

            checkpoints.append({
                'epoch': epoch,
                'q': q_val,
                'kappa': kappa_val,
                'd_eff_sig': d_eff_sig_val,
                'd_eff_gram': d_eff_gram_val,
                'kappa_eff_sig': kappa_eff_sig,
                'logit_q': logit_q,
                'trW_sig': float(trW_sig),
                'n_sig': n_sig,
                'lambda_margin': lam_margin,
                'lambda_etf': lam_etf,
                'lambda_within': lam_within,
            })
            model.train()

    final_ck = checkpoints[-1] if checkpoints else {}
    log(f"  DONE [seed={seed} arm={arm}]: q={final_ck.get('q', 0):.4f} "
        f"kappa={final_ck.get('kappa', 0):.4f} "
        f"d_eff_sig={final_ck.get('d_eff_sig', 0):.3f} "
        f"kappa_eff_sig={final_ck.get('kappa_eff_sig', 0):.4f}")

    return {
        'seed': seed, 'arm': arm,
        'lam_margin': lam_margin, 'lam_etf': lam_etf, 'lam_within': lam_within,
        'final_q': final_ck.get('q', 0),
        'final_kappa': final_ck.get('kappa', 0),
        'final_d_eff_sig': final_ck.get('d_eff_sig', 0),
        'final_d_eff_gram': final_ck.get('d_eff_gram', 0),
        'final_kappa_eff_sig': final_ck.get('kappa_eff_sig', 0),
        'final_logit_q': final_ck.get('logit_q', 0),
        'checkpoints': checkpoints,
    }


def analyze_rescue_results(all_results):
    """
    Pre-registered analysis:
    1. Fit C from CE arm alone (1 parameter)
    2. Predict logit(q) = A_renorm * kappa_eff_sig + C for ALL arms (0 additional params)
    3. Compute R2 and RMSE
    4. Rescue test: |logit(q_rescue) - logit(q_CE)| < 0.1
    5. Iso-product invariance: pairs with similar kappa_eff_sig should have similar logit_q
    """
    all_snaps = []
    for arm, arm_res in all_results.items():
        for res in arm_res:
            for ck in res.get('checkpoints', []):
                all_snaps.append({
                    'arm': arm, 'seed': res['seed'],
                    'kappa_eff_sig': ck['kappa_eff_sig'],
                    'logit_q': ck['logit_q'],
                    'q': ck['q'], 'kappa': ck['kappa'],
                    'd_eff_sig': ck['d_eff_sig'],
                })

    if len(all_snaps) < 3:
        return {'status': 'insufficient_data'}

    kappa_eff_sigs = np.array([s['kappa_eff_sig'] for s in all_snaps])
    logit_qs = np.array([s['logit_q'] for s in all_snaps])

    # Step 1: Fit C from CE arm only
    ce_snaps = [s for s in all_snaps if s['arm'] == 'ce']
    if ce_snaps:
        ce_ke = np.array([s['kappa_eff_sig'] for s in ce_snaps])
        ce_lq = np.array([s['logit_q'] for s in ce_snaps])
        C_ce = float(np.mean(ce_lq - A_RENORM_K20 * ce_ke))
    else:
        C_ce = float(np.mean(logit_qs - A_RENORM_K20 * kappa_eff_sigs))

    # Step 2: Predict ALL arms (zero additional free params)
    preds = A_RENORM_K20 * kappa_eff_sigs + C_ce
    residuals = logit_qs - preds
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((logit_qs - logit_qs.mean()) ** 2))
    r2_all = 1 - ss_res / (ss_tot + 1e-10)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Step 3: Per-arm analysis
    per_arm = {}
    for arm in all_results:
        arm_snaps = [s for s in all_snaps if s['arm'] == arm]
        if arm_snaps:
            arm_ke = np.array([s['kappa_eff_sig'] for s in arm_snaps])
            arm_lq = np.array([s['logit_q'] for s in arm_snaps])
            arm_preds = A_RENORM_K20 * arm_ke + C_ce
            arm_resid = arm_lq - arm_preds
            per_arm[arm] = {
                'n': len(arm_snaps),
                'mean_q': float(np.mean([s['q'] for s in arm_snaps])),
                'mean_kappa_eff_sig': float(np.mean(arm_ke)),
                'mean_d_eff_sig': float(np.mean([s['d_eff_sig'] for s in arm_snaps])),
                'mean_kappa': float(np.mean([s['kappa'] for s in arm_snaps])),
                'mean_logit_q': float(np.mean(arm_lq)),
                'mean_pred': float(np.mean(arm_preds)),
                'mean_residual': float(np.mean(arm_resid)),
                'max_abs_residual': float(np.max(np.abs(arm_resid))),
            }

    # Step 4: Rescue test
    rescue_test = None
    if 'rescue' in per_arm and 'ce' in per_arm:
        rescue_logit = per_arm['rescue']['mean_logit_q']
        ce_logit = per_arm['ce']['mean_logit_q']
        delta_logit = abs(rescue_logit - ce_logit)
        pred_delta = abs(per_arm['rescue']['mean_kappa_eff_sig']
                         - per_arm['ce']['mean_kappa_eff_sig']) * A_RENORM_K20
        rescue_test = {
            'logit_rescue': rescue_logit,
            'logit_ce': ce_logit,
            'actual_delta_logit': float(rescue_logit - ce_logit),
            'abs_delta_logit': float(delta_logit),
            'PASS_threshold': 0.1,
            'PASS': bool(delta_logit < 0.1),
            'pred_delta_from_product': float(pred_delta),
        }

    # Step 5: Iso-product invariance
    # Find pairs with kappa_eff_sig within 0.05 of each other from DIFFERENT arms
    iso_pairs = []
    for i, s1 in enumerate(all_snaps):
        for j, s2 in enumerate(all_snaps[i+1:], i+1):
            if s1['arm'] != s2['arm']:
                delta_ke = abs(s1['kappa_eff_sig'] - s2['kappa_eff_sig'])
                if delta_ke < 0.05:
                    delta_lq = abs(s1['logit_q'] - s2['logit_q'])
                    iso_pairs.append({
                        'arm1': s1['arm'], 'arm2': s2['arm'],
                        'ke1': s1['kappa_eff_sig'], 'ke2': s2['kappa_eff_sig'],
                        'lq1': s1['logit_q'], 'lq2': s2['logit_q'],
                        'delta_ke': float(delta_ke), 'delta_lq': float(delta_lq),
                        'consistent': bool(delta_lq < 0.1),
                    })

    return {
        'C_ce': float(C_ce),
        'r2_all_arms': float(r2_all),
        'rmse_all_arms': float(rmse),
        'n_total_snaps': len(all_snaps),
        'PASS_r2': bool(r2_all > 0.85),
        'PASS_rmse': bool(rmse < 0.15),
        'per_arm': per_arm,
        'rescue_test': rescue_test,
        'iso_pairs': iso_pairs[:20],  # first 20 iso-product pairs
        'n_iso_pairs': len(iso_pairs),
        'iso_consistency_rate': float(
            np.mean([p['consistent'] for p in iso_pairs])
            if iso_pairs else float('nan')
        ),
    }


def main():
    log("=" * 70)
    log("Orthogonal Intervention + Rescue Study")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, N_SEEDS={N_SEEDS}")
    log(f"A_renorm={A_RENORM_K20} (pre-registered, zero free params)")
    log(f"Arms: {list(ARM_CONFIGS.keys())}")
    log("")
    log("PRE-REGISTERED PREDICTIONS:")
    log(f"  1. R2 > 0.85 for ALL arms with slope=A_renorm, C=CE-fit-only")
    log(f"  2. RESCUE arm: |logit(q_rescue) - logit(q_CE)| < 0.1 logit units")
    log(f"  3. Iso-product pairs (delta_kappa_eff_sig < 0.05) have delta_logit < 0.1")
    log("")

    train_ds, test_ds = get_cifar_coarse()

    all_results = {arm: [] for arm in ARM_CONFIGS}
    result = {'status': 'running', 'results': all_results}

    if os.path.exists(RESULT_PATH):
        try:
            with open(RESULT_PATH) as f:
                result = json.load(f)
            all_results = result.get('results', {arm: [] for arm in ARM_CONFIGS})
            log(f"Resuming from {RESULT_PATH}")
        except Exception:
            pass

    def save():
        result['results'] = all_results
        with open(RESULT_PATH, 'w') as f:
            json.dump(result, f)

    for arm in ARM_CONFIGS:
        done_seeds = {r['seed'] for r in all_results.get(arm, [])}
        log(f"\n=== ARM: {arm} (lam_margin={ARM_CONFIGS[arm][0]}, "
            f"lam_etf={ARM_CONFIGS[arm][1]}, lam_within={ARM_CONFIGS[arm][2]}) ===")
        for seed in range(N_SEEDS):
            if seed in done_seeds:
                log(f"  seed={seed}: already done, skipping")
                continue
            log(f"\n--- seed={seed} ---")
            res = train_one_arm(seed, arm, train_ds, test_ds)
            if arm not in all_results:
                all_results[arm] = []
            all_results[arm].append(res)
            save()

    # Final analysis
    log("\n" + "=" * 70)
    log("RESCUE CAUSAL ANALYSIS (PRE-REGISTERED)")
    log("=" * 70)
    analysis = analyze_rescue_results(all_results)
    result['analysis'] = analysis
    result['status'] = 'done'
    save()

    log(f"\nR2 (ALL arms, slope=A_renorm={A_RENORM_K20}, C from CE): "
        f"{analysis['r2_all_arms']:.4f} {'PASS' if analysis['PASS_r2'] else 'FAIL'}")
    log(f"RMSE: {analysis['rmse_all_arms']:.4f} {'PASS' if analysis['PASS_rmse'] else 'FAIL'}")
    log(f"C_ce (intercept from CE arm): {analysis['C_ce']:.4f}")

    if analysis.get('rescue_test'):
        rt = analysis['rescue_test']
        log(f"\nRESCUE ARM TEST:")
        log(f"  logit(q_rescue) = {rt['logit_rescue']:.4f}")
        log(f"  logit(q_CE)     = {rt['logit_ce']:.4f}")
        log(f"  |delta_logit|   = {rt['abs_delta_logit']:.4f} (threshold: 0.1)")
        log(f"  PASS: {rt['PASS']}")

    log(f"\nIso-product pairs found: {analysis['n_iso_pairs']}")
    if analysis['n_iso_pairs'] > 0:
        log(f"  Consistency rate (delta_logit < 0.1): "
            f"{analysis['iso_consistency_rate']:.3f}")

    log(f"\nPer-arm summary:")
    for arm, stats in analysis.get('per_arm', {}).items():
        log(f"  {arm:15s}: q={stats['mean_q']:.4f} kappa={stats['mean_kappa']:.4f} "
            f"d_eff_sig={stats['mean_d_eff_sig']:.3f} "
            f"kappa_eff_sig={stats['mean_kappa_eff_sig']:.4f} "
            f"resid={stats['mean_residual']:+.4f}")

    log("\nDONE.")


if __name__ == '__main__':
    main()
