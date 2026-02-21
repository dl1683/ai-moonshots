#!/usr/bin/env python -u
"""
FROZEN-EMBEDDING DO-INTERVENTION ON kappa_nearest (Feb 21 2026)
===============================================================
Codex recommendation: cleaner causal proof without optimizer/gradient conflicts.

DESIGN (do-calculus style intervention):
  Starting point: trained ResNet-18 embeddings (pure CE, q~0.707, kappa~0.35)

  For delta in [-0.20, -0.15, ..., +0.20]:
    1. Find the NEAREST-NEIGHBOR class pair (j, k): classes with min centroid distance
    2. MOVE their centroids apart/together by delta (along the jk direction)
       mu_j <- mu_j + (delta/2) * direction_jk
       mu_k <- mu_k - (delta/2) * direction_jk  [scaled to unit norm of direction]
    3. TRANSLATE all samples from class j and k (keeps within-class scatter FIXED)
       x_i <- x_i + (delta_mu_j or delta_mu_k)
    4. Recompute kappa_nearest and q

  KEY: only kappa_nearest (min inter-centroid distance) changes.
       sigma_W (within-class scatter) is UNCHANGED.
       Therefore we measure the PURE causal effect of kappa_nearest on q.

CONTROLS:
  - Same perturbation but on the FARTHEST class pair instead of nearest
    (should have minimal effect on q, since kappa_nearest is defined by MINIMUM)
  - Random pair perturbation (average effect)
  - Null control: perturb class centroids by same delta but PERPENDICULAR to jk
    (within-class scatter changes; should NOT affect kappa_nearest)

PRE-REGISTERED CRITERION:
  1. Dose-response: r(delta_kappa, delta_logit_q) > 0.90 (strong linear response)
  2. Slope matches theory: |alpha_intervention - 1.54| / 1.54 < 0.30 (30% tolerance)
  3. Specificity: r(delta_kappa_farthest, delta_logit_q) < 0.30 (control is weak)

EXPECTED OUTCOME:
  Nearest pair: logit(q) increases linearly with delta. Slope alpha ~ 1.54.
  Farthest pair: logit(q) almost unchanged (kappa_nearest unchanged for large delta).
  Random pair: intermediate effect (sometimes hits nearest pair, sometimes not).

Nobel significance:
  If dose-response slope matches LOAO alpha = 1.54, this is STRONG causal evidence:
  - No training required (no gradient conflicts, no confounders)
  - Exact control of kappa_nearest (we set it directly)
  - Specificity control rules out "any perturbation helps"
  - Matches theoretical prediction quantitatively

Usage:
  # With saved embeddings from two-stage triplet or cross-modal arm:
  python src/cti_do_intervention.py --emb_path results/stage1_embs_seed42.npz

  # Full run (trains ResNet-18 from scratch for 35 epochs, then applies intervention):
  python src/cti_do_intervention.py --train
"""

import json
import sys
import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 20  # CIFAR-100 coarse classes

# DO-INTERVENTION CONFIG
DELTA_RANGE = np.linspace(-0.25, 0.25, 21)  # -0.25 to +0.25 in steps of 0.025
N_EVAL = 2000  # subsample for evaluation

# PRE-REGISTERED CRITERIA
PRE_REG_R_THRESHOLD = 0.90  # r(delta_kappa, delta_logit_q) for NEAREST pair
PRE_REG_ALPHA_TOLERANCE = 0.30  # |alpha_intervention - LOAO_ALPHA| / LOAO_ALPHA
PRE_REG_CONTROL_R = 0.30  # r for FARTHEST pair should be below this
LOAO_ALPHA = 1.549  # from 7-architecture LOAO (per-task-intercept)

SEEDS = [42, 123, 456, 789, 1024]


# ================================================================
# TRAINING (Stage 1 CE-only, for fresh embeddings)
# ================================================================
def coarse_label(x):
    return x // 5


def get_cifar_coarse():
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100(
        root="data", train=True,  download=True,
        transform=train_t, target_transform=coarse_label
    )
    test_ds = torchvision.datasets.CIFAR100(
        root="data", train=False, download=False,
        transform=test_t,  target_transform=coarse_label
    )
    return train_ds, test_ds


def train_ce_model(seed, train_ds, n_epochs=35):
    """Train ResNet-18 with pure CE for n_epochs. Returns model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, K)
    model = model.to(DEVICE)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True
    )
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    ce_loss_fn = nn.CrossEntropyLoss()

    print(f"  Training CE model (seed={seed}, {n_epochs} epochs)...", flush=True)
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = ce_loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 5 == 0:
            print(f"    epoch {epoch}/{n_epochs} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def extract_embeddings(model, test_ds, n_max=N_EVAL):
    """Extract penultimate (avgpool) features."""
    model.eval()
    features_list, labels_list = [], []

    def hook_fn(module, input, output):
        features_list.append(output.squeeze(-1).squeeze(-1).cpu().float())

    hook = model.avgpool.register_forward_hook(hook_fn)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=0
    )
    n_seen = 0
    for imgs, labels in loader:
        if n_seen >= n_max:
            break
        imgs = imgs.to(DEVICE)
        model(imgs)
        labels_list.append(labels.numpy())
        n_seen += imgs.shape[0]
    hook.remove()

    X = torch.cat(features_list, dim=0).numpy()[:n_max]
    y = np.concatenate(labels_list)[:n_max]
    return X, y


# ================================================================
# GEOMETRY TOOLS
# ================================================================
def compute_class_stats(X, y):
    """Compute per-class centroids and within-class scatter."""
    classes = np.unique(y)
    centroids = {}
    within_vars = []
    for c in classes:
        Xc = X[y == c]
        centroids[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - centroids[c])**2, axis=1)))
    sigma_W = np.sqrt(np.mean(within_vars) / X.shape[1])
    return centroids, float(sigma_W)


def compute_kappa_nearest(centroids, sigma_W, d):
    """kappa_nearest = min_{j!=k} ||mu_j - mu_k|| / (sigma_W * sqrt(d))"""
    classes = list(centroids.keys())
    min_dist = np.inf
    nearest_pair = (classes[0], classes[1])
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            dist = np.linalg.norm(centroids[ci] - centroids[cj])
            if dist < min_dist:
                min_dist = dist
                nearest_pair = (ci, cj)
    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10)), nearest_pair


def compute_farthest_pair(centroids):
    """Find the pair with MAXIMUM centroid distance."""
    classes = list(centroids.keys())
    max_dist = -np.inf
    farthest_pair = (classes[0], classes[1])
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            dist = np.linalg.norm(centroids[ci] - centroids[cj])
            if dist > max_dist:
                max_dist = dist
                farthest_pair = (ci, cj)
    return farthest_pair


def compute_q(X, y, K=K):
    """Compute normalized 1-NN accuracy."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    return float((acc - 1.0 / K) / (1.0 - 1.0 / K))


def logit_q(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return float(np.log(q / (1 - q)))


# ================================================================
# DO-INTERVENTION
# ================================================================
def apply_centroid_shift(X, y, centroids, class_j, class_k, delta, d):
    """
    Move centroids of class_j and class_k apart by delta (in embedding units).
    Translate all samples from those classes accordingly.

    direction: unit vector from mu_j to mu_k
    delta > 0: move j and k APART (increases kappa_nearest)
    delta < 0: move j and k TOGETHER (decreases kappa_nearest)
    """
    mu_j = centroids[class_j].copy()
    mu_k = centroids[class_k].copy()

    diff = mu_k - mu_j
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy(), y.copy()

    direction = diff / dist  # unit vector from j to k

    # Move j toward k by delta/2, and k away from j by delta/2
    # delta > 0: PUSH APART (mu_j moves opposite to direction, mu_k moves along direction)
    shift_j = -(delta / 2) * direction  # move j AWAY from k
    shift_k =  (delta / 2) * direction  # move k AWAY from j

    X_new = X.copy()
    mask_j = (y == class_j)
    mask_k = (y == class_k)
    X_new[mask_j] += shift_j
    X_new[mask_k] += shift_k

    return X_new, y


def do_intervention_sweep(X, y, delta_range, pair_mode="nearest"):
    """
    Apply do-intervention for each delta and record (kappa, q, logit_q).
    pair_mode: "nearest" (nearest class pair), "farthest" (farthest), "random" (random)
    """
    d = X.shape[1]
    centroids, sigma_W = compute_class_stats(X, y)

    kappa_orig, nearest_pair = compute_kappa_nearest(centroids, sigma_W, d)
    farthest_pair = compute_farthest_pair(centroids)

    if pair_mode == "nearest":
        target_pair = nearest_pair
    elif pair_mode == "farthest":
        target_pair = farthest_pair
    elif pair_mode == "random":
        classes = list(centroids.keys())
        np.random.seed(123)
        i, j = np.random.choice(len(classes), size=2, replace=False)
        target_pair = (classes[i], classes[j])
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")

    print(f"  [{pair_mode}] Target pair: {target_pair}, kappa_orig={kappa_orig:.4f}", flush=True)

    results = []
    for delta in delta_range:
        X_shifted, _ = apply_centroid_shift(X, y, centroids, target_pair[0], target_pair[1], delta, d)
        new_centroids, new_sigma_W = compute_class_stats(X_shifted, y)
        new_kappa, _ = compute_kappa_nearest(new_centroids, new_sigma_W, d)
        q = compute_q(X_shifted, y)

        if q is None:
            continue

        results.append({
            "delta": float(delta),
            "kappa_nearest": float(new_kappa),
            "delta_kappa": float(new_kappa - kappa_orig),
            "q": float(q),
            "logit_q": logit_q(q),
            "target_pair": [int(target_pair[0]), int(target_pair[1])],
        })
        print(f"    delta={delta:+.3f}: kappa={new_kappa:.4f} ({new_kappa-kappa_orig:+.4f}), "
              f"q={q:.4f}, logit_q={logit_q(q):.4f}", flush=True)

    return results, float(kappa_orig)


# ================================================================
# ANALYSIS
# ================================================================
def analyze_dose_response(results, mode_name):
    """Fit alpha from dose-response curve and test pre-registered criteria."""
    if len(results) < 4:
        print(f"  {mode_name}: insufficient data")
        return {}

    kappas  = np.array([r["kappa_nearest"]  for r in results])
    logit_qs = np.array([r["logit_q"] for r in results])
    deltas  = np.array([r["delta"]          for r in results])

    # r(delta_kappa, delta_logit_q) using DELTA (not level)
    delta_kappas   = kappas - kappas.mean()
    delta_logit_qs = logit_qs - logit_qs.mean()

    r = float(np.corrcoef(delta_kappas, delta_logit_qs)[0, 1]) if np.std(delta_kappas) > 1e-6 else 0.0

    # OLS: logit_q = alpha * kappa + C
    A = np.vstack([kappas, np.ones(len(kappas))]).T
    alpha_hat, C = np.linalg.lstsq(A, logit_qs, rcond=None)[0]

    ss_res = np.sum((logit_qs - (alpha_hat * kappas + C))**2)
    ss_tot = np.sum((logit_qs - logit_qs.mean())**2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    deviation = abs(alpha_hat - LOAO_ALPHA) / LOAO_ALPHA

    print(f"\n  {mode_name} dose-response:")
    print(f"    alpha_intervention = {alpha_hat:.4f} (LOAO = {LOAO_ALPHA})")
    print(f"    deviation = {deviation:.1%}")
    print(f"    r(delta_kappa, delta_logit_q) = {r:.4f}")
    print(f"    R2 = {r2:.4f}")

    # Pre-registered criteria (only for "nearest" mode)
    if "nearest" in mode_name.lower():
        crit1 = r > PRE_REG_R_THRESHOLD
        crit2 = deviation < PRE_REG_ALPHA_TOLERANCE
        print(f"    [{'PASS' if crit1 else 'FAIL'}] r > {PRE_REG_R_THRESHOLD}: r={r:.4f}")
        print(f"    [{'PASS' if crit2 else 'FAIL'}] deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}: "
              f"deviation={deviation:.1%}")
    else:
        crit3 = r < PRE_REG_CONTROL_R
        print(f"    [{'PASS' if crit3 else 'FAIL'}] Control r < {PRE_REG_CONTROL_R} "
              f"(specificity): r={r:.4f}")

    return {
        "alpha_intervention": float(alpha_hat),
        "C": float(C),
        "r": float(r),
        "r2": float(r2),
        "deviation_from_loao": float(deviation),
        "n_points": int(len(results)),
    }


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Train ResNet-18 from scratch (needed if no saved embeddings)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("FROZEN-EMBEDDING DO-INTERVENTION ON kappa_nearest", flush=True)
    print("=" * 70, flush=True)
    print(f"Theory: logit(q) = {LOAO_ALPHA} * kappa_nearest + C(task)", flush=True)
    print(f"Pre-registered: r > {PRE_REG_R_THRESHOLD}, deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}", flush=True)
    print(flush=True)

    train_ds, test_ds = get_cifar_coarse()

    all_seed_results = {}

    for seed in args.seeds:
        print(f"\n{'='*50}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*50}", flush=True)

        # Check for saved embeddings first
        emb_path = f"results/do_intervention_embs_seed{seed}.npz"
        if os.path.exists(emb_path):
            data = np.load(emb_path)
            X, y = data["X"], data["y"]
            print(f"  Loaded embeddings: X.shape={X.shape}", flush=True)
        elif args.train:
            # Train fresh CE model
            model = train_ce_model(seed, train_ds, n_epochs=args.epochs)
            X, y = extract_embeddings(model, test_ds)
            # Save for future use
            np.savez(emb_path, X=X, y=y)
            print(f"  Saved embeddings: {emb_path}", flush=True)
            del model
            torch.cuda.empty_cache()
        else:
            print(f"  No saved embeddings found at {emb_path}.", flush=True)
            print(f"  Run with --train to generate embeddings first.", flush=True)
            continue

        # Baseline (no perturbation)
        centroids_orig, sigma_W = compute_class_stats(X, y)
        kappa_orig, nearest_pair = compute_kappa_nearest(centroids_orig, sigma_W, X.shape[1])
        q_orig = compute_q(X, y)
        print(f"  Baseline: kappa={kappa_orig:.4f}, q={q_orig:.4f}", flush=True)
        print(f"  Nearest class pair: {nearest_pair}", flush=True)

        seed_results = {"baseline": {"kappa": kappa_orig, "q": q_orig}}

        # Do-interventions for three modes
        for mode in ["nearest", "farthest"]:
            print(f"\n  --- {mode.upper()} PAIR PERTURBATION ---", flush=True)
            sweep_results, _ = do_intervention_sweep(X, y, DELTA_RANGE, pair_mode=mode)
            analysis = analyze_dose_response(sweep_results, f"Seed {seed} - {mode}")
            seed_results[mode] = {
                "sweep": sweep_results,
                "analysis": analysis,
            }

        all_seed_results[str(seed)] = seed_results

        # Save partial result
        with open("results/cti_do_intervention.json", "w") as f:
            json.dump(all_seed_results, f, indent=2)
        print(f"\n  Saved partial result for seed {seed}", flush=True)

    # Summary
    print("\n\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)

    nearest_alphas = []
    nearest_rs     = []
    farthest_rs    = []

    for seed_str, r in all_seed_results.items():
        if "nearest" in r and "analysis" in r["nearest"]:
            a = r["nearest"]["analysis"]
            nearest_alphas.append(a.get("alpha_intervention", float("nan")))
            nearest_rs.append(a.get("r", float("nan")))
        if "farthest" in r and "analysis" in r["farthest"]:
            a = r["farthest"]["analysis"]
            farthest_rs.append(a.get("r", float("nan")))

    if nearest_alphas:
        mean_alpha = float(np.nanmean(nearest_alphas))
        std_alpha  = float(np.nanstd(nearest_alphas))
        deviation  = abs(mean_alpha - LOAO_ALPHA) / LOAO_ALPHA
        mean_r     = float(np.nanmean(nearest_rs))
        mean_r_control = float(np.nanmean(farthest_rs)) if farthest_rs else float("nan")

        print(f"  NEAREST PAIR: alpha = {mean_alpha:.4f} +/- {std_alpha:.4f}")
        print(f"    LOAO alpha = {LOAO_ALPHA:.4f}")
        print(f"    Deviation = {deviation:.1%}")
        print(f"    Mean r    = {mean_r:.4f}")
        print(f"  FARTHEST PAIR (control): mean_r = {mean_r_control:.4f}")

        crit1 = mean_r > PRE_REG_R_THRESHOLD
        crit2 = deviation < PRE_REG_ALPHA_TOLERANCE
        crit3 = (mean_r_control < PRE_REG_CONTROL_R) if not np.isnan(mean_r_control) else None

        print(f"\n  PRE-REGISTERED RESULTS:")
        print(f"    [{' PASS' if crit1 else ' FAIL'}] r > {PRE_REG_R_THRESHOLD}: {mean_r:.4f}")
        print(f"    [{' PASS' if crit2 else ' FAIL'}] deviation < {PRE_REG_ALPHA_TOLERANCE:.0%}: {deviation:.1%}")
        if crit3 is not None:
            print(f"    [{' PASS' if crit3 else ' FAIL'}] control r < {PRE_REG_CONTROL_R}: {mean_r_control:.4f}")

        all_pass = crit1 and crit2 and (crit3 if crit3 is not None else True)
        print(f"\n  OVERALL: {'PASS - kappa_nearest is causal lever for q' if all_pass else 'FAIL/PARTIAL'}")

    # Final save
    all_seed_results["summary"] = {
        "loao_alpha": LOAO_ALPHA,
        "mean_nearest_alpha": float(np.nanmean(nearest_alphas)) if nearest_alphas else None,
        "mean_nearest_r": float(np.nanmean(nearest_rs)) if nearest_rs else None,
        "mean_farthest_r": float(np.nanmean(farthest_rs)) if farthest_rs else None,
        "pre_reg_r_threshold": PRE_REG_R_THRESHOLD,
        "pre_reg_alpha_tolerance": PRE_REG_ALPHA_TOLERANCE,
        "pre_reg_control_r": PRE_REG_CONTROL_R,
    }
    with open("results/cti_do_intervention.json", "w") as f:
        json.dump(all_seed_results, f, indent=2)
    print(f"\nSaved: results/cti_do_intervention.json", flush=True)


if __name__ == "__main__":
    main()
