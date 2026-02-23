"""
Direct N_eff pairwise measurement from full centroid geometry.

This computes N_eff empirically by:
1. Extracting all K class centroids from real embeddings
2. Computing all K*(K-1)/2 pairwise normalized distances kappa_kj
3. For each class k: N_eff_k = sum_{j!=k} exp(-alpha * (kappa_kj - kappa_nearest_k))
4. Testing if mean N_eff ~ sqrt(K-1) across K subsets

Uses pythia-160m on dbpedia (K=14), subsampling classes to get K in {4,6,8,10,12,14}.

Pre-registration:
- For each K: compute N_eff_obs
- Pre-registered criterion: Pearson r(log(N_eff_obs), log(K-1)) > 0.90
- Slope of log-log regression should be in [0.35, 0.65] (sparse competition range)

Output: results/cti_neff_pairwise.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, linregress
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_neff_pairwise.json")

ALPHA_UNIVERSAL = 3.598
K_SUBSETS = [4, 6, 8, 10, 12, 14]
N_SEEDS = 20      # number of random class subsets per K
N_SAMPLES_PER_CLASS = 500
PEARSON_THRESHOLD = 0.90
SLOPE_LOW  = 0.35
SLOPE_HIGH = 0.65


def compute_neff_from_centroids(mu_dict, sigma_W, d, alpha=ALPHA_UNIVERSAL):
    """
    Given class centroids and within-class std, compute N_eff for each class.
    N_eff_k = sum_{j!=k} exp(-alpha * (kappa_kj - kappa_k_nearest))
    where kappa_kj = ||mu_k - mu_j|| / (sigma_W * sqrt(d))
    """
    classes = list(mu_dict.keys())
    K = len(classes)
    if K < 2:
        return None

    # Compute all pairwise kappa_kj
    kappa_mat = np.zeros((K, K))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i == j:
                continue
            dist = float(np.linalg.norm(mu_dict[ci] - mu_dict[cj]))
            kappa_mat[i, j] = dist / (sigma_W * np.sqrt(d))

    # For each class, compute N_eff
    neff_list = []
    for i in range(K):
        kappa_others = kappa_mat[i, [j for j in range(K) if j != i]]
        kappa_nearest = np.min(kappa_others)
        # N_eff = sum_j exp(-alpha * (kappa_kj - kappa_nearest))
        delta_kappas = kappa_others - kappa_nearest
        neff_k = float(np.sum(np.exp(-alpha * delta_kappas)))
        neff_list.append(neff_k)

    return float(np.mean(neff_list)), float(np.std(neff_list))


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    MODEL_NAME = "EleutherAI/pythia-160m"
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dbpedia dataset...")
    ds = load_dataset("fancyzhx/dbpedia_14", split="test")
    texts = ds["content"]
    labels = ds["label"]
    all_classes = sorted(set(labels))
    K_full = len(all_classes)
    print(f"DBpedia: K={K_full}, {len(texts)} samples")

    # Extract embeddings for all classes
    print(f"\nExtracting embeddings (N_SAMPLES_PER_CLASS={N_SAMPLES_PER_CLASS})...")
    class_embeddings = {}
    for ci in all_classes:
        idx_ci = [i for i, l in enumerate(labels) if l == ci][:N_SAMPLES_PER_CLASS]
        texts_ci = [texts[i] for i in idx_ci]
        embs_ci = []
        batch_size = 32
        for b_start in range(0, len(texts_ci), batch_size):
            batch = texts_ci[b_start:b_start+batch_size]
            tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=128).to(device)
            with torch.no_grad():
                out = model(**tok)
            emb = out.last_hidden_state[:, -1, :].cpu().float().numpy()
            embs_ci.extend(emb)
        class_embeddings[ci] = np.array(embs_ci)
        sys.stdout.write(f"  Class {ci}: {len(embs_ci)} samples\r")
        sys.stdout.flush()
    print()
    d = class_embeddings[all_classes[0]].shape[1]
    print(f"Embedding dim: {d}")

    # Compute N_eff for each K subset
    results_per_K = {}
    print(f"\nComputing N_eff for K subsets: {K_SUBSETS}")
    for K in K_SUBSETS:
        neff_list = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            classes_k = list(rng.choice(all_classes, size=K, replace=False))

            # Compute class centroids and sigma_W
            mu = {}
            within_var_sum = 0.0
            n_total = 0
            for ci in classes_k:
                embs = class_embeddings[ci]
                mu[ci] = embs.mean(0)
                within_var_sum += np.sum((embs - mu[ci])**2)
                n_total += len(embs)
            sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))

            neff_mean, neff_std = compute_neff_from_centroids(mu, sigma_W, d)
            if neff_mean is not None:
                neff_list.append(neff_mean)

        if neff_list:
            neff_obs = float(np.mean(neff_list))
            neff_theory = float(np.sqrt(K - 1))
            ratio = neff_obs / neff_theory
            results_per_K[K] = {
                "neff_obs": neff_obs,
                "neff_std": float(np.std(neff_list)),
                "neff_theory_sqrt": neff_theory,
                "ratio_obs_theory": ratio,
                "n_seeds": len(neff_list),
            }
            print(f"  K={K:>3}: N_eff_obs={neff_obs:.3f}, sqrt(K-1)={neff_theory:.3f}, "
                  f"ratio={ratio:.3f}, n_seeds={len(neff_list)}")

    # Fit log-log
    K_vals = sorted(results_per_K.keys())
    log_km1 = np.array([np.log(K - 1) for K in K_vals])
    log_neff = np.array([np.log(results_per_K[K]["neff_obs"]) for K in K_vals])

    slope, intercept, r_val, p_val, se = linregress(log_km1, log_neff)
    r_pearson, p_pearson = pearsonr(log_km1, log_neff)

    print(f"\nLog-log regression:")
    print(f"  slope (beta_neff): {slope:.4f} +/- {se:.4f}")
    print(f"  Pearson r: {r_pearson:.4f}, p={p_pearson:.4f}")

    pr_slope = SLOPE_LOW <= slope <= SLOPE_HIGH
    pr_r = r_pearson >= PEARSON_THRESHOLD
    print(f"\nPre-registered:")
    print(f"  PR_SLOPE (in [{SLOPE_LOW},{SLOPE_HIGH}]): {'PASS' if pr_slope else 'FAIL'} (slope={slope:.4f})")
    print(f"  PR_R (r >= {PEARSON_THRESHOLD}): {'PASS' if pr_r else 'FAIL'} (r={r_pearson:.4f})")

    output = {
        "experiment": "neff_pairwise_direct",
        "model": MODEL_NAME,
        "dataset": "dbpedia_14",
        "K_full": K_full,
        "n_samples_per_class": N_SAMPLES_PER_CLASS,
        "n_seeds": N_SEEDS,
        "alpha_universal": ALPHA_UNIVERSAL,
        "per_K": results_per_K,
        "log_log_slope": float(slope),
        "log_log_slope_se": float(se),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "pre_reg": {
            "PR_SLOPE": f"slope in [{SLOPE_LOW},{SLOPE_HIGH}]",
            "PR_R": f"Pearson r >= {PEARSON_THRESHOLD}",
        },
        "verdict": {
            "PR_SLOPE": bool(pr_slope),
            "PR_R": bool(pr_r),
            "n_pass": int(pr_slope) + int(pr_r),
        },
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
