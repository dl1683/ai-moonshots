"""
N_eff pairwise measurement on Banking77 (sparse regime).
Companion to cti_neff_pairwise.py which tested DBpedia (dense regime).

Banking77 K=77 fine-grained intents — expected SPARSE competition (beta~0.5).
DBpedia K=14 broad topics     — DENSE competition (beta~0.94, matches M1 at K=14).

This tests: can we observe the REGIME TRANSITION from pairwise geometry?

Pre-registration:
- Banking77 slope in [0.35, 0.65] (sparse) -> PASS
- DBpedia slope in [0.75, 1.05] (dense) -> PASS (already confirmed: slope=0.934)
- The contrast (banking slope < dbpedia slope) confirms regime-dependent competition

Output: results/cti_neff_pairwise_banking.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, linregress

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_neff_pairwise_banking.json")

ALPHA_UNIVERSAL = 3.598
K_SUBSETS = [4, 8, 16, 32, 48, 64]   # wider range to better fit log-log slope
N_SEEDS = 15
N_SAMPLES_PER_CLASS = 300
SLOPE_LOW_SPARSE  = 0.35
SLOPE_HIGH_SPARSE = 0.65


def compute_neff_from_centroids(mu_dict, sigma_W, d, alpha=ALPHA_UNIVERSAL):
    classes = list(mu_dict.keys())
    K = len(classes)
    if K < 2:
        return None, None

    kappa_mat = np.zeros((K, K))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i == j:
                continue
            dist = float(np.linalg.norm(mu_dict[ci] - mu_dict[cj]))
            kappa_mat[i, j] = dist / (sigma_W * np.sqrt(d))

    neff_list = []
    kappa_nearest_list = []
    for i in range(K):
        kappa_others = kappa_mat[i, [j for j in range(K) if j != i]]
        kappa_nearest = np.min(kappa_others)
        delta_kappas = kappa_others - kappa_nearest
        neff_k = float(np.sum(np.exp(-alpha * delta_kappas)))
        neff_list.append(neff_k)
        kappa_nearest_list.append(kappa_nearest)

    return float(np.mean(neff_list)), float(np.mean(kappa_nearest_list))


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    MODEL_NAME = "EleutherAI/pythia-160m"
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Banking77 dataset...")
    ds = load_dataset("PolyAI/banking77", split="test")
    texts = [x["text"] for x in ds]
    labels = [x["label"] for x in ds]
    all_classes = sorted(set(labels))
    K_full = len(all_classes)
    print(f"Banking77: K={K_full}, {len(texts)} samples")

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
        sys.stdout.write(f"  Class {ci}/{K_full}: {len(embs_ci)} samples\r")
        sys.stdout.flush()
    print()
    d = class_embeddings[all_classes[0]].shape[1]
    print(f"Embedding dim: {d}")

    results_per_K = {}
    print(f"\nComputing N_eff for K subsets: {K_SUBSETS}")
    for K in K_SUBSETS:
        if K > K_full:
            print(f"  K={K}: skipped (> K_full={K_full})")
            continue
        neff_list = []
        kappa_list = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 100)  # different seed range from dbpedia run
            classes_k = list(rng.choice(all_classes, size=K, replace=False))
            mu = {}
            within_var_sum = 0.0
            n_total = 0
            for ci in classes_k:
                embs = class_embeddings[ci]
                mu[ci] = embs.mean(0)
                within_var_sum += np.sum((embs - mu[ci])**2)
                n_total += len(embs)
            sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))
            neff_mean, kappa_mean = compute_neff_from_centroids(mu, sigma_W, d)
            if neff_mean is not None:
                neff_list.append(neff_mean)
                kappa_list.append(kappa_mean)

        if neff_list:
            neff_obs = float(np.mean(neff_list))
            neff_theory = float(np.sqrt(K - 1))
            ratio = neff_obs / neff_theory
            kappa_avg = float(np.mean(kappa_list))
            results_per_K[K] = {
                "neff_obs": neff_obs,
                "neff_std": float(np.std(neff_list)),
                "neff_theory_sqrt": neff_theory,
                "ratio_obs_theory": ratio,
                "kappa_nearest_mean": kappa_avg,
                "n_seeds": len(neff_list),
            }
            print(f"  K={K:>3}: N_eff_obs={neff_obs:.3f}, sqrt(K-1)={neff_theory:.3f}, "
                  f"ratio={ratio:.3f}, kappa={kappa_avg:.4f}")

    K_vals = sorted(results_per_K.keys())
    log_km1 = np.array([np.log(K - 1) for K in K_vals])
    log_neff = np.array([np.log(results_per_K[K]["neff_obs"]) for K in K_vals])

    slope, intercept, r_val, p_val, se = linregress(log_km1, log_neff)
    r_pearson, p_pearson = pearsonr(log_km1, log_neff)

    print(f"\nLog-log regression (Banking77):")
    print(f"  slope (beta_neff): {slope:.4f} +/- {se:.4f}")
    print(f"  Pearson r: {r_pearson:.4f}, p={p_pearson:.4f}")

    pr_slope = SLOPE_LOW_SPARSE <= slope <= SLOPE_HIGH_SPARSE
    print(f"\nPre-registered:")
    print(f"  PR_SLOPE_SPARSE (in [{SLOPE_LOW_SPARSE},{SLOPE_HIGH_SPARSE}]): "
          f"{'PASS' if pr_slope else 'FAIL'} (slope={slope:.4f})")

    # Compare to DBpedia
    dbpedia_path = os.path.join(RESULTS_DIR, "cti_neff_pairwise.json")
    dbpedia_slope = None
    if os.path.exists(dbpedia_path):
        with open(dbpedia_path) as f:
            dbp = json.load(f)
        dbpedia_slope = dbp.get("log_log_slope")
        pr_contrast = (slope < dbpedia_slope) if dbpedia_slope else None
        print(f"  DBpedia slope: {dbpedia_slope:.4f}")
        print(f"  PR_CONTRAST (banking slope < dbpedia slope): "
              f"{'PASS' if pr_contrast else 'FAIL'}")

    output = {
        "experiment": "neff_pairwise_banking77",
        "model": MODEL_NAME,
        "dataset": "banking77",
        "K_full": K_full,
        "n_samples_per_class": N_SAMPLES_PER_CLASS,
        "n_seeds": N_SEEDS,
        "alpha_universal": ALPHA_UNIVERSAL,
        "per_K": results_per_K,
        "log_log_slope": float(slope),
        "log_log_slope_se": float(se),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "dbpedia_slope": dbpedia_slope,
        "pr_sparse_pass": bool(pr_slope),
        "regime_contrast": "banking_sparse_vs_dbpedia_dense" if dbpedia_slope else None,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
