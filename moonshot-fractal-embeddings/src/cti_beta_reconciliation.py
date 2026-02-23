"""
Beta reconciliation: within-architecture K-dependence of logit(q) vs N_eff geometric slope.

CORE QUESTION: Why does cross-dataset regression give beta=0.478 while
geometric N_eff from centroid geometry gives slope=0.93?

HYPOTHESIS: The within-architecture, within-dataset beta (computed by subsampling K classes
from a fixed architecture+dataset) = geometric N_eff slope ~ 0.93.
The cross-dataset regression beta=0.478 is attenuated because different datasets
have different C_0 intercepts, and C_0 absorbs some of the K-dependence.

DESIGN:
For pythia-160m on DBpedia (K_full=14):
  1. For K in {4, 6, 8, 10, 12, 14}: sample K classes, compute actual q (1-NN) AND N_eff geometry
  2. Fit beta_logit = slope of logit(q_norm) vs log(K-1)  [within arch+dataset]
  3. Compare to geometric N_eff slope = 0.934 (from cti_neff_pairwise.json)
  4. If beta_logit ~ 0.93: cross-dataset beta=0.478 is a C_0 variation artifact
  5. If beta_logit ~ 0.5: sparse competition is genuine, geometric N_eff is different quantity

Also run for Banking77 (K_full=77, wider K range: 4,8,16,32,48,64,77).

PRE-REGISTRATION (committed before running):
- H1: beta_logit (DBpedia, within-arch) in [0.80, 1.00]  -> N_eff interpretation consistent
- H2: beta_logit (Banking77, within-arch) in [0.80, 1.00] -> same
- H3: both beta_logit values within 0.20 of geometric N_eff slopes
- If H1+H2+H3 PASS: beta=0.478 is cross-dataset attenuation artifact, not sparse competition
- If H1/H2 FAIL (beta_logit ~ 0.5): sparse competition is a real within-arch phenomenon

Output: results/cti_beta_reconciliation.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.special import logit as scipy_logit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_beta_reconciliation.json")

ALPHA_UNIVERSAL = 3.598
N_SEEDS = 20
N_SAMPLES_PER_CLASS = 500

EXPERIMENTS = [
    {
        "dataset": "dbpedia_14",
        "K_subsets": [4, 6, 8, 10, 12, 14],
        "K_full": 14,
        "geometric_neff_slope": 0.9339,  # from cti_neff_pairwise.json
    },
    {
        "dataset": "banking77",
        "K_subsets": [4, 8, 16, 32, 48, 64, 77],
        "K_full": 77,
        "geometric_neff_slope": 0.9176,  # from cti_neff_pairwise_banking.json
    },
]

# Pre-reg thresholds
H1_SLOPE_LOW = 0.80
H1_SLOPE_HIGH = 1.00
H3_MAX_DIFF = 0.20


def compute_1nn_accuracy(embs_train, labels_train, embs_test, labels_test):
    """Simple 1-NN classification."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(embs_train, labels_train)
    preds = knn.predict(embs_test)
    return float(np.mean(preds == labels_test))


def compute_neff_from_centroids(mu_dict, sigma_W, d, alpha=ALPHA_UNIVERSAL):
    classes = list(mu_dict.keys())
    K = len(classes)
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


def run_dataset_experiment(exp_config, class_embeddings, all_classes, d):
    """For each K subset, compute both logit(q) AND geometric N_eff."""
    K_subsets = exp_config["K_subsets"]
    K_full = exp_config["K_full"]
    dataset = exp_config["dataset"]
    geometric_slope = exp_config["geometric_neff_slope"]

    results_per_K = {}
    print(f"\nDataset: {dataset} (K_full={K_full})")
    print(f"K subsets: {K_subsets}")

    for K in K_subsets:
        if K > K_full:
            print(f"  K={K}: skipped")
            continue

        logit_q_list = []
        neff_list = []
        kappa_list = []

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 300)
            classes_k = list(rng.choice(all_classes, size=K, replace=False))

            # Collect embeddings
            embs_by_class = {}
            for ci in classes_k:
                embs_by_class[ci] = class_embeddings[ci]

            # Compute centroids + within-class statistics
            mu = {}
            within_var_sum = 0.0
            n_total = 0
            for ci in classes_k:
                embs = embs_by_class[ci]
                mu[ci] = embs.mean(0)
                within_var_sum += np.sum((embs - mu[ci])**2)
                n_total += len(embs)
            sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))

            # Compute geometric N_eff
            neff_mean, kappa_mean = compute_neff_from_centroids(mu, sigma_W, d)

            # Compute actual 1-NN accuracy (80/20 split within each class)
            train_embs, train_labels, test_embs, test_labels = [], [], [], []
            for ci in classes_k:
                embs = embs_by_class[ci]
                n = len(embs)
                n_train = max(1, int(0.8 * n))
                idx = rng.permutation(n)
                train_embs.append(embs[idx[:n_train]])
                train_labels.extend([ci] * n_train)
                test_embs.append(embs[idx[n_train:]])
                test_labels.extend([ci] * (n - n_train))

            train_embs = np.vstack(train_embs)
            test_embs = np.vstack(test_embs)
            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)

            q_raw = compute_1nn_accuracy(train_embs, train_labels, test_embs, test_labels)
            # Normalize: q_norm = (q_raw - 1/K) / (1 - 1/K)
            q_norm = (q_raw - 1.0/K) / (1.0 - 1.0/K)
            # Clip to avoid logit extremes
            q_norm_clipped = float(np.clip(q_norm, 0.01, 0.99))
            logit_q = float(scipy_logit(q_norm_clipped))

            logit_q_list.append(logit_q)
            neff_list.append(neff_mean)
            kappa_list.append(kappa_mean)

        if logit_q_list:
            results_per_K[K] = {
                "logit_q_mean": float(np.mean(logit_q_list)),
                "logit_q_std": float(np.std(logit_q_list)),
                "neff_mean": float(np.mean(neff_list)),
                "neff_std": float(np.std(neff_list)),
                "kappa_nearest_mean": float(np.mean(kappa_list)),
                "n_seeds": len(logit_q_list),
            }
            print(f"  K={K:>3}: logit(q)={np.mean(logit_q_list):.3f}+/-{np.std(logit_q_list):.3f}, "
                  f"N_eff={np.mean(neff_list):.3f}, kappa={np.mean(kappa_list):.4f}")

    # Fit beta_logit: slope of logit(q_norm) vs log(K-1)
    K_vals = sorted(results_per_K.keys())
    log_km1 = np.array([np.log(K - 1) for K in K_vals])
    logit_q_vals = np.array([results_per_K[K]["logit_q_mean"] for K in K_vals])
    log_neff_vals = np.array([np.log(results_per_K[K]["neff_mean"]) for K in K_vals])

    # Beta_logit: direct logit(q) slope (this is what the law predicts as beta)
    slope_logit, _, r_logit, _, se_logit = linregress(log_km1, logit_q_vals)
    r_logit_p, p_logit = pearsonr(log_km1, logit_q_vals)

    # Beta_neff: geometric N_eff slope (should match pairwise experiment)
    slope_neff, _, r_neff, _, se_neff = linregress(log_km1, log_neff_vals)
    r_neff_p, p_neff = pearsonr(log_km1, log_neff_vals)

    h1_pass = H1_SLOPE_LOW <= slope_logit <= H1_SLOPE_HIGH
    h3_pass = abs(slope_logit - geometric_slope) <= H3_MAX_DIFF

    print(f"\n  beta_logit (logit(q) vs log(K-1)): {slope_logit:.4f} +/- {se_logit:.4f}, "
          f"r={r_logit_p:.4f}")
    print(f"  beta_neff (N_eff slope, this run): {slope_neff:.4f} +/- {se_neff:.4f}, "
          f"r={r_neff_p:.4f}")
    print(f"  geometric N_eff slope (from prereg): {geometric_slope:.4f}")
    print(f"  |beta_logit - geometric_slope| = {abs(slope_logit-geometric_slope):.4f}")
    print(f"  H1 (beta_logit in [{H1_SLOPE_LOW},{H1_SLOPE_HIGH}]): {'PASS' if h1_pass else 'FAIL'}")
    print(f"  H3 (|diff| <= {H3_MAX_DIFF}): {'PASS' if h3_pass else 'FAIL'}")

    return {
        "dataset": dataset,
        "geometric_neff_slope_baseline": geometric_slope,
        "per_K": results_per_K,
        "beta_logit": float(slope_logit),
        "beta_logit_se": float(se_logit),
        "beta_logit_r": float(r_logit_p),
        "beta_logit_p": float(p_logit),
        "beta_neff_this_run": float(slope_neff),
        "beta_neff_r": float(r_neff_p),
        "diff_logit_vs_geometric": float(abs(slope_logit - geometric_slope)),
        "h1_pass": bool(h1_pass),
        "h3_pass": bool(h3_pass),
    }


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

    all_dataset_results = []

    for exp_config in EXPERIMENTS:
        dataset_name = exp_config["dataset"]
        K_full = exp_config["K_full"]

        print(f"\n{'='*60}")
        print(f"Loading {dataset_name}...")

        if dataset_name == "dbpedia_14":
            ds = load_dataset("dbpedia_14", split="train")
            texts = [x["content"] for x in ds]
            labels = [x["label"] for x in ds]
        elif dataset_name == "banking77":
            ds = load_dataset("mteb/banking77", split="test")
            texts = [x["text"] for x in ds]
            labels = [x["label"] for x in ds]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        all_classes = sorted(set(labels))
        print(f"  K={len(all_classes)}, {len(texts)} samples")

        print(f"\nExtracting embeddings (N_SAMPLES_PER_CLASS={N_SAMPLES_PER_CLASS})...")
        class_embeddings = {}
        for ci in all_classes:
            idx_ci = [i for i, l in enumerate(labels) if l == ci][:N_SAMPLES_PER_CLASS]
            texts_ci = [texts[i] for i in idx_ci]
            embs_ci = []
            batch_size = 32
            for b_start in range(0, len(texts_ci), batch_size):
                batch = texts_ci[b_start:b_start+batch_size]
                tok = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    out = model(**tok)
                emb = out.last_hidden_state[:, -1, :].cpu().float().numpy()
                embs_ci.extend(emb)
            class_embeddings[ci] = np.array(embs_ci)
            sys.stdout.write(f"  Class {ci}/{len(all_classes)}: {len(embs_ci)} samples\r")
            sys.stdout.flush()
        print()
        d = class_embeddings[all_classes[0]].shape[1]
        print(f"  Embedding dim: {d}")

        res = run_dataset_experiment(exp_config, class_embeddings, all_classes, d)
        all_dataset_results.append(res)

    # Overall verdict
    n_h1_pass = sum(1 for r in all_dataset_results if r["h1_pass"])
    n_h3_pass = sum(1 for r in all_dataset_results if r["h3_pass"])

    print(f"\n{'='*60}")
    print(f"OVERALL RECONCILIATION VERDICT")
    print(f"{'='*60}")
    for r in all_dataset_results:
        print(f"  {r['dataset']}: beta_logit={r['beta_logit']:.4f}, "
              f"geometric={r['geometric_neff_slope_baseline']:.4f}, "
              f"diff={r['diff_logit_vs_geometric']:.4f}, "
              f"H1={'PASS' if r['h1_pass'] else 'FAIL'}, H3={'PASS' if r['h3_pass'] else 'FAIL'}")

    cross_dataset_beta_regression = 0.478  # from comprehensive universality
    print(f"\n  Cross-dataset regression beta: {cross_dataset_beta_regression}")
    within_betas = [round(r['beta_logit'], 4) for r in all_dataset_results]
    print(f"  Within-arch betas: {within_betas}")

    if n_h1_pass >= 2 and n_h3_pass >= 2:
        interpretation = "CONSISTENT: within-arch beta~0.93; cross-dataset beta=0.478 is C_0-variation artifact"
    elif n_h1_pass == 0:
        interpretation = "SPARSE_GENUINE: within-arch beta~0.5 too; sparse competition is real within-arch phenomenon"
    else:
        interpretation = "MIXED: partial evidence, further investigation needed"

    print(f"\n  INTERPRETATION: {interpretation}")

    output = {
        "experiment": "beta_reconciliation",
        "model": MODEL_NAME,
        "n_seeds": N_SEEDS,
        "n_samples_per_class": N_SAMPLES_PER_CLASS,
        "alpha_universal": ALPHA_UNIVERSAL,
        "cross_dataset_regression_beta": cross_dataset_beta_regression,
        "datasets": all_dataset_results,
        "n_h1_pass": n_h1_pass,
        "n_h3_pass": n_h3_pass,
        "interpretation": interpretation,
        "pre_reg": {
            "H1": f"Within-arch beta_logit in [{H1_SLOPE_LOW},{H1_SLOPE_HIGH}]",
            "H2": "H1 for Banking77 too",
            "H3": f"|beta_logit - geometric_slope| <= {H3_MAX_DIFF} for both datasets",
        },
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
