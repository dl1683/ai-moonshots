"""
Multi-architecture N_eff pairwise universality test.

Tests if the geometric N_eff exponent (slope of log N_eff vs log(K-1))
is universal across architectures.

Baseline (pythia-160m, from previous experiment):
  DBpedia slope = 0.934, r = 0.9999

Pre-registered (results/cti_neff_pairwise_multiarch_prereg.json):
  PR1: Each architecture: slope in [0.85, 1.00]
  PR2: Each architecture: Pearson r > 0.99
  PR3: CV of slopes across all 4 architectures (including pythia-160m) < 10%
  PASS: PR1+PR2 for >=3/3 new architectures, PR3 passes

Output: results/cti_neff_pairwise_multiarch.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, linregress

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_neff_pairwise_multiarch.json")

ALPHA_UNIVERSAL = 3.598
K_SUBSETS = [4, 6, 8, 10, 12, 14]
N_SEEDS = 20
N_SAMPLES_PER_CLASS = 500
SLOPE_LOW = 0.85
SLOPE_HIGH = 1.00
R_THRESHOLD = 0.99
CV_THRESHOLD = 0.10

MODELS = [
    "EleutherAI/gpt-neo-125M",
    "allenai/OLMo-1B-hf",
    "Qwen/Qwen3-0.6B",
]

BASELINE_PYTHIA = {
    "model": "EleutherAI/pythia-160m",
    "slope": 0.9339427789622784,
    "r": 0.9999429666130499,
}


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


def run_model(model_name, device, tokenizer, model, all_classes, class_embeddings, K_full, d):
    """Run N_eff pairwise measurement for one model."""
    results_per_K = {}

    print(f"\nComputing N_eff for K subsets: {K_SUBSETS}")
    for K in K_SUBSETS:
        if K > K_full:
            print(f"  K={K}: skipped (> K_full={K_full})")
            continue
        neff_list = []
        kappa_list = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 200)
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

    print(f"\nLog-log regression:")
    print(f"  slope (N_eff exponent): {slope:.4f} +/- {se:.4f}")
    print(f"  Pearson r: {r_pearson:.4f}, p={p_pearson:.4f}")

    pr1 = SLOPE_LOW <= slope <= SLOPE_HIGH
    pr2 = r_pearson >= R_THRESHOLD
    print(f"\nPre-registered:")
    print(f"  PR1 slope in [{SLOPE_LOW},{SLOPE_HIGH}]: {'PASS' if pr1 else 'FAIL'} (slope={slope:.4f})")
    print(f"  PR2 r >= {R_THRESHOLD}: {'PASS' if pr2 else 'FAIL'} (r={r_pearson:.4f})")

    return {
        "model": model_name,
        "per_K": results_per_K,
        "log_log_slope": float(slope),
        "log_log_slope_se": float(se),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "pr1_slope_pass": bool(pr1),
        "pr2_r_pass": bool(pr2),
    }


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading DBpedia dataset...")
    ds = load_dataset("dbpedia_14", split="train", trust_remote_code=True)
    texts = [x["content"] for x in ds]
    labels = [x["label"] for x in ds]
    all_classes = sorted(set(labels))
    K_full = len(all_classes)
    print(f"DBpedia: K={K_full}, {len(texts)} samples")

    model_results = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        m = AutoModel.from_pretrained(model_name).to(device)
        m.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
                    out = m(**tok)
                emb = out.last_hidden_state[:, -1, :].cpu().float().numpy()
                embs_ci.extend(emb)
            class_embeddings[ci] = np.array(embs_ci)
            sys.stdout.write(f"  Class {ci}/{K_full}: {len(embs_ci)} samples\r")
            sys.stdout.flush()
        print()
        d = class_embeddings[all_classes[0]].shape[1]
        print(f"Embedding dim: {d}")

        res = run_model(model_name, device, tokenizer, m, all_classes,
                        class_embeddings, K_full, d)
        model_results.append(res)

        # Free GPU memory
        del m
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-architecture CV
    all_slopes = [BASELINE_PYTHIA["slope"]] + [r["log_log_slope"] for r in model_results]
    slope_cv = float(np.std(all_slopes) / np.abs(np.mean(all_slopes)))
    pr3 = slope_cv < CV_THRESHOLD

    print(f"\n{'='*60}")
    print(f"CROSS-ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline (pythia-160m): slope={BASELINE_PYTHIA['slope']:.4f}")
    for r in model_results:
        pr_str = "PASS" if (r["pr1_slope_pass"] and r["pr2_r_pass"]) else "FAIL"
        print(f"  {r['model']}: slope={r['log_log_slope']:.4f}, "
              f"r={r['pearson_r']:.4f}, [{pr_str}]")
    print(f"All slopes: {[f'{s:.4f}' for s in all_slopes]}")
    print(f"CV across all 4 architectures: {slope_cv:.4f} ({'PASS' if pr3 else 'FAIL'} threshold={CV_THRESHOLD})")

    n_pr12_pass = sum(1 for r in model_results if r["pr1_slope_pass"] and r["pr2_r_pass"])
    overall_pass = (n_pr12_pass >= 3) and pr3
    print(f"\nOVERALL VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  PR1+PR2 pass: {n_pr12_pass}/3 architectures")
    print(f"  PR3 (CV<10%): {'PASS' if pr3 else 'FAIL'}")

    output = {
        "experiment": "neff_pairwise_multi_arch",
        "dataset": "dbpedia_14",
        "K_full": K_full,
        "alpha_universal": ALPHA_UNIVERSAL,
        "n_seeds": N_SEEDS,
        "n_samples_per_class": N_SAMPLES_PER_CLASS,
        "baseline_pythia160m": BASELINE_PYTHIA,
        "model_results": model_results,
        "all_slopes": all_slopes,
        "slope_cv": slope_cv,
        "pr1_range": [SLOPE_LOW, SLOPE_HIGH],
        "pr2_r_threshold": R_THRESHOLD,
        "pr3_cv_threshold": CV_THRESHOLD,
        "n_pr12_pass": n_pr12_pass,
        "pr3_pass": bool(pr3),
        "overall_pass": bool(overall_pass),
        "verdict": "PASS" if overall_pass else "FAIL",
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
