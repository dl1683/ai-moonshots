"""
Extended N_eff pairwise: 3 more architectures to strengthen competition density correlation.

Adds TinyLlama-1.1B, rwkv-4-169m, Qwen2.5-0.5B to the 4 existing data points.
Goal: 7 architectures for competition density correlation (r=0.996 from 4 archs).

Pre-registration: same as cti_neff_pairwise_multiarch_prereg.json
Output: results/cti_neff_pairwise_extended.json (appended to existing multi-arch results)
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, linregress, spearmanr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_neff_pairwise_extended.json")
MULTIARCH_PATH = os.path.join(RESULTS_DIR, "cti_neff_pairwise_multiarch.json")

ALPHA_UNIVERSAL = 3.598
K_SUBSETS = [4, 6, 8, 10, 12, 14]
N_SEEDS = 20
N_SAMPLES_PER_CLASS = 500

MODELS = [
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "RWKV/rwkv-4-169m-pile",
    "Qwen/Qwen2.5-0.5B",
]


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


def run_model(model_name, class_embeddings, all_classes, K_full, d):
    results_per_K = {}
    print(f"\nComputing N_eff for K subsets: {K_SUBSETS}")
    for K in K_SUBSETS:
        if K > K_full:
            continue
        neff_list = []
        kappa_list = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 400)
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
    slope, _, r_val, _, se = linregress(log_km1, log_neff)
    r_pearson, p_pearson = pearsonr(log_km1, log_neff)
    neff_at_14 = results_per_K.get(14, {}).get("neff_obs", None)
    competition_density = neff_at_14 / 13.0 if neff_at_14 else None

    print(f"\n  slope={slope:.4f}, r={r_pearson:.4f}, N_eff@14={neff_at_14:.3f}, "
          f"density={competition_density:.3f}")

    return {
        "model": model_name,
        "per_K": results_per_K,
        "log_log_slope": float(slope),
        "log_log_slope_se": float(se),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "neff_at_K14": float(neff_at_14) if neff_at_14 else None,
        "competition_density_K14": float(competition_density) if competition_density else None,
    }


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading DBpedia dataset...")
    ds = load_dataset("dbpedia_14", split="train")
    texts = [x["content"] for x in ds]
    labels = [x["label"] for x in ds]
    all_classes = sorted(set(labels))
    K_full = len(all_classes)
    print(f"DBpedia: K={K_full}, {len(texts)} samples")

    new_results = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            m = AutoModel.from_pretrained(model_name).to(device)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue

        m.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"\nExtracting embeddings (N_SAMPLES_PER_CLASS={N_SAMPLES_PER_CLASS})...")
        class_embeddings = {}
        for ci in all_classes:
            idx_ci = [i for i, l in enumerate(labels) if l == ci][:N_SAMPLES_PER_CLASS]
            texts_ci = [texts[i] for i in idx_ci]
            embs_ci = []
            for b_start in range(0, len(texts_ci), 32):
                batch = texts_ci[b_start:b_start+32]
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

        res = run_model(model_name, class_embeddings, all_classes, K_full, d)
        new_results.append(res)

        del m
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load existing multi-arch results
    with open(MULTIARCH_PATH) as f:
        multiarch = json.load(f)

    # Build full competition density dataset
    all_archs = []
    all_slopes = []
    all_densities = []

    # Baseline pythia-160m
    with open(os.path.join(RESULTS_DIR, "cti_neff_pairwise.json")) as f:
        pyth = json.load(f)
    neff14_pythia = pyth["per_K"]["14"]["neff_obs"]
    all_archs.append("pythia-160m")
    all_slopes.append(multiarch["baseline_pythia160m"]["slope"])
    all_densities.append(neff14_pythia / 13.0)

    for mr in multiarch["model_results"]:
        neff14 = mr["per_K"]["14"]["neff_obs"]
        all_archs.append(mr["model"].split("/")[-1])
        all_slopes.append(mr["log_log_slope"])
        all_densities.append(neff14 / 13.0)

    for nr in new_results:
        neff14 = nr["neff_at_K14"]
        density = nr["competition_density_K14"]
        all_archs.append(nr["model"].split("/")[-1])
        all_slopes.append(nr["log_log_slope"])
        all_densities.append(density)

    # Correlation analysis
    r_pearson, p_pearson = pearsonr(all_densities, all_slopes)
    r_spearman, p_spearman = spearmanr(all_densities, all_slopes)
    n = len(all_archs)

    print(f"\n{'='*60}")
    print(f"COMPETITION DENSITY vs N_EFF EXPONENT ({n} architectures)")
    print(f"{'='*60}")
    for arch, slope, density in zip(all_archs, all_slopes, all_densities):
        print(f"  {arch:45s}: slope={slope:.4f}, density={density:.3f}")
    print(f"\nCorrelation: Pearson r={r_pearson:.4f} (p={p_pearson:.4f})")
    print(f"             Spearman rho={r_spearman:.4f} (p={p_spearman:.4f})")

    output = {
        "experiment": "neff_pairwise_extended",
        "note": "Adds TinyLlama, RWKV, Qwen2.5 to competition density analysis",
        "n_architectures_total": n,
        "competition_density_correlation": {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_rho": float(r_spearman),
            "spearman_p": float(p_spearman),
        },
        "all_architectures": [
            {"arch": arch, "neff_slope": float(s), "competition_density_K14": float(d)}
            for arch, s, d in zip(all_archs, all_slopes, all_densities)
        ],
        "new_model_results": new_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
