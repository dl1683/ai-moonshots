#!/usr/bin/env python
"""
CTI Cross-Family Validation: OLMo-2-1B checkpoint sweep.

Tests whether the Gaussian depth law fitted on Pythia transfers to a
completely different model family (OLMo-2). This is the CRITICAL test:
if cross-family transfer works, the law has real predictive power.

Usage:
    python -u src/cti_cross_family.py --sweep     # Run OLMo sweep
    python -u src/cti_cross_family.py --predict    # Test Pythia->OLMo prediction
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

# OLMo-2-1B: trained on ~4T tokens, batch_size varies
# Tokens per step: from branch names, step 10000 = 21B tokens -> ~2.1M tokens/step
OLMO_TOKENS_PER_STEP = 2_097_152  # ~2.1M (same as Pythia by coincidence)

OLMO_MODEL = {
    "hf_path": "allenai/OLMo-2-0425-1B",
    "N": 1_241_513_984,  # Will verify at runtime
    "num_layers": 16,
    "hidden_dim": 2048,
    "family": "olmo2",
}

# Sampled checkpoints (log-spaced, covering early to late training)
# Verified against actual HuggingFace branch names
OLMO_CHECKPOINTS = [
    ("stage1-step300-tokens1B", 300, 1_000_000_000),
    ("stage1-step10000-tokens21B", 10000, 21_000_000_000),
    ("stage1-step20000-tokens42B", 20000, 42_000_000_000),
    ("stage1-step50000-tokens105B", 50000, 105_000_000_000),
    ("stage1-step80000-tokens168B", 80000, 168_000_000_000),
    ("stage1-step150000-tokens315B", 150000, 315_000_000_000),
    ("stage1-step260000-tokens546B", 260000, 546_000_000_000),
    ("stage1-step430000-tokens902B", 430000, 902_000_000_000),
    ("stage1-step720000-tokens1510B", 720000, 1_510_000_000_000),
    ("stage1-step1170000-tokens2454B", 1170000, 2_454_000_000_000),
    ("stage1-step1907359-tokens4001B", 1907359, 4_001_000_000_000),
]

DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


def load_olmo_at_checkpoint(revision, device="cuda"):
    """Load OLMo-2-1B at a specific training checkpoint."""
    from transformers import AutoModel, AutoTokenizer

    print(f"    Loading OLMo-2-1B revision={revision}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        OLMO_MODEL["hf_path"], revision=revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        OLMO_MODEL["hf_path"],
        revision=revision,
        torch_dtype=torch.float16,
    )
    model = model.to(device).eval()

    # Verify param count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Loaded in {time.time()-t0:.1f}s, {n_params:,} params")

    return model, tokenizer, n_params


@torch.no_grad()
def extract_layer_reps(model, tokenizer, texts, device="cuda",
                       batch_size=16, max_seq_len=256):
    """Extract L2-normalized representations from all layers (last-token pooling)."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = texts[i * batch_size:(i + 1) * batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=max_seq_len, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, return_dict=True)

        if out.hidden_states is None:
            raise RuntimeError("No hidden states returned")

        mask = enc["attention_mask"]

        for li, hs in enumerate(out.hidden_states):
            hs_f = hs.float()
            # Last-token pooling for decoder
            seq_lens = mask.sum(dim=1) - 1
            pooled = hs_f[torch.arange(hs_f.size(0)), seq_lens]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            if li not in all_hidden:
                all_hidden[li] = []
            all_hidden[li].append(pooled.cpu().numpy())

    return {li: np.concatenate(arrs, axis=0)
            for li, arrs in sorted(all_hidden.items())}


def knn_accuracy(embs, labels, k=20):
    """kNN accuracy with cosine similarity."""
    n = len(embs)
    if n < k + 1:
        return 0.0

    correct = 0
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = embs[start:end] @ embs.T
        for i in range(end - start):
            sims[i, start + i] = -float("inf")
        topk = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(end - start):
            votes = labels[topk[i]]
            pred = np.bincount(votes).argmax()
            if pred == labels[start + i]:
                correct += 1

    return correct / n


def load_eval_datasets(dataset_names, max_samples=2000):
    """Load and cache evaluation datasets."""
    datasets = {}
    for name in dataset_names:
        print(f"Loading {name}...")
        ds = load_hierarchical_dataset(name, split="test", max_samples=max_samples)
        texts = [s.text for s in ds.samples]
        l0 = np.array([s.level0_label for s in ds.samples])
        l1 = np.array([s.level1_label for s in ds.samples])
        datasets[name] = {"texts": texts, "l0_labels": l0, "l1_labels": l1}
        print(f"  {name}: {len(texts)} samples, {len(set(l0))} L0, {len(set(l1))} L1")
    return datasets


def sweep(device="cuda"):
    """Run OLMo-2-1B checkpoint sweep."""
    ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]
    datasets = load_eval_datasets(ds_names)

    results = []
    out_path = RESULTS_DIR / "cti_olmo2_sweep.json"

    print("=" * 70)
    print("  CTI Cross-Family: OLMo-2-1B Checkpoint Sweep")
    print("=" * 70)

    for revision, step, tokens in OLMO_CHECKPOINTS:
        print(f"\n--- Checkpoint: {revision} (step={step}) ---")
        t0 = time.time()

        try:
            model, tokenizer, n_params = load_olmo_at_checkpoint(revision, device)
            L = OLMO_MODEL["num_layers"]
            C = 6 * n_params * tokens

            result = {
                "model": "olmo2-1b",
                "family": "olmo2",
                "step": step,
                "tokens_seen": tokens,
                "C_flops": C,
                "N_params": n_params,
                "num_layers": L,
                "hidden_dim": OLMO_MODEL["hidden_dim"],
                "revision": revision,
                "datasets": {},
            }

            for ds_name, ds_data in datasets.items():
                texts = ds_data["texts"]
                l0_labels = ds_data["l0_labels"]
                l1_labels = ds_data["l1_labels"]

                layer_reps = extract_layer_reps(model, tokenizer, texts, device,
                                                batch_size=16)

                layer_results = {}
                for li in sorted(layer_reps.keys()):
                    x = li / L
                    reps = layer_reps[li]
                    knn_l0 = knn_accuracy(reps, l0_labels, k=20)
                    knn_l1 = knn_accuracy(reps, l1_labels, k=20)
                    layer_results[li] = {
                        "layer": li,
                        "x": round(x, 4),
                        "knn_l0": round(knn_l0, 4),
                        "knn_l1": round(knn_l1, 4),
                    }

                best_l1 = max(layer_results.values(), key=lambda r: r["knn_l1"])
                best_layer = best_l1["layer"]

                result["datasets"][ds_name] = {
                    "layers": layer_results,
                    "best_layer": best_layer,
                    "best_x": best_l1["x"],
                    "best_knn_l1": best_l1["knn_l1"],
                    "final_knn_l1": layer_results[max(layer_results.keys())]["knn_l1"],
                    "n_samples": len(texts),
                }

                print(f"    {ds_name}: best_layer={best_layer}/{L} (x={best_l1['x']:.2f}) "
                      f"best_knn_l1={best_l1['knn_l1']:.3f} "
                      f"final_knn_l1={layer_results[max(layer_results.keys())]['knn_l1']:.3f}")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results.append(result)
            elapsed = time.time() - t0
            print(f"    Checkpoint done in {elapsed:.0f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"model": "olmo2-1b", "step": step, "error": str(e)})

        # Save incrementally
        out = {
            "experiment": "CTI Cross-Family: OLMo-2-1B Sweep",
            "models": ["olmo2-1b"],
            "datasets": ds_names,
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, (np.floating,)) else x)

    print(f"\nSaved {len(results)} results to {out_path}")


def predict():
    """Test Pythia-fitted parameters on OLMo-2-1B data."""
    from scipy.special import expit

    # Load Pythia fit parameters
    holdout_path = RESULTS_DIR / "cti_holdout_prediction.json"
    with open(holdout_path) as f:
        pythia_fit = json.load(f)

    params = pythia_fit["fit_params"]
    alpha = params["alpha"]
    beta = params["beta"]
    mu_0 = params["mu_0"]
    mu_1 = params["mu_1"]
    b_d = params["b_d"]

    print("Pythia-fitted parameters:")
    print(f"  alpha={alpha:.4f}, beta={beta:.4f}, mu_0={mu_0:.4f}, mu_1={mu_1:.6f}")
    for ds, b in b_d.items():
        print(f"  b_{ds}={b:.4f}")

    # Load OLMo sweep
    olmo_path = RESULTS_DIR / "cti_olmo2_sweep.json"
    with open(olmo_path) as f:
        olmo_data = json.load(f)

    datasets = sorted(DS_CLASSES.keys())

    # Build observations
    obs = []
    for result in olmo_data["results"]:
        if "error" in result:
            continue
        if result["step"] == 0:
            continue

        N = float(result["N_params"])
        C = float(result["C_flops"])
        L = int(result["num_layers"])

        for ds_name, ds_data in result["datasets"].items():
            n_classes = DS_CLASSES.get(ds_name, 100)
            Q_chance = 1.0 / n_classes

            for li_str, layer_data in ds_data["layers"].items():
                li = int(li_str)
                x = li / L
                Q_raw = layer_data["knn_l1"]
                Q_norm = (Q_raw - Q_chance) / (1.0 - Q_chance)
                Q_norm = np.clip(Q_norm, 0.001, 0.999)

                obs.append({
                    "x": x, "Q": Q_norm, "Q_raw": Q_raw, "dataset": ds_name,
                    "step": result["step"], "layer": li, "L": L,
                    "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                })

    Q_obs = np.array([o["Q"] for o in obs])

    # Predict using Pythia parameters
    Q_pred = []
    for o in obs:
        x_star = mu_0 + mu_1 * o["log_r"]
        logit_Q = b_d.get(o["dataset"], -3.0) + alpha * o["log_r"] - beta * (o["x"] - x_star) ** 2
        Q_pred.append(expit(np.clip(logit_Q, -20, 20)))
    Q_pred = np.array(Q_pred)

    # Overall metrics
    residuals = Q_obs - Q_pred
    mae = np.mean(np.abs(residuals))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((Q_obs - Q_obs.mean()) ** 2)

    print("\n" + "=" * 70)
    print("  CROSS-FAMILY PREDICTION: Pythia -> OLMo-2-1B")
    print("=" * 70)
    print(f"Total observations: {len(obs)}")
    print(f"Overall: MAE={mae:.4f}, R2={r2:.4f}")

    # Per-dataset
    print("\nPer-dataset:")
    ds_results = {}
    for ds in datasets:
        mask = np.array([o["dataset"] == ds for o in obs])
        if mask.sum() == 0:
            continue
        Q_te = Q_obs[mask]
        Q_pr = Q_pred[mask]
        res = Q_te - Q_pr
        ds_mae = np.mean(np.abs(res))
        ds_r2 = 1 - np.sum(res ** 2) / np.sum((Q_te - Q_te.mean()) ** 2) if np.sum((Q_te - Q_te.mean()) ** 2) > 0 else 0
        ds_results[ds] = {"mae": float(ds_mae), "r2": float(ds_r2), "n": int(mask.sum())}
        print(f"  {ds:>16s}: MAE={ds_mae:.4f}, R2={ds_r2:.4f} (n={mask.sum()})")

    # Shape universality: compare OLMo vs Pythia profiles
    from scipy.stats import spearmanr

    # Load Pythia data for shape comparison
    pythia_path = RESULTS_DIR / "cti_checkpoint_sweep_all.json"
    with open(pythia_path) as f:
        pythia_data = json.load(f)

    x_common = np.linspace(0, 1, 21)

    def get_profiles(data, model_filter=None):
        profiles = {}
        for result in data["results"]:
            if "error" in result or result["step"] == 0:
                continue
            if model_filter and result["model"] != model_filter:
                continue
            L = int(result["num_layers"])
            for ds_name, ds_data in result["datasets"].items():
                layers = ds_data["layers"]
                qs = []
                xs = []
                for li in sorted(layers.keys(), key=int):
                    qs.append(layers[li]["knn_l1"])
                    xs.append(layers[li]["x"])
                qs = np.array(qs)
                xs = np.array(xs)
                if qs.max() > qs.min():
                    qs_norm = (qs - qs.min()) / (qs.max() - qs.min())
                    profiles[(result.get("model", "olmo2-1b"), result["step"], ds_name)] = \
                        np.interp(x_common, xs, qs_norm)
        return profiles

    pythia_profiles = get_profiles(pythia_data)
    olmo_profiles = get_profiles(olmo_data)

    # Cross-family shape correlation
    shape_rhos = []
    for okey, oprof in olmo_profiles.items():
        for pkey, pprof in pythia_profiles.items():
            if okey[2] == pkey[2]:  # Same dataset
                rho, _ = spearmanr(oprof, pprof)
                shape_rhos.append({"olmo_step": okey[1], "pythia_model": pkey[0],
                                   "pythia_step": pkey[1], "dataset": okey[2],
                                   "rho": float(rho)})

    if shape_rhos:
        rhos = np.array([r["rho"] for r in shape_rhos])
        print(f"\nCross-family shape correlation (OLMo vs Pythia):")
        print(f"  Mean rho = {rhos.mean():.3f}")
        print(f"  Median rho = {np.median(rhos):.3f}")
        print(f"  Min rho = {rhos.min():.3f}")
        print(f"  >0.7: {np.mean(rhos > 0.7):.1%}")

    # Save
    out = {
        "experiment": "CTI Cross-Family: Pythia -> OLMo-2-1B",
        "overall": {"mae": float(mae), "r2": float(r2), "n": len(obs)},
        "per_dataset": ds_results,
        "shape_correlation": {
            "mean_rho": float(rhos.mean()) if shape_rhos else None,
            "median_rho": float(np.median(rhos)) if shape_rhos else None,
            "frac_above_07": float(np.mean(rhos > 0.7)) if shape_rhos else None,
            "n": len(shape_rhos),
        },
        "pythia_params_used": params,
    }
    out_path = RESULTS_DIR / "cti_cross_family_prediction.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run OLMo sweep")
    parser.add_argument("--predict", action="store_true", help="Test Pythia->OLMo prediction")
    args = parser.parse_args()

    if args.sweep:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sweep(device)
    elif args.predict:
        predict()
    else:
        print("Use --sweep or --predict")


if __name__ == "__main__":
    main()
