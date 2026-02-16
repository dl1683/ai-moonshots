#!/usr/bin/env python
"""CTI BLOOM Blind Test: frozen Pythia parameters predict BLOOM depth profiles.

BLOOM uses ALiBi positional encoding (not RoPE or absolute) and was trained
by BigScience with a different optimizer/schedule than Pythia/OPT/GPT-2.
This makes it a genuinely novel architecture for blind prediction.

Models:
  - BLOOM-560M: 24 layers, 1024 hidden, 16 heads, 8 training checkpoints
  - BLOOM-1.1B: 24 layers, 1536 hidden, 16 heads, 8 training checkpoints
  - BLOOM-1.7B: 24 layers, 2048 hidden, 16 heads, 7 training checkpoints

Protocol:
  1. Load FROZEN Pythia-fitted parameters (from cti_holdout_prediction.json)
  2. Load each BLOOM checkpoint
  3. Extract layer-wise representations
  4. Compute kNN accuracy at each layer
  5. Predict with frozen params (ZERO BLOOM-specific fitting)
  6. Report MAE, R2, shape correlation
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}

# BLOOM models with training checkpoints
# NOTE: Tokenizer must be loaded from final model repos, not intermediate
# (intermediate repos don't include proper tokenizer files per revision)
BLOOM_MODELS = {
    "bloom-560m": {
        "hf_path": "bigscience/bloom-560m-intermediate",
        "tokenizer_path": "bigscience/bloom-560m",  # final model for tokenizer
        "checkpoints": [
            ("global_step1000", 1000),
            ("global_step10000", 10000),
            ("global_step100000", 100000),
            ("global_step200000", 200000),
            ("global_step300000", 300000),
            ("global_step400000", 400000),
            ("global_step500000", 500000),
            ("global_step600000", 600000),
        ],
        "num_layers": 24,
        "hidden_dim": 1024,
        "tokens_per_step": 2048 * 512,  # seqlen * batch_size (BLOOM training: ~1M tokens/step)
    },
    "bloom-1.1b": {
        "hf_path": "bigscience/bloom-1b1-intermediate",
        "tokenizer_path": "bigscience/bloom-1b1",
        "checkpoints": [
            ("global_step1000", 1000),
            ("global_step10000", 10000),
            ("global_step100000", 100000),
            ("global_step200000", 200000),
            ("global_step300000", 300000),
            ("global_step400000", 400000),
            ("global_step500000", 500000),
            ("global_step600000", 600000),
        ],
        "num_layers": 24,
        "hidden_dim": 1536,
        "tokens_per_step": 2048 * 512,
    },
    "bloom-1.7b": {
        "hf_path": "bigscience/bloom-1b7-intermediate",
        "tokenizer_path": "bigscience/bloom-1b7",
        "checkpoints": [
            ("global_step1000", 1000),
            ("global_step50000", 50000),
            ("global_step100000", 100000),
            ("global_step150000", 150000),
            ("global_step200000", 200000),
            ("global_step250000", 250000),
            ("global_step300000", 300000),
        ],
        "num_layers": 24,
        "hidden_dim": 2048,
        "tokens_per_step": 2048 * 512,
    },
}


@torch.no_grad()
def extract_layer_reps(model, tokenizer, texts, device="cuda",
                       batch_size=16, max_seq_len=256):
    """Extract last-token pooled representations from all layers."""
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
            seq_lens = mask.sum(dim=1) - 1
            pooled = hs_f[torch.arange(hs_f.size(0)), seq_lens]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            if li not in all_hidden:
                all_hidden[li] = []
            all_hidden[li].append(pooled.cpu().numpy())

    return {li: np.concatenate(arrs, axis=0)
            for li, arrs in sorted(all_hidden.items())}


def knn_accuracy(embs, labels, k=20):
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


def predict_frozen(params_dict, obs_list, datasets):
    """Predict using FROZEN Pythia parameters. ZERO BLOOM fitting."""
    alpha = params_dict["alpha"]
    beta = params_dict["beta"]
    mu_0 = params_dict["mu_0"]
    mu_1 = params_dict["mu_1"]
    b_d = params_dict["b_d"]

    preds = []
    for o in obs_list:
        x_star = mu_0 + mu_1 * o["log_r"]
        logit_Q = b_d.get(o["dataset"], 0) + alpha * o["log_r"] - beta * (o["x"] - x_star) ** 2
        preds.append(expit(np.clip(logit_Q, -20, 20)))
    return np.array(preds)


def main():
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]

    # Load FROZEN Pythia parameters
    with open(RESULTS_DIR / "cti_holdout_prediction.json") as f:
        holdout = json.load(f)
    frozen_params = holdout["fit_params"]
    print("FROZEN Pythia parameters (NO BLOOM FITTING):")
    print(f"  alpha={frozen_params['alpha']:.4f}, beta={frozen_params['beta']:.4f}")
    print(f"  mu_0={frozen_params['mu_0']:.4f}, mu_1={frozen_params['mu_1']:.4f}")

    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    for name in ds_names:
        ds = load_hierarchical_dataset(name, split="test", max_samples=2000)
        texts = [s.text for s in ds.samples]
        l0 = np.array([s.level0_label for s in ds.samples])
        l1 = np.array([s.level1_label for s in ds.samples])
        datasets[name] = {"texts": texts, "l0_labels": l0, "l1_labels": l1}
        print(f"  {name}: {len(texts)} samples")

    # Process each BLOOM model + checkpoint
    all_obs = []
    results = []

    print("\n" + "=" * 70)
    print("  CTI BLOOM BLIND TEST")
    print("  Parameters frozen from Pythia. ZERO BLOOM-specific fitting.")
    print("=" * 70)

    for model_key, info in BLOOM_MODELS.items():
        print(f"\n{'='*50}")
        print(f"  {model_key}")
        print(f"{'='*50}")

        # Load tokenizer ONCE from the final model (not intermediate checkpoints)
        tokenizer = AutoTokenizer.from_pretrained(info["tokenizer_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Loaded tokenizer from {info['tokenizer_path']}")

        for ckpt_name, step in info["checkpoints"]:
            t0 = time.time()
            tokens_seen = step * info["tokens_per_step"]
            print(f"\n  --- {model_key} step={step} (tokens={tokens_seen/1e9:.1f}B) ---")

            try:

                model = AutoModel.from_pretrained(
                    info["hf_path"],
                    revision=ckpt_name,
                    torch_dtype=torch.float16,
                )
                model = model.to(device).eval()
                n_params = sum(p.numel() for p in model.parameters())
                L = info["num_layers"]
                C = 6 * n_params * tokens_seen

                print(f"    Params: {n_params:,}, L={L}, C={C:.2e}")

                result = {
                    "model": model_key,
                    "family": "bloom",
                    "step": step,
                    "checkpoint": ckpt_name,
                    "tokens_seen": tokens_seen,
                    "C_flops": C,
                    "N_params": n_params,
                    "num_layers": L,
                    "hidden_dim": info["hidden_dim"],
                    "datasets": {},
                }

                bs = 32 if n_params < 5e8 else (16 if n_params < 1e9 else 8)

                for ds_name, ds_data in datasets.items():
                    texts = ds_data["texts"]
                    l1_labels = ds_data["l1_labels"]
                    n_classes = DS_CLASSES[ds_name]
                    Q_chance = 1.0 / n_classes

                    layer_reps = extract_layer_reps(model, tokenizer, texts, device,
                                                    batch_size=bs)

                    layer_results = {}
                    for li in sorted(layer_reps.keys()):
                        x = li / L
                        reps = layer_reps[li]
                        knn_l1 = knn_accuracy(reps, l1_labels, k=20)
                        Q_norm = (knn_l1 - Q_chance) / (1.0 - Q_chance)
                        Q_norm = np.clip(Q_norm, 0.001, 0.999)

                        layer_results[li] = {
                            "layer": li, "x": round(x, 4),
                            "knn_l1": round(knn_l1, 4),
                            "Q_norm": round(float(Q_norm), 4),
                        }

                        log_r = np.log(float(C)) - np.log(float(n_params)) if C > 0 else 0
                        all_obs.append({
                            "x": x, "Q": float(Q_norm), "dataset": ds_name,
                            "model": model_key, "family": "bloom",
                            "step": step, "layer": li, "L": L,
                            "N": n_params, "C": C, "log_r": log_r,
                        })

                    best_l1 = max(layer_results.values(), key=lambda r: r["knn_l1"])
                    result["datasets"][ds_name] = {
                        "layers": layer_results,
                        "best_layer": best_l1["layer"],
                        "best_x": best_l1["x"],
                        "best_knn_l1": best_l1["knn_l1"],
                        "final_knn_l1": layer_results[max(layer_results.keys())]["knn_l1"],
                    }

                    print(f"    {ds_name}: best=L{best_l1['layer']}/{L} "
                          f"(knn_l1={best_l1['knn_l1']:.3f}, final={layer_results[max(layer_results.keys())]['knn_l1']:.3f})")

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                results.append(result)
                print(f"    Done in {time.time()-t0:.0f}s")

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({"model": model_key, "step": step, "error": str(e)})

            # Save incrementally
            _save_interim(results, all_obs, frozen_params, ds_names)

    # === BLIND PREDICTION ===
    print("\n" + "=" * 70)
    print("  BLIND PREDICTION RESULTS")
    print("=" * 70)

    # Filter out step=0 (if C=0, log_r undefined) and errors
    valid_obs = [o for o in all_obs if o["C"] > 0]

    if not valid_obs:
        print("  No valid observations!")
        return

    Q_obs = np.array([o["Q"] for o in valid_obs])
    Q_pred = predict_frozen(frozen_params, valid_obs, ds_names)

    residuals = Q_obs - Q_pred
    overall_mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    overall_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\nOverall: MAE={overall_mae:.4f}, R2={overall_r2:.4f}, N={len(valid_obs)}")

    # Per-dataset
    per_ds = {}
    for ds in ds_names:
        ds_idx = [i for i, o in enumerate(valid_obs) if o["dataset"] == ds]
        if not ds_idx:
            continue
        ds_Q = Q_obs[ds_idx]
        ds_pred = Q_pred[ds_idx]
        ds_res = ds_Q - ds_pred
        mae = float(np.mean(np.abs(ds_res)))
        ss_r = float(np.sum(ds_res ** 2))
        ss_t = float(np.sum((ds_Q - ds_Q.mean()) ** 2))
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        per_ds[ds] = {"mae": mae, "r2": r2, "n": len(ds_idx)}
        print(f"  {ds:20s}: MAE={mae:.4f}, R2={r2:.4f}, N={len(ds_idx)}")

    # Per-model
    per_model = {}
    for mk in BLOOM_MODELS:
        mk_idx = [i for i, o in enumerate(valid_obs) if o["model"] == mk]
        if not mk_idx:
            continue
        mk_Q = Q_obs[mk_idx]
        mk_pred = Q_pred[mk_idx]
        mk_res = mk_Q - mk_pred
        mae = float(np.mean(np.abs(mk_res)))
        ss_r = float(np.sum(mk_res ** 2))
        ss_t = float(np.sum((mk_Q - mk_Q.mean()) ** 2))
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        per_model[mk] = {"mae": mae, "r2": r2, "n": len(mk_idx)}
        print(f"  {mk:20s}: MAE={mae:.4f}, R2={r2:.4f}, N={len(mk_idx)}")

    # Shape correlation
    groups = {}
    for i, o in enumerate(valid_obs):
        key = (o["model"], o["dataset"], o["step"])
        if key not in groups:
            groups[key] = []
        groups[key].append((o["x"], o["Q"], Q_pred[i]))

    rhos = []
    for key, entries in groups.items():
        if len(entries) < 4:
            continue
        entries.sort(key=lambda e: e[0])
        obs_vals = [e[1] for e in entries]
        pred_vals = [e[2] for e in entries]
        rho, _ = spearmanr(obs_vals, pred_vals)
        if not np.isnan(rho):
            rhos.append(rho)

    if rhos:
        print(f"\nShape correlation: mean_rho={np.mean(rhos):.3f}, "
              f"median={np.median(rhos):.3f}, frac>0.7={np.mean(np.array(rhos)>0.7):.3f}")

    # Degradation check
    deg_count = 0
    deg_total = 0
    for mk in BLOOM_MODELS:
        for ds in ds_names:
            for step_info in BLOOM_MODELS[mk]["checkpoints"]:
                step = step_info[1]
                profile = [o for o in valid_obs if o["model"] == mk and o["dataset"] == ds and o["step"] == step]
                if len(profile) < 3:
                    continue
                profile.sort(key=lambda o: o["x"])
                best_x = max(profile, key=lambda o: o["Q"])["x"]
                best_Q = max(o["Q"] for o in profile)
                final_Q = profile[-1]["Q"]
                gap = best_Q - final_Q
                deg_total += 1
                if gap > 0.01 and best_x < 0.95:
                    deg_count += 1

    print(f"\nDegradation: {deg_count}/{deg_total} profiles ({deg_count/deg_total*100:.0f}%)")

    # Save final
    output = {
        "experiment": "CTI BLOOM Blind Test",
        "description": "Pythia-fitted parameters (FROZEN) applied to BLOOM family (ZERO BLOOM fitting)",
        "architecture_novelty": "BLOOM uses ALiBi positional encoding, different from RoPE (Pythia/OLMo) and absolute (GPT-2/OPT)",
        "frozen_params": frozen_params,
        "bloom_results": results,
        "blind_prediction": {
            "overall": {"mae": overall_mae, "r2": overall_r2, "n": len(valid_obs)},
            "per_dataset": per_ds,
            "per_model": per_model,
            "shape_correlation": {
                "mean_rho": float(np.mean(rhos)) if rhos else None,
                "median_rho": float(np.median(rhos)) if rhos else None,
                "frac_above_07": float(np.mean(np.array(rhos) > 0.7)) if rhos else None,
                "n": len(rhos),
            },
            "degradation": {
                "count": deg_count, "total": deg_total,
                "frac": deg_count / deg_total if deg_total > 0 else 0,
            },
        },
    }

    out_path = RESULTS_DIR / "cti_bloom_blind_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer,))
                  else float(x) if isinstance(x, (np.floating,)) else x)
    print(f"\nSaved to {out_path}")


def _save_interim(results, all_obs, frozen_params, ds_names):
    """Save intermediate results."""
    out = {
        "experiment": "CTI BLOOM Blind Test (interim)",
        "frozen_params": frozen_params,
        "bloom_results": results,
        "n_observations": len(all_obs),
    }
    out_path = RESULTS_DIR / "cti_bloom_blind_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer,))
                  else float(x) if isinstance(x, (np.floating,)) else x)


if __name__ == "__main__":
    main()
