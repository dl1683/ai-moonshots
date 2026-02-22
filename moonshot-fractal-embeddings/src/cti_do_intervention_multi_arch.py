#!/usr/bin/env python -u
"""
Multi-Arch Frozen Do-Intervention: DBpedia K=14
===============================================
Pre-registered: results/cti_multi_arch_dointerv_prereg.json

Extends the one clean causal result (pythia-160m/DBpedia, alpha=1.601, r=0.974)
to 5 diverse architectures on DBpedia K=14.

Design:
  For each model, at best layer:
  1. Extract frozen embeddings (500 per class x K=14 = 7000 samples)
  2. DO-INTERVENTION: move nearest centroid pair apart/together by delta
     while keeping within-class residuals FIXED (exact kappa control)
  3. Measure q dose-response
  4. Fit alpha_intervention and compare to alpha_prereg=1.477

Specific negative control: same sweep on FARTHEST centroid pair.
Prediction: farthest pair r << nearest pair r (specificity).

Models and best layers (from kappa_near_cache):
  EleutherAI/pythia-160m     layer 12  kappa=0.613  q=0.833
  EleutherAI/pythia-410m     layer 3   kappa=0.541  q=0.790
  google/electra-small-discriminator layer 3 kappa=0.534 q=0.758
  RWKV/rwkv-4-169m-pile      layer 12  kappa=0.805  q=0.833
  google-bert/bert-base-uncased layer 10 kappa=0.759  q=0.838

Pass criteria (pre-registered):
  Per model:
    r_nearest > 0.90
    |alpha_obs - 1.477| / 1.477 < 0.30
    r_farthest < 0.50 (specificity control)
  Aggregate:
    n_pass_alpha >= 4/5 (majority pass)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.special import logit as sp_logit

import torch
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-registered constants
ALPHA_PREREG = 1.477
K = 14
N_PER_CLASS = 500
DELTA_RANGE = np.linspace(-3.0, 3.0, 21)

# Models and their best layers on DBpedia K=14
MODELS = [
    ("EleutherAI/pythia-160m",  12, "pythia-160m"),
    ("EleutherAI/pythia-410m",   3, "pythia-410m"),
    ("google/electra-small-discriminator", 3, "electra-small"),
    ("RWKV/rwkv-4-169m-pile",   12, "rwkv-4-169m"),
    ("google-bert/bert-base-uncased", 10, "bert-base-uncased"),
]


# -------------------------------------------------------
# Embedding extraction
# -------------------------------------------------------

def extract_embeddings(model_name, layer_idx, model_short):
    """Extract frozen embeddings for DBpedia K=14, 500 per class."""
    from datasets import load_dataset
    from collections import defaultdict

    cache_path = RESULTS_DIR / f"dointerv_multi_{model_short}_l{layer_idx}.npz"
    if cache_path.exists():
        print(f"  Loading cache: {cache_path.name}")
        loaded = np.load(str(cache_path))
        return loaded["X"], loaded["y"]

    print(f"  Extracting {model_short} layer {layer_idx}...")
    ds = load_dataset("fancyzhx/dbpedia_14", split="train")

    # Balance: N_PER_CLASS per class
    class_indices = defaultdict(list)
    for i, item in enumerate(ds):
        if len(class_indices[item["label"]]) < N_PER_CLASS:
            class_indices[item["label"]].append(i)
    indices = []
    for c in sorted(class_indices.keys()):
        indices.extend(class_indices[c][:N_PER_CLASS])
    ds_sub = ds.select(indices)

    labels = np.array([ds_sub[i]["label"] for i in range(len(ds_sub))])
    texts = [ds_sub[i]["content"][:512] for i in range(len(ds_sub))]

    # Handle RWKV separately (uses different tokenizer/model)
    is_rwkv = "rwkv" in model_name.lower()

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if is_rwkv:
        from transformers import RwkvModel
        model = RwkvModel.from_pretrained(
            model_name, output_hidden_states=True, trust_remote_code=True
        ).to(DEVICE).eval()
    else:
        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(DEVICE).eval()

    batch_size = 32
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=128, padding=True).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states  # (n_layers+1, B, T, d)
        h = hs[layer_idx].float()  # (B, T, d)
        # Mean pool over tokens (with attention mask)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        all_embs.append(np.nan_to_num(emb.cpu().numpy(), nan=0, posinf=0, neginf=0))

    del model
    torch.cuda.empty_cache()

    X = np.vstack(all_embs)
    print(f"    X.shape={X.shape}, K={len(np.unique(labels))}")
    np.savez(str(cache_path), X=X, y=labels)
    return X, labels


# -------------------------------------------------------
# Kappa computation
# -------------------------------------------------------

def compute_kappa_q(X, y):
    """Compute kappa_nearest and q_norm."""
    from sklearn.neighbors import KNeighborsClassifier
    classes = np.unique(y)
    mu = {c: X[y == c].mean(0) for c in classes if (y == c).sum() >= 2}
    if len(mu) < 2:
        return None, None, None, None

    within_var = sum(np.sum((X[y == c] - mu[c])**2) for c in mu)
    n_total = sum((y == c).sum() for c in mu)
    sigma_W = float(np.sqrt(within_var / (n_total * X.shape[1])))
    if sigma_W < 1e-10:
        return None, None, None, None

    # All pairwise centroid distances
    class_list = sorted(mu.keys())
    pairwise = {(i, j): np.linalg.norm(mu[i] - mu[j])
                for i in class_list for j in class_list if i < j}
    nearest_pair = min(pairwise, key=pairwise.get)
    farthest_pair = max(pairwise, key=pairwise.get)
    kappa = float(min(pairwise.values()) / (sigma_W * np.sqrt(X.shape[1])))

    # q_norm
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
    knn.fit(X[tr], y[tr])
    acc = float(knn.score(X[te], y[te]))
    q = (acc - 1.0/K) / (1.0 - 1.0/K)

    return kappa, q, nearest_pair, farthest_pair


# -------------------------------------------------------
# Do-intervention sweep
# -------------------------------------------------------

def do_intervention_sweep(X, y, pair, direction, sigma_W):
    """Move centroids of pair in direction by delta, keep residuals fixed."""
    classes = np.unique(y)
    mu = {c: X[y == c].mean(0) for c in classes}
    ci, cj = pair

    # Direction unit vector between centroids
    diff = mu[cj] - mu[ci]
    diff_norm = np.linalg.norm(diff)
    if diff_norm < 1e-10:
        return None
    unit = diff / diff_norm

    sweep = []
    for delta in DELTA_RANGE:
        # New centroid positions: move apart (delta>0) or together (delta<0)
        shift = delta * sigma_W * unit
        new_mu = dict(mu)
        new_mu[ci] = mu[ci] - shift / 2
        new_mu[cj] = mu[cj] + shift / 2

        # Reconstruct embeddings: new_X = residuals + new centroids
        X_new = X.copy()
        for c in [ci, cj]:
            mask = y == c
            residuals = X[mask] - mu[c]
            X_new[mask] = residuals + new_mu[c]

        kappa_new, q_new, _, _ = compute_kappa_q(X_new, y)
        if kappa_new is None:
            continue

        logit_q = float(sp_logit(float(np.clip(q_new, 0.01, 0.99))))
        sweep.append({
            "delta": float(delta),
            "kappa_nearest": float(kappa_new),
            "delta_kappa": float(kappa_new - sweep[0]["kappa_nearest"]) if sweep else 0.0,
            "q": float(q_new),
            "logit_q": logit_q,
        })

    # Fix delta_kappa relative to delta=0 point
    baseline = next((s for s in sweep if abs(s["delta"]) < 1e-6), None)
    if baseline:
        baseline_kappa = baseline["kappa_nearest"]
        baseline_logit = baseline["logit_q"]
        for s in sweep:
            s["delta_kappa"] = s["kappa_nearest"] - baseline_kappa
            s["delta_logit_q"] = s["logit_q"] - baseline_logit

    # Fit
    if len(sweep) < 5:
        return None
    kappas = np.array([s["kappa_nearest"] for s in sweep])
    logit_qs = np.array([s["logit_q"] for s in sweep])
    valid = np.isfinite(kappas) & np.isfinite(logit_qs)
    if valid.sum() < 5:
        return None
    coeffs = np.polyfit(kappas[valid], logit_qs[valid], 1)
    r, _ = pearsonr(kappas[valid], logit_qs[valid])
    alpha_obs = float(coeffs[0])
    deviation = abs(alpha_obs - ALPHA_PREREG) / ALPHA_PREREG

    return {
        "sweep": sweep,
        "alpha": alpha_obs,
        "C": float(coeffs[1]),
        "r": float(r),
        "deviation_from_prereg": float(deviation),
        "n_points": int(valid.sum()),
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    print("=" * 65)
    print("Multi-Arch Frozen Do-Intervention: DBpedia K=14")
    print("=" * 65)
    print(f"alpha_prereg={ALPHA_PREREG}, K={K}, n_per_class={N_PER_CLASS}")
    print(f"Device: {DEVICE}")
    print(f"Pass: r_nearest>0.90 AND |alpha-1.477|/1.477<0.30 AND r_farthest<0.50")
    print()

    all_results = {}
    model_summaries = []

    for model_name, layer_idx, model_short in MODELS:
        print(f"\n{'='*55}")
        print(f"Model: {model_short} (layer {layer_idx})")
        print(f"{'='*55}")

        try:
            X, y = extract_embeddings(model_name, layer_idx, model_short)
        except Exception as e:
            print(f"  EXTRACTION ERROR: {e}")
            continue

        # Baseline
        kappa_bl, q_bl, nearest_pair, farthest_pair = compute_kappa_q(X, y)
        if kappa_bl is None:
            print("  FAILED baseline computation")
            continue

        sigma_W = float(np.sqrt(
            sum(np.sum((X[y == c] - X[y == c].mean(0))**2) for c in np.unique(y))
            / (len(X) * X.shape[1])
        ))

        print(f"  Baseline: kappa={kappa_bl:.4f}, q={q_bl:.4f}")
        print(f"  Nearest pair: {nearest_pair}, Farthest pair: {farthest_pair}")
        print(f"  sigma_W={sigma_W:.4f}")

        # Nearest pair sweep
        print(f"  Running nearest-pair sweep...")
        nearest_result = do_intervention_sweep(X, y, nearest_pair, "nearest", sigma_W)
        if nearest_result:
            print(f"  Nearest: alpha={nearest_result['alpha']:.4f}, r={nearest_result['r']:.4f}, "
                  f"deviation={nearest_result['deviation_from_prereg']:.3f}")

        # Farthest pair sweep (control)
        print(f"  Running farthest-pair sweep (control)...")
        farthest_result = do_intervention_sweep(X, y, farthest_pair, "farthest", sigma_W)
        if farthest_result:
            print(f"  Farthest: alpha={farthest_result['alpha']:.4f}, r={farthest_result['r']:.4f}")

        # Pass/fail
        pass_r = nearest_result['r'] > 0.90 if nearest_result else False
        pass_alpha = nearest_result['deviation_from_prereg'] < 0.30 if nearest_result else False
        pass_specific = farthest_result['r'] < 0.50 if farthest_result else False
        overall_pass = pass_r and pass_alpha and pass_specific

        print(f"  VERDICT: r_pass={pass_r}, alpha_pass={pass_alpha}, "
              f"specific={pass_specific} -> {'PASS' if overall_pass else 'FAIL'}")

        all_results[model_short] = {
            "model": model_name,
            "layer": layer_idx,
            "baseline_kappa": float(kappa_bl),
            "baseline_q": float(q_bl),
            "nearest_pair": list(nearest_pair),
            "farthest_pair": list(farthest_pair),
            "nearest": nearest_result,
            "farthest": farthest_result,
            "pass_r": bool(pass_r),
            "pass_alpha": bool(pass_alpha),
            "pass_specificity": bool(pass_specific),
            "overall_pass": bool(overall_pass),
        }
        model_summaries.append({
            "model": model_short,
            "alpha": nearest_result['alpha'] if nearest_result else None,
            "r": nearest_result['r'] if nearest_result else None,
            "r_farthest": farthest_result['r'] if farthest_result else None,
            "pass": overall_pass,
        })

    # Aggregate summary
    print("\n" + "=" * 65)
    print("AGGREGATE RESULTS")
    print("=" * 65)
    print(f"{'Model':<25} {'alpha':>8} {'r':>7} {'r_far':>7} {'pass':>6}")
    print("-" * 60)
    for s in model_summaries:
        alpha_str = f"{s['alpha']:.4f}" if s['alpha'] is not None else "  N/A "
        r_str = f"{s['r']:.4f}" if s['r'] is not None else "  N/A "
        r_far_str = f"{s['r_farthest']:.4f}" if s['r_farthest'] is not None else "  N/A "
        print(f"  {s['model']:<23} {alpha_str:>8} {r_str:>7} {r_far_str:>7} {'PASS' if s['pass'] else 'FAIL':>6}")

    n_pass = sum(1 for s in model_summaries if s['pass'])
    alpha_vals = [s['alpha'] for s in model_summaries if s['alpha'] is not None]
    print(f"\n  n_pass = {n_pass}/{len(model_summaries)}")
    if alpha_vals:
        print(f"  alpha: mean={np.mean(alpha_vals):.4f}, std={np.std(alpha_vals):.4f}, "
              f"CV={np.std(alpha_vals)/np.mean(alpha_vals):.3f}")

    primary_pass = n_pass >= 4

    # Save
    out = {
        "experiment": "do_intervention_multi_arch_dbpedia",
        "prereg_alpha": ALPHA_PREREG,
        "K": K,
        "dataset": "dbpedia14",
        "n_per_class": N_PER_CLASS,
        "primary_pass": bool(primary_pass),
        "n_pass": n_pass,
        "n_models": len(model_summaries),
        "alpha_mean": float(np.mean(alpha_vals)) if alpha_vals else None,
        "alpha_std": float(np.std(alpha_vals)) if alpha_vals else None,
        "alpha_cv": float(np.std(alpha_vals)/np.mean(alpha_vals)) if len(alpha_vals) >= 2 else None,
        "model_summaries": model_summaries,
        "model_results": {k: {kk: vv for kk, vv in v.items() if kk not in ['nearest', 'farthest']}
                          for k, v in all_results.items()},
    }
    # Convert numpy types for JSON serialization
    def np_clean(obj):
        if isinstance(obj, dict):
            return {k: np_clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [np_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = RESULTS_DIR / "cti_do_intervention_multi_arch.json"
    with open(out_path, "w") as f:
        json.dump(np_clean(out), f, indent=2)
    print(f"\nResults saved to {out_path.name}")
    print(f"PRIMARY PASS (>=4/5 models): {'PASS' if primary_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
