#!/usr/bin/env python
"""
TRAINING GEOMETRY: TRACK ETA AND EXPLAIN QUALITY-KAPPA DIVERGENCE

The training dynamics show kNN quality peaks then DECREASES even as
kappa keeps rising. This script investigates WHY by tracking additional
geometric quantities through training:

1. eta (within-class isotropy) - does the geometry change qualitatively?
2. kappa*eta (effective discriminant) - does this peak where quality peaks?
3. Intrinsic dimensionality of representations
4. Nearest-different-class vs nearest-same-class distance ratio

If eta*kappa explains the quality peak, we have a BETTER order parameter
than Fisher SNR alone — one that captures BOTH separability AND geometry.

Uses cached checkpoints from cti_training_dynamics.
"""

import json
import sys
import gc
import time
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset


def sigmoid(x, a, b, c, d_param):
    return d_param + (a - d_param) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def load_pythia_checkpoint(model_id, step, device="cuda"):
    """Load a specific Pythia training checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = f"step{step}" if step > 0 else "step0"

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()

    return model, tokenizer


def extract_full_geometry(model, tokenizer, texts, labels, device="cuda",
                           batch_size=32):
    """Extract representations and compute full geometric statistics."""
    all_reps = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = texts[i * batch_size:(i + 1) * batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)
        mask = enc.get("attention_mask",
                       torch.ones(enc["input_ids"].shape, device=device))
        # Use penultimate layer
        hs = out.hidden_states[-2].float()
        m = mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        all_reps.append(pooled.cpu().numpy())

    X = np.concatenate(all_reps, axis=0)
    d = X.shape[1]
    classes = np.unique(labels)
    K = len(classes)
    grand_mean = X.mean(axis=0)

    # --- kappa and eta ---
    S_W_trace = 0.0
    S_B_trace = 0.0

    # For eta, need eigenvalues of within-class scatter
    # Accumulate within-class covariance (d x d can be large, use trace trick)
    within_sq_frob = 0.0  # ||S_W||_F^2 = sum of squared eigenvalues = trace(S_W^2)

    for c in classes:
        X_c = X[labels == c]
        n_c = len(X_c)
        class_mean = X_c.mean(axis=0)
        diff = X_c - class_mean
        S_W_trace += np.sum(diff ** 2)

        mean_diff = class_mean - grand_mean
        S_B_trace += n_c * np.sum(mean_diff ** 2)

        # trace(S_W^2) contribution from this class
        # S_W_c = (1/n_c) * diff.T @ diff
        # We want trace(S_W_total^2) but approximate per-class
        # ||diff.T @ diff||_F^2 / n_c^2 gives trace(S_W_c^2) * n_c
        # Actually for eta we want the pooled S_W
        # Simpler: use sample covariance contribution
        if n_c > 1:
            centered = diff / np.sqrt(n_c)
            # trace(C^2) = ||C||_F^2 where C = centered.T @ centered
            gram = centered.T @ centered  # d x d
            within_sq_frob += np.sum(gram ** 2)

    kappa = S_B_trace / max(S_W_trace, 1e-10)

    # eta = trace(S_W)^2 / (d * trace(S_W^2))
    # trace(S_W) = S_W_trace / n (we use unnormalized above)
    n = len(X)
    trace_sw = S_W_trace / n
    trace_sw2 = within_sq_frob / n  # approximate
    eta = trace_sw ** 2 / max(d * trace_sw2, 1e-10)

    # --- kNN accuracy ---
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, labels)
    _, indices = knn.kneighbors(X)
    correct = sum(1 for i in range(len(X)) if labels[indices[i, 1]] == labels[i])
    knn_acc = correct / len(X)
    q = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)

    # --- Distance ratio: nearest-same vs nearest-different ---
    # Sample subset for speed
    n_sample = min(500, len(X))
    idx_sample = np.random.choice(len(X), n_sample, replace=False)

    same_dists = []
    diff_dists = []
    for i in idx_sample:
        dists = np.sum((X - X[i]) ** 2, axis=1)
        dists[i] = np.inf  # exclude self

        same_mask = labels == labels[i]
        diff_mask = labels != labels[i]
        same_mask[i] = False

        if same_mask.any():
            same_dists.append(np.min(dists[same_mask]))
        if diff_mask.any():
            diff_dists.append(np.min(dists[diff_mask]))

    mean_same = float(np.mean(same_dists)) if same_dists else 1.0
    mean_diff = float(np.mean(diff_dists)) if diff_dists else 1.0
    dist_ratio = mean_diff / max(mean_same, 1e-10)

    # --- Effective dimensionality ---
    cov_diag = np.var(X, axis=0)
    total_var = cov_diag.sum()
    if total_var > 0:
        probs_dim = cov_diag / total_var
        probs_dim = probs_dim[probs_dim > 1e-10]
        eff_dim = float(np.exp(-np.sum(probs_dim * np.log(probs_dim))))
    else:
        eff_dim = 1.0

    return {
        "kappa": float(kappa),
        "eta": float(eta),
        "kappa_eta": float(kappa * eta),
        "knn": float(knn_acc),
        "q": float(q),
        "K": int(K),
        "dist_ratio": float(dist_ratio),
        "eff_dim": float(eff_dim),
    }


def main():
    print("=" * 70)
    print("TRAINING GEOMETRY: ETA AND KAPPA*ETA THROUGH TRAINING")
    print("=" * 70)

    # Load dataset
    ds_name = "clinc"
    print(f"\nLoading {ds_name}...")
    ds = load_hierarchical_dataset(ds_name, split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels_raw = np.array([s.level1_label for s in ds.samples])
    unique_labels = np.unique(labels_raw)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels_raw])
    K = len(unique_labels)
    print(f"  {len(texts)} samples, K={K}")

    # Checkpoint steps — use fewer for speed since each needs full geometry
    steps = [0, 64, 128, 256, 512,
             1000, 2000, 4000, 8000, 16000,
             32000, 64000, 100000, 143000]

    # Cache for resumability
    cache_path = RESULTS_DIR / "cti_training_geometry_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    models = [
        ("EleutherAI/pythia-160m", "pythia-160m"),
        ("EleutherAI/pythia-410m", "pythia-410m"),
    ]

    all_results = {}

    for model_id, model_name in models:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        cache_key = f"{model_name}_clinc_geom"
        if cache_key not in cache:
            cache[cache_key] = {}

        step_results = []

        for step in steps:
            step_key = str(step)
            if step_key in cache[cache_key]:
                r = cache[cache_key][step_key]
                print(f"  step={step:>6}: kappa={r['kappa']:.4f}, eta={r['eta']:.6f}, "
                      f"k*e={r['kappa_eta']:.6f}, q={r['q']:.4f} [cached]")
                step_results.append({"step": step, **r})
                continue

            print(f"  Loading step {step}...", end=" ", flush=True)
            t0 = time.time()

            try:
                model, tokenizer = load_pythia_checkpoint(model_id, step)
                result = extract_full_geometry(model, tokenizer, texts, labels)
                dt = time.time() - t0
                print(f"kappa={result['kappa']:.4f}, eta={result['eta']:.6f}, "
                      f"k*e={result['kappa_eta']:.6f}, q={result['q']:.4f}, "
                      f"dr={result['dist_ratio']:.4f} ({dt:.1f}s)")

                cache[cache_key][step_key] = result
                step_results.append({"step": step, **result})

                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2,
                              default=lambda x: float(x) if hasattr(x, "__float__") else str(x))

                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

        all_results[model_name] = step_results

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: WHICH ORDER PARAMETER TRACKS QUALITY BEST?")
    print(f"{'='*70}")

    for model_name, step_results in all_results.items():
        if len(step_results) < 5:
            continue

        steps_arr = np.array([r["step"] for r in step_results])
        kappas = np.array([r["kappa"] for r in step_results])
        etas = np.array([r["eta"] for r in step_results])
        kappa_etas = np.array([r["kappa_eta"] for r in step_results])
        qs = np.array([r["q"] for r in step_results])
        dist_ratios = np.array([r["dist_ratio"] for r in step_results])

        print(f"\n  --- {model_name} ---")

        # Find quality peak
        peak_idx = np.argmax(qs)
        peak_step = steps_arr[peak_idx]
        print(f"  Quality peak at step {peak_step} (q={qs[peak_idx]:.4f})")

        # Correlations with quality
        for name, arr in [("kappa", kappas), ("eta", etas),
                           ("kappa*eta", kappa_etas),
                           ("dist_ratio", dist_ratios)]:
            rho, _ = spearmanr(arr, qs)
            r, _ = pearsonr(arr, qs)

            # Find peak of this metric
            peak_metric_idx = np.argmax(arr)
            peak_metric_step = steps_arr[peak_metric_idx]

            print(f"    {name:>12}: rho={rho:.4f}, r={r:.4f}, "
                  f"peaks at step {peak_metric_step}")

        # Key test: does kappa*eta peak where quality peaks?
        kappa_eta_peak_idx = np.argmax(kappa_etas)
        print(f"\n  CRITICAL TEST: quality peaks at step {peak_step}, "
              f"kappa*eta peaks at step {steps_arr[kappa_eta_peak_idx]}")
        print(f"  Match: {'YES' if peak_step == steps_arr[kappa_eta_peak_idx] else 'NO'}")

        # Sigmoid fits: q = sigmoid(x) for different x variables
        print(f"\n  Sigmoid fits across training:")
        for name, arr in [("kappa", kappas), ("kappa*eta", kappa_etas),
                           ("dist_ratio", dist_ratios)]:
            if np.std(arr) < 1e-10:
                continue
            try:
                popt, _ = curve_fit(sigmoid, arr, qs,
                                    p0=[0.6, 10, np.median(arr), 0.0],
                                    maxfev=10000)
                pred = sigmoid(arr, *popt)
                ss_tot = np.sum((qs - qs.mean()) ** 2)
                r2 = 1 - np.sum((qs - pred) ** 2) / max(ss_tot, 1e-10)
                mae = float(np.mean(np.abs(qs - pred)))
                print(f"    q = sigmoid({name:>12}): R^2={r2:.4f}, MAE={mae:.4f}")
            except Exception:
                print(f"    q = sigmoid({name:>12}): fit failed")

        # Print trajectory for visualization
        print(f"\n  Training trajectory:")
        print(f"  {'step':>6} {'kappa':>8} {'eta':>10} {'k*eta':>10} "
              f"{'q':>8} {'dr':>8}")
        for r in step_results:
            marker = " <-- Q peak" if r["step"] == peak_step else ""
            print(f"  {r['step']:>6} {r['kappa']:>8.4f} {r['eta']:>10.6f} "
                  f"{r['kappa_eta']:>10.6f} {r['q']:>8.4f} "
                  f"{r['dist_ratio']:>8.4f}{marker}")

    # ============================================================
    # POOLED ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("POOLED ANALYSIS: BEST ORDER PARAMETER ACROSS ALL TRAINING")
    print(f"{'='*70}")

    all_kappas = []
    all_etas = []
    all_kappa_etas = []
    all_qs = []
    all_drs = []

    for model_name, step_results in all_results.items():
        for r in step_results:
            all_kappas.append(r["kappa"])
            all_etas.append(r["eta"])
            all_kappa_etas.append(r["kappa_eta"])
            all_qs.append(r["q"])
            all_drs.append(r["dist_ratio"])

    all_kappas = np.array(all_kappas)
    all_etas = np.array(all_etas)
    all_kappa_etas = np.array(all_kappa_etas)
    all_qs = np.array(all_qs)
    all_drs = np.array(all_drs)

    for name, arr in [("kappa", all_kappas), ("eta", all_etas),
                       ("kappa*eta", all_kappa_etas),
                       ("dist_ratio", all_drs)]:
        rho, _ = spearmanr(arr, all_qs)
        try:
            popt, _ = curve_fit(sigmoid, arr, all_qs,
                                p0=[0.6, 10, np.median(arr), 0.0],
                                maxfev=10000)
            pred = sigmoid(arr, *popt)
            r2 = 1 - np.sum((all_qs - pred) ** 2) / np.sum((all_qs - all_qs.mean()) ** 2)
            mae = float(np.mean(np.abs(all_qs - pred)))
            print(f"  {name:>12}: rho={rho:.4f}, R^2={r2:.4f}, MAE={mae:.4f}")
        except Exception:
            print(f"  {name:>12}: rho={rho:.4f}, sigmoid fit failed")

    # ============================================================
    # SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    # Check key hypotheses
    kappa_rho = float(spearmanr(all_kappas, all_qs)[0])
    kappa_eta_rho = float(spearmanr(all_kappa_etas, all_qs)[0])
    dr_rho = float(spearmanr(all_drs, all_qs)[0])

    checks = [
        ("kappa*eta tracks quality better than kappa alone",
         abs(kappa_eta_rho) > abs(kappa_rho),
         f"|rho(k*e)|={abs(kappa_eta_rho):.4f} vs |rho(k)|={abs(kappa_rho):.4f}"),
        ("dist_ratio tracks quality better than kappa",
         abs(dr_rho) > abs(kappa_rho),
         f"|rho(dr)|={abs(dr_rho):.4f} vs |rho(k)|={abs(kappa_rho):.4f}"),
        ("eta changes significantly during training",
         any(np.std([r["eta"] for r in sr]) / np.mean([r["eta"] for r in sr]) > 0.1
             for sr in all_results.values() if len(sr) >= 3),
         "CV(eta) per model"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save results
    save_results = {
        "experiment": "training_geometry",
        "dataset": ds_name,
        "K": K,
        "models": {},
        "pooled_correlations": {
            "kappa_rho": kappa_rho,
            "kappa_eta_rho": kappa_eta_rho,
            "dist_ratio_rho": dr_rho,
        },
    }
    for model_name, step_results in all_results.items():
        save_results["models"][model_name] = step_results

    out_path = RESULTS_DIR / "cti_training_geometry.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
