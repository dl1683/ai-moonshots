#!/usr/bin/env python
"""
CTI Residual Surgery: Causal test of the residual-strength hypothesis.

Key experiment: Does residual connection strength control the depth profile shape?

Hypothesis:
- At alpha=1 (full residuals): flat/monotonic profiles (transformer behavior)
- At alpha=0 (no residuals): bell-shaped profiles (SSM behavior)
- Smooth transition between these as alpha varies

This would establish CAUSAL control, not just observation.

Usage:
    python -u src/cti_residual_surgery.py --model Qwen/Qwen3-0.6B --alphas 0.0,0.25,0.5,0.75,1.0
    python -u src/cti_residual_surgery.py --model state-spaces/mamba-370m-hf --mode add-residual
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset


DATASETS = ["clinc", "dbpedia_classes", "trec"]


def load_model(model_id: str, device: str = "cuda"):
    """Load model for surgery."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16,
            attn_implementation="eager",
        )
    except Exception:
        try:
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.float16,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.float16,
            )

    model = model.to(device).eval()
    cfg = getattr(model, "config", None)
    num_layers = getattr(cfg, "num_hidden_layers", 24) if cfg else 24
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M, Layers: {num_layers}")
    return model, tokenizer, num_layers, n_params


def find_residual_layers(model) -> List[Tuple[str, nn.Module]]:
    """Find layer modules where residual connections happen."""
    candidates = []
    for name, module in model.named_modules():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0 and parts[i-1] in ("layers", "blocks", "h", "layer"):
                remaining = ".".join(parts[i+1:])
                if not remaining:  # Top-level layer module
                    candidates.append((int(p), name, module))
                break
    # Deduplicate by layer number
    seen = {}
    for layer_num, name, module in candidates:
        if layer_num not in seen:
            seen[layer_num] = (name, module)
    return [(k, v[0], v[1]) for k, v in sorted(seen.items())]


class ResidualScaler:
    """Context manager that scales residual connections by alpha.

    For transformer layers with structure: output = residual + f(input),
    we modify to: output = alpha * residual + f(input)

    We achieve this by hooking into each layer's forward and scaling
    the residual component.
    """

    def __init__(self, model, alpha: float):
        self.model = model
        self.alpha = alpha
        self.hooks = []
        self.layer_inputs = {}

    def __enter__(self):
        if abs(self.alpha - 1.0) < 1e-6:
            return self  # No modification needed

        layers = find_residual_layers(self.model)
        if not layers:
            print(f"  WARNING: No hookable layers found")
            return self

        for layer_num, name, module in layers:
            # Pre-hook: capture input
            def pre_hook(mod, inp, ln=layer_num):
                if isinstance(inp, tuple) and len(inp) > 0:
                    self.layer_inputs[ln] = inp[0].detach().clone()
                return None

            # Post-hook: scale the residual component
            def post_hook(mod, inp, output, ln=layer_num):
                if ln not in self.layer_inputs:
                    return output

                residual = self.layer_inputs[ln]

                if isinstance(output, tuple):
                    hidden = output[0]
                    # residual component = hidden - f(input) is not separable
                    # Instead: new_output = alpha * residual + (hidden - residual)
                    # = hidden + (alpha - 1) * residual
                    modified = hidden + (self.alpha - 1.0) * residual.to(hidden.dtype)
                    return (modified,) + output[1:]
                elif isinstance(output, torch.Tensor):
                    modified = output + (self.alpha - 1.0) * residual.to(output.dtype)
                    return modified
                return output

            self.hooks.append(module.register_forward_pre_hook(pre_hook))
            self.hooks.append(module.register_forward_hook(post_hook))

        print(f"  Installed residual scaling hooks (alpha={self.alpha}) on {len(layers)} layers")
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.layer_inputs.clear()


@torch.no_grad()
def extract_depth_profile(
    model, tokenizer, texts: List[str], labels: np.ndarray, n_classes: int,
    alpha: float = 1.0,
    batch_size: int = 32, device: str = "cuda",
) -> Optional[Dict]:
    """Extract depth profile with given residual scaling."""

    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch_texts = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)

            try:
                outputs = model(**enc, output_hidden_states=True, return_dict=True)
            except Exception as e:
                print(f"    Forward pass failed: {e}")
                return None

            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                print(f"    No hidden states returned")
                return None

            mask = enc.get("attention_mask", torch.ones(enc["input_ids"].shape, device=device))

            for layer_idx, hs in enumerate(hidden_states):
                hs_f = hs.float()
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (hs_f * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if layer_idx not in all_hidden:
                    all_hidden[layer_idx] = []
                all_hidden[layer_idx].append(pooled.cpu().numpy())

    # Concatenate and compute kNN
    from cti_cross_paradigm import knn_accuracy

    num_layers = max(all_hidden.keys())
    Q_chance = 1.0 / n_classes
    profile = {}

    for layer_idx in sorted(all_hidden.keys()):
        emb = np.concatenate(all_hidden[layer_idx], axis=0)
        acc = knn_accuracy(emb, labels, k=5)
        x = layer_idx / max(num_layers, 1)
        Q_norm = max((acc - Q_chance) / (1.0 - Q_chance), 0.001)
        profile[layer_idx] = {"x": float(x), "knn_l1": float(acc), "Q_norm": float(Q_norm)}

    return profile


def fit_bell(xs: np.ndarray, qs: np.ndarray):
    """Fit bell shape and return R2, mu, beta."""
    from scipy.optimize import minimize
    from scipy.special import expit

    Q_clip = np.clip(qs, 0.01, 0.99)

    def loss(params):
        b, beta, mu = params
        pred = expit(b - beta * (xs - mu)**2)
        return np.mean((Q_clip - pred)**2)

    best = None
    best_loss = float("inf")
    for mu_init in [0.3, 0.5, 0.7]:
        for beta_init in [1.0, 5.0, 15.0]:
            try:
                res = minimize(loss, [0.0, beta_init, mu_init],
                             method="L-BFGS-B",
                             bounds=[(-10, 10), (0.01, 100), (-0.5, 1.5)])
                if res.fun < best_loss:
                    best_loss = res.fun
                    best = res
            except Exception:
                continue

    if best is None:
        return None

    b, beta, mu = best.x
    pred = expit(b - beta * (xs - mu)**2)
    ss_res = np.sum((Q_clip - pred)**2)
    ss_tot = np.sum((Q_clip - Q_clip.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Linear baseline
    def lin_loss(p):
        return np.mean((Q_clip - expit(p[0] + p[1] * xs))**2)
    try:
        lin_res = minimize(lin_loss, [0, 0], method="L-BFGS-B", bounds=[(-10,10),(-20,20)])
        lin_pred = expit(lin_res.x[0] + lin_res.x[1] * xs)
        lin_r2 = 1 - np.sum((Q_clip - lin_pred)**2) / ss_tot if ss_tot > 0 else 0
    except Exception:
        lin_r2 = 0

    return {
        "bell_r2": float(r2), "lin_r2": float(lin_r2),
        "delta_r2": float(r2 - lin_r2),
        "mu": float(mu), "beta": float(beta), "b": float(b),
    }


def main():
    parser = argparse.ArgumentParser(description="CTI Residual Surgery")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    output_path = Path(args.output) if args.output else RESULTS_DIR / "cti_residual_surgery.json"

    print("=" * 70)
    print("CTI RESIDUAL SURGERY EXPERIMENT")
    print("Causal test: Does residual strength control depth profile shape?")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Alphas: {alphas}")
    print(f"Datasets: {datasets}")
    print()

    model, tokenizer, num_layers, n_params = load_model(args.model, args.device)

    results = {
        "model_id": args.model,
        "num_layers": num_layers,
        "n_params": int(n_params),
        "alphas": alphas,
        "experiments": {},
    }

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*50}")

        try:
            data = load_hierarchical_dataset(ds_name, split="test", max_samples=args.max_samples)
        except Exception as e:
            print(f"  SKIP ({e})")
            continue

        texts = [s.text for s in data.samples]
        l1_labels = np.array([s.level1_label for s in data.samples])
        n_classes = len(data.level1_names)

        ds_results = {}
        for alpha in alphas:
            print(f"\n  Alpha = {alpha:.2f}")
            t0 = time.time()

            profile = extract_depth_profile(
                model, tokenizer, texts, l1_labels, n_classes,
                alpha=alpha, batch_size=args.batch_size, device=args.device,
            )

            if profile is None:
                print(f"    FAILED")
                ds_results[str(alpha)] = {"status": "error"}
                continue

            # Extract arrays for fitting
            xs = np.array([profile[l]["x"] for l in sorted(profile.keys())])
            qs = np.array([profile[l]["Q_norm"] for l in sorted(profile.keys())])

            fit = fit_bell(xs, qs)
            elapsed = time.time() - t0

            if fit:
                print(f"    Bell R2={fit['bell_r2']:.4f} mu={fit['mu']:.3f} beta={fit['beta']:.2f} "
                      f"| Lin R2={fit['lin_r2']:.4f} | Delta={fit['delta_r2']:+.4f} [{elapsed:.1f}s]")
            else:
                print(f"    Fit failed [{elapsed:.1f}s]")

            ds_results[str(alpha)] = {
                "status": "ok",
                "profile": {str(k): v for k, v in profile.items()},
                "fit": fit,
                "runtime_sec": round(elapsed, 1),
            }

        results["experiments"][ds_name] = ds_results

    # Summary table
    print(f"\n{'='*70}")
    print("RESIDUAL SURGERY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'Alpha':>6} {'Bell R2':>8} {'Lin R2':>8} {'Delta':>8} {'mu':>6} {'beta':>7}")
    print("-" * 65)

    for ds_name, ds_res in results["experiments"].items():
        for alpha_str in sorted(ds_res.keys(), key=float):
            r = ds_res[alpha_str]
            if r.get("status") != "ok" or not r.get("fit"):
                continue
            f = r["fit"]
            print(f"{ds_name:<20} {float(alpha_str):>6.2f} {f['bell_r2']:>8.4f} "
                  f"{f['lin_r2']:>8.4f} {f['delta_r2']:>+8.4f} {f['mu']:>6.3f} {f['beta']:>7.2f}")

    # Analyze the trend
    print(f"\n{'='*70}")
    print("ALPHA vs BELL STRENGTH TREND")
    print(f"{'='*70}")
    for ds_name, ds_res in results["experiments"].items():
        alpha_delta_pairs = []
        for alpha_str, r in ds_res.items():
            if r.get("status") == "ok" and r.get("fit"):
                alpha_delta_pairs.append((float(alpha_str), r["fit"]["delta_r2"], r["fit"]["beta"]))
        if len(alpha_delta_pairs) >= 3:
            alpha_delta_pairs.sort()
            alphas_arr = np.array([p[0] for p in alpha_delta_pairs])
            deltas_arr = np.array([p[1] for p in alpha_delta_pairs])
            betas_arr = np.array([p[2] for p in alpha_delta_pairs])

            from scipy.stats import spearmanr
            rho_delta, p_delta = spearmanr(alphas_arr, deltas_arr)
            rho_beta, p_beta = spearmanr(alphas_arr, betas_arr)

            print(f"\n  {ds_name}:")
            print(f"    Alpha vs Delta R2: rho={rho_delta:+.3f} (p={p_delta:.4f})")
            print(f"    Alpha vs Beta:     rho={rho_beta:+.3f} (p={p_beta:.4f})")

            if rho_delta < -0.5:
                print(f"    PREDICTION CONFIRMED: Higher alpha -> flatter profiles")
            elif rho_delta > 0.5:
                print(f"    PREDICTION REVERSED: Higher alpha -> more bell-shaped")
            else:
                print(f"    INCONCLUSIVE: No clear monotonic relationship")

    # Save
    results["completed_at"] = datetime.now(timezone.utc).isoformat()

    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2,
                 default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
