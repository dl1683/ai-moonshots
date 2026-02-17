#!/usr/bin/env python
"""
CTI Cross-Paradigm Experiment: Test bell-shaped depth profiles across 7 architecture paradigms.

Nobel-track experiment: Does the Competition Model's prediction hold beyond transformers?
If yes: evidence for a universal principle of information processing.
If no: identifies what breaks the theory and which assumption fails.

Architecture paradigms tested:
1. Transformer (Qwen3, Gemma-3, Granite-4.0)
2. SSM/Mamba (pure state-space, NO attention)
3. Hybrid (Falcon-H1, Zamba2 - mixed Mamba+Transformer)
4. Liquid (LFM2 - LIV convolution + GQA)
5. RWKV (linear attention)
6. Reasoning (DeepSeek-R1-Distill, Phi-4-reasoning)
7. xLSTM (extended LSTM with exponential gating)

Usage:
    python -u src/cti_cross_paradigm.py --quick          # QUICK_TEST (6 models, ~30 min)
    python -u src/cti_cross_paradigm.py --tier1           # All Tier 1 (~2 hours)
    python -u src/cti_cross_paradigm.py --models Qwen/Qwen3-0.6B,state-spaces/mamba-370m-hf
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

# ── Model Registry ───────────────────────────────────────────────────

QUICK_TEST = [
    "Qwen/Qwen3-0.6B",
    "state-spaces/mamba-370m-hf",
    "tiiuae/Falcon-H1-0.5B-Instruct",
    "LiquidAI/LFM2-1.2B-Exp",
    "RWKV/RWKV7-Goose-World3-1.5B-HF",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

TIER1 = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "google/gemma-3-1b-it",
    "ibm-granite/granite-4.0-350m",
    "ibm-granite/granite-4.0-1b",
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b-hf",
    "tiiuae/Falcon-H1-0.5B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Instruct",
    "ibm-granite/granite-4.0-h-350m",
    "ibm-granite/granite-4.0-h-1b",
    "Zyphra/Zamba2-1.2B",
    "LiquidAI/LFM2-350M-Exp",
    "LiquidAI/LFM2-1.2B-Exp",
    "LiquidAI/LFM2-2.6B-Exp",
    "RWKV/RWKV7-Goose-World3-1.5B-HF",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

DATASETS = ["clinc", "dbpedia_classes", "agnews", "trec"]


def detect_paradigm(model_id: str) -> str:
    """Detect architecture paradigm from model ID."""
    mid = model_id.lower()
    if "mamba" in mid and "falcon" not in mid and "zamba" not in mid:
        return "ssm"
    if "falcon-h" in mid or "zamba" in mid or "granite-4.0-h" in mid or "nemotron-h" in mid:
        return "hybrid"
    if "lfm" in mid or "liquid" in mid:
        return "liquid"
    if "rwkv" in mid:
        return "rwkv"
    if "xlstm" in mid:
        return "xlstm"
    if "r1-distill" in mid or "reasoning" in mid:
        return "reasoning"
    return "transformer"


# ── Model Loading ────────────────────────────────────────────────────

def load_model(model_id: str, device: str = "cuda"):
    """Load any model with architecture-appropriate settings."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    paradigm = detect_paradigm(model_id)
    print(f"Loading {model_id} (paradigm: {paradigm})...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder models need left padding for last-token pooling
    tokenizer.padding_side = "left"

    # Try AutoModel first (returns hidden states more reliably), fall back to CausalLM
    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # Avoid flash attention issues
        )
    except Exception:
        try:
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        except Exception:
            # Some models only work as CausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

    model = model.to(device).eval()

    # Detect number of layers
    num_layers = _detect_num_layers(model)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Params: {n_params/1e6:.1f}M, Layers: {num_layers}, Paradigm: {paradigm}")
    return model, tokenizer, num_layers, n_params, paradigm


def _detect_num_layers(model) -> int:
    """Detect number of layers from model config or structure."""
    cfg = getattr(model, "config", None)
    if cfg:
        for attr in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            if hasattr(cfg, attr):
                return getattr(cfg, attr)

    # Try common layer container attributes
    for attr in ["layers", "blocks", "h", "transformer.h", "encoder.layer",
                 "model.layers", "backbone.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__"):
                return len(obj)
        except AttributeError:
            continue

    # Last resort: count modules with layer-like names
    layer_modules = set()
    for name, _ in model.named_modules():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0 and parts[i-1] in ("layers", "blocks", "h", "layer"):
                layer_modules.add(int(p))
    if layer_modules:
        return max(layer_modules) + 1

    print("  WARNING: Could not detect layer count, defaulting to 12")
    return 12


# ── Hidden State Extraction ──────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states_standard(
    model, tokenizer, texts: List[str],
    batch_size: int = 32, device: str = "cuda", max_seq_len: int = 512,
) -> Optional[Dict[int, np.ndarray]]:
    """Try standard HuggingFace hidden state extraction."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_seq_len, return_tensors="pt",
        ).to(device)

        try:
            outputs = model(**enc, output_hidden_states=True, return_dict=True)
        except TypeError:
            # Model doesn't accept output_hidden_states
            return None

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            return None

        mask = enc.get("attention_mask", torch.ones(enc["input_ids"].shape, device=device))

        for layer_idx, hs in enumerate(hidden_states):
            hs_f = hs.float()
            # Mean pooling (most robust across architectures)
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (hs_f * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            # L2 normalize
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(pooled.cpu().numpy())

    result = {}
    for layer_idx in sorted(all_hidden.keys()):
        result[layer_idx] = np.concatenate(all_hidden[layer_idx], axis=0)
    return result


@torch.no_grad()
def extract_hidden_states_hooks(
    model, tokenizer, texts: List[str], num_layers: int,
    batch_size: int = 32, device: str = "cuda", max_seq_len: int = 512,
) -> Dict[int, np.ndarray]:
    """Fallback: use forward hooks to capture intermediate activations."""
    print("  Using hook-based extraction (model doesn't support output_hidden_states)")

    # Find layer modules to hook
    target_modules = []
    for name, module in model.named_modules():
        # Common layer container patterns
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0 and parts[i-1] in ("layers", "blocks", "h", "layer"):
                layer_num = int(p)
                # Only hook the top-level layer module (not sub-modules)
                remaining = ".".join(parts[i+1:])
                if not remaining:  # This IS the layer module
                    target_modules.append((layer_num, name, module))
                break

    # Deduplicate by layer number
    seen_layers = {}
    for layer_num, name, module in target_modules:
        if layer_num not in seen_layers:
            seen_layers[layer_num] = (name, module)

    if not seen_layers:
        raise RuntimeError("Could not find any layer modules to hook")

    print(f"  Found {len(seen_layers)} hookable layers")

    all_hidden = {}
    hooks = []
    batch_activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, dict):
                hidden = output.get("hidden_states", output.get("last_hidden_state", None))
                if hidden is None:
                    for v in output.values():
                        if isinstance(v, torch.Tensor) and v.dim() == 3:
                            hidden = v
                            break
            else:
                hidden = output
            if hidden is not None and isinstance(hidden, torch.Tensor):
                batch_activations[layer_idx] = hidden.detach()
        return hook_fn

    # Register hooks
    for layer_idx in sorted(seen_layers.keys()):
        _, module = seen_layers[layer_idx]
        hooks.append(module.register_forward_hook(make_hook(layer_idx)))

    try:
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(n_batches):
            batch_texts = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=max_seq_len, return_tensors="pt",
            ).to(device)

            batch_activations.clear()
            try:
                model(**enc)
            except Exception:
                # Some models need specific kwargs
                model(enc["input_ids"])

            mask = enc.get("attention_mask", torch.ones(enc["input_ids"].shape, device=device))

            for layer_idx, hidden in batch_activations.items():
                hs_f = hidden.float()
                if hs_f.dim() == 3:
                    mask_expanded = mask.unsqueeze(-1).float()
                    if hs_f.shape[1] == mask_expanded.shape[1]:
                        pooled = (hs_f * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                    else:
                        pooled = hs_f.mean(dim=1)
                elif hs_f.dim() == 2:
                    pooled = hs_f
                else:
                    continue
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if layer_idx not in all_hidden:
                    all_hidden[layer_idx] = []
                all_hidden[layer_idx].append(pooled.cpu().numpy())
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for layer_idx in sorted(all_hidden.keys()):
        result[layer_idx] = np.concatenate(all_hidden[layer_idx], axis=0)
    return result


def extract_all_layers(
    model, tokenizer, texts: List[str], num_layers: int,
    batch_size: int = 32, device: str = "cuda",
) -> Dict[int, np.ndarray]:
    """Extract hidden states using best available method."""
    # Try standard first
    result = extract_hidden_states_standard(model, tokenizer, texts, batch_size, device)
    if result and len(result) > 1:
        print(f"  Extracted {len(result)} layers via standard API")
        return result

    # Fall back to hooks
    result = extract_hidden_states_hooks(model, tokenizer, texts, num_layers, batch_size, device)
    if result and len(result) > 1:
        print(f"  Extracted {len(result)} layers via hooks")
        return result

    raise RuntimeError("Could not extract hidden states from model")


# ── kNN Evaluation ───────────────────────────────────────────────────

def knn_accuracy(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Fast kNN accuracy using batch cosine similarity."""
    n = len(embeddings)
    if n == 0:
        return 0.0
    chunk_size = 500
    correct = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embeddings[start:end]
        sims = chunk @ embeddings.T
        for i in range(end - start):
            sims[i, start + i] = -float('inf')
        top_k_idx = np.argpartition(-sims, min(k, sims.shape[1]-1), axis=1)[:, :k]
        for i in range(end - start):
            neighbor_labels = labels[top_k_idx[i]]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            pred = unique[np.argmax(counts)]
            if pred == labels[start + i]:
                correct += 1
    return correct / n


# ── Bell-shape fitting ───────────────────────────────────────────────

def fit_bell_shape(x: np.ndarray, Q: np.ndarray):
    """Fit logit(Q) = b - beta*(x - mu)^2 to a single depth profile.

    Returns: (b, beta, mu, r2) or None if fitting fails.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    # Clip Q to avoid logit explosion
    Q_clip = np.clip(Q, 0.01, 0.99)

    def loss(params):
        b, beta, mu = params
        pred = expit(b - beta * (x - mu)**2)
        return np.mean((Q_clip - pred)**2)

    best = None
    best_loss = float("inf")
    for mu_init in [0.3, 0.5, 0.7]:
        for beta_init in [1.0, 5.0, 15.0]:
            try:
                res = minimize(loss, [0.0, beta_init, mu_init],
                             method="L-BFGS-B",
                             bounds=[(-10, 10), (0.01, 100), (-0.5, 1.5)],
                             options={"maxiter": 2000})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best = res
            except Exception:
                continue

    if best is None:
        return None

    b, beta, mu = best.x
    pred = expit(b - beta * (x - mu)**2)
    ss_res = np.sum((Q_clip - pred)**2)
    ss_tot = np.sum((Q_clip - Q_clip.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {"b": float(b), "beta": float(beta), "mu": float(mu), "r2": float(r2)}


def fit_linear_baseline(x: np.ndarray, Q: np.ndarray):
    """Fit logit(Q) = a + c*x (linear in logit space) as baseline."""
    from scipy.optimize import minimize
    from scipy.special import expit

    Q_clip = np.clip(Q, 0.01, 0.99)

    def loss(params):
        a, c = params
        pred = expit(a + c * x)
        return np.mean((Q_clip - pred)**2)

    try:
        res = minimize(loss, [0.0, 0.0], method="L-BFGS-B",
                      bounds=[(-10, 10), (-20, 20)])
        pred = expit(res.x[0] + res.x[1] * x)
        ss_res = np.sum((Q_clip - pred)**2)
        ss_tot = np.sum((Q_clip - Q_clip.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"a": float(res.x[0]), "c": float(res.x[1]), "r2": float(r2)}
    except Exception:
        return None


# ── Main Experiment ──────────────────────────────────────────────────

def run_model(
    model_id: str,
    datasets: List[str],
    k: int = 5,
    max_samples: int = 2000,
    device: str = "cuda",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run full depth-profile experiment for one model."""

    model, tokenizer, num_layers, n_params, paradigm = load_model(model_id, device)
    result = {
        "model_id": model_id,
        "paradigm": paradigm,
        "num_layers": num_layers,
        "n_params": int(n_params),
        "datasets": {},
    }

    for ds_name in datasets:
        print(f"\n  Dataset: {ds_name}")
        t0 = time.time()

        try:
            data = load_hierarchical_dataset(ds_name, split="test", max_samples=max_samples)
        except Exception as e:
            print(f"    SKIP ({e})")
            result["datasets"][ds_name] = {"status": "error", "error": str(e)}
            continue

        texts = [s.text for s in data.samples]
        l0_labels = np.array([s.level0_label for s in data.samples])
        l1_labels = np.array([s.level1_label for s in data.samples])

        print(f"    Samples: {len(texts)}, L0: {len(data.level0_names)}, L1: {len(data.level1_names)}")

        # Extract all layer representations
        try:
            layer_reps = extract_all_layers(model, tokenizer, texts, num_layers, batch_size, device)
        except Exception as e:
            print(f"    EXTRACTION FAILED: {e}")
            traceback.print_exc()
            result["datasets"][ds_name] = {"status": "error", "error": str(e)}
            continue

        # Evaluate kNN at each layer
        layers_data = {}
        for layer_idx in sorted(layer_reps.keys()):
            emb = layer_reps[layer_idx]
            acc_l1 = knn_accuracy(emb, l1_labels, k=k)
            x = layer_idx / max(num_layers, 1)
            layers_data[layer_idx] = {
                "layer": layer_idx,
                "x": float(x),
                "knn_l1": float(acc_l1),
            }
            print(f"    L{layer_idx:>2}: L1={acc_l1:.4f} x={x:.3f}")

        # Fit bell shape to the depth profile
        if len(layers_data) >= 4:
            xs = np.array([layers_data[l]["x"] for l in sorted(layers_data.keys())])
            qs = np.array([layers_data[l]["knn_l1"] for l in sorted(layers_data.keys())])

            # Normalize Q
            n_classes = len(data.level1_names)
            Q_chance = 1.0 / n_classes
            Q_norm = np.clip((qs - Q_chance) / (1.0 - Q_chance), 0.001, 0.999)

            bell_fit = fit_bell_shape(xs, Q_norm)
            linear_fit = fit_linear_baseline(xs, Q_norm)

            if bell_fit and linear_fit:
                delta_r2 = bell_fit["r2"] - linear_fit["r2"]
                print(f"    BELL: R2={bell_fit['r2']:.4f}, mu={bell_fit['mu']:.3f}, beta={bell_fit['beta']:.2f}")
                print(f"    LINEAR: R2={linear_fit['r2']:.4f}")
                print(f"    Delta R2 (bell - linear) = {delta_r2:+.4f}")
            elif bell_fit:
                delta_r2 = None
                print(f"    BELL: R2={bell_fit['r2']:.4f}, mu={bell_fit['mu']:.3f}, beta={bell_fit['beta']:.2f}")
            else:
                delta_r2 = None
                print(f"    BELL FIT FAILED")
        else:
            bell_fit = None
            linear_fit = None
            delta_r2 = None

        elapsed = time.time() - t0
        result["datasets"][ds_name] = {
            "status": "ok",
            "n_samples": len(texts),
            "n_l1_classes": len(data.level1_names),
            "layers": layers_data,
            "bell_fit": bell_fit,
            "linear_fit": linear_fit,
            "delta_r2": delta_r2,
            "runtime_sec": round(elapsed, 1),
        }

        # Free layer reps
        del layer_reps
        gc.collect()

    # Cleanup model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="CTI Cross-Paradigm Experiment")
    parser.add_argument("--quick", action="store_true", help="Run QUICK_TEST models")
    parser.add_argument("--tier1", action="store_true", help="Run all Tier 1 models")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated HuggingFace model IDs")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        model_ids = QUICK_TEST
    elif args.tier1:
        model_ids = TIER1
    elif args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    else:
        model_ids = QUICK_TEST  # Default to quick

    datasets = [d.strip() for d in args.datasets.split(",")]
    output_path = Path(args.output) if args.output else RESULTS_DIR / "cti_cross_paradigm.json"

    # Load existing results for resume
    all_results = {}
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)

    print("=" * 70)
    print("CTI CROSS-PARADIGM EXPERIMENT")
    print("Nobel-track: Testing bell-shaped depth profiles across architectures")
    print("=" * 70)
    print(f"Models: {len(model_ids)}")
    for m in model_ids:
        print(f"  [{detect_paradigm(m):>12}] {m}")
    print(f"Datasets: {datasets}")
    print(f"k={args.k}, max_samples={args.max_samples}")
    print()

    for model_id in model_ids:
        if model_id in all_results and all_results[model_id].get("status") == "complete":
            print(f"\nSKIP {model_id} (already complete)")
            continue

        print(f"\n{'='*70}")
        print(f"Model: {model_id}")
        print(f"{'='*70}")

        try:
            result = run_model(
                model_id=model_id,
                datasets=datasets,
                k=args.k,
                max_samples=args.max_samples,
                device=args.device,
                batch_size=args.batch_size,
            )
            result["status"] = "complete"
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            all_results[model_id] = result
        except Exception as e:
            print(f"\n  FATAL ERROR: {e}")
            traceback.print_exc()
            all_results[model_id] = {
                "status": "error",
                "error": str(e),
                "paradigm": detect_paradigm(model_id),
            }

        # Save after each model (resume-safe)
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
            json.dump(convert(all_results), f, indent=2,
                     default=lambda x: float(x) if hasattr(x, 'item') else x)
        print(f"\nSaved to {output_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-PARADIGM SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<45} {'Paradigm':<12} {'Bell R2':>8} {'Lin R2':>8} {'Delta':>8} {'mu':>6} {'beta':>6}")
    print("-" * 95)

    paradigm_results = {}
    for model_id in model_ids:
        r = all_results.get(model_id, {})
        if r.get("status") != "complete":
            print(f"{model_id:<45} {'ERROR':<12}")
            continue
        paradigm = r.get("paradigm", "?")
        if paradigm not in paradigm_results:
            paradigm_results[paradigm] = []

        for ds_name, ds in r.get("datasets", {}).items():
            if ds.get("status") != "ok":
                continue
            bell = ds.get("bell_fit")
            lin = ds.get("linear_fit")
            delta = ds.get("delta_r2")
            if bell:
                short_model = model_id.split("/")[-1][:30]
                print(f"{short_model:<45} {paradigm:<12} "
                      f"{bell['r2']:>8.4f} {lin['r2'] if lin else 0:>8.4f} "
                      f"{delta if delta is not None else 0:>+8.4f} "
                      f"{bell['mu']:>6.3f} {bell['beta']:>6.2f}")
                paradigm_results[paradigm].append({
                    "model": model_id, "dataset": ds_name,
                    "bell_r2": bell["r2"], "delta_r2": delta,
                    "mu": bell["mu"], "beta": bell["beta"],
                })

    # Per-paradigm summary
    print(f"\n{'='*70}")
    print("PER-PARADIGM AGGREGATE")
    print(f"{'='*70}")
    for paradigm in sorted(paradigm_results.keys()):
        entries = paradigm_results[paradigm]
        if not entries:
            continue
        bell_r2s = [e["bell_r2"] for e in entries]
        deltas = [e["delta_r2"] for e in entries if e["delta_r2"] is not None]
        mus = [e["mu"] for e in entries]
        betas = [e["beta"] for e in entries]
        n = len(entries)
        print(f"\n  {paradigm.upper()} ({n} profiles):")
        print(f"    Bell R2:  mean={np.mean(bell_r2s):.4f} +/- {np.std(bell_r2s):.4f}")
        if deltas:
            print(f"    Delta R2: mean={np.mean(deltas):+.4f} +/- {np.std(deltas):.4f} "
                  f"(bell > linear: {sum(1 for d in deltas if d > 0)}/{len(deltas)})")
        print(f"    Peak mu:  mean={np.mean(mus):.3f} +/- {np.std(mus):.3f}")
        print(f"    Curvature beta: mean={np.mean(betas):.2f} +/- {np.std(betas):.2f}")

    # Overall verdict
    all_bell_r2 = []
    all_deltas = []
    for entries in paradigm_results.values():
        for e in entries:
            all_bell_r2.append(e["bell_r2"])
            if e["delta_r2"] is not None:
                all_deltas.append(e["delta_r2"])

    if all_bell_r2:
        print(f"\n{'='*70}")
        print(f"VERDICT: Bell R2 mean={np.mean(all_bell_r2):.4f} across {len(all_bell_r2)} profiles")
        print(f"         Bell > Linear in {sum(1 for d in all_deltas if d > 0)}/{len(all_deltas)} cases")
        n_paradigms = len(paradigm_results)
        print(f"         Tested across {n_paradigms} architecture paradigms")
        if np.mean(all_bell_r2) > 0.7 and sum(1 for d in all_deltas if d > 0) > 0.7 * len(all_deltas):
            print(f"         STATUS: STRONG evidence for cross-paradigm bell shape")
        elif np.mean(all_bell_r2) > 0.5:
            print(f"         STATUS: MODERATE evidence, some paradigms may deviate")
        else:
            print(f"         STATUS: WEAK or NO evidence for universality")


if __name__ == "__main__":
    main()
