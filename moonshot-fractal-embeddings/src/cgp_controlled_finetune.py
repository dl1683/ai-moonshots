#!/usr/bin/env python
"""
cgp_controlled_finetune.py

CGP Week 1: Controlled multi-objective fine-tuning.

Takes Pythia-160M as base model and fine-tunes with LoRA using 4 different
objectives at matched FLOPs:
  1. Contrastive (SimCSE-style: same sentence augmented via dropout)
  2. MLM (masked language modeling via causal masking adaptation)
  3. Classification (hierarchical label prediction)
  4. LM (original causal language modeling objective)

After training, measures per-layer geometry and kNN quality.

Key question: Does Objective -> Geometry -> Quality hold as a causal chain?

Usage:
    python -u src/cgp_controlled_finetune.py --objective contrastive
    python -u src/cgp_controlled_finetune.py --objective mlm
    python -u src/cgp_controlled_finetune.py --objective classification
    python -u src/cgp_controlled_finetune.py --objective lm
    python -u src/cgp_controlled_finetune.py --objective all
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

# ── Configuration ─────────────────────────────────────────────────────

MODEL_KEY = "pythia-160m"
HF_PATH = "EleutherAI/pythia-160m"
HIDDEN_DIM = 768
NUM_LAYERS = 12

# Training hyperparameters (matched FLOPs)
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 16
MAX_SEQ_LEN = 128
NUM_STEPS = 500  # Matched across all objectives
LR = 2e-4
WARMUP_STEPS = 50

DATASET_NAME = "clinc"  # Primary dataset for training/eval
EVAL_DATASETS = ["clinc", "dbpedia_classes", "yahoo", "20newsgroups"]


class LoRALayer(nn.Module):
    """Simple LoRA adapter for linear layers."""

    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: int = 32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRAModel(nn.Module):
    """Wraps a pretrained model with LoRA adapters on attention layers."""

    def __init__(self, base_model, hidden_dim: int, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base_model = base_model

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add LoRA to attention query/value projections
        self.lora_layers = nn.ModuleDict()
        for i, layer in enumerate(self._get_transformer_layers()):
            attn = self._get_attention(layer)
            if attn is not None:
                self.lora_layers[f"q_{i}"] = LoRALayer(hidden_dim, hidden_dim, r, alpha)
                self.lora_layers[f"v_{i}"] = LoRALayer(hidden_dim, hidden_dim, r, alpha)

        # Register hooks to inject LoRA
        self._hooks = []
        self._register_hooks()

    def _get_transformer_layers(self):
        if hasattr(self.base_model, "gpt_neox"):
            return self.base_model.gpt_neox.layers
        elif hasattr(self.base_model, "transformer"):
            if hasattr(self.base_model.transformer, "h"):
                return self.base_model.transformer.h
        elif hasattr(self.base_model, "model"):
            if hasattr(self.base_model.model, "layers"):
                return self.base_model.model.layers
        raise ValueError(f"Cannot find transformer layers in {type(self.base_model)}")

    def _get_attention(self, layer):
        if hasattr(layer, "attention"):
            return layer.attention
        elif hasattr(layer, "self_attn"):
            return layer.self_attn
        return None

    def _register_hooks(self):
        """Register forward hooks that add LoRA output to Q and V projections."""
        layers = self._get_transformer_layers()
        for i, layer in enumerate(layers):
            attn = self._get_attention(layer)
            if attn is None:
                continue

            # Find query and value projection layers
            q_proj = None
            v_proj = None
            if hasattr(attn, "query_key_value"):
                # Pythia uses fused QKV
                q_proj = attn.query_key_value
            elif hasattr(attn, "q_proj"):
                q_proj = attn.q_proj
                v_proj = attn.v_proj

            if q_proj is not None:
                hook = self._make_hook(f"q_{i}", q_proj)
                self._hooks.append(q_proj.register_forward_hook(hook))

            if v_proj is not None:
                hook = self._make_hook(f"v_{i}", v_proj)
                self._hooks.append(v_proj.register_forward_hook(hook))

    def _make_hook(self, key, target_module):
        def hook_fn(module, input, output):
            if key in self.lora_layers:
                lora_out = self.lora_layers[key](input[0])
                return output + lora_out
        return hook_fn

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Objectives ────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt"
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


class LabeledDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def pool_last_token(hidden_states, attention_mask):
    """Pool using last non-padding token (for decoder models)."""
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.size(0)
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]


def train_contrastive(model, tokenizer, texts, device, num_steps=500):
    """SimCSE-style contrastive learning: same text through model twice with different dropout."""
    print("  Training with CONTRASTIVE objective (SimCSE-style)")
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    model.train()
    step = 0
    total_loss = 0

    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass twice (different dropout masks)
            out1 = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            out2 = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

            # Pool last hidden state
            z1 = pool_last_token(out1.hidden_states[-1], batch["attention_mask"])
            z2 = pool_last_token(out2.hidden_states[-1], batch["attention_mask"])

            # Normalize
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

            # InfoNCE loss
            sim = z1 @ z2.T / 0.05  # temperature
            labels = torch.arange(z1.size(0), device=device)
            loss = F.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 100 == 0:
                print(f"    Step {step}/{num_steps}: loss={total_loss/step:.4f}")

    return total_loss / max(step, 1)


def train_lm(model, tokenizer, texts, device, num_steps=500):
    """Causal language modeling objective."""
    print("  Training with LM objective (causal)")
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    model.train()
    step = 0
    total_loss = 0

    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Shift for causal LM
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # ignore padding

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get logits from final hidden state
            hidden = outputs.hidden_states[-1]
            # Use the base model's embedding to get logits
            if hasattr(model.base_model, "embed_out"):
                logits = model.base_model.embed_out(hidden)
            elif hasattr(model.base_model, "lm_head"):
                logits = model.base_model.lm_head(hidden)
            else:
                # Use weight tying
                logits = hidden @ model.base_model.gpt_neox.embed_in.weight.T

            # Shift logits and labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 100 == 0:
                print(f"    Step {step}/{num_steps}: loss={total_loss/step:.4f}")

    return total_loss / max(step, 1)


def train_classification(model, tokenizer, texts, labels, n_classes, device, num_steps=500):
    """Classification head on top of pooled representation."""
    print(f"  Training with CLASSIFICATION objective ({n_classes} classes)")
    dataset = LabeledDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Classification head
    cls_head = nn.Linear(HIDDEN_DIM, n_classes).to(device)

    params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(cls_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)

    model.train()
    cls_head.train()
    step = 0
    total_loss = 0

    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            pooled = pool_last_token(outputs.hidden_states[-1], attention_mask)
            logits = cls_head(pooled)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 100 == 0:
                acc = (logits.argmax(dim=-1) == targets).float().mean().item()
                print(f"    Step {step}/{num_steps}: loss={total_loss/step:.4f}, acc={acc:.3f}")

    return total_loss / max(step, 1)


def train_mlm(model, tokenizer, texts, device, num_steps=500, mask_prob=0.15):
    """Pseudo-MLM: randomly mask tokens and predict them (adapted for decoder)."""
    print("  Training with MLM objective (masked prediction)")
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    model.train()
    step = 0
    total_loss = 0
    vocab_size = tokenizer.vocab_size

    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Create mask
            mask = (torch.rand_like(input_ids.float()) < mask_prob) & (attention_mask == 1)
            # Don't mask special tokens
            mask[:, 0] = False

            labels = input_ids.clone()
            labels[~mask] = -100  # Only predict masked positions

            # Replace masked tokens with random tokens
            noise = torch.randint(0, vocab_size, input_ids.shape, device=device)
            corrupted = torch.where(mask, noise, input_ids)

            outputs = model(
                input_ids=corrupted,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden = outputs.hidden_states[-1]
            if hasattr(model.base_model, "embed_out"):
                logits = model.base_model.embed_out(hidden)
            elif hasattr(model.base_model, "lm_head"):
                logits = model.base_model.lm_head(hidden)
            else:
                logits = hidden @ model.base_model.gpt_neox.embed_in.weight.T

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 100 == 0:
                print(f"    Step {step}/{num_steps}: loss={total_loss/step:.4f}")

    return total_loss / max(step, 1)


# ── Geometry Measurement ──────────────────────────────────────────────

def measure_geometry(model, tokenizer, texts, device, max_samples=500):
    """Measure per-layer geometric properties."""
    from cti_geometry_analysis import (
        compute_anisotropy,
        compute_effective_rank,
        compute_spectral_concentration,
        compute_intrinsic_dim_mle,
    )

    model.eval()
    all_layer_reps = {}
    batch_size = 32

    texts = texts[:max_samples]

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                          max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)

            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

            attn_mask = enc["attention_mask"]
            for layer_idx, hs in enumerate(outputs.hidden_states):
                # Pool last non-padding token
                seq_lens = attn_mask.sum(dim=1) - 1
                pooled = hs[torch.arange(hs.size(0)), seq_lens]
                if layer_idx not in all_layer_reps:
                    all_layer_reps[layer_idx] = []
                all_layer_reps[layer_idx].append(pooled.cpu().numpy())

    results = {}
    for layer_idx in sorted(all_layer_reps.keys()):
        reps = np.concatenate(all_layer_reps[layer_idx], axis=0).astype(np.float32)
        s_relative = layer_idx / NUM_LAYERS

        results[layer_idx] = {
            "layer": layer_idx,
            "s_relative": float(s_relative),
            "anisotropy": compute_anisotropy(reps),
            "effective_rank": compute_effective_rank(reps),
            "spectral_concentration_top10": compute_spectral_concentration(reps, top_k=10),
            "intrinsic_dim": compute_intrinsic_dim_mle(reps, k=5),
        }

    return results


def measure_knn_quality(model, tokenizer, dataset_name, device, max_samples=1000):
    """Measure kNN quality at each layer (reusing atlas pipeline)."""
    from cti_knn_sweep import extract_all_layer_representations, evaluate_knn_per_layer

    data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_samples)
    texts = [s.text for s in data.samples]
    l0_labels = np.array([s.level0_label for s in data.samples])
    l1_labels = np.array([s.level1_label for s in data.samples])

    model.eval()
    layer_reps = extract_all_layer_representations(
        model.base_model if hasattr(model, 'base_model') else model,
        tokenizer, texts, "last", device, batch_size=32
    )

    results = evaluate_knn_per_layer(layer_reps, l0_labels, l1_labels, k=5)
    return results


# ── Main Pipeline ─────────────────────────────────────────────────────

def run_objective(objective: str, device: str = "cuda"):
    """Run one objective: fine-tune, measure geometry, measure quality."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"CGP CONTROLLED FINE-TUNE: {objective.upper()}")
    print(f"{'='*70}")

    # Load base model
    print(f"\nLoading {HF_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, torch_dtype=torch.float16)
    base_model = base_model.to(device)

    # Wrap with LoRA
    lora_model = LoRAModel(base_model, HIDDEN_DIM, LORA_R, LORA_ALPHA)
    lora_model = lora_model.to(device)
    print(f"  LoRA trainable params: {lora_model.trainable_params():,}")

    # Load training data
    data = load_hierarchical_dataset(DATASET_NAME, split="train", max_samples=5000)
    texts = [s.text for s in data.samples]
    l0_labels = [s.level0_label for s in data.samples]
    l1_labels = [s.level1_label for s in data.samples]
    n_l1 = len(data.level1_names)

    print(f"  Training data: {len(texts)} samples, {n_l1} fine classes")

    # ── Train ─────────────────────────────────────────────────────────
    t0 = time.time()

    if objective == "contrastive":
        avg_loss = train_contrastive(lora_model, tokenizer, texts, device, NUM_STEPS)
    elif objective == "lm":
        avg_loss = train_lm(lora_model, tokenizer, texts, device, NUM_STEPS)
    elif objective == "classification":
        avg_loss = train_classification(
            lora_model, tokenizer, texts, l1_labels, n_l1, device, NUM_STEPS
        )
    elif objective == "mlm":
        avg_loss = train_mlm(lora_model, tokenizer, texts, device, NUM_STEPS)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s, avg loss={avg_loss:.4f}")

    # ── Measure Geometry ──────────────────────────────────────────────
    print(f"\n  Measuring per-layer geometry...")
    test_data = load_hierarchical_dataset(DATASET_NAME, split="test", max_samples=800)
    test_texts = [s.text for s in test_data.samples]

    geometry = measure_geometry(lora_model, tokenizer, test_texts, device)

    print(f"  Geometry measured at {len(geometry)} layers")
    for li in sorted(geometry.keys()):
        g = geometry[li]
        print(f"    L{li:2d} (s={g['s_relative']:.3f}): "
              f"aniso={g['anisotropy']:.4f}  eff_rank={g['effective_rank']:.1f}  "
              f"intr_dim={g['intrinsic_dim']:.1f}")

    # ── Measure Quality ───────────────────────────────────────────────
    print(f"\n  Measuring kNN quality on eval datasets...")
    quality = {}
    for ds_name in EVAL_DATASETS:
        try:
            q = measure_knn_quality(lora_model, tokenizer, ds_name, device)
            quality[ds_name] = q
            best_l1 = max(q.values(), key=lambda x: x.get("knn_l1", 0)).get("knn_l1", 0)
            print(f"    {ds_name}: best L1={best_l1:.4f}")
        except Exception as e:
            print(f"    {ds_name}: ERROR - {e}")
            quality[ds_name] = {"error": str(e)}

    # ── Save Results ──────────────────────────────────────────────────
    result = {
        "objective": objective,
        "model": MODEL_KEY,
        "lora_r": LORA_R,
        "num_steps": NUM_STEPS,
        "avg_loss": float(avg_loss),
        "train_time_sec": float(train_time),
        "geometry": {str(k): v for k, v in geometry.items()},
        "quality": quality,
    }

    output_path = RESULTS_DIR / f"cgp_finetune_{objective}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\n  Saved to {output_path}")

    # Cleanup
    del lora_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_baseline(device: str = "cuda"):
    """Measure geometry and quality for the UNTRAINED base model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"CGP BASELINE: No fine-tuning")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, torch_dtype=torch.float16)
    base_model = base_model.to(device).eval()

    # Wrap minimally for API compatibility
    class BaseWrapper:
        def __init__(self, model):
            self.base_model = model
        def eval(self):
            self.base_model.eval()
            return self
        def __call__(self, **kwargs):
            return self.base_model(**kwargs)
        def parameters(self):
            return self.base_model.parameters()

    wrapper = BaseWrapper(base_model)

    # Measure geometry
    test_data = load_hierarchical_dataset(DATASET_NAME, split="test", max_samples=800)
    test_texts = [s.text for s in test_data.samples]

    print("  Measuring baseline geometry...")
    geometry = measure_geometry(wrapper, tokenizer, test_texts, device)

    for li in sorted(geometry.keys()):
        g = geometry[li]
        print(f"    L{li:2d}: aniso={g['anisotropy']:.4f}  eff_rank={g['effective_rank']:.1f}  "
              f"intr_dim={g['intrinsic_dim']:.1f}")

    # Measure quality
    print("  Measuring baseline quality...")
    quality = {}
    for ds_name in EVAL_DATASETS:
        try:
            q = measure_knn_quality(wrapper, tokenizer, ds_name, device)
            quality[ds_name] = q
            best_l1 = max(q.values(), key=lambda x: x.get("knn_l1", 0)).get("knn_l1", 0)
            print(f"    {ds_name}: best L1={best_l1:.4f}")
        except Exception as e:
            print(f"    {ds_name}: ERROR - {e}")

    result = {
        "objective": "baseline",
        "model": MODEL_KEY,
        "geometry": {str(k): v for k, v in geometry.items()},
        "quality": quality,
    }

    output_path = RESULTS_DIR / "cgp_finetune_baseline.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"  Saved to {output_path}")

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", default="all",
                       choices=["contrastive", "mlm", "classification", "lm", "baseline", "all"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    objectives = ["baseline", "contrastive", "mlm", "classification", "lm"] if args.objective == "all" else [args.objective]

    all_results = {}
    for obj in objectives:
        if obj == "baseline":
            result = run_baseline(args.device)
        else:
            result = run_objective(obj, args.device)
        all_results[obj] = result

    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CGP SUMMARY: Geometry Comparison Across Objectives")
        print(f"{'='*70}")

        for obj, result in all_results.items():
            geom = result.get("geometry", {})
            if not geom:
                continue
            # Get final layer geometry
            last_key = str(max(int(k) for k in geom.keys()))
            g = geom[last_key]
            print(f"\n  {obj:15s}: aniso={g['anisotropy']:.4f}  "
                  f"eff_rank={g['effective_rank']:.1f}  "
                  f"intr_dim={g['intrinsic_dim']:.1f}")

            # Get best quality
            qual = result.get("quality", {})
            for ds_name, ds_qual in qual.items():
                if isinstance(ds_qual, dict) and "error" not in ds_qual:
                    best_l1 = max(
                        (v.get("knn_l1", 0) for v in ds_qual.values()
                         if isinstance(v, dict)),
                        default=0
                    )
                    print(f"    {ds_name}: best L1={best_l1:.4f}")

    # Save combined results
    combined_path = RESULTS_DIR / "cgp_finetune_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
