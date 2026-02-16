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


def apply_lora(base_model, r=16, alpha=32):
    """Apply PEFT LoRA to the base model."""
    from peft import LoraConfig, get_peft_model

    # Find target modules for Pythia (GPT-NeoX architecture)
    target_modules = ["query_key_value"]  # Pythia uses fused QKV

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base_model, config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    return peft_model


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

            # CausalLM models return logits directly when labels are provided
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            loss = outputs.loss

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

    # Classification head (half precision to match model)
    cls_head = nn.Linear(HIDDEN_DIM, n_classes).half().to(device)

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

            # CausalLM with labels computes loss internally
            outputs = model(
                input_ids=corrupted,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            loss = outputs.loss

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
    from cti_knn_sweep import extract_all_layer_representations, knn_accuracy

    data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_samples)
    texts = [s.text for s in data.samples]
    l0_labels = np.array([s.level0_label for s in data.samples])
    l1_labels = np.array([s.level1_label for s in data.samples])

    # PEFT models wrap the base model; extract_all_layer_representations
    # needs the HF model that supports output_hidden_states
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        # PEFT CausalLM wraps as model.base_model.model
        extract_model = model
    elif hasattr(model, 'base_model'):
        extract_model = model.base_model
    else:
        extract_model = model

    layer_reps = extract_all_layer_representations(
        extract_model, tokenizer, texts, pooling="last", device=device, batch_size=32
    )

    results = {}
    for layer_idx, reps in layer_reps.items():
        knn_l0 = knn_accuracy(reps, l0_labels, k=5)
        knn_l1 = knn_accuracy(reps, l1_labels, k=5)
        results[layer_idx] = {
            "layer": layer_idx,
            "knn_l0": float(knn_l0),
            "knn_l1": float(knn_l1),
        }
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

    # Apply LoRA via PEFT
    lora_model = apply_lora(base_model, LORA_R, LORA_ALPHA)

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

    # Measure geometry
    test_data = load_hierarchical_dataset(DATASET_NAME, split="test", max_samples=800)
    test_texts = [s.text for s in test_data.samples]

    print("  Measuring baseline geometry...")
    geometry = measure_geometry(base_model, tokenizer, test_texts, device)

    for li in sorted(geometry.keys()):
        g = geometry[li]
        print(f"    L{li:2d}: aniso={g['anisotropy']:.4f}  eff_rank={g['effective_rank']:.1f}  "
              f"intr_dim={g['intrinsic_dim']:.1f}")

    # Measure quality
    print("  Measuring baseline quality...")
    quality = {}
    for ds_name in EVAL_DATASETS:
        try:
            q = measure_knn_quality(base_model, tokenizer, ds_name, device)
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
