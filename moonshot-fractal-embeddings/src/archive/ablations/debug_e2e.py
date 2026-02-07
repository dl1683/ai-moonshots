"""Debug E2E gradient flow."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

from multi_model_pipeline import MODELS
from hierarchical_datasets import load_hierarchical_dataset

device = "cuda"

# Load data
print("Loading data...")
train_data = load_hierarchical_dataset("yahoo", split="train", max_samples=1000)
num_l0 = len(set(item.level0_label for item in train_data))
num_l1 = len(set(item.level1_label for item in train_data))

# Load backbone
print("Loading backbone...")
model_config = MODELS["qwen3-0.6b"]
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_config.hf_path)
tokenizer.padding_side = "left"

backbone = AutoModel.from_pretrained(
    model_config.hf_path,
    torch_dtype=torch.float16,
).to(device)

# Freeze all, then unfreeze last 2 layers
for p in backbone.parameters():
    p.requires_grad = False

# Find layers in Qwen3 structure
if hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
    layers = backbone.model.layers
elif hasattr(backbone, 'layers'):
    layers = backbone.layers
else:
    print(f"Backbone type: {type(backbone)}")
    print(f"Backbone children: {[name for name, _ in backbone.named_children()]}")
    raise ValueError("Could not find layers")

for i, layer in enumerate(layers):
    if i >= len(layers) - 2:
        for p in layer.parameters():
            p.requires_grad = True
print(f"Unfroze last 2 of {len(layers)} layers")

# Simple classifier
classifier = nn.Sequential(
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, num_l1)
).to(device)

# Check trainable params
backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"Backbone trainable params: {backbone_trainable:,}")

# Setup optimizer
optimizer = AdamW([
    {'params': [p for p in backbone.parameters() if p.requires_grad], 'lr': 1e-6},
    {'params': classifier.parameters(), 'lr': 1e-4},
], weight_decay=0.01)

# Test batch
texts = [item.text for item in train_data[:8]]
labels = torch.tensor([item.level1_label for item in train_data[:8]], device=device)

print("\n=== Test 1: Forward pass ===")
backbone.eval()  # Keep in eval mode
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
outputs = backbone(**inputs)
hidden = outputs.last_hidden_state[:, -1, :].float()
print(f"Hidden shape: {hidden.shape}")
print(f"Hidden requires_grad: {hidden.requires_grad}")

logits = classifier(hidden)
loss = F.cross_entropy(logits, labels)
print(f"Loss: {loss.item()}")

print("\n=== Test 2: Backward pass ===")
loss.backward()

# Check if backbone gradients exist
backbone_grads = 0
for name, p in backbone.named_parameters():
    if p.requires_grad and p.grad is not None:
        backbone_grads += 1
        print(f"  {name}: grad norm = {p.grad.norm().item():.6f}")

print(f"\nBackbone params with gradients: {backbone_grads}")

# Check classifier gradients
for name, p in classifier.named_parameters():
    if p.grad is not None:
        print(f"  classifier.{name}: grad norm = {p.grad.norm().item():.6f}")

print("\n=== Test 3: Optimizer step ===")
# Get param norms before
backbone_norm_before = sum(p.norm().item() for p in backbone.parameters() if p.requires_grad)
classifier_norm_before = sum(p.norm().item() for p in classifier.parameters())

optimizer.step()

# Get param norms after
backbone_norm_after = sum(p.norm().item() for p in backbone.parameters() if p.requires_grad)
classifier_norm_after = sum(p.norm().item() for p in classifier.parameters())

print(f"Backbone param norm: {backbone_norm_before:.4f} -> {backbone_norm_after:.4f}")
print(f"Classifier param norm: {classifier_norm_before:.4f} -> {classifier_norm_after:.4f}")

print("\n=== Test 4: Multiple training steps ===")
for step in range(5):
    optimizer.zero_grad()
    outputs = backbone(**inputs)
    hidden = outputs.last_hidden_state[:, -1, :].float()
    logits = classifier(hidden)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()

    # Check predictions
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    print(f"  Step {step+1}: loss={loss.item():.4f}, acc={acc:.4f}")

print("\n=== Done ===")
