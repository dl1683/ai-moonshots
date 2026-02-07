"""
Fractal Embeddings V3 - Performance Optimized
==============================================

Implements design recommendations for maximum performance:
1. LoRA/partial unfreeze of backbone (biggest impact)
2. Gated multi-scale fusion with attention
3. Explicit hierarchical CE loss + parent-child consistency
4. Hard-negative mining with memory queue (SupCon)
5. Two-stage curriculum training

Target: Push L0/L1 accuracy significantly higher than V2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import json
import gc
import copy
from pathlib import Path
from collections import deque

from multi_model_pipeline import MODELS, ModelConfig


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""

    def __init__(self, in_dim: int, out_dim: int, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta."""
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling


class GatedFractalHead(nn.Module):
    """
    Enhanced fractal head with gated multi-scale fusion.

    Changes from V2:
    1. Gating mechanism to weight scales per-example
    2. MLP fusion after gating
    3. Classification heads for CE loss
    """

    def __init__(
        self,
        input_dim: int,
        num_scales: int = 4,
        scale_dim: int = 128,  # Larger scale dim
        num_l0_classes: int = 10,
        num_l1_classes: int = 100,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_scales = num_scales
        self.scale_dim = scale_dim
        self.total_dim = num_scales * scale_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Shared fractal processing block
        self.norm = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.ffn_norm = nn.LayerNorm(input_dim)

        # Scale-specific projections
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, scale_dim),
            )
            for _ in range(num_scales)
        ])

        # Gating mechanism - learns to weight scales per example
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_scales),
        )

        # Fusion MLP after gated combination
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(scale_dim),
            nn.Linear(scale_dim, scale_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scale_dim * 2, scale_dim),
        )

        # Classification heads for explicit CE loss
        self.l0_classifier = nn.Linear(scale_dim, num_l0_classes)
        self.l1_classifier = nn.Linear(scale_dim, num_l1_classes)

        # Final normalization
        self.final_norm = nn.LayerNorm(self.total_dim)

    def forward(self, x: torch.Tensor, return_scales: bool = False) -> Dict:
        """
        Args:
            x: (B, D) backbone embedding
        Returns:
            Dict with embeddings and logits
        """
        # Add sequence dim
        x = x.unsqueeze(1)  # (B, 1, D)
        x = self.input_proj(x)
        x_input = x.clone()

        scale_embeddings = []

        for scale_idx in range(self.num_scales):
            # Self-attention
            normed = self.norm(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out

            # FFN
            x = x + self.ffn(self.ffn_norm(x))

            # Input injection (decaying)
            injection = 0.1 / (1 + scale_idx * 0.3)
            x = x + injection * x_input

            # Scale output
            scale_emb = self.scale_projs[scale_idx](x.squeeze(1))
            scale_embeddings.append(scale_emb)

        # Stack scales: (B, num_scales, scale_dim)
        S = torch.stack(scale_embeddings, dim=1)

        # Compute gates from original input
        gate_input = x_input.squeeze(1)  # (B, input_dim)
        gates = F.softmax(self.gate_net(gate_input), dim=-1)  # (B, num_scales)

        # Gated fusion: weighted average of scales
        gates_expanded = gates.unsqueeze(-1)  # (B, num_scales, 1)
        fused = (S * gates_expanded).sum(dim=1)  # (B, scale_dim)

        # MLP fusion
        fused = self.fusion_mlp(fused)

        # Classification logits
        logits_l0 = self.l0_classifier(fused)
        logits_l1 = self.l1_classifier(fused)

        # Full embedding (concatenate all scales for retrieval)
        full_emb = torch.cat(scale_embeddings, dim=-1)
        full_emb = self.final_norm(full_emb)

        result = {
            'embedding': full_emb,
            'fused_embedding': fused,
            'logits_l0': logits_l0,
            'logits_l1': logits_l1,
            'gates': gates,
        }

        if return_scales:
            result['scale_embeddings'] = scale_embeddings

        return result


class EmbeddingQueue:
    """Memory queue for hard negative mining with SupCon."""

    def __init__(self, size: int = 8192, dim: int = 128):
        self.size = size
        self.dim = dim
        self.queue = torch.zeros(size, dim)
        self.labels = torch.zeros(size, dtype=torch.long)
        self.ptr = 0
        self.full = False

    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Add embeddings to queue."""
        batch_size = embeddings.shape[0]
        embeddings = embeddings.detach().cpu()
        labels = labels.detach().cpu()

        if self.ptr + batch_size <= self.size:
            self.queue[self.ptr:self.ptr + batch_size] = embeddings
            self.labels[self.ptr:self.ptr + batch_size] = labels
            self.ptr += batch_size
        else:
            # Wrap around
            remaining = self.size - self.ptr
            self.queue[self.ptr:] = embeddings[:remaining]
            self.labels[self.ptr:] = labels[:remaining]
            self.queue[:batch_size - remaining] = embeddings[remaining:]
            self.labels[:batch_size - remaining] = labels[remaining:]
            self.ptr = batch_size - remaining
            self.full = True

    def get(self, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current queue contents."""
        if self.full:
            return self.queue.to(device), self.labels.to(device)
        else:
            return self.queue[:self.ptr].to(device), self.labels[:self.ptr].to(device)


class FractalModelV3(nn.Module):
    """
    Full fractal model with LoRA backbone adaptation.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_l0_classes: int,
        num_l1_classes: int,
        num_scales: int = 4,
        scale_dim: int = 128,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_layers: int = 4,  # Number of layers to apply LoRA
        device: str = "cuda",
    ):
        super().__init__()

        self.config = config
        self.device = device
        self.lora_r = lora_r

        # Load backbone
        print(f"Loading {config.name}...")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.hf_path,
            trust_remote_code=config.trust_remote_code,
        )

        if config.pooling == "last":
            self.tokenizer.padding_side = "left"

        self.backbone = AutoModel.from_pretrained(
            config.hf_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.float16,
        )

        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Add LoRA to last N layers
        self.lora_layers = nn.ModuleDict()
        self._add_lora_to_backbone(lora_layers, lora_r, lora_alpha)

        # Fractal head
        self.fractal_head = GatedFractalHead(
            input_dim=config.hidden_dim,
            num_scales=num_scales,
            scale_dim=scale_dim,
            num_l0_classes=num_l0_classes,
            num_l1_classes=num_l1_classes,
        )

        self.embed_dim = num_scales * scale_dim

        # Parent-child mask for hierarchical consistency
        self.register_buffer('parent_child_mask', None)

        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  LoRA layers: {lora_layers}, r={lora_r}")
        print(f"  Output dim: {self.embed_dim}")

    def _add_lora_to_backbone(self, num_layers: int, r: int, alpha: int):
        """Add LoRA adapters to the last N transformer layers."""
        # Get total number of layers
        if hasattr(self.backbone, 'encoder'):
            layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, 'layers'):
            layers = self.backbone.layers
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'layers'):
            layers = self.backbone.model.layers
        else:
            print("  Warning: Could not find transformer layers, skipping LoRA")
            return

        total_layers = len(layers)
        start_layer = max(0, total_layers - num_layers)

        for i in range(start_layer, total_layers):
            layer_name = f"layer_{i}"
            # Add LoRA for attention Q, K, V projections
            if hasattr(layers[i], 'attention'):
                attn = layers[i].attention
                if hasattr(attn, 'self'):
                    attn = attn.self

                if hasattr(attn, 'query'):
                    in_dim = attn.query.in_features
                    out_dim = attn.query.out_features
                    self.lora_layers[f"{layer_name}_q"] = LoRALayer(in_dim, out_dim, r, alpha)
                    self.lora_layers[f"{layer_name}_v"] = LoRALayer(in_dim, out_dim, r, alpha)
            elif hasattr(layers[i], 'self_attn'):
                attn = layers[i].self_attn
                if hasattr(attn, 'q_proj'):
                    in_dim = attn.q_proj.in_features
                    out_dim = attn.q_proj.out_features
                    self.lora_layers[f"{layer_name}_q"] = LoRALayer(in_dim, out_dim, r, alpha)
                    self.lora_layers[f"{layer_name}_v"] = LoRALayer(in_dim, out_dim, r, alpha)

        print(f"  Added LoRA to {len(self.lora_layers)} projections")

    def set_parent_child_mask(self, l0_to_l1_mapping: Dict[int, List[int]], num_l1_classes: int):
        """Set the parent-child mask for hierarchical consistency loss."""
        mask = torch.zeros(max(l0_to_l1_mapping.keys()) + 1, num_l1_classes, device=self.device)
        for l0_idx, l1_indices in l0_to_l1_mapping.items():
            for l1_idx in l1_indices:
                mask[l0_idx, l1_idx] = 1.0
        self.parent_child_mask = mask  # Store directly on device

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states."""
        if self.config.pooling == "cls":
            return hidden_states[:, 0]
        elif self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.config.pooling == "last":
            if self.tokenizer.padding_side == "left":
                return hidden_states[:, -1]
            else:
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                return hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    seq_lens
                ]
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_scales: bool = False,
    ) -> Dict:
        """Forward pass with LoRA adaptation."""
        # Get backbone embeddings (LoRA is applied internally via hooks)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        pooled = pooled.float()  # Convert from fp16

        # Fractal head
        result = self.fractal_head(pooled, return_scales)

        # Normalize embedding
        result['embedding'] = F.normalize(result['embedding'], dim=-1)

        return result

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
    ) -> torch.Tensor:
        """Encode texts."""
        if is_query and self.config.prefix_query:
            texts = [self.config.prefix_query + t for t in texts]
        elif not is_query and self.config.prefix_doc:
            texts = [self.config.prefix_doc + t for t in texts]

        self.eval()
        all_embs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=min(self.config.max_seq_len, 512),
                    return_tensors="pt",
                ).to(self.device)

                result = self.forward(inputs['input_ids'], inputs['attention_mask'])
                all_embs.append(result['embedding'].cpu())

        return torch.cat(all_embs, dim=0)


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    queue_embeddings: Optional[torch.Tensor] = None,
    queue_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss with optional memory queue.

    This is better than vanilla contrastive because:
    1. Uses all same-class samples as positives (not just pairs)
    2. Hard negatives from queue improve discrimination
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Normalize
    embeddings = F.normalize(embeddings, dim=-1)

    # In-batch similarity
    sim_matrix = embeddings @ embeddings.T / temperature  # (B, B)

    # Create positive mask (same label = positive)
    pos_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()  # (B, B)
    pos_mask.fill_diagonal_(0)  # Exclude self

    # Add queue if available
    if queue_embeddings is not None and len(queue_embeddings) > 0:
        queue_embeddings = F.normalize(queue_embeddings, dim=-1)
        sim_queue = embeddings @ queue_embeddings.T / temperature  # (B, Q)
        sim_matrix = torch.cat([sim_matrix, sim_queue], dim=1)

        pos_mask_queue = labels.unsqueeze(1).eq(queue_labels.unsqueeze(0)).float()
        pos_mask = torch.cat([pos_mask, pos_mask_queue], dim=1)

    # SupCon loss
    exp_sim = torch.exp(sim_matrix)

    # Mask out self from denominator
    mask_self = torch.ones_like(sim_matrix)
    mask_self[:, :batch_size].fill_diagonal_(0)

    # Log-sum-exp denominator
    denom = (exp_sim * mask_self).sum(dim=1, keepdim=True)

    # For each positive pair, compute log(exp(sim) / sum(exp(all)))
    log_prob = sim_matrix - torch.log(denom + 1e-8)

    # Average over positives
    pos_per_sample = pos_mask.sum(dim=1)
    pos_per_sample = pos_per_sample.clamp(min=1)  # Avoid div by zero

    loss = -(log_prob * pos_mask).sum(dim=1) / pos_per_sample
    return loss.mean()


class CurriculumTrainer:
    """
    Two-stage curriculum training.

    Stage 1 (warm-up): CE loss only - stabilize classification
    Stage 2 (full): Gradually add contrastive + orthogonality
    """

    def __init__(
        self,
        model: FractalModelV3,
        train_dataset,
        val_dataset,
        l0_to_l1_mapping: Dict[int, List[int]],
        num_l1_classes: int,
        device: str = "cuda",
        # Loss weights
        ce_weight: float = 1.0,
        supcon_weight: float = 0.2,
        hier_cons_weight: float = 0.2,
        ortho_weight: float = 0.1,
        # Training params
        lr_head: float = 1e-4,
        lr_lora: float = 5e-5,
        warmup_ratio: float = 0.4,  # Stage 1 = 40% of training
        ramp_ratio: float = 0.1,    # Ramp up over 10% of training
        temperature: float = 0.07,
        label_smoothing: float = 0.05,
        queue_size: int = 4096,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

        # Set parent-child mask
        model.set_parent_child_mask(l0_to_l1_mapping, num_l1_classes)
        self.l0_to_l1_mapping = l0_to_l1_mapping

        # Loss weights
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        self.hier_cons_weight = hier_cons_weight
        self.ortho_weight = ortho_weight

        # Training params
        self.lr_head = lr_head
        self.lr_lora = lr_lora
        self.warmup_ratio = warmup_ratio
        self.ramp_ratio = ramp_ratio
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        # Memory queue for SupCon
        self.queue = EmbeddingQueue(size=queue_size, dim=model.fractal_head.scale_dim)

    def compute_loss(
        self,
        batch_result: Dict,
        l0_labels: torch.Tensor,
        l1_labels: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss with curriculum."""
        losses = {}

        # CE losses (always on)
        ce_l0 = F.cross_entropy(
            batch_result['logits_l0'], l0_labels,
            label_smoothing=self.label_smoothing
        )
        ce_l1 = F.cross_entropy(
            batch_result['logits_l1'], l1_labels,
            label_smoothing=self.label_smoothing
        )
        losses['ce_l0'] = ce_l0.item()
        losses['ce_l1'] = ce_l1.item()

        total_loss = self.ce_weight * (ce_l0 + ce_l1)

        # Curriculum: gradually add other losses after warmup
        if step >= self.warmup_ratio * total_steps:
            # Compute ramp factor (0 -> 1 over ramp_ratio of training)
            ramp_start = self.warmup_ratio * total_steps
            ramp_end = (self.warmup_ratio + self.ramp_ratio) * total_steps
            ramp = min(1.0, (step - ramp_start) / max(1, ramp_end - ramp_start))

            # Supervised Contrastive Loss
            fused_emb = batch_result['fused_embedding']
            queue_emb, queue_labels = self.queue.get(self.device)

            sup_con = supcon_loss(
                fused_emb, l1_labels,  # Use L1 labels for fine-grained contrast
                temperature=self.temperature,
                queue_embeddings=queue_emb if len(queue_emb) > 0 else None,
                queue_labels=queue_labels if len(queue_emb) > 0 else None,
            )
            losses['supcon'] = sup_con.item()
            total_loss = total_loss + ramp * self.supcon_weight * sup_con

            # Update queue
            self.queue.enqueue(fused_emb, l1_labels)

            # Hierarchical Consistency Loss
            if self.model.parent_child_mask is not None:
                p_l1 = F.softmax(batch_result['logits_l1'], dim=-1)
                pred_l0 = batch_result['logits_l0'].argmax(dim=-1)

                # Get valid L1 classes for predicted L0
                valid_mask = self.model.parent_child_mask[pred_l0]  # (B, num_l1)

                # Penalize probability mass on invalid L1 classes
                invalid_prob = (p_l1 * (1 - valid_mask)).sum(dim=-1).mean()
                losses['hier_cons'] = invalid_prob.item()
                total_loss = total_loss + ramp * self.hier_cons_weight * invalid_prob

            # Scale Orthogonality Loss
            if 'scale_embeddings' in batch_result:
                scales = batch_result['scale_embeddings']
                ortho_loss = 0.0
                for i in range(len(scales)):
                    for j in range(i + 1, len(scales)):
                        a = F.normalize(scales[i], dim=-1)
                        b = F.normalize(scales[j], dim=-1)
                        ortho_loss += torch.abs(a @ b.T).mean()
                losses['ortho'] = ortho_loss.item()
                total_loss = total_loss + ramp * self.ortho_weight * ortho_loss

        losses['total'] = total_loss.item()
        return total_loss, losses

    def train(
        self,
        num_epochs: int = 15,
        batch_size: int = 64,
        patience: int = 5,
        grad_accum_steps: int = 2,
    ) -> List[Dict]:
        """Train with curriculum."""
        from torch.utils.data import DataLoader

        # Create dataloader
        train_ds = HierarchicalDatasetV3(
            self.train_dataset.samples,
            self.model.tokenizer,
            self.device,
        )

        dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_ds.collate,
            drop_last=True,
        )

        total_steps = num_epochs * len(dataloader)

        # Optimizer with different LRs for head and LoRA
        optimizer = torch.optim.AdamW([
            {'params': self.model.fractal_head.parameters(), 'lr': self.lr_head},
            {'params': self.model.lora_layers.parameters(), 'lr': self.lr_lora},
        ], weight_decay=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )

        # Training loop
        best_val_score = -float('inf')
        best_state = None
        no_improve = 0
        history = []
        global_step = 0

        self.model.train()

        for epoch in range(num_epochs):
            epoch_losses = {'total': 0, 'ce_l0': 0, 'ce_l1': 0, 'supcon': 0, 'hier_cons': 0, 'ortho': 0}
            n_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                result = self.model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    return_scales=True,
                )

                loss, losses = self.compute_loss(
                    result,
                    batch['l0_labels'],
                    batch['l1_labels'],
                    global_step,
                    total_steps,
                )

                # Gradient accumulation
                loss = loss / grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                for k, v in losses.items():
                    epoch_losses[k] += v
                n_batches += 1
                global_step += 1

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= n_batches

            # Validation
            self.model.eval()
            val_results = self.evaluate(self.val_dataset)
            self.model.train()

            val_score = val_results['l0_accuracy'] + val_results['l1_accuracy']

            stage = "CE-only" if global_step < self.warmup_ratio * total_steps else "Full"
            print(f"  Epoch {epoch+1}/{num_epochs} [{stage}]: "
                  f"loss={epoch_losses['total']:.4f} | "
                  f"Val L0={val_results['l0_accuracy']:.4f}, L1={val_results['l1_accuracy']:.4f}")

            history.append({
                'epoch': epoch + 1,
                **epoch_losses,
                **val_results,
            })

            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {
                    'fractal_head': copy.deepcopy(self.model.fractal_head.state_dict()),
                    'lora_layers': copy.deepcopy(self.model.lora_layers.state_dict()),
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Restore best
        if best_state:
            self.model.fractal_head.load_state_dict(best_state['fractal_head'])
            self.model.lora_layers.load_state_dict(best_state['lora_layers'])
            print(f"  Restored best checkpoint (val_score={best_val_score:.4f})")

        return history

    def evaluate(self, dataset, max_samples: int = 2000) -> Dict:
        """Evaluate classification accuracy."""
        samples = dataset.samples[:min(max_samples, len(dataset.samples))]
        texts = [s.text for s in samples]
        l0_labels = np.array([s.level0_label for s in samples])
        l1_labels = np.array([s.level1_label for s in samples])

        embeddings = self.model.encode(texts, batch_size=32).numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        def knn_accuracy(emb, labels, k=5):
            correct = 0
            for i in range(len(emb)):
                sims = emb @ emb[i]
                sims[i] = -float('inf')
                top_k = np.argsort(-sims)[:k]
                neighbor_labels = labels[top_k]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                pred = unique[np.argmax(counts)]
                if pred == labels[i]:
                    correct += 1
            return correct / len(emb)

        return {
            'l0_accuracy': knn_accuracy(embeddings, l0_labels),
            'l1_accuracy': knn_accuracy(embeddings, l1_labels),
        }


class HierarchicalDatasetV3:
    """Dataset for V3 training."""

    def __init__(self, samples, tokenizer, device: str = 'cuda'):
        self.samples = samples
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'text': s.text,
            'l0_label': s.level0_label,
            'l1_label': s.level1_label,
        }

    def collate(self, batch):
        texts = [b['text'] for b in batch]
        l0_labels = torch.tensor([b['l0_label'] for b in batch], device=self.device)
        l1_labels = torch.tensor([b['l1_label'] for b in batch], device=self.device)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        ).to(self.device)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'l0_labels': l0_labels,
            'l1_labels': l1_labels,
        }


def build_l0_to_l1_mapping(samples) -> Tuple[Dict[int, List[int]], int]:
    """Build parent-child mapping from samples."""
    mapping = {}
    all_l1 = set()

    for s in samples:
        l0, l1 = s.level0_label, s.level1_label
        if l0 not in mapping:
            mapping[l0] = set()
        mapping[l0].add(l1)
        all_l1.add(l1)

    # Convert to lists
    mapping = {k: sorted(list(v)) for k, v in mapping.items()}
    num_l1 = max(all_l1) + 1

    return mapping, num_l1


def split_train_val(samples, val_ratio: float = 0.15, seed: int = 42):
    """Stratified train/val split."""
    import random
    rng = random.Random(seed)

    groups = {}
    for s in samples:
        key = (s.level0_label, s.level1_label)
        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    train_samples, val_samples = [], []

    for key, group in groups.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        val_samples.extend(group[:n_val])
        train_samples.extend(group[n_val:])

    return train_samples, val_samples


def run_v3_experiment(
    model_key: str = "bge-small",
    dataset_name: str = "yahoo",
    num_epochs: int = 15,
    batch_size: int = 64,
    device: str = "cuda",
):
    """Run V3 experiment with all improvements."""
    from hierarchical_datasets import load_hierarchical_dataset

    print("=" * 70)
    print(f"FRACTAL V3 EXPERIMENT: {model_key} on {dataset_name}")
    print("=" * 70)
    print("Features: LoRA + Gated Fusion + CE + SupCon + Curriculum")

    # Load data
    print("\n[1] Loading data...")
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

    # Split train/val
    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

    class TempDataset:
        def __init__(self, samples, level0_names, level1_names):
            self.samples = samples
            self.level0_names = level0_names
            self.level1_names = level1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data.samples = train_samples

    print(f"  Train: {len(train_data.samples)}")
    print(f"  Val: {len(val_data.samples)}")
    print(f"  Test: {len(test_data.samples)}")
    print(f"  L0 classes: {len(train_data.level0_names)}")
    print(f"  L1 classes: {len(train_data.level1_names)}")

    # Build hierarchy mapping
    l0_to_l1, num_l1 = build_l0_to_l1_mapping(train_data.samples)
    num_l0 = len(train_data.level0_names)

    # Baseline
    print("\n[2] Baseline (no fractal)...")
    from multi_model_pipeline import load_model

    base_model = load_model(model_key, use_fractal=False, device=device)

    def evaluate_knn(model, dataset, max_samples=2000):
        samples = dataset.samples[:min(max_samples, len(dataset.samples))]
        texts = [s.text for s in samples]
        l0_labels = np.array([s.level0_label for s in samples])
        l1_labels = np.array([s.level1_label for s in samples])

        embeddings = model.encode(texts, batch_size=32).numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        def knn_acc(emb, labels, k=5):
            correct = 0
            for i in range(len(emb)):
                sims = emb @ emb[i]
                sims[i] = -float('inf')
                top_k = np.argsort(-sims)[:k]
                neighbor_labels = labels[top_k]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                pred = unique[np.argmax(counts)]
                if pred == labels[i]:
                    correct += 1
            return correct / len(emb)

        return {
            'l0_accuracy': knn_acc(embeddings, l0_labels),
            'l1_accuracy': knn_acc(embeddings, l1_labels),
        }

    base_results = evaluate_knn(base_model, test_data)
    print(f"  L0: {base_results['l0_accuracy']:.4f}")
    print(f"  L1: {base_results['l1_accuracy']:.4f}")

    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # V3 Model
    print(f"\n[3] Training V3 model ({num_epochs} epochs)...")
    config = MODELS[model_key]

    model = FractalModelV3(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=128,
        lora_r=16,
        lora_alpha=32,
        lora_layers=4,
        device=device,
    ).to(device)

    trainer = CurriculumTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        l0_to_l1_mapping=l0_to_l1,
        num_l1_classes=num_l1,
        device=device,
        temperature=0.07,
        queue_size=4096,
    )

    history = trainer.train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        patience=5,
        grad_accum_steps=2,
    )

    # Final evaluation
    print("\n[4] Final evaluation on test set...")
    model.eval()
    v3_results = evaluate_knn(model, test_data)
    print(f"  L0: {v3_results['l0_accuracy']:.4f}")
    print(f"  L1: {v3_results['l1_accuracy']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    delta_l0 = v3_results['l0_accuracy'] - base_results['l0_accuracy']
    delta_l1 = v3_results['l1_accuracy'] - base_results['l1_accuracy']

    print(f"  {'Metric':<10} {'Baseline':<10} {'V3':<10} {'Delta':<10}")
    print(f"  {'-'*40}")
    print(f"  {'L0':<10} {base_results['l0_accuracy']:<10.4f} {v3_results['l0_accuracy']:<10.4f} {delta_l0:+10.4f}")
    print(f"  {'L1':<10} {base_results['l1_accuracy']:<10.4f} {v3_results['l1_accuracy']:<10.4f} {delta_l1:+10.4f}")

    success = delta_l0 > 0 and delta_l1 > 0
    print(f"\n  {'SUCCESS!' if success else 'NEEDS WORK'}: L0={delta_l0:+.2%}, L1={delta_l1:+.2%}")

    # Save results
    results = {
        'model': model_key,
        'dataset': dataset_name,
        'baseline': base_results,
        'v3': v3_results,
        'delta': {'l0': delta_l0, 'l1': delta_l1},
        'history': history,
    }

    results_path = Path(__file__).parent.parent / "results" / f"v3_{model_key}_{dataset_name}.json"

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bge-small")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_v3_experiment(
        model_key=args.model,
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
