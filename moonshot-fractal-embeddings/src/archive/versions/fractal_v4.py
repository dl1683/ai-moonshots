"""
Fractal Embeddings V4 - Optimized for Real Gains
=================================================

Design recommendations for genuine 10%+ improvement:

1. Two-stage PEFT: Head-only first, then LoRA+head with low LR
2. Large memory bank (50k+) with hard negative mining
3. Multi-task supervision: contrastive + margin + classification

Key insight: Previous results were inflated by data leakage.
Real baseline with proper train/test: ~2-4% improvement.
Target: 10%+ genuine improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import json
import gc
import copy
from pathlib import Path
from collections import deque
import faiss

from multi_model_pipeline import MODELS, ModelConfig


class MemoryBank:
    """
    Large memory bank for hard negative mining (MoCo-style).

    Target: 50k-200k embeddings for effective hard negatives.
    """

    def __init__(self, size: int = 65536, dim: int = 256):
        self.size = size
        self.dim = dim
        self.embeddings = torch.zeros(size, dim)
        self.labels = torch.zeros(size, dtype=torch.long)
        self.ptr = 0
        self.full = False

        # FAISS index for fast ANN search
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized

    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Add embeddings to bank."""
        batch_size = embeddings.shape[0]
        embeddings = F.normalize(embeddings.detach().cpu(), dim=-1)
        labels = labels.detach().cpu()

        if self.ptr + batch_size <= self.size:
            self.embeddings[self.ptr:self.ptr + batch_size] = embeddings
            self.labels[self.ptr:self.ptr + batch_size] = labels
            self.ptr += batch_size
        else:
            # Wrap around
            remaining = self.size - self.ptr
            self.embeddings[self.ptr:] = embeddings[:remaining]
            self.labels[self.ptr:] = labels[:remaining]
            self.embeddings[:batch_size - remaining] = embeddings[remaining:]
            self.labels[:batch_size - remaining] = labels[remaining:]
            self.ptr = batch_size - remaining
            self.full = True

    def rebuild_index(self):
        """Rebuild FAISS index for hard negative mining."""
        if self.full:
            data = self.embeddings.numpy()
        else:
            data = self.embeddings[:self.ptr].numpy()

        self.index.reset()
        self.index.add(data)

    def mine_hard_negatives(
        self,
        queries: torch.Tensor,
        query_labels: torch.Tensor,
        k: int = 50,
        hard_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mine hard negatives using ANN search.

        Returns mix of hard negatives (similar but different class)
        and random negatives.
        """
        if self.index.ntotal == 0:
            return None, None

        queries_np = F.normalize(queries, dim=-1).detach().cpu().numpy()

        # Find k nearest neighbors
        D, I = self.index.search(queries_np, k)

        hard_negs = []
        hard_labels = []

        for i, (indices, query_label) in enumerate(zip(I, query_labels)):
            # Filter to different-class neighbors (hard negatives)
            neg_mask = self.labels[indices] != query_label.item()
            neg_indices = indices[neg_mask.numpy()]

            if len(neg_indices) > 0:
                # Take hardest (most similar but different class)
                n_hard = int(len(neg_indices) * hard_ratio)
                n_hard = max(1, n_hard)
                selected = neg_indices[:n_hard]
                hard_negs.append(self.embeddings[selected])
                hard_labels.append(self.labels[selected])

        if hard_negs:
            return torch.cat(hard_negs, dim=0), torch.cat(hard_labels, dim=0)
        return None, None

    def get_size(self) -> int:
        return self.size if self.full else self.ptr


class FractalHeadV4(nn.Module):
    """
    Fractal head with classification outputs for multi-task learning.
    """

    def __init__(
        self,
        input_dim: int,
        num_scales: int = 4,
        scale_dim: int = 64,
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

        # Shared fractal block
        self.norm = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.ffn_norm = nn.LayerNorm(input_dim)

        # Scale projections
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, scale_dim),
            )
            for _ in range(num_scales)
        ])

        # Classification heads for multi-task
        self.l0_classifier = nn.Sequential(
            nn.LayerNorm(self.total_dim),
            nn.Linear(self.total_dim, num_l0_classes),
        )
        self.l1_classifier = nn.Sequential(
            nn.LayerNorm(self.total_dim),
            nn.Linear(self.total_dim, num_l1_classes),
        )

        self.final_norm = nn.LayerNorm(self.total_dim)

    def forward(self, x: torch.Tensor, return_scales: bool = False) -> Dict:
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x_input = x.clone()

        scale_embeddings = []

        for scale_idx in range(self.num_scales):
            normed = self.norm(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            x = x + self.ffn(self.ffn_norm(x))

            injection = 0.1 / (1 + scale_idx * 0.3)
            x = x + injection * x_input

            scale_emb = self.scale_projs[scale_idx](x.squeeze(1))
            scale_embeddings.append(scale_emb)

        full_emb = torch.cat(scale_embeddings, dim=-1)
        full_emb = self.final_norm(full_emb)

        result = {
            'embedding': full_emb,
            'logits_l0': self.l0_classifier(full_emb),
            'logits_l1': self.l1_classifier(full_emb),
        }

        if return_scales:
            result['scale_embeddings'] = scale_embeddings

        return result


class FractalModelV4(nn.Module):
    """
    V4 model with proper PEFT support.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_l0_classes: int,
        num_l1_classes: int,
        num_scales: int = 4,
        scale_dim: int = 64,
        device: str = "cuda",
    ):
        super().__init__()

        self.config = config
        self.device = device

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

        # Start with backbone frozen
        self.freeze_backbone()

        # Fractal head
        self.fractal_head = FractalHeadV4(
            input_dim=config.hidden_dim,
            num_scales=num_scales,
            scale_dim=scale_dim,
            num_l0_classes=num_l0_classes,
            num_l1_classes=num_l1_classes,
        )

        self.embed_dim = num_scales * scale_dim
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Output dim: {self.embed_dim}")

    def freeze_backbone(self):
        """Freeze entire backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 4):
        """Unfreeze last N transformer layers + final layernorm."""
        # Find layers
        if hasattr(self.backbone, 'encoder'):
            layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, 'layers'):
            layers = self.backbone.layers
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'layers'):
            layers = self.backbone.model.layers
        else:
            print("  Warning: Could not find layers, skipping unfreeze")
            return

        total = len(layers)
        for i in range(max(0, total - n), total):
            for param in layers[i].parameters():
                param.requires_grad = True

        # Also unfreeze final layernorm
        for name, param in self.backbone.named_parameters():
            if 'final' in name.lower() or 'ln_f' in name.lower() or 'norm' in name.lower():
                if 'layer' not in name.lower():  # Not per-layer norm
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"  Unfroze last {n} layers: {trainable:,} / {total_params:,} params")

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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

    def forward(self, input_ids, attention_mask, return_scales=False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask).float()

        result = self.fractal_head(pooled, return_scales)
        result['embedding'] = F.normalize(result['embedding'], dim=-1)

        return result

    def encode(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> torch.Tensor:
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
                    batch, padding=True, truncation=True,
                    max_length=min(self.config.max_seq_len, 512),
                    return_tensors="pt"
                ).to(self.device)

                result = self.forward(inputs['input_ids'], inputs['attention_mask'])
                all_embs.append(result['embedding'].cpu())

        return torch.cat(all_embs, dim=0)


def margin_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    hard_neg_embs: Optional[torch.Tensor] = None,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss on hard negatives.
    """
    if hard_neg_embs is None:
        return torch.tensor(0.0, device=embeddings.device)

    embeddings = F.normalize(embeddings, dim=-1)
    hard_neg_embs = F.normalize(hard_neg_embs, dim=-1)

    # For each sample, compute margin loss against its hard negatives
    batch_size = embeddings.shape[0]

    # Positive pairs: same-class samples in batch
    pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    pos_mask.fill_diagonal_(False)

    losses = []
    for i in range(batch_size):
        anchor = embeddings[i:i+1]

        # Positive: same class
        pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:
            continue
        pos = embeddings[pos_indices[:1]]  # Take one positive

        # Negative: hard negatives
        neg = hard_neg_embs[:min(5, len(hard_neg_embs))]  # Use up to 5 hard negs

        # Margin loss: d(a,n) - d(a,p) + margin > 0
        pos_dist = 1 - (anchor @ pos.T).squeeze()
        neg_dists = 1 - (anchor @ neg.T).squeeze()

        loss = F.relu(pos_dist - neg_dists.min() + margin)
        losses.append(loss)

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=embeddings.device)


class TwoStageTrainer:
    """
    Two-stage training:
    Stage 1: Head-only (1-2 epochs)
    Stage 2: Head + last N layers with lower LR
    """

    def __init__(
        self,
        model: FractalModelV4,
        train_dataset,
        val_dataset,
        device: str = "cuda",
        # Stage 1 params
        stage1_epochs: int = 2,
        stage1_lr: float = 1e-4,
        # Stage 2 params
        stage2_epochs: int = 8,
        stage2_lr_head: float = 2e-5,  # Lower to prevent collapse
        stage2_lr_backbone: float = 1e-6,  # Much lower to prevent collapse
        unfreeze_layers: int = 4,
        # Memory bank
        memory_size: int = 65536,
        mine_negatives_every: int = 5,  # Mine every N batches
        # Loss weights
        contrastive_weight: float = 1.0,
        margin_weight: float = 0.5,
        class_weight: float = 0.3,
        # Other
        temperature: float = 0.07,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

        # Training params
        self.stage1_epochs = stage1_epochs
        self.stage1_lr = stage1_lr
        self.stage2_epochs = stage2_epochs
        self.stage2_lr_head = stage2_lr_head
        self.stage2_lr_backbone = stage2_lr_backbone
        self.unfreeze_layers = unfreeze_layers

        # Memory bank
        self.memory = MemoryBank(size=memory_size, dim=model.embed_dim)
        self.mine_negatives_every = mine_negatives_every

        # Loss weights
        self.contrastive_weight = contrastive_weight
        self.margin_weight = margin_weight
        self.class_weight = class_weight
        self.temperature = temperature

    def train(self, batch_size: int = 32, patience: int = 5) -> List[Dict]:
        from torch.utils.data import DataLoader

        train_ds = ContrastiveDataset(self.train_dataset.samples, self.model.tokenizer)

        def collate(batch):
            return collate_fn(batch, self.model.tokenizer, self.device)

        dataloader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate, drop_last=True
        )

        history = []
        global_best_score = -float('inf')
        global_best_state = None

        # ==================== STAGE 1: Head only ====================
        print(f"\n[Stage 1] Training HEAD ONLY ({self.stage1_epochs} epochs)")
        self.model.freeze_backbone()

        optimizer = torch.optim.AdamW(
            self.model.fractal_head.parameters(),
            lr=self.stage1_lr,
            weight_decay=0.01
        )

        for epoch in range(self.stage1_epochs):
            self.model.train()
            epoch_loss = self._train_epoch(dataloader, optimizer, epoch, 1)

            val_score = self._evaluate()
            history.append({'stage': 1, 'epoch': epoch + 1, 'loss': epoch_loss, **val_score})

            print(f"  Stage 1 Epoch {epoch+1}: loss={epoch_loss:.4f}, "
                  f"L0={val_score['l0_accuracy']:.4f}, L1={val_score['l1_accuracy']:.4f}")

            if val_score['l0_accuracy'] + val_score['l1_accuracy'] > global_best_score:
                global_best_score = val_score['l0_accuracy'] + val_score['l1_accuracy']
                global_best_state = copy.deepcopy(self.model.state_dict())

        # ==================== STAGE 2: Head + Backbone ====================
        print(f"\n[Stage 2] Training HEAD + BACKBONE ({self.stage2_epochs} epochs)")
        self.model.unfreeze_last_n_layers(self.unfreeze_layers)

        # Different LR for head vs backbone
        head_params = list(self.model.fractal_head.parameters())
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW([
            {'params': head_params, 'lr': self.stage2_lr_head},
            {'params': backbone_params, 'lr': self.stage2_lr_backbone},
        ], weight_decay=0.01)

        no_improve = 0

        for epoch in range(self.stage2_epochs):
            self.model.train()
            epoch_loss = self._train_epoch(dataloader, optimizer, epoch, 2)

            val_score = self._evaluate()
            history.append({'stage': 2, 'epoch': epoch + 1, 'loss': epoch_loss, **val_score})

            print(f"  Stage 2 Epoch {epoch+1}: loss={epoch_loss:.4f}, "
                  f"L0={val_score['l0_accuracy']:.4f}, L1={val_score['l1_accuracy']:.4f}")

            score = val_score['l0_accuracy'] + val_score['l1_accuracy']
            if score > global_best_score:
                global_best_score = score
                global_best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at Stage 2 Epoch {epoch + 1}")
                    break

        # Restore best
        if global_best_state:
            self.model.load_state_dict(global_best_state)
            print(f"\nRestored best model (score={global_best_score:.4f})")

        return history

    def _train_epoch(self, dataloader, optimizer, epoch: int, stage: int) -> float:
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward
            anchor = self.model(batch['anchor_ids'], batch['anchor_mask'], return_scales=True)
            l0_pos = self.model(batch['l0_pos_ids'], batch['l0_pos_mask'])
            l1_pos = self.model(batch['l1_pos_ids'], batch['l1_pos_mask'])

            # Contrastive loss
            anchor_emb = anchor['embedding']
            l0_pos_emb = l0_pos['embedding']
            l1_pos_emb = l1_pos['embedding']

            # L0 contrastive
            l0_logits = anchor_emb @ l0_pos_emb.T / self.temperature
            l0_targets = torch.arange(len(anchor_emb), device=self.device)
            l0_loss = F.cross_entropy(l0_logits, l0_targets)

            # L1 contrastive
            l1_logits = anchor_emb @ l1_pos_emb.T / self.temperature
            l1_loss = F.cross_entropy(l1_logits, l0_targets)

            contrastive_loss = l0_loss + l1_loss

            # Classification loss (multi-task)
            class_loss = (
                F.cross_entropy(anchor['logits_l0'], batch['l0_labels']) +
                F.cross_entropy(anchor['logits_l1'], batch['l1_labels'])
            )

            # Margin loss with hard negatives
            margin_loss_val = torch.tensor(0.0, device=self.device)
            if self.memory.get_size() > 1000 and batch_idx % self.mine_negatives_every == 0:
                hard_negs, _ = self.memory.mine_hard_negatives(
                    anchor_emb, batch['l1_labels'], k=50
                )
                if hard_negs is not None:
                    hard_negs = hard_negs.to(self.device)
                    margin_loss_val = margin_loss(anchor_emb, batch['l1_labels'], hard_negs)

            # Update memory bank
            self.memory.enqueue(anchor_emb, batch['l1_labels'])

            # Rebuild index periodically
            if batch_idx % (self.mine_negatives_every * 10) == 0:
                self.memory.rebuild_index()

            # Total loss (with NaN protection)
            loss = self.contrastive_weight * contrastive_loss + self.class_weight * class_loss
            if not torch.isnan(margin_loss_val) and margin_loss_val > 0:
                loss = loss + self.margin_weight * margin_loss_val

            # Skip if NaN
            if torch.isnan(loss):
                print(f"    Warning: NaN loss at batch {batch_idx}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _evaluate(self) -> Dict:
        self.model.eval()
        samples = self.val_dataset.samples[:min(1000, len(self.val_dataset.samples))]
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


class ContrastiveDataset:
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

        self.l0_index = {}
        self.l1_index = {}
        for i, s in enumerate(samples):
            if s.level0_label not in self.l0_index:
                self.l0_index[s.level0_label] = []
            self.l0_index[s.level0_label].append(i)

            if s.level1_label not in self.l1_index:
                self.l1_index[s.level1_label] = []
            self.l1_index[s.level1_label].append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor = self.samples[idx]

        l0_indices = self.l0_index[anchor.level0_label]
        l0_pos_idx = np.random.choice(l0_indices)
        while l0_pos_idx == idx and len(l0_indices) > 1:
            l0_pos_idx = np.random.choice(l0_indices)
        l0_pos = self.samples[l0_pos_idx]

        l1_indices = self.l1_index[anchor.level1_label]
        l1_pos_idx = np.random.choice(l1_indices)
        while l1_pos_idx == idx and len(l1_indices) > 1:
            l1_pos_idx = np.random.choice(l1_indices)
        l1_pos = self.samples[l1_pos_idx]

        return {
            'anchor_text': anchor.text,
            'l0_pos_text': l0_pos.text,
            'l1_pos_text': l1_pos.text,
            'l0_label': anchor.level0_label,
            'l1_label': anchor.level1_label,
        }


def collate_fn(batch, tokenizer, device):
    anchors = [b['anchor_text'] for b in batch]
    l0_pos = [b['l0_pos_text'] for b in batch]
    l1_pos = [b['l1_pos_text'] for b in batch]

    l0_labels = torch.tensor([b['l0_label'] for b in batch], device=device)
    l1_labels = torch.tensor([b['l1_label'] for b in batch], device=device)

    anchor_enc = tokenizer(anchors, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
    l0_enc = tokenizer(l0_pos, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
    l1_enc = tokenizer(l1_pos, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)

    return {
        'anchor_ids': anchor_enc['input_ids'],
        'anchor_mask': anchor_enc['attention_mask'],
        'l0_pos_ids': l0_enc['input_ids'],
        'l0_pos_mask': l0_enc['attention_mask'],
        'l1_pos_ids': l1_enc['input_ids'],
        'l1_pos_mask': l1_enc['attention_mask'],
        'l0_labels': l0_labels,
        'l1_labels': l1_labels,
    }


def split_train_val(samples, val_ratio=0.15, seed=42):
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


def run_v4_experiment(
    model_key: str = "bge-small",
    dataset_name: str = "yahoo",
    stage1_epochs: int = 2,
    stage2_epochs: int = 8,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Run V4 experiment with two-stage training."""
    from hierarchical_datasets import load_hierarchical_dataset

    print("=" * 70)
    print(f"FRACTAL V4: {model_key} on {dataset_name}")
    print("=" * 70)
    print("Features: Two-stage PEFT + Hard negative mining + Multi-task")

    # Load data
    print("\n[1] Loading data...")
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

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

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)

    # Baseline
    print("\n[2] Baseline...")
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

    # V4 Model
    print(f"\n[3] Training V4...")
    config = MODELS[model_key]

    model = FractalModelV4(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = TwoStageTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        unfreeze_layers=4,
        memory_size=32768,  # Smaller for faster testing
    )

    history = trainer.train(batch_size=batch_size, patience=5)

    # Final eval
    print("\n[4] Final evaluation...")
    model.eval()
    v4_results = evaluate_knn(model, test_data)
    print(f"  L0: {v4_results['l0_accuracy']:.4f}")
    print(f"  L1: {v4_results['l1_accuracy']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    delta_l0 = v4_results['l0_accuracy'] - base_results['l0_accuracy']
    delta_l1 = v4_results['l1_accuracy'] - base_results['l1_accuracy']

    print(f"  {'Metric':<10} {'Baseline':<10} {'V4':<10} {'Delta':<10}")
    print(f"  {'-'*40}")
    print(f"  {'L0':<10} {base_results['l0_accuracy']:<10.4f} {v4_results['l0_accuracy']:<10.4f} {delta_l0:+10.4f}")
    print(f"  {'L1':<10} {base_results['l1_accuracy']:<10.4f} {v4_results['l1_accuracy']:<10.4f} {delta_l1:+10.4f}")

    success = delta_l0 > 0.05 or delta_l1 > 0.05  # Target 5%+
    print(f"\n  {'SUCCESS!' if success else 'NEEDS MORE WORK'}: L0={delta_l0:+.2%}, L1={delta_l1:+.2%}")

    # Save
    results = {
        'model': model_key,
        'dataset': dataset_name,
        'baseline': base_results,
        'v4': v4_results,
        'delta': {'l0': delta_l0, 'l1': delta_l1},
        'history': history,
    }

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = Path(__file__).parent.parent / "results" / f"v4_{model_key}_{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bge-small")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_v4_experiment(
        model_key=args.model,
        dataset_name=args.dataset,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
    )
