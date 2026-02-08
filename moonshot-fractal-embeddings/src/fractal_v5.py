"""
Fractal Embeddings V5 - Hierarchy-Aligned Progressive Prefix Supervision
=========================================================================

Architecture:
- Progressive prefix supervision: sample j ∈ {1,2,3,4} with P(j)=[0.4,0.3,0.2,0.1]
- Block-dropout: zero blocks > j for prefix, keep probs [0.95,0.9,0.8,0.7] for full
- Two heads: Head_top (top-level from prefix), Head_leaf (leaf from full)
- Losses: L_full + 0.6*L_prefix

This directly enforces scale separation by construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import gc
import copy
from pathlib import Path

from multi_model_pipeline import MODELS, ModelConfig

# For mixed precision training (fix for NaN in Stage 2)
from torch.cuda.amp import autocast, GradScaler


class FractalHeadV5(nn.Module):
    """
    Fractal head with hierarchy-aligned prefix supervision.

    Key differences from V4:
    - Separate classification heads for prefix (top-level) and full (leaf)
    - Block-dropout support
    - Returns scale blocks separately for prefix construction
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

        # Shared fractal block (applied recursively)
        self.norm = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.ffn_norm = nn.LayerNorm(input_dim)

        # Scale projections (one per scale)
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, scale_dim),
            )
            for _ in range(num_scales)
        ])

        # Separate heads per design spec:
        # Head_top: predicts TOP-LEVEL label from PREFIX (j < 4)
        # Head_leaf: predicts LEAF label from FULL (j = 4)

        # For prefix, we need to handle variable length, so we use max dim
        self.head_top = nn.Sequential(
            nn.LayerNorm(self.total_dim),  # Will zero-pad shorter prefixes
            nn.Linear(self.total_dim, num_l0_classes),
        )

        self.head_leaf = nn.Sequential(
            nn.LayerNorm(self.total_dim),
            nn.Linear(self.total_dim, num_l1_classes),
        )

        self.final_norm = nn.LayerNorm(self.total_dim)

    def forward(
        self,
        x: torch.Tensor,
        block_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass returning block embeddings for flexible prefix construction.

        Args:
            x: Input tensor [batch, hidden_dim]
            block_dropout_mask: Optional [batch, num_scales] mask for block dropout
                               1 = keep, 0 = drop

        Returns:
            Dict with:
            - blocks: List of [batch, scale_dim] tensors for each scale
            - full_embedding: [batch, total_dim] normalized full embedding
        """
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # [batch, 1, hidden]
        x = self.input_proj(x)
        x_input = x.clone()

        blocks = []

        for scale_idx in range(self.num_scales):
            # Shared transformer block
            normed = self.norm(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            x = x + self.ffn(self.ffn_norm(x))

            # Residual injection (decreasing with scale)
            injection = 0.1 / (1 + scale_idx * 0.3)
            x = x + injection * x_input

            # Project to scale embedding
            block_emb = self.scale_projs[scale_idx](x.squeeze(1))  # [batch, scale_dim]
            blocks.append(block_emb)

        # Apply block dropout if provided
        if block_dropout_mask is not None:
            for i in range(self.num_scales):
                # Expand mask for broadcasting: [batch, 1] * [batch, scale_dim]
                mask = block_dropout_mask[:, i:i+1]  # [batch, 1]
                blocks[i] = blocks[i] * mask

        # Full embedding (all blocks concatenated)
        full_emb = torch.cat(blocks, dim=-1)  # [batch, total_dim]
        full_emb = self.final_norm(full_emb)
        full_emb = F.normalize(full_emb, dim=-1)

        return {
            'blocks': blocks,
            'full_embedding': full_emb,
        }

    def get_prefix_embedding(self, blocks: List[torch.Tensor], prefix_len: int) -> torch.Tensor:
        """
        Construct prefix embedding from first prefix_len blocks.
        Zero-pads to total_dim for consistent classifier input.
        """
        batch_size = blocks[0].shape[0]
        device = blocks[0].device

        # Concatenate first prefix_len blocks
        prefix_blocks = blocks[:prefix_len]
        prefix_emb = torch.cat(prefix_blocks, dim=-1)  # [batch, prefix_len * scale_dim]

        # Zero-pad to full dimension
        pad_size = self.total_dim - prefix_emb.shape[-1]
        if pad_size > 0:
            padding = torch.zeros(batch_size, pad_size, device=device)
            prefix_emb = torch.cat([prefix_emb, padding], dim=-1)

        # Normalize the non-zero part
        prefix_emb = F.normalize(prefix_emb, dim=-1)

        return prefix_emb

    def classify_top(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classify top-level (L0) from prefix embedding."""
        return self.head_top(embedding)

    def classify_leaf(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classify leaf-level (L1) from full embedding."""
        return self.head_leaf(embedding)


class FractalModelV5(nn.Module):
    """
    V5 model with hierarchy-aligned prefix supervision.
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
        self.num_scales = num_scales
        self.scale_dim = scale_dim

        print(f"Loading {config.name}...")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.hf_path,
            trust_remote_code=config.trust_remote_code,
        )

        if config.pooling == "last":
            self.tokenizer.padding_side = "left"

        # Load in FP32 - let AMP handle mixed precision during training
        # (FP16 loading caused dtype mismatch issues in Stage 2)
        self.backbone = AutoModel.from_pretrained(
            config.hf_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.float32,
        )

        # Start with backbone frozen
        self.freeze_backbone()

        # Fractal head
        self.fractal_head = FractalHeadV5(
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
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 4):
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

        for name, param in self.backbone.named_parameters():
            if 'final' in name.lower() or 'ln_f' in name.lower() or 'norm' in name.lower():
                if 'layer' not in name.lower():
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

    def forward(
        self,
        input_ids,
        attention_mask,
        block_dropout_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask).float()

        result = self.fractal_head(pooled, block_dropout_mask)
        return result

    def encode(self, texts: List[str], batch_size: int = 32, prefix_len: Optional[int] = None) -> torch.Tensor:
        """
        Encode texts. If prefix_len specified, return prefix embedding instead of full.
        """
        if self.config.prefix_query:
            texts = [self.config.prefix_query + t for t in texts]

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

                if prefix_len is not None and prefix_len < self.num_scales:
                    emb = self.fractal_head.get_prefix_embedding(result['blocks'], prefix_len)
                else:
                    emb = result['full_embedding']

                all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)


class V5Trainer:
    """
    Trainer implementing hierarchy-aligned progressive prefix supervision.

    Key features:
    - Prefix sampling: j ∈ {1,2,3,4} with P=[0.4, 0.3, 0.2, 0.1]
    - Block-dropout on full path: keep_probs=[0.95, 0.9, 0.8, 0.7]
    - L_full = L_contrastive(full) + 0.5*L_margin(full) + 1.0*L_cls_leaf(full)
    - L_prefix = L_contrastive(prefix) + 0.5*L_margin(prefix) + 1.0*L_cls_top(prefix)
    - Total: L = L_full + 0.6*L_prefix
    """

    # Hyperparameters
    PREFIX_PROBS = [0.4, 0.3, 0.2, 0.1]  # P(j=1), P(j=2), P(j=3), P(j=4)
    BLOCK_KEEP_PROBS = [0.95, 0.9, 0.8, 0.7]  # Keep prob for blocks 0,1,2,3
    PREFIX_WEIGHT = 0.6  # Weight for L_prefix in total loss
    MARGIN_WEIGHT = 0.5  # Weight for margin loss
    CLASS_WEIGHT = 1.0   # Weight for classification loss

    def __init__(
        self,
        model: FractalModelV5,
        train_dataset,
        val_dataset,
        device: str = "cuda",
        # Stage 1 params
        stage1_epochs: int = 2,
        stage1_lr: float = 1e-4,
        # Stage 2 params
        stage2_epochs: int = 8,
        stage2_lr_head: float = 2e-5,
        stage2_lr_backbone: float = 1e-6,
        unfreeze_layers: int = 4,
        # Other
        temperature: float = 0.07,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

        self.stage1_epochs = stage1_epochs
        self.stage1_lr = stage1_lr
        self.stage2_epochs = stage2_epochs
        self.stage2_lr_head = stage2_lr_head
        self.stage2_lr_backbone = stage2_lr_backbone
        self.unfreeze_layers = unfreeze_layers
        self.temperature = temperature

        # Mixed precision scaler for Stage 2 (fix for NaN)
        self.scaler = GradScaler()

    def sample_prefix_length(self, batch_size: int) -> torch.Tensor:
        """Sample prefix lengths for batch per prefix distribution."""
        # P(j=1)=0.4, P(j=2)=0.3, P(j=3)=0.2, P(j=4)=0.1
        probs = torch.tensor(self.PREFIX_PROBS)
        lengths = torch.multinomial(probs.expand(batch_size, -1), num_samples=1).squeeze(-1)
        return lengths + 1  # Convert 0-indexed to 1-indexed (j ∈ {1,2,3,4})

    def create_block_dropout_mask(self, batch_size: int, prefix_lengths: torch.Tensor) -> torch.Tensor:
        """
        Create block dropout mask for prefix path.
        Zeros out blocks > prefix_length for each sample.
        """
        device = prefix_lengths.device
        mask = torch.zeros(batch_size, self.model.num_scales, device=device)

        for i in range(batch_size):
            j = prefix_lengths[i].item()
            mask[i, :j] = 1.0  # Keep first j blocks

        return mask

    def create_full_dropout_mask(self, batch_size: int) -> torch.Tensor:
        """
        Create block dropout mask for full path per design spec.
        keep_probs = [0.95, 0.9, 0.8, 0.7] for blocks 0,1,2,3
        """
        device = self.device
        mask = torch.ones(batch_size, self.model.num_scales, device=device)

        for block_idx, keep_prob in enumerate(self.BLOCK_KEEP_PROBS):
            # Randomly drop this block for some samples
            drop_mask = torch.rand(batch_size, device=device) > keep_prob
            mask[drop_mask, block_idx] = 0.0

        return mask

    def contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        logits = anchor @ positive.T / self.temperature
        targets = torch.arange(len(anchor), device=anchor.device)
        return F.cross_entropy(logits, targets)

    def margin_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.2
    ) -> torch.Tensor:
        """In-batch margin ranking loss."""
        batch_size = embeddings.shape[0]

        # Similarity matrix
        sims = embeddings @ embeddings.T

        # Same-class mask
        same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
        same_class.fill_diagonal_(False)

        # Different-class mask
        diff_class = ~same_class
        diff_class.fill_diagonal_(False)

        losses = []
        for i in range(batch_size):
            # Positive pairs (same class)
            pos_mask = same_class[i]
            if not pos_mask.any():
                continue
            pos_sims = sims[i][pos_mask]

            # Negative pairs (different class)
            neg_mask = diff_class[i]
            if not neg_mask.any():
                continue
            neg_sims = sims[i][neg_mask]

            # Margin loss: want pos_sim > neg_sim + margin
            # Loss = max(0, neg_sim - pos_sim + margin)
            for pos_sim in pos_sims:
                loss = F.relu(neg_sims - pos_sim + margin).mean()
                losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=embeddings.device)

    def train(self, batch_size: int = 32, patience: int = 5) -> List[Dict]:
        from torch.utils.data import DataLoader

        train_ds = ContrastiveDatasetV5(self.train_dataset.samples, self.model.tokenizer)

        def collate(batch):
            return collate_fn_v5(batch, self.model.tokenizer, self.device)

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
            epoch_loss, epoch_metrics = self._train_epoch(dataloader, optimizer, epoch, 1)

            val_score = self._evaluate()
            history.append({
                'stage': 1, 'epoch': epoch + 1,
                'loss': epoch_loss,
                **epoch_metrics,
                **val_score
            })

            print(f"  Stage 1 Epoch {epoch+1}: loss={epoch_loss:.4f}, "
                  f"L0={val_score['l0_accuracy']:.4f}, L1={val_score['l1_accuracy']:.4f}")

            if val_score['l0_accuracy'] + val_score['l1_accuracy'] > global_best_score:
                global_best_score = val_score['l0_accuracy'] + val_score['l1_accuracy']
                global_best_state = copy.deepcopy(self.model.state_dict())

        # ==================== STAGE 2: Head + Backbone ====================
        if self.stage2_epochs > 0:
            print(f"\n[Stage 2] Training HEAD + BACKBONE ({self.stage2_epochs} epochs)")
            self.model.unfreeze_last_n_layers(self.unfreeze_layers)

            head_params = list(self.model.fractal_head.parameters())
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]

            optimizer = torch.optim.AdamW([
                {'params': head_params, 'lr': self.stage2_lr_head},
                {'params': backbone_params, 'lr': self.stage2_lr_backbone},
            ], weight_decay=0.01)

            no_improve = 0

        for epoch in range(self.stage2_epochs):
            self.model.train()
            epoch_loss, epoch_metrics = self._train_epoch(dataloader, optimizer, epoch, 2)

            val_score = self._evaluate()
            history.append({
                'stage': 2, 'epoch': epoch + 1,
                'loss': epoch_loss,
                **epoch_metrics,
                **val_score
            })

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

    def _train_epoch(self, dataloader, optimizer, epoch: int, stage: int) -> Tuple[float, Dict]:
        total_loss = 0
        total_loss_full = 0
        total_loss_prefix = 0
        num_batches = 0

        # Use mixed precision for Stage 2 (fix for NaN)
        use_amp = (stage == 2)

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = batch['anchor_ids'].shape[0]

            # Sample prefix lengths per design spec
            prefix_lengths = self.sample_prefix_length(batch_size).to(self.device)

            # Mixed precision context
            with autocast(enabled=use_amp):
                # ===== FULL PATH =====
                # Block dropout for full path
                full_dropout_mask = self.create_full_dropout_mask(batch_size)

                anchor_full = self.model(
                    batch['anchor_ids'], batch['anchor_mask'],
                    block_dropout_mask=full_dropout_mask
                )
                l1_pos_full = self.model(batch['l1_pos_ids'], batch['l1_pos_mask'])

                full_emb = anchor_full['full_embedding']
                l1_pos_emb = l1_pos_full['full_embedding']

                # L_full components
                full_contrastive = self.contrastive_loss(full_emb, l1_pos_emb)
                full_margin = self.margin_loss(full_emb, batch['l1_labels'])
                full_cls = F.cross_entropy(
                    self.model.fractal_head.classify_leaf(full_emb),
                    batch['l1_labels']
                )

                loss_full = full_contrastive + self.MARGIN_WEIGHT * full_margin + self.CLASS_WEIGHT * full_cls

                # ===== PREFIX PATH =====
                # Get prefix embeddings (no dropout, just use first j blocks)
                prefix_dropout_mask = self.create_block_dropout_mask(batch_size, prefix_lengths)

                anchor_prefix = self.model(
                    batch['anchor_ids'], batch['anchor_mask'],
                    block_dropout_mask=prefix_dropout_mask
                )
                l0_pos_prefix = self.model(batch['l0_pos_ids'], batch['l0_pos_mask'])

                # Get prefix embeddings based on sampled lengths
                # For simplicity, we use the most common prefix length for the positive
                # (use same negatives for prefix and full)
                mode_prefix_len = prefix_lengths.mode().values.item()

                prefix_emb = self.model.fractal_head.get_prefix_embedding(
                    anchor_prefix['blocks'], mode_prefix_len
                )
                l0_pos_emb = l0_pos_prefix['full_embedding']  # Use full for positive (it's same class)

                # L_prefix components
                prefix_contrastive = self.contrastive_loss(prefix_emb, l0_pos_emb)
                prefix_margin = self.margin_loss(prefix_emb, batch['l0_labels'])
                prefix_cls = F.cross_entropy(
                    self.model.fractal_head.classify_top(prefix_emb),
                    batch['l0_labels']
                )

                loss_prefix = prefix_contrastive + self.MARGIN_WEIGHT * prefix_margin + self.CLASS_WEIGHT * prefix_cls

                # ===== TOTAL LOSS =====
                loss = loss_full + self.PREFIX_WEIGHT * loss_prefix

            # Skip if NaN
            if torch.isnan(loss):
                print(f"    Warning: NaN loss at batch {batch_idx}, skipping")
                continue

            # Use scaler for Stage 2, regular backward for Stage 1
            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_loss_full += loss_full.item()
            total_loss_prefix += loss_prefix.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {
            'loss_full': total_loss_full / max(num_batches, 1),
            'loss_prefix': total_loss_prefix / max(num_batches, 1),
        }
        return avg_loss, metrics

    def _evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        samples = self.val_dataset.samples[:min(1000, len(self.val_dataset.samples))]
        texts = [s.text for s in samples]
        l0_labels = np.array([s.level0_label for s in samples])
        l1_labels = np.array([s.level1_label for s in samples])

        # Full embedding evaluation
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

    def evaluate_prefix_accuracy(self) -> Dict:
        """
        Metric: evaluate accuracy at each prefix length.
        This tests if scale separation is actually being learned.
        """
        self.model.eval()
        samples = self.val_dataset.samples[:min(500, len(self.val_dataset.samples))]
        texts = [s.text for s in samples]
        l0_labels = np.array([s.level0_label for s in samples])
        l1_labels = np.array([s.level1_label for s in samples])

        results = {}

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

        for j in [1, 2, 3, 4]:
            prefix_len = j if j < 4 else None  # j=4 means full
            emb = self.model.encode(texts, batch_size=32, prefix_len=prefix_len).numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

            results[f'j{j}_l0'] = knn_accuracy(emb, l0_labels)
            results[f'j{j}_l1'] = knn_accuracy(emb, l1_labels)

        return results


class ContrastiveDatasetV5:
    """Dataset for V5 training with both L0 and L1 positives."""

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

        # L0 positive (same top-level category)
        l0_indices = self.l0_index[anchor.level0_label]
        l0_pos_idx = np.random.choice(l0_indices)
        while l0_pos_idx == idx and len(l0_indices) > 1:
            l0_pos_idx = np.random.choice(l0_indices)
        l0_pos = self.samples[l0_pos_idx]

        # L1 positive (same leaf category)
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


def collate_fn_v5(batch, tokenizer, device):
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


def run_v5_experiment(
    model_key: str = "bge-small",
    dataset_name: str = "yahoo",
    stage1_epochs: int = 5,
    stage2_epochs: int = 0,  # Skip Stage 2 (causes overfitting)
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 42,
):
    """Run V5 experiment with hierarchy-aligned prefix supervision."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from hierarchical_datasets import load_hierarchical_dataset

    print("=" * 70)
    print(f"FRACTAL V5: {model_key} on {dataset_name}")
    print("=" * 70)
    print("Features: Hierarchy-aligned progressive prefix supervision + block-dropout")
    print(f"Config: PREFIX_PROBS={V5Trainer.PREFIX_PROBS}, BLOCK_KEEP={V5Trainer.BLOCK_KEEP_PROBS}")

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

    def evaluate_knn(model, dataset, max_samples=2000, is_v5=False):
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

    # V5 Model
    print(f"\n[3] Training V5...")
    config = MODELS[model_key]

    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = V5Trainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        unfreeze_layers=4,
    )

    history = trainer.train(batch_size=batch_size, patience=5)

    # Final eval
    print("\n[4] Final evaluation on TEST set...")
    model.eval()
    v5_results = evaluate_knn(model, test_data, is_v5=True)
    print(f"  L0: {v5_results['l0_accuracy']:.4f}")
    print(f"  L1: {v5_results['l1_accuracy']:.4f}")

    # Evaluate at each prefix length
    print("\n[5] Prefix-length accuracy...")
    test_data_temp = TempDataset(test_data.samples, test_data.level0_names, test_data.level1_names)
    trainer.val_dataset = test_data_temp
    prefix_results = trainer.evaluate_prefix_accuracy()

    print("  Prefix Length | L0 Acc | L1 Acc")
    print("  " + "-" * 35)
    for j in [1, 2, 3, 4]:
        l0 = prefix_results[f'j{j}_l0']
        l1 = prefix_results[f'j{j}_l1']
        print(f"  j={j} ({j*64}d)    | {l0:.4f} | {l1:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    delta_l0 = v5_results['l0_accuracy'] - base_results['l0_accuracy']
    delta_l1 = v5_results['l1_accuracy'] - base_results['l1_accuracy']

    print(f"  {'Metric':<10} {'Baseline':<10} {'V5':<10} {'Delta':<10}")
    print(f"  {'-'*40}")
    print(f"  {'L0':<10} {base_results['l0_accuracy']:<10.4f} {v5_results['l0_accuracy']:<10.4f} {delta_l0:+10.4f}")
    print(f"  {'L1':<10} {base_results['l1_accuracy']:<10.4f} {v5_results['l1_accuracy']:<10.4f} {delta_l1:+10.4f}")

    success = delta_l0 > 0.05 or delta_l1 > 0.05
    print(f"\n  {'SUCCESS!' if success else 'NEEDS MORE WORK'}: L0={delta_l0:+.2%}, L1={delta_l1:+.2%}")

    # Save results
    results = {
        'model': model_key,
        'dataset': dataset_name,
        'baseline': base_results,
        'v5': v5_results,
        'delta': {'l0': delta_l0, 'l1': delta_l1},
        'prefix_accuracy': prefix_results,
        'history': history,
        'training_config': {
            'prefix_probs': V5Trainer.PREFIX_PROBS,
            'block_keep_probs': V5Trainer.BLOCK_KEEP_PROBS,
            'prefix_weight': V5Trainer.PREFIX_WEIGHT,
        }
    }

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = Path(__file__).parent.parent / "results" / f"v5_{model_key}_{dataset_name}.json"
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

    run_v5_experiment(
        model_key=args.model,
        dataset_name=args.dataset,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
    )
