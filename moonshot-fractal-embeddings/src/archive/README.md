# Archive: Deprecated Code

This directory contains deprecated implementations and ablation experiments that are **not recommended for use**.

These files are preserved for:
1. **Reproducibility** - Understanding the research progression
2. **Reference** - Seeing what approaches were tried and failed
3. **Appendix material** - Supporting negative results in publications

## Directory Structure

### `versions/` - Deprecated Implementations

| File | Version | Why Deprecated |
|------|---------|----------------|
| `fractal_embeddings.py` | V1 | Basic multi-scale, no hierarchy awareness |
| `fractal_embeddings_v2.py` | V2 | Original hierarchy-aware, superseded by V5 |
| `fractal_v3.py` | V3 | LoRA + gated fusion - overly complex, no improvement |
| `fractal_v4.py` | V4 | Two-stage PEFT - backbone fine-tuning causes overfitting |

**Current version**: Use `fractal_v5.py` in the parent `src/` directory.

### `ablations/` - Failed Experiments (Negative Controls)

These experiments proved that **head-only training is superior** to alternative approaches:

| File | Approach | Finding |
|------|----------|---------|
| `train_e2e_hierarchical.py` | End-to-end backbone fine-tuning | Overfitting, NaN gradients |
| `train_golden_hierarchical.py` | Golden ratio loss weighting | No improvement |
| `train_hierarchical_contrastive.py` | Tree-distance contrastive | Overfitting |
| `train_hierarchical_contrastive_v2.py` | Projection learning | Marginal improvement only |
| `train_label_matching.py` | Label-text semantic matching | Not viable |
| `train_hfh.py` | Hyperbolic fractal head | Unvalidated |
| `fractal_vq.py` | Vector quantization | Worse than baseline |
| `recursive_fractal.py` | Recursive residual conditioning | No improvement |
| `fractal_comparison.py` | Early A/B testing framework | Superseded |
| `debug_e2e.py` | Debug script | Development only |

## Key Learnings

1. **Backbone fine-tuning fails** on small hierarchical datasets (overfitting)
2. **Head-only training** with frozen backbone is more robust
3. **Complex architectures** (LoRA, PEFT, VQ) don't help when the backbone is good
4. **Structure matters** - but only correct structure (see hierarchy randomization experiment)

## Do NOT Use These Files

For new experiments, use only the files in the parent `src/` directory:
- `fractal_v5.py` - Current implementation
- `multi_model_pipeline.py` - Model abstraction
- `hierarchical_datasets.py` - Data loaders
- `evaluation.py` - Metrics
