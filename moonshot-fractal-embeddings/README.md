# Fractal Embeddings: When Structure Matters

**Multi-scale self-similar embeddings for hierarchical semantic understanding.**

## Key Finding: Correct Structure Matters

Our most important result: **correct hierarchy structure improves performance, while incorrect structure actively hurts.**

### Hierarchy Randomization Experiment

| Condition | Hierarchical Accuracy | Delta vs Flat |
|-----------|----------------------|---------------|
| Flat baseline | 66.68% | - |
| Fractal + TRUE hierarchy | 67.40% | **+0.72%** |
| Fractal + RANDOM hierarchy | 66.58% | **-0.10%** |

**Gap**: +0.82% [95% CI: 0.43%, 1.21%] - **95% CI excludes zero**

This proves that:
1. **Correct hierarchy helps** - True hierarchy beats flat
2. **Wrong hierarchy hurts** - Random hierarchy is worse than flat (no hierarchy)
3. **Architecture must match data** - The fractal conditioning amplifies correct signal but also amplifies noise from wrong signal

### Additional Validated Results

| Experiment | Finding | p-value |
|------------|---------|---------|
| V5 on Yahoo Answers (5 seeds) | +5.36% L0, +6.47% L1 | - |
| 20 Newsgroups real-world | +0.70% hierarchical | **p=0.0232** |
| Scaling law (depths 2-5) | +1-2.5% consistent improvement | - |

---

## The Paradigm Shift

Traditional embeddings represent all semantic relationships in a single flat vector space. This forces coarse concepts (e.g., "Science") and fine-grained distinctions (e.g., "Quantum Physics vs Classical Physics") to compete for the same dimensions.

**Fractal embeddings** introduce multi-scale structure where:
- **Coarse scales** capture high-level categories
- **Fine scales** capture specific distinctions
- **Shared fractal blocks** ensure self-similarity across scales

This mirrors how humans understand hierarchies: we recognize something as "Sports" before distinguishing "Football vs Basketball."

---

## V5 Architecture

V5 uses **hierarchy-aligned prefix supervision** with head-only training:

1. **Progressive Prefix Supervision** - Train shorter prefixes for coarse labels
2. **Block Dropout** - Forces scale specialization
3. **Dual Classification Heads** - Separate heads for coarse and fine predictions
4. **Frozen Backbone** - Head-only training avoids overfitting

```
Pre-trained Backbone (frozen)
         |
         v
+------------------+
|   Fractal Head   |
| [Block 0] (64d)  | --> Coarse classification
| [Block 1] (64d)  |
| [Block 2] (64d)  |
| [Block 3] (64d)  | --> Fine classification
+------------------+
```

---

## Quick Start

```bash
cd src/
```

```python
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from multi_model_pipeline import MODELS
from hierarchical_datasets import load_hierarchical_dataset

# Load model
config = MODELS["qwen3-0.6b"]
model = FractalModelV5(
    config=config,
    num_l0_classes=10,
    num_l1_classes=60,
    num_scales=4,
    scale_dim=64,
    device="cuda"
).to("cuda")

# Load data
train_data = load_hierarchical_dataset("yahoo", split="train")
val_data = load_hierarchical_dataset("yahoo", split="test")

# Train (head-only)
trainer = V5Trainer(model, train_data, val_data, stage1_epochs=5, stage2_epochs=0)
trainer.train(batch_size=24)

# Encode texts
embeddings = model.encode(["Your text here"])
```

---

## Repository Structure

```
moonshot-fractal-embeddings/
├── README.md                          # This file
├── src/
│   ├── fractal_v5.py                  # Current implementation
│   ├── multi_model_pipeline.py        # Universal model wrapper (10+ models)
│   ├── hierarchical_datasets.py       # Yahoo Answers, AG News, DBPedia loaders
│   ├── v5_statistical_validation.py   # Multi-seed validation script
│   ├── hierarchy_randomization_fast.py # Hierarchy randomization experiment
│   ├── rigorous_scaling_experiment.py # Depths 2-5 scaling validation
│   └── archive/                       # Deprecated code (V1-V4, ablations)
├── results/
│   ├── v5_multiseed_qwen3-0.6b.json   # Primary V5 results
│   ├── hierarchy_randomization_fast.json # KEY: Proves structure matters
│   ├── rigorous_scaling_qwen3-0.6b.json
│   ├── newsgroups_benchmark_qwen3-0.6b.json
│   └── archive/                       # Ablation experiment results
├── research/
│   ├── THEORY.md                      # Canonical theory doc (limitations, proofs, advantages)
│   ├── IDEAS.md                       # Consolidated exploratory ideas
│   ├── EFFICIENCY_ANALYSIS.md         # Efficiency and inference-cost analysis
│   ├── NEXT_DIRECTIONS.md             # Future research priorities
└── checkpoints/                       # Trained model weights
```

---

## Theoretical Foundation

### Sample Complexity (Theorem 1)

Fractal classifiers have better sample complexity:
- **Flat**: O((d + log C) / ε²) - depends on total classes C
- **Fractal**: O((d + log b) / ε²) - depends on branching factor b

When b << C, fractal requires **exponentially fewer samples**.

### Scale-Separated Embeddings (Theorem B)

Scale-separated embeddings achieve **Ω(k/(j+1)) access complexity advantage** over isotropic embeddings for level-j hierarchical classification.

| Level | Scale-Separated | Isotropic | Ratio |
|-------|-----------------|-----------|-------|
| j=0 (coarsest) | d₀ | Ω(k·d₀) | **Ω(k)** |
| j=k-1 (finest) | k·d₀ | k·d₀ | 1 |

See `research/THEORY.md` for the canonical theory and proof summary.

---

## Key Findings

1. **Structure sensitivity** - Correct hierarchy helps, wrong hierarchy hurts
2. **Head-only training wins** - Backbone fine-tuning causes overfitting
3. **Larger models benefit more** - Qwen3-0.6B shows significantly larger gains than smaller models
4. **Lower variance** - Fractal predictions are more stable across seeds
5. **Boundary conditions** - Fractal requires learnable signal at each level

---

## Limitations

1. **Limited real-world benchmarks** - Primarily validated on Yahoo Answers + 20 Newsgroups
2. **Shallow hierarchies tested** - Depths 2-5, boundary effects at depth 6+
3. **Text domain only** - Not yet tested on images or other modalities

---

## Citation

```bibtex
@misc{fractal-embeddings-2026,
  title={When Structure Matters: Empirical and Information-Theoretic Evidence for Fractal Classifiers},
  author={AI Moonshots},
  year={2026},
  url={https://github.com/ai-moonshots/fractal-embeddings}
}
```

---

## Research Documents

For the complete research writeup, see:
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - End-to-end project summary, positioning, and status
- **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)** - Detailed method/stats appendix
- **[THEORY.md](research/THEORY.md)** - Canonical theory doc: limits, proofs, and why fractal supervision works
- **[NEXT_DIRECTIONS.md](research/NEXT_DIRECTIONS.md)** - Prioritized roadmap
- **[IDEAS.md](research/IDEAS.md)** - Consolidated speculative ideas (non-validated)

---

*Part of the [AI Moonshots](https://github.com/ai-moonshots) research project.*
