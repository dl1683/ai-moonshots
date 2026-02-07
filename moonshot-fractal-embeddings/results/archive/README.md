# Archive: Ablation Experiment Results

This directory contains results from ablation experiments that proved inferior to the V5 approach.

These results are preserved for:
1. **Reproducibility** - Understanding what was tried
2. **Negative controls** - Proving that alternatives don't work
3. **Appendix material** - Supporting the claim that head-only training is best

## Files

| File | Experiment | Key Finding |
|------|-----------|-------------|
| `e2e_hierarchical_qwen3-0.6b.json` | End-to-end backbone fine-tuning | Worse than frozen baseline |
| `fractal_vq_qwen3-0.6b.json` | Vector quantization | Worse than baseline |
| `golden_hier_qwen3-0.6b.json` | Golden ratio loss weighting | No improvement |
| `hier_contrastive_qwen3-0.6b.json` | Tree-distance contrastive | Degraded performance |
| `hierarchical_contrastive_v2_qwen3-0.6b.json` | Projection learning | Marginal only |
| `label_matching_qwen3-0.6b.json` | Label-text matching | No improvement over frozen baseline |
| `recursive_fractal_qwen3-0.6b.json` | Recursive residual | No improvement |
| `fractal_comparison_qwen3-0.6b.json` | Early A/B testing | Superseded |
| `simple_hier_qwen3-0.6b.json` | Simple hierarchical baseline | Reference |
| `hierarchy_randomization_checkpoint.json` | Partial checkpoint | Superseded by full results |

## Current Results

For current validated results, see the parent `results/` directory:
- `v5_multiseed_qwen3-0.6b.json` - Primary V5 results (+5.36%/+6.47%)
- `hierarchy_randomization_fast.json` - Proves structure matters (+0.82% gap)
- `rigorous_scaling_qwen3-0.6b.json` - Depths 2-5 validation
- `newsgroups_benchmark_qwen3-0.6b.json` - Real-world validation (p=0.0232)
