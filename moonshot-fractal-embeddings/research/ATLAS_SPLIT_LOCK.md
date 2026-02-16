# Atlas Experiment: Pre-Registered Splits

## Locked BEFORE any analysis (2026-02-15)

### Fit Models (14)
- bge-small, bge-base, bge-large (CLS-pooled encoders)
- e5-small, e5-base, e5-large (mean-pooled encoders)
- minilm, mpnet, nomic, embedding-gemma (diverse encoders)
- multilingual-e5-large (multilingual encoder)
- pythia-410m (decoder)
- gte-qwen2-1.5b, stella-1.5b (LLM-based encoders)

### Holdout Models (4) - DO NOT ANALYZE UNTIL FIT MODELS DONE
- bge-m3 (trilingual encoder)
- nomic-v2 (MoE encoder)
- qwen3-0.6b (decoder-to-encoder)
- gte-large (large encoder)

### Fit Datasets (8)
clinc, dbpedia_classes, trec, yahoo, 20newsgroups, goemotions, arxiv, wos

### Holdout Datasets (3) - DO NOT ANALYZE UNTIL FIT DATASETS DONE
agnews, amazon, dbpedia

### Pre-Registered Predictions (to be filed after fit analysis)
1. Best functional form per architecture class
2. Predicted universality class for holdout models
3. Predicted peak layer for holdout models
4. Predicted curve shape (parameters) for holdout models
