# AI Moonshots - Model Directory

## Hardware Profile

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5090 Laptop |
| VRAM | ~24GB |
| Effective Capacity | ~13B full precision, ~70B+ quantized |

## Quantization Strategy

| Model Size | Strategy |
|------------|----------|
| **<1B parameters** | Full precision (FP16/BF16) |
| **1B - 7B parameters** | Q8 or Q6_K (light quantization) |
| **7B - 30B parameters** | Q4_K_M or Q5_K_M |
| **30B+ parameters** | Q4_K_S or Q3_K aggressive |

---

## 1. TRANSFORMER MODELS (Standard Attention)

### Sub-1B (Full Precision)

| Model | Params | Context | Strengths | HF Link |
|-------|--------|---------|-----------|---------|
| **Qwen3-0.6B** | 600M | 32K | Best sub-1B, multilingual (100+ langs), agent-friendly | `Qwen/Qwen3-0.6B` |
| **Qwen2-0.5B** | 500M | 32K | Ultra-lightweight baseline | `Qwen/Qwen2-0.5B` |
| **SmolLM2-360M** | 360M | 8K | HuggingFace's tiny model | `HuggingFaceTB/SmolLM2-360M` |
| **Pythia-410M** | 410M | 2K | EleutherAI, great for research/interpretability | `EleutherAI/pythia-410m` |
| **Pythia-160M** | 160M | 2K | Smallest Pythia, fast experiments | `EleutherAI/pythia-160m` |

### 1B - 7B (Light Quantization Q6-Q8)

| Model | Params | Context | Strengths | Quantized VRAM |
|-------|--------|---------|-----------|----------------|
| **Qwen3-1.7B** | 1.7B | 32K | Strong reasoning for size | ~2GB Q8 |
| **Qwen3-4B** | 4B | 32K | Sweet spot for experiments | ~4GB Q8 |
| **Llama-3.2-3B** | 3B | 128K | Long context, Meta quality | ~3GB Q8 |
| **Gemma-3-4B** | 4B | 128K | Google, pan & scan vision | ~4GB Q8 |
| **Phi-4** | 3.8B | 16K | Microsoft, strong reasoning | ~4GB Q8 |

### 7B - 30B (Medium Quantization Q4-Q5)

| Model | Params | Context | Strengths | Quantized VRAM |
|-------|--------|---------|-----------|----------------|
| **Qwen3-8B** | 8B | 32K | Excellent all-rounder | ~6GB Q4_K_M |
| **Llama-3.1-8B** | 8B | 128K | Industry standard | ~6GB Q4_K_M |
| **Mistral-7B-v0.3** | 7B | 32K | Fast, efficient | ~5GB Q4_K_M |
| **Gemma-3-12B** | 12B | 128K | Multilingual, Google | ~8GB Q4_K_M |
| **DeepSeek-V2-Lite** | 16B | 128K | MoE, only 2.4B active | ~10GB Q4_K_M |
| **Qwen3-14B** | 14B | 32K | Strong mid-size | ~10GB Q4_K_M |
| **Gemma-3-27B** | 27B | 128K | Near-frontier quality | ~18GB Q4_K_M |

### 30B+ (Aggressive Quantization Q3-Q4)

| Model | Total/Active Params | Context | Strengths | Quantized VRAM |
|-------|---------------------|---------|-----------|----------------|
| **Qwen3-32B** | 32B | 32K | Frontier-class open | ~22GB Q4_K_S |
| **DeepSeek-V3** | 671B/37B active | 128K | MoE, SOTA open | ~24GB Q3_K (fits tight) |
| **Llama-4-Scout** | 109B/17B active | 10M | Insane context window | ~20GB Q4 |
| **Kimi-K2** | 1T/32B active | 128K | SOTA agentic | ~22GB Q3_K |

---

## 2. STATE SPACE MODELS (SSM/Mamba)

Pure SSM architecture - linear scaling with sequence length, 5x faster inference than transformers.

### Sub-1B (Full Precision)

| Model | Params | Strengths | HF Link |
|-------|--------|-----------|---------|
| **Mamba-130M** | 130M | Smallest pure Mamba | `state-spaces/mamba-130m` |
| **Mamba-370M** | 370M | Fast experiments | `state-spaces/mamba-370m` |
| **Mamba-790M** | 790M | Good balance | `state-spaces/mamba-790m` |
| **Mamba2-130M** | 130M | Mamba2 architecture | `state-spaces/mamba2-130m` |
| **Mamba2-370M** | 370M | Mamba2, improved | `state-spaces/mamba2-370m` |
| **Mamba2-780M** | 780M | Mamba2, sweet spot | `state-spaces/mamba2-780m` |

### 1B+ (Quantized)

| Model | Params | Strengths | Quantized VRAM |
|-------|--------|-----------|----------------|
| **Mamba-1.4B** | 1.4B | Pure Mamba, Pile trained | ~1.5GB Q8 |
| **Mamba-2.8B** | 2.8B | Largest pure Mamba | ~2.5GB Q8 |
| **Mamba2-1.3B** | 1.3B | Mamba2, better | ~1.5GB Q8 |
| **Mamba2-2.7B** | 2.7B | Mamba2, SOTA SSM | ~2.5GB Q8 |
| **Codestral-Mamba-7B** | 7B | Mistral, code-focused | ~5GB Q4_K_M |

---

## 3. HYBRID MODELS (Transformer + SSM)

Best of both worlds - attention for complex reasoning, SSM for efficiency.

### IBM Granite 4.0 Family (Apache 2.0)

| Model | Params | Architecture | Strengths | VRAM |
|-------|--------|--------------|-----------|------|
| **Granite-4.0-Micro** | ~350M | Hybrid | Tiny, browser-runnable | ~400MB FP16 |
| **Granite-4.0-Micro-H** | ~350M | Hybrid (Mamba2) | SSM variant | ~400MB FP16 |
| **Granite-4.0-Tiny** | ~1B | Hybrid | Small but capable | ~1GB FP16 |
| **Granite-4.0-1B** | ~1.5B | Transformer | Nano family | ~1.5GB FP16 |
| **Granite-4.0-H-1B** | ~1.5B | Hybrid | Mamba2 layers, 70-80% less memory | ~1.2GB FP16 |
| **Granite-4.0-Small** | 3B | Hybrid | General purpose | ~3GB Q8 |
| **Granite-4.0-H-Small** | 32B/9B active | MoE Hybrid | SOTA efficiency | ~8GB Q4_K_M |

### AI21 Jamba Family

| Model | Params | Architecture | Strengths | VRAM |
|-------|--------|--------------|-----------|------|
| **Jamba-v0.1** | 52B/12B active | Transformer+Mamba+MoE | First major hybrid | ~10GB Q4 |
| **Jamba-1.5-Mini** | 52B/12B active | Improved Jamba | Better efficiency | ~10GB Q4 |
| **Jamba-1.5-Large** | 398B/94B active | Large hybrid | Near-frontier | ~24GB Q3 (tight) |

### Falcon-H1 (TII)

| Model | Params | Architecture | Strengths | VRAM |
|-------|--------|--------------|-----------|------|
| **Falcon-H1-0.5B** | 500M | Transformer+Mamba | Tiny hybrid | ~600MB FP16 |
| **Falcon-H1-1.5B** | 1.5B | Transformer+Mamba | Small hybrid | ~1.5GB FP16 |
| **Falcon-H1-3B** | 3B | Transformer+Mamba | Mid hybrid | ~3GB Q8 |
| **Falcon-H1-7B** | 7B | Transformer+Mamba | Efficient | ~5GB Q4_K_M |
| **Falcon-H1-34B** | 34B | Transformer+Mamba | Large hybrid | ~22GB Q4_K_S |

---

## 4. LIQUID AI MODELS (Novel Architecture)

Liquid Foundation Models - hybrid architecture optimized for on-device deployment.

| Model | Params | Strengths | VRAM |
|-------|--------|-----------|------|
| **LFM2.5-1.2B-Base** | 1.2B | 28T token pretrain | ~1.2GB FP16 |
| **LFM2.5-1.2B-Instruct** | 1.2B | SOTA 1B-class, beats Llama-3.2-1B | ~1.2GB FP16 |
| **LFM2.5-1.2B-Vision** | 1.2B | VLM variant | ~1.5GB FP16 |
| **LFM2.5-1.2B-Audio** | 1.2B | Audio understanding | ~1.5GB FP16 |
| **LFM2-2.6B-Exp** | 2.6B | Pure RL trained, dynamic reasoning | ~2.5GB Q8 |

---

## 5. VISION-LANGUAGE MODELS (VLM)

### Sub-1B / Small (Full Precision)

| Model | Params | Strengths | VRAM |
|-------|--------|-----------|------|
| **DeepSeek-VL-1.3B** | 1.3B | Smallest strong VLM | ~1.5GB FP16 |
| **Phi-4-Vision** | 3.8B | Microsoft, efficient | ~4GB FP16 |
| **LFM2.5-1.2B-Vision** | 1.2B | Liquid AI | ~1.5GB FP16 |

### 7B+ (Quantized)

| Model | Params | Strengths | VRAM |
|-------|--------|-----------|------|
| **Qwen2.5-VL-7B** | 7B | Video input, 29 languages, object localization | ~5GB Q4_K_M |
| **Llama-3.2-Vision-11B** | 11B | Strong OCR, 128K context | ~8GB Q4_K_M |
| **Gemma-3-4B-IT** | 4B | Pan & scan, Google | ~4GB Q8 |
| **Kimi-VL-A3B-Thinking** | 16B/2.8B active | Long video, PDFs, MoE | ~4GB Q4 |
| **GLM-4.6V** | varies | Tool use, visual agents, 128K | ~6GB Q4_K_M |
| **Qwen3-VL-30B-A3B** | 30B/3B active | MoE, rivals GPT-5 | ~5GB Q4 |

---

## 6. DIFFUSION MODELS (Image Generation)

### FLUX Family (Black Forest Labs)

| Model | Params | Strengths | VRAM |
|-------|--------|-----------|------|
| **FLUX.1-schnell** | ~12B | Fast (0.5-1s), good quality | ~12GB FP16, ~8GB Q8 |
| **FLUX.1-dev** | ~12B | Higher quality, slower | ~12GB FP16 |
| **FLUX.2** | ~12B | Multi-reference (up to 10 images) | ~12GB FP16 |

### Stable Diffusion Family

| Model | Params | Strengths | VRAM |
|-------|--------|-----------|------|
| **SD 3.5-Medium** | 2B | Balanced, accurate text | ~4GB FP16 |
| **SD 3.5-Large** | 8B | High quality | ~10GB FP16 |
| **SDXL-Lightning** | ~2.5B | <1s generation (2-4 steps) | ~6GB FP16 |
| **SDXL-Turbo** | ~2.5B | Single-step generation | ~6GB FP16 |

### Other Notable

| Model | Params | Strengths | VRAM |
|-------|--------|-----------|------|
| **HunyuanImage-3.0** | varies | Tencent, near closed-model quality | ~12GB |
| **Juggernaut XL v10** | ~2.5B | Cinematic, photorealistic | ~6GB FP16 |

---

## 7. EMBEDDING MODELS

### Traditional Encoder Models (Sub-1B)

| Model | Params | Dims | Strengths | VRAM | HF Link |
|-------|--------|------|-----------|------|---------|
| **EmbeddingGemma-300M** | 308M | 768 | SOTA <500M, 100+ langs, bi-directional Gemma3 | ~400MB | `google/embedding-gemma-308m` |
| **bge-small-en-v1.5** | 33M | 384 | Fast, good quality | ~70MB | `BAAI/bge-small-en-v1.5` |
| **bge-base-en-v1.5** | 109M | 768 | Balanced | ~220MB | `BAAI/bge-base-en-v1.5` |
| **bge-large-en-v1.5** | 335M | 1024 | Best BGE English | ~700MB | `BAAI/bge-large-en-v1.5` |
| **e5-small-v2** | 33M | 384 | 16ms latency, fastest | ~70MB | `intfloat/e5-small-v2` |
| **e5-base-v2** | 109M | 768 | Balanced | ~220MB | `intfloat/e5-base-v2` |
| **e5-large-v2** | 335M | 1024 | Best E5 English | ~700MB | `intfloat/e5-large-v2` |
| **nomic-embed-text-v1.5** | 137M | 768 | Matryoshka, flexible dims | ~300MB | `nomic-ai/nomic-embed-text-v1.5` |
| **bge-m3** | 568M | 1024 | Trilingual, dense+sparse | ~700MB | `BAAI/bge-m3` |

### LLM-Based Embedding Models (1B-8B) - SOTA Performance

These models are decoder LLMs converted to encoders with bi-directional attention, achieving SOTA on MTEB.

| Model | Params | Dims | MTEB Score | Strengths | VRAM |
|-------|--------|------|------------|-----------|------|
| **Qwen3-Embedding-0.6B** | 600M | 32-1024 | 65.5 | Best <1B, 100+ langs, instruction-aware | ~700MB |
| **Qwen3-Embedding-4B** | 4B | 32-1024 | 68.2 | High quality, multilingual | ~4GB Q8 |
| **Qwen3-Embedding-8B** | 8B | 32-1024 | **70.58** | **#1 MTEB multilingual (Jun 2025)** | ~6GB Q4 |
| **GTE-Qwen2-1.5B-instruct** | 1.5B | 4096 | 67.20 | Alibaba, long context | ~1.5GB |
| **GTE-Qwen2-7B-instruct** | 7.6B | 4096 | 70.72 | Near SOTA, long context | ~6GB Q4 |
| **stella_en_1.5B_v5** | 1.5B | 4096 | 69.43 | English-only, efficient | ~1.5GB |
| **llama-embed-nemotron-8b** | 8B | 4096 | **#1 MTEB multi (Oct 2025)** | NVIDIA, cross-lingual | ~6GB Q4 |
| **llama-nemotron-embed-1b-v2** | 1B | 4096 | 66.8 | NVIDIA, commercial use OK | ~1GB |
| **e5-mistral-7b-instruct** | 7B | 4096 | 66.6 | First LLM-based, 4K context | ~5GB Q4 |
| **SFR-Embedding-Mistral** | 7B | 4096 | 67.6 | Salesforce, top retrieval | ~5GB Q4 |
| **GritLM-7B** | 7.2B | 4096 | 67.07 | Best retrieval on multilingual | ~5GB Q4 |
| **multilingual-e5-large-instruct** | 560M | 1024 | 65.53 | Best efficiency/performance ratio | ~600MB |
| **nomic-embed-text-v2** | ~1B | 768 | 66.1 | First MoE embedding model | ~1GB |

### Pure Vision Models (for CTI cross-modal LOAO test)

| Model | Params | Dims | Architecture | VRAM | HF ID |
|-------|--------|------|--------------|------|-------|
| **ViT-Base-Patch16** | 86M | 768 | 12-layer ViT, ImageNet-21k | ~200MB | `google/vit-base-patch16-224` |
| **ViT-Large-Patch16** | 307M | 1024 | 24-layer ViT, ImageNet-21k | ~700MB | `google/vit-large-patch16-224` |
| **DINOv2-ViT-S/14** | 21M | 384 | Self-supervised, excellent features | ~80MB | `facebook/dinov2-small` |
| **DINOv2-ViT-B/14** | 86M | 768 | Self-supervised, SOTA linear probe | ~200MB | `facebook/dinov2-base` |

### Multimodal Embedding Models

| Model | Params | Modalities | Strengths | VRAM |
|-------|--------|------------|-----------|------|
| **Jina-embeddings-v4** | 3B | Text+Image | Built on Qwen2.5-VL, charts/tables | ~3GB Q8 |
| **llama-nemotron-embed-vl-1b-v2** | 1B | Text+Image | NVIDIA, commercial use | ~1GB |
| **Omni-Embed-Nemotron** | 3B | Text+Image+Audio+Video | Built on Qwen-Omni | ~3GB Q8 |
| **nomic-embed-vision-v1.5** | 137M | Text+Image | Lightweight multimodal | ~300MB |

### Code Embedding Models

| Model | Params | Strengths | VRAM |
|-------|--------|-----------|------|
| **nomic-embed-code** | ~137M | SOTA code retrieval | ~300MB |
| **CodeSage-Large** | 1.3B | Multi-language code | ~1.5GB |
| **StarCoder-Embed** | varies | Code + docstrings | varies |

### Key Insight: LLM-Based vs Traditional

| Aspect | Traditional (BERT-based) | LLM-Based (7B+) |
|--------|-------------------------|-----------------|
| MTEB Score | 60-66 | 66-71 |
| Latency | 5-20ms | 50-200ms |
| Context | 512 tokens | 4096-8192 tokens |
| VRAM | <1GB | 5-8GB (Q4) |
| Best For | Low-latency, edge | Maximum quality |

**Fractal Embeddings Compatibility:**
- ✅ Traditional encoders: All tested, work well
- ✅ Qwen3-Embedding: Tested, BEST compatibility
- ⚠️ LLM-based (7B+): Untested, potential for even larger gains due to richer representations
- ⚠️ Multimodal: Untested, interesting future direction

---

## 8. AUDIO/SPEECH MODELS

| Model | Params | Type | Strengths | VRAM |
|-------|--------|------|-----------|------|
| **Whisper-large-v3** | 1.5B | ASR | OpenAI, SOTA transcription | ~3GB |
| **Whisper-medium** | 769M | ASR | Balanced | ~1.5GB |
| **Whisper-small** | 244M | ASR | Fast | ~500MB |
| **LFM2.5-1.2B-Audio** | 1.2B | Audio LM | Liquid AI | ~1.5GB |
| **Qwen2-Audio-7B** | 7B | Audio LM | Audio understanding | ~5GB Q4 |

---

## Model Selection by Experiment Type

### For Activation Analysis / Interpretability
- **Pythia series** (160M-2.8B) - Designed for interpretability research
- **Mamba series** - Simpler recurrent structure
- **Granite-4.0-Micro/Tiny** - Small enough to trace everything

### For Architecture Comparison
- Run same task on:
  - Pure Transformer: Qwen3-0.6B, Llama-3.2-3B
  - Pure SSM: Mamba-790M, Mamba2-2.7B
  - Hybrid: Granite-4.0-H-1B, Falcon-H1-1.5B, Jamba

### For Scaling Law Experiments
- **Pythia**: 160M → 410M → 1B → 1.4B → 2.8B
- **Mamba**: 130M → 370M → 790M → 1.4B → 2.8B
- **Qwen3**: 0.6B → 1.7B → 4B → 8B → 14B → 32B

### For Frontier Capability Testing
- **DeepSeek-V3** (Q3 quantized) - Current SOTA open
- **Kimi-K2** (Q3 quantized) - SOTA agentic
- **Qwen3-32B** (Q4 quantized) - Strong reasoning

---

## Quick Reference: What Fits in 24GB

| Precision | Max Model Size |
|-----------|---------------|
| FP32 | ~6B |
| FP16/BF16 | ~12B |
| Q8 | ~24B |
| Q6_K | ~30B |
| Q5_K_M | ~40B |
| Q4_K_M | ~50B |
| Q4_K_S | ~60B |
| Q3_K | ~70B+ |
| MoE (active params matter) | 600B+ total if <25B active |

---

## Sources

- [BentoML: Best Open-Source LLMs 2026](https://www.bentoml.com/blog/navigating-the-world-of-open-source-large-language-models)
- [DataCamp: Top Open-Source LLMs](https://www.datacamp.com/blog/top-open-source-llms)
- [IBM Granite 4.0 Announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- [Liquid AI LFM2.5 Blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- [Mamba GitHub](https://github.com/state-spaces/mamba)
- [HuggingFace VLMs 2025](https://huggingface.co/blog/vlms-2025)
- [BentoML: Image Generation Models](https://www.bentoml.com/blog/a-guide-to-open-source-image-generation-models)
- [BentoML: Embedding Models](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)

---

*Last Updated: January 2026*
