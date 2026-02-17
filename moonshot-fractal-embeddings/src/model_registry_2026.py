"""
Model Registry 2026 - Cutting Edge Only
Updated: February 2026
Policy: Only use models from 2025-2026. No legacy models.
"""

# TIER 1 - Fast experiments (all < 3B)
TIER1_2026 = [
    # Transformers
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "google/gemma-3-1b-it",
    "HuggingFaceTB/SmolLM3-3B",
    "ibm-granite/granite-4.0-350m",
    "ibm-granite/granite-4.0-1b",

    # SSM (Mamba - using HF format for transformers compatibility)
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b-hf",

    # Hybrid
    "tiiuae/Falcon-H1-0.5B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Instruct",
    "ibm-granite/granite-4.0-h-350m",
    "ibm-granite/granite-4.0-h-1b",
    "Zyphra/Zamba2-1.2B",                       # NEW: Hybrid Mamba2 + shared transformer

    # Liquid AI (novel LIV convolution + GQA hybrid architecture)
    "LiquidAI/LFM2-350M-Exp",                   # NEW: Liquid Foundation Model, 350M
    "LiquidAI/LFM2-1.2B-Exp",                   # NEW: Liquid Foundation Model, 1.2B
    "LiquidAI/LFM2-2.6B-Exp",                   # NEW: Liquid Foundation Model, 2.6B

    # RWKV
    "RWKV/RWKV7-Goose-World3-1.5B-HF",

    # Reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

# TIER 2 - Standard experiments (3B-8B)
TIER2_2026 = [
    # Transformers
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "google/gemma-3-4b-it",
    "microsoft/Phi-4-mini-instruct",
    "allenai/Olmo-3-7B",
    "ibm-granite/granite-4.0-micro",
    "ibm-granite/granite-4.0-tiny-preview",

    # SSM
    "tiiuae/falcon-mamba-7b",                    # NEW: Pure Mamba SSM, 7B, trained on 5.8T tokens

    # Hybrid
    "tiiuae/Falcon-H1-3B-Instruct",
    "tiiuae/Falcon-H1-7B-Instruct",
    "nvidia/Nemotron-H-4B-Instruct-128K",
    "Zyphra/Zamba2-2.7B",
    "Zyphra/Zamba2-7B",                          # NEW: Hybrid Mamba2 + shared transformer, 7B
    "ibm-granite/granite-4.0-h-tiny",

    # xLSTM (novel extended LSTM architecture)
    "NX-AI/xLSTM-7b",                            # NEW: xLSTM with exponential gating + matrix memory

    # RWKV
    "RWKV-Red-Team/ARWKV-R1-7B",

    # Reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "microsoft/Phi-4-mini-reasoning",             # NEW: 3.8B reasoning model, MIT license

    # Diffusion
    "ML-GSAI/LLaDA-8B-Base",
]

# TIER 3 - Validation (8B+, quantize aggressively)
TIER3_2026 = [
    # Transformers
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "google/gemma-3-12b-it",                     # NEW
    "google/gemma-3-27b-it",                     # NEW
    "openai/gpt-oss-20b",
    "allenai/Olmo-3-1125-32B",

    # Hybrid
    "tiiuae/Falcon-H1-34B-Instruct",
    "ibm-granite/granite-4.0-h-small",

    # Reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "microsoft/Phi-4-reasoning",                  # NEW: 14B reasoning, approaches full DeepSeek-R1

    # Diffusion
    "inclusionAI/LLaDA2.0-mini",
]

# Quick test set for rapid iteration
QUICK_TEST = [
    "Qwen/Qwen3-0.6B",                           # Transformer
    "state-spaces/mamba-370m-hf",                  # SSM (Mamba 1 HF format)
    "tiiuae/Falcon-H1-0.5B-Instruct",             # Hybrid (Mamba+Transformer)
    "LiquidAI/LFM2-1.2B-Exp",                     # Liquid (LIV convolution+GQA)
    "RWKV/RWKV7-Goose-World3-1.5B-HF",            # RWKV
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Reasoning
]

# Paradigm groups
PARADIGMS = {
    "transformer": [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "google/gemma-3-1b-it",
        "ibm-granite/granite-4.0-1b",
    ],
    "ssm": [
        "state-spaces/mamba-370m-hf",
        "state-spaces/mamba-790m-hf",
        "state-spaces/mamba-1.4b-hf",
        "tiiuae/falcon-mamba-7b",          # Pure Mamba, 7B
    ],
    "hybrid": [
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "tiiuae/Falcon-H1-1.5B-Instruct",
        "ibm-granite/granite-4.0-h-1b",
        "Zyphra/Zamba2-1.2B",              # Mamba2 + shared transformer
        "Zyphra/Zamba2-2.7B",
    ],
    "liquid": [
        "LiquidAI/LFM2-350M-Exp",         # Novel LIV convolution + GQA hybrid
        "LiquidAI/LFM2-1.2B-Exp",
        "LiquidAI/LFM2-2.6B-Exp",
    ],
    "xlstm": [
        "NX-AI/xLSTM-7b",                 # Extended LSTM with exponential gating
    ],
    "rwkv": [
        "RWKV/RWKV7-Goose-World3-1.5B-HF",
    ],
    "reasoning": [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "microsoft/Phi-4-mini-reasoning",   # 3.8B reasoning, MIT
    ],
}


def get_tier1_models():
    """Get all Tier 1 models for fast experiments."""
    return TIER1_2026


def get_quick_test_models():
    """Get quick test set covering all paradigms."""
    return QUICK_TEST


def get_models_by_paradigm(paradigm: str):
    """Get models for a specific paradigm."""
    return PARADIGMS.get(paradigm, [])


def get_all_paradigms():
    """Get list of all paradigms."""
    return list(PARADIGMS.keys())
