"""Head-to-head BPB comparison: Sutra vs competitors on same test set.

Runs ALL models on the SAME test text and computes byte-level perplexity.
This is the only fair comparison between byte-level and token-level models.

Usage:
    python eval/head_to_head.py --sutra-model results/sutra_production_best.pt
"""

import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))


def compute_sutra_bpb(model_path, test_text, dim=5120, seq_len=512):
    """Compute BPB for Sutra model."""
    from sutra_v04 import SutraV04
    model = SutraV04(dim=dim, patch_size=4, max_rounds=6, k_retrieval=16, max_seq=seq_len)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cuda"))
    model = model.cuda().eval()

    data = torch.tensor(list(test_text.encode("utf-8")), dtype=torch.long)
    total_loss = total_bytes = 0

    with torch.no_grad():
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i + seq_len].unsqueeze(0).cuda()
            y = data[i + 1:i + seq_len + 1].unsqueeze(0).cuda()
            logits, _ = model(x)
            Tc = min(logits.size(1), y.size(1))
            loss = F.cross_entropy(logits[:, :Tc].reshape(-1, 256), y[:, :Tc].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_bytes += y[:, :Tc].numel()

    del model
    torch.cuda.empty_cache()
    return total_loss / (total_bytes * math.log(2))


def compute_ollama_bpb(model_name, test_text, chunk_size=2000):
    """Compute BPB for an ollama model by measuring log-probability of test text.

    We use the completion API and measure cross-entropy on the model's predictions.
    This is an APPROXIMATION since we can't get per-token log-probs from ollama easily.
    Instead we measure: can the model predict the next chunk given context?
    """
    # For now: use generation quality as proxy
    # TODO: compute actual log-probs via HuggingFace for fair comparison

    # Instead, let's use HuggingFace models directly for accurate BPB
    print(f"  {model_name}: using HuggingFace for accurate BPB (TODO)")
    return None


def compute_hf_bpb(model_name, test_text, max_chars=50000):
    """Compute BPB for a HuggingFace model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  transformers not installed")
        return None

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    text = test_text[:max_chars]
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(model.device)

    total_loss = 0
    total_tokens = 0
    seq_len = 512
    model.eval()

    with torch.no_grad():
        for i in range(0, input_ids.size(1) - seq_len, seq_len):
            chunk = input_ids[:, i:i + seq_len + 1]
            x = chunk[:, :-1]
            y = chunk[:, 1:]
            outputs = model(x)
            logits = outputs.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    # Convert token-level CE to byte-level BPB
    # BPB = CE_tokens * tokens_per_byte / ln(2)
    text_bytes = len(text.encode("utf-8"))
    text_tokens = total_tokens
    tokens_per_byte = text_tokens / text_bytes
    bpb = (total_loss / total_tokens) * tokens_per_byte / math.log(2)

    del model
    torch.cuda.empty_cache()
    return bpb


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sutra-model", help="Path to Sutra checkpoint")
    parser.add_argument("--test-text", default=str(REPO / "data" / "corpus_test.txt"))
    parser.add_argument("--competitors", default="EleutherAI/pythia-410m",
                        help="Comma-separated HF model names")
    args = parser.parse_args()

    # Load test text
    with open(args.test_text, "r", encoding="utf-8") as f:
        test_text = f.read()[:50000]
    print(f"Test text: {len(test_text):,} chars")

    results = {}

    # Sutra
    if args.sutra_model and Path(args.sutra_model).exists():
        print(f"\nEvaluating Sutra...")
        bpb = compute_sutra_bpb(args.sutra_model, test_text)
        results["Sutra-475M"] = bpb
        print(f"  Sutra-475M BPB: {bpb:.4f}")

    # Competitors
    for model_name in args.competitors.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        print(f"\nEvaluating {model_name}...")
        bpb = compute_hf_bpb(model_name, test_text)
        if bpb is not None:
            results[model_name] = bpb
            print(f"  {model_name} BPB: {bpb:.4f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"HEAD-TO-HEAD BPB COMPARISON")
    print(f"{'='*50}")
    for name, bpb in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:30s}: {bpb:.4f}")

    with open(REPO / "results" / "head_to_head.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
