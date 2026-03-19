"""lm-evaluation-harness wrapper for Sutra byte-level models.

Bridges Sutra (raw byte input/output) with lm-eval (expects tokenizer + log-probs).

Usage:
    # After training, evaluate:
    python eval/sutra_lm_eval_wrapper.py --model results/sutra_production_best.pt --tasks hellaswag,arc_easy

For now: provides BPB evaluation on standard test sets as a proxy.
Full lm-eval integration requires custom model class (TODO).
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))
from sutra_v04 import SutraV04


def evaluate_bpb(model, text, seq_len=512, device="cuda"):
    """Compute bits-per-byte on a text string."""
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    model.eval()
    total_loss = 0
    total_bytes = 0

    with torch.no_grad():
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i + seq_len].unsqueeze(0).to(device)
            y = data[i + 1:i + seq_len + 1].unsqueeze(0).to(device)
            logits, _ = model(x)
            Tc = min(logits.size(1), y.size(1))
            loss = F.cross_entropy(
                logits[:, :Tc].reshape(-1, 256),
                y[:, :Tc].reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_bytes += y[:, :Tc].numel()

    return total_loss / (total_bytes * math.log(2))


def evaluate_on_datasets(model, device="cuda"):
    """Evaluate Sutra on standard test sets."""
    results = {}

    # WikiText-2 (download if needed)
    try:
        from datasets import load_dataset
        wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        wt2_text = "\n".join(wt2["text"])[:500_000]
        results["wikitext2"] = evaluate_bpb(model, wt2_text, device=device)
        print(f"WikiText-2 BPB: {results['wikitext2']:.4f}")
    except Exception as e:
        print(f"WikiText-2 failed: {e}")

    # Our test corpus
    test_path = REPO / "data" / "corpus_test.txt"
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            test_text = f.read()[:500_000]
        results["corpus_test"] = evaluate_bpb(model, test_text, device=device)
        print(f"Corpus test BPB: {results['corpus_test']:.4f}")

    return results


def load_model(checkpoint_path, dim=5120, device="cuda"):
    """Load a trained Sutra model."""
    model = SutraV04(
        dim=dim, patch_size=4, max_rounds=6, k_retrieval=16, max_seq=512
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--dim", type=int, default=5120)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading Sutra model from {args.model}")
    model = load_model(args.model, dim=args.dim, device=args.device)
    print(f"Params: {model.count_params():,}")

    results = evaluate_on_datasets(model, device=args.device)

    # Save
    out = REPO / "results" / "sutra_eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    # Compare against known baselines
    print(f"\n{'='*50}")
    print(f"COMPARISON (approximate published BPB on WikiText-2):")
    print(f"  Pythia-410M:  ~0.92 BPB")
    print(f"  SmolLM-360M:  ~0.85 BPB (estimated)")
    print(f"  Qwen3-0.6B:   ~0.75 BPB (estimated)")
    if "wikitext2" in results:
        print(f"  Sutra-475M:   {results['wikitext2']:.4f} BPB")
        if results["wikitext2"] < 0.92:
            print(f"  SUTRA BEATS Pythia-410M!")
        if results["wikitext2"] < 0.85:
            print(f"  SUTRA BEATS SmolLM-360M!")
        if results["wikitext2"] < 0.75:
            print(f"  SUTRA BEATS Qwen3-0.6B!")


if __name__ == "__main__":
    main()
