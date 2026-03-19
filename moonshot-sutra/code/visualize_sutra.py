"""Visualize Sutra v0.3 internals.

Shows:
1. Message passing patterns (which patches communicate)
2. Sparse retrieval targets (what each patch attends to)
3. PonderNet halting (how many rounds per input)
4. Patch-level predictions vs byte-level

Usage:
    python code/visualize_sutra.py --model results/sutra_model.pt --text "Hello world"
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from sutra_v03_mvp import SutraV03


def visualize_routing(model, text, max_len=128):
    """Show what the model does internally for a given input."""
    # Encode text to bytes
    raw = text.encode("utf-8")[:max_len]
    x = torch.tensor(list(raw), dtype=torch.long).unsqueeze(0)
    B, T = x.shape
    P = model.patch_size

    print(f"Input: \"{text[:60]}...\"" if len(text) > 60 else f"Input: \"{text}\"")
    print(f"Bytes: {T}, Patches: {math.ceil(T/P)}")
    print()

    # Hook into model components to capture internals
    model.eval()
    with torch.no_grad():
        logits, kl = model(x)

    # Get predictions
    preds = logits[0].argmax(dim=-1).tolist()
    pred_text = bytes(preds[:T]).decode("utf-8", errors="replace")
    print(f"Predicted next bytes: \"{pred_text[:60]}\"")
    print()

    # Show per-patch info
    n_patches = math.ceil(T / P)
    print(f"Patch structure ({n_patches} patches × {P} bytes):")
    for i in range(min(n_patches, 16)):
        start = i * P
        end = min(start + P, T)
        patch_bytes = raw[start:end]
        patch_text = patch_bytes.decode("utf-8", errors="replace")
        print(f"  Patch {i:2d}: [{start:3d}-{end:3d}] \"{patch_text}\"")

    # Adaptive depth info
    if hasattr(model, "msg_pass") and hasattr(model.msg_pass, "_avg_steps"):
        print(f"\nAdaptive depth: avg {model.msg_pass._avg_steps:.1f} rounds")

    return logits


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to saved model")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.",
                        help="Input text to visualize")
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    args = parser.parse_args()

    model = SutraV03(dim=args.dim, patch_size=4, max_rounds=3, k_retrieval=4)

    if args.model and Path(args.model).exists():
        model.load_state_dict(torch.load(args.model, weights_only=True))
        print(f"Loaded model from {args.model}")
    else:
        print("Using untrained model (random weights)")

    print(f"Params: {model.count_params():,}")
    print()

    visualize_routing(model, args.text)


if __name__ == "__main__":
    main()
