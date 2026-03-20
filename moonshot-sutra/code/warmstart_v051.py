"""Warm-start v0.5.1 from v0.5 checkpoint.

Transfers 94% of parameters (60/64). Only the switching kernel gate,
mode logits, and halting head start fresh.

Usage:
    python code/warmstart_v051.py --checkpoint results/checkpoints_v05/step_10000.pt
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))


def warmstart(checkpoint_path, dim=768, ff_dim=1536, max_steps=6, output_path=None):
    from sutra_v05_ssm import SutraV05
    from sutra_v051 import SutraV051

    # Load v0.5 checkpoint
    print(f"Loading v0.5 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    old_state = ckpt["model"] if "model" in ckpt else ckpt

    # Create v0.5.1 model
    new_model = SutraV051(vocab_size=50257, dim=dim, ff_dim=ff_dim, max_steps=max_steps)
    new_state = new_model.state_dict()

    # Transfer compatible weights
    transferred = 0
    fresh = 0

    # Direct name matches
    for name in list(new_state.keys()):
        if name in old_state and new_state[name].shape == old_state[name].shape:
            new_state[name] = old_state[name]
            transferred += 1
        else:
            fresh += 1

    # Handle renamed transition kernel: v0.5 net.* -> v0.5.1 base.*
    remap = {
        "transition.net.0.weight": "transition.base.0.weight",
        "transition.net.0.bias": "transition.base.0.bias",
        "transition.net.2.weight": "transition.base.2.weight",
        "transition.net.2.bias": "transition.base.2.bias",
    }
    for old_name, new_name in remap.items():
        if old_name in old_state and new_name in new_state:
            if old_state[old_name].shape == new_state[new_name].shape:
                new_state[new_name] = old_state[old_name]
                transferred += 1
                fresh -= 1
                print(f"  Remapped: {old_name} -> {new_name}")

    new_model.load_state_dict(new_state)

    total = transferred + fresh
    print(f"\nTransferred: {transferred}/{total} ({transferred/total*100:.0f}%)")
    print(f"Fresh (random init): {fresh}/{total}")
    print(f"  New modules: switching gate, mode logits, halting head")

    # Save
    if output_path is None:
        output_path = REPO / "results" / "v051_warmstart.pt"
    torch.save({"model": new_model.state_dict(), "step": ckpt.get("step", 0),
                "warmstart_from": str(checkpoint_path)}, output_path)
    print(f"Saved to {output_path}")
    print(f"v0.5.1 params: {new_model.count_params():,}")
    return new_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    warmstart(args.checkpoint, output_path=args.output)
