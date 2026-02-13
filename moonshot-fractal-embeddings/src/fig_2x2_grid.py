"""
Generate the 2x2 viral grid figure: V5 vs MRL at 64d vs 256d.
Shows "semantic zoom" effect for V5 vs "quality degradation" for MRL.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def load_clinc_data():
    """Load 5-seed CLINC benchmark data."""
    with open("results/benchmark_bge-small_clinc.json") as f:
        data = json.load(f)

    v5_seeds = []
    mrl_seeds = []

    for seed in data["seeds"]:
        seed_str = str(seed)
        v5 = data["v5"][seed_str]["prefix_accuracy"]
        mrl = data["mrl"][seed_str]["prefix_accuracy"]
        v5_seeds.append(v5)
        mrl_seeds.append(mrl)

    # Average across seeds
    keys = ["j1_l0", "j1_l1", "j2_l0", "j2_l1", "j3_l0", "j3_l1", "j4_l0", "j4_l1"]
    v5_mean = {k: np.mean([s[k] for s in v5_seeds]) for k in keys}
    v5_std = {k: np.std([s[k] for s in v5_seeds]) for k in keys}
    mrl_mean = {k: np.mean([s[k] for s in mrl_seeds]) for k in keys}
    mrl_std = {k: np.std([s[k] for s in mrl_seeds]) for k in keys}

    return v5_mean, v5_std, mrl_mean, mrl_std


def make_2x2_grid():
    """Create the viral 2x2 grid figure."""
    v5, v5_s, mrl, mrl_s = load_clinc_data()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Colors
    v5_color = "#2196F3"  # Blue
    mrl_color = "#FF9800"  # Orange
    coarse_color = "#4CAF50"  # Green
    fine_color = "#9C27B0"  # Purple

    dims = [64, 128, 192, 256]
    dim_labels = ["64d", "128d", "192d", "256d"]

    # V5 data
    v5_l0 = [v5["j1_l0"], v5["j2_l0"], v5["j3_l0"], v5["j4_l0"]]
    v5_l1 = [v5["j1_l1"], v5["j2_l1"], v5["j3_l1"], v5["j4_l1"]]

    # MRL data
    mrl_l0 = [mrl["j1_l0"], mrl["j2_l0"], mrl["j3_l0"], mrl["j4_l0"]]
    mrl_l1 = [mrl["j1_l1"], mrl["j2_l1"], mrl["j3_l1"], mrl["j4_l1"]]

    # === Top-left: V5 L0 (Coarse) ===
    ax = axes[0, 0]
    ax.bar(dims, [x * 100 for x in v5_l0], width=25, color=v5_color, alpha=0.85, edgecolor="white")
    ax.set_ylim(85, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("V5 (Fractal) -- Coarse (L0)", fontsize=12, fontweight="bold", color=v5_color)
    ax.set_xticks(dims)
    ax.set_xticklabels(dim_labels)
    for i, v in enumerate(v5_l0):
        ax.text(dims[i], v * 100 + 0.3, f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(y=v5_l0[0] * 100, color=coarse_color, linestyle="--", alpha=0.3)
    # Add annotation
    ramp = (v5_l0[0] - v5_l0[3]) * 100
    ax.annotate(f"Stable ({ramp:+.1f}pp)", xy=(256, v5_l0[3] * 100),
                fontsize=9, color=coarse_color, ha="right")

    # === Top-right: MRL L0 (Coarse) ===
    ax = axes[0, 1]
    ax.bar(dims, [x * 100 for x in mrl_l0], width=25, color=mrl_color, alpha=0.85, edgecolor="white")
    ax.set_ylim(85, 100)
    ax.set_title("MRL (Standard) -- Coarse (L0)", fontsize=12, fontweight="bold", color=mrl_color)
    ax.set_xticks(dims)
    ax.set_xticklabels(dim_labels)
    for i, v in enumerate(mrl_l0):
        ax.text(dims[i], v * 100 + 0.3, f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(y=mrl_l0[0] * 100, color=coarse_color, linestyle="--", alpha=0.3)
    ramp = (mrl_l0[3] - mrl_l0[0]) * 100
    ax.annotate(f"Flat ({ramp:+.1f}pp)", xy=(256, mrl_l0[3] * 100),
                fontsize=9, color=mrl_color, ha="right")

    # === Bottom-left: V5 L1 (Fine) ===
    ax = axes[1, 0]
    bars = ax.bar(dims, [x * 100 for x in v5_l1], width=25, color=v5_color, alpha=0.85, edgecolor="white")
    ax.set_ylim(45, 75)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Embedding Dimensions", fontsize=11)
    ax.set_title("V5 (Fractal) -- Fine (L1)", fontsize=12, fontweight="bold", color=v5_color)
    ax.set_xticks(dims)
    ax.set_xticklabels(dim_labels)
    for i, v in enumerate(v5_l1):
        ax.text(dims[i], v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    # Highlight the ramp
    ramp = (v5_l1[3] - v5_l1[0]) * 100
    ax.annotate(f"ZOOM: +{ramp:.1f}pp",
                xy=(160, (v5_l1[0] + v5_l1[3]) / 2 * 100),
                fontsize=11, color=fine_color, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor=fine_color))
    # Arrow from 64d to 256d
    ax.annotate("", xy=(250, v5_l1[3] * 100 - 0.5), xytext=(70, v5_l1[0] * 100 + 0.5),
                arrowprops=dict(arrowstyle="->", color=fine_color, lw=2))

    # === Bottom-right: MRL L1 (Fine) ===
    ax = axes[1, 1]
    ax.bar(dims, [x * 100 for x in mrl_l1], width=25, color=mrl_color, alpha=0.85, edgecolor="white")
    ax.set_ylim(45, 75)
    ax.set_xlabel("Embedding Dimensions", fontsize=11)
    ax.set_title("MRL (Standard) -- Fine (L1)", fontsize=12, fontweight="bold", color=mrl_color)
    ax.set_xticks(dims)
    ax.set_xticklabels(dim_labels)
    for i, v in enumerate(mrl_l1):
        ax.text(dims[i], v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ramp = (mrl_l1[3] - mrl_l1[0]) * 100
    ax.annotate(f"Flat: {ramp:+.1f}pp",
                xy=(160, (mrl_l1[0] + mrl_l1[3]) / 2 * 100),
                fontsize=11, color="gray", fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"))

    # Overall title
    fig.suptitle("Semantic Zoom: V5 Truncation Changes Meaning, MRL Truncation Loses Quality",
                 fontsize=13, fontweight="bold", y=0.98)

    # Add text box with key takeaway
    fig.text(0.5, 0.01,
             "V5: 64d encodes COARSE semantics (L0=96.4%, L1=53.5%)  |  "
             "256d encodes FINE semantics (L0=95.4%, L1=67.6%)\n"
             "MRL: All dimensions encode the SAME thing at varying quality  |  "
             "Steerability: V5 = +0.150, MRL = +0.007",
             ha="center", fontsize=9, style="italic",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    # Save
    out_dir = Path("results/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_2x2_semantic_zoom.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_2x2_semantic_zoom.pdf", bbox_inches="tight")
    print(f"Saved to {out_dir / 'fig_2x2_semantic_zoom.png'}")
    plt.close()


if __name__ == "__main__":
    make_2x2_grid()
