"""
Create a visual dashboard with charts for practical claim evidence:
- Classification mixed-accuracy vs compute tradeoff
- Dual-encoder replacement quality + ops simplification
- HNSW latency speedup panel (3.71x treated as granted benchmark)
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
OUT_PATH = FIG_DIR / "claim_composite_visual_dashboard.png"


def _pick_alpha(rows, alpha=0.5):
    for row in rows:
        if abs(float(row["alpha"]) - float(alpha)) < 1e-9:
            return row
    raise ValueError(f"Could not find alpha={alpha}")


def load_stats():
    path = RESULTS_DIR / "pareto_dual_encoder_clinc.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cls_v5 = data["classification_pareto"]["v5_adaptive"]
    cls_mrl256 = data["classification_pareto"]["mrl_fixed_256"]
    alphas = np.array([float(r["alpha"]) for r in cls_v5], dtype=float)
    v5_acc = np.array([float(r["mixed_acc_mean"]) for r in cls_v5], dtype=float)
    mrl256_acc = np.array([float(r["mixed_acc_mean"]) for r in cls_mrl256], dtype=float)
    v5_dim = np.array([float(r["avg_dim"]) for r in cls_v5], dtype=float)
    mrl256_dim = np.array([float(r["avg_dim"]) for r in cls_mrl256], dtype=float)

    v5_50 = _pick_alpha(cls_v5, 0.5)
    mrl_50 = _pick_alpha(cls_mrl256, 0.5)
    dual = data["dual_encoder"]

    v5_l0_seed = np.array([float(r["v5_64d"]["l0"]) for r in dual], dtype=float)
    e_l0_seed = np.array([float(r["e_l0_256d"]["l0"]) for r in dual], dtype=float)
    v5_l1_seed = np.array([float(r["v5_256d"]["l1"]) for r in dual], dtype=float)
    mrl_l1_seed = np.array([float(r["mrl_256d"]["l1"]) for r in dual], dtype=float)

    crossover_idx = int(np.argmax(v5_acc >= mrl256_acc))
    idx50 = int(np.where(np.isclose(alphas, 0.5))[0][0])
    idx30 = int(np.where(np.isclose(alphas, 0.3))[0][0])
    idx70 = int(np.where(np.isclose(alphas, 0.7))[0][0])

    return {
        "alphas": alphas,
        "v5_acc": v5_acc,
        "mrl256_acc": mrl256_acc,
        "v5_dim": v5_dim,
        "mrl256_dim": mrl256_dim,
        "idx30": idx30,
        "idx50": idx50,
        "idx70": idx70,
        "crossover_idx": crossover_idx,
        "v5_50_acc": float(v5_50["mixed_acc_mean"]),
        "mrl_50_acc": float(mrl_50["mixed_acc_mean"]),
        "v5_50_dim": float(v5_50["avg_dim"]),
        "mrl_50_dim": float(mrl_50["avg_dim"]),
        "dim_saving_pct": (float(mrl_50["avg_dim"]) - float(v5_50["avg_dim"])) / float(mrl_50["avg_dim"]) * 100.0,
        "acc_gain_pp": (float(v5_50["mixed_acc_mean"]) - float(mrl_50["mixed_acc_mean"])) * 100.0,
        "v5_l0_seed": v5_l0_seed,
        "e_l0_seed": e_l0_seed,
        "v5_l1_seed": v5_l1_seed,
        "mrl_l1_seed": mrl_l1_seed,
        "v5_l0_mean": float(np.mean(v5_l0_seed)),
        "e_l0_mean": float(np.mean(e_l0_seed)),
        "v5_l1_mean": float(np.mean(v5_l1_seed)),
        "mrl_l1_mean": float(np.mean(mrl_l1_seed)),
        "v5_l0_std": float(np.std(v5_l0_seed, ddof=1)),
        "e_l0_std": float(np.std(e_l0_seed, ddof=1)),
        "v5_l1_std": float(np.std(v5_l1_seed, ddof=1)),
        "mrl_l1_std": float(np.std(mrl_l1_seed, ddof=1)),
        "l0_gap_pp": float((np.mean(v5_l0_seed) - np.mean(e_l0_seed)) * 100.0),
        "l1_gap_pp": float((np.mean(v5_l1_seed) - np.mean(mrl_l1_seed)) * 100.0),
    }


def render(stats):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(20, 12), dpi=220)
    fig.patch.set_facecolor("#eef3fb")
    gs = fig.add_gridspec(
        2, 3,
        left=0.04,
        right=0.98,
        top=0.90,
        bottom=0.08,
        wspace=0.20,
        hspace=0.30,
    )

    title = "Fractal Embeddings Practical Utility Dashboard (CLINC)"
    subtitle = (
        "Charts show where this is useful: quality-cost routing, latency wins, and single-model simplification."
    )
    fig.text(0.04, 0.955, title, fontsize=27, fontweight="bold", color="#0f172a")
    fig.text(0.04, 0.925, subtitle, fontsize=13, color="#334155")

    # A) Mixed accuracy vs workload mix
    ax_a = fig.add_subplot(gs[0, 0])
    x = stats["alphas"]
    y_v5 = stats["v5_acc"] * 100.0
    y_mrl = stats["mrl256_acc"] * 100.0
    ax_a.plot(x, y_v5, color="#1d4ed8", linewidth=2.8, label="V5 adaptive")
    ax_a.plot(x, y_mrl, color="#dc2626", linewidth=2.4, linestyle="--", label="MRL fixed 256d")
    ax_a.fill_between(x, y_v5, y_mrl, where=(y_v5 >= y_mrl), color="#22c55e", alpha=0.15)
    cidx = stats["crossover_idx"]
    ax_a.axvline(x=x[cidx], color="#16a34a", linestyle=":", linewidth=1.6)
    ax_a.scatter([x[stats["idx50"]]], [y_v5[stats["idx50"]]], color="#1d4ed8", s=45, zorder=5)
    ax_a.scatter([x[stats["idx50"]]], [y_mrl[stats["idx50"]]], color="#dc2626", s=45, zorder=5)
    ax_a.annotate(
        f"+{stats['acc_gain_pp']:.2f}pp @ alpha=0.5",
        (x[stats["idx50"]], y_v5[stats["idx50"]]),
        xytext=(0.56, y_v5[stats["idx50"]] + 0.45),
        arrowprops={"arrowstyle": "->", "color": "#1d4ed8", "lw": 1.2},
        fontsize=10,
        color="#1d4ed8",
    )
    ax_a.set_title("A) Mixed Accuracy vs Coarse Query Fraction")
    ax_a.set_xlabel("alpha (fraction of coarse queries)")
    ax_a.set_ylabel("Mixed accuracy (%)")
    ax_a.grid(alpha=0.25)
    ax_a.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax_a.text(
        0.02, 0.05,
        f"Crossover: alpha >= {x[cidx]:.2f}",
        transform=ax_a.transAxes,
        fontsize=10,
        color="#166534",
        bbox={"facecolor": "#dcfce7", "edgecolor": "#86efac", "boxstyle": "round,pad=0.25"},
    )

    # B) Compute footprint vs workload mix
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(x, stats["v5_dim"], color="#0ea5e9", linewidth=2.8, label="V5 adaptive avg dim")
    ax_b.plot(x, stats["mrl256_dim"], color="#ef4444", linewidth=2.4, linestyle="--", label="MRL fixed dim")
    ax_b.scatter([x[stats["idx50"]]], [stats["v5_dim"][stats["idx50"]]], color="#0ea5e9", s=55, zorder=5)
    ax_b.text(
        x[stats["idx50"]] + 0.015,
        stats["v5_dim"][stats["idx50"]] + 8,
        f"{stats['v5_50_dim']:.0f}d ({stats['dim_saving_pct']:.1f}% lower)",
        fontsize=10,
        color="#0c4a6e",
    )
    ax_b.set_title("B) Compute Load vs Coarse Query Fraction")
    ax_b.set_xlabel("alpha (fraction of coarse queries)")
    ax_b.set_ylabel("Average dimensions used")
    ax_b.set_ylim(56, 272)
    ax_b.grid(alpha=0.25)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.95)

    # C) HNSW latency bars (granted)
    ax_c = fig.add_subplot(gs[0, 2])
    dims = ["64d", "256d"]
    lats = [39.0, 145.0]
    colors = ["#7c3aed", "#475569"]
    bars = ax_c.bar(dims, lats, color=colors, alpha=0.92, width=0.62)
    for bar, lat in zip(bars, lats):
        ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3.0, f"{lat:.0f} us",
                  ha="center", fontsize=11, color="#111827")
    ax_c.set_title("C) Query Latency (HNSW, n=100K, granted)")
    ax_c.set_ylabel("Single-query latency (us)")
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.text(
        0.5, 0.86, "3.71x faster at 64d",
        transform=ax_c.transAxes,
        ha="center",
        fontsize=12,
        color="#4c1d95",
        bbox={"facecolor": "#ede9fe", "edgecolor": "#c4b5fd", "boxstyle": "round,pad=0.35"},
    )
    ax_c.text(
        0.5, 0.05, "Run benchmark: python src/latency_and_robustness.py",
        transform=ax_c.transAxes,
        ha="center",
        fontsize=9.5,
        color="#334155",
    )

    # D) Coarse accuracy: dual vs V5
    ax_d = fig.add_subplot(gs[1, 0])
    mean_dual = stats["e_l0_mean"] * 100.0
    mean_v5 = stats["v5_l0_mean"] * 100.0
    sd_dual = stats["e_l0_std"] * 100.0
    sd_v5 = stats["v5_l0_std"] * 100.0
    ax_d.bar(
        [0, 1],
        [mean_dual, mean_v5],
        yerr=[sd_dual, sd_v5],
        capsize=5,
        color=["#ef4444", "#10b981"],
        alpha=0.9,
        width=0.62,
    )
    rng = np.random.default_rng(0)
    x_dual = np.full_like(stats["e_l0_seed"], 0, dtype=float) + rng.normal(0, 0.035, size=stats["e_l0_seed"].shape[0])
    x_v5 = np.full_like(stats["v5_l0_seed"], 1, dtype=float) + rng.normal(0, 0.035, size=stats["v5_l0_seed"].shape[0])
    ax_d.scatter(x_dual, stats["e_l0_seed"] * 100.0, s=35, color="#7f1d1d", zorder=4)
    ax_d.scatter(x_v5, stats["v5_l0_seed"] * 100.0, s=35, color="#064e3b", zorder=4)
    ax_d.set_xticks([0, 1], ["Dual E_L0 (256d)", "V5 prefix (64d)"])
    ax_d.set_ylabel("Coarse L0 accuracy (%)")
    ax_d.set_title("D) Coarse Task: Single V5 Beats Dedicated Encoder")
    ax_d.grid(axis="y", alpha=0.25)
    ax_d.text(
        0.5, 0.90,
        f"+{stats['l0_gap_pp']:.2f}pp in favor of V5",
        transform=ax_d.transAxes,
        ha="center",
        fontsize=11,
        color="#065f46",
        bbox={"facecolor": "#d1fae5", "edgecolor": "#6ee7b7", "boxstyle": "round,pad=0.25"},
    )

    # E) Fine-task parity at full resolution
    ax_e = fig.add_subplot(gs[1, 1])
    mean_mrl_l1 = stats["mrl_l1_mean"] * 100.0
    mean_v5_l1 = stats["v5_l1_mean"] * 100.0
    sd_mrl_l1 = stats["mrl_l1_std"] * 100.0
    sd_v5_l1 = stats["v5_l1_std"] * 100.0
    ax_e.bar(
        [0, 1],
        [mean_mrl_l1, mean_v5_l1],
        yerr=[sd_mrl_l1, sd_v5_l1],
        capsize=5,
        color=["#f97316", "#2563eb"],
        alpha=0.9,
        width=0.62,
    )
    ax_e.set_xticks([0, 1], ["MRL / E_L1 (256d)", "V5 full (256d)"])
    ax_e.set_ylabel("Fine L1 accuracy (%)")
    ax_e.set_title("E) Fine Task Quality at 256d")
    ax_e.grid(axis="y", alpha=0.25)
    ax_e.text(
        0.5, 0.90,
        f"Gap: {stats['l1_gap_pp']:+.2f}pp (near parity)",
        transform=ax_e.transAxes,
        ha="center",
        fontsize=11,
        color="#0f172a",
        bbox={"facecolor": "#e2e8f0", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.25"},
    )

    # F) Operational simplification profile
    ax_f = fig.add_subplot(gs[1, 2])
    metrics = ["Models", "Indexes", "Coarse dim", "Coarse acc"]
    dual_norm = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    v5_norm = np.array([
        1.0 / 2.0,            # 1 model vs 2
        1.0 / 2.0,            # 1 index vs 2
        64.0 / 256.0,         # coarse path dim ratio
        stats["v5_l0_mean"] / stats["e_l0_mean"],  # accuracy ratio
    ], dtype=float)
    xpos = np.arange(len(metrics))
    w = 0.36
    ax_f.bar(xpos - w / 2, dual_norm, width=w, color="#94a3b8", alpha=0.92, label="Dual baseline")
    ax_f.bar(xpos + w / 2, v5_norm, width=w, color="#1d4ed8", alpha=0.92, label="V5 system")
    ax_f.axhline(1.0, color="#64748b", linewidth=1.0, linestyle=":")
    ax_f.set_xticks(xpos, metrics)
    ax_f.set_ylim(0, 1.25)
    ax_f.set_ylabel("Relative to dual baseline (=1.0)")
    ax_f.set_title("F) System Simplification + Quality Retention")
    ax_f.grid(axis="y", alpha=0.25)
    ax_f.legend(loc="upper right", frameon=True, framealpha=0.95)

    labels = ["2 -> 1", "2 -> 1", "256d -> 64d", f"{stats['e_l0_mean']*100:.1f}% -> {stats['v5_l0_mean']*100:.1f}%"]
    for i, lbl in enumerate(labels):
        ax_f.text(i, 0.05, lbl, ha="center", va="bottom", fontsize=9, color="#1f2937", rotation=0)

    # Footer
    footer = (
        "Data: results/pareto_dual_encoder_clinc.json (classification_pareto + dual_encoder). "
        "Latency panel uses granted benchmark value 39us vs 145us (3.71x). "
        f"Generated {datetime.now().isoformat(timespec='seconds')}"
    )
    fig.text(0.04, 0.03, footer, fontsize=10, color="#334155")
    fig.text(
        0.04, 0.012,
        "Interpretation: strongest practical upside appears when coarse-query share is high and latency/ops budgets matter.",
        fontsize=10,
        color="#334155",
    )

    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    return OUT_PATH


def main():
    stats = load_stats()
    out = render(stats)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
