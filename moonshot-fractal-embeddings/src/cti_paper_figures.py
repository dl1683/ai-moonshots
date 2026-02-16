#!/usr/bin/env python
"""Generate publication-quality figures for the CTI paper.

Figures:
1. Teaser: depth profile examples showing universal shape
2. LOFO results: predicted vs observed scatter
3. Cross-family transfer table (text-based, for LaTeX)
4. BLOOM blind test: shape match despite absolute offset
5. Late-training degradation heatmap
6. Parameter stability across LOFO folds
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit
from scipy.stats import spearmanr
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures" / "cti"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}

# Color scheme
FAMILY_COLORS = {
    "pythia": "#1f77b4",
    "olmo2": "#ff7f0e",
    "cerebras-gpt": "#2ca02c",
    "opt": "#d62728",
    "gpt2": "#9467bd",
    "bloom": "#8c564b",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


def load_all_observations():
    """Load all observations (same as universal prediction)."""
    obs = []
    for source, path, family_fn in [
        ("pythia", "cti_checkpoint_sweep_all.json", lambda r: "pythia"),
        ("olmo2", "cti_olmo2_sweep.json", lambda r: "olmo2"),
        ("multi", "cti_multi_family.json", lambda r: r["family"]),
    ]:
        with open(RESULTS_DIR / path) as f:
            data = json.load(f)
        for r in data["results"]:
            if "error" in r:
                continue
            if source == "pythia" and r.get("step", 0) == 0:
                continue
            for ds_name, ds_data in r["datasets"].items():
                n_c = DS_CLASSES.get(ds_name, 100)
                Q_ch = 1.0 / n_c
                for li_str, ld in ds_data["layers"].items():
                    li = int(li_str)
                    L = int(r["num_layers"])
                    N = float(r["N_params"])
                    C = float(r["C_flops"])
                    Q = (ld["knn_l1"] - Q_ch) / (1.0 - Q_ch)
                    Q = np.clip(Q, 0.001, 0.999)
                    obs.append({
                        "x": li / L, "Q": Q, "dataset": ds_name,
                        "model": r["model"], "family": family_fn(r),
                        "step": r.get("step", -1), "L": L,
                        "N": N, "C": C, "log_r": np.log(C) - np.log(N),
                        "knn_l1_raw": ld["knn_l1"],
                    })
    return obs


def predict_ds(params, obs_list, ds_list):
    alpha = params["alpha"]
    beta = params["beta"]
    mu_0 = params["mu_0"]
    mu_1 = params["mu_1"]
    b_d = params["b_d"]
    preds = []
    for o in obs_list:
        x_star = mu_0 + mu_1 * o["log_r"]
        logit_Q = b_d.get(o["dataset"], 0) + alpha * o["log_r"] - beta * (o["x"] - x_star) ** 2
        preds.append(expit(np.clip(logit_Q, -20, 20)))
    return np.array(preds)


def fig1_depth_profiles():
    """Figure 1: Depth profiles showing universal shape across families."""
    obs = load_all_observations()

    # Select representative models (1 per family, latest checkpoint)
    representatives = {
        "pythia": ("pythia-1.4b", 143000),
        "olmo2": ("olmo2-1b", 1907359),
        "cerebras-gpt": ("cerebras-gpt-1.3b", -1),
        "opt": ("opt-1.3b", -1),
        "gpt2": ("gpt2-large", -1),
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]
    ds_labels = ["CLINC-150", "DBPedia", "AG News", "TREC"]

    for di, (ds, label) in enumerate(zip(ds_names, ds_labels)):
        ax = axes[di]
        for fam, (model, step) in representatives.items():
            fam_obs = [o for o in obs if o["model"] == model and o["step"] == step and o["dataset"] == ds]
            if not fam_obs:
                continue
            fam_obs.sort(key=lambda o: o["x"])
            xs = [o["x"] for o in fam_obs]
            qs = [o["Q"] for o in fam_obs]
            ax.plot(xs, qs, "-o", color=FAMILY_COLORS[fam], label=fam if di == 0 else "",
                   markersize=3, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Normalized depth (l/L)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        if di == 0:
            ax.set_ylabel("Q_norm (chance-normalized)")

    axes[0].legend(loc="lower right", fontsize=7)
    plt.suptitle("Depth-wise representation quality profiles across 5 model families", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_depth_profiles.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig1_depth_profiles.png", bbox_inches="tight")
    plt.close()
    print("  Fig 1: depth profiles")


def fig2_lofo_scatter():
    """Figure 2: LOFO predicted vs observed scatter plot."""
    obs = load_all_observations()

    with open(RESULTS_DIR / "cti_holdout_prediction.json") as f:
        params = json.load(f)["fit_params"]

    # Compute LOFO predictions (simplified: use Pythia params for all)
    from scipy.optimize import minimize as scipy_minimize

    datasets = sorted(DS_CLASSES.keys())
    families = sorted(set(o["family"] for o in obs))

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))

    for fi, holdout_fam in enumerate(families):
        ax = axes[fi]
        train = [o for o in obs if o["family"] != holdout_fam]
        test = [o for o in obs if o["family"] == holdout_fam]

        # Fit on train
        n_ds = len(datasets)
        bounds = [(-1, 1), (0.01, 50), (-2, 2), (-0.5, 0.5)] + [(-10, 10)] * n_ds
        Q_tr = np.array([o["Q"] for o in train])

        def loss(p):
            alpha, beta, mu_0, mu_1 = p[:4]
            b_d = {ds: p[4 + i] for i, ds in enumerate(datasets)}
            preds = []
            for o in train:
                x_star = mu_0 + mu_1 * o["log_r"]
                logit_Q = b_d.get(o["dataset"], 0) + alpha * o["log_r"] - beta * (o["x"] - x_star) ** 2
                preds.append(expit(np.clip(logit_Q, -20, 20)))
            return np.mean((Q_tr - np.array(preds)) ** 2)

        best = None
        best_loss = float("inf")
        for trial in range(20):
            rng = np.random.RandomState(trial)
            x0 = [rng.uniform(b[0], b[1]) for b in bounds]
            try:
                res = scipy_minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": 3000, "ftol": 1e-12})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best = res
            except Exception:
                continue

        # Predict test
        Q_test = np.array([o["Q"] for o in test])
        fit_params = {
            "alpha": best.x[0], "beta": best.x[1],
            "mu_0": best.x[2], "mu_1": best.x[3],
            "b_d": {ds: best.x[4 + i] for i, ds in enumerate(datasets)},
        }
        Q_pred = predict_ds(fit_params, test, datasets)

        ss_res = np.sum((Q_test - Q_pred) ** 2)
        ss_tot = np.sum((Q_test - Q_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Color by dataset
        for ds in datasets:
            idx = [i for i, o in enumerate(test) if o["dataset"] == ds]
            if idx:
                ax.scatter(Q_test[idx], Q_pred[idx], s=3, alpha=0.4,
                          label=ds if fi == 0 else "")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Observed Q")
        if fi == 0:
            ax.set_ylabel("Predicted Q")
        ax.set_title(f"Holdout: {holdout_fam}\nR2={r2:.3f}")
        ax.set_xlim(0, 0.7)
        ax.set_ylim(0, 0.7)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=6, loc="lower right")
    plt.suptitle("Leave-One-Family-Out: Predicted vs Observed Quality", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_lofo_scatter.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig2_lofo_scatter.png", bbox_inches="tight")
    plt.close()
    print("  Fig 2: LOFO scatter")


def fig3_degradation():
    """Figure 3: Late-training degradation across families."""
    obs = load_all_observations()

    families = sorted(set(o["family"] for o in obs))
    ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]
    ds_labels = ["CLINC", "DBPedia", "AG News", "TREC"]

    # Compute best_x for latest checkpoint of each model
    models = sorted(set((o["model"], o["family"]) for o in obs))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Bar chart: degradation gap per family per dataset
    x_pos = np.arange(len(ds_names))
    width = 0.15

    for fi, fam in enumerate(families):
        gaps = []
        for ds in ds_names:
            fam_obs = [o for o in obs if o["family"] == fam and o["dataset"] == ds]
            if not fam_obs:
                gaps.append(0)
                continue
            # Get latest step per model
            model_steps = {}
            for o in fam_obs:
                key = o["model"]
                if key not in model_steps or o["step"] > model_steps[key]:
                    model_steps[key] = o["step"]

            model_gaps = []
            for model, step in model_steps.items():
                profile = [o for o in fam_obs if o["model"] == model and o["step"] == step]
                if len(profile) < 3:
                    continue
                best_Q = max(o["Q"] for o in profile)
                final_Q = max(profile, key=lambda o: o["x"])["Q"]
                model_gaps.append(best_Q - final_Q)

            gaps.append(np.mean(model_gaps) if model_gaps else 0)

        ax.bar(x_pos + fi * width, gaps, width, label=fam,
              color=FAMILY_COLORS[fam], alpha=0.8)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Quality gap (best layer - final layer)")
    ax.set_title("Late-training quality degradation across families")
    ax.set_xticks(x_pos + width * (len(families) - 1) / 2)
    ax.set_xticklabels(ds_labels)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_degradation.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig3_degradation.png", bbox_inches="tight")
    plt.close()
    print("  Fig 3: degradation")


def fig4_bloom_blind():
    """Figure 4: BLOOM blind test — shape matches, level shifted."""
    with open(RESULTS_DIR / "cti_bloom_blind_test.json") as f:
        data = json.load(f)

    with open(RESULTS_DIR / "cti_holdout_prediction.json") as f:
        params = json.load(f)["fit_params"]

    # Pick bloom-1.7b final checkpoint
    bloom_result = None
    for r in data["bloom_results"]:
        if r.get("model") == "bloom-1.7b" and r.get("step") == 300000:
            bloom_result = r
            break

    if bloom_result is None:
        print("  Fig 4: SKIPPED (no bloom-1.7b step=300000)")
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]
    ds_labels = ["CLINC-150", "DBPedia", "AG News", "TREC"]

    N = float(bloom_result["N_params"])
    C = float(bloom_result["C_flops"])
    L = int(bloom_result["num_layers"])
    log_r = np.log(C) - np.log(N)

    for di, (ds, label) in enumerate(zip(ds_names, ds_labels)):
        ax = axes[di]
        layers = bloom_result["datasets"][ds]["layers"]

        # Observed
        xs = []
        qs_obs = []
        for li_str, ld in sorted(layers.items(), key=lambda x: int(x[0])):
            li = int(li_str)
            xs.append(li / L)
            qs_obs.append(ld["Q_norm"])

        ax.plot(xs, qs_obs, "o-", color=FAMILY_COLORS["bloom"], label="BLOOM observed",
               markersize=4, linewidth=1.5)

        # Predicted (frozen Pythia)
        n_c = DS_CLASSES[ds]
        x_star = params["mu_0"] + params["mu_1"] * log_r
        preds = []
        for x in xs:
            logit_Q = params["b_d"][ds] + params["alpha"] * log_r - params["beta"] * (x - x_star) ** 2
            preds.append(expit(np.clip(logit_Q, -20, 20)))
        ax.plot(xs, preds, "s--", color="#333333", label="Pythia prediction",
               markersize=3, linewidth=1, alpha=0.7)

        # Calibrated prediction
        bd_shift = -0.75  # approximate
        preds_cal = []
        for x in xs:
            logit_Q = (params["b_d"][ds] + bd_shift) + params["alpha"] * log_r - params["beta"] * (x - x_star) ** 2
            preds_cal.append(expit(np.clip(logit_Q, -20, 20)))
        ax.plot(xs, preds_cal, "^:", color="#e377c2", label="Calibrated",
               markersize=3, linewidth=1, alpha=0.7)

        ax.set_xlabel("Normalized depth")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        if di == 0:
            ax.set_ylabel("Q_norm")

    axes[-1].legend(fontsize=7, loc="lower left")
    plt.suptitle("BLOOM-1.7B blind test: shape universal, level needs calibration", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_bloom_blind.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig4_bloom_blind.png", bbox_inches="tight")
    plt.close()
    print("  Fig 4: BLOOM blind test")


def main():
    print("Generating CTI paper figures...")
    fig1_depth_profiles()
    fig2_lofo_scatter()
    fig3_degradation()
    fig4_bloom_blind()
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
