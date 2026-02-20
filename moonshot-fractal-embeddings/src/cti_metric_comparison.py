#!/usr/bin/env python -u
"""
METRIC COMPARISON: kappa vs Fisher Criterion vs CKA vs MI vs ID

Critical reviewer question: "Is kappa just renamed Fisher SNR?"

We compare predictive power of:
  1. kappa = tr(S_B) / tr(S_W)          [our metric, trace ratio]
  2. fisher = tr(S_W^{-1} S_B)          [classic Fisher criterion, inverse-weighted]
  3. dist_ratio = E[NN_inter]/E[NN_intra] [our observable OP]
  4. CKA = ||Y^T X||_F^2 / (||X^TX||_F * ||Y^TY||_F)  [centered kernel alignment]
  5. MI_knn = kNN-based MI estimate       [mutual information proxy]
  6. eff_rank = exp(H(singular values))  [spectral effective rank]

All metrics predict logit(q) for kNN classification quality.

Dataset: Pythia-160m, Pythia-410m on CLINC150 + TREC (all layers)
Output: R^2, Spearman rho, cross-model generalization table
"""

import json
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model
from hierarchical_datasets import load_hierarchical_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
]
DATASETS = ["clinc", "trec"]
MAX_SAMPLES = 1500   # per dataset
BATCH_SIZE = 32


# ============================================================
# Metric 1: kappa = tr(S_B) / tr(S_W)
# ============================================================

def compute_kappa(X, labels):
    """Trace ratio: tr(S_B) / tr(S_W)."""
    unique = np.unique(labels)
    grand = X.mean(0)
    tr_sb = 0.0
    tr_sw = 0.0
    for lbl in unique:
        m = labels == lbl
        if m.sum() < 2:
            continue
        Xk = X[m]; muk = Xk.mean(0)
        tr_sb += float(m.sum()) * float(((muk - grand) ** 2).sum())
        tr_sw += float(((Xk - muk) ** 2).sum())
    return tr_sb / (tr_sw + 1e-10)


# ============================================================
# Metric 2: Fisher criterion = tr(S_W^{-1} S_B)
# (truncated SVD for efficiency when d >> K)
# ============================================================

def compute_fisher_criterion(X, labels, max_rank=50):
    """
    Classic Fisher criterion: tr(S_W^{-1} S_B).

    Uses SVD of X to get S_W = X_c^T X_c / n where X_c is class-mean-centered.
    Computes S_W^{-1} via pseudoinverse on top-max_rank components.

    Note: S_W has rank min(n-K, d). For d >> K, full inversion is expensive.
    We use the K-dimensional subspace of S_B (rank K-1) and project.
    """
    unique = np.unique(labels)
    K = len(unique)
    n, d = X.shape

    # Compute class means and grand mean
    grand = X.mean(0)
    class_means = np.array([X[labels == lbl].mean(0) for lbl in unique])  # [K, d]
    n_per_class = np.array([np.sum(labels == lbl) for lbl in unique])     # [K]

    # Within-class centered matrix [n, d]
    X_w = X.copy()
    for i, lbl in enumerate(unique):
        X_w[labels == lbl] -= class_means[i]

    # S_W = X_w^T X_w / (n - K)
    # Use SVD of X_w for efficient pseudoinverse
    # X_w is [n, d], SVD: X_w = U S V^T
    # S_W = V S^2 V^T / (n-K)
    # S_W^{-1} ~ V diag(1/s_i^2) V^T * (n-K) (truncated)
    rank = min(max_rank, n - K, d, 2 * K)
    U, sv, Vt = np.linalg.svd(X_w, full_matrices=False)
    # Take top `rank` components
    sv_r = sv[:rank]; Vt_r = Vt[:rank]  # [rank, d]
    sw_inv_diag = (n - K) / (sv_r ** 2 + 1e-8)  # [rank]

    # Between-class: B = sqrt(n_k) * (mu_k - grand)  [K, d]
    B = np.sqrt(n_per_class[:, None]) * (class_means - grand)   # [K, d]
    # Project B onto low-rank S_W^{-1} basis:
    # tr(S_W^{-1} S_B) = sum_k (B_k^T S_W^{-1} B_k)
    # S_W^{-1} B_k = Vt_r^T diag(sw_inv_diag) Vt_r B_k
    Vt_B = Vt_r @ B.T   # [rank, K]
    fisher = float(np.sum(sw_inv_diag[:, None] * Vt_B ** 2))
    return fisher


# ============================================================
# Metric 3: dist_ratio = E[NN_inter] / E[NN_intra]
# ============================================================

def compute_dist_ratio(X, labels, n_sample=500):
    """Ratio of mean inter-class NN to intra-class NN distances."""
    idx = np.random.choice(len(labels), min(n_sample, len(labels)), replace=False)
    Xs = X[idx]; ys = labels[idx]
    Xs_n = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    sim = Xs_n @ Xs_n.T
    D = (2 - 2 * sim).clip(0)
    intra, inter = [], []
    for i in range(len(ys)):
        same = ys == ys[i]
        same[i] = False
        if same.sum() > 0:
            intra.append(D[i, same].min())
        diff = ~(ys == ys[i])
        if diff.sum() > 0:
            inter.append(D[i, diff].min())
    if not intra or not inter:
        return 1.0
    return float(np.mean(inter)) / (float(np.mean(intra)) + 1e-10)


# ============================================================
# Metric 4: Linear CKA
# ============================================================

def compute_cka(X, labels):
    """
    Linear CKA between embeddings and one-hot label matrix.
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    n = len(labels)
    unique = np.unique(labels)
    K = len(unique)
    # One-hot Y, centered
    Y = np.zeros((n, K))
    for i, lbl in enumerate(unique):
        Y[labels == lbl, i] = 1.0
    # Center both
    X_c = X - X.mean(0)
    Y_c = Y - Y.mean(0)
    # HSIC(X, Y) = tr(X_c X_c^T Y_c Y_c^T) / n^2
    # = ||Y_c^T X_c||_F^2 / n^2
    YtX = Y_c.T @ X_c    # [K, d]
    XtX = X_c.T @ X_c    # [d, d]
    YtY = Y_c.T @ Y_c    # [K, K]
    hsic_xy = float(np.sum(YtX ** 2)) / n ** 2
    hsic_xx = float(np.sum(XtX ** 2)) / n ** 2
    hsic_yy = float(np.sum(YtY ** 2)) / n ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / (denom + 1e-10)


# ============================================================
# Metric 5: kNN-based MI estimate (kraskov estimator)
# ============================================================

def compute_mi_knn(X, labels, k=5, tr_idx=None, te_idx=None):
    """
    Approximate MI(X; Y) using kNN accuracy entropy.
    MI ~ H(Y) - H(Y|X) ~ log(K) - entropy_of_kNN_predictions.

    Uses same train/test split as kNN quality for consistency.
    """
    K = len(np.unique(labels))
    if tr_idx is None:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(sss.split(X, labels))

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X[tr_idx], labels[tr_idx])
    proba = knn.predict_proba(X[te_idx])   # [n_te, K]
    h_yx = -np.sum(proba * np.log(proba + 1e-10), axis=1).mean()
    h_y = np.log(K)
    return max(0.0, float(h_y - h_yx))


# ============================================================
# Metric 6: Effective rank
# ============================================================

def compute_eff_rank(X):
    """Effective rank = exp(entropy of normalized singular values)."""
    X_c = X - X.mean(0)
    sv = np.linalg.svd(X_c, compute_uv=False)
    sv = sv[sv > 1e-10]
    sv_n = sv / sv.sum()
    entropy = -np.sum(sv_n * np.log(sv_n + 1e-10))
    return float(np.exp(entropy))


# ============================================================
# Embedding extraction
# ============================================================

@torch.no_grad()
def extract_all_layers(model, tokenizer, texts, device=DEVICE):
    """Extract mean-pooled embeddings at every layer."""
    all_hidden = {}
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, return_dict=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for idx, hs in enumerate(out.hidden_states):
            pooled = (hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            if idx not in all_hidden:
                all_hidden[idx] = []
            all_hidden[idx].append(pooled.cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in all_hidden.items()}


# ============================================================
# Main
# ============================================================

def run_model_dataset(model_id, dataset_name):
    """Run all metrics for one model x dataset combination."""
    print(f"\n  Loading {model_id} on {dataset_name}...", flush=True)

    model, tokenizer = load_model(model_id, device=DEVICE)
    model.eval()

    data = load_hierarchical_dataset(dataset_name)
    texts = data["texts"][:MAX_SAMPLES]
    labels = np.array(data["fine_labels"][:MAX_SAMPLES])

    K = len(np.unique(labels))
    print(f"  n={len(texts)}, K={K}", flush=True)

    layer_reps = extract_all_layers(model, tokenizer, texts)
    n_layers = len(layer_reps)

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for layer_idx in range(n_layers):
        X = layer_reps[layer_idx]

        # kNN quality — 80/20 stratified split
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(sss.split(X, labels))
        knn_clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn_clf.fit(X[tr_idx], labels[tr_idx])
        knn_acc = float(knn_clf.score(X[te_idx], labels[te_idx]))
        q = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)
        logit_q = float(np.log(q / (1 - q) + 1e-10)) if 0 < q < 1 else (3.0 if q >= 1 else -3.0)

        # All metrics (geometry computed on full X; kNN split only for quality estimate)
        k_val = compute_kappa(X, labels)
        fish = compute_fisher_criterion(X, labels)
        dr = compute_dist_ratio(X, labels)
        cka = compute_cka(X, labels)
        mi = compute_mi_knn(X, labels, tr_idx=tr_idx, te_idx=te_idx)
        er = compute_eff_rank(X)

        rows.append({
            "layer": layer_idx,
            "knn_acc": knn_acc,
            "q": float(q),
            "logit_q": logit_q,
            "kappa": float(k_val),
            "fisher": float(fish),
            "dist_ratio": float(dr),
            "cka": float(cka),
            "mi_knn": float(mi),
            "eff_rank": float(er),
            "K": K,
        })
        if layer_idx % 4 == 0:
            print(f"    Layer {layer_idx:2d}: q={q:.3f} kappa={k_val:.3f} "
                  f"fisher={fish:.1f} dr={dr:.3f} cka={cka:.4f}", flush=True)

    return rows


def compute_r2(x, y):
    """R^2 of linear regression of y on x."""
    if len(x) < 3 or np.std(x) < 1e-10:
        return float("nan")
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    ss_res = np.sum((y - reg.predict(x)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def main():
    print("=" * 70)
    print("METRIC COMPARISON: kappa vs Fisher vs CKA vs MI vs eff_rank")
    print("=" * 70)

    all_data = {}   # model_id x dataset -> list of rows

    for model_id in MODELS:
        for dataset_name in DATASETS:
            key = f"{model_id.split('/')[-1]}__{dataset_name}"
            t0 = time.time()
            try:
                rows = run_model_dataset(model_id, dataset_name)
                all_data[key] = rows
                print(f"  Done {key}: {len(rows)} layers in {time.time()-t0:.0f}s",
                      flush=True)
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
                continue

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("WITHIN-MODEL ANALYSIS (Spearman rho vs logit_q)")
    print(f"{'='*70}")

    metrics = ["kappa", "fisher", "dist_ratio", "cka", "mi_knn", "eff_rank"]
    metric_rho = {m: [] for m in metrics}
    metric_r2 = {m: [] for m in metrics}

    for key, rows in all_data.items():
        logit_q = [r["logit_q"] for r in rows]
        print(f"\n  {key} (n_layers={len(rows)}):")
        for m in metrics:
            vals = [r[m] for r in rows]
            if all(np.isfinite(v) for v in vals + logit_q):
                rho = float(spearmanr(vals, logit_q).statistic)
                r2 = compute_r2(vals, logit_q)
                metric_rho[m].append(rho)
                metric_r2[m].append(r2)
                print(f"    {m:>12}: rho={rho:+.3f}, R2={r2:.3f}")
            else:
                print(f"    {m:>12}: NaN (non-finite values)")

    # ============================================================
    # CROSS-MODEL COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("CROSS-MODEL R^2 (pooling all models x datasets x layers)")
    print(f"{'='*70}")

    all_rows = [r for rows in all_data.values() for r in rows]
    logit_q_all = [r["logit_q"] for r in all_rows]
    print(f"\n  Total data points: {len(all_rows)}")
    print(f"\n  {'Metric':>15} {'Cross-model R2':>16} {'Mean within-rho':>17} {'Mean within-R2':>15}")
    print(f"  {'-'*70}")

    results_summary = {}
    for m in metrics:
        vals_all = [r[m] for r in all_rows]
        if all(np.isfinite(v) for v in vals_all + logit_q_all):
            cross_r2 = compute_r2(vals_all, logit_q_all)
            mean_rho = np.mean(metric_rho[m]) if metric_rho[m] else float("nan")
            mean_r2 = np.mean(metric_r2[m]) if metric_r2[m] else float("nan")
            print(f"  {m:>15} {cross_r2:>16.3f} {mean_rho:>17.3f} {mean_r2:>15.3f}")
            results_summary[m] = {
                "cross_model_r2": cross_r2,
                "mean_within_rho": float(mean_rho),
                "mean_within_r2": float(mean_r2),
                "per_model_rho": {k: v for k, v in zip(all_data.keys(), metric_rho[m])},
            }

    # ============================================================
    # KEY QUESTION: Is kappa different from Fisher criterion?
    # ============================================================
    print(f"\n{'='*70}")
    print("KEY QUESTION: kappa vs Fisher criterion")
    print(f"{'='*70}")

    kappa_vals = [r["kappa"] for r in all_rows]
    fisher_vals = [r["fisher"] for r in all_rows]
    if all(np.isfinite(v) for v in kappa_vals + fisher_vals):
        rho_kf = float(spearmanr(kappa_vals, fisher_vals).statistic)
        print(f"\n  Spearman(kappa, fisher) = {rho_kf:.4f}")
        print(f"  (If ~1.0: identical ranking power. If <0.9: genuinely different)")

        # Fisher gives different ranking?
        logit_q_all_arr = np.array(logit_q_all)
        kappa_arr = np.array(kappa_vals)
        fisher_arr = np.array(fisher_vals)

        # Residual of kappa on logit_q — does fisher add info?
        reg_kappa = LinearRegression().fit(kappa_arr.reshape(-1,1), logit_q_all_arr)
        resid_kappa = logit_q_all_arr - reg_kappa.predict(kappa_arr.reshape(-1,1))
        r2_fisher_on_resid = compute_r2(fisher_arr, resid_kappa)
        print(f"  R2 of fisher on kappa-residual = {r2_fisher_on_resid:.4f}")
        print(f"  (>0.05 = fisher adds independent information beyond kappa)")

        results_summary["kappa_vs_fisher"] = {
            "spearman_rho": rho_kf,
            "r2_fisher_on_kappa_residual": r2_fisher_on_resid,
        }

    # ============================================================
    # SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")
    if "kappa" in results_summary and "fisher" in results_summary:
        k_r2 = results_summary["kappa"]["cross_model_r2"]
        f_r2 = results_summary["fisher"]["cross_model_r2"]
        dr_r2 = results_summary["dist_ratio"]["cross_model_r2"]
        cka_r2 = results_summary["cka"]["cross_model_r2"]
        mi_r2 = results_summary["mi_knn"]["cross_model_r2"]

        checks = [
            ("kappa > Fisher (different and better)",
             k_r2 > f_r2,
             f"kappa_r2={k_r2:.3f} vs fisher_r2={f_r2:.3f}"),
            ("dist_ratio >= kappa (observable OP better)",
             dr_r2 >= k_r2,
             f"dr_r2={dr_r2:.3f} vs kappa_r2={k_r2:.3f}"),
            ("kappa > CKA",
             k_r2 > cka_r2,
             f"kappa_r2={k_r2:.3f} vs cka_r2={cka_r2:.3f}"),
            ("kappa > MI proxy",
             k_r2 > mi_r2,
             f"kappa_r2={k_r2:.3f} vs mi_r2={mi_r2:.3f}"),
        ]
        passes = sum(1 for _, p, _ in checks if p)
        for crit, passed, val in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {crit}: {val}")
        print(f"\n  TOTAL: {passes}/{len(checks)}")
        results_summary["scorecard_passes"] = int(passes)
        results_summary["scorecard_total"] = len(checks)

    # ============================================================
    # SAVE
    # ============================================================
    out_path = RESULTS_DIR / "cti_metric_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "models": MODELS,
            "datasets": DATASETS,
            "n_total_points": len(all_rows),
            "metrics_compared": metrics,
            "results": results_summary,
            "per_model_rows": all_data,
        }, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
