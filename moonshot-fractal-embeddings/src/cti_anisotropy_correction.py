#!/usr/bin/env python -u
"""
ANISOTROPY CORRECTION: d_eff replaces d in Gumbel Race Law

THEORETICAL GAP: The zero-param probit theory gives rho=0.37 on real NNs
because it assumes isotropic within-class covariance (Sigma_W = sigma^2 I_d).
Real neural embeddings are HIGHLY anisotropic: a few large eigenvalues dominate.

THEORY: For anisotropic Sigma_W with eigenvalues {lambda_i}:
  D^2(x, mu_k) = sum_j lambda_j z_j^2  (weighted chi^2)
  By CLT:  D^2 ~ N(tr(Sigma_W), 2*tr(Sigma_W^2))
  Effective dimension: d_eff = tr(Sigma_W)^2 / tr(Sigma_W^2)
  Anisotropy index: eta = d / d_eff = d * tr(Sigma_W^2) / tr(Sigma_W)^2
  (eta=1 isotropic, eta>>1 highly anisotropic / low-rank)

CORRECTED GUMBEL RACE LAW:
  A_eff(m, d_eff) = C_corr * sqrt(d_eff * log(m))
  logit(q) = A_eff * kappa_eff - log(K-1) + C
  where kappa_eff = tr(S_B) / tr(S_W)  [kappa unchanged, only A changes]

PREDICTION: Replacing d -> d_eff in the zero-param formula should improve
rho from 0.37 to >>0.7 on real neural network embeddings.

EXPERIMENT:
1. Extract Pythia-160m embeddings on CLINC150, all layers
2. Compute eta = d * tr(Sigma_W^2) / tr(Sigma_W)^2 at each layer
3. Compute d_eff = d / eta = tr(Sigma_W)^2 / tr(Sigma_W^2)
4. Test zero-param formula with d vs d_eff
5. Compare rho(zero_param_isotropic, logit_q) vs rho(zero_param_aniso, logit_q)
"""

import json
import sys
import gc
import time
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr, norm
from scipy.special import ndtri
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model
from hierarchical_datasets import load_hierarchical_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "EleutherAI/pythia-160m"
DATASETS = ["clinc", "trec"]
MAX_SAMPLES = 1500
BATCH_SIZE = 32


# ============================================================
# Anisotropy & effective dimension
# ============================================================

def compute_within_scatter_spectrum(X, labels):
    """
    Compute eigenvalue spectrum of within-class scatter matrix S_W.

    S_W = (1/n) * sum_k sum_{i in k} (x_i - mu_k)(x_i - mu_k)^T

    Returns: tr(S_W), tr(S_W^2), d_eff, eta
    """
    unique = np.unique(labels)
    n, d = X.shape

    # Center each class
    X_w = X.copy()
    for lbl in unique:
        m = labels == lbl
        X_w[m] -= X_w[m].mean(0)

    # tr(S_W) = tr(X_w^T X_w / n) = ||X_w||_F^2 / n
    tr_sw = float(np.sum(X_w ** 2)) / n

    # tr(S_W^2) = tr((X_w^T X_w / n)^2) = ||X_w^T X_w||_F^2 / n^2
    # = ||X_w X_w^T||_F^2 / n^4 * n^2 ... use Frobenius identity:
    # tr((A^T A)^2) = ||A^T A||_F^2 = sum_{ij} (X_w^T X_w)_{ij}^2
    # For d < n: X_w^T X_w is d x d, compute directly
    # For d > n: use X_w X_w^T (n x n) and trace relation

    if d <= min(2000, n):
        SwSw = (X_w.T @ X_w) / n    # [d, d]
        tr_sw2 = float(np.sum(SwSw ** 2)) / n   # tr(S_W^2) = ||S_W||_F^2
    else:
        # Gram matrix: G = X_w X_w^T / n, tr(G^2) = tr(S_W^2) (same eigenvalues)
        # But G is n x n — too large. Use SVD instead.
        # tr(S_W^2) = sum_i sv_i^4 / n^2
        sv = np.linalg.svd(X_w / np.sqrt(n), compute_uv=False)
        tr_sw2 = float(np.sum(sv ** 4))

    d_eff = (tr_sw ** 2) / (tr_sw2 + 1e-10)
    eta = d / d_eff if d_eff > 0 else d   # anisotropy index (1 = isotropic)
    return tr_sw, tr_sw2, float(d_eff), float(eta)


# ============================================================
# Zero-param probit formula (isotropic)
# ============================================================

def zero_param_probit_isotropic(kappa, K, d, m, n_per_class):
    """
    Isotropic Gaussian probit prediction.
    q = Phi(mu_M / sigma_M)
    For symmetric mean configuration with rank d:
    mu_M ~ kappa * d / (K * sqrt(d)) = kappa * sqrt(d) / K  [rough]
    sigma_M ~ sqrt(2d) / sqrt(K) * (from inter-class variance)

    More careful: use the EVT-derived formula.
    logit(q) = A(m,d) * kappa - log(K-1) + C
    A(m,d) = C_corr * sqrt(d * log(m))
    C_corr = 1.075 for m >= 50
    C(m,d) derived from beta_gumbel correction
    """
    C_corr = 1.075 if m >= 50 else 1.25
    A = C_corr * np.sqrt(d * np.log(max(m, 2)))
    C_offset = -0.5   # empirical intercept
    logit_q = A * kappa - np.log(max(K - 1, 1)) + C_offset
    return float(logit_q)


def zero_param_probit_anisotropic(kappa, K, d_eff, m, n_per_class):
    """
    Anisotropic correction: replace d -> d_eff in A formula.
    A_eff(m, d_eff) = C_corr * sqrt(d_eff * log(m))
    """
    C_corr = 1.075 if m >= 50 else 1.25
    A_eff = C_corr * np.sqrt(max(d_eff, 1.0) * np.log(max(m, 2)))
    C_offset = -0.5
    logit_q = A_eff * kappa - np.log(max(K - 1, 1)) + C_offset
    return float(logit_q)


# ============================================================
# Embedding extraction
# ============================================================

@torch.no_grad()
def extract_all_layers(model, tokenizer, texts):
    all_hidden = {}
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(DEVICE)
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

def main():
    print("=" * 70)
    print("ANISOTROPY CORRECTION: d_eff in Gumbel Race Law")
    print("=" * 70)

    print(f"\nLoading {MODEL_ID}...", flush=True)
    model, tokenizer, _, _ = load_model(MODEL_ID, device=DEVICE)
    model.eval()

    all_results = {}

    for dataset_name in DATASETS:
        print(f"\n{'='*70}\nDataset: {dataset_name}\n{'='*70}", flush=True)

        data = load_hierarchical_dataset(dataset_name)
        texts = data["texts"][:MAX_SAMPLES]
        labels = np.array(data["fine_labels"][:MAX_SAMPLES])
        K = len(np.unique(labels))
        n = len(texts)
        m = n // K   # approx samples per class

        print(f"  n={n}, K={K}, m~{m}", flush=True)

        layer_reps = extract_all_layers(model, tokenizer, texts)
        n_layers = len(layer_reps)

        # kNN split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(sss.split(np.zeros(n), labels))

        rows = []
        for layer_idx in range(n_layers):
            X = layer_reps[layer_idx]
            d = X.shape[1]

            # kNN quality
            knn_clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
            knn_clf.fit(X[tr_idx], labels[tr_idx])
            knn_acc = float(knn_clf.score(X[te_idx], labels[te_idx]))
            q = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)
            logit_q = float(np.log(q / (1 - q) + 1e-10)) if 0 < q < 1 else (3.0 if q >= 1 else -5.0)

            # kappa
            unique = np.unique(labels)
            grand = X.mean(0)
            tr_sb, tr_sw = 0.0, 0.0
            for lbl in unique:
                mk = labels == lbl
                Xk = X[mk]; muk = Xk.mean(0)
                tr_sb += float(mk.sum()) * float(((muk - grand)**2).sum())
                tr_sw += float(((Xk - muk)**2).sum())
            kappa = tr_sb / (tr_sw + 1e-10)

            # Anisotropy
            tr_sw_val, tr_sw2, d_eff, eta = compute_within_scatter_spectrum(X, labels)

            # Zero-param predictions
            logit_iso = zero_param_probit_isotropic(kappa, K, d, m, n // K)
            logit_aniso = zero_param_probit_anisotropic(kappa, K, d_eff, m, n // K)

            rows.append({
                "layer": layer_idx,
                "knn_acc": knn_acc,
                "q": float(q),
                "logit_q": logit_q,
                "kappa": float(kappa),
                "d": d,
                "d_eff": d_eff,
                "eta": eta,
                "logit_pred_iso": logit_iso,
                "logit_pred_aniso": logit_aniso,
            })

            if layer_idx % 3 == 0:
                print(f"  Layer {layer_idx:2d}: q={q:.3f} kappa={kappa:.3f} "
                      f"d_eff={d_eff:.1f} eta={eta:.2f} | "
                      f"logit_q={logit_q:.2f} iso={logit_iso:.2f} aniso={logit_aniso:.2f}",
                      flush=True)

        all_results[dataset_name] = rows

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("RESULTS: Isotropic vs Anisotropic Zero-Param Theory")
    print(f"{'='*70}")

    summary = {}
    for dataset_name, rows in all_results.items():
        logit_q = [r["logit_q"] for r in rows]
        logit_iso = [r["logit_pred_iso"] for r in rows]
        logit_aniso = [r["logit_pred_aniso"] for r in rows]
        eta_vals = [r["eta"] for r in rows]
        d_eff_vals = [r["d_eff"] for r in rows]

        rho_iso = float(spearmanr(logit_iso, logit_q).statistic)
        rho_aniso = float(spearmanr(logit_aniso, logit_q).statistic)
        mae_iso = float(np.mean(np.abs(np.array(logit_iso) - np.array(logit_q))))
        mae_aniso = float(np.mean(np.abs(np.array(logit_aniso) - np.array(logit_q))))
        mean_eta = float(np.mean(eta_vals))
        mean_d_eff = float(np.mean(d_eff_vals))

        print(f"\n  Dataset: {dataset_name}")
        print(f"  Mean eta (anisotropy) = {mean_eta:.2f}  (1=isotropic, >>1=concentrated)")
        print(f"  Mean d_eff = {mean_d_eff:.1f}  (vs d={rows[0]['d']})")
        print(f"  rho (isotropic zero-param) = {rho_iso:.4f}")
        print(f"  rho (anisotropic zero-param) = {rho_aniso:.4f}  [improvement: {rho_aniso - rho_iso:+.4f}]")
        print(f"  MAE (isotropic) = {mae_iso:.3f}")
        print(f"  MAE (anisotropic) = {mae_aniso:.3f}  [improvement: {mae_iso - mae_aniso:+.3f}]")

        summary[dataset_name] = {
            "rho_iso": rho_iso,
            "rho_aniso": rho_aniso,
            "mae_iso": mae_iso,
            "mae_aniso": mae_aniso,
            "mean_eta": mean_eta,
            "mean_d_eff": mean_d_eff,
            "n_layers": len(rows),
        }

    # SCORECARD
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")
    checks = []
    for dataset_name, s in summary.items():
        checks.append((
            f"Anisotropic > Isotropic rho [{dataset_name}]",
            s["rho_aniso"] > s["rho_iso"],
            f"aniso={s['rho_aniso']:.3f} > iso={s['rho_iso']:.3f}"
        ))
        checks.append((
            f"rho(anisotropic) > 0.5 [{dataset_name}]",
            s["rho_aniso"] > 0.5,
            f"rho={s['rho_aniso']:.3f}"
        ))

    passes = sum(1 for _, p, _ in checks if p)
    for crit, passed, val in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {crit}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    print(f"\n{'='*70}")
    print("IMPLICATION")
    print(f"{'='*70}")
    print("""
  If anisotropic correction improves rho significantly:
  -> The zero-param theory DOES work, but requires d_eff not d
  -> This is a THEORETICAL RESULT: d_eff = tr(Sigma_W)^2 / tr(Sigma_W^2) is the
     correct effective dimension for Euclidean kNN under anisotropic covariance
  -> This fills the gap in the Observable Order-Parameter Theorem (Theorem 5
     in research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md)
  -> Nobel-track: a universal formula with NO FREE PARAMS that predicts
     kNN quality from geometry alone, even for anisotropic neural networks
""")

    # Save
    out_path = RESULTS_DIR / "cti_anisotropy_correction.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "datasets": DATASETS,
            "summary": summary,
            "scorecard_passes": passes,
            "scorecard_total": len(checks),
            "per_dataset_rows": all_results,
        }, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
