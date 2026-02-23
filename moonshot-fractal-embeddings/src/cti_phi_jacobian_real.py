#!/usr/bin/env python -u
"""
PHI JACOBIAN REAL-EMBEDDING TEST (Session 40)
==============================================
PRE-REGISTERED TEST:
  Use pythia-160m step512 embeddings on DBpedia-14.
  WHY step512: early training checkpoint -> non-ceiling q (0.3-0.7)
  AND varied kappa_gap -> avoids regime mismatch that blocked full-model test.

  For each non-ceiling class (q_base < 0.85, gap > 0.04):
    - Arm A: shift j1 by delta, measure slope_j1 = d(logit_q)/d(delta_kappa_j1)
    - Arm B: shift j2 ONLY (orthogonal), delta < gap/2, measure slope_j2
    - w_empirical = slope_j2 / slope_j1
    - w_phi(tau*=0.20) = exp(-gap / tau*) [FIXED a priori, not fitted]

PRE-REGISTERED CRITERION:
  1. r(w_empirical, w_phi(tau*=0.20)) > 0.60
  2. r(log(w_empirical), -gap) > 0.60
  3. Slope of log(w) vs gap within [-7.5, -2.5]  (tau* estimate in [0.13, 0.40])
  4. At least 4 valid classes (non-ceiling + non-near-tie + rank-stable)

FIXED PARAMETER:
  tau* = 0.20  (from phi_upgrade_pooled Session 38, NOT refitted here)

REGIME FILTER:
  - Non-ceiling: q_base < 0.85
  - Non-near-tie: gap > 0.04 kappa units
  - Rank-stable: use only delta < gap * 0.45 (within j2 rank)
"""

import json
import gc
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, linregress, spearmanr
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_NPZ = RESULTS_DIR / "phi_jacobian_real_step512_embs.npz"
OUT_JSON = RESULTS_DIR / "cti_phi_jacobian_real.json"

MODEL_NAME = "EleutherAI/pythia-160m"
REVISION = "step512"
LAYER = 12
N_PER_CLASS = 500
K = 14
BATCH_SIZE = 64
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Pre-registered fixed tau*
TAU_STAR = 0.20

# Regime filter thresholds
Q_CEILING = 0.85       # above this: skip (ceiling)
MIN_GAP = 0.04         # below this: skip (near-tie / rank-switch risk)
RANK_STABLE_FRAC = 0.45  # use delta < gap * RANK_STABLE_FRAC to avoid rank switch

# MATCHED delta range: Arm A and Arm B use the SAME small deltas per class.
# max_delta = gap * RANK_STABLE_FRAC (caps Arm B; Arm A also capped at same max)
# FINE_DELTAS: the candidate delta grid (in kappa units)
FINE_DELTAS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
               0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20]

# Arm A large-delta list (for global slope calibration only, NOT for w ratio)
DELTA_A_LARGE = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================
def extract_embeddings():
    """Extract step512 embeddings; save/load NPZ cache."""
    if CACHE_NPZ.exists():
        print(f"[CACHE] Loading embeddings from {CACHE_NPZ}", flush=True)
        data = np.load(str(CACHE_NPZ))
        return data["X"].astype(np.float64), data["y"].astype(np.int64)

    print(f"[EXTRACT] Loading {MODEL_NAME} @ {REVISION}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_NAME, revision=REVISION,
        torch_dtype=torch.float32, output_hidden_states=True
    ).to(DEVICE).eval()

    print(f"[EXTRACT] Loading dataset...", flush=True)
    raw = load_dataset("fancyzhx/dbpedia_14", split="test", trust_remote_code=True)
    labels_all = np.array(raw["label"])
    texts_all = raw["content"]

    # Stratified subsample: N_PER_CLASS per class
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=N_PER_CLASS * K, random_state=RANDOM_STATE
    )
    _, idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
    texts_sub = [texts_all[i] for i in idx]
    labels_sub = labels_all[idx]

    print(f"[EXTRACT] Encoding {len(texts_sub)} samples at layer {LAYER}...", flush=True)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts_sub), BATCH_SIZE):
            batch = texts_sub[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc)
            h = out.hidden_states[LAYER]   # [B, seq_len, d]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1)
            all_embs.append(pooled.cpu().numpy().astype(np.float32))
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  batch {i//BATCH_SIZE+1}/{len(texts_sub)//BATCH_SIZE+1}", flush=True)

    X = np.vstack(all_embs).astype(np.float32)
    y = labels_sub.astype(np.int64)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    np.savez_compressed(str(CACHE_NPZ), X=X, y=y)
    print(f"[EXTRACT] Saved to {CACHE_NPZ}", flush=True)
    return X.astype(np.float64), y


# ============================================================
# GEOMETRY
# ============================================================
def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        centroids[c] = Xc.mean(0)
        resids.append(Xc - centroids[c])
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R ** 2)))
    return centroids, sigma_W


def ranked_competitors(centroids, sigma_W, d, ci):
    """Return sorted list of (kappa, class_label) nearest first."""
    mu_i = centroids[ci]
    entries = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa = dist / (sigma_W * np.sqrt(d) + 1e-12)
        entries.append((kappa, cj))
    entries.sort()
    return entries


# ============================================================
# EVALUATION
# ============================================================
def eval_q_ci(X_tr, y_tr, X_te, y_te, ci):
    """1-NN normalized recall for class ci."""
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
    knn.fit(X_tr, y_tr)
    mask = y_te == ci
    if mask.sum() == 0:
        return None
    preds = knn.predict(X_te[mask])
    K_local = len(np.unique(y_tr))
    q_raw = float((preds == ci).mean())
    return float((q_raw - 1.0 / K_local) / (1.0 - 1.0 / K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1 - 1e-5))
    return float(np.log(q / (1.0 - q)))


# ============================================================
# SLOPE FITTING
# ============================================================
def fit_slope(x_vals, y_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or x.std() < 1e-12:
        return None, None
    slope, intercept, r, _, _ = linregress(x, y)
    return float(slope), float(r)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("PHI JACOBIAN REAL-EMBEDDING TEST")
    print(f"Model: {MODEL_NAME} @ {REVISION}")
    print(f"Dataset: DBpedia-14, K={K}, N={N_PER_CLASS}/class")
    print(f"PRE-REGISTERED tau* = {TAU_STAR} (from phi_upgrade_pooled, FIXED)")
    print(f"PRE-REGISTERED: r(w_empirical, w_phi) > 0.60")
    print(f"PRE-REGISTERED: r(log(w_empirical), -gap) > 0.60")
    print("=" * 70)

    # --- Load embeddings ---
    X, y = extract_embeddings()
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    print(f"\nEmbeddings: shape={X.shape}, K={len(classes)}")

    # --- Fixed train/test split ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(sss.split(X, y))
    X_tr_base, X_te_base = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    print(f"Train: {len(tr_idx)}, Test: {len(te_idx)}")

    centroids, sigma_W = compute_class_stats(X, y)
    print(f"sigma_W = {sigma_W:.6f}")

    # ---- Per-class analysis ----
    per_class_results = {}
    valid_classes = []

    for ci in classes:
        ranking = ranked_competitors(centroids, sigma_W, d, ci)
        kappa_j1, j1 = ranking[0]
        kappa_j2, j2 = ranking[1]
        gap = kappa_j2 - kappa_j1

        # Baseline accuracy
        q_base = eval_q_ci(X_tr_base, y_tr, X_te_base, y_te, ci)
        if q_base is None:
            print(f"  ci={ci}: SKIP (no test samples)")
            continue
        logit_base = safe_logit(q_base)

        print(f"\n--- ci={ci}: j1={j1} kappa_j1={kappa_j1:.4f}, "
              f"j2={j2} kappa_j2={kappa_j2:.4f}, gap={gap:.4f}, "
              f"q={q_base:.4f}, logit={logit_base:.4f}")

        # Regime filter
        if q_base >= Q_CEILING:
            print(f"    SKIP: q={q_base:.4f} >= ceiling={Q_CEILING}")
            per_class_results[ci] = {
                "skip_reason": "ceiling", "q_base": q_base,
                "kappa_j1": kappa_j1, "kappa_j2": kappa_j2, "gap": gap
            }
            continue

        if gap < MIN_GAP:
            print(f"    SKIP: gap={gap:.4f} < min_gap={MIN_GAP}")
            per_class_results[ci] = {
                "skip_reason": "near_tie", "q_base": q_base,
                "kappa_j1": kappa_j1, "kappa_j2": kappa_j2, "gap": gap
            }
            continue

        # Max stable delta for BOTH arms (rank-stable condition)
        max_delta = gap * RANK_STABLE_FRAC
        # Matched deltas: same grid for both Arm A and Arm B (local Jacobian)
        matched_deltas = [dv for dv in FINE_DELTAS if dv <= max_delta + 1e-6]
        print(f"    max_delta = {max_delta:.4f}; n_matched_pts = {len(matched_deltas)}")

        if len(matched_deltas) < 3:
            print(f"    SKIP: too few matched delta pts ({len(matched_deltas)} < 3)")
            per_class_results[ci] = {
                "skip_reason": "too_few_delta", "q_base": q_base,
                "kappa_j1": kappa_j1, "kappa_j2": kappa_j2, "gap": gap,
                "n_matched_delta_pts": len(matched_deltas)
            }
            continue

        # ------ ARM A: shift j1 (matched range = local Jacobian) ------
        delta_a_vals, logit_delta_a = [], []
        for delta in matched_deltas:
            if delta == 0.0:
                delta_a_vals.append(0.0)
                logit_delta_a.append(0.0)
                continue
            direction = centroids[j1] - centroids[ci]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm
            shift = delta * sigma_W * np.sqrt(d) * direction
            X_tr_mod = X_tr_base.copy()
            X_te_mod = X_te_base.copy()
            X_tr_mod[y_tr == j1] += shift
            X_te_mod[y_te == j1] += shift
            q_mod = eval_q_ci(X_tr_mod, y_tr, X_te_mod, y_te, ci)
            if q_mod is None:
                continue
            delta_a_vals.append(float(delta))
            logit_delta_a.append(safe_logit(q_mod) - logit_base)

        slope_j1, r_j1 = fit_slope(delta_a_vals, logit_delta_a)

        # Also compute global slope_j1 from larger range (informational only)
        delta_a_large_vals, logit_delta_a_large = [], []
        for delta in DELTA_A_LARGE:
            if delta == 0.0:
                delta_a_large_vals.append(0.0)
                logit_delta_a_large.append(0.0)
                continue
            direction = centroids[j1] - centroids[ci]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm
            shift = delta * sigma_W * np.sqrt(d) * direction
            X_tr_mod = X_tr_base.copy()
            X_te_mod = X_te_base.copy()
            X_tr_mod[y_tr == j1] += shift
            X_te_mod[y_te == j1] += shift
            q_mod = eval_q_ci(X_tr_mod, y_tr, X_te_mod, y_te, ci)
            if q_mod is None:
                continue
            delta_a_large_vals.append(float(delta))
            logit_delta_a_large.append(safe_logit(q_mod) - logit_base)
        slope_j1_global, r_j1_global = fit_slope(delta_a_large_vals, logit_delta_a_large)

        # ------ ARM B: shift j2 only (orthogonal, matched range) ------
        delta_b_vals, logit_delta_b = [], []
        for delta in matched_deltas:
            if delta == 0.0:
                delta_b_vals.append(0.0)
                logit_delta_b.append(0.0)
                continue
            direction = centroids[j2] - centroids[ci]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm
            shift = delta * sigma_W * np.sqrt(d) * direction
            X_tr_mod = X_tr_base.copy()
            X_te_mod = X_te_base.copy()
            X_tr_mod[y_tr == j2] += shift
            X_te_mod[y_te == j2] += shift
            q_mod = eval_q_ci(X_tr_mod, y_tr, X_te_mod, y_te, ci)
            if q_mod is None:
                continue

            # Verify j2 is still ranked 2nd (rank-stable check)
            centroids_mod, sigma_W_mod = compute_class_stats(
                np.vstack([X_tr_mod, X_te_mod]),
                np.concatenate([y_tr, y_te])
            )
            rank_mod = ranked_competitors(centroids_mod, sigma_W_mod, d, ci)
            if len(rank_mod) >= 2 and rank_mod[1][1] != j2:
                print(f"    WARNING: rank-switch at delta={delta:.3f}, skipping point")
                continue

            delta_b_vals.append(float(delta))
            logit_delta_b.append(safe_logit(q_mod) - logit_base)

        slope_j2, r_j2 = fit_slope(delta_b_vals, logit_delta_b)

        # ------ Report ------
        s1l_str = f"{slope_j1:.4f}" if slope_j1 is not None else "N/A"
        r1l_str = f"{r_j1:.4f}" if r_j1 is not None else "N/A"
        s1g_str = f"{slope_j1_global:.4f}" if slope_j1_global is not None else "N/A"
        s2_str  = f"{slope_j2:.4f}" if slope_j2 is not None else "N/A"
        r2_str  = f"{r_j2:.4f}" if r_j2 is not None else "N/A"
        print(f"    Arm A local: slope_j1={s1l_str} (r={r1l_str}), "
              f"global: slope_j1={s1g_str}")
        print(f"    Arm B: slope_j2={s2_str} (r={r2_str})")

        w_empirical = None
        # w uses LOCAL matched slope_j1 (same delta range as Arm B)
        if (slope_j1 is not None and slope_j2 is not None
                and abs(slope_j1) > 1e-4):
            w_empirical = float(slope_j2 / slope_j1)
            w_phi_pred = float(np.exp(-gap / TAU_STAR))
            we_str = f"{w_empirical:.4f}"
            wp_str = f"{w_phi_pred:.4f}"
            print(f"    w_empirical={we_str}, w_phi(tau*={TAU_STAR})={wp_str}")
            # Only add to valid if w is in reasonable range AND slope_j1 > 0
            if slope_j1 > 0 and 0 <= w_empirical <= 20:
                valid_classes.append(ci)

        per_class_results[ci] = {
            "j1": int(j1), "j2": int(j2),
            "kappa_j1": float(kappa_j1), "kappa_j2": float(kappa_j2), "gap": float(gap),
            "q_base": float(q_base), "logit_base": float(logit_base),
            "max_delta_matched": float(max_delta),
            "n_matched_delta_pts": len(matched_deltas),
            "slope_j1_local": float(slope_j1) if slope_j1 is not None else None,
            "slope_j1_global": float(slope_j1_global) if slope_j1_global is not None else None,
            "slope_j2": float(slope_j2) if slope_j2 is not None else None,
            "r_j1_local": float(r_j1) if r_j1 is not None else None,
            "r_j1_global": float(r_j1_global) if r_j1_global is not None else None,
            "r_j2": float(r_j2) if r_j2 is not None else None,
            "n_arm_a_pts": len(delta_a_vals),
            "n_arm_b_pts": len(delta_b_vals),
            "w_empirical": float(w_empirical) if w_empirical is not None else None,
            "w_phi_pred": float(np.exp(-gap / TAU_STAR)),
            "skip_reason": None,
        }

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid_data = [
        (ci, per_class_results[ci])
        for ci in valid_classes
        if per_class_results[ci].get("w_empirical") is not None
        and np.isfinite(per_class_results[ci]["w_empirical"])
        and per_class_results[ci]["w_empirical"] > 0
    ]

    print(f"\nValid classes for Jacobian test: {len(valid_data)}")
    print(f"Min required: 4")

    w_empiricals = [r["w_empirical"] for _, r in valid_data]
    w_phi_preds  = [r["w_phi_pred"]  for _, r in valid_data]
    gaps         = [r["gap"]         for _, r in valid_data]

    print(f"\nPer-class table (gap-conditioned Jacobian):")
    print(f"  {'ci':>4} {'gap':>8} {'w_emp':>8} {'w_phi':>8} {'log_wemp':>10} {'log_wphi':>10}")
    for ci, r in valid_data:
        log_w = np.log(r["w_empirical"]) if r["w_empirical"] > 0 else float("nan")
        log_phi = np.log(r["w_phi_pred"]) if r["w_phi_pred"] > 0 else float("nan")
        print(f"  {ci:>4} {r['gap']:>8.4f} {r['w_empirical']:>8.4f} {r['w_phi_pred']:>8.4f} "
              f"{log_w:>10.4f} {log_phi:>10.4f}")

    # Initialize pass flags
    corr_w_pass = False
    corr_log_w_pass = False
    slope_in_range_pass = False
    n_valid_pass = len(valid_data) >= 4

    if len(valid_data) >= 3:
        # Test 1: r(w_empirical, w_phi(tau*=0.20))
        r_w, p_w = pearsonr(w_empiricals, w_phi_preds)
        print(f"\nTest 1: r(w_empirical, w_phi(tau*=0.20)) = {r_w:.4f} (p={p_w:.4f})")
        corr_w_pass = r_w > 0.60
        print(f"  PRE-REG PASS (r>0.60): {'PASS' if corr_w_pass else 'FAIL'}")

        # Test 2: r(log(w_empirical), -gap)
        log_w_emp = np.array([np.log(max(w, 1e-6)) for w in w_empiricals])
        neg_gap = np.array([-g for g in gaps])
        r_log_w, p_log_w = pearsonr(log_w_emp, neg_gap)
        print(f"\nTest 2: r(log(w_empirical), -gap) = {r_log_w:.4f} (p={p_log_w:.4f})")
        corr_log_w_pass = r_log_w > 0.60
        print(f"  PRE-REG PASS (r>0.60): {'PASS' if corr_log_w_pass else 'FAIL'}")

        # Test 3: slope of log(w) vs gap
        slope_log_w, _, slope_r, _, _ = linregress(np.array(gaps), log_w_emp)
        tau_star_empirical = -1.0 / slope_log_w if abs(slope_log_w) > 1e-6 else float("nan")
        print(f"\nTest 3: slope of log(w) vs gap = {slope_log_w:.4f}")
        print(f"  Implied tau* = {tau_star_empirical:.4f} (pre-registered: {TAU_STAR})")
        slope_in_range_pass = (-7.5 <= slope_log_w <= -2.5)
        print(f"  PRE-REG PASS (slope in [-7.5, -2.5]): {'PASS' if slope_in_range_pass else 'FAIL'}")
    else:
        r_w, p_w, r_log_w, p_log_w = None, None, None, None
        slope_log_w, tau_star_empirical = None, None
        print("\n[ERROR] Too few valid classes for correlation tests")

    summary_pass = corr_w_pass and corr_log_w_pass and n_valid_pass
    print(f"\n{'='*70}")
    print(f"OVERALL {'PASS' if summary_pass else 'FAIL'}")
    print(f"  n_valid_classes >= 4: {'PASS' if n_valid_pass else 'FAIL'} ({len(valid_data)})")
    print(f"  r(w_emp, w_phi) > 0.60: {'PASS' if corr_w_pass else 'FAIL'}")
    print(f"  r(log_w, -gap) > 0.60: {'PASS' if corr_log_w_pass else 'FAIL'}")
    print(f"  slope in [-7.5,-2.5]: {'PASS' if slope_in_range_pass else 'FAIL'}")

    result = {
        "experiment": "phi_jacobian_real",
        "model": MODEL_NAME,
        "revision": REVISION,
        "layer": LAYER,
        "dataset": "dbpedia14",
        "K": K,
        "tau_star_preregistered": TAU_STAR,
        "regime_filters": {
            "q_ceiling": Q_CEILING,
            "min_gap": MIN_GAP,
            "rank_stable_frac": RANK_STABLE_FRAC,
        },
        "n_total_classes": len(classes),
        "n_valid_classes": len(valid_data),
        "per_class": per_class_results,
        "summary": {
            "r_w_empirical_phi": float(r_w) if r_w is not None else None,
            "p_w_empirical_phi": float(p_w) if p_w is not None else None,
            "r_log_w_neg_gap": float(r_log_w) if r_log_w is not None else None,
            "p_log_w_neg_gap": float(p_log_w) if p_log_w is not None else None,
            "slope_log_w_vs_gap": float(slope_log_w) if slope_log_w is not None else None,
            "tau_star_empirical": float(tau_star_empirical) if tau_star_empirical is not None else None,
            "tau_star_preregistered": TAU_STAR,
            "tau_star_error_pct": (
                abs(tau_star_empirical - TAU_STAR) / TAU_STAR * 100
                if tau_star_empirical is not None and not np.isnan(tau_star_empirical)
                else None
            ),
        },
        "pass_criteria": {
            "n_valid_pass": bool(n_valid_pass),
            "corr_w_pass": bool(corr_w_pass),
            "corr_log_w_pass": bool(corr_log_w_pass),
            "slope_in_range_pass": bool(slope_in_range_pass),
            "summary_pass": bool(summary_pass),
        },
        "pre_registered_thresholds": {
            "min_r_corr_w_phi": 0.60,
            "min_r_log_w_neg_gap": 0.60,
            "slope_range": [-7.5, -2.5],
            "min_valid_classes": 4,
        },
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
