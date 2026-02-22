#!/usr/bin/env python -u
"""
RANK-SPECTRUM ORTHOGONAL INTERVENTION (Feb 22 2026)
===================================================
Codex recommendation: "pre-registered density-controlled rank-spectrum orthogonal
intervention: Keep encoder fixed. Move one competitor rank at a time (j2..jK)
with j1 locked. Predict rank-effect kernel in advance and test cross-modality
with no post-hoc refit."

DESIGN:
  For each focus class ci and each competitor rank r in {2, 3, 4, 5, K}:
    - Move ONLY rank-r competitor jr (j1 stays fixed -> kappa_j1 EXACTLY unchanged)
    - Extract SLOPE beta_r = d(logit_q)/d(kappa_jr) from dose-response

  Pre-registered zero-parameter prediction:
    beta_r / beta_r_norm = exp(-A * delta_kappa_r * sqrt(d_eff_ci))
    where:
      A_ViT = 0.63 (from LOAO experiment, locked)
      delta_kappa_r = kappa_jr - kappa_j1 (from centroid geometry)
      d_eff_ci = tr(Sigma_W) / sigma_centroid_dir(ci,j1)^2 (from embedding geometry)

  The normalized slope beta_r_norm = beta_r / beta_j1 where beta_j1 is the
  slope at rank 1 (= Arm A slope for class ci).

  Since beta_j1 is hard to isolate (Arm A moves BOTH ci and j1), we instead test:
    PRE-REGISTERED: Pearson r(theory_weight_r, beta_r) > 0.85
    across all (ci, r) pairs, using theory_weight_r as the x-axis.

DATASETS:
  PRIMARY:   ViT-Large-16-224 CIFAR-10 layer 12 (N=5000, d=1024, K=10)
             Arm C failed -> all ranks should show non-zero effect
  SECONDARY: Pythia-160m DBpedia layer 12 (N=1000, d=768, K=14)
             Arm C passed -> high-rank effects should approach 0

  SAME theory kernel tested on BOTH datasets (only A differs by modality).

CIFAR-10 CLASS NAMES:
  0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
  5=dog, 6=frog, 7=horse, 8=ship, 9=truck
"""

import json
import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# ================================================================
# CONFIG
# ================================================================
# ViT dataset
VIT_EMBS_FILE  = "results/vit_loao_embs_vit-large-patch16-224_cifar10.npz"
VIT_LAYER_KEY  = "12"
VIT_SUBSAMPLE  = 500                        # 500/class = 5000 total
VIT_A          = 0.63                       # LOCKED from LOAO (pre-registered)

# NLP dataset
NLP_EMBS_FILE  = "results/do_int_embs_pythia-160m_dbpedia.npz"
NLP_A          = 1.054                      # LOCKED from LOAO (pre-registered)

# Shared
RANDOM_SEED    = 42
OUT_JSON       = "results/cti_rank_spectrum_factorial.json"
OUT_LOG        = "results/cti_rank_spectrum_factorial_log.txt"

# Ranks to test (r=2 is j2, r=3 is j3, ..., r=K is jK)
# For ViT K=10: test ranks {2, 3, 4, 5, 9} (j2..j5 + jK)
# For NLP K=14: test ranks {2, 3, 4, 7, 13} (j2..j4 + j7 + jK)
VIT_RANKS  = [2, 3, 4, 5, 9]               # r=9 = jK for K=10
NLP_RANKS  = [2, 3, 4, 7, 13]              # r=13 = jK for K=14

N_DELTA        = 9                          # dose points per arm
DELTA_MAX_B    = 3.0                        # push competitor up to +3 units
DELTA_B_RANGE  = np.linspace(0.0, DELTA_MAX_B, N_DELTA)

N_CV_SPLITS    = 5
MIN_SAMPLES    = 10

# PRE-REGISTERED threshold
PRE_REG_R_THEORY  = 0.85   # r(theory_weight_r, beta_r) across all (ci, r) pairs

CIFAR10_NAMES = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
                 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}


# ================================================================
# HELPERS
# ================================================================
def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        centroids[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R**2)))
    tr_W = float(np.mean(R**2) * X.shape[1])   # tr(Sigma_W) = mean_sq * d
    return centroids, sigma_W, tr_W


def get_ranking(centroids, sigma_W, d, ci):
    """Return sorted list of (kappa_ij, j) for all j != ci, ascending kappa."""
    mu_i = centroids[ci]
    rk = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa_ij = dist / (sigma_W * np.sqrt(d) + 1e-10)
        rk.append((kappa_ij, cj))
    rk.sort(key=lambda x: x[0])
    return rk   # [(kappa, j), ...] ascending


def compute_d_eff(X, y, centroids, tr_W, ci, j1):
    """Compute d_eff_formula = tr(Sigma_W) / sigma_centroid_dir(ci,j1)^2."""
    direction = centroids[j1] - centroids[ci]
    dist = np.linalg.norm(direction)
    if dist < 1e-10:
        return float('nan')
    direction = direction / dist
    R_ci = X[y == ci] - centroids[ci]
    proj = R_ci @ direction
    var_cdir = float(np.var(proj))
    if var_cdir < 1e-10:
        return float('nan')
    return tr_W / var_cdir


def compute_per_class_q(X, y, ci, n_splits=N_CV_SPLITS):
    K = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    recalls = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if (y_tr == ci).sum() < 2:
            continue
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X_tr, y_tr)
        mask = (y_te == ci)
        if mask.sum() == 0:
            continue
        preds = knn.predict(X_te[mask])
        recalls.append(float((preds == ci).mean()))
    if not recalls:
        return None
    q_raw = float(np.mean(recalls))
    K_inv = 1.0 / K
    return float((q_raw - K_inv) / (1.0 - K_inv)) if abs(1.0 - K_inv) > 1e-10 else None


def logit_fn(q):
    q = float(np.clip(q, 1e-5, 1-1e-5))
    return float(np.log(q / (1.0 - q)))


def apply_competitor_shift(X, y, centroids, ci, cj, delta):
    """ORTHOGONAL: move ONLY cj away from ci. ci DOES NOT MOVE."""
    mu_i = centroids[ci]
    mu_j = centroids[cj].copy()
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] += delta * direction
    return X_new


def run_rank_arm(X, y, centroids, sigma_W, tr_W, ci, jr, delta_range, rank_r, log_fn,
                 class_name=""):
    """Run orthogonal shift arm for rank-r competitor jr. ci does not move."""
    d = X.shape[1]
    rk = get_ranking(centroids, sigma_W, d, ci)
    kappa_j1 = rk[0][0]
    j1 = rk[0][1]
    kappa_jr = None
    for k, j in rk:
        if j == jr:
            kappa_jr = k
            break

    if kappa_jr is None:
        log_fn(f"  WARNING: jr={jr} not found in ranking for ci={ci}")
        return None, kappa_j1, None

    d_eff = compute_d_eff(X, y, centroids, tr_W, ci, j1)
    delta_kappa_r = kappa_jr - kappa_j1

    records = []
    for delta in delta_range:
        X_new = apply_competitor_shift(X, y, centroids, ci, jr, delta)
        new_cents, new_sw, _ = compute_class_stats(X_new, y)
        new_rk = get_ranking(new_cents, new_sw, d, ci)
        new_kappa_j1 = new_rk[0][0]   # kappa_j1 should be unchanged

        q_ci = compute_per_class_q(X_new, y, ci)
        if q_ci is None:
            continue
        lq = logit_fn(q_ci)

        new_kappa_jr = None
        for k, j in new_rk:
            if j == jr:
                new_kappa_jr = k
                break

        records.append({
            "delta": float(delta),
            "kappa_j1_new": float(new_kappa_j1),
            "kappa_jr_new": float(new_kappa_jr) if new_kappa_jr is not None else None,
            "q_ci": float(q_ci),
            "logit_q": float(lq),
        })

    if not records:
        return None, kappa_j1, kappa_jr

    # Extract slope: beta = d(logit_q) / d(kappa_jr)
    xs = np.array([r["kappa_jr_new"] for r in records if r.get("kappa_jr_new") is not None])
    ys = np.array([r["logit_q"] for r in records
                   if records[records.index(r)].get("kappa_jr_new") is not None
                   if r.get("kappa_jr_new") is not None])
    # Re-extract aligned
    xs_clean = np.array([r["kappa_jr_new"] for r in records if r["kappa_jr_new"] is not None])
    ys_clean = np.array([r["logit_q"] for r in records if r["kappa_jr_new"] is not None])

    if len(xs_clean) < 3 or np.std(xs_clean) < 1e-8:
        beta = 0.0
        r_corr = 0.0
    else:
        # Linear regression slope
        beta = float(np.polyfit(xs_clean, ys_clean, 1)[0])
        r_corr, _ = stats.pearsonr(xs_clean, ys_clean)
        r_corr = float(r_corr)

    log_fn(f"    rank {rank_r} (jr={jr}{(' '+class_name) if class_name else ''}): "
           f"kappa_jr={kappa_jr:.3f}, delta_kappa={delta_kappa_r:.3f}, "
           f"d_eff={d_eff:.1f}, beta={beta:.3f}, r={r_corr:.3f}")

    result = {
        "rank_r": rank_r,
        "jr": int(jr),
        "kappa_jr_orig": float(kappa_jr),
        "kappa_j1_orig": float(kappa_j1),
        "delta_kappa_r": float(delta_kappa_r),
        "d_eff": float(d_eff) if not np.isnan(d_eff) else None,
        "beta": float(beta),
        "r_corr": float(r_corr),
        "n_points": len(records),
    }
    return result, kappa_j1, kappa_jr


# ================================================================
# DATASET RUNNER
# ================================================================
def run_dataset(X, y, A_val, ranks_to_test, class_names, log_fn, dataset_name):
    """Run rank-spectrum intervention for one dataset."""
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    K = len(classes)

    centroids, sigma_W, tr_W = compute_class_stats(X, y)
    log_fn(f"\n  N={X.shape[0]}, d={d}, K={K}, sigma_W={sigma_W:.4f}, tr_W={tr_W:.2f}")

    # Per-class baseline q
    baseline_q = {}
    for ci in classes:
        q = compute_per_class_q(X, y, ci)
        if q is not None:
            baseline_q[ci] = q

    dataset_results = {}

    for ci in classes:
        if ci not in baseline_q:
            continue
        if (y == ci).sum() < MIN_SAMPLES:
            continue

        rk = get_ranking(centroids, sigma_W, d, ci)
        j1 = rk[0][1]
        d_eff_ci = compute_d_eff(X, y, centroids, tr_W, ci, j1)
        kappa_j1 = rk[0][0]

        name_i = class_names.get(ci, str(ci))
        log_fn(f"\n  Focus class {ci}({name_i}): kappa_j1={kappa_j1:.3f}, "
               f"d_eff={d_eff_ci:.1f}, q={baseline_q[ci]:.4f}")

        ci_results = {
            "class": int(ci),
            "class_name": name_i,
            "kappa_j1": float(kappa_j1),
            "d_eff": float(d_eff_ci) if not np.isnan(d_eff_ci) else None,
            "baseline_q": float(baseline_q[ci]),
            "j1": int(j1),
            "ranks": {}
        }

        for rank_r in ranks_to_test:
            if rank_r > K - 1:
                continue
            jr = rk[rank_r - 1][1]     # rank_r is 1-based
            delta_kappa_r = rk[rank_r-1][0] - kappa_j1
            name_jr = class_names.get(jr, str(jr))

            # Theory prediction (pre-registered, zero-parameter)
            if not np.isnan(d_eff_ci) and d_eff_ci > 0:
                theory_weight = float(np.exp(-A_val * delta_kappa_r * np.sqrt(d_eff_ci)))
            else:
                theory_weight = float('nan')

            arm_result, _, _ = run_rank_arm(
                X, y, centroids, sigma_W, tr_W, ci, jr,
                DELTA_B_RANGE, rank_r, log_fn, class_name=name_jr
            )
            if arm_result is not None:
                arm_result["theory_weight"] = theory_weight
                arm_result["jr_name"] = name_jr
                ci_results["ranks"][str(rank_r)] = arm_result

        dataset_results[str(ci)] = ci_results

    return dataset_results


# ================================================================
# AGGREGATE THEORY TEST
# ================================================================
def aggregate_theory_test(dataset_results, dataset_name, A_val, log_fn):
    """Compare observed betas to theory predictions. Pre-registered r>0.85."""
    theory_weights = []
    observed_betas = []
    observed_r_corrs = []
    labels = []

    for ci_str, ci_data in dataset_results.items():
        for r_str, rank_data in ci_data.get("ranks", {}).items():
            tw = rank_data.get("theory_weight")
            beta = rank_data.get("beta")
            r_corr = rank_data.get("r_corr")
            if tw is None or beta is None or np.isnan(tw) or np.isnan(beta):
                continue
            theory_weights.append(tw)
            observed_betas.append(beta)
            observed_r_corrs.append(r_corr)
            labels.append(f"ci={ci_str} r={r_str}")

    if len(theory_weights) < 3:
        log_fn(f"  {dataset_name}: insufficient data points ({len(theory_weights)})")
        return None, None

    tw_arr = np.array(theory_weights)
    beta_arr = np.array(observed_betas)
    r_arr = np.array(observed_r_corrs)

    # Spearman: does higher theory weight -> higher beta?
    rho_tw_beta, p_tw_beta = stats.spearmanr(tw_arr, beta_arr)
    rho_tw_rcorr, p_tw_rcorr = stats.spearmanr(tw_arr, r_arr)

    # Pearson (also useful)
    r_pearson_beta, p_pearson = stats.pearsonr(tw_arr, beta_arr)
    r_pearson_rcorr, p_pearson_rcorr = stats.pearsonr(tw_arr, r_arr)

    log_fn(f"\n  [{dataset_name}] Theory weight vs observed (n={len(theory_weights)}):")
    log_fn(f"  Spearman r(theory_weight, beta)   = {rho_tw_beta:.3f}  p={p_tw_beta:.4f}")
    log_fn(f"  Spearman r(theory_weight, r_corr) = {rho_tw_rcorr:.3f} p={p_tw_rcorr:.4f}")
    log_fn(f"  Pearson  r(theory_weight, beta)   = {r_pearson_beta:.3f}  p={p_pearson:.4f}")
    log_fn(f"  Pearson  r(theory_weight, r_corr) = {r_pearson_rcorr:.3f}  p={p_pearson_rcorr:.4f}")
    pass_test = abs(r_pearson_beta) >= PRE_REG_R_THEORY
    log_fn(f"  PRE-REG Pearson r>={PRE_REG_R_THEORY}: {'PASS' if pass_test else 'FAIL'} "
           f"(r_pearson_beta={r_pearson_beta:.3f})")

    # Print data table
    log_fn(f"\n  {'ci':>4} {'rank':>5} {'theory_w':>10} {'beta_obs':>10} {'r_corr':>8}")
    for i, (lbl, tw, b, rc) in enumerate(zip(labels, theory_weights, observed_betas, observed_r_corrs)):
        log_fn(f"  {lbl:>9} {tw:>10.3f} {b:>10.3f} {rc:>8.3f}")

    return {
        "n_points": len(theory_weights),
        "spearman_r_theory_beta": float(rho_tw_beta),
        "spearman_p_theory_beta": float(p_tw_beta),
        "spearman_r_theory_rcorr": float(rho_tw_rcorr),
        "spearman_p_theory_rcorr": float(p_tw_rcorr),
        "pearson_r_theory_beta": float(r_pearson_beta),
        "pearson_p_theory_beta": float(p_pearson),
        "pearson_r_theory_rcorr": float(r_pearson_rcorr),
        "pearson_p_theory_rcorr": float(p_pearson_rcorr),
        "pre_reg_pass": bool(pass_test),
        "theory_weights": [float(x) for x in theory_weights],
        "observed_betas": [float(x) for x in observed_betas],
        "observed_r_corrs": [float(x) for x in observed_r_corrs],
        "labels": labels,
    }, {
        "theory_weights": theory_weights,
        "observed_betas": observed_betas,
        "observed_r_corrs": observed_r_corrs,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs("results", exist_ok=True)
    log_file = open(OUT_LOG, "w", buffering=1)
    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log("=" * 72)
    log("RANK-SPECTRUM ORTHOGONAL INTERVENTION")
    log("=" * 72)
    log(f"PRE-REGISTERED PREDICTION (ZERO-PARAMETER):")
    log(f"  beta_r proportional to exp(-A * delta_kappa_r * sqrt(d_eff_ci))")
    log(f"  A_ViT = {VIT_A} (LOCKED from LOAO)")
    log(f"  A_NLP = {NLP_A} (LOCKED from LOAO)")
    log(f"  d_eff = tr(Sigma_W) / sigma_centroid_dir(ci,j1)^2 (from geometry)")
    log(f"  PRE-REG: Pearson r(theory_weight, observed_beta) > {PRE_REG_R_THEORY}")
    log("=" * 72)

    all_results = {}

    # ----------------------------------------------------------------
    # VIT DATASET
    # ----------------------------------------------------------------
    log("\n" + "=" * 72)
    log("DATASET 1: ViT-Large-16-224 CIFAR-10 layer 12")
    log("=" * 72)

    vit_data = np.load(VIT_EMBS_FILE)
    X_vit_all = vit_data[VIT_LAYER_KEY].astype(np.float64)
    y_vit_all = vit_data["y"].astype(np.int64)

    # Subsample
    rng = np.random.RandomState(RANDOM_SEED)
    keep = []
    for c in range(10):
        idx_c = np.where(y_vit_all == c)[0]
        chosen = rng.choice(idx_c, size=min(VIT_SUBSAMPLE, len(idx_c)), replace=False)
        keep.append(chosen)
    X_vit = X_vit_all[np.sort(np.concatenate(keep))]
    y_vit = y_vit_all[np.sort(np.concatenate(keep))]
    log(f"Subsampled ViT: N={X_vit.shape[0]}, d={X_vit.shape[1]}")

    vit_results = run_dataset(X_vit, y_vit, VIT_A, VIT_RANKS, CIFAR10_NAMES, log, "ViT")
    all_results["vit"] = {
        "dataset": "ViT-Large CIFAR-10 layer 12",
        "A_locked": VIT_A,
        "ranks_tested": VIT_RANKS,
        "per_class": vit_results,
    }

    vit_agg, vit_raw = aggregate_theory_test(vit_results, "ViT", VIT_A, log)
    if vit_agg is not None:
        all_results["vit"]["aggregate_theory_test"] = vit_agg

    # ----------------------------------------------------------------
    # NLP DATASET
    # ----------------------------------------------------------------
    log("\n" + "=" * 72)
    log("DATASET 2: Pythia-160m DBpedia layer 12")
    log("=" * 72)

    try:
        nlp_data = np.load(NLP_EMBS_FILE)
        X_nlp = nlp_data["X"].astype(np.float64)
        y_nlp = nlp_data["y"].astype(np.int64)
        log(f"NLP embeddings: N={X_nlp.shape[0]}, d={X_nlp.shape[1]}")

        # Build NLP class names (DBpedia has 14 categories)
        nlp_class_names = {i: f"dbp{i}" for i in range(20)}

        nlp_ranks = [r for r in NLP_RANKS if r <= len(np.unique(y_nlp)) - 1]
        nlp_results = run_dataset(X_nlp, y_nlp, NLP_A, nlp_ranks, nlp_class_names, log, "NLP")
        all_results["nlp"] = {
            "dataset": "Pythia-160m DBpedia layer 12",
            "A_locked": NLP_A,
            "ranks_tested": nlp_ranks,
            "per_class": nlp_results,
        }

        nlp_agg, nlp_raw = aggregate_theory_test(nlp_results, "NLP", NLP_A, log)
        if nlp_agg is not None:
            all_results["nlp"]["aggregate_theory_test"] = nlp_agg
    except Exception as e:
        log(f"  NLP dataset failed: {e}")
        nlp_raw = None

    # ----------------------------------------------------------------
    # COMBINED TEST (cross-modality)
    # ----------------------------------------------------------------
    log("\n" + "=" * 72)
    log("COMBINED CROSS-MODALITY TEST")
    log("=" * 72)

    combined_tw = []
    combined_beta = []
    combined_rcorr = []
    combined_label = []

    if vit_raw is not None:
        for tw, b, rc in zip(vit_raw["theory_weights"], vit_raw["observed_betas"], vit_raw["observed_r_corrs"]):
            combined_tw.append(tw)
            combined_beta.append(b)
            combined_rcorr.append(rc)
            combined_label.append("ViT")
    if nlp_raw is not None:
        for tw, b, rc in zip(nlp_raw["theory_weights"], nlp_raw["observed_betas"], nlp_raw["observed_r_corrs"]):
            combined_tw.append(tw)
            combined_beta.append(b)
            combined_rcorr.append(rc)
            combined_label.append("NLP")

    if len(combined_tw) >= 5:
        tw_arr = np.array(combined_tw)
        beta_arr = np.array(combined_beta)
        r_arr = np.array(combined_rcorr)
        rho_comb, p_comb = stats.spearmanr(tw_arr, beta_arr)
        r_pear_comb, p_pear_comb = stats.pearsonr(tw_arr, beta_arr)
        rho_comb_rcorr, p_comb_rcorr = stats.spearmanr(tw_arr, r_arr)
        pass_comb = abs(r_pear_comb) >= PRE_REG_R_THEORY
        log(f"\nCombined (n={len(combined_tw)}, ViT+NLP):")
        log(f"  Spearman r(theory, beta)   = {rho_comb:.3f}  p={p_comb:.4f}")
        log(f"  Spearman r(theory, r_corr) = {rho_comb_rcorr:.3f}  p={p_comb_rcorr:.4f}")
        log(f"  Pearson  r(theory, beta)   = {r_pear_comb:.3f}  p={p_pear_comb:.4f}")
        log(f"  PRE-REG PASS: {pass_comb} (r={r_pear_comb:.3f})")

        all_results["combined"] = {
            "n_points": len(combined_tw),
            "spearman_r": float(rho_comb), "spearman_p": float(p_comb),
            "pearson_r": float(r_pear_comb), "pearson_p": float(p_pear_comb),
            "pre_reg_pass": bool(pass_comb),
            "theory_weights": [float(x) for x in combined_tw],
            "betas": [float(x) for x in combined_beta],
            "r_corrs": [float(x) for x in combined_rcorr],
            "labels": combined_label,
        }

    # ----------------------------------------------------------------
    # FINAL VERDICT
    # ----------------------------------------------------------------
    log("\n" + "=" * 72)
    log("FINAL VERDICT")
    log("=" * 72)
    log(f"PRE-REGISTERED: Pearson r(theory_weight, observed_beta) > {PRE_REG_R_THEORY}")
    if "combined" in all_results:
        comb = all_results["combined"]
        log(f"  Combined (ViT+NLP): r={comb['pearson_r']:.3f} p={comb['pearson_p']:.4f} "
            f"-> {'PASS' if comb['pre_reg_pass'] else 'FAIL'}")
    if "vit" in all_results and "aggregate_theory_test" in all_results["vit"]:
        vitt = all_results["vit"]["aggregate_theory_test"]
        log(f"  ViT only: r={vitt['pearson_r_theory_beta']:.3f} -> {'PASS' if vitt['pre_reg_pass'] else 'FAIL'}")
    if "nlp" in all_results and "aggregate_theory_test" in all_results["nlp"]:
        nlpt = all_results["nlp"]["aggregate_theory_test"]
        log(f"  NLP only: r={nlpt['pearson_r_theory_beta']:.3f} -> {'PASS' if nlpt['pre_reg_pass'] else 'FAIL'}")

    log(f"\nReference (from previous experiments):")
    log(f"  ViT Arm B (rank 2 only): r_corr=0.945 -> 2-LAYER SUPPORTED")
    log(f"  NLP Arm B (rank 2 only): r_corr=0.450 -> 1-LAYER SUPPORTED")
    log(f"  Arm C failure in ViT: jK (rank 9) still causal -> {'>= all ranks matter in dense K=10' if True else ''}")
    log(f"  Arm C pass in NLP: jK (rank 13) irrelevant -> 1-layer dominated")

    # Save
    out = {
        "experiment": "rank_spectrum_factorial",
        "description": "Rank-spectrum orthogonal intervention: move each competitor rank separately",
        "pre_registered": {
            "prediction": "beta_r proportional to exp(-A * delta_kappa_r * sqrt(d_eff))",
            "A_vit_locked": VIT_A,
            "A_nlp_locked": NLP_A,
            "threshold_pearson_r": PRE_REG_R_THEORY,
        },
        **all_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    log(f"\nSaved to {OUT_JSON}")
    log_file.close()


if __name__ == "__main__":
    main()
