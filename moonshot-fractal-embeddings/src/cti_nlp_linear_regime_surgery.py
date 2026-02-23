"""
NLP Linear-Regime Global Surgery Test
Pre-registered: src/PREREGISTRATION_nlp_linear_regime_surgery.md
Pre-reg commit: 4a36d65

Tests whether the 1/d_eff attenuation holds for NLP embeddings in the linear regime
(kappa_eff ~ 1.0), as a regime-controlled follow-up to the sub-linear FAIL (commit 8ea7288).

Layer selection: pick the layer per model where |kappa_eff - 1.0| is minimized.
"""
import os
import json
import numpy as np
import torch
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_nlp_linear_regime_surgery.json")
LOG_PATH = os.path.join(RESULTS_DIR, "cti_nlp_linear_regime_surgery_log.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K = 20
A_RENORM_K20 = 1.0535
SURGERY_LEVELS = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
PREREGISTERED_COMMIT = "4a36d65"
KAPPA_EFF_TARGET = 1.0
KAPPA_EFF_VALID_LO = 0.7
KAPPA_EFF_VALID_HI = 1.5

N_SAMPLE = 5000
BATCH_SIZE = 64

MODELS = [
    {"name": "pythia-1b",   "hf": "EleutherAI/pythia-1b",   "layers": [4, 8, 12, 16]},
    {"name": "OLMo-1B-hf",  "hf": "allenai/OLMo-1B-hf",     "layers": [4, 8, 12, 16]},
    {"name": "pythia-410m", "hf": "EleutherAI/pythia-410m",  "layers": [3, 6, 9, 12]},
    {"name": "pythia-160m", "hf": "EleutherAI/pythia-160m",  "layers": [3, 6, 9, 12]},
]

log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)


def extract_embeddings(model_hf, layers, n_sample=N_SAMPLE, batch_size=BATCH_SIZE):
    """Extract embeddings at specified layers for 20newsgroups."""
    log(f"  Loading model {model_hf}...")
    tokenizer = AutoTokenizer.from_pretrained(model_hf)
    model = AutoModel.from_pretrained(model_hf, output_hidden_states=True)
    model.eval()
    model.to(DEVICE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"  Loading 20newsgroups...")
    ds = load_dataset("SetFit/20_newsgroups", split="test")
    texts = ds["text"]
    raw_labels = ds["label_text"] if "label_text" in ds.column_names else ds["label"]

    # Encode labels
    unique_labels = sorted(set(raw_labels))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    all_labels = np.array([label2id[l] for l in raw_labels], dtype=np.int64)

    # Stratified sample
    rng = np.random.RandomState(42)
    indices = np.arange(len(all_labels))
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=n_sample, random_state=42)
    try:
        idx, _ = next(sss.split(indices, all_labels))
    except Exception:
        idx = rng.choice(len(all_labels), size=n_sample, replace=False)
    idx = np.sort(idx)
    sample_texts = [texts[i] for i in idx]
    sample_labels = all_labels[idx]

    log(f"  Sampling {len(idx)} examples, K={len(unique_labels)}")

    # Extract embeddings
    layer_embeddings = {l: [] for l in layers}
    with torch.no_grad():
        for i in range(0, len(sample_texts), batch_size):
            batch_texts = sample_texts[i:i + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128,
                            return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hidden = out.hidden_states  # tuple of (batch, seq, d)
            # Mean pooling over non-padding tokens
            for target_layer in layers:
                h = hidden[target_layer]  # (batch, seq, d)
                mask = enc["attention_mask"].float().unsqueeze(-1)  # (batch, seq, 1)
                pooled = (h * mask).sum(1) / mask.sum(1)  # (batch, d)
                layer_embeddings[target_layer].append(pooled.cpu().float().numpy())
            if (i // batch_size) % 10 == 0:
                log(f"    Batch {i//batch_size + 1}/{(len(sample_texts)+batch_size-1)//batch_size}")

    # Stack and sanitize NaN
    for l in layers:
        X = np.vstack(layer_embeddings[l])
        # Replace NaN/Inf with 0
        X = np.where(np.isfinite(X), X, 0.0)
        layer_embeddings[l] = X

    del model
    torch.cuda.empty_cache()
    return layer_embeddings, sample_labels


# ============================================================
# Surgery functions (same as cti_global_vs_single_surgery.py)
# ============================================================

def compute_geometry(X_tr, y_tr):
    classes = np.unique(y_tr)
    K_actual = len(classes)
    N = len(X_tr)
    d = X_tr.shape[1]
    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])
    trW = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        Xc_c = Xc - centroids[c]
        trW += float(np.sum(Xc_c ** 2)) / N
    sigma_W_global = float(np.sqrt(trW / d))
    min_dist, min_i, min_j = float('inf'), 0, 1
    for i in range(K_actual):
        for j in range(i + 1, K_actual):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j
    delta_min = float(min_dist)
    kappa_nearest = float(delta_min / (sigma_W_global * np.sqrt(d) + 1e-10))
    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)
    sigma_centroid_sq = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        proj = Xc_c @ Delta_hat
        sigma_centroid_sq += (n_c / N) * float(np.mean(proj ** 2))
    d_eff_formula = float(trW / (sigma_centroid_sq + 1e-10))
    kappa_eff = kappa_nearest * float(np.sqrt(d_eff_formula))
    grand_mean = centroids.mean(0)
    C_c = (centroids - grand_mean).astype(np.float64)
    # Use randomized SVD via sklearn for robustness with high-d embeddings
    from sklearn.utils.extmath import randomized_svd
    n_components = min(K_actual - 1, d - 1)
    _, _, Vt = randomized_svd(C_c, n_components=n_components, random_state=42)
    Vt = Vt.astype(np.float32)
    n_sig = min(K_actual - 1, d, Vt.shape[0])
    P_B = Vt[:n_sig, :]
    trW_sig = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        Xc_proj = Xc_c @ P_B.T
        trW_sig += (n_c / N) * float(np.sum(Xc_proj ** 2)) / n_c
    trW_null = trW - trW_sig
    return dict(centroids=centroids, Delta_hat=Delta_hat, trW=trW,
                trW_sig=trW_sig, trW_null=trW_null, sigma_W_global=sigma_W_global,
                sigma_centroid_sq=sigma_centroid_sq, d_eff_formula=d_eff_formula,
                kappa_nearest=kappa_nearest, kappa_eff=kappa_eff,
                K_actual=K_actual, d=d, P_B=P_B, n_sig=n_sig)


def apply_single_surgery(X, y, geo, r):
    centroids, Delta_hat = geo['centroids'], geo['Delta_hat']
    trW, scsq = geo['trW'], geo['sigma_centroid_sq']
    classes = np.unique(y)
    min_r = float(scsq / (trW + 1e-10)) * 1.001
    r_eff = max(r, min_r)
    scale_along = 1.0 / float(np.sqrt(r_eff))
    num = trW - scsq / r_eff
    denom = trW - scsq
    scale_perp = float(np.sqrt(max(0.0, num / (denom + 1e-12))))
    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        z = X[mask] - centroids[c]
        proj = z @ Delta_hat
        z_along = proj[:, None] * Delta_hat[None, :]
        z_perp = z - z_along
        X_new[mask] = centroids[c] + scale_along * z_along + scale_perp * z_perp
    return X_new, float(scale_along), float(scale_perp)


def apply_global_surgery(X, y, geo, r):
    centroids, P_B = geo['centroids'], geo['P_B']
    trW, trW_sig, trW_null = geo['trW'], geo['trW_sig'], geo['trW_null']
    classes = np.unique(y)
    r_min_valid = trW_sig / (trW + 1e-10)
    scale_sig = 1.0 / float(np.sqrt(r))
    valid = (r >= r_min_valid)
    scale_null_sq = (trW - trW_sig / r) / trW_null if trW_null > 1e-10 else 1.0
    if scale_null_sq < 0:
        scale_null_sq = 0.0
    scale_null = float(np.sqrt(scale_null_sq))
    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        z = X[mask] - centroids[c]
        z_sig = (z @ P_B.T) @ P_B
        z_null = z - z_sig
        X_new[mask] = centroids[c] + scale_sig * z_sig + scale_null * z_null
    return X_new, float(scale_sig), float(scale_null), valid, float(r_min_valid)


def compute_q_knn(X_tr, y_tr, X_te, y_te):
    K_actual = len(np.unique(y_tr))
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn.fit(X_tr, y_tr)
    acc = float(np.mean(knn.predict(X_te) == y_te))
    q_norm = float(np.clip((acc - 1.0 / K_actual) / (1.0 - 1.0 / K_actual), 0.001, 0.999))
    return q_norm, acc


def safe_logit(q):
    q = float(np.clip(q, 0.001, 0.999))
    return float(np.log(q / (1 - q)))


def run_surgery_for_arch(arch_name, best_layer, X_full, y_full, arch_idx):
    log(f"\n{'='*60}")
    log(f"ARCH: {arch_name} (layer {best_layer})")
    log(f"{'='*60}")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=arch_idx)
    tr_idx, te_idx = next(sss.split(X_full, y_full))
    X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
    X_te, y_te = X_full[te_idx], y_full[te_idx]
    log(f"  Split: {len(X_tr)} train, {len(X_te)} test, K={K}, d={X_full.shape[1]}")
    geo = compute_geometry(X_tr, y_tr)
    d_eff = geo['d_eff_formula']
    kappa = geo['kappa_nearest']
    kappa_eff = geo['kappa_eff']
    log(f"  Geometry: d_eff={d_eff:.3f}, kappa={kappa:.4f}, kappa_eff={kappa_eff:.3f}")
    log(f"  Signal dims: n_sig={geo['n_sig']}, trW_sig={geo['trW_sig']:.4f}, trW_null={geo['trW_null']:.4f}")
    q_base, _ = compute_q_knn(X_tr, y_tr, X_te, y_te)
    logit_base = safe_logit(q_base)
    C_fit = logit_base - A_RENORM_K20 * kappa * float(np.sqrt(d_eff))
    log(f"  Baseline: q={q_base:.4f}, logit={logit_base:.4f}, C_fit={C_fit:.4f}")

    records = []
    for r in SURGERY_LEVELS:
        log(f"\n  [r={r:.2f}] ...")
        X_tr_s, sa, sp = apply_single_surgery(X_tr, y_tr, geo, r)
        X_te_s, _, _ = apply_single_surgery(X_te, y_te, geo, r)
        q_single, _ = compute_q_knn(X_tr_s, y_tr, X_te_s, y_te)
        delta_single = safe_logit(q_single) - logit_base
        geo_s = compute_geometry(X_tr_s, y_tr)
        kappa_chg_s = abs(geo_s['kappa_nearest'] - kappa) / (kappa + 1e-10) * 100

        X_tr_g, sg, sn, valid, r_min_v = apply_global_surgery(X_tr, y_tr, geo, r)
        X_te_g, _, _, _, _ = apply_global_surgery(X_te, y_te, geo, r)
        q_global, _ = compute_q_knn(X_tr_g, y_tr, X_te_g, y_te)
        delta_global = safe_logit(q_global) - logit_base
        geo_g = compute_geometry(X_tr_g, y_tr)
        kappa_chg_g = abs(geo_g['kappa_nearest'] - kappa) / (kappa + 1e-10) * 100

        if not valid:
            log(f"    WARNING: r={r} < r_min_valid={r_min_v:.3f}: global surgery invalid. Excluding.")

        delta_pred_full = A_RENORM_K20 * kappa * float(np.sqrt(d_eff)) * (float(np.sqrt(r)) - 1)
        delta_pred_att = delta_pred_full / d_eff

        ratio = float('nan')
        if abs(delta_single) > 1e-6:
            ratio = delta_global / delta_single

        log(f"    Single: q={q_single:.4f}, delta={delta_single:.4f}, kappa_chg={kappa_chg_s:.3f}%")
        log(f"    Global: q={q_global:.4f}, delta={delta_global:.4f}, kappa_chg={kappa_chg_g:.3f}%")
        log(f"    Pred(full)={delta_pred_full:.4f}, Pred(att)={delta_pred_att:.4f}")
        if not np.isnan(ratio):
            log(f"    Ratio global/single={ratio:.2f} (d_eff={d_eff:.2f})")

        records.append({
            'arch': arch_name, 'layer': best_layer, 'arch_idx': arch_idx, 'r': r,
            'q_base': q_base, 'logit_base': logit_base,
            'q_single': q_single, 'delta_single': delta_single, 'kappa_chg_single': kappa_chg_s,
            'q_global': q_global, 'delta_global': delta_global, 'kappa_chg_global': kappa_chg_g,
            'global_valid': valid, 'r_min_valid': float(r_min_v),
            'ratio_global_over_single': ratio,
            'd_eff_base': d_eff, 'kappa_base': kappa, 'kappa_eff_base': kappa_eff,
            'delta_pred_full': delta_pred_full, 'delta_pred_attenuated': delta_pred_att,
        })
    return records, d_eff


def main():
    log("=" * 70)
    log("NLP LINEAR-REGIME GLOBAL VS SINGLE SURGERY TEST")
    log("=" * 70)
    log(f"Pre-reg commit: {PREREGISTERED_COMMIT}")
    log(f"Target kappa_eff: {KAPPA_EFF_TARGET} ± valid range [{KAPPA_EFF_VALID_LO}, {KAPPA_EFF_VALID_HI}]")
    log(f"A_RENORM_K20 = {A_RENORM_K20}, K = {K}")
    log("")

    all_records = []
    arch_summaries = []

    for arch_idx, arch_info in enumerate(MODELS):
        arch_name = arch_info['name']
        model_hf = arch_info['hf']
        layers = arch_info['layers']

        log(f"\n{'='*60}")
        log(f"MODEL {arch_idx}: {arch_name}")
        log(f"{'='*60}")

        # Extract embeddings at all specified layers
        try:
            layer_embeddings, labels = extract_embeddings(model_hf, layers)
        except Exception as e:
            log(f"  ERROR extracting embeddings: {e}")
            continue

        # Find best layer (kappa_eff closest to 1.0)
        log(f"\n  Layer selection (target kappa_eff={KAPPA_EFF_TARGET}):")
        best_layer = None
        best_dist = float('inf')
        layer_stats = {}

        for layer in layers:
            X = layer_embeddings[layer]
            y = labels
            # Compute geometry on full dataset (rough estimate)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
            tr_idx, _ = next(sss.split(X, y))
            X_tr = X[tr_idx]
            y_tr = y[tr_idx]
            geo = compute_geometry(X_tr, y_tr)
            kappa = geo['kappa_nearest']
            d_eff = geo['d_eff_formula']
            kappa_eff = geo['kappa_eff']
            dist = abs(kappa_eff - KAPPA_EFF_TARGET)
            layer_stats[layer] = {'kappa': kappa, 'd_eff': d_eff, 'kappa_eff': kappa_eff}
            log(f"    Layer {layer}: kappa={kappa:.4f}, d_eff={d_eff:.2f}, kappa_eff={kappa_eff:.3f}")
            if dist < best_dist:
                best_dist = dist
                best_layer = layer

        best = layer_stats[best_layer]
        log(f"\n  Selected layer: {best_layer} (kappa_eff={best['kappa_eff']:.3f}, "
            f"d_eff={best['d_eff']:.2f})")

        if not (KAPPA_EFF_VALID_LO <= best['kappa_eff'] <= KAPPA_EFF_VALID_HI):
            log(f"  SKIP: kappa_eff={best['kappa_eff']:.3f} outside valid range "
                f"[{KAPPA_EFF_VALID_LO}, {KAPPA_EFF_VALID_HI}]. Marking as not-in-regime.")
            arch_summaries.append({
                'arch': arch_name, 'in_regime': False,
                'best_layer': best_layer, **best
            })
            continue

        # Run surgery on best layer
        X_best = layer_embeddings[best_layer]
        records, d_eff = run_surgery_for_arch(arch_name, best_layer, X_best, labels, arch_idx)
        all_records.extend(records)
        arch_summaries.append({
            'arch': arch_name, 'in_regime': True,
            'best_layer': best_layer, **best
        })

    # ================================================================
    # PRE-REGISTERED EVALUATION
    # ================================================================
    log(f"\n{'='*70}")
    log("PRE-REGISTERED EVALUATION")
    log(f"{'='*70}")

    excluded = [(r['arch'], r['r'], r['r_min_valid'])
                for r in all_records if not r['global_valid'] and r['r'] != 1.0]
    valid_nb = [r for r in all_records
                if r['global_valid'] and r['r'] != 1.0 and not np.isnan(r['ratio_global_over_single'])]

    if excluded:
        log(f"\nNOTE: {len(excluded)} records excluded (global surgery invalid): {excluded}")

    ratios = [r['ratio_global_over_single'] for r in valid_nb]
    d_effs = [r['d_eff_base'] for r in valid_nb]
    log(f"\nH1 (ratio in [d_eff/3, d_eff*3]):")
    log(f"  Ratios: {[round(x, 2) for x in ratios]}")
    log(f"  d_effs: {[round(x, 2) for x in d_effs]}")

    h1_pass = False
    if ratios:
        med_ratio = float(np.median(ratios))
        med_d_eff = float(np.median(d_effs))
        lo, hi = med_d_eff / 3, med_d_eff * 3
        h1_pass = lo <= med_ratio <= hi
        log(f"  Median ratio={med_ratio:.2f}, Median d_eff={med_d_eff:.2f}")
        log(f"  H1: {'PASS' if h1_pass else 'FAIL'} (interval [{lo:.2f}, {hi:.2f}])")

    h2_pairs = [(r['r'], r['delta_global']) for r in valid_nb]
    h2_correct = sum(1 for rr, dg in h2_pairs if (rr > 1 and dg > 0) or (rr < 1 and dg < 0))
    h2_pass = h2_correct >= max(1, int(0.75 * len(h2_pairs)))
    log(f"\nH2: direction correct {h2_correct}/{len(h2_pairs)}: {'PASS' if h2_pass else 'FAIL'}")

    single_chgs = [r['kappa_chg_single'] for r in valid_nb]
    global_chgs = [r['kappa_chg_global'] for r in valid_nb]
    max_s = max(single_chgs) if single_chgs else 0
    max_g = max(global_chgs) if global_chgs else 0
    h3_pass = max_s < 0.5 and max_g < 0.5
    log(f"\nH3: kappa_chg single={max_s:.4f}%, global={max_g:.4f}%: {'PASS' if h3_pass else 'FAIL'}")

    overall = h1_pass and h2_pass and h3_pass
    log(f"\n{'='*70}")
    log(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    if overall:
        log("  1/d_eff attenuation CONFIRMED in NLP linear-regime embeddings.")
        log("  The mechanism is domain-agnostic when kappa_eff ~ 1.0.")
    else:
        log("  NLP linear-regime surgery did not confirm 1/d_eff attenuation.")

    # Save
    result = {
        'experiment': 'nlp_linear_regime_global_vs_single_surgery',
        'preregistered_commit': PREREGISTERED_COMMIT,
        'cifar_reference_commit': '59faa5d',
        'cross_domain_fail_commit': '8ea7288',
        'A_RENORM_K20': A_RENORM_K20,
        'K': K,
        'surgery_levels': SURGERY_LEVELS,
        'kappa_eff_target': KAPPA_EFF_TARGET,
        'kappa_eff_valid_range': [KAPPA_EFF_VALID_LO, KAPPA_EFF_VALID_HI],
        'arch_summaries': arch_summaries,
        'all_records': all_records,
        'evaluation': {
            'h1_pass': h1_pass, 'h2_pass': h2_pass, 'h3_pass': h3_pass,
            'overall_pass': overall,
            'n_valid': len(valid_nb),
            'n_excluded': len(excluded),
            'excluded': excluded,
            'ratios': ratios,
            'd_effs': d_effs,
            'median_ratio': float(np.median(ratios)) if ratios else None,
            'median_d_eff': float(np.median(d_effs)) if d_effs else None,
            'h2_correct': h2_correct,
            'h2_total': len(h2_pairs),
            'max_kappa_chg_single': max_s,
            'max_kappa_chg_global': max_g,
        }
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\nSaved to {OUTPUT_PATH}")
    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))
    return overall


if __name__ == '__main__':
    main()
