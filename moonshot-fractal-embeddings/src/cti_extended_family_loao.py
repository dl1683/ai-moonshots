#!/usr/bin/env python -u
"""
EXTENDED FAMILY LOAO: Decoder + Encoder architecture family analysis
Pre-registered: commit 9e151d5 (BEFORE running)

Tests:
1. GPT-2 and Phi-2 (decoder family) LOAO against the 12-arch set
2. Encoder family within-group LOAO (deberta, electra, bert)
3. Documents last-layer causal LM artifact (if any) for GPT-2

Uses existing kappa_near_cache files -- NO model loading needed.
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# ==============================================================
# PRE-REGISTERED ARCHITECTURE SETS
# ==============================================================

# 12 original decoder/hybrid archs from kappa_loao_per_dataset.json
LOAO_12_ARCHS = [
    'Falcon-H1-0.5B-Base',
    'Mistral-7B-v0.3',
    'OLMo-1B-hf',
    'Qwen2.5-0.5B',
    'Qwen3-0.6B',
    'Qwen3-1.7B',
    'TinyLlama-1.1B-intermediate-step-1431k-3T',
    'gpt-neo-125m',
    'pythia-160m',
    'pythia-1b',
    'pythia-410m',
    'rwkv-4-169m-pile',
]

# New decoder archs to test (H1, H2)
NEW_DECODER_ARCHS = ['gpt2', 'phi2']

# Encoder archs to test (H3, H4)
ENCODER_ARCHS = ['deberta-base', 'electra-small', 'bert-base-uncased']

# Datasets used in the 12-arch analysis
DATASETS = ['20newsgroups', 'agnews', 'dbpedia', 'go_emotions']

# Pre-registered 12-arch alpha reference
ALPHA_12_MEAN = 1.477
ALPHA_12_STD = 0.034
# 5-sigma bounds for H1/H2
ALPHA_LOWER_5SIGMA = ALPHA_12_MEAN - 5 * ALPHA_12_STD
ALPHA_UPPER_5SIGMA = ALPHA_12_MEAN + 5 * ALPHA_12_STD
print(f"5-sigma bounds for H1/H2: [{ALPHA_LOWER_5SIGMA:.3f}, {ALPHA_UPPER_5SIGMA:.3f}]")


def load_kappa_cache(arch: str, datasets: list) -> list:
    """Load kappa_near_cache files for a given architecture."""
    records = []
    for ds in datasets:
        # Try multiple name patterns
        candidates = [
            RESULTS_DIR / f"kappa_near_cache_{ds}_{arch}.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    recs = json.load(f)
                for r in recs:
                    r['dataset'] = ds
                    r['arch'] = arch
                records.extend(recs)
                break
    return records


def fit_per_dataset_intercept(records: list, datasets: list) -> dict:
    """
    Fit logit(q) = alpha * kappa_nearest + C_d (per-dataset intercept) via OLS.
    Returns: alpha, beta (not used here), r2, mae, predicted values.
    """
    if len(records) < 3:
        return None

    kappas = np.array([r['kappa_nearest'] for r in records])
    logit_q = np.array([r['logit_q'] for r in records])
    ds_names = [r['dataset'] for r in records]
    ds_unique = sorted(set(ds_names))

    if len(ds_unique) < 1:
        return None

    # Design matrix: [kappa, D1, D2, ...] where D_i is dataset dummy
    n_ds = len(ds_unique)
    X = np.zeros((len(records), 1 + n_ds))
    X[:, 0] = kappas
    for i, r in enumerate(records):
        ds_idx = ds_unique.index(r['dataset'])
        X[i, 1 + ds_idx] = 1.0

    # OLS
    beta, res, rank, sv = np.linalg.lstsq(X, logit_q, rcond=None)
    alpha = float(beta[0])
    y_pred = X @ beta
    ss_res = float(np.sum((logit_q - y_pred)**2))
    ss_tot = float(np.sum((logit_q - logit_q.mean())**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    mae = float(np.mean(np.abs(y_pred - logit_q)))

    return {
        'alpha': alpha,
        'r2': r2,
        'mae': mae,
        'n': len(records),
        'datasets': ds_unique,
        'C_per_dataset': {ds: float(beta[1 + i]) for i, ds in enumerate(ds_unique)},
    }


def run_loao_on_set(arch_list: list, new_archs: list, datasets: list,
                    alpha_ref: float, alpha_std: float) -> dict:
    """
    LOAO: for each new_arch, fit on arch_list, predict new_arch.
    Returns per-arch LOAO alpha and validation stats.
    """
    # Load training set data
    train_records = []
    for arch in arch_list:
        recs = load_kappa_cache(arch, datasets)
        train_records.extend(recs)

    print(f"\nTraining set: {len(arch_list)} archs, {len(train_records)} records")

    # Fit global (training-only) model
    global_fit = fit_per_dataset_intercept(train_records, datasets)
    if global_fit is None:
        return {'error': 'insufficient training data'}

    print(f"Training alpha: {global_fit['alpha']:.4f}, R2={global_fit['r2']:.4f}")

    # For each new arch: load data, fit on train, predict new
    results = {}
    for new_arch in new_archs:
        new_recs = load_kappa_cache(new_arch, datasets)
        if not new_recs:
            print(f"  {new_arch}: NO DATA FOUND")
            results[new_arch] = {'error': 'no data'}
            continue

        print(f"\n  Testing {new_arch} ({len(new_recs)} records):")

        # Fit on all 12 training archs (don't exclude new arch since it's a prospective test)
        # Use global_fit alpha directly to predict new arch
        kappas_new = np.array([r['kappa_nearest'] for r in new_recs])
        logit_q_new = np.array([r['logit_q'] for r in new_recs])
        ds_new = [r['dataset'] for r in new_recs]
        ds_unique_new = sorted(set(ds_new))

        # Also fit new arch independently to get its own alpha
        own_fit = fit_per_dataset_intercept(new_recs, datasets)

        # Predict using frozen training alpha, fit only C_d for new arch
        # For each dataset in new arch, find C_d from training model intercepts
        # If dataset exists in training, use training C_d; else skip
        train_ds_intercepts = global_fit['C_per_dataset']

        preds_global = []
        actuals_global = []
        for r in new_recs:
            if r['dataset'] in train_ds_intercepts:
                pred = global_fit['alpha'] * r['kappa_nearest'] + train_ds_intercepts[r['dataset']]
                preds_global.append(pred)
                actuals_global.append(r['logit_q'])

        if len(preds_global) > 1:
            preds_global = np.array(preds_global)
            actuals_global = np.array(actuals_global)
            mae_global = float(np.mean(np.abs(preds_global - actuals_global)))
            r_global, p_global = stats.pearsonr(preds_global, actuals_global)
        else:
            mae_global, r_global, p_global = float('nan'), float('nan'), float('nan')

        print(f"    Own alpha: {own_fit['alpha']:.4f}, R2={own_fit['r2']:.4f}")
        print(f"    Global pred (frozen alpha): MAE={mae_global:.4f}, r={r_global:.4f}")

        # Check H1/H2: is own alpha within bounds?
        lower = alpha_ref - 5 * alpha_std
        upper = alpha_ref + 5 * alpha_std
        h_pass = lower <= own_fit['alpha'] <= upper

        results[new_arch] = {
            'own_alpha': own_fit['alpha'] if own_fit else None,
            'own_r2': own_fit['r2'] if own_fit else None,
            'own_n': own_fit['n'] if own_fit else 0,
            'global_frozen_mae': mae_global,
            'global_frozen_r': r_global,
            'alpha_in_5sigma': h_pass,
            'alpha_bounds': [float(lower), float(upper)],
        }

    return {
        'training_alpha': global_fit['alpha'],
        'training_r2': global_fit['r2'],
        'n_training': len(train_records),
        'results': results,
    }


def run_encoder_group_loao(encoder_archs: list, datasets: list) -> dict:
    """
    Within-encoder LOAO: fit on 2 of 3 encoders, predict the 3rd.
    """
    print(f"\n{'='*60}")
    print(f"ENCODER GROUP LOAO: {encoder_archs}")

    # First, get each encoder's own alpha
    encoder_data = {}
    for arch in encoder_archs:
        recs = load_kappa_cache(arch, datasets)
        if recs:
            fit = fit_per_dataset_intercept(recs, datasets)
            encoder_data[arch] = {'records': recs, 'own_fit': fit}
            if fit:
                print(f"  {arch}: own alpha={fit['alpha']:.4f}, R2={fit['r2']:.4f}, n={fit['n']}")
        else:
            print(f"  {arch}: NO DATA")

    valid_archs = [a for a in encoder_archs if a in encoder_data and encoder_data[a].get('own_fit')]
    if len(valid_archs) < 2:
        return {'error': 'insufficient encoder data', 'encoder_data': {}}

    own_alphas = [encoder_data[a]['own_fit']['alpha'] for a in valid_archs]
    alpha_mean = float(np.mean(own_alphas))
    alpha_std_enc = float(np.std(own_alphas))
    alpha_cv = float(alpha_std_enc / abs(alpha_mean)) if abs(alpha_mean) > 0.01 else float('nan')

    print(f"\n  Encoder own-alpha summary:")
    print(f"  Mean: {alpha_mean:.4f}, Std: {alpha_std_enc:.4f}, CV: {alpha_cv:.4f}")
    print(f"  H3 (CV < 0.50): {'PASS' if alpha_cv < 0.50 else 'FAIL'}")
    print(f"  H4 (encoder alpha >> 1.477): {'PASS' if alpha_mean > 2 * ALPHA_12_MEAN else 'FAIL'}")

    # Within-encoder LOAO
    loao_results = {}
    for test_arch in valid_archs:
        train_archs = [a for a in valid_archs if a != test_arch]
        train_recs = []
        for a in train_archs:
            train_recs.extend(encoder_data[a]['records'])

        train_fit = fit_per_dataset_intercept(train_recs, datasets)
        if not train_fit:
            continue

        test_recs = encoder_data[test_arch]['records']
        test_kappas = np.array([r['kappa_nearest'] for r in test_recs])
        test_logit_q = np.array([r['logit_q'] for r in test_recs])
        test_ds = [r['dataset'] for r in test_recs]

        # Predict test arch using frozen train alpha + train C_d
        preds = []
        actuals = []
        for r in test_recs:
            if r['dataset'] in train_fit['C_per_dataset']:
                pred = train_fit['alpha'] * r['kappa_nearest'] + train_fit['C_per_dataset'][r['dataset']]
                preds.append(pred)
                actuals.append(r['logit_q'])

        if len(preds) > 1:
            mae = float(np.mean(np.abs(np.array(preds) - np.array(actuals))))
            r_val, p_val = stats.pearsonr(preds, actuals)
        else:
            mae, r_val, p_val = float('nan'), float('nan'), float('nan')

        loao_results[test_arch] = {
            'train_alpha': train_fit['alpha'],
            'predicted_mae': mae,
            'predicted_r': r_val,
            'test_own_alpha': encoder_data[test_arch]['own_fit']['alpha'],
        }
        print(f"    LOAO {test_arch}: train_alpha={train_fit['alpha']:.4f}, r={r_val:.4f}")

    return {
        'own_alphas': {a: encoder_data[a]['own_fit']['alpha'] for a in valid_archs if a in encoder_data},
        'alpha_mean': alpha_mean,
        'alpha_std': alpha_std_enc,
        'alpha_cv': alpha_cv,
        'h3_pass': alpha_cv < 0.50,
        'h4_pass': alpha_mean > 2 * ALPHA_12_MEAN,
        'loao_results': loao_results,
        'n_valid_archs': len(valid_archs),
    }


def check_last_layer_pattern(arch: str, datasets: list) -> dict:
    """
    Check if last layer shows kappa-q degeneration (causal LM artifact).
    A causal LM may have kappa increase at last layer but q decrease.
    """
    all_recs = load_kappa_cache(arch, datasets)
    if not all_recs:
        return {}

    # Group by dataset, check monotonicity of q vs layer index
    by_dataset = {}
    for r in all_recs:
        ds = r['dataset']
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(r)

    results = {}
    for ds, recs in by_dataset.items():
        recs_sorted = sorted(recs, key=lambda x: x['layer'])
        layers = [r['layer'] for r in recs_sorted]
        kappas = [r['kappa_nearest'] for r in recs_sorted]
        qs = [r['q'] for r in recs_sorted]
        logit_qs = [r['logit_q'] for r in recs_sorted]

        # Check if last layer has highest kappa but NOT highest q
        last_kappa_rank = sorted(range(len(kappas)), key=lambda i: kappas[i]).index(len(kappas)-1) + 1
        last_q_rank = sorted(range(len(qs)), key=lambda i: qs[i]).index(len(qs)-1) + 1

        results[ds] = {
            'layers': layers,
            'kappas': [round(k, 4) for k in kappas],
            'qs': [round(q, 4) for q in qs],
            'last_layer_kappa_rank': int(last_kappa_rank),  # 4 = highest
            'last_layer_q_rank': int(last_q_rank),
            'last_layer_degenerate': last_kappa_rank >= 3 and last_q_rank <= 2,
        }

    n_degenerate = sum(1 for v in results.values() if v['last_layer_degenerate'])
    return {
        'per_dataset': results,
        'n_degenerate_datasets': n_degenerate,
        'total_datasets': len(results),
        'last_layer_causal_lm_artifact': n_degenerate >= 2,
    }


def main():
    print("=" * 70)
    print("EXTENDED FAMILY LOAO (Pre-registered: commit 9e151d5)")
    print("=" * 70)

    results = {
        'experiment': 'extended_family_loao',
        'preregistered_commit': '9e151d5',
        'alpha_12_reference': ALPHA_12_MEAN,
        'alpha_12_std': ALPHA_12_STD,
        'alpha_bounds_5sigma': [round(ALPHA_LOWER_5SIGMA, 4), round(ALPHA_UPPER_5SIGMA, 4)],
    }

    # ============================================================
    # PART 1: Check gpt2 last-layer artifact
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1: GPT-2 last-layer analysis (pre-reg H1 context)")
    print("=" * 60)

    gpt2_artifact = check_last_layer_pattern('gpt2', DATASETS)
    results['gpt2_last_layer_artifact'] = gpt2_artifact

    if gpt2_artifact.get('last_layer_causal_lm_artifact'):
        print("  GPT-2 LAST-LAYER CAUSAL LM ARTIFACT CONFIRMED:")
        print(f"  {gpt2_artifact['n_degenerate_datasets']}/{gpt2_artifact['total_datasets']} datasets show kappa high but q low at last layer")
    else:
        print("  GPT-2 last-layer pattern: no consistent degeneration")

    # ============================================================
    # PART 2: Decoder prospective LOAO (H1: gpt2, H2: phi2)
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: Decoder prospective LOAO (H1: gpt2, H2: phi2)")
    print("=" * 60)

    decoder_loao = run_loao_on_set(
        arch_list=LOAO_12_ARCHS,
        new_archs=NEW_DECODER_ARCHS,
        datasets=DATASETS,
        alpha_ref=ALPHA_12_MEAN,
        alpha_std=ALPHA_12_STD,
    )
    results['decoder_prospective_loao'] = decoder_loao

    # Evaluate H1 and H2
    for i, arch in enumerate(NEW_DECODER_ARCHS, 1):
        if arch in decoder_loao.get('results', {}):
            arch_res = decoder_loao['results'][arch]
            if 'error' not in arch_res:
                h_pass = arch_res.get('alpha_in_5sigma', False)
                own_alpha = arch_res.get('own_alpha', 'N/A')
                print(f"  H{i} ({arch}): own_alpha={own_alpha:.4f}, "
                      f"in_5sigma={'PASS' if h_pass else 'FAIL'}")

    # ============================================================
    # PART 3: Encoder family within-group LOAO (H3, H4)
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 3: Encoder family LOAO (H3: CV, H4: alpha separation)")
    print("=" * 60)

    encoder_loao = run_encoder_group_loao(ENCODER_ARCHS, DATASETS)
    results['encoder_group_loao'] = encoder_loao

    # ============================================================
    # PART 4: Decoder-only per-dataset alpha across ALL decoder archs
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 4: Full 14-arch decoder LOAO (12 original + gpt2 + phi2)")
    print("=" * 60)

    # Only include gpt2 and phi2 if their last-layer issue is NOT present for all datasets
    # Based on pre-reg: include all layers for proper comparison
    all_14_archs = LOAO_12_ARCHS + NEW_DECODER_ARCHS

    all_decoder_alphas = {}
    for arch in all_14_archs:
        recs = load_kappa_cache(arch, DATASETS)
        if recs:
            fit = fit_per_dataset_intercept(recs, DATASETS)
            if fit:
                all_decoder_alphas[arch] = fit['alpha']
                print(f"  {arch}: alpha={fit['alpha']:.4f}")

    valid_alphas = list(all_decoder_alphas.values())
    if valid_alphas:
        alpha_mean_14 = float(np.mean(valid_alphas))
        alpha_std_14 = float(np.std(valid_alphas))
        alpha_cv_14 = float(alpha_std_14 / abs(alpha_mean_14)) if abs(alpha_mean_14) > 0.01 else float('nan')
        print(f"\n  14-arch alpha: mean={alpha_mean_14:.4f}, std={alpha_std_14:.4f}, CV={alpha_cv_14:.4f}")

    results['all_14_decoder_alphas'] = all_decoder_alphas
    results['alpha_14_mean'] = float(np.mean(valid_alphas)) if valid_alphas else None
    results['alpha_14_cv'] = float(np.std(valid_alphas) / abs(np.mean(valid_alphas))) if valid_alphas else None

    # ============================================================
    # EVALUATION
    # ============================================================
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # H1: gpt2 alpha in bounds
    gpt2_res = decoder_loao.get('results', {}).get('gpt2', {})
    gpt2_own_alpha = gpt2_res.get('own_alpha') if gpt2_res else None
    h1_pass = gpt2_res.get('alpha_in_5sigma', False) if gpt2_res and 'error' not in gpt2_res else False

    # H2: phi2 alpha in bounds
    phi2_res = decoder_loao.get('results', {}).get('phi2', {})
    phi2_own_alpha = phi2_res.get('own_alpha') if phi2_res else None
    h2_pass = phi2_res.get('alpha_in_5sigma', False) if phi2_res and 'error' not in phi2_res else False

    # H3: encoder CV < 0.50
    h3_pass = encoder_loao.get('h3_pass', False)
    encoder_cv = encoder_loao.get('alpha_cv', float('nan'))

    # H4: encoder alpha >> decoder alpha
    h4_pass = encoder_loao.get('h4_pass', False)
    encoder_alpha_mean = encoder_loao.get('alpha_mean', float('nan'))

    gpt2_alpha_str = f"{gpt2_own_alpha:.4f}" if gpt2_own_alpha is not None else "N/A"
    phi2_alpha_str = f"{phi2_own_alpha:.4f}" if phi2_own_alpha is not None else "N/A"
    print(f"H1 (gpt2 alpha in 5-sigma bounds [{ALPHA_LOWER_5SIGMA:.3f}, {ALPHA_UPPER_5SIGMA:.3f}]): "
          f"alpha={gpt2_alpha_str} -> {'PASS' if h1_pass else 'FAIL'}")
    print(f"H2 (phi2 alpha in 5-sigma bounds): "
          f"alpha={phi2_alpha_str} -> {'PASS' if h2_pass else 'FAIL'}")
    print(f"H3 (encoder CV < 0.50): CV={encoder_cv:.4f} -> {'PASS' if h3_pass else 'FAIL'}")
    print(f"H4 (encoder alpha > 2x decoder): encoder_mean={encoder_alpha_mean:.4f} vs decoder=1.477 -> {'PASS' if h4_pass else 'FAIL'}")

    if gpt2_artifact.get('last_layer_causal_lm_artifact'):
        print(f"\nNOTE: GPT-2 last-layer causal LM artifact detected -- H1 failure is expected from this mechanism")

    evaluation = {
        'h1_gpt2_alpha_bounds': h1_pass,
        'h1_gpt2_alpha_value': float(gpt2_own_alpha) if gpt2_own_alpha is not None else None,
        'h2_phi2_alpha_bounds': h2_pass,
        'h2_phi2_alpha_value': float(phi2_own_alpha) if phi2_own_alpha is not None else None,
        'h3_encoder_cv_pass': h3_pass,
        'h3_encoder_cv': float(encoder_cv),
        'h4_encoder_separation_pass': h4_pass,
        'h4_encoder_alpha_mean': float(encoder_alpha_mean),
        'gpt2_last_layer_artifact': bool(gpt2_artifact.get('last_layer_causal_lm_artifact', False)),
        'alpha_14_cv': results.get('alpha_14_cv'),
    }
    results['evaluation'] = evaluation

    # Save
    out_path = RESULTS_DIR / "cti_extended_family_loao.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    main()
