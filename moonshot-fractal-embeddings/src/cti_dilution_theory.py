"""
Dilution Theory: alpha_DO = alpha_LOAO * dilution_factor

Theory (from Session 8 analysis):
  When we move the nearest class pair by delta in embedding space,
  the observed slope alpha_DO is ATTENUATED relative to alpha_LOAO
  by the fraction of misclassifications concentrated at that pair:

    alpha_DO = alpha_LOAO * dilution_factor

  where:
    dilution_factor = P(confuse with nearest class | misclassified)
                    = sum_c CM[c, nn[c]] / sum_off_diagonal(CM)

If this holds across ALL tested conditions, the "failed" replications
become CONFIRMATORY evidence of the dilution mechanism.

Test: for each do-intervention condition, compute:
  1. dilution_factor from baseline embeddings (confusion matrix)
  2. predicted alpha_DO = 1.549 * dilution_factor
  3. Compare to observed alpha_DO from intervention sweeps

This test turns failures into confirmations: the law IS the law,
but the intervention only "writes" to the fraction of confusions
at the perturbed pair.
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

LOAO_ALPHA = 1.549

# ================================================================
# DILUTION FACTOR COMPUTATION
# ================================================================
def compute_dilution_factor(X, y, method='nearest_pair'):
    """
    Compute the dilution factor from baseline embeddings.

    method='nearest_pair':
      dilution_factor = (CM[c*, c*'] + CM[c*', c*]) / total_off_diagonal
      where (c*, c*') is the nearest class centroid pair.

    method='nearest_per_class':
      dilution_factor = sum_c CM[c, nn[c]] / total_off_diagonal
      where nn[c] is each class's nearest neighbor.
    """
    classes = np.unique(y)
    K = len(classes)
    d = X.shape[1]

    # Compute centroids
    centroids = np.zeros((K, d))
    for i, c in enumerate(classes):
        centroids[i] = X[y == c].mean(0)

    # Find nearest class pair
    min_dist = np.inf
    nearest_c1, nearest_c2 = 0, 1
    nn_per_class = {}
    for i in range(K):
        for j in range(i + 1, K):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_dist:
                min_dist = dist
                nearest_c1, nearest_c2 = classes[i], classes[j]
        # nn for class i
        dists_i = [np.linalg.norm(centroids[i] - centroids[j]) for j in range(K) if j != i]
        nn_per_class[classes[i]] = classes[np.argmin(dists_i)]

    # 1-NN confusion matrix using proper train/test split (avoid trivial self-prediction)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    n_tr = int(0.7 * len(X))
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1)
    knn.fit(X[idx[:n_tr]], y[idx[:n_tr]])
    y_pred = knn.predict(X[idx[n_tr:]])
    y_true = y[idx[n_tr:]]
    CM = confusion_matrix(y_true, y_pred, labels=classes)

    # Off-diagonal (misclassification) total
    total_wrong = CM.sum() - np.diag(CM).sum()
    if total_wrong == 0:
        return 0.0, 0.0, nn_per_class, (nearest_c1, nearest_c2), CM

    if method == 'nearest_pair':
        # Only the SINGLE nearest pair
        i1 = list(classes).index(nearest_c1)
        i2 = list(classes).index(nearest_c2)
        pair_confusions = CM[i1, i2] + CM[i2, i1]
        dilution_nearest = pair_confusions / total_wrong
        # Also compute per-class version
        per_class_near = 0
        for ci, c in enumerate(classes):
            nn_c = nn_per_class[c]
            nn_idx = list(classes).index(nn_c)
            per_class_near += CM[ci, nn_idx]
        dilution_perclass = per_class_near / total_wrong
        return float(dilution_nearest), float(dilution_perclass), nn_per_class, (nearest_c1, nearest_c2), CM

    return 0.0, 0.0, {}, (0, 0), CM


def compute_baseline_q(X, y, K):
    """Compute normalized 1-NN accuracy q."""
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1)
    # LOO: approximate with train=test (ceiling)
    knn.fit(X, y)
    acc = float(knn.score(X, y))  # Overfit estimate
    # Better: use a split
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    n_tr = int(0.7 * n)
    knn2 = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1)
    knn2.fit(X[idx[:n_tr]], y[idx[:n_tr]])
    acc = float(knn2.score(X[idx[n_tr:]], y[idx[n_tr:]]))
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    return float(q), float(acc)


# ================================================================
# LOAD DO-INTERVENTION RESULTS
# ================================================================
def load_do_intervention_conditions():
    """Load all available do-intervention results."""
    conditions = []

    # V2 text results
    with open('results/cti_do_intervention_text.json') as f:
        d = json.load(f)
    for key in ['pythia-160m_dbpedia', 'pythia-160m_agnews',
                'gpt-neo-125m_dbpedia', 'gpt-neo-125m_agnews']:
        cond = d.get(key, {})
        nearest = cond.get('nearest', {}).get('analysis', {})
        baseline = cond.get('baseline', {})
        parts = key.split('_', 1)
        model, dataset = parts[0], parts[1]
        emb_file = f'results/do_int_embs_{model}_{dataset}.npz'
        if os.path.exists(emb_file) and nearest:
            conditions.append({
                'label': key,
                'model': model,
                'dataset': dataset,
                'alpha_DO': nearest.get('alpha_intervention'),
                'r': nearest.get('r'),
                'baseline_q': baseline.get('q'),
                'baseline_kappa': baseline.get('kappa'),
                'emb_file': emb_file,
                'source': 'v2_text',
            })

    # Replication results
    try:
        with open('results/cti_causal_replication.json') as f:
            rep = json.load(f)
        replication_data = rep.get('replication', rep.get('results', {}))
        for key, val in replication_data.items():
            if isinstance(val, dict) and 'alpha' in val:
                # Try to find matching emb file
                safe_key = key.replace('/', '_').replace(' ', '_')
                # Match to available npz files
                for prefix in ['do_int_repl_', 'do_int_layer_']:
                    for f in os.listdir('results/'):
                        if f.endswith('.npz') and prefix in f:
                            if safe_key.lower() in f.lower() or key.split('/')[0].lower() in f.lower():
                                conditions.append({
                                    'label': key,
                                    'alpha_DO': val.get('alpha'),
                                    'r': val.get('r'),
                                    'emb_file': f'results/{f}',
                                    'source': 'replication',
                                })
                                break
    except Exception as e:
        print(f"  Warning: could not load replication results: {e}")

    # Adaptive do-intervention
    try:
        with open('results/cti_adaptive_do_intervention.json') as f:
            adap = json.load(f)
        nearest_res = adap.get('nearest_result', {})
        # Find emb file
        model_k = adap.get('model', '').replace('/', '_').replace('openai-community_', '')
        dataset_k = adap.get('dataset', '').replace('_14', '')
        emb_file = f'results/do_int_repl_gpt2_dbpedia_L12.npz'
        if os.path.exists(emb_file):
            conditions.append({
                'label': f"gpt2/dbpedia-L12 (adaptive)",
                'model': 'gpt2',
                'dataset': 'dbpedia',
                'layer': adap.get('layer'),
                'alpha_DO': nearest_res.get('alpha'),
                'r': nearest_res.get('r'),
                'emb_file': emb_file,
                'source': 'adaptive',
            })
    except Exception as e:
        print(f"  Warning: could not load adaptive results: {e}")

    return conditions


# ================================================================
# MAIN ANALYSIS
# ================================================================
def main():
    print("Dilution Theory Validation")
    print("=" * 60)
    print(f"Hypothesis: alpha_DO = {LOAO_ALPHA} * dilution_factor")
    print()

    conditions = load_do_intervention_conditions()
    print(f"Found {len(conditions)} do-intervention conditions")
    print()

    results = []
    for cond in conditions:
        label = cond['label']
        alpha_DO = cond.get('alpha_DO')
        emb_file = cond.get('emb_file')

        if alpha_DO is None or emb_file is None or not os.path.exists(emb_file):
            print(f"  SKIP {label}: missing data")
            continue

        print(f"Analyzing: {label}")
        print(f"  emb_file: {emb_file}")
        print(f"  alpha_DO (observed): {alpha_DO:.4f}")

        # Load embeddings
        data = np.load(emb_file)
        X = data['X']
        y = data['y'].astype(int)
        K = len(np.unique(y))
        print(f"  embeddings: {X.shape}, K={K}")

        # Compute dilution factor
        dil_nearest, dil_perclass, nn_map, nearest_pair, CM = compute_dilution_factor(X, y)
        q_val, acc_val = compute_baseline_q(X, y, K)

        # Predictions
        alpha_pred_nearest = LOAO_ALPHA * dil_nearest
        alpha_pred_perclass = LOAO_ALPHA * dil_perclass

        # Error
        err_nearest = abs(alpha_DO - alpha_pred_nearest)
        err_perclass = abs(alpha_DO - alpha_pred_perclass)
        rel_err_nearest = err_nearest / max(alpha_DO, 0.01)
        rel_err_perclass = err_perclass / max(alpha_DO, 0.01)

        print(f"  dilution_nearest_pair: {dil_nearest:.4f}")
        print(f"  dilution_per_class:    {dil_perclass:.4f}")
        print(f"  alpha_pred (nearest):  {alpha_pred_nearest:.4f}  err={err_nearest:.4f} ({rel_err_nearest:.1%})")
        print(f"  alpha_pred (perclass): {alpha_pred_perclass:.4f}  err={err_perclass:.4f} ({rel_err_perclass:.1%})")
        print(f"  baseline q: {q_val:.4f}")
        print()

        results.append({
            'label': label,
            'alpha_DO': float(alpha_DO),
            'r': float(cond.get('r', 0)),
            'K': int(K),
            'baseline_q': float(q_val),
            'dilution_nearest': float(dil_nearest),
            'dilution_perclass': float(dil_perclass),
            'alpha_pred_nearest': float(alpha_pred_nearest),
            'alpha_pred_perclass': float(alpha_pred_perclass),
            'err_nearest': float(err_nearest),
            'err_perclass': float(err_perclass),
            'rel_err_nearest': float(rel_err_nearest),
            'rel_err_perclass': float(rel_err_perclass),
            'loao_alpha': LOAO_ALPHA,
            'source': cond.get('source'),
        })

    if not results:
        print("No results to analyze")
        return

    # ================================================================
    # SUMMARY STATISTICS
    # ================================================================
    print("=" * 60)
    print("SUMMARY: Dilution Theory Predictions vs Observed alpha_DO")
    print("=" * 60)
    print(f"{'Condition':<35} {'Obs':>6} {'Pred':>6} {'Err':>6} {'Dil':>6}")
    print("-" * 65)
    for r in results:
        print(f"  {r['label'][:33]:<33} {r['alpha_DO']:>6.3f} {r['alpha_pred_nearest']:>6.3f} "
              f"{r['rel_err_nearest']:>6.1%} {r['dilution_nearest']:>6.3f}")

    obs = np.array([r['alpha_DO'] for r in results])
    pred = np.array([r['alpha_pred_nearest'] for r in results])
    dils = np.array([r['dilution_nearest'] for r in results])

    # Correlation between observed and predicted
    if len(obs) >= 3:
        corr, pval = stats.pearsonr(obs, pred)
        r2 = corr ** 2
        mae = float(np.mean(np.abs(obs - pred)))
        print(f"\nPrediction quality:")
        print(f"  Pearson r(obs, pred) = {corr:.4f}  (r2={r2:.4f})")
        print(f"  MAE = {mae:.4f}")
        print(f"  p-value = {pval:.4f}")

        # Also test alpha_DO = alpha_LOAO * dilution (regression through origin)
        from numpy.linalg import lstsq
        slope, _, _, _ = lstsq(dils.reshape(-1, 1), obs, rcond=None)
        print(f"\nRegression: alpha_DO = slope * dilution")
        print(f"  slope = {slope[0]:.4f}  (expected {LOAO_ALPHA})")
        print(f"  devation from LOAO alpha: {abs(slope[0] - LOAO_ALPHA) / LOAO_ALPHA:.1%}")

    # Pass criterion: MAE < 0.3 and r > 0.8
    pass_test = (len(obs) >= 3 and
                 mae < 0.3 and
                 corr > 0.8 and
                 abs(slope[0] - LOAO_ALPHA) / LOAO_ALPHA < 0.3)

    print(f"\nPASS (MAE<0.3, r>0.8, slope within 30% of LOAO): {pass_test}")

    output = {
        'hypothesis': f'alpha_DO = {LOAO_ALPHA} * dilution_factor',
        'n_conditions': len(results),
        'pearson_r': float(corr) if len(obs) >= 3 else None,
        'r2': float(r2) if len(obs) >= 3 else None,
        'mae': float(mae) if len(obs) >= 3 else None,
        'regression_slope': float(slope[0]) if len(obs) >= 3 else None,
        'pass': bool(pass_test) if len(obs) >= 3 else False,
        'results': results,
    }

    with open('results/cti_dilution_theory.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to results/cti_dilution_theory.json")


if __name__ == '__main__':
    main()
