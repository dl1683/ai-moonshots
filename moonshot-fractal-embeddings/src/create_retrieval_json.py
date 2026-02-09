"""Create retrieval benchmark JSON from printed output (all 3 seeds completed but aggregation crashed)."""
import json, numpy as np
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

d = {'experiment': 'retrieval_benchmark', 'model': 'bge-small', 'dataset': 'clinc',
     'seeds': [42, 123, 456], 'ks': [1, 5, 10, 20], 'stage1_epochs': 5,
     'max_train_samples': 10000, 'max_test_samples': 2000,
     'timestamp': datetime.now().isoformat(), 'results_by_seed': {}}

def mk(r1,r5,r10,r20,mrr):
    return {'recall@1':r1,'recall@5':r5,'recall@10':r10,'recall@20':r20,'mrr':mrr}

# Seed 42
d['results_by_seed']['42'] = {
  'v5': {
    '1': {'L0': mk(0.973,0.989,0.991,0.994,0.980), 'L1': mk(0.878,0.957,0.976,0.983,0.911)},
    '2': {'L0': mk(0.978,0.987,0.991,0.994,0.983), 'L1': mk(0.940,0.980,0.985,0.990,0.957)},
    '3': {'L0': mk(0.980,0.989,0.993,0.998,0.985), 'L1': mk(0.949,0.980,0.988,0.995,0.963)},
    '4': {'L0': mk(0.980,0.990,0.995,0.999,0.985), 'L1': mk(0.954,0.982,0.989,0.996,0.967)},
  },
  'mrl': {
    '1': {'L0': mk(0.974,0.989,0.998,0.999,0.982), 'L1': mk(0.931,0.975,0.992,0.996,0.951)},
    '2': {'L0': mk(0.975,0.993,0.997,1.000,0.982), 'L1': mk(0.934,0.980,0.990,0.996,0.955)},
    '3': {'L0': mk(0.975,0.995,0.997,1.000,0.983), 'L1': mk(0.934,0.981,0.989,0.995,0.955)},
    '4': {'L0': mk(0.974,0.995,0.997,1.000,0.983), 'L1': mk(0.934,0.981,0.989,0.995,0.955)},
  },
}

# Seed 123
d['results_by_seed']['123'] = {
  'v5': {
    '1': {'L0': mk(0.966,0.988,0.990,0.992,0.977), 'L1': mk(0.853,0.954,0.969,0.984,0.898)},
    '2': {'L0': mk(0.978,0.993,0.993,0.996,0.985), 'L1': mk(0.911,0.970,0.984,0.992,0.938)},
    '3': {'L0': mk(0.976,0.996,0.996,0.998,0.985), 'L1': mk(0.922,0.977,0.990,0.994,0.947)},
    '4': {'L0': mk(0.974,0.996,0.997,0.998,0.984), 'L1': mk(0.913,0.981,0.991,0.994,0.943)},
  },
  'mrl': {
    '1': {'L0': mk(0.976,0.992,0.997,0.999,0.984), 'L1': mk(0.932,0.981,0.991,0.996,0.954)},
    '2': {'L0': mk(0.978,0.994,0.996,0.999,0.985), 'L1': mk(0.935,0.980,0.991,0.994,0.955)},
    '3': {'L0': mk(0.981,0.996,0.998,0.999,0.987), 'L1': mk(0.940,0.983,0.992,0.994,0.958)},
    '4': {'L0': mk(0.980,0.996,0.998,0.998,0.986), 'L1': mk(0.938,0.981,0.992,0.994,0.956)},
  },
}

# Seed 456
d['results_by_seed']['456'] = {
  'v5': {
    '1': {'L0': mk(0.976,0.990,0.993,0.998,0.983), 'L1': mk(0.881,0.963,0.979,0.990,0.916)},
    '2': {'L0': mk(0.978,0.992,0.995,1.000,0.985), 'L1': mk(0.929,0.978,0.991,0.999,0.951)},
    '3': {'L0': mk(0.983,0.995,0.998,0.999,0.988), 'L1': mk(0.940,0.988,0.997,0.999,0.960)},
    '4': {'L0': mk(0.982,0.995,0.998,0.998,0.988), 'L1': mk(0.935,0.987,0.997,0.998,0.957)},
  },
  'mrl': {
    '1': {'L0': mk(0.980,0.998,1.000,1.000,0.988), 'L1': mk(0.946,0.989,0.999,1.000,0.965)},
    '2': {'L0': mk(0.986,0.998,1.000,1.000,0.991), 'L1': mk(0.949,0.989,0.999,1.000,0.966)},
    '3': {'L0': mk(0.989,1.000,1.000,1.000,0.993), 'L1': mk(0.955,0.989,1.000,1.000,0.969)},
    '4': {'L0': mk(0.988,0.999,1.000,1.000,0.992), 'L1': mk(0.956,0.989,0.998,1.000,0.970)},
  },
}

out_path = RESULTS_DIR / "retrieval_benchmark_bge-small_clinc.json"
with open(out_path, 'w') as f:
    json.dump(d, f, indent=2)
print(f'Saved {out_path}')

# Summary
seeds = [42, 123, 456]
for level in ['L0', 'L1']:
    print(f'\n--- {level} Recall@1 ---')
    for j in ['1','2','3','4']:
        v = [d['results_by_seed'][str(s)]['v5'][j][level]['recall@1'] for s in seeds]
        m = [d['results_by_seed'][str(s)]['mrl'][j][level]['recall@1'] for s in seeds]
        print(f'  j={j} ({int(j)*64}d): V5={np.mean(v):.3f}+-{np.std(v):.3f}  MRL={np.mean(m):.3f}+-{np.std(m):.3f}  delta={np.mean(v)-np.mean(m):+.3f}')

vr = [d['results_by_seed'][str(s)]['v5']['4']['L1']['recall@1']-d['results_by_seed'][str(s)]['v5']['1']['L1']['recall@1'] for s in seeds]
mr = [d['results_by_seed'][str(s)]['mrl']['4']['L1']['recall@1']-d['results_by_seed'][str(s)]['mrl']['1']['L1']['recall@1'] for s in seeds]
print(f'\nL1 R@1 ramp (256d-64d):')
print(f'  V5:  {np.mean(vr):+.3f} +/- {np.std(vr):.3f}')
print(f'  MRL: {np.mean(mr):+.3f} +/- {np.std(mr):.3f}')
print(f'  Ratio: {np.mean(vr)/max(np.mean(mr),0.001):.0f}x')
