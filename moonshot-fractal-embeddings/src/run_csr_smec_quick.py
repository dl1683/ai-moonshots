"""
Quick CSR/SMEC negative control runs (seed 42 only on 9 main datasets).
======================================================================

Run this after HEAL completes to get negative controls for the paper table.

Usage:
    python -u src/run_csr_smec_quick.py
"""

import sys
import os
import json
import time
import gc
import torch
from pathlib import Path

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))

from external_baselines.common import RESULTS_DIR, load_cached_embeddings
from external_baselines.csr_baseline import run_csr
from external_baselines.smec_baseline import run_smec

# 9 main datasets only (skip deep hierarchy for now)
DATASETS = [
    "yahoo", "goemotions", "newsgroups", "trec", "arxiv",
    "dbpedia_classes", "clinc", "wos", "hupd_sec_cls",
]
SEED = 42


def main():
    print("=" * 60)
    print("  Quick CSR/SMEC negative control runs (seed 42)")
    print("=" * 60)

    methods = {"csr": run_csr, "smec": run_smec}
    total = 0
    done = 0

    for method_name, runner in methods.items():
        for ds in DATASETS:
            result_path = RESULTS_DIR / method_name / f"{ds}_seed{SEED}.json"
            if result_path.exists():
                r = json.load(open(result_path))
                print(f"  SKIP {method_name}/{ds} (exists, S={r['steerability']:+.4f})")
                continue
            total += 1

    print(f"\n  {total} runs needed\n")

    start = time.time()
    for method_name, runner in methods.items():
        print(f"\n--- {method_name.upper()} ---")
        for ds in DATASETS:
            result_path = RESULTS_DIR / method_name / f"{ds}_seed{SEED}.json"
            if result_path.exists():
                continue

            done += 1
            print(f"\n[{done}/{total}] {method_name}/{ds}/seed{SEED}...")

            try:
                result = runner(ds, SEED)
                elapsed = time.time() - start
                print(f"  S={result['steerability']:+.4f} ({elapsed/60:.1f}min)")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - start
    print(f"\nDone! {done} runs in {elapsed/60:.1f} minutes")

    # Print summary
    print("\n" + "=" * 60)
    print("  NEGATIVE CONTROLS SUMMARY")
    print("=" * 60)
    for method_name in ["csr", "smec"]:
        steers = []
        for ds in DATASETS:
            path = RESULTS_DIR / method_name / f"{ds}_seed{SEED}.json"
            if path.exists():
                r = json.load(open(path))
                steers.append((ds, r["steerability"]))
        print(f"\n{method_name.upper()}:")
        for ds, s in steers:
            print(f"  {ds}: S={s:+.4f}")
        if steers:
            vals = [s for _, s in steers]
            print(f"  Mean: {sum(vals)/len(vals):+.4f}, "
                  f"Max|S|: {max(abs(v) for v in vals):.4f}")


if __name__ == "__main__":
    main()
