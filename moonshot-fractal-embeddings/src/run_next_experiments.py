"""Chain the next experiments: deep hierarchy + backbone RAG + semantic zoom.

Run after the head-only RAG demo (bffb622) finishes.

Usage: python src/run_next_experiments.py
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("EXPERIMENT CHAIN: Deep Hierarchy + Backbone RAG + Semantic Zoom")
print("=" * 70)

# Phase 1: Deep hierarchy experiments (4 configs x 5 seeds)
print("\n[PHASE 1] Deep Hierarchy Experiments")
print("=" * 70)
result = subprocess.run(
    [sys.executable, "-u", "src/run_deep_hierarchy_experiments.py", "all"],
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)
print(f"Deep hierarchy exit code: {result.returncode}")

# Phase 2: Backbone RAG demo (3 datasets x 3 seeds)
print("\n[PHASE 2] Backbone RAG Demo")
print("=" * 70)
result = subprocess.run(
    [sys.executable, "-u", "src/adaptive_rag_demo.py",
     "--datasets", "clinc", "dbpedia_classes", "trec",
     "--seeds", "42", "123", "456",
     "--device", "cuda",
     "--backbone"],
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)
print(f"Backbone RAG exit code: {result.returncode}")

# Phase 3: Semantic zoom demo (CLINC + DBPedia)
print("\n[PHASE 3] Semantic Zoom Demo")
print("=" * 70)
for ds in ["clinc", "dbpedia_classes"]:
    result = subprocess.run(
        [sys.executable, "-u", "src/semantic_zoom_demo.py", ds],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print(f"Semantic zoom {ds} exit code: {result.returncode}")

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
