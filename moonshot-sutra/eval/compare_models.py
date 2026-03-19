"""Compare multiple models on the Sutra eval suite.

Runs the eval on all specified ollama models and produces a comparison table.

Usage:
    python eval/compare_models.py --models qwen3:4b,phi3:mini,granite4:3b
    python eval/compare_models.py --all-local  # Run on all installed ollama models
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
EVAL_DIR = REPO_ROOT / "eval"


def get_ollama_models():
    """Get list of installed ollama models."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def query_ollama(model, prompt, max_tokens=4096, timeout=180):
    """Query an ollama model and return the response."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            # Handle thinking models
            response = result.get("response", "")
            thinking = result.get("thinking", "")
            if not response and thinking:
                response = thinking
            return {
                "response": response,
                "thinking": thinking,
                "eval_count": result.get("eval_count", 0),
                "total_duration_s": result.get("total_duration", 0) / 1e9,
            }
    except Exception as e:
        return {"response": f"ERROR: {e}", "thinking": "", "eval_count": 0, "total_duration_s": 0}


def load_eval(eval_path):
    """Load eval questions from JSONL."""
    questions = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_model_on_eval(model, questions, max_questions=None):
    """Run a model on all eval questions."""
    results = []
    total = len(questions) if max_questions is None else min(len(questions), max_questions)
    questions = questions[:total]

    print(f"\n  Running {model} on {total} questions...")
    start = time.time()

    for i, q in enumerate(questions):
        prompt = f"Answer the following question carefully and completely.\n\nQuestion: {q['question']}\n\nAnswer:"
        result = query_ollama(model, prompt)
        results.append({
            "id": q["id"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "response": result["response"],
            "tokens": result["eval_count"],
            "duration_s": result["total_duration_s"],
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1}/{total}] {rate:.1f} q/s, ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"  Completed {model}: {total} questions in {elapsed:.1f}s ({total/elapsed:.1f} q/s)")
    return results


def score_results(questions, model_results):
    """Score model results against eval questions."""
    sys.path.insert(0, str(EVAL_DIR))
    from score import score_question

    scored = []
    for q, r in zip(questions, model_results):
        score_result = score_question(q, r["response"])
        score_result["id"] = q["id"]
        score_result["category"] = q.get("category", "")
        score_result["difficulty"] = q.get("difficulty", "")
        scored.append(score_result)
    return scored


def generate_comparison(models_data, output_path):
    """Generate a comparison table and save results."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON — Sutra Eval")
    print(f"{'='*80}")

    # Header
    models = list(models_data.keys())
    header = f"{'Category':<25s}"
    for m in models:
        header += f" {m:>15s}"
    print(header)
    print("-" * len(header))

    # Collect scores by category per model
    all_categories = set()
    all_difficulties = set()
    for model, data in models_data.items():
        for item in data["scored"]:
            all_categories.add(item.get("category", "unknown"))
            all_difficulties.add(item.get("difficulty", "unknown"))

    # By category
    for cat in sorted(all_categories):
        row = f"{cat:<25s}"
        for model in models:
            items = [s for s in models_data[model]["scored"] if s.get("category") == cat]
            scored_items = [s for s in items if s.get("score") is not None]
            if scored_items:
                avg = sum(s["score"] for s in scored_items) / len(scored_items)
                row += f" {avg:>14.1%}"
            else:
                row += f" {'N/A':>15s}"
        print(row)

    # By difficulty
    print("-" * len(header))
    for diff in ["easy", "medium", "hard", "extreme"]:
        if diff not in all_difficulties:
            continue
        row = f"  [{diff}]{'':>{20-len(diff)}s}"
        for model in models:
            items = [s for s in models_data[model]["scored"] if s.get("difficulty") == diff]
            scored_items = [s for s in items if s.get("score") is not None]
            if scored_items:
                avg = sum(s["score"] for s in scored_items) / len(scored_items)
                row += f" {avg:>14.1%}"
            else:
                row += f" {'N/A':>15s}"
        print(row)

    # Overall
    print("-" * len(header))
    row = f"{'OVERALL':<25s}"
    for model in models:
        scored_items = [s for s in models_data[model]["scored"] if s.get("score") is not None]
        if scored_items:
            avg = sum(s["score"] for s in scored_items) / len(scored_items)
            row += f" {avg:>14.1%}"
        else:
            row += f" {'N/A':>15s}"
    print(row)

    # Timing
    print(f"\n{'Avg response time (s)':<25s}", end="")
    for model in models:
        durations = [r["duration_s"] for r in models_data[model]["raw_results"]]
        avg_dur = sum(durations) / len(durations) if durations else 0
        print(f" {avg_dur:>14.2f}s", end="")
    print()

    # Save full results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            model: {
                "scores_by_category": {},
                "scores_by_difficulty": {},
                "overall": None,
                "raw_count": len(data["raw_results"]),
            }
            for model, data in models_data.items()
        },
    }

    for model, data in models_data.items():
        scored = data["scored"]
        # Overall
        scored_items = [s for s in scored if s.get("score") is not None]
        if scored_items:
            save_data["models"][model]["overall"] = sum(s["score"] for s in scored_items) / len(scored_items)
        # By category
        for cat in all_categories:
            items = [s for s in scored if s.get("category") == cat and s.get("score") is not None]
            if items:
                save_data["models"][model]["scores_by_category"][cat] = sum(s["score"] for s in items) / len(items)
        # By difficulty
        for diff in all_difficulties:
            items = [s for s in scored if s.get("difficulty") == diff and s.get("score") is not None]
            if items:
                save_data["models"][model]["scores_by_difficulty"][diff] = sum(s["score"] for s in items) / len(items)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nFull results saved to {output_path}")
    return save_data


def main():
    parser = argparse.ArgumentParser(description="Compare models on Sutra eval")
    parser.add_argument("--models", help="Comma-separated model names (ollama)")
    parser.add_argument("--all-local", action="store_true", help="Run on all installed ollama models")
    parser.add_argument("--eval", default=str(EVAL_DIR / "sutra_eval_500.jsonl"), help="Path to eval JSONL")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit questions per model (for testing)")
    parser.add_argument("--output", default=None, help="Output path")
    args = parser.parse_args()

    if args.all_local:
        models = get_ollama_models()
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        print("Specify --models or --all-local")
        print(f"\nAvailable ollama models: {get_ollama_models()}")
        return

    eval_path = Path(args.eval)
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}")
        print("Build it first by compiling question batches into eval/sutra_eval_500.jsonl")
        return

    questions = load_eval(eval_path)
    print(f"Loaded {len(questions)} eval questions from {eval_path}")
    print(f"Models to compare: {models}")

    models_data = {}
    for model in models:
        raw_results = run_model_on_eval(model, questions, args.max_questions)
        scored = score_results(questions[:len(raw_results)], raw_results)
        models_data[model] = {"raw_results": raw_results, "scored": scored}

    output_path = Path(args.output) if args.output else RESULTS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    generate_comparison(models_data, output_path)


if __name__ == "__main__":
    main()
