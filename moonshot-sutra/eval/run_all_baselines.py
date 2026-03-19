"""Run all baseline models on the Sutra eval, one at a time.

Saves per-model responses and a final comparison summary.
Designed to run unattended in background.
"""

import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent.parent
EVAL_PATH = REPO / "eval" / "sutra_eval_500.jsonl"
RESULTS_DIR = REPO / "results"
LEDGER_PATH = REPO / "experiments" / "ledger.jsonl"

MODELS = [
    "granite4:350m",
    "qwen2.5:0.5b",
    "qwen3:0.6b",
    "qwen3:1.7b",
    "llama3.2:3b",
    "granite4:3b",
    "phi3:mini",
    "qwen3:4b",
    "gemma3:4b",
    "mistral:7b",
    "phi4:latest",
]


def query_ollama(model, prompt, max_tokens=4096, timeout=300):
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


def load_eval():
    questions = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_model(model, questions):
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"Questions: {len(questions)}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    responses = []
    start = time.time()

    for i, q in enumerate(questions):
        prompt = f"Answer the following question carefully and completely.\n\nQuestion: {q['question']}\n\nAnswer:"
        result = query_ollama(model, prompt)
        responses.append({
            "id": q["id"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "scoring": q.get("scoring", ""),
            "response": result["response"],
            "tokens": result["eval_count"],
            "duration_s": result["total_duration_s"],
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(questions) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(questions)}] {rate:.2f} q/s, ETA {eta/60:.1f}min", flush=True)

    elapsed = time.time() - start
    print(f"  DONE: {len(questions)} questions in {elapsed/60:.1f}min ({len(questions)/elapsed:.2f} q/s)")

    # Save responses
    safe_name = model.replace(":", "_").replace("/", "_")
    resp_path = RESULTS_DIR / f"baseline_{safe_name}.jsonl"
    with open(resp_path, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return responses, elapsed


def compute_stats(questions, responses):
    """Compute basic stats (constraint_check only for now since rubric needs LLM judge)."""
    sys.path.insert(0, str(REPO / "eval"))
    from score import score_question

    stats = {"total": len(questions), "by_category": defaultdict(lambda: {"n": 0, "scored": 0, "sum": 0.0}),
             "by_difficulty": defaultdict(lambda: {"n": 0, "scored": 0, "sum": 0.0})}

    for q, r in zip(questions, responses):
        result = score_question(q, r["response"])
        cat = q.get("category", "?")
        diff = q.get("difficulty", "?")
        stats["by_category"][cat]["n"] += 1
        stats["by_difficulty"][diff]["n"] += 1

        score = result.get("score")
        if isinstance(score, dict):
            score = score.get("score")
        if score is not None:
            stats["by_category"][cat]["scored"] += 1
            stats["by_category"][cat]["sum"] += score
            stats["by_difficulty"][diff]["scored"] += 1
            stats["by_difficulty"][diff]["sum"] += score

    return stats


def log_to_ledger(model, elapsed, stats):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "id": f"baseline_{model.replace(':', '_')}",
        "purpose": f"Baseline eval of {model} on Sutra 500-question eval",
        "command": f"python eval/run_all_baselines.py (model={model})",
        "metrics": {
            "total_questions": stats["total"],
            "wall_time_s": round(elapsed, 1),
            "by_category": {k: {"n": v["n"], "scored": v["scored"],
                                "avg": round(v["sum"]/v["scored"], 4) if v["scored"] > 0 else None}
                           for k, v in stats["by_category"].items()},
            "by_difficulty": {k: {"n": v["n"], "scored": v["scored"],
                                  "avg": round(v["sum"]/v["scored"], 4) if v["scored"] > 0 else None}
                             for k, v in stats["by_difficulty"].items()},
        },
        "status": "DONE",
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_eval()
    print(f"Loaded {len(questions)} eval questions")
    print(f"Models to run: {len(MODELS)}")
    print(f"Estimated total: ~{len(questions) * len(MODELS) * 3 / 60:.0f} min (assuming ~3s/question avg)\n")

    all_stats = {}
    for model in MODELS:
        responses, elapsed = run_model(model, questions)
        stats = compute_stats(questions, responses)
        log_to_ledger(model, elapsed, stats)
        all_stats[model] = stats
        print(f"\n  Stats for {model}:")
        for cat, v in sorted(stats["by_category"].items()):
            avg = f"{v['sum']/v['scored']:.1%}" if v["scored"] > 0 else "N/A"
            print(f"    {cat:30s}: {v['scored']}/{v['n']} scored, avg={avg}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": list(MODELS),
        "total_questions": len(questions),
        "results": {},
    }
    for model, stats in all_stats.items():
        overall_scored = sum(v["scored"] for v in stats["by_category"].values())
        overall_sum = sum(v["sum"] for v in stats["by_category"].values())
        summary["results"][model] = {
            "overall": round(overall_sum / overall_scored, 4) if overall_scored > 0 else None,
            "scored_count": overall_scored,
            "by_category": {k: round(v["sum"]/v["scored"], 4) if v["scored"] > 0 else None
                           for k, v in stats["by_category"].items()},
        }

    summary_path = RESULTS_DIR / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
