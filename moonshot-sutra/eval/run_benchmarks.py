"""Sutra Unified Eval Runner.

Runs all benchmarks against a model and produces a unified report:
  1. Standard benchmarks via lm-evaluation-harness (MMLU, GSM8K, ARC, etc.)
  2. Custom Sutra 500-question eval
  3. ARC-AGI-2 abstract reasoning

Usage:
    # Run all benchmarks on an ollama model
    python eval/run_benchmarks.py --model ollama:qwen3:4b --all

    # Run only standard benchmarks
    python eval/run_benchmarks.py --model ollama:qwen3:4b --standard

    # Run only custom Sutra eval
    python eval/run_benchmarks.py --model ollama:qwen3:4b --sutra

    # Run only ARC-AGI-2
    python eval/run_benchmarks.py --arc

    # Run specific benchmarks
    python eval/run_benchmarks.py --model ollama:qwen3:4b --benchmarks mmlu,gsm8k,arc_challenge

    # Run on a HuggingFace model
    python eval/run_benchmarks.py --model hf:Qwen/Qwen3-4B --standard
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
EVAL_DIR = REPO_ROOT / "eval"
ARC_AGI_DIR = Path("C:/Users/devan/OneDrive/Desktop/Projects/ARC AGI 2")

# Standard benchmarks available via lm-evaluation-harness
STANDARD_BENCHMARKS = {
    # Core intelligence benchmarks
    "mmlu": {"description": "Massive Multitask Language Understanding (57 subjects)", "metric": "acc"},
    "gsm8k": {"description": "Grade School Math (8.5K problems)", "metric": "exact_match"},
    "arc_challenge": {"description": "AI2 Reasoning Challenge (hard)", "metric": "acc_norm"},
    "arc_easy": {"description": "AI2 Reasoning Challenge (easy)", "metric": "acc_norm"},
    "hellaswag": {"description": "Sentence completion (commonsense)", "metric": "acc_norm"},
    "winogrande": {"description": "Winograd schema (coreference)", "metric": "acc"},
    "truthfulqa": {"description": "Truthfulness under adversarial framing", "metric": "mc2"},
    # Advanced reasoning
    "gpqa": {"description": "Graduate-level QA (PhD-level science)", "metric": "acc_norm"},
    "bbh": {"description": "BIG-Bench Hard (23 challenging tasks)", "metric": "acc"},
    # Instruction following
    "ifeval": {"description": "Instruction Following Eval", "metric": "prompt_level_strict_acc"},
}

# Benchmark groups for convenience
BENCHMARK_GROUPS = {
    "core": ["mmlu", "gsm8k", "arc_challenge", "hellaswag", "winogrande", "truthfulqa"],
    "reasoning": ["gsm8k", "arc_challenge", "gpqa", "bbh"],
    "all_standard": list(STANDARD_BENCHMARKS.keys()),
}


def parse_model_spec(model_str):
    """Parse model specification string.

    Formats:
        ollama:model_name  -> use ollama backend
        hf:org/model_name  -> use huggingface backend
        local:/path/to/model -> use local model path
    """
    if ":" not in model_str:
        # Default to ollama
        return "ollama", model_str

    backend, model_id = model_str.split(":", 1)
    return backend.lower(), model_id


def run_lm_eval(model_backend, model_id, benchmarks, output_dir, batch_size=1, num_fewshot=None):
    """Run lm-evaluation-harness benchmarks."""
    results = {}

    # Build the lm_eval command based on backend
    if model_backend == "ollama":
        # Use local-completions or openai-completions backend for ollama
        model_args = f"model={model_id},base_url=http://localhost:11434/v1,tokenizer_backend=huggingface,num_concurrent=1"
        model_type = "local-completions"
    elif model_backend == "hf":
        model_args = f"pretrained={model_id},trust_remote_code=True"
        model_type = "hf"
    else:
        print(f"Unknown backend: {model_backend}")
        return results

    task_str = ",".join(benchmarks)
    output_path = output_dir / f"lm_eval_{model_id.replace('/', '_').replace(':', '_')}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", task_str,
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
        "--log_samples",
    ]

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    print(f"\n{'='*60}")
    print(f"Running lm-eval: {task_str}")
    print(f"Model: {model_backend}:{model_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            print(f"lm-eval failed:\n{proc.stderr[:2000]}")
            results["error"] = proc.stderr[:500]
        else:
            # Parse results from output
            results["raw_output"] = proc.stdout[-3000:]  # Last 3K chars
            # Try to load the JSON results
            result_files = list(output_path.rglob("results.json"))
            if result_files:
                with open(result_files[0]) as f:
                    results["parsed"] = json.load(f)
    except subprocess.TimeoutExpired:
        results["error"] = "Timeout (2 hours)"
    except Exception as e:
        results["error"] = str(e)

    return results


def run_sutra_eval(model_backend, model_id, eval_path, output_dir):
    """Run the custom Sutra 500-question eval."""
    if not eval_path.exists():
        print(f"Sutra eval not found at {eval_path}")
        return {"error": "Eval file not found"}

    print(f"\n{'='*60}")
    print(f"Running Sutra Custom Eval")
    print(f"Model: {model_backend}:{model_id}")
    print(f"Eval: {eval_path}")
    print(f"{'='*60}\n")

    # Load eval questions
    questions = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions")

    # Generate responses via ollama
    if model_backend == "ollama":
        return run_sutra_eval_ollama(model_id, questions, output_dir)
    elif model_backend == "hf":
        return run_sutra_eval_hf(model_id, questions, output_dir)
    else:
        return {"error": f"Unsupported backend: {model_backend}"}


def run_sutra_eval_ollama(model_id, questions, output_dir):
    """Run Sutra eval using ollama API."""
    import urllib.request

    responses = []
    total = len(questions)

    for i, q in enumerate(questions):
        qid = q["id"]
        question_text = q["question"]

        # Build prompt — append /no_think for thinking models to get direct answers
        prompt = f"Answer the following question carefully and completely.\n\nQuestion: {question_text}\n\nAnswer:"

        payload = json.dumps({
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 4096},
        }).encode()

        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                # Handle thinking models (Qwen3 etc.) — combine thinking + response
                thinking = result.get("thinking", "")
                response_text = result.get("response", "")
                if not response_text and thinking:
                    response_text = thinking  # Use thinking if response is empty
        except Exception as e:
            response_text = f"ERROR: {e}"

        responses.append({
            "id": qid,
            "response": response_text,
            "model": model_id,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {qid}: {response_text[:80]}...")

    # Save responses
    resp_path = output_dir / f"sutra_responses_{model_id.replace(':', '_')}.jsonl"
    with open(resp_path, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Score using our scoring framework
    sys.path.insert(0, str(EVAL_DIR))
    from score import score_question, generate_report

    scored = []
    for q, r in zip(questions, responses):
        result = score_question(q, r["response"])
        result["id"] = q["id"]
        scored.append(result)

    report = generate_report(questions, scored)

    return {
        "responses_path": str(resp_path),
        "report": report,
    }


def run_sutra_eval_hf(model_id, questions, output_dir):
    """Run Sutra eval using HuggingFace transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return {"error": "transformers not installed"}

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    responses = []
    total = len(questions)

    for i, q in enumerate(questions):
        qid = q["id"]
        prompt = f"Answer the following question carefully and completely.\n\nQuestion: {q['question']}\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False,
            )
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        responses.append({"id": qid, "response": response_text, "model": model_id})

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {qid}: {response_text[:80]}...")

    # Save and score
    resp_path = output_dir / f"sutra_responses_{model_id.replace('/', '_')}.jsonl"
    with open(resp_path, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sys.path.insert(0, str(EVAL_DIR))
    from score import score_question, generate_report

    scored = []
    for q, r in zip(questions, responses):
        result = score_question(q, r["response"])
        result["id"] = q["id"]
        scored.append(result)

    report = generate_report(questions, scored)
    return {"responses_path": str(resp_path), "report": report}


def run_arc_agi2():
    """Run ARC-AGI-2 evaluation using the existing solver."""
    if not ARC_AGI_DIR.exists():
        return {"error": f"ARC-AGI-2 not found at {ARC_AGI_DIR}"}

    print(f"\n{'='*60}")
    print(f"Running ARC-AGI-2")
    print(f"Directory: {ARC_AGI_DIR}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, str(ARC_AGI_DIR / "run.py"), "--eval"]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(ARC_AGI_DIR),
        )
        return {
            "stdout": proc.stdout[-3000:],
            "stderr": proc.stderr[-1000:] if proc.returncode != 0 else "",
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Timeout (1 hour)"}
    except Exception as e:
        return {"error": str(e)}


def generate_unified_report(model_spec, standard_results, sutra_results, arc_results, output_path):
    """Generate a unified report combining all benchmark results."""
    report = {
        "model": model_spec,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    # Standard benchmarks
    if standard_results and "parsed" in standard_results:
        parsed = standard_results["parsed"]
        if "results" in parsed:
            for task_name, task_results in parsed["results"].items():
                clean_name = task_name.split(",")[0] if "," in task_name else task_name
                bench_info = STANDARD_BENCHMARKS.get(clean_name, {})
                metric_key = bench_info.get("metric", "acc")
                score = None
                for key, val in task_results.items():
                    if metric_key in key:
                        score = val
                        break
                report["benchmarks"][clean_name] = {
                    "description": bench_info.get("description", ""),
                    "score": score,
                    "metric": metric_key,
                    "raw": task_results,
                }

    # Sutra custom eval
    if sutra_results and "report" in sutra_results:
        sutra_report = sutra_results["report"]
        report["benchmarks"]["sutra_custom"] = {
            "description": "Sutra 500-question custom eval (reasoning, strategy, synthesis)",
            "total_questions": sutra_report.get("total_questions", 0),
            "auto_scored": sutra_report.get("scored_questions", 0),
            "needs_llm_judge": sutra_report.get("needs_llm_judge", 0),
            "by_category": sutra_report.get("by_category", {}),
            "by_difficulty": sutra_report.get("by_difficulty", {}),
        }

    # ARC-AGI-2
    if arc_results and "error" not in arc_results:
        report["benchmarks"]["arc_agi_2"] = {
            "description": "ARC-AGI-2 Abstract Reasoning Challenge",
            "raw_output": arc_results.get("stdout", ""),
        }

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"UNIFIED EVAL REPORT")
    print(f"Model: {model_spec}")
    print(f"{'='*60}")
    for name, data in report["benchmarks"].items():
        score = data.get("score")
        metric = data.get("metric", "")
        if score is not None:
            print(f"  {name:25s} {score:.4f} ({metric})")
        else:
            desc = data.get("description", "")
            print(f"  {name:25s} {desc}")
    print(f"\nFull report: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Sutra Unified Eval Runner")
    parser.add_argument("--model", help="Model spec (e.g., ollama:qwen3:4b, hf:Qwen/Qwen3-4B)")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--standard", action="store_true", help="Run standard lm-eval benchmarks")
    parser.add_argument("--sutra", action="store_true", help="Run custom Sutra eval")
    parser.add_argument("--arc", action="store_true", help="Run ARC-AGI-2")
    parser.add_argument("--benchmarks", help="Comma-separated list of specific benchmarks")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for lm-eval")
    parser.add_argument("--output", default=None, help="Output path for unified report")
    args = parser.parse_args()

    if not any([args.all, args.standard, args.sutra, args.arc, args.benchmarks]):
        parser.print_help()
        print("\nAvailable standard benchmarks:")
        for name, info in STANDARD_BENCHMARKS.items():
            print(f"  {name:25s} {info['description']}")
        print(f"\nAvailable ollama models:")
        subprocess.run(["ollama", "list"], check=False)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_spec = args.model or "none"
    backend, model_id = parse_model_spec(model_spec) if args.model else ("none", "none")

    standard_results = None
    sutra_results = None
    arc_results = None

    # Standard benchmarks
    if args.all or args.standard or args.benchmarks:
        if not args.model:
            print("ERROR: --model required for standard/sutra benchmarks")
            return

        if args.benchmarks:
            benchmarks = args.benchmarks.split(",")
        elif args.standard or args.all:
            benchmarks = BENCHMARK_GROUPS["core"]
        else:
            benchmarks = []

        if benchmarks:
            standard_results = run_lm_eval(backend, model_id, benchmarks, RESULTS_DIR, args.batch_size)

    # Custom Sutra eval
    if args.all or args.sutra:
        if not args.model:
            print("ERROR: --model required for Sutra eval")
            return
        eval_path = EVAL_DIR / "sutra_eval_500.jsonl"
        sutra_results = run_sutra_eval(backend, model_id, eval_path, RESULTS_DIR)

    # ARC-AGI-2
    if args.all or args.arc:
        arc_results = run_arc_agi2()

    # Unified report
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_unified_{model_id.replace(':', '_').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    generate_unified_report(model_spec, standard_results, sutra_results, arc_results, output_path)


if __name__ == "__main__":
    main()
