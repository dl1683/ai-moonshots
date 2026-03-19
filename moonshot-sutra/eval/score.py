"""Sutra Eval Scoring Framework.

Scores model responses against the 500-question eval set.
Three scoring modes: exact_match, constraint_check, rubric (LLM-as-judge).

Usage:
    python eval/score.py --eval eval/sutra_eval_500.jsonl --responses responses.jsonl --output results.json
    python eval/score.py --eval eval/sutra_eval_500.jsonl --model <model_name> --output results.json
"""

import json
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def score_exact_match(response, expected):
    """Score exact match questions. Returns 0 or 1."""
    if expected is None:
        return None  # Cannot score without expected answer
    # Normalize whitespace and case for comparison
    resp_norm = " ".join(response.lower().split())
    exp_norm = " ".join(expected.lower().split())
    if resp_norm == exp_norm:
        return 1.0
    # Try numeric comparison
    try:
        resp_num = float(response.replace(",", "").strip())
        exp_num = float(expected.replace(",", "").strip())
        if abs(resp_num - exp_num) < 1e-6 * max(abs(exp_num), 1):
            return 1.0
    except (ValueError, AttributeError):
        pass
    # Check if expected answer is contained in response
    if exp_norm in resp_norm:
        return 0.8  # Partial credit for containing the answer
    return 0.0


def score_constraint_check(response, constraints):
    """Score constraint-check questions. Returns fraction of constraints met."""
    if not constraints:
        return None
    met = 0
    results = []
    for constraint in constraints:
        # Basic automated constraint checking
        passed = check_single_constraint(response, constraint)
        results.append({"constraint": constraint, "met": passed})
        if passed:
            met += 1
    return {
        "score": met / len(constraints),
        "met": met,
        "total": len(constraints),
        "details": results,
    }


def check_single_constraint(response, constraint):
    """Check a single constraint. Returns True/False.

    This handles common programmatic constraints. Complex constraints
    fall back to manual/LLM review (returns None -> treated as unscored).
    """
    c = constraint.lower()

    # Word count constraints
    m = re.search(r"exactly (\d+) words", c)
    if m:
        target = int(m.group(1))
        actual = len(response.split())
        return actual == target

    # Sentence count constraints
    m = re.search(r"exactly (\d+) sentences?", c)
    if m:
        target = int(m.group(1))
        # Simple sentence counting by terminal punctuation
        actual = len(re.findall(r"[.!?]+", response))
        return actual == target

    # Paragraph count constraints
    m = re.search(r"(\d+) paragraphs?", c)
    if m:
        target = int(m.group(1))
        actual = len([p for p in response.split("\n\n") if p.strip()])
        return actual == target

    # "does not contain" / "do not use" constraints
    m = re.search(r"(?:does not contain|do not use|no|without)(?: the word[s]?)? ['\"]?(\w+)['\"]?", c)
    if m:
        forbidden = m.group(1).lower()
        return forbidden not in response.lower()

    # "starts with" constraints
    m = re.search(r"starts? with (?:a )?(\w+)", c)
    if m:
        target = m.group(1).lower()
        first_word = response.split()[0].lower() if response.split() else ""
        return first_word == target or response.lower().startswith(target)

    # "ends with" constraints
    m = re.search(r"ends? with (?:a )?(\w+)", c)
    if m:
        target = m.group(1).lower()
        return response.strip().lower().endswith(target)

    # Cannot check programmatically — needs LLM judge
    return None


def score_rubric(response, rubric, question):
    """Score rubric-based questions. Returns structured rubric scores.

    For automated scoring, this returns a placeholder that needs LLM-as-judge.
    """
    if not rubric:
        return None
    return {
        "score": None,  # Needs LLM judge
        "rubric_items": rubric,
        "needs_llm_judge": True,
        "question_preview": question[:100],
        "response_preview": response[:200],
    }


def score_question(question_item, response_text):
    """Score a single question based on its scoring type."""
    scoring = question_item.get("scoring", "rubric")

    if scoring == "exact_match":
        expected = question_item.get("expected_answer")
        return {
            "type": "exact_match",
            "score": score_exact_match(response_text, expected),
        }
    elif scoring == "constraint_check":
        constraints = question_item.get("constraints", [])
        result = score_constraint_check(response_text, constraints)
        return {"type": "constraint_check", **result} if result else {"type": "constraint_check", "score": None}
    elif scoring == "rubric":
        rubric = question_item.get("rubric", [])
        result = score_rubric(response_text, rubric, question_item.get("question", ""))
        return {"type": "rubric", **(result or {})}
    else:
        return {"type": "unknown", "score": None}


def generate_report(eval_items, scored_results):
    """Generate a summary report from scored results."""
    report = {
        "total_questions": len(eval_items),
        "scored_questions": 0,
        "needs_llm_judge": 0,
        "by_category": defaultdict(lambda: {"count": 0, "scored": 0, "total_score": 0.0}),
        "by_difficulty": defaultdict(lambda: {"count": 0, "scored": 0, "total_score": 0.0}),
        "by_scoring_type": defaultdict(lambda: {"count": 0, "scored": 0, "total_score": 0.0}),
    }

    for item, result in zip(eval_items, scored_results):
        cat = item.get("category", "unknown")
        diff = item.get("difficulty", "unknown")
        stype = result.get("type", "unknown")

        report["by_category"][cat]["count"] += 1
        report["by_difficulty"][diff]["count"] += 1
        report["by_scoring_type"][stype]["count"] += 1

        score = result.get("score")
        if score is not None:
            report["scored_questions"] += 1
            report["by_category"][cat]["scored"] += 1
            report["by_category"][cat]["total_score"] += score
            report["by_difficulty"][diff]["scored"] += 1
            report["by_difficulty"][diff]["total_score"] += score
            report["by_scoring_type"][stype]["scored"] += 1
            report["by_scoring_type"][stype]["total_score"] += score
        else:
            report["needs_llm_judge"] += 1

    # Compute averages
    for group in [report["by_category"], report["by_difficulty"], report["by_scoring_type"]]:
        for key, val in group.items():
            if val["scored"] > 0:
                val["avg_score"] = val["total_score"] / val["scored"]
            else:
                val["avg_score"] = None

    # Convert defaultdicts to regular dicts for JSON
    report["by_category"] = dict(report["by_category"])
    report["by_difficulty"] = dict(report["by_difficulty"])
    report["by_scoring_type"] = dict(report["by_scoring_type"])

    return report


def main():
    parser = argparse.ArgumentParser(description="Sutra Eval Scoring Framework")
    parser.add_argument("--eval", required=True, help="Path to eval JSONL file")
    parser.add_argument("--responses", help="Path to responses JSONL file (pre-generated)")
    parser.add_argument("--output", default="results/eval_results.json", help="Output path for results")
    args = parser.parse_args()

    eval_items = load_jsonl(args.eval)
    print(f"Loaded {len(eval_items)} eval questions")

    if args.responses:
        responses = load_jsonl(args.responses)
        response_map = {r["id"]: r["response"] for r in responses}
    else:
        print("No responses file provided. Run with --responses to score.")
        # Print eval stats instead
        cats = defaultdict(int)
        diffs = defaultdict(int)
        stypes = defaultdict(int)
        for item in eval_items:
            cats[item.get("category", "unknown")] += 1
            diffs[item.get("difficulty", "unknown")] += 1
            stypes[item.get("scoring", "unknown")] += 1
        print(f"\nBy category: {dict(cats)}")
        print(f"By difficulty: {dict(diffs)}")
        print(f"By scoring type: {dict(stypes)}")
        return

    # Score each question
    scored = []
    for item in eval_items:
        qid = item["id"]
        response_text = response_map.get(qid, "")
        result = score_question(item, response_text)
        result["id"] = qid
        scored.append(result)

    # Generate report
    report = generate_report(eval_items, scored)
    report["individual_scores"] = scored

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResults written to {out_path}")
    print(f"Total: {report['total_questions']} questions")
    print(f"Auto-scored: {report['scored_questions']}")
    print(f"Needs LLM judge: {report['needs_llm_judge']}")
    print(f"\nBy category:")
    for cat, vals in report["by_category"].items():
        avg = f"{vals['avg_score']:.3f}" if vals["avg_score"] is not None else "N/A"
        print(f"  {cat}: {vals['scored']}/{vals['count']} scored, avg={avg}")


if __name__ == "__main__":
    main()
