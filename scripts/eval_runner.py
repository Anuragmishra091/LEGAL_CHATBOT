"""
Legal Chatbot Evaluation Runner.

Runs the chatbot against a test set and computes all evaluation metrics.

Usage:
    python eval_runner.py                     # Run full evaluation
    python eval_runner.py --limit 5           # Run only first 5 questions
    python eval_runner.py --category constitution  # Run only Constitution questions
    python eval_runner.py --no-llm            # Skip faithfulness (no API calls)
"""

import json
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from main import LegalChatbot
from eval_metrics import evaluate_single

TESTSET_PATH = Path(__file__).parent / "eval_testset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"


def load_testset(category: str = None) -> List[Dict]:
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)
    if category:
        testset = [t for t in testset if t.get("category") == category]
    return testset


def run_evaluation(
    limit: int = None,
    category: str = None,
    skip_llm: bool = False,
    chatbot=None,
    on_progress=None,
) -> Dict:
    """
    Run the chatbot on each test question, compute metrics, return summary.

    Args:
        chatbot:     Optional pre-created LegalChatbot instance to reuse.
        on_progress: Optional callable(done: int, total: int) called after each question.
    """
    testset = load_testset(category)
    if limit:
        testset = testset[:limit]

    print(f"=" * 70)
    print(f"  LEGAL CHATBOT EVALUATION")
    print(f"  Questions: {len(testset)}")
    if category:
        print(f"  Category : {category}")
    print(f"=" * 70)

    _chatbot = chatbot if chatbot is not None else LegalChatbot()
    all_results: List[Dict] = []

    for i, test in enumerate(testset, 1):
        qid      = test["id"]
        question = test["question"]
        expected = test.get("expected_answer", "")
        exp_kw   = test.get("expected_keywords", [])
        exp_src  = test.get("expected_sources", [])

        print(f"\n[{i}/{len(testset)}] {qid}: {question[:60]}...")

        # Get chatbot response
        start = time.time()
        result = _chatbot.ask(question, n_context=5)
        elapsed = time.time() - start

        answer  = result["answer"]
        sources = result["sources"]

        print(f"  Response time: {elapsed:.1f}s | Sources: {len(sources)}")
        print(f"  Answer preview: {answer[:100]}...")

        # Compute metrics
        metrics = evaluate_single(
            question=question,
            answer=answer,
            sources=sources,
            expected_answer=expected if expected else None,
            expected_keywords=exp_kw if exp_kw else None,
            expected_sources=exp_src if exp_src else None,
        )

        # Skip LLM-based faithfulness if requested
        if skip_llm:
            metrics.pop("faithfulness", None)

        # Print individual scores
        for metric, score in metrics.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    {metric:22s}: {score:.3f} {bar}")

        all_results.append({
            "id":       qid,
            "question": question,
            "category": test.get("category", ""),
            "answer":   answer[:500],
            "sources":  [s.get("section", "") for s in sources],
            "metrics":  metrics,
            "time_s":   round(elapsed, 2),
        })

        if on_progress:
            on_progress(i, len(testset))

    # ── Aggregate scores ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    metric_names = set()
    for r in all_results:
        metric_names.update(r["metrics"].keys())

    summary = {}
    for metric in sorted(metric_names):
        scores = [r["metrics"][metric] for r in all_results if metric in r["metrics"]]
        if scores:
            avg = sum(scores) / len(scores)
            summary[metric] = {
                "mean":  round(avg, 3),
                "min":   round(min(scores), 3),
                "max":   round(max(scores), 3),
                "count": len(scores),
            }
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            print(f"  {metric:22s}: {avg:.3f} {bar}  (min={min(scores):.2f}, max={max(scores):.2f})")

    # Category breakdown
    categories = set(r["category"] for r in all_results if r["category"])
    if len(categories) > 1:
        print(f"\n{'─' * 70}")
        print(f"  PER-CATEGORY BREAKDOWN")
        print(f"{'─' * 70}")
        for cat in sorted(categories):
            cat_results = [r for r in all_results if r["category"] == cat]
            # Average of all metrics for this category
            all_scores = []
            for r in cat_results:
                all_scores.extend(r["metrics"].values())
            if all_scores:
                avg = sum(all_scores) / len(all_scores)
                print(f"  {cat:20s}: avg={avg:.3f} ({len(cat_results)} questions)")

    # Overall score (mean of all metric means)
    overall = sum(s["mean"] for s in summary.values()) / len(summary) if summary else 0
    print(f"\n  {'OVERALL SCORE':22s}: {overall:.3f}")
    print(f"{'=' * 70}")

    # Save results
    output = {
        "summary":    summary,
        "overall":    round(overall, 3),
        "details":    all_results,
        "test_count": len(all_results),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {RESULTS_PATH}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate Legal Chatbot")
    parser.add_argument("--limit",    type=int, default=None, help="Run only first N questions")
    parser.add_argument("--category", type=str, default=None,
                        choices=["constitution", "bns", "bnss", "bsa", "judgment", "cross_domain"],
                        help="Filter by category")
    parser.add_argument("--no-llm",   action="store_true",
                        help="Skip LLM-based faithfulness metric (saves API calls)")
    args = parser.parse_args()

    run_evaluation(
        limit=args.limit,
        category=args.category,
        skip_llm=args.no_llm,
    )


if __name__ == "__main__":
    main()
