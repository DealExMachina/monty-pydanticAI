"""Main benchmark runner.

Runs all three agent variants on the financial dataset and produces
a comparison report with per-question and aggregate results.

Usage:
    uv run python -m benchmark.run
    uv run python -m benchmark.run --no-pydantic
    uv run python -m benchmark.run --no-dspy --no-optimized
    uv run python -m benchmark.run --limit 5     # quick test with 5 questions
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import dspy

from benchmark.dataset import FinancialExample, load_dataset
from benchmark.metrics import AgentResult, AggregateMetrics, QuestionMetrics, aggregate, evaluate_single
from benchmark.agent_pydantic import run_pydantic_agent
from benchmark.agent_dspy import configure_dspy_lm, make_tool_function, run_dspy_agent
from benchmark.agent_dspy_optimized import load_optimized_agent, OPTIMIZED_STATE_PATH
from benchmark.sandbox import SandboxSession

RESULTS_DIR = Path("benchmark/results")

# ── Display helpers ─────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    print(f"\n{BOLD}{'=' * 64}")
    print(f"  {text}")
    print(f"{'=' * 64}{RESET}\n")


def status_icon(correct: bool) -> str:
    return f"{GREEN}✓{RESET}" if correct else f"{RED}✗{RESET}"


# ── Runners ─────────────────────────────────────────────────────────


async def run_pydantic_benchmark(
    examples: list[FinancialExample],
) -> list[tuple[FinancialExample, AgentResult]]:
    """Run all examples through PydanticAI agent."""
    results = []
    for i, ex in enumerate(examples):
        print(
            f"  [{i + 1:3d}/{len(examples)}] {ex.category:20s} "
            f"{ex.question[:50]}...",
            end="",
            flush=True,
        )
        result = await run_pydantic_agent(ex.question)
        ok = "✓" if result.extracted_number is not None and abs(
            result.extracted_number - ex.expected_answer
        ) <= ex.tolerance else "✗"
        print(
            f"  {ok}  got={result.extracted_number}  "
            f"exp={ex.expected_answer:.4f}  "
            f"t={result.wall_time_s:.1f}s  "
            f"calls={result.sandbox_calls}"
        )
        results.append((ex, result))
    return results


def run_dspy_benchmark(
    examples: list[FinancialExample],
    optimized: bool = False,
) -> list[tuple[FinancialExample, AgentResult]]:
    """Run all examples through DSPy ReAct agent."""
    lm = configure_dspy_lm()
    dspy.configure(lm=lm)

    agent_name = "dspy_optimized" if optimized else "dspy_react"

    # Load optimized agent if requested
    compiled_react = None
    if optimized:
        compiled_react = load_optimized_agent()
        if compiled_react is None:
            print(f"  {YELLOW}WARNING: No optimized state at {OPTIMIZED_STATE_PATH}{RESET}")
            print(f"  Run 'python -m benchmark.agent_dspy_optimized' first.")
            print(f"  Falling back to vanilla DSPy ReAct.{RESET}")

    results = []
    for i, ex in enumerate(examples):
        print(
            f"  [{i + 1:3d}/{len(examples)}] {ex.category:20s} "
            f"{ex.question[:50]}...",
            end="",
            flush=True,
        )

        if compiled_react is not None:
            # Rebuild with fresh session + load optimized state
            session = SandboxSession(max_retries=5)
            tool_fn = make_tool_function(session)
            fresh_react = dspy.ReAct(
                signature="question -> answer",
                tools=[tool_fn],
                max_iters=5,
            )
            fresh_react.load(OPTIMIZED_STATE_PATH)
            result = run_dspy_agent(ex.question, react=fresh_react, agent_name=agent_name)
        else:
            result = run_dspy_agent(ex.question, agent_name=agent_name)

        ok = "✓" if result.extracted_number is not None and abs(
            result.extracted_number - ex.expected_answer
        ) <= ex.tolerance else "✗"
        print(
            f"  {ok}  got={result.extracted_number}  "
            f"exp={ex.expected_answer:.4f}  "
            f"t={result.wall_time_s:.1f}s  "
            f"calls={result.sandbox_calls}"
        )
        results.append((ex, result))
    return results


# ── Summary table ───────────────────────────────────────────────────


def print_summary_table(summaries: list[AggregateMetrics]) -> None:
    """Print a side-by-side comparison table."""
    col_w = 22
    headers = ["Metric"] + [s.agent_name for s in summaries]

    sep = "+" + "+".join("-" * col_w for _ in headers) + "+"
    eq_sep = "+" + "+".join("=" * col_w for _ in headers) + "+"

    print(sep)
    print("|" + "|".join(h.center(col_w) for h in headers) + "|")
    print(eq_sep)

    rows = [
        ("Accuracy", [f"{s.accuracy:.1%}" for s in summaries]),
        ("Correct / Total", [f"{s.correct_count}/{s.total_questions}" for s in summaries]),
        ("Exec Success Rate", [f"{s.execution_success_rate:.1%}" for s in summaries]),
        ("Mean Latency (s)", [f"{s.mean_wall_time_s:.2f}" for s in summaries]),
        ("Median Latency (s)", [f"{s.median_wall_time_s:.2f}" for s in summaries]),
        ("P95 Latency (s)", [f"{s.p95_wall_time_s:.2f}" for s in summaries]),
        ("Mean Sandbox Calls", [f"{s.mean_sandbox_calls:.2f}" for s in summaries]),
        ("Total Errors", [str(s.total_errors) for s in summaries]),
    ]

    for label, values in rows:
        cells = [label] + values
        print("|" + "|".join(c.center(col_w) for c in cells) + "|")

    print(sep)

    # Category breakdown
    print(f"\n{BOLD}Per-category accuracy:{RESET}")
    all_cats = sorted(
        set(cat for s in summaries for cat in s.category_accuracy)
    )
    cat_header = ["Category"] + [s.agent_name for s in summaries]
    print("  " + " | ".join(h.ljust(20) for h in cat_header))
    print("  " + "-" * (22 * len(cat_header)))
    for cat in all_cats:
        row = [cat]
        for s in summaries:
            val = s.category_accuracy.get(cat, 0.0)
            row.append(f"{val:.0%}")
        print("  " + " | ".join(v.ljust(20) for v in row))


# ── Save results ────────────────────────────────────────────────────


def save_results(
    all_qm: dict[str, list[QuestionMetrics]],
    summaries: list[AggregateMetrics],
) -> Path:
    """Save all results to a timestamped JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    data: dict = {
        "timestamp": timestamp,
        "model": "qwen3:8b",
        "summaries": {},
        "per_question": {},
    }

    for s in summaries:
        data["summaries"][s.agent_name] = {
            "accuracy": s.accuracy,
            "correct_count": s.correct_count,
            "total_questions": s.total_questions,
            "execution_success_rate": s.execution_success_rate,
            "mean_wall_time_s": s.mean_wall_time_s,
            "median_wall_time_s": s.median_wall_time_s,
            "p95_wall_time_s": s.p95_wall_time_s,
            "mean_sandbox_calls": s.mean_sandbox_calls,
            "total_errors": s.total_errors,
            "category_accuracy": s.category_accuracy,
        }

    for agent_name, qm_list in all_qm.items():
        data["per_question"][agent_name] = [
            {
                "question": m.question,
                "category": m.category,
                "correct": m.correct,
                "execution_success": m.execution_success,
                "extracted_number": m.extracted_number,
                "expected_answer": m.expected_answer,
                "absolute_error": m.absolute_error,
                "wall_time_s": m.wall_time_s,
                "sandbox_calls": m.sandbox_calls,
                "error": m.error,
            }
            for m in qm_list
        ]

    path.write_text(json.dumps(data, indent=2, default=str))
    return path


# ── Main ────────────────────────────────────────────────────────────


async def main(
    run_pydantic: bool = True,
    run_dspy_vanilla: bool = True,
    run_dspy_opt: bool = True,
    limit: int | None = None,
) -> None:
    """Run the full benchmark."""
    examples = load_dataset()
    if limit:
        examples = examples[:limit]
    print(f"Loaded {len(examples)} financial calculation examples.\n")

    all_qm: dict[str, list[QuestionMetrics]] = {}
    summaries: list[AggregateMetrics] = []

    # --- Agent A: PydanticAI ---
    if run_pydantic:
        print_header("Agent A: PydanticAI + Monty")
        results = await run_pydantic_benchmark(examples)
        qm_list = [
            evaluate_single(ex.question, ex.category, ex.expected_answer, ex.tolerance, res)
            for ex, res in results
        ]
        agg = aggregate(qm_list)
        all_qm["pydantic_ai"] = qm_list
        summaries.append(agg)
        print(f"\n  {BOLD}Accuracy: {agg.accuracy:.1%}{RESET}")

    # --- Agent B: DSPy ReAct (vanilla) ---
    if run_dspy_vanilla:
        print_header("Agent B: DSPy ReAct (vanilla)")
        results = run_dspy_benchmark(examples, optimized=False)
        qm_list = [
            evaluate_single(ex.question, ex.category, ex.expected_answer, ex.tolerance, res)
            for ex, res in results
        ]
        agg = aggregate(qm_list)
        all_qm["dspy_react"] = qm_list
        summaries.append(agg)
        print(f"\n  {BOLD}Accuracy: {agg.accuracy:.1%}{RESET}")

    # --- Agent C: DSPy ReAct (optimized) ---
    if run_dspy_opt:
        print_header("Agent C: DSPy ReAct (optimized)")
        results = run_dspy_benchmark(examples, optimized=True)
        qm_list = [
            evaluate_single(ex.question, ex.category, ex.expected_answer, ex.tolerance, res)
            for ex, res in results
        ]
        agg = aggregate(qm_list)
        all_qm["dspy_optimized"] = qm_list
        summaries.append(agg)
        print(f"\n  {BOLD}Accuracy: {agg.accuracy:.1%}{RESET}")

    # --- Summary ---
    if summaries:
        print_header("COMPARISON SUMMARY")
        print_summary_table(summaries)

        path = save_results(all_qm, summaries)
        print(f"\n{DIM}Results saved to: {path}{RESET}")


if __name__ == "__main__":
    flags = set(sys.argv[1:])

    run_p = "--no-pydantic" not in flags
    run_d = "--no-dspy" not in flags
    run_o = "--no-optimized" not in flags

    # Parse --limit N
    limit = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--limit" and i < len(sys.argv) - 1:
            limit = int(sys.argv[i + 1])

    asyncio.run(main(run_pydantic=run_p, run_dspy_vanilla=run_d, run_dspy_opt=run_o, limit=limit))
