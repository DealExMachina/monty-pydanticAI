"""Main benchmark runner.

Runs four agent variants on the financial dataset and produces
a comparison report with per-question and aggregate results.

Agents:
    A: PydanticAI + Monty      (code execution sandbox)
    B: DSPy vanilla ReAct      (unoptimized, Monty sandbox)
    C: DSPy optimized ReAct    (BootstrapFewShot compiled, Monty sandbox)
    D: PydanticAI + Tool Calls (pre-built quant functions, no Monty)

Usage:
    uv run python -m benchmark.run                     # all 4 agents
    uv run python -m benchmark.run --no-pydantic
    uv run python -m benchmark.run --no-dspy --no-optimized
    uv run python -m benchmark.run --no-tools-baseline # skip Agent D
    uv run python -m benchmark.run --limit 5           # quick test with 5 questions
    uv run python -m benchmark.run --concurrency 8     # parallel requests (default: 4)
    uv run python -m benchmark.run --direct            # include no-tools baseline for sandbox delta
    uv run python -m benchmark.run --dspy-optimized-on-test   # DSPy optimized on 30%% held-out test set
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import dspy

from benchmark.dataset import (
    FinancialExample, load_dataset, load_dataset_split,
    SINGLE_FORMULA_CATEGORIES, CODE_COMPOSITION_CATEGORIES,
)
from benchmark.metrics import AgentResult, AggregateMetrics, QuestionMetrics, aggregate, evaluate_single
from benchmark.agent_pydantic import run_pydantic_agent
from benchmark.agent_direct import run_direct_agent
from benchmark.agent_dspy import configure_dspy_lm, make_tool_function, run_dspy_agent
from benchmark.agent_dspy_optimized import load_optimized_agent, OPTIMIZED_STATE_PATH
from benchmark.agent_tools import run_tools_agent
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


async def _run_pydantic_one(
    ex: FinancialExample,
    semaphore: asyncio.Semaphore,
) -> tuple[FinancialExample, AgentResult]:
    """Run a single example. Semaphore limits concurrent requests."""
    async with semaphore:
        result = await run_pydantic_agent(ex.question)
    return (ex, result)


async def _run_direct_one(
    ex: FinancialExample,
    semaphore: asyncio.Semaphore,
) -> tuple[FinancialExample, AgentResult]:
    """Run a single example through direct (no-tools) agent."""
    async with semaphore:
        result = await run_direct_agent(ex.question)
    return (ex, result)


async def run_direct_benchmark(
    examples: list[FinancialExample],
    concurrency: int = 4,
) -> list[tuple[FinancialExample, AgentResult]]:
    """Run all examples through direct (no-tools) baseline agent."""
    semaphore = asyncio.Semaphore(concurrency)
    if concurrency > 1:
        print(f"  {DIM}Running with concurrency={concurrency}{RESET}\n")

    tasks = [_run_direct_one(ex, semaphore) for ex in examples]
    raw_results = await asyncio.gather(*tasks)

    results = list(raw_results)
    for i, (ex, result) in enumerate(results):
        ok = "✓" if result.extracted_number is not None and abs(
            result.extracted_number - ex.expected_answer
        ) <= ex.tolerance else "✗"
        print(
            f"  [{i + 1:3d}/{len(examples)}] {ex.category:20s} "
            f"{ex.question[:50]}...  "
            f"{ok}  got={result.extracted_number}  "
            f"exp={ex.expected_answer:.4f}  "
            f"t={result.wall_time_s:.1f}s  "
            f"(no tools)"
        )
    return results


async def run_pydantic_benchmark(
    examples: list[FinancialExample],
    concurrency: int = 4,
) -> list[tuple[FinancialExample, AgentResult]]:
    """Run all examples through PydanticAI agent (parallel)."""
    semaphore = asyncio.Semaphore(concurrency)
    if concurrency > 1:
        print(f"  {DIM}Running with concurrency={concurrency}{RESET}\n")

    tasks = [_run_pydantic_one(ex, semaphore) for ex in examples]
    raw_results = await asyncio.gather(*tasks)

    # Print in original order
    results = list(raw_results)
    for i, (ex, result) in enumerate(results):
        ok = "✓" if result.extracted_number is not None and abs(
            result.extracted_number - ex.expected_answer
        ) <= ex.tolerance else "✗"
        print(
            f"  [{i + 1:3d}/{len(examples)}] {ex.category:20s} "
            f"{ex.question[:50]}...  "
            f"{ok}  got={result.extracted_number}  "
            f"exp={ex.expected_answer:.4f}  "
            f"t={result.wall_time_s:.1f}s  "
            f"calls={result.sandbox_calls}"
        )
    return results


async def _run_tools_one(
    ex: FinancialExample,
    semaphore: asyncio.Semaphore,
) -> tuple[FinancialExample, AgentResult]:
    """Run a single example through the pre-built tools agent."""
    async with semaphore:
        result = await run_tools_agent(ex.question)
    return (ex, result)


async def run_tools_benchmark(
    examples: list[FinancialExample],
    concurrency: int = 4,
) -> list[tuple[FinancialExample, AgentResult]]:
    """Run all examples through PydanticAI + pre-built tools agent (parallel)."""
    semaphore = asyncio.Semaphore(concurrency)
    if concurrency > 1:
        print(f"  {DIM}Running with concurrency={concurrency}{RESET}\n")

    tasks = [_run_tools_one(ex, semaphore) for ex in examples]
    raw_results = await asyncio.gather(*tasks)

    results = list(raw_results)
    for i, (ex, result) in enumerate(results):
        ok = "✓" if result.extracted_number is not None and abs(
            result.extracted_number - ex.expected_answer
        ) <= ex.tolerance else "✗"
        print(
            f"  [{i + 1:3d}/{len(examples)}] {ex.category:20s} "
            f"{ex.question[:50]}...  "
            f"{ok}  got={result.extracted_number}  "
            f"exp={ex.expected_answer:.4f}  "
            f"t={result.wall_time_s:.1f}s  "
            f"(tools)"
        )
    return results


def _run_dspy_one(
    ex: FinancialExample,
    compiled_react: dspy.ReAct | None,
    agent_name: str,
) -> tuple[FinancialExample, AgentResult]:
    """Run a single DSPy example (for use in thread)."""
    if compiled_react is not None:
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
    return (ex, result)


async def _run_dspy_one_async(
    ex: FinancialExample,
    compiled_react: dspy.ReAct | None,
    agent_name: str,
    semaphore: asyncio.Semaphore,
) -> tuple[FinancialExample, AgentResult]:
    """Run DSPy in thread pool with semaphore."""
    async with semaphore:
        return await asyncio.to_thread(_run_dspy_one, ex, compiled_react, agent_name)


async def run_dspy_benchmark(
    examples: list[FinancialExample],
    optimized: bool = False,
    concurrency: int = 4,
) -> list[tuple[FinancialExample, AgentResult]]:
    """Run all examples through DSPy ReAct agent (parallel)."""
    lm = configure_dspy_lm()
    dspy.configure(lm=lm)

    agent_name = "dspy_optimized" if optimized else "dspy_react"

    compiled_react = None
    if optimized:
        compiled_react = load_optimized_agent()
        if compiled_react is None:
            print(f"  {YELLOW}WARNING: No optimized state at {OPTIMIZED_STATE_PATH}{RESET}")
            print(f"  Run 'python -m benchmark.agent_dspy_optimized' first.")
            print(f"  Falling back to vanilla DSPy ReAct.{RESET}")

    if concurrency > 1:
        print(f"  {DIM}Running with concurrency={concurrency}{RESET}\n")

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _run_dspy_one_async(ex, compiled_react, agent_name, semaphore)
        for ex in examples
    ]
    raw_results = await asyncio.gather(*tasks)

    results = list(raw_results)
    for i, (ex, result) in enumerate(results):
        ok = "✓" if result.extracted_number is not None and abs(
            result.extracted_number - ex.expected_answer
        ) <= ex.tolerance else "✗"
        print(
            f"  [{i + 1:3d}/{len(examples)}] {ex.category:20s} "
            f"{ex.question[:50]}...  "
            f"{ok}  got={result.extracted_number}  "
            f"exp={ex.expected_answer:.4f}  "
            f"t={result.wall_time_s:.1f}s  "
            f"calls={result.sandbox_calls}"
        )
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

    # Sandbox delta: when direct baseline is present, show improvement over no-tools
    direct_acc = next((s.accuracy for s in summaries if s.agent_name == "direct"), None)
    delta_values = []
    for s in summaries:
        if s.agent_name == "direct":
            delta_values.append("(baseline)")
        elif direct_acc is not None:
            delta = s.accuracy - direct_acc
            sign = "+" if delta >= 0 else ""
            delta_values.append(f"{sign}{delta:.1%}")
        else:
            delta_values.append("-")

    rows = [
        ("Accuracy", [f"{s.accuracy:.1%}" for s in summaries]),
        ("Correct / Total", [f"{s.correct_count}/{s.total_questions}" for s in summaries]),
    ]
    if direct_acc is not None:
        rows.append(("Sandbox Delta", delta_values))
    rows.extend([
        ("Exec Success Rate", [f"{s.execution_success_rate:.1%}" for s in summaries]),
        ("Mean Latency (s)", [f"{s.mean_wall_time_s:.2f}" for s in summaries]),
        ("Median Latency (s)", [f"{s.median_wall_time_s:.2f}" for s in summaries]),
        ("P95 Latency (s)", [f"{s.p95_wall_time_s:.2f}" for s in summaries]),
        ("Mean Sandbox Calls", [f"{s.mean_sandbox_calls:.2f}" for s in summaries]),
        ("Mean Prompt Tokens", [f"{s.mean_prompt_tokens:.0f}" for s in summaries]),
        ("Mean Completion Tokens", [f"{s.mean_completion_tokens:.0f}" for s in summaries]),
        ("Total Tokens", [str(s.total_tokens) for s in summaries]),
        ("Est. Cost (USD)", [f"{s.estimated_cost_usd:.4f}" for s in summaries]),
        ("Total Errors", [str(s.total_errors) for s in summaries]),
    ])

    for label, values in rows:
        cells = [label] + values
        print("|" + "|".join(c.center(col_w) for c in cells) + "|")

    print(sep)

    # Category breakdown by type
    all_cats = sorted(
        set(cat for s in summaries for cat in s.category_accuracy)
    )
    cat_header = ["Category"] + [s.agent_name for s in summaries]

    sf_cats = [c for c in all_cats if c in SINGLE_FORMULA_CATEGORIES]
    cc_cats = [c for c in all_cats if c in CODE_COMPOSITION_CATEGORIES]

    if sf_cats:
        print(f"\n{BOLD}Single-formula categories:{RESET}")
        print("  " + " | ".join(h.ljust(20) for h in cat_header))
        print("  " + "-" * (22 * len(cat_header)))
        for cat in sf_cats:
            row = [cat]
            for s in summaries:
                val = s.category_accuracy.get(cat, 0.0)
                row.append(f"{val:.0%}")
            print("  " + " | ".join(v.ljust(20) for v in row))

    if cc_cats:
        print(f"\n{BOLD}Code-composition categories:{RESET}")
        print("  " + " | ".join(h.ljust(20) for h in cat_header))
        print("  " + "-" * (22 * len(cat_header)))
        for cat in cc_cats:
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
        "model": "nemotron",
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
            "mean_prompt_tokens": s.mean_prompt_tokens,
            "mean_completion_tokens": s.mean_completion_tokens,
            "total_tokens": s.total_tokens,
            "estimated_cost_usd": s.estimated_cost_usd,
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
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
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
    run_tools: bool = True,
    run_direct: bool = False,
    dspy_optimized_on_test: bool = False,
    limit: int | None = None,
    concurrency: int = 4,
) -> None:
    """Run the full benchmark."""
    examples = load_dataset()
    if limit:
        examples = examples[:limit]

    sf_count = sum(1 for ex in examples if ex.category in SINGLE_FORMULA_CATEGORIES)
    cc_count = sum(1 for ex in examples if ex.category in CODE_COMPOSITION_CATEGORIES)
    print(f"Loaded {len(examples)} financial calculation examples "
          f"({sf_count} single-formula, {cc_count} code-composition).\n")

    # When --dspy-optimized-on-test: DSPy optimized runs on held-out test set only
    dspy_opt_examples = examples
    if dspy_optimized_on_test and run_dspy_opt:
        _, test_examples = load_dataset_split(train_ratio=0.7, seed=42)
        if limit:
            test_examples = test_examples[:limit]
        dspy_opt_examples = test_examples
        print(f"{DIM}DSPy optimized will run on {len(dspy_opt_examples)} held-out test examples (70/30 split).{RESET}\n")

    all_qm: dict[str, list[QuestionMetrics]] = {}
    summaries: list[AggregateMetrics] = []

    # --- Baseline: Direct (no tools) ---
    if run_direct:
        print_header("Baseline: Direct (no tools)")
        results = await run_direct_benchmark(examples, concurrency=concurrency)
        qm_list = [
            evaluate_single(ex.question, ex.category, ex.expected_answer, ex.tolerance, res)
            for ex, res in results
        ]
        agg = aggregate(qm_list)
        all_qm["direct"] = qm_list
        summaries.append(agg)
        print(f"\n  {BOLD}Accuracy: {agg.accuracy:.1%}{RESET}")

    # --- Agent D: PydanticAI + Pre-built Tool Calls (naive baseline) ---
    if run_tools:
        print_header("Agent D: PydanticAI + Pre-built Tool Calls")
        results = await run_tools_benchmark(examples, concurrency=concurrency)
        qm_list = [
            evaluate_single(ex.question, ex.category, ex.expected_answer, ex.tolerance, res)
            for ex, res in results
        ]
        agg = aggregate(qm_list)
        all_qm["pydantic_tools"] = qm_list
        summaries.append(agg)
        print(f"\n  {BOLD}Accuracy: {agg.accuracy:.1%}{RESET}")

    # --- Agent A: PydanticAI + Monty ---
    if run_pydantic:
        print_header("Agent A: PydanticAI + Monty")
        results = await run_pydantic_benchmark(examples, concurrency=concurrency)
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
        results = await run_dspy_benchmark(examples, optimized=False, concurrency=concurrency)
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
        results = await run_dspy_benchmark(dspy_opt_examples, optimized=True, concurrency=concurrency)
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
    run_t = "--no-tools-baseline" not in flags
    run_direct = "--direct" in flags
    dspy_optimized_on_test = "--dspy-optimized-on-test" in flags

    limit = None
    concurrency = 4
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
        elif arg == "--concurrency" and i + 1 < len(args):
            concurrency = int(args[i + 1])

    asyncio.run(
        main(
            run_pydantic=run_p,
            run_dspy_vanilla=run_d,
            run_dspy_opt=run_o,
            run_tools=run_t,
            run_direct=run_direct,
            dspy_optimized_on_test=dspy_optimized_on_test,
            limit=limit,
            concurrency=concurrency,
        )
    )
