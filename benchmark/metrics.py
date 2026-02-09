"""Evaluation metrics for the benchmark.

Compares agent answers against expected values with tolerance,
and aggregates results per agent and per category.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class AgentResult:
    """Standardized result from any agent variant."""

    agent_name: str
    question: str
    raw_answer: str  # full text answer from the agent
    extracted_number: float | None  # parsed numeric answer, or None
    sandbox_calls: int
    sandbox_success: bool  # did at least one sandbox call succeed?
    wall_time_s: float  # total wall-clock seconds
    error: str | None = None


@dataclass
class QuestionMetrics:
    """Metrics for a single question across one agent."""

    question: str
    category: str
    agent_name: str
    correct: bool
    execution_success: bool
    extracted_number: float | None
    expected_answer: float
    absolute_error: float | None
    wall_time_s: float
    sandbox_calls: int
    error: str | None


@dataclass
class AggregateMetrics:
    """Summary metrics for one agent across all questions."""

    agent_name: str
    total_questions: int
    correct_count: int
    accuracy: float
    execution_success_count: int
    execution_success_rate: float
    mean_wall_time_s: float
    median_wall_time_s: float
    p95_wall_time_s: float
    mean_sandbox_calls: float
    total_errors: int
    category_accuracy: dict[str, float] = field(default_factory=dict)


# ── Evaluation functions ────────────────────────────────────────────


def evaluate_single(
    question: str,
    category: str,
    expected_answer: float,
    tolerance: float,
    result: AgentResult,
) -> QuestionMetrics:
    """Evaluate a single agent result against the expected answer."""
    extracted = result.extracted_number
    expected = expected_answer

    if extracted is not None:
        abs_error = abs(extracted - expected)
        correct = abs_error <= tolerance
    else:
        abs_error = None
        correct = False

    return QuestionMetrics(
        question=question,
        category=category,
        agent_name=result.agent_name,
        correct=correct,
        execution_success=result.sandbox_success,
        extracted_number=extracted,
        expected_answer=expected,
        absolute_error=abs_error,
        wall_time_s=result.wall_time_s,
        sandbox_calls=result.sandbox_calls,
        error=result.error,
    )


def aggregate(metrics: list[QuestionMetrics]) -> AggregateMetrics:
    """Compute aggregate statistics from per-question metrics."""
    if not metrics:
        raise ValueError("No metrics to aggregate")

    agent_name = metrics[0].agent_name
    total = len(metrics)
    correct_count = sum(1 for m in metrics if m.correct)
    exec_success = sum(1 for m in metrics if m.execution_success)
    errors = sum(1 for m in metrics if m.error is not None)

    times = sorted(m.wall_time_s for m in metrics)
    mean_time = sum(times) / len(times)
    median_time = times[len(times) // 2]
    p95_idx = int(len(times) * 0.95)
    p95_time = times[min(p95_idx, len(times) - 1)]

    mean_calls = sum(m.sandbox_calls for m in metrics) / total

    # Per-category breakdown
    cat_correct: dict[str, int] = defaultdict(int)
    cat_total: dict[str, int] = defaultdict(int)
    for m in metrics:
        cat_total[m.category] += 1
        if m.correct:
            cat_correct[m.category] += 1
    cat_accuracy = {
        cat: cat_correct[cat] / cat_total[cat] for cat in sorted(cat_total)
    }

    return AggregateMetrics(
        agent_name=agent_name,
        total_questions=total,
        correct_count=correct_count,
        accuracy=correct_count / total,
        execution_success_count=exec_success,
        execution_success_rate=exec_success / total,
        mean_wall_time_s=mean_time,
        median_wall_time_s=median_time,
        p95_wall_time_s=p95_time,
        mean_sandbox_calls=mean_calls,
        total_errors=errors,
        category_accuracy=cat_accuracy,
    )
