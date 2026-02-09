"""DSPy BootstrapFewShot optimization for the ReAct agent.

Run standalone:  python -m benchmark.agent_dspy_optimized
This trains the agent on 70% of the dataset and saves the optimized state.
"""

from __future__ import annotations

import os
import re

import dspy
from dspy.teleprompt import BootstrapFewShot

from benchmark.agent_dspy import configure_dspy_lm, make_tool_function, run_dspy_agent
from benchmark.dataset import as_dspy_examples, load_dataset
from benchmark.sandbox import SandboxSession

OPTIMIZED_STATE_PATH = "benchmark/optimized_react_state.json"


# ── Metric ──────────────────────────────────────────────────────────


def optimization_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    """Does the predicted answer match expected within tolerance?

    Returns 1.0 if correct, 0.0 otherwise.
    Uses 1% relative tolerance (minimum 0.01 absolute).
    """
    try:
        raw = pred.answer if hasattr(pred, "answer") else str(pred)
        extracted = None

        # Try ANSWER: tag first
        m = re.search(r"ANSWER:\s*\$?([-\d,]+\.?\d*)", raw)
        if m:
            extracted = float(m.group(1).replace(",", ""))

        # Fallback: last number in text
        if extracted is None:
            numbers = re.findall(r"-?[\d,]+\.\d+|-?\d{2,}", raw)
            if numbers:
                extracted = float(numbers[-1].replace(",", ""))

        if extracted is None:
            return 0.0

        expected = float(example.answer)
        tolerance = max(abs(expected) * 0.01, 0.01)
        return 1.0 if abs(extracted - expected) <= tolerance else 0.0

    except (ValueError, TypeError):
        return 0.0


# ── Optimization ────────────────────────────────────────────────────


def optimize(
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 8,
    max_rounds: int = 1,
    train_ratio: float = 0.7,
) -> dspy.ReAct:
    """Run BootstrapFewShot on the financial dataset.

    Split: 70 train / 30 test (by default).
    """
    lm = configure_dspy_lm()
    dspy.configure(lm=lm)

    # Load and split
    all_examples = load_dataset()
    split_idx = int(len(all_examples) * train_ratio)
    train_dspy = as_dspy_examples(all_examples[:split_idx])
    test_dspy = as_dspy_examples(all_examples[split_idx:])

    print(f"Train: {len(train_dspy)}, Test: {len(test_dspy)}")

    # Build student
    session = SandboxSession(max_retries=5)
    tool_fn = make_tool_function(session)

    student = dspy.ReAct(
        signature="question -> answer",
        tools=[tool_fn],
        max_iters=5,
    )

    # Optimize
    optimizer = BootstrapFewShot(
        metric=optimization_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        max_errors=20,
    )

    print("Starting BootstrapFewShot compilation...")
    compiled = optimizer.compile(student=student, trainset=train_dspy)

    # Save
    compiled.save(OPTIMIZED_STATE_PATH)
    print(f"Saved optimized state to {OPTIMIZED_STATE_PATH}")

    return compiled


def load_optimized_agent() -> dspy.ReAct | None:
    """Load a previously optimized agent, or None if not found."""
    if not os.path.exists(OPTIMIZED_STATE_PATH):
        return None

    session = SandboxSession(max_retries=5)
    tool_fn = make_tool_function(session)

    react = dspy.ReAct(
        signature="question -> answer",
        tools=[tool_fn],
        max_iters=5,
    )
    react.load(OPTIMIZED_STATE_PATH)
    return react


# ── CLI entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    optimize()
