"""DSPy ReAct agent runner for the benchmark.

Uses dspy.ReAct with a run_python tool that calls the shared Monty sandbox.
Configures DSPy to use qwen3:8b via Ollama.
"""

from __future__ import annotations

import time

import dspy

from benchmark.metrics import AgentResult
from benchmark.sandbox import SandboxSession
from benchmark.agent_pydantic import extract_number

# ── DSPy LM configuration ──────────────────────────────────────────


def configure_dspy_lm() -> dspy.LM:
    """Configure DSPy to use qwen3:8b via Ollama."""
    lm = dspy.LM(
        "ollama_chat/qwen3:8b",
        api_base="http://localhost:11434",
        api_key="",
        temperature=0.3,
        cache=False,
    )
    return lm


# ── Tool factory ────────────────────────────────────────────────────


def make_tool_function(session: SandboxSession):
    """Create a run_python tool function bound to a SandboxSession.

    The returned function has proper type hints and docstring for DSPy
    to infer the tool schema.
    """

    def run_python(code: str) -> str:
        """Execute Python code in a secure Monty sandbox for financial calculations.

        The sandbox provides math functions you can call directly (no import):
        sqrt(x), ln(x), exp(x), norm_cdf(x), norm_pdf(x), abs(x), max_val(a,b).

        Code formatting rules:
        - All top-level statements must start at column 0.
        - Use exactly 4 spaces for indentation inside blocks.
        - The result of the last expression is returned.

        Example (Black-Scholes call):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        d1 = (ln(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        call = S * norm_cdf(d1) - K * exp(-r*T) * norm_cdf(d2)
        round(call, 4)

        Args:
            code: Python source code to execute in the sandbox.
        """
        return session.execute(code)

    return run_python


# ── Runner ──────────────────────────────────────────────────────────


def run_dspy_agent(
    question: str,
    react: dspy.ReAct | None = None,
    agent_name: str = "dspy_react",
) -> AgentResult:
    """Run a single question through the DSPy ReAct agent.

    If react is None, builds a fresh one. If provided (e.g., an optimized
    compiled agent), rebuilds it with a fresh SandboxSession tool.
    """
    session = SandboxSession(max_retries=5)
    tool_fn = make_tool_function(session)

    if react is None:
        react_agent = dspy.ReAct(
            signature="question -> answer",
            tools=[tool_fn],
            max_iters=5,
        )
    else:
        # For optimized agent: we rebuild with fresh tool but same demos
        react_agent = react

    t0 = time.perf_counter()
    try:
        result = react_agent(question=question)
        wall_time = time.perf_counter() - t0

        raw_answer = result.answer if hasattr(result, "answer") else str(result)

        # Primary: use sandbox raw value (the actual computed result)
        extracted = None
        if session.last_successful_value is not None:
            val = session.last_successful_value
            if isinstance(val, (int, float)):
                extracted = float(val)

        # Fallback: parse from agent text if sandbox didn't return a number
        if extracted is None:
            extracted = extract_number(raw_answer)

        return AgentResult(
            agent_name=agent_name,
            question=question,
            raw_answer=raw_answer,
            extracted_number=extracted,
            sandbox_calls=session.call_count,
            sandbox_success=any(r.success for r in session.results),
            wall_time_s=wall_time,
        )
    except Exception as exc:
        wall_time = time.perf_counter() - t0
        return AgentResult(
            agent_name=agent_name,
            question=question,
            raw_answer="",
            extracted_number=None,
            sandbox_calls=session.call_count,
            sandbox_success=False,
            wall_time_s=wall_time,
            error=str(exc),
        )
