"""DSPy ReAct agent runner for the benchmark.

Uses dspy.ReAct with a stateful_python_code_exec tool that calls the shared
Monty sandbox. The tool is named to match Nemotron's trained tool name.
Configures DSPy to use Nemotron via vLLM on Koyeb.
"""

from __future__ import annotations

import time

import dspy

from benchmark.metrics import AgentResult
from benchmark.sandbox import SandboxSession
from benchmark.agent_pydantic import extract_number

# ── Model configuration ────────────────────────────────────────────

VLLM_BASE_URL = "https://nemotron-3-inference-dealexmachina-53d19e1c.koyeb.app/v1"
VLLM_MODEL = "nemotron"
VLLM_API_KEY = "dummy"  # vLLM doesn't require auth but the client needs a value

# ── DSPy LM configuration ──────────────────────────────────────────


def configure_dspy_lm() -> dspy.LM:
    """Configure DSPy to use Nemotron via vLLM on Koyeb."""
    lm = dspy.LM(
        f"openai/{VLLM_MODEL}",
        api_base=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0.3,
        cache=False,
    )
    return lm


# ── Tool factory ────────────────────────────────────────────────────


def make_tool_function(session: SandboxSession):
    """Create a stateful_python_code_exec tool function bound to a SandboxSession.

    The tool is named to match Nemotron's trained tool name. The returned
    function has proper type hints and docstring for DSPy to infer the tool schema.
    """

    def stateful_python_code_exec(code: str) -> str:
        """Execute Python code in a sandboxed environment for financial calculations.

        External math functions (call directly, no import):
        sqrt(x), ln(x), exp(x), norm_cdf(x), norm_pdf(x).

        Native builtins also available:
        min, max, abs, round, sum, len, range, pow, sorted, enumerate.

        Supports for/while loops, if/else, function definitions (def),
        and list comprehensions.

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

    return stateful_python_code_exec


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

    # Capture history length before call to get delta (LM is shared)
    history_len_before = 0
    try:
        lm = dspy.settings.lm
        if lm:
            history_len_before = len(lm.history)
    except Exception:
        pass

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

        # Token usage from LM history (only new entries for this call)
        prompt_tokens, completion_tokens = 0, 0
        try:
            lm = dspy.settings.lm
            if lm and lm.history:
                for entry in lm.history[history_len_before:]:
                    usage = entry.get("usage") if isinstance(entry, dict) else None
                    if usage:
                        prompt_tokens += usage.get("prompt_tokens", 0) or usage.get(
                            "input_tokens", 0
                        )
                        completion_tokens += usage.get(
                            "completion_tokens", 0
                        ) or usage.get("output_tokens", 0)
        except Exception:
            pass

        return AgentResult(
            agent_name=agent_name,
            question=question,
            raw_answer=raw_answer,
            extracted_number=extracted,
            sandbox_calls=session.call_count,
            sandbox_success=any(r.success for r in session.results),
            wall_time_s=wall_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
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
            prompt_tokens=0,
            completion_tokens=0,
        )
