"""PydanticAI agent runner for the benchmark.

Creates a PydanticAI Agent with a run_python tool that calls the shared
Monty sandbox. Returns structured AgentResult for evaluation.
"""

from __future__ import annotations

import asyncio
import re
import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from benchmark.metrics import AgentResult
from benchmark.sandbox import SandboxSession, EXTERNAL_FUNCTION_NAMES

# ── Number extraction ───────────────────────────────────────────────


def extract_number(text: str) -> float | None:
    """Extract a numeric answer from agent text output.

    Tries (in order):
    1. Explicit ANSWER: <number> tag
    2. Number after common result phrases (is, equals, result, total)
    3. Last float-like number in the text
    """
    if not text:
        return None

    # Pattern 1: explicit ANSWER: tag
    m = re.search(r"ANSWER:\s*\$?([-\d,]+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # Pattern 2: number after common phrases
    m = re.search(
        r"(?:is|equals?|=|result|total|answer|price|value|payment)[:\s]+\$?([-\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # Pattern 3: last float-like number in text
    numbers = re.findall(r"-?[\d,]+\.\d+|-?\d{2,}", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a quantitative finance calculator. When asked a calculation,
use the `run_python` tool to execute code in a sandboxed Python environment.

## Available math functions (no import needed)
The sandbox provides these functions directly — just call them:
- sqrt(x)    — square root
- ln(x)      — natural logarithm
- exp(x)     — exponential (e^x)
- norm_cdf(x) — standard normal CDF (cumulative distribution)
- norm_pdf(x) — standard normal PDF (density)
- abs(x)     — absolute value
- max_val(a, b) — maximum of two values

## Code formatting (CRITICAL)
- All top-level statements MUST start at column 0 (no leading spaces).
- Use exactly 4 spaces per indent level inside blocks.
- After a for/while/if block, the next top-level line must be at column 0.
- The value of the LAST expression is the result returned to you.

## Example: Black-Scholes call price
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
d1 = (ln(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
call = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
round(call, 4)

## Answer format
After getting the sandbox result, state the final numeric answer as:
ANSWER: <number>
"""


# ── Agent factory ───────────────────────────────────────────────────


def _build_agent(session: SandboxSession) -> Agent:
    """Build a fresh PydanticAI agent bound to a SandboxSession."""
    ollama_model = OpenAIChatModel(
        model_name="qwen3:8b",
        provider=OllamaProvider(base_url="http://localhost:11434/v1"),
    )
    agent = Agent(
        ollama_model,
        system_prompt=SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.3),
    )

    @agent.tool_plain
    def run_python(code: str) -> str:
        """Execute Python code in a secure Monty sandbox.

        Available math functions (call directly, no import):
        sqrt, ln, exp, norm_cdf, norm_pdf, abs, max_val.

        All top-level statements must start at column 0.
        Use 4 spaces for indentation inside blocks.
        The result of the last expression is returned.

        Args:
            code: Python source code to execute.
        """
        return session.execute(code)

    return agent


# ── Runner ──────────────────────────────────────────────────────────


async def run_pydantic_agent(question: str) -> AgentResult:
    """Run a single question through the PydanticAI agent."""
    session = SandboxSession(max_retries=3)
    agent = _build_agent(session)

    t0 = time.perf_counter()
    try:
        result = await agent.run(question)
        wall_time = time.perf_counter() - t0

        raw_answer = result.output

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
            agent_name="pydantic_ai",
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
            agent_name="pydantic_ai",
            question=question,
            raw_answer="",
            extracted_number=None,
            sandbox_calls=session.call_count,
            sandbox_success=False,
            wall_time_s=wall_time,
            error=str(exc),
        )
