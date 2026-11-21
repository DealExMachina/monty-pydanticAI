"""PydanticAI agent runner for the benchmark.

Creates a PydanticAI Agent with a stateful_python_code_exec tool that calls
the shared Monty sandbox. The tool is named to match Nemotron's trained tool
name so the model naturally calls it without hallucinating an unknown tool.
Returns structured AgentResult for evaluation.
"""

from __future__ import annotations

import asyncio
import re
import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from benchmark.metrics import AgentResult
from benchmark.sandbox import SandboxSession, EXTERNAL_FUNCTION_NAMES

# ── Model configuration ─────────────────────────────────────────────

VLLM_BASE_URL = "https://nemotron-3-inference-dealexmachina-53d19e1c.koyeb.app/v1"
VLLM_MODEL = "nemotron"
VLLM_API_KEY = "dummy"  # vLLM doesn't require auth but the client needs a value

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
use the `stateful_python_code_exec` tool to execute code in a sandboxed Python environment.

## External math functions (no import needed)
These are provided by the sandbox — call them directly:
- sqrt(x)     — square root
- ln(x)       — natural logarithm
- exp(x)      — exponential (e^x)
- norm_cdf(x) — standard normal CDF (cumulative distribution)
- norm_pdf(x) — standard normal PDF (density)

## Native Python builtins (always available)
- min(), max(), abs(), round(), sum(), len(), range(), pow(), sorted(), enumerate()
- for/while loops, if/else, function definitions (def), list comprehensions

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
    model = OpenAIChatModel(
        model_name=VLLM_MODEL,
        provider=OpenAIProvider(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY),
    )
    agent = Agent(
        model,
        system_prompt=SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.3),
        # NOTE: To force the model to always call our tool (no free-text), add:
        #   extra_body={"tool_choice": {"type": "function",
        #               "function": {"name": "stateful_python_code_exec"}}}
        # to ModelSettings. This uses vLLM structured output to guarantee the
        # model emits a valid tool call. Kept off by default so the model can
        # also answer directly when no calculation is needed.
        retries=3,  # tool call retries (default 1); more attempts for Monty syntax errors
        output_retries=3,  # retry on validation failure to reduce spurious abort
    )

    @agent.tool_plain
    def stateful_python_code_exec(code: str) -> str:
        """Execute Python code in a sandboxed environment.

        External math functions (call directly, no import):
        sqrt, ln, exp, norm_cdf, norm_pdf.

        Native builtins also available:
        min, max, abs, round, sum, len, range, pow, sorted, enumerate.

        Supports for/while loops, if/else, function definitions (def),
        and list comprehensions.

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

        # Token usage from PydanticAI
        prompt_tokens, completion_tokens = 0, 0
        try:
            usage = result.usage()
            if usage:
                prompt_tokens = getattr(usage, "request_tokens", None) or getattr(
                    usage, "input_tokens", 0
                )
                completion_tokens = getattr(usage, "response_tokens", None) or getattr(
                    usage, "output_tokens", 0
                )
        except Exception:
            pass

        return AgentResult(
            agent_name="pydantic_ai",
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
            agent_name="pydantic_ai",
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
