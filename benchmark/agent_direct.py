"""Direct (no-tools) baseline agent for the benchmark.

Sends the question to the LLM with instructions to compute and output
ANSWER: <number>. No sandbox — model must rely on reasoning/arithmetic only.
Uses the same Nemotron endpoint for fair comparison.
"""

from __future__ import annotations

import re
import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from benchmark.agent_pydantic import extract_number
from benchmark.metrics import AgentResult

# ── Model configuration ─────────────────────────────────────────────

VLLM_BASE_URL = "https://nemotron-3-inference-dealexmachina-53d19e1c.koyeb.app/v1"
VLLM_MODEL = "nemotron"
VLLM_API_KEY = "dummy"

# ── System prompt (no tools) ────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a quantitative finance calculator. When asked a calculation,
reason through the problem step by step and compute the answer.

## Answer format
After your reasoning, state the final numeric answer as:
ANSWER: <number>

Example:
Question: Calculate the future value of $10,000 at 5% for 10 years.
Reasoning: FV = PV * (1 + r)^n = 10000 * (1.05)^10 = ...
ANSWER: 16288.95
"""


def _build_agent() -> Agent:
    """Build a PydanticAI agent with no tools (direct LLM call only)."""
    model = OpenAIChatModel(
        model_name=VLLM_MODEL,
        provider=OpenAIProvider(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY),
    )
    return Agent(
        model,
        system_prompt=SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.3),
        output_retries=2,
    )


# ── Runner ──────────────────────────────────────────────────────────


async def run_direct_agent(question: str) -> AgentResult:
    """Run a single question through the direct (no-tools) agent."""
    agent = _build_agent()

    t0 = time.perf_counter()
    try:
        result = await agent.run(question)
        wall_time = time.perf_counter() - t0

        raw_answer = result.output
        extracted = extract_number(raw_answer)

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
            agent_name="direct",
            question=question,
            raw_answer=raw_answer,
            extracted_number=extracted,
            sandbox_calls=0,
            sandbox_success=False,
            wall_time_s=wall_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        wall_time = time.perf_counter() - t0
        return AgentResult(
            agent_name="direct",
            question=question,
            raw_answer="",
            extracted_number=None,
            sandbox_calls=0,
            sandbox_success=False,
            wall_time_s=wall_time,
            error=str(exc),
            prompt_tokens=0,
            completion_tokens=0,
        )
