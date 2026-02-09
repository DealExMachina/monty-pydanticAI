"""
Demo: PydanticAI agent with Monty sandbox execution via Ollama.

The agent has access to a `run_python` tool that executes code inside
Monty — a minimal, secure Python interpreter written in Rust by Pydantic.
All reasoning steps and sandbox calls are traced to stdout.
"""

from __future__ import annotations

import asyncio
import textwrap
import time

import pydantic_monty
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

# ── helpers ──────────────────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

MAX_TOOL_RETRIES = 3  # stop the agent from looping on the same error


def trace(label: str, body: str, color: str = BLUE) -> None:
    """Print a coloured trace block to stdout."""
    header = f"{color}{BOLD}[{label}]{RESET}"
    indented = textwrap.indent(body.strip(), "  ")
    print(f"\n{header}\n{indented}\n")


def clean_code(code: str) -> str:
    """Best-effort fix for common LLM code-gen issues.

    Small local models frequently emit code with:
    - a stray leading/trailing space on top-level lines after a block,
    - inconsistent indent (tabs vs spaces mix).

    We apply textwrap.dedent and normalize indent to 4 spaces.
    """
    # Strip leading/trailing blank lines
    code = code.strip("\n")
    # Dedent the whole block (fixes uniform extra indent)
    code = textwrap.dedent(code)
    # Replace tabs with 4 spaces
    code = code.replace("\t", "    ")
    return code


# ── Monty sandbox wrapper ───────────────────────────────────────────

_sandbox_call_count: int = 0  # track calls per agent.run()


def reset_sandbox_counter() -> None:
    global _sandbox_call_count
    _sandbox_call_count = 0


def execute_in_sandbox(code: str) -> str:
    """Run *code* inside the Monty sandbox and return stdout + result.

    Monty is a minimal Python interpreter written in Rust.  It blocks
    filesystem / network / env access by default, so the code cannot
    escape the sandbox.
    """
    global _sandbox_call_count
    _sandbox_call_count += 1

    if _sandbox_call_count > MAX_TOOL_RETRIES:
        msg = (
            f"Too many sandbox attempts ({_sandbox_call_count}). "
            "Stop retrying and answer the user directly with what you know."
        )
        trace(f"SANDBOX ⚠ retry cap", msg, RED)
        return msg

    # Clean up common LLM indentation issues
    code = clean_code(code)

    trace("SANDBOX ← code", code, YELLOW)
    t0 = time.perf_counter()

    try:
        m = pydantic_monty.Monty(code)
        output = m.run()
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        result_str = repr(output) if output is not None else "(no return value)"
        trace(
            f"SANDBOX → result  ({elapsed_us:.0f} µs)",
            result_str,
            GREEN,
        )
        return f"Execution succeeded.\nResult: {result_str}"

    except Exception as exc:
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        err_msg = f"{type(exc).__name__}: {exc}"
        trace(f"SANDBOX ✗ error  ({elapsed_us:.0f} µs)", err_msg, RED)
        return (
            f"Execution failed.\nError: {err_msg}\n\n"
            "IMPORTANT: If this is an indentation error, make sure ALL "
            "top-level statements start at column 0 with NO leading spaces. "
            "Use exactly 4 spaces for each indent level inside blocks."
        )


# ── Agent setup ─────────────────────────────────────────────────────

ollama_model = OpenAIChatModel(
    model_name="qwen3:8b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

SYSTEM_PROMPT = """\
You are a helpful assistant that can run Python code.
When the user asks you to compute something or write a program,
use the `run_python` tool to execute code in a sandboxed environment.

## Sandbox rules (Monty)
- Monty is a Python subset: arithmetic, strings, lists, dicts, tuples,
  functions (def), for/while loops, if/elif/else, comprehensions, f-strings.
- NOT supported: import, class, with, match, yield, eval, exec, open.
- The value of the LAST expression is returned as the result.

## Code formatting (CRITICAL)
- All top-level statements MUST start at column 0 (no leading spaces).
- Use exactly 4 spaces per indent level inside blocks.
- After a for/while/if block, the next top-level line must be at column 0.
- Example of CORRECT code:
  result = []
  for i in range(10):
      result.append(i * i)
  result

## On errors
- If the sandbox returns an error, try a DIFFERENT approach.
  Do NOT repeat the same code.
- If you cannot fix the code after 2 tries, answer directly from your
  knowledge without using the tool.
"""

agent = Agent(
    ollama_model,
    system_prompt=SYSTEM_PROMPT,
    model_settings=ModelSettings(temperature=0.3),
)


@agent.tool_plain
def run_python(code: str) -> str:
    """Execute Python code in a secure Monty sandbox.

    The sandbox supports basic Python: arithmetic, strings, lists, dicts,
    functions, loops, comprehensions.  No imports, no file or network access.
    The result of the last expression is returned.

    IMPORTANT: Top-level statements must start at column 0 (no leading spaces).
    Use exactly 4 spaces for indentation inside blocks.

    Args:
        code: The Python source code to execute.
    """
    return execute_in_sandbox(code)


# ── main loop ───────────────────────────────────────────────────────

async def main() -> None:
    print(f"{BOLD}╔══════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║  PydanticAI + Monty Sandbox Demo (Ollama local) ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════╝{RESET}")
    print(f"Model : qwen3:8b via Ollama")
    print(f"Sandbox: Monty {pydantic_monty.__version__}")
    print(f"Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input(f"{BOLD}You > {RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not user_input.strip():
            continue

        trace("USER", user_input)
        reset_sandbox_counter()

        result = await agent.run(user_input)

        # Trace reasoning / message history
        for msg in result.new_messages():
            kind = msg.kind
            if kind == "response" and hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "content"):
                        trace("REASONING", str(part.content), BLUE)
                    elif hasattr(part, "tool_name"):
                        trace(
                            f"TOOL CALL → {part.tool_name}",
                            str(getattr(part, "args", "")),
                            YELLOW,
                        )
            elif kind == "tool-return":
                trace("TOOL RETURN", str(getattr(msg, "content", "")), GREEN)

        print(f"\n{BOLD}Agent >{RESET} {result.output}\n")


if __name__ == "__main__":
    asyncio.run(main())
