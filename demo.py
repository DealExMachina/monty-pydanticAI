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

# ── helpers ──────────────────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def trace(label: str, body: str, color: str = BLUE) -> None:
    """Print a coloured trace block to stdout."""
    header = f"{color}{BOLD}[{label}]{RESET}"
    indented = textwrap.indent(body.strip(), "  ")
    print(f"\n{header}\n{indented}\n")


# ── Monty sandbox wrapper ───────────────────────────────────────────

def execute_in_sandbox(code: str) -> str:
    """Run *code* inside the Monty sandbox and return stdout + result.

    Monty is a minimal Python interpreter written in Rust.  It blocks
    filesystem / network / env access by default, so the code cannot
    escape the sandbox.
    """
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
        return f"Execution failed.\nError: {err_msg}"


# ── Agent setup ─────────────────────────────────────────────────────

ollama_model = OpenAIChatModel(
    model_name="qwen3:8b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

agent = Agent(
    ollama_model,
    system_prompt=(
        "You are a helpful assistant that can run Python code. "
        "When the user asks you to compute something or write a program, "
        "use the `run_python` tool to execute code in a sandboxed environment. "
        "Always show your reasoning before writing code. "
        "The sandbox runs a Python subset (Monty): no imports, no classes, "
        "no file/network access. Basic arithmetic, strings, lists, dicts, "
        "functions, loops, and comprehensions all work. "
        "The code you write is evaluated as an expression or a series of "
        "statements; the value of the last expression is returned."
    ),
)


@agent.tool_plain
def run_python(code: str) -> str:
    """Execute Python code in a secure Monty sandbox.

    The sandbox supports basic Python: arithmetic, strings, lists, dicts,
    functions, loops, comprehensions.  No imports, no file or network access.
    The result of the last expression is returned.

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
