"""Shared Monty sandbox with math external functions.

Both PydanticAI and DSPy agents call the same sandbox, ensuring a
fair comparison. The sandbox exposes sqrt, ln, exp, norm_cdf, norm_pdf
as external functions so the LLM can write readable quant code.
"""

from __future__ import annotations

import math
import textwrap
import time
from dataclasses import dataclass, field

import pydantic_monty

# ── External functions (host-side implementations) ──────────────────

PI = math.pi
E = math.e


def _sqrt(x: float) -> float:
    return math.sqrt(x)


def _ln(x: float) -> float:
    return math.log(x)


def _exp(x: float) -> float:
    return math.exp(x)


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return (1.0 / math.sqrt(2.0 * PI)) * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution (Abramowitz & Stegun 26.2.17)."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(
        -x_abs * x_abs / 2.0
    )
    return 0.5 * (1.0 + sign * y)


def _abs(x: float) -> float:
    return abs(x)


def _max_fn(*args: float) -> float:
    return max(args)


EXTERNAL_FUNCTIONS = {
    "sqrt": _sqrt,
    "ln": _ln,
    "exp": _exp,
    "norm_cdf": _norm_cdf,
    "norm_pdf": _norm_pdf,
    "abs": _abs,
    "max_val": _max_fn,
}

EXTERNAL_FUNCTION_NAMES = list(EXTERNAL_FUNCTIONS.keys())


# ── Sandbox types ───────────────────────────────────────────────────


@dataclass
class SandboxResult:
    """Result of a single sandbox execution."""

    success: bool
    output: str  # string returned to the agent
    raw_value: object  # actual Python object from Monty, or None
    elapsed_us: float  # microseconds
    error: str | None = None


@dataclass
class SandboxSession:
    """Tracks sandbox calls for one question. Enforces retry cap."""

    max_retries: int = 5
    call_count: int = field(default=0, init=False)
    results: list[SandboxResult] = field(default_factory=list, init=False)

    def execute(self, code: str) -> str:
        """Run code in Monty with external math functions. Returns a message."""
        self.call_count += 1

        if self.call_count > self.max_retries:
            msg = (
                f"Too many sandbox attempts ({self.call_count}). "
                "Stop retrying and answer the user directly with what you know."
            )
            self.results.append(
                SandboxResult(
                    success=False,
                    output=msg,
                    raw_value=None,
                    elapsed_us=0,
                    error="retry_cap_exceeded",
                )
            )
            return msg

        code = _clean_code(code)
        t0 = time.perf_counter()

        try:
            m = pydantic_monty.Monty(
                code,
                external_functions=EXTERNAL_FUNCTION_NAMES,
            )

            # Execute with start/resume loop for external function calls
            result = m.start()
            while isinstance(result, pydantic_monty.MontySnapshot):
                fn_name = result.function_name
                fn_args = result.args

                if fn_name in EXTERNAL_FUNCTIONS:
                    fn = EXTERNAL_FUNCTIONS[fn_name]
                    try:
                        ret = fn(*fn_args)
                    except Exception as fn_exc:
                        ret = float("nan")
                else:
                    ret = None

                result = result.resume(return_value=ret)

            # result is now MontyComplete
            output = result.output
            elapsed_us = (time.perf_counter() - t0) * 1_000_000

            result_str = repr(output) if output is not None else "(no return value)"
            msg = f"Execution succeeded.\nResult: {result_str}"
            self.results.append(
                SandboxResult(
                    success=True,
                    output=msg,
                    raw_value=output,
                    elapsed_us=elapsed_us,
                )
            )
            return msg

        except Exception as exc:
            elapsed_us = (time.perf_counter() - t0) * 1_000_000
            err_msg = f"{type(exc).__name__}: {exc}"
            msg = (
                f"Execution failed.\nError: {err_msg}\n\n"
                "IMPORTANT: All top-level statements must start at column 0. "
                "Use exactly 4 spaces for each indent level inside blocks."
            )
            self.results.append(
                SandboxResult(
                    success=False,
                    output=msg,
                    raw_value=None,
                    elapsed_us=elapsed_us,
                    error=err_msg,
                )
            )
            return msg

    @property
    def last_successful_value(self) -> object | None:
        """Return the raw_value from the last successful execution."""
        for r in reversed(self.results):
            if r.success:
                return r.raw_value
        return None

    @property
    def total_sandbox_time_us(self) -> float:
        return sum(r.elapsed_us for r in self.results)


# ── Helpers ─────────────────────────────────────────────────────────


def _clean_code(code: str) -> str:
    """Best-effort fix for common LLM code-gen issues."""
    code = code.strip("\n")
    code = textwrap.dedent(code)
    code = code.replace("\t", "    ")
    return code
