"""PydanticAI agent with pre-built quant tool calls (naive baseline).

This is Agent D: the approach most people would take without knowing
about DSPy or Monty. Each financial formula is wrapped as an individual
PydanticAI tool_plain. The LLM picks the right tool and extracts
parameters from the question.

This agent is expected to score well on single-formula categories but
fail on code-composition categories where no matching tool exists.
"""

from __future__ import annotations

import math
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

# ── Math helpers (host-side, same as sandbox.py) ─────────────────────

PI = math.pi


def _norm_cdf(x: float) -> float:
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


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * PI)) * math.exp(-0.5 * x * x)


def _bs_d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _bs_d2(S, K, T, r, sigma):
    return _bs_d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a quantitative finance calculator. You have access to pre-built
financial tools. To answer a question, call the appropriate tool with
the correct parameters extracted from the question.

If no tool matches the question exactly, do your best to reason through
the problem and provide a numeric answer.

## Answer format
After getting the tool result, state the final numeric answer as:
ANSWER: <number>
"""


# ── Agent factory ────────────────────────────────────────────────────


def _build_tools_agent() -> Agent:
    """Build a PydanticAI agent with pre-built quant tool calls."""
    model = OpenAIChatModel(
        model_name=VLLM_MODEL,
        provider=OpenAIProvider(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY),
    )
    agent = Agent(
        model,
        system_prompt=SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.3),
        retries=3,
        output_retries=3,
    )

    # ── Option pricing tools ────────────────────────────────────────

    @agent.tool_plain
    def black_scholes_call(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Price a European call option using the Black-Scholes formula.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate (annualized, e.g. 0.05 for 5%).
            sigma: Volatility (annualized, e.g. 0.20 for 20%).
        """
        d1 = _bs_d1(S, K, T, r, sigma)
        d2 = _bs_d2(S, K, T, r, sigma)
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

    @agent.tool_plain
    def black_scholes_put(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Price a European put option using the Black-Scholes formula.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate (annualized).
            sigma: Volatility (annualized).
        """
        d1 = _bs_d1(S, K, T, r, sigma)
        d2 = _bs_d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    # ── Greeks tools ────────────────────────────────────────────────

    @agent.tool_plain
    def call_delta(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate the delta of a European call option.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate.
            sigma: Volatility.
        """
        d1 = _bs_d1(S, K, T, r, sigma)
        return _norm_cdf(d1)

    @agent.tool_plain
    def put_delta(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate the delta of a European put option.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate.
            sigma: Volatility.
        """
        d1 = _bs_d1(S, K, T, r, sigma)
        return _norm_cdf(d1) - 1.0

    @agent.tool_plain
    def option_gamma(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate the gamma of a European option (same for call and put).

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate.
            sigma: Volatility.
        """
        d1 = _bs_d1(S, K, T, r, sigma)
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))

    @agent.tool_plain
    def option_vega_pct(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate the vega of a European option (per 1% change in volatility).

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate.
            sigma: Volatility.
        """
        d1 = _bs_d1(S, K, T, r, sigma)
        return S * _norm_pdf(d1) * math.sqrt(T) / 100.0

    @agent.tool_plain
    def call_theta(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate the theta (per year) of a European call option.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate.
            sigma: Volatility.
        """
        d1 = _bs_d1(S, K, T, r, sigma)
        d2 = _bs_d2(S, K, T, r, sigma)
        return (
            -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
            - r * K * math.exp(-r * T) * _norm_cdf(d2)
        )

    @agent.tool_plain
    def call_rho(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate the rho of a European call option.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate.
            sigma: Volatility.
        """
        d2 = _bs_d2(S, K, T, r, sigma)
        return K * T * math.exp(-r * T) * _norm_cdf(d2)

    # ── Bond pricing tools ──────────────────────────────────────────

    @agent.tool_plain
    def bond_price(
        face: float, coupon_rate: float, ytm: float, periods: int
    ) -> float:
        """Calculate the price of an annual coupon bond.

        Args:
            face: Face (par) value of the bond.
            coupon_rate: Annual coupon rate (e.g. 0.05 for 5%).
            ytm: Yield to maturity (e.g. 0.05 for 5%).
            periods: Number of years to maturity.
        """
        C = face * coupon_rate
        price = sum(C / (1 + ytm) ** t for t in range(1, periods + 1))
        price += face / (1 + ytm) ** periods
        return price

    @agent.tool_plain
    def bond_duration(
        face: float, coupon_rate: float, ytm: float, periods: int
    ) -> float:
        """Calculate the Macaulay duration of an annual coupon bond.

        Args:
            face: Face value.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            periods: Number of years to maturity.
        """
        C = face * coupon_rate
        P = sum(C / (1 + ytm) ** t for t in range(1, periods + 1)) + face / (1 + ytm) ** periods
        dur = sum(t * C / (1 + ytm) ** t for t in range(1, periods + 1)) + periods * face / (1 + ytm) ** periods
        return dur / P

    # ── Compound interest tools ─────────────────────────────────────

    @agent.tool_plain
    def compound_interest(
        pv: float, rate: float, n: int, t: int
    ) -> float:
        """Calculate the future value with discrete compounding.

        Args:
            pv: Present value (initial investment).
            rate: Annual interest rate (e.g. 0.05 for 5%).
            n: Compounding frequency per year (1=annual, 12=monthly, 365=daily).
            t: Number of years.
        """
        return pv * (1 + rate / n) ** (n * t)

    @agent.tool_plain
    def continuous_compounding(
        pv: float, rate: float, t: float
    ) -> float:
        """Calculate the future value with continuous compounding.

        Args:
            pv: Present value.
            rate: Annual interest rate.
            t: Number of years.
        """
        return pv * math.exp(rate * t)

    # ── Loan tools ──────────────────────────────────────────────────

    @agent.tool_plain
    def loan_monthly_payment(
        principal: float, annual_rate: float, years: int
    ) -> float:
        """Calculate the monthly payment for a fully amortizing loan.

        Args:
            principal: Loan amount.
            annual_rate: Annual interest rate (e.g. 0.06 for 6%).
            years: Loan term in years.
        """
        r = annual_rate / 12.0
        n = years * 12
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    @agent.tool_plain
    def loan_remaining_balance(
        principal: float, annual_rate: float, years: int, months_paid: int
    ) -> float:
        """Calculate the remaining balance after k monthly payments.

        Args:
            principal: Original loan amount.
            annual_rate: Annual interest rate.
            years: Original loan term in years.
            months_paid: Number of monthly payments already made.
        """
        r = annual_rate / 12.0
        n = years * 12
        pmt = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        return principal * (1 + r) ** months_paid - pmt * ((1 + r) ** months_paid - 1) / r

    @agent.tool_plain
    def loan_total_interest(
        principal: float, annual_rate: float, years: int
    ) -> float:
        """Calculate the total interest paid over the life of a loan.

        Args:
            principal: Loan amount.
            annual_rate: Annual interest rate.
            years: Loan term in years.
        """
        r = annual_rate / 12.0
        n = years * 12
        pmt = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        return pmt * n - principal

    # ── NPV tool ────────────────────────────────────────────────────

    @agent.tool_plain
    def npv_calc(rate: float, cashflows: list[float]) -> float:
        """Calculate the Net Present Value of a series of cash flows.

        Args:
            rate: Discount rate (e.g. 0.10 for 10%).
            cashflows: List of cash flows where cashflows[0] is at time 0.
        """
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))

    # ── VaR tool ────────────────────────────────────────────────────

    @agent.tool_plain
    def parametric_var(
        portfolio_value: float,
        daily_volatility: float,
        z_score: float,
        holding_days: int,
        trading_days_per_year: int = 252,
    ) -> float:
        """Calculate parametric Value at Risk.

        Args:
            portfolio_value: Total portfolio value.
            daily_volatility: Daily volatility as decimal (e.g. 0.02 for 2%).
            z_score: Z-score for confidence level (1.6449 for 95%, 2.3263 for 99%).
            holding_days: Holding period in trading days.
            trading_days_per_year: Trading days per year (default 252).
        """
        return portfolio_value * z_score * daily_volatility * math.sqrt(
            holding_days / trading_days_per_year
        )

    # ── Forward price tool ──────────────────────────────────────────

    @agent.tool_plain
    def forward_price(
        spot: float, rate: float, time: float, div_yield: float = 0.0
    ) -> float:
        """Calculate the forward price of an asset.

        Args:
            spot: Current spot price.
            rate: Risk-free interest rate.
            time: Time to maturity in years.
            div_yield: Continuous dividend yield (default 0).
        """
        return spot * math.exp((rate - div_yield) * time)

    # ── Depreciation tools ──────────────────────────────────────────

    @agent.tool_plain
    def straight_line_depreciation(
        cost: float, salvage: float, useful_life: int
    ) -> float:
        """Calculate annual straight-line depreciation.

        Args:
            cost: Original cost of the asset.
            salvage: Salvage value at end of useful life.
            useful_life: Useful life in years.
        """
        return (cost - salvage) / useful_life

    @agent.tool_plain
    def declining_balance_depreciation(
        cost: float, rate: float, year: int
    ) -> float:
        """Calculate declining balance depreciation for a specific year.

        Args:
            cost: Original cost of the asset.
            rate: Depreciation rate (e.g. 0.20 for 20%).
            year: Which year to calculate (1-indexed).
        """
        return cost * (1 - rate) ** (year - 1) * rate

    # ── Portfolio metrics tools ─────────────────────────────────────

    @agent.tool_plain
    def sharpe_ratio(
        portfolio_return: float, risk_free_rate: float, volatility: float
    ) -> float:
        """Calculate the Sharpe ratio.

        Args:
            portfolio_return: Portfolio return (e.g. 0.14 for 14%).
            risk_free_rate: Risk-free rate (e.g. 0.03 for 3%).
            volatility: Portfolio volatility (e.g. 0.18 for 18%).
        """
        return (portfolio_return - risk_free_rate) / volatility

    return agent


# ── Runner ──────────────────────────────────────────────────────────


async def run_tools_agent(question: str) -> AgentResult:
    """Run a single question through the pre-built tools agent."""
    agent = _build_tools_agent()

    t0 = time.perf_counter()
    try:
        result = await agent.run(question)
        wall_time = time.perf_counter() - t0

        raw_answer = result.output

        # Try to extract a number from the answer
        extracted = extract_number(raw_answer)

        # Token usage
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
            agent_name="pydantic_tools",
            question=question,
            raw_answer=raw_answer,
            extracted_number=extracted,
            sandbox_calls=0,  # no sandbox
            sandbox_success=False,
            wall_time_s=wall_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        wall_time = time.perf_counter() - t0
        return AgentResult(
            agent_name="pydantic_tools",
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
