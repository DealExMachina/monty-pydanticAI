# Monty vs Plain Tool Calls: Benchmark Evaluation

This benchmark compares four agent setups for quantitative finance calculations,
all using the same Nemotron model served via vLLM. The goal is to demonstrate
why a code-execution sandbox (Monty) outperforms pre-built tool calls for
financial computations.

## Why Monty?

The conventional approach to giving an LLM calculation abilities is to wrap
library functions as individual tools: `black_scholes_call(S, K, T, r, sigma)`,
`bond_price(face, coupon, ytm, n)`, etc. This works for simple, well-defined
formulas but breaks down when problems require:

- **Iterative solvers** (implied volatility via Newton-Raphson)
- **Numerical methods** (finite-difference Greeks with parameter bumping)
- **Custom compositions** (multi-leg option strategies with arbitrary payoffs)
- **Stateful loops** (amortization schedules with mid-term rate changes)
- **Bespoke products** (structured notes with caps, floors, barriers)

A code-execution sandbox like Monty lets the LLM write arbitrary Python to
solve any of these, without needing a pre-built tool for every possible
financial calculation. This benchmark quantifies the accuracy gap.

## The Four Agents

All agents use the same Nemotron model via vLLM on Koyeb.

### Agent A: PydanticAI + Monty

- File: `benchmark/agent_pydantic.py`
- Tool: `stateful_python_code_exec` -- executes arbitrary Python in Monty
- The LLM writes code to solve any problem; Monty runs it client-side

### Agent B: DSPy vanilla ReAct + Monty

- File: `benchmark/agent_dspy.py`
- Same Monty tool, but orchestrated through DSPy ReAct (unoptimized)
- Tests whether DSPy's ReAct prompting strategy helps or hurts accuracy

### Agent C: DSPy optimized ReAct + Monty

- File: `benchmark/agent_dspy_optimized.py`
- BootstrapFewShot-compiled DSPy ReAct with Monty
- Includes automatically discovered few-shot examples in the prompt
- Expected to achieve the highest accuracy of all four agents

### Agent D: PydanticAI + Pre-built Tool Calls (naive baseline)

- File: `benchmark/agent_tools.py`
- ~20 individual `@agent.tool_plain` functions (no Monty, no code generation)
- This is **what most people would do** without knowing about Monty or DSPy
- Expected to score well on single-formula categories but fail on code-composition

## Benchmark Design

### Dataset: 150 questions across 15 categories

**Single-formula categories (100 questions, 10 categories):**

Problems solvable with a single function call. Agent D should handle these well.

- `option_pricing` -- Black-Scholes call/put pricing
- `greeks` -- Delta, gamma, vega, theta, rho
- `var_portfolio` -- Parametric Value at Risk
- `bond_pricing` -- Bond prices, duration, convexity
- `compound_interest` -- Future value with various compounding
- `loan_amortization` -- Monthly payments, balances, total interest
- `npv_irr` -- Net present value calculations
- `portfolio_metrics` -- Sharpe ratio, portfolio variance
- `derivatives_misc` -- Put-call parity, forwards, covered calls
- `depreciation_tax` -- Straight-line, declining balance, tax shields

**Code-composition categories (50 questions, 5 categories):**

Problems requiring loops, conditionals, helper functions, or multi-step logic.
Agent D has no tools for these and should fail or score poorly.

- `implied_volatility` -- Newton-Raphson iterative solver
- `numerical_greeks` -- Finite-difference Greeks with parameter bumping
- `multi_leg_options` -- Iron condors, straddles, butterflies
- `structured_products` -- Capital-protected notes, reverse convertibles
- `amortization_events` -- Loan schedules with extra payments and rate changes

### Metrics

- **Accuracy**: Extracted numeric answer within tolerance of expected answer
- **Execution success rate**: Whether the sandbox produced a result (Monty agents only)
- **Latency**: Wall-clock time per question (mean, median, P95)
- **Token usage**: Prompt + completion tokens
- **Per-category accuracy**: Breakdown by single-formula vs code-composition

### Expected Results

| Category slice | Agent D (tools) | Agent A (Monty) | Agent B (DSPy) | Agent C (DSPy opt) |
|---|---|---|---|---|
| Single-formula (100q) | Good | Good | Good | Best |
| Code-composition (50q) | Poor/Fail | Good | Good | Best |

## How to Run

```bash
# Full 4-agent benchmark (default)
uv run python -m benchmark.run

# Quick test with 5 questions
uv run python -m benchmark.run --limit 5

# Skip specific agents
uv run python -m benchmark.run --no-tools-baseline   # skip Agent D
uv run python -m benchmark.run --no-pydantic          # skip Agent A
uv run python -m benchmark.run --no-dspy              # skip Agent B
uv run python -m benchmark.run --no-optimized         # skip Agent C

# Include the no-tools direct baseline (LLM arithmetic only)
uv run python -m benchmark.run --direct

# Control concurrency
uv run python -m benchmark.run --concurrency 8

# DSPy optimized on held-out test set (30%)
uv run python -m benchmark.run --dspy-optimized-on-test
```

### Re-optimizing DSPy

After dataset changes, re-run the DSPy optimization:

```bash
uv run python -m benchmark.agent_dspy_optimized
```

This trains on 70% of the 150 questions and saves
`benchmark/optimized_react_state.json`.

## Results Interpretation

The summary table at the end of a run shows accuracy, latency, and token
usage for each agent. The per-category breakdown is split into
"Single-formula" and "Code-composition" sections.

The key comparison is between Agent D (tools) and the Monty agents (A/B/C)
on code-composition categories. A large accuracy gap here demonstrates why
Monty is the better approach for financial calculations.

### When to use which approach

- **Pre-built tools** (Agent D style): When the problem domain is
  well-defined, the set of possible calculations is small and fixed, and
  reliability on those specific calculations matters most.

- **Monty sandbox** (Agents A/B/C style): When problems are diverse,
  require composition, or the set of possible calculations cannot be
  anticipated. This is the general-purpose approach.

- **DSPy optimization** (Agent C): When you have a training set of
  question-answer pairs and want to maximize accuracy through automatic
  prompt engineering. Layer this on top of Monty for the best results.

## Technical Notes

### Tool name alignment

Nemotron was trained with a tool called `stateful_python_code_exec`.
Our Monty-backed tool is named to match, so the model's training bias
works in our favor. See the git history for details on the original
`run_python` -> `stateful_python_code_exec` rename.

### Monty builtins

Monty natively provides `min()`, `max()`, `abs()`, `round()`, `sum()`,
`len()`, `range()`, `pow()`, `sorted()`, `enumerate()` as builtins.
The sandbox only exposes 5 external functions: `sqrt`, `ln`, `exp`,
`norm_cdf`, `norm_pdf`.

### References

- [PydanticAI](https://ai.pydantic.dev/) -- Agent framework
- [pydantic-monty](https://github.com/pydantic/monty) -- Rust-based Python sandbox
- [DSPy](https://dspy.ai/) -- Programmatic prompt optimization
- [vLLM](https://docs.vllm.ai/) -- Inference engine for Nemotron
