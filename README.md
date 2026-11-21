# PydanticAI + Monty Sandbox Demo

A simple interactive agent built with [PydanticAI](https://ai.pydantic.dev/) that executes Python code in a secure [Monty](https://github.com/pydantic/monty) sandbox, powered by a local [Ollama](https://ollama.com/) model.

## What is this?

The agent has a `run_python` tool that sends LLM-generated code to **Monty** — a minimal Python interpreter written in Rust by the Pydantic team. Monty blocks filesystem, network, and environment access by default, so the code cannot escape the sandbox.

All reasoning steps and sandbox calls are traced to stdout with color-coded output.

## Prerequisites

- **Python** >= 3.11
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **[Ollama](https://ollama.com/)** running locally with a model pulled (default: `qwen3:8b`)

```bash
# Pull the model if you haven't already
ollama pull qwen3:8b
```

## Quick start

```bash
git clone https://github.com/DealExMachina/monty-pydanticAI.git
cd monty-pydanticAI
uv sync
uv run python demo.py
```

## Trace output

The demo prints color-coded traces as the agent works:

| Label | Color | Meaning |
|---|---|---|
| `[USER]` | blue | Your input |
| `[REASONING]` | blue | Agent's chain-of-thought |
| `[SANDBOX ← code]` | yellow | Code sent to Monty |
| `[SANDBOX → result]` | green | Result + execution time (µs) |
| `[SANDBOX ✗ error]` | red | Sandbox error |

Example session:

```
You > Calcule la somme des carrés des 10 premiers entiers

[SANDBOX ← code]
  sum(i**2 for i in range(1, 11))

[SANDBOX → result  (62 µs)]
  385

Agent > La somme des carrés des 10 premiers entiers est **385**.
```

## Benchmark: PydanticAI vs DSPy ReAct

The `benchmark/` directory compares agent approaches on **100 quantitative finance calculations** (Black-Scholes, Greeks, VaR, bond pricing, NPV, Sharpe ratio, etc.):

| Agent | Description |
|---|---|
| **PydanticAI + Monty** | Hand-crafted system prompt |
| **DSPy ReAct + Monty** | DSPy vanilla (auto-generated prompt) |
| **DSPy ReAct optimized** | After BootstrapFewShot optimization |
| **Direct (baseline)** | No tools; LLM reasoning only — use `--direct` |

**Benchmark model:** The benchmark requires **Nemotron** (vLLM on a Koyeb endpoint). Ollama is used only for the interactive `demo.py`.

### Run the benchmark

```bash
# Verify the dataset (100 examples, 10 categories)
uv run python -c "from benchmark.dataset import verify_dataset; verify_dataset()"

# Quick test (3 questions, PydanticAI only)
uv run python -m benchmark.run --limit 3 --no-dspy --no-optimized

# Full benchmark (all 3 agents, 100 questions each)
uv run python -m benchmark.run

# Include no-tools baseline to measure sandbox value
uv run python -m benchmark.run --direct

# DSPy optimization on held-out test set (recommended for fair comparison)
uv run python -m benchmark.run --dspy-optimized-on-test

# DSPy optimization (optional, ~30-60 min)
uv run python -m benchmark.agent_dspy_optimized
uv run python -m benchmark.run
```

### Benchmark flags

| Flag | Description |
|---|---|
| `--limit N` | Run on first N questions only |
| `--concurrency N` | Parallel requests (default: 4) |
| `--direct` | Include no-tools baseline; report "Sandbox Delta" vs tool-using agents |
| `--dspy-optimized-on-test` | Run DSPy optimized on 30% held-out test set (70/30 split) |
| `--no-pydantic` | Skip PydanticAI agent |
| `--no-dspy` | Skip DSPy vanilla |
| `--no-optimized` | Skip DSPy optimized |

### Dataset categories

| Category | Examples |
|---|---|
| option_pricing | Black-Scholes European call/put pricing |
| greeks | Delta, Gamma, Vega, Theta, Rho |
| var_portfolio | Parametric VaR (variance-covariance) |
| bond_pricing | Bond price, duration, convexity |
| compound_interest | Discrete and continuous compounding |
| loan_amortization | Monthly payments (PMT formula) |
| npv_irr | Net present value calculations |
| portfolio_metrics | Sharpe ratio, weighted returns, portfolio volatility |
| derivatives_misc | Put-call parity, forwards, swaps |
| depreciation_tax | Straight-line, declining balance, tax shields |

### Output

Results are saved to `benchmark/results/benchmark_YYYYMMDD_HHMMSS.json` with per-question details and aggregate metrics:

- **Accuracy** and **Sandbox Delta** (vs no-tools baseline when `--direct` is used)
- **Latency** (mean, median, P95)
- **Sandbox calls** per question
- **Token economics**: prompt tokens, completion tokens, total tokens, estimated cost (USD)

## Monty sandbox

Monty runs a **subset of Python**. Supported: arithmetic, strings, lists, dicts, functions, loops, comprehensions, type hints. Not supported: imports, classes, match statements, context managers, generators, standard library modules.

For the benchmark, the sandbox exposes **external functions** (`sqrt`, `ln`, `exp`, `norm_cdf`, `norm_pdf`) so the LLM can write readable quant code without reimplementing math.

## Stack

- [PydanticAI](https://ai.pydantic.dev/) — Agent framework
- [DSPy](https://dspy.ai/) — Declarative LLM programming & prompt optimization
- [pydantic-monty](https://github.com/pydantic/monty) — Secure Python sandbox (Rust)
- [Ollama](https://ollama.com/) — Local LLM inference

## License

MIT
