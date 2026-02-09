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

## Monty sandbox limitations

Monty runs a **subset of Python**. Supported: arithmetic, strings, lists, dicts, functions, loops, comprehensions, type hints. Not supported: imports, classes, match statements, context managers, generators, standard library modules.

## Changing the model

Edit the `model_name` in `demo.py` to use any model available in your Ollama instance:

```python
ollama_model = OpenAIChatModel(
    model_name="qwen3:8b",  # change this
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)
```

## Stack

- [PydanticAI](https://ai.pydantic.dev/) — Agent framework
- [pydantic-monty](https://github.com/pydantic/monty) — Secure Python sandbox (Rust)
- [Ollama](https://ollama.com/) — Local LLM inference

## License

MIT
