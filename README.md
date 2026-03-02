# Agentic Backtesting System with Schema-Constrained, Documentation-Grounded, and Test-Driven Code Generation

This repository contains the experimental framework for evaluating how structured prompt engineering treatments affect the quality of LLM-generated algorithmic trading strategies. The system uses an agentic workflow where an LLM generates vectorbt-based trading strategy code, receives verification feedback, and iteratively refines its output.

## Experimental Design

The experiment uses a **2³ full factorial design** testing three independent treatments:

- **S (Schema)**: Structured input/output schemas with type hints, parameter constraints, and ranges
- **D (Documentation)**: RAG-based documentation grounding via AWS OpenSearch, providing real API docs for numpy, pandas, scipy, and vectorbt
- **T (TDD)**: Test-Driven Development with invariant tests and strategy-specific tests that generated code must pass

This yields **8 experimental conditions**:

| Condition | Schema | Docs | TDD | Description |
|-----------|--------|------|-----|-------------|
| C0 | off | off | off | Control (minimal prompt) |
| C1 | on | off | off | Schema only |
| C2 | off | on | off | Documentation grounding only |
| C3 | off | off | on | TDD only |
| C4 | on | on | off | Schema + Docs |
| C5 | on | off | on | Schema + TDD |
| C6 | off | on | on | Docs + TDD |
| C7 | on | on | on | All treatments |

Each condition is tested across **3 strategy complexity levels**:

| Level | Strategy | vectorbt Interface |
|-------|----------|-------------------|
| Simple | RSI Mean Reversion | `Portfolio.from_signals` |
| Medium | MACD + ATR Trailing Stop | `Portfolio.from_order_func(flexible=False)` |
| Complex | Pairs Trading (two assets) | `Portfolio.from_order_func(flexible=True)` |

Each condition-complexity pair is run **10 times** at two max-turn settings (10 and 20), producing **480 total experiment runs**.

## Repository Structure

```
.
├── run_experiment.py          # Main experiment runner (agent loop, verification, backtesting)
├── build_rag.py               # RAG index builder (extracts docs into AWS OpenSearch)
├── run_experiments.sh          # Batch script for all 480 experiment runs
├── pyproject.toml              # Project metadata and dependencies (used by uv)
├── uv.lock                     # Locked dependency versions
├── .env.example                # Template for environment variables (create .env from this)
│
├── prompts/                    # Prompt definitions for each condition
│   ├── c0_control.py           # C0: Control (no treatments)
│   ├── c1_schema.py            # C1: Schema only
│   ├── c2_docs.py              # C2: Documentation grounding only
│   ├── c3_tdd.py               # C3: TDD only
│   ├── c4_schema_docs.py       # C4: Schema + Docs
│   ├── c5_schema_tdd.py        # C5: Schema + TDD
│   ├── c6_docs_tdd.py          # C6: Docs + TDD
│   ├── c7_all.py               # C7: All treatments
│   └── shared/                 # Shared prompt components
│       ├── strategy_base.py    #   Strategy descriptions and output specs
│       ├── schemas.py          #   DATA_SCHEMA and PARAM_SCHEMA definitions
│       ├── signatures.py       #   Function signature specifications
│       ├── api_citation.py     #   API citation requirements for D-on conditions
│       ├── vas.py              #   RAG tool description for D-on conditions
│       ├── invariant_tests.py  #   Property-based tests (all strategies)
│       ├── strategy_tests.py   #   Strategy-specific test suites
│       └── validation.py       #   Validation requirement strings per condition
│
├── verify/                     # Verification modules (one per condition)
│   ├── verify_c0.py ... verify_c7.py
│   └── shared/                 # Shared verification logic
│       ├── schema.py           #   Schema conformance checking
│       ├── vas.py              #   Documentation grounding verification
│       ├── tests.py            #   Test execution engine
│       └── passthrough.py      #   Passthrough for disabled treatments
│
├── backtests/                  # Backtest runners using vectorbt
│   ├── simple.py               # RSI Mean Reversion backtest
│   ├── medium.py               # MACD + ATR backtest
│   ├── complex.py              # Pairs Trading backtest
│   └── shared/
│       ├── data.py             #   Synthetic data generation (reproducible, seed=42)
│       ├── metrics.py          #   Metrics extraction from vectorbt portfolios
│       └── result.py           #   BacktestResult dataclass
│
├── common/                     # Shared types and utilities
│   ├── types.py                #   VerificationResult dataclass
│   └── utils.py                #   Module loading helpers
│
├── reference-implementation/   # Gold-standard strategy implementations
│   ├── simple.py               #   RSI Mean Reversion reference
│   ├── medium.py               #   MACD + ATR reference
│   └── complex.py              #   Pairs Trading reference
│
├── results_10/                 # Experiment results (max_turns=10), 240 runs
│   └── {condition}_{complexity}_{run}/
│       ├── code.py             #   LLM-generated strategy code
│       └── results.json        #   Metadata, verification, and backtest metrics
│
└── results_20/                 # Experiment results (max_turns=20), 240 runs
    └── {condition}_{complexity}_{run}/
        ├── code.py
        └── results.json
```

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** -- fast Python package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **OpenAI API key** with access to the model specified in `.env` (default: `gpt-5-mini`)
- **AWS credentials** for OpenSearch Serverless (required only for D-on conditions: C2, C4, C6, C7)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd experiments_imp_2
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

   This creates a `.venv` virtual environment and installs all dependencies from `pyproject.toml` and `uv.lock`.

3. **Configure environment variables:**

   Create a `.env` file in the project root with the following variables:

   ```
   OPENAI_API_KEY="your-openai-api-key"
   OPENAI_ENDPOINT="https://api.openai.com/v1"
   OPENAI_MODEL="gpt-5-mini"
   AWS_OPENSEARCH_URL="your-opensearch-endpoint"
   AWS_ACCESS_KEY_ID="your-aws-access-key"
   AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
   ```

   > **Note:** The AWS credentials and OpenSearch URL are only required for running D-on conditions (C2, C4, C6, C7) which use the RAG documentation search tool. Conditions C0, C1, C3, and C5 only require the OpenAI credentials.

## Usage

### Running a Single Experiment

```bash
# Basic usage: run condition C0 (control) on the simple strategy
uv run run-experiment c0 simple

# Specify max agent turns and run number
uv run run-experiment c3 medium --max-turns 10 --run-number 1

# Run all treatments (C7) on the complex strategy with 20 max turns
uv run run-experiment c7 complex --max-turns 20 --run-number 1
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `condition` | Condition ID: `c0` through `c7` |
| `complexity` | Strategy complexity: `simple`, `medium`, or `complex` |
| `--max-turns N` | Maximum agent turns for iterative refinement (default: 20) |
| `--run-number N` | Explicit run number (auto-increments if omitted) |
| `--skip-backtest` | Skip the backtest phase after verification |
| `--verify-only` | Re-run verification on existing generated code |
| `--backtest-only` | Re-run backtest on existing generated code |
| `--quiet` | Suppress progress output |

### Running All Experiments

The `run_experiments.sh` script contains commands for all 480 experiment runs. Uncomment the desired batch and execute:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

Each line runs an experiment in the background (`&`), allowing parallel execution.

### Building the RAG Documentation Index

Required before running D-on conditions (C2, C4, C6, C7) for the first time:

```bash
# Index all libraries (numpy, pandas, scipy, vectorbt)
uv run build-rag

# Index a single library
uv run build-rag --lib numpy

# Preview without indexing
uv run build-rag --dry-run

# Delete and recreate the index
uv run build-rag --recreate
```

### Inspecting Results

Each experiment run produces a directory under `results_10/` or `results_20/` containing:

- **`code.py`** -- the LLM-generated Python strategy code
- **`results.json`** -- structured results including:
  - `condition`, `complexity`, `run_number` -- experiment identifiers
  - `submissions` -- number of code submissions the agent made
  - `total_tokens` -- token usage (`input_tokens`, `output_tokens`, `total_tokens`)
  - `generation.success` -- whether the agent produced code
  - `verification.results` -- per-category pass/fail (schema, documentation, tests)
  - `backtest.metrics` -- `total_return`, `sharpe_ratio`, `max_drawdown`, `total_trades`

Example result:

```json
{
  "condition": "c0",
  "complexity": "simple",
  "run_number": 1,
  "max_turns": 10,
  "submissions": 1,
  "total_tokens": {"input_tokens": 8600, "output_tokens": 5625, "total_tokens": 14225},
  "generation": {"success": true, "error": null},
  "verification": {
    "success": true,
    "results": {
      "schema": {"passed": true, "errors": []},
      "documentation": {"passed": true, "errors": []},
      "tests": {"passed": true, "errors": []}
    }
  },
  "backtest": {
    "success": true,
    "metrics": {
      "total_return": 0.338,
      "sharpe_ratio": 0.845,
      "max_drawdown": 0.199,
      "total_trades": 1
    }
  }
}
```

## How It Works

1. **Prompt Construction**: Each condition module (`prompts/c0_control.py` ... `prompts/c7_all.py`) assembles a prompt from shared components based on which treatments are enabled.

2. **Agent Execution**: `run_experiment.py` creates an OpenAI agent (via the `openai-agents` SDK) with:
   - A `submit_code` tool that saves, verifies, and backtests generated code
   - A `search_docs` tool (D-on conditions only) that queries the AWS OpenSearch RAG index

3. **Iterative Refinement**: The agent generates code, submits it via the tool, receives structured feedback (verification errors, backtest failures), and refines its code up to `max_turns` iterations.

4. **Verification**: Each condition verifies different aspects:
   - **Schema (S-on)**: Checks function signatures, parameter types/ranges, return value structure
   - **Documentation (D-on)**: Verifies that API calls match real library signatures
   - **TDD (T-on)**: Runs invariant tests and strategy-specific test suites against the generated code

5. **Backtesting**: Successfully verified code is backtested using vectorbt with synthetic data (reproducible via `seed=42`) to produce performance metrics.

## Pre-computed Results

The `results_10/` and `results_20/` directories contain the complete set of 480 pre-computed experiment results. These can be analyzed directly without re-running experiments. Each directory contains 240 result folders (8 conditions x 3 complexities x 10 runs).
