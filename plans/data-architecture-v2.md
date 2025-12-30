# Data Architecture v2: Unified Time-Series Data Model

> **Status:** DRAFT - All decisions pending review

## Decision Summary

All major decisions have been made. See **Decided** section below.

> **Note:** Execution frequency and price are configured via built-in parameters in PARAM_SCHEMA. All data sources are resampled to the `execution_frequency` before being passed to the strategy.

> **Decided:**
> - **LLM Output Format**: Structured JSON output with separate `data_schema`, `param_schema`, and `code` fields. No parsing of Python code needed.
> - **DATA_SCHEMA** uses rich structure (frequency, columns, description). No `resample` field - resampling is controlled globally.
> - **PARAM_SCHEMA** always includes two built-in parameters: `execution_frequency` (enum: 1Min, 5Min, 1Hour, 1Day, 1Week, 1Month) and `execution_price` (enum: open, close). All data is resampled to `execution_frequency` before being passed to the strategy.
> - **Symbol Handling**: Symbols are NEVER in DATA_SCHEMA. DATA_SCHEMA defines data slots (e.g., `prices`, `asset_a`, `asset_b`). Symbols are always injected at runtime, making strategies reusable across different symbols.
> - **Multi-Symbol Support**: Naturally handled by DATA_SCHEMA slots. A pairs strategy defines `asset_a` and `asset_b` slots; runtime maps them to actual symbols (e.g., SPY, QQQ). Same logic works for any pair.
> - **Symbol Extraction**: ChatGPT extracts symbols from user prompts and passes them separately from the strategy logic. Symbol changes don't require code regeneration.
> - **Error Handling**: Fail fast with structured ERROR_SCHEMA. Errors are returned to the widget for user-friendly visualization.
> - **Caching**: No caching for now. Fetch fresh data on every request. Caching can be added later if needed.
> - **Fundamental Data Provider**: Financial Modeling Prep (FMP) for earnings, balance sheet, and other fundamental data.

## Overview

This document outlines the architecture for a flexible, agentic backtesting system that treats all data uniformly as time-series and allows AI-generated strategies to declare their data dependencies.

**Note:** The "Reference Implementation" section below shows ONE possible approach combining several options. It is provided for illustration only and does not represent decided choices.

## Core Principles

1. **Data is data** - No distinction between "price data" and "fundamental data". Everything is a time-indexed DataFrame.
2. **Schema/Instance separation** - Strategies define schemas (shape/constraints), runtime provides instances (specific values).
3. **Agentic orchestration** - The orchestrator validates instances against schemas and dispatches to appropriate data providers.
4. **Separation of concerns** - Logic, data schema, and param schema are independently modifiable.

## Conversation Flow

This section illustrates how user prompts flow through the system, demonstrating the separation between strategy logic and runtime parameters.

### Example 1: Creating a New Strategy

**User:** "Create a MA crossover strategy with SPY"

ChatGPT parses this prompt and extracts:
- **Strategy logic:** MA crossover (needs code generation)
- **Symbol:** SPY (runtime parameter)

ChatGPT calls the backtest tool:
```python
backtest(
    prompt="MA crossover strategy",      # sent to Codex for code generation
    symbols={"prices": "SPY"},           # runtime parameter, NOT sent to Codex
)
```

Codex generates a **generic, symbol-agnostic** strategy:
```json
{
  "data_schema": {
    "prices": {"frequency": "1Day", "columns": ["close"]}
  },
  "param_schema": {
    "fast_window": {"type": "int", "default": 10},
    "slow_window": {"type": "int", "default": 30}
  },
  "code": "def generate_signals(data, params): ..."
}
```

Orchestrator:
1. Maps `symbols={"prices": "SPY"}` to DATA_SCHEMA slots
2. Fetches SPY data from provider
3. Runs backtest with default params

### Example 2: Changing Symbol (No Code Regeneration)

**User:** "Show me what this looks like with QQQ instead"

ChatGPT recognizes this is a **symbol change only** - no logic change needed.

ChatGPT calls the backtest tool with **same code**, different symbol:
```python
backtest(
    base_code=<previous strategy code>,   # reuse existing
    symbols={"prices": "QQQ"},            # only this changes
)
```

Orchestrator:
1. Skips code generation (base_code provided)
2. Fetches QQQ data
3. Runs backtest

**Result:** Same strategy logic, different data, no LLM call needed.

### Example 3: Changing Parameters (No Code Regeneration)

**User:** "Try with a 20-period fast MA instead"

ChatGPT recognizes this is a **parameter change** - no logic change needed.

```python
backtest(
    base_code=<previous strategy code>,   # reuse existing
    symbols={"prices": "QQQ"},            # keep current symbol
    params={"fast_window": 20},           # override default
)
```

**Result:** Same code, same data schema, just different param values.

### Example 4: Changing Logic (Code Regeneration Required)

**User:** "Actually, let's use RSI instead of MA crossover"

ChatGPT recognizes this requires **new strategy logic**.

```python
backtest(
    prompt="RSI strategy",                # new logic, needs Codex
    symbols={"prices": "QQQ"},            # keep current symbol
)
```

Codex generates new DATA_SCHEMA, PARAM_SCHEMA, and code for RSI strategy.

### Summary: What Triggers Code Regeneration?

| User Request | Code Regeneration? | What Changes |
|--------------|-------------------|--------------|
| "Create X strategy with SPY" | ✅ Yes | New strategy |
| "Try QQQ instead" | ❌ No | Only symbols |
| "Use 20-period MA" | ❌ No | Only params |
| "Add a stop-loss" | ✅ Yes | Logic change |
| "Switch to RSI strategy" | ✅ Yes | New strategy |

## Schema vs Instance Model

The architecture separates **schemas** (abstract shape/constraints) from **instances** (concrete values):

| Component | Schema (Shape) | Instance (Values) |
|-----------|---------------|-------------------|
| **Data** | DATA_SCHEMA - what data structure the strategy expects | DATA_PARAMS - specific providers, symbols, timeframes |
| **Parameters** | PARAM_SCHEMA - parameter types, bounds, descriptions | params - specific values (defaults extracted from PARAM_SCHEMA) |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STRATEGY DEFINITION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DATA_SCHEMA                          PARAM_SCHEMA                       │
│  ┌─────────────────────────┐         ┌─────────────────────────┐        │
│  │ "prices": {             │         │ "pe_threshold": {       │        │
│  │   frequency: "1Day",    │         │   type: "float",        │        │
│  │   columns: [ohlcv]      │         │   min: 0, max: 100,     │        │
│  │ },                      │         │   default: 15           │        │
│  │ "earnings": {           │         │ },                      │        │
│  │   frequency: "quarterly"│         │ "ma_window": {          │        │
│  │   columns: [eps]        │         │   type: "int",          │        │
│  │ }                       │         │   min: 1, max: 500      │        │
│  │                         │         │ }                       │        │
│  │                         │         └─────────────────────────┘        │
│  └─────────────────────────┘                     │                       │
│              │                           (validated against)             │
│              │                                   │                       │
│      (validated against)                         ▼                       │
│              │                       params (instance)                   │
│              ▼                       ┌─────────────────────────┐        │
│  DATA_PARAMS (instance)              │ pe_threshold: 15        │        │
│  ┌─────────────────────────┐         │ ma_window: 20           │        │
│  │ provider: "alpaca"      │         │                         │        │
│  │ symbol: "SPY"           │         └─────────────────────────┘        │
│  │ timeframe: "1Day"       │                                             │
│  └─────────────────────────┘                                             │
│                                                                          │
│  STRATEGY_CODE (logic that uses data and params)                         │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ def generate_signals(data: dict, params: dict):             │        │
│  │     # all data resampled to same frequency before this call │        │
│  │     ...                                                     │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Resampling and Execution Frequency

The `execution_frequency` parameter (in PARAM_SCHEMA) controls how often the strategy runs. All data sources are resampled to this frequency:

- DATA_SCHEMA specifies `frequency` - what to fetch from the provider
- PARAM_SCHEMA specifies `execution_frequency` - what to resample to

The orchestrator:
1. Fetches each data series at its `frequency`
2. Resamples all data to `execution_frequency` (upsample with ffill, or downsample with aggregation)
3. Passes aligned data to `generate_signals()`

```
Example: Daily execution with quarterly fundamentals

DATA_SCHEMA:
  prices:   frequency="1Day"       → no change (already daily)
  earnings: frequency="quarterly"  → forward-fill to daily

PARAM_SCHEMA:
  execution_frequency: "1Day"

Quarterly EPS:      Q1────────────────Q2────────────────Q3
                     │                 │                 │
                     ▼ (forward-fill)  ▼                 ▼
Daily (resampled): D1 D2 D3 D4 ... D60 D61 D62 ... D120 D121 ...


Example: Weekly rebalancing with daily data

DATA_SCHEMA:
  prices:   frequency="1Day"       → downsample to weekly (last)
  earnings: frequency="quarterly"  → forward-fill to weekly

PARAM_SCHEMA:
  execution_frequency: "1Week"

Daily prices:      D1 D2 D3 D4 D5 D6 D7 D8 D9 ...
                                  │           │
                                  ▼ (last)    ▼
Weekly (resampled):              W1          W2
```

### Benefits of Schema/Instance Separation

| Benefit | Description |
|---------|-------------|
| **Reusability** | Same strategy (schema + code) works for any symbol |
| **Validation** | Orchestrator validates instances against schemas before execution |
| **Swappable data sources** | Change provider without touching strategy code |
| **UI generation** | Auto-generate forms from PARAM_SCHEMA |
| **Parameter optimization** | PARAM_SCHEMA provides bounds for grid search |
| **Documentation** | Schemas include descriptions for each field |

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER REQUEST                                   │
│  "Create a PE ratio strategy" + symbols: ["SPY", "AAPL"]                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CODEX AGENT                                    │
│  Generates STRATEGY DEFINITION:                                          │
│  - DATA_SCHEMA (what data shape is needed)                              │
│  - PARAM_SCHEMA (what parameters exist, with bounds)                    │
│  - STRATEGY_CODE (the logic)                                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CODEX AGENT (or separate call)                 │
│  Generates DATA_PARAMS based on:                                         │
│  - DATA_SCHEMA (what's needed)                                          │
│  - User's symbols (SPY, AAPL)                                           │
│  - Available providers                                                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                   │
│  1. Validate DATA_PARAMS against DATA_SCHEMA                            │
│  2. Validate params against PARAM_SCHEMA                                │
│  3. Fetch data from providers based on DATA_PARAMS                      │
│  4. Extract defaults from PARAM_SCHEMA, merge with user overrides       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PROVIDERS                                 │
│  Alpaca (prices), FMP (fundamentals), Yahoo (fallback)                  │
│  Each fetches data according to DATA_PARAMS                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXECUTOR                                       │
│  generate_signals(data, params) → (entries, exits)                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           VECTORBT                                       │
│  Portfolio.from_signals() → backtest results                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Generated Code Structure

A strategy definition consists of three parts generated by the LLM:

### 1. DATA_SCHEMA (Data Requirements)

Defines what data the strategy needs. The orchestrator fetches at `frequency` and resamples to `execution_frequency` (from PARAM_SCHEMA):

```python
DATA_SCHEMA = {
    "prices": {
        "frequency": "1Day",       # raw data frequency to fetch
        "columns": ["open", "high", "low", "close", "volume"],
        "description": "Daily OHLCV price data",
    },
    "earnings": {
        "frequency": "quarterly",  # raw data frequency to fetch (will be resampled)
        "columns": ["eps"],
        "description": "Quarterly earnings per share",
    },
}
```

**Fields:**
- `frequency`: What frequency to fetch from the data provider
- `columns`: Required columns in the DataFrame
- `description`: Human-readable description

**Note:** All data sources are resampled to `execution_frequency` (from PARAM_SCHEMA) before being passed to the strategy. Lower-frequency data (e.g., quarterly) is forward-filled; higher-frequency data is downsampled.

### 2. PARAM_SCHEMA (Parameter Constraints)

Defines what parameters the strategy accepts. **Every strategy automatically includes two built-in parameters** for execution control:

```python
PARAM_SCHEMA = {
    # ===== BUILT-IN PARAMETERS (always present) =====
    "execution_frequency": {
        "type": "enum",
        "values": ["1Min", "5Min", "15Min", "1Hour", "1Day", "1Week", "1Month"],
        "default": "1Day",
        "description": "Frequency at which the strategy executes. All data is resampled to this frequency.",
    },
    "execution_price": {
        "type": "enum",
        "values": ["open", "close"],
        "default": "close",
        "description": "Price column used for trade execution.",
    },

    # ===== STRATEGY-SPECIFIC PARAMETERS =====
    "pe_threshold": {
        "type": "float",
        "min": 0,
        "max": 100,
        "default": 15,
        "description": "PE ratio below which to buy",
    },
    "pe_exit": {
        "type": "float",
        "min": 0,
        "max": 100,
        "default": 25,
        "description": "PE ratio above which to sell",
    },
}
```

**Built-in Parameters:**
- `execution_frequency`: Controls how often signals are generated. All data sources are resampled to this frequency before being passed to `generate_signals()`.
- `execution_price`: Which price column vectorbt uses for trade execution (`open` or `close`).

### 3. STRATEGY_CODE (Logic)

The function that generates trading signals:

```python
def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Args:
        data: Dict[str, pd.DataFrame] keyed by DATA_SCHEMA keys.
              All DataFrames are pre-resampled to the same frequency.
        params: Strategy parameters, validated against PARAM_SCHEMA.

    Returns:
        (entries, exits) - boolean Series aligned with data index
    """
    prices = data['prices']      # guaranteed: ohlcv columns, resampled frequency
    earnings = data['earnings']  # guaranteed: eps column, same frequency (pre-resampled)

    # No manual reindexing needed - orchestrator already aligned everything
    pe_ratio = prices['close'] / earnings['eps']

    entries = (pe_ratio < params['pe_threshold']).fillna(False).astype(bool)
    exits = (pe_ratio > params['pe_exit']).fillna(False).astype(bool)

    return entries, exits
```

**Note:** The strategy code no longer needs to handle resampling/reindexing. The orchestrator aligns all data to the common `execution_frequency` before calling `generate_signals()`.

### 4. Runtime Inputs (Symbols)

Symbols are passed at runtime, mapped to DATA_SCHEMA slots:

```python
# User: "Run this on SPY"
symbols = {"prices": "SPY", "earnings": "SPY"}

# User: "Run pairs trading on GLD vs SLV"
symbols = {"asset_a": "GLD", "asset_b": "SLV"}
```

The orchestrator uses these to fetch data from appropriate providers.

### 5. ERROR_SCHEMA (Error Response)

When errors occur, a structured error response is returned to the widget for visualization:

```python
ERROR_SCHEMA = {
    "success": False,
    "error": {
        "type": "data_fetch_error",      # error category
        "message": "QQQ data could not be loaded",  # user-friendly message
        "details": {                      # optional technical details
            "symbol": "QQQ",
            "provider": "alpaca",
            "status_code": 404,
        },
    },
}
```

**Error Types:**

| Type | Description | Example Message |
|------|-------------|-----------------|
| `data_fetch_error` | Failed to fetch data from provider | "QQQ data could not be loaded, please try again" |
| `invalid_symbol_error` | Symbol not recognized | "ASDF is not a valid ticker" |
| `code_generation_error` | Codex failed to generate valid code | "Failed to generate strategy after 5 attempts" |
| `validation_error` | Generated code failed validation | "Strategy code is missing generate_signals function" |
| `execution_error` | Strategy code threw an error | "Division by zero in strategy calculation" |
| `backtest_error` | vectorbt backtest failed | "Insufficient data for backtest period" |
| `schema_mismatch_error` | Symbols don't match DATA_SCHEMA slots | "Missing symbol mapping for 'asset_b'" |

**Widget Handling:**

The widget should display errors with:
- Clear error type indicator (icon/color)
- User-friendly message
- Optional "Show Details" for technical info
- Suggested actions when applicable (e.g., "Try a different ticker")

### Complete Strategy File Example

```python
# ============== STRATEGY DEFINITION (generated by LLM) ==============

DATA_SCHEMA = {
    "prices": {
        "frequency": "1Day",
        "columns": ["open", "high", "low", "close", "volume"],
        "description": "Daily OHLCV price data",
    },
    "earnings": {
        "frequency": "quarterly",
        "columns": ["eps"],
        "description": "Quarterly earnings per share",
    },
}

PARAM_SCHEMA = {
    # Built-in parameters (always present)
    "execution_frequency": {
        "type": "enum",
        "values": ["1Min", "5Min", "15Min", "1Hour", "1Day", "1Week", "1Month"],
        "default": "1Day",
        "description": "Frequency at which the strategy executes",
    },
    "execution_price": {
        "type": "enum",
        "values": ["open", "close"],
        "default": "close",
        "description": "Price column used for trade execution",
    },
    # Strategy-specific parameters
    "pe_threshold": {
        "type": "float",
        "min": 0,
        "max": 100,
        "default": 15,
        "description": "PE ratio below which to buy",
    },
    "pe_exit": {
        "type": "float",
        "min": 0,
        "max": 100,
        "default": 25,
        "description": "PE ratio above which to sell",
    },
}

def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    """Returns signals at execution_frequency."""
    prices = data['prices']
    earnings = data['earnings']

    # No reindexing needed - all data already aligned by orchestrator
    pe_ratio = prices['close'] / earnings['eps']

    entries = (pe_ratio < params['pe_threshold']).fillna(False).astype(bool)
    exits = (pe_ratio > params['pe_exit']).fillna(False).astype(bool)

    return entries, exits
```


## Reference Implementation (One Possible Approach)

The following shows ONE possible implementation combining several options above. This is NOT the decided approach.

### 4. Data Provider Interface

```python
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class DataProvider(ABC):
    """Base interface for all data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier used in DATA_PARAMS."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """List of data types this provider supports."""
        ...

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        data_type: str,
        start: datetime,
        end: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data for a symbol.

        Args:
            symbol: Ticker symbol
            data_type: Type of data (e.g., "ohlcv", "quarterly_earnings")
            start: Start datetime
            end: End datetime
            **kwargs: Additional parameters (timeframe, fields, etc.)

        Returns:
            DataFrame with DatetimeIndex
        """
        ...
```

### 5. Provider Registry

```python
class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, DataProvider] = {}

    def register(self, provider: DataProvider):
        self._providers[provider.name] = provider

    def get(self, name: str) -> DataProvider:
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}")
        return self._providers[name]

    def list_providers(self) -> dict:
        """Return provider documentation for system prompt."""
        return {
            name: {
                "types": provider.supported_types,
                "description": provider.__doc__,
            }
            for name, provider in self._providers.items()
        }

# Global registry
PROVIDERS = ProviderRegistry()
PROVIDERS.register(AlpacaProvider())
PROVIDERS.register(FMPProvider())  # Financial Modeling Prep for fundamentals
```

## Updated BacktestInput Schema

```python
class BacktestInput(BaseModel):
    """Schema for the backtest tool."""

    # Strategy specification
    prompt: str  # Natural language description or modification request
    base_code: Optional[str] = None  # Existing code to modify

    # Runtime parameters
    symbols: List[str]  # Symbols to backtest
    params: Optional[Dict[str, Any]] = None  # Override defaults from PARAM_SCHEMA

    # Time range
    start: Optional[str] = None
    end: Optional[str] = None

    # Execution settings
    init_cash: float = 10000.0
```

## Orchestrator Logic

```python
class BacktestOrchestrator:
    def __init__(self, agent: CodexAgent, providers: ProviderRegistry):
        self.agent = agent
        self.providers = providers
        self.executor = StrategyExecutor()
        self.cache = DataCache()

    def run(self, payload: BacktestInput) -> dict:
        # 1. Generate or modify code
        session = self.agent.create_session(payload.prompt, payload.base_code)
        code = self.agent.generate(session)

        # 2. Validate code
        is_valid, error = self.agent.validate_code(code)
        if not is_valid:
            # Feed back to agent for fixing...
            pass

        # 3. Extract schemas from structured LLM response (already JSON)
        data_schema = response['data_schema']
        param_schema = response['param_schema']
        code = response['code']

        # 4. Extract default params from schema, merge with user overrides
        default_params = {k: v['default'] for k, v in param_schema.items()}
        params = {**default_params, **(payload.params or {})}

        # 5. Get execution_frequency from params (built-in parameter)
        exec_freq = params.get('execution_frequency', '1Day')

        # 6. Run for each symbol
        return self._run_single_symbol(code, data_schema, params, payload, exec_freq)

    def _run_single_symbol(self, code, data_schema, params, payload, exec_freq):
        results = {}
        for symbol in payload.symbols:
            # Fetch and resample data to execution_frequency
            data = self._fetch_and_resample(data_schema, symbol, payload.start, payload.end, exec_freq)

            # Execute strategy
            result = self.executor.execute(code, data, params)
            results[symbol] = result

        return results

    def _fetch_and_resample(self, data_schema, symbol, start, end, exec_freq):
        data = {}
        for name, schema in data_schema.items():
            # Fetch at source frequency
            provider = self.providers.infer_provider(schema)
            df = self.cache.get_or_fetch(
                provider=provider,
                symbol=schema.get('symbol', symbol),  # use fixed symbol if specified
                frequency=schema['frequency'],
                start=start,
                end=end,
            )

            # Resample to execution_frequency
            if schema['frequency'] != exec_freq:
                df = self._resample_to_exec_freq(df, schema['frequency'], exec_freq)

            data[name] = df
        return data

    def _resample_to_exec_freq(self, df, source_freq, exec_freq):
        """Resample DataFrame to execution_frequency."""
        if is_higher_frequency(source_freq, exec_freq):
            # Downsample: daily -> weekly (use last value)
            return df.resample(exec_freq).last()
        else:
            # Upsample: quarterly -> daily (forward-fill)
            return df.resample(exec_freq).ffill()
```

## System Prompt Updates

The CodexAgent system prompt must document available providers:

```
AVAILABLE DATA PROVIDERS:

alpaca:
  - type: "ohlcv"
  - timeframes: "1Min", "5Min", "15Min", "1Hour", "1Day"
  - columns returned: open, high, low, close, volume

fmp (Financial Modeling Prep):
  - type: "quarterly_earnings"
  - fields: eps, reported_eps, estimated_eps, surprise, surprise_percentage
  - type: "income_statement"
  - fields: revenue, gross_profit, operating_income, net_income
  - type: "balance_sheet"
  - fields: total_assets, total_liabilities, book_value

yahoo:
  - type: "ohlcv"
  - type: "dividends"
  - type: "splits"
```

## Error Handling

### Data Fetch Errors

When a provider fails or returns no data:

1. **Unknown provider** - Validation error, feed back to Codex to fix
2. **API error** - Retry with backoff, then fail with clear message
3. **No data for symbol** - Skip symbol, include warning in results
4. **Missing fields** - Feed back to Codex to adjust requirements

### Codex Feedback Loop

```python
def feed_data_error(self, session: CodexSession, error: str, requirement: dict):
    """Feed a data fetching error back to the agent."""
    error_message = f"""The data requirement could not be fulfilled:

Requirement: {requirement}
Error: {error}

Please check:
1. Provider name is valid (alpaca, fmp, yahoo)
2. Data type is supported by the provider
3. Required fields are available

Output the corrected code with fixed DATA_SCHEMA."""

    session.messages.append({"role": "user", "content": error_message})
```

## Caching Strategy

```python
class DataCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[tuple, pd.DataFrame] = {}
        self._timestamps: dict[tuple, datetime] = {}
        self.ttl = ttl_seconds

    def _make_key(self, provider, symbol, data_type, start, end, **kwargs) -> tuple:
        return (provider.name, symbol, data_type, str(start), str(end), frozenset(kwargs.items()))

    def get_or_fetch(self, provider, symbol, data_type, start, end, **kwargs) -> pd.DataFrame:
        key = self._make_key(provider, symbol, data_type, start, end, **kwargs)

        # Check cache
        if key in self._cache:
            if datetime.now() - self._timestamps[key] < timedelta(seconds=self.ttl):
                return self._cache[key]

        # Fetch and cache
        df = provider.fetch(symbol, data_type, start, end, **kwargs)
        self._cache[key] = df
        self._timestamps[key] = datetime.now()
        return df
```

## Implementation Phases

### Phase 1: Core Infrastructure (Priority: High)

- [ ] Provider interface (`DataProvider` ABC)
- [ ] Provider registry
- [ ] Alpaca provider (refactor existing)
- [ ] Structured JSON output from LLM (data_schema, param_schema, code)
- [ ] Resampling logic (upsample with ffill, downsample with aggregation)
- [ ] Updated system prompt with provider documentation
- [ ] Updated `generate_signals(data, params)` signature
- [ ] Orchestrator with single-symbol mode

### Phase 2: Enhanced Data Support (Priority: Medium)

- [ ] FMP provider (Financial Modeling Prep for fundamental data)
- [ ] In-memory caching
- [ ] Error handling with Codex feedback for data issues
- [ ] Yahoo Finance provider (fallback/alternative for price data)

### Phase 3: Advanced Features (Priority: Low)

- [ ] Multi-symbol strategy examples (pairs trading, relative value)
- [ ] Parameter schema with validation and bounds
- [ ] Persistent caching (SQLite or Redis)
- [ ] Strategy metadata (name, description, version)
- [ ] Parameter optimization integration

## Open Questions

### Data & Providers

1. ~~**Which fundamental data provider?**~~ **DECIDED:** Financial Modeling Prep (FMP) for earnings, balance sheet, and other fundamental data. See Decision section.

2. **How to handle timezone alignment?** Price data may be in exchange timezone, fundamentals in UTC. Options:
   - Normalize everything to UTC
   - Keep original timezones, let strategy handle
   - Configurable per provider

3. **Should we support custom/user-provided data?** e.g., user uploads a CSV of proprietary signals. Options:
   - Not supported (v1)
   - File upload endpoint
   - Reference external URLs

4. **What about alternative data?** Sentiment, news, satellite imagery, etc. Same provider pattern or separate?

### Execution & Backtesting

5. **Execution price column?** Currently assumes `close`. See Decision #7.

6. **Slippage and transaction costs?** Currently only fees. Should we model:
   - Slippage (market impact)
   - Bid-ask spread
   - Different fee structures

7. **Position sizing?** Current system is all-in/all-out. Should we support:
   - Fractional positions
   - Kelly criterion
   - Risk parity

8. **Which vectorbt API?** Currently `Portfolio.from_signals()`. Consider:
   - `from_orders()` for more control
   - `from_order_func()` for full flexibility

### Architecture

9. **Strategy persistence?** Should successful strategies be saved to disk? Options:
   - Not persisted (current)
   - Save to `strategies/` folder
   - Database storage

10. **Strategy versioning?** When a strategy is modified, keep history? Git-like versioning?

11. **Concurrent execution?** Run multiple symbols in parallel? Options:
    - Sequential (current)
    - ThreadPoolExecutor
    - async/await

12. **Rate limiting?** How to handle provider rate limits gracefully?

### LLM Integration

13. **Model selection?** Currently hardcoded to `gpt-5.1`. Should this be configurable?

14. **Token/cost tracking?** Should we track API costs per backtest?

15. **Prompt caching?** Can we cache system prompts or use fine-tuned models?

## Appendix: Example Strategies

### Simple RSI Strategy (Price Only)

```python
# ============== STRATEGY DEFINITION ==============

DATA_SCHEMA = {
    "prices": {
        "frequency": "1Day",
        "columns": ["open", "high", "low", "close", "volume"],
        "description": "Daily OHLCV price data",
    },
}

PARAM_SCHEMA = {
    # Built-in (always present)
    "execution_frequency": {"type": "enum", "values": ["1Day", "1Week", ...], "default": "1Day"},
    "execution_price": {"type": "enum", "values": ["open", "close"], "default": "close"},
    # Strategy-specific
    "rsi_window": {
        "type": "int",
        "min": 2,
        "max": 100,
        "default": 14,
        "description": "RSI calculation window",
    },
    "oversold": {
        "type": "float",
        "min": 0,
        "max": 50,
        "default": 30,
        "description": "RSI level considered oversold (buy signal)",
    },
    "overbought": {
        "type": "float",
        "min": 50,
        "max": 100,
        "default": 70,
        "description": "RSI level considered overbought (sell signal)",
    },
}

def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    """Returns signals at execution_frequency."""
    prices = data['prices']
    rsi = vbt.RSI.run(prices['close'], window=params['rsi_window']).rsi

    entries = (rsi < params['oversold']).fillna(False).astype(bool)
    exits = (rsi > params['overbought']).fillna(False).astype(bool)

    return entries, exits
```

**Runtime Input** (for "Run on SPY"):
```python
symbols = {"prices": "SPY"}
```

---

### PE Ratio Strategy (Price + Fundamentals)

```python
# ============== STRATEGY DEFINITION ==============

DATA_SCHEMA = {
    "prices": {
        "frequency": "1Day",
        "columns": ["open", "high", "low", "close", "volume"],
        "description": "Daily OHLCV price data",
    },
    "earnings": {
        "frequency": "quarterly",
        "columns": ["eps"],
        "description": "Quarterly earnings per share",
    },
}

PARAM_SCHEMA = {
    # Built-in (always present)
    "execution_frequency": {"type": "enum", "values": ["1Day", "1Week", ...], "default": "1Day"},
    "execution_price": {"type": "enum", "values": ["open", "close"], "default": "close"},
    # Strategy-specific
    "pe_buy_threshold": {
        "type": "float",
        "min": 0,
        "max": 50,
        "default": 15,
        "description": "PE ratio below which to buy",
    },
    "pe_sell_threshold": {
        "type": "float",
        "min": 10,
        "max": 100,
        "default": 25,
        "description": "PE ratio above which to sell",
    },
}

def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    """Returns signals at execution_frequency."""
    prices = data['prices']
    earnings = data['earnings']

    # No manual reindexing - orchestrator already aligned to execution_frequency
    pe_ratio = prices['close'] / earnings['eps']

    entries = (pe_ratio < params['pe_buy_threshold']).fillna(False).astype(bool)
    exits = (pe_ratio > params['pe_sell_threshold']).fillna(False).astype(bool)

    return entries, exits
```

**Runtime Input** (for "Run on AAPL"):
```python
symbols = {"prices": "AAPL", "earnings": "AAPL"}
```

---

### Pairs Trading Strategy (Multi-Symbol)

```python
# ============== STRATEGY DEFINITION ==============

DATA_SCHEMA = {
    "asset_a": {
        "frequency": "1Day",
        "columns": ["open", "high", "low", "close", "volume"],
        "description": "First asset in the pair",
    },
    "asset_b": {
        "frequency": "1Day",
        "columns": ["open", "high", "low", "close", "volume"],
        "description": "Second asset in the pair",
    },
}

PARAM_SCHEMA = {
    # Built-in (always present)
    "execution_frequency": {"type": "enum", "values": ["1Day", "1Week", ...], "default": "1Day"},
    "execution_price": {"type": "enum", "values": ["open", "close"], "default": "close"},
    # Strategy-specific
    "zscore_entry": {
        "type": "float",
        "min": 0.5,
        "max": 5.0,
        "default": 2.0,
        "description": "Z-score threshold to enter trade",
    },
    "zscore_exit": {
        "type": "float",
        "min": 0,
        "max": 2.0,
        "default": 0.5,
        "description": "Z-score threshold to exit trade",
    },
    "lookback": {
        "type": "int",
        "min": 5,
        "max": 100,
        "default": 20,
        "description": "Lookback period for mean/std calculation",
    },
}

def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    """Returns signals at execution_frequency."""
    a = data['asset_a']['close']
    b = data['asset_b']['close']

    # Calculate spread
    spread = a / b
    spread_mean = spread.rolling(params['lookback']).mean()
    spread_std = spread.rolling(params['lookback']).std()
    zscore = (spread - spread_mean) / spread_std

    # Mean reversion signals
    entries = (zscore < -params['zscore_entry']).fillna(False).astype(bool)
    exits = (abs(zscore) < params['zscore_exit']).fillna(False).astype(bool)

    return entries, exits
```

**Runtime Input** (symbols injected, strategy is reusable):
```python
# User: "Run pairs trading on SPY vs QQQ"
symbols = {"asset_a": "SPY", "asset_b": "QQQ"}

# User: "Run pairs trading on GLD vs SLV"
symbols = {"asset_a": "GLD", "asset_b": "SLV"}

# Same strategy logic works for any pair!
```

**Note:** The strategy defines abstract slots (`asset_a`, `asset_b`). Symbols are mapped at runtime, making the strategy reusable across any pair of assets.
