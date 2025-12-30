# Data Architecture v2: Unified Time-Series Data Model

> **Status:** DRAFT - All decisions pending review

## Decision Summary

All major decisions have been made. See **Decided** section below.

> **Note:** Execution frequency and price are configured via built-in parameters in PARAM_SCHEMA. All data sources are resampled to the `execution_frequency` before being passed to the strategy.

> **Decided:**
> - **LLM Output Format**: Structured JSON output using OpenAI JSON mode (`response_format: {"type": "json_object"}`) with separate `data_schema`, `param_schema`, and `code` fields. No parsing of Python code needed.
> - **DATA_SCHEMA** uses rich structure (frequency, columns, description). No `resample` field - resampling is controlled globally via `execution_frequency`.
> - **PARAM_SCHEMA** always includes three built-in parameters: `execution_frequency` (enum: 1Min, 5Min, 1Hour, 1Day, 1Week, 1Month), `execution_price` (enum: open, close), and `direction` (enum: longonly, shortonly, both). All data is resampled to `execution_frequency` before being passed to the strategy.
> - **Symbol Handling**: Symbols are NEVER in DATA_SCHEMA. DATA_SCHEMA defines data slots (e.g., `prices`, `asset_a`, `asset_b`). Symbols are always injected at runtime, making strategies reusable across different symbols.
> - **Multi-Symbol Support**: Naturally handled by DATA_SCHEMA slots. A pairs strategy defines `asset_a` and `asset_b` slots; runtime maps them to actual symbols (e.g., SPY, QQQ). Same logic works for any pair.
> - **Symbol Extraction**: ChatGPT extracts symbols from user prompts and passes them separately from the strategy logic. Symbol changes don't require code regeneration.
> - **Error Handling**: Fail fast with structured ERROR_SCHEMA. Errors are returned to the widget for user-friendly visualization.
> - **Caching**: No caching for now. Fetch fresh data on every request. Caching can be added later if needed.
> - **Data Providers**: Alpaca for OHLCV price data, Financial Modeling Prep (FMP) for fundamentals. No Yahoo - keeping it simple with one provider per data type.
> - **Strategy Return Format**: Strategies return **position targets** as `dict[str, pd.Series]`, where each Series contains target positions (+1 long, 0 flat, -1 short) for each data slot. Orchestrator converts position changes to entry/exit signals for vectorbt. This unified format handles single-asset, long/short, and multi-asset (pairs) strategies consistently using `from_signals()`.
> - **Strategy Types**: Two types planned - position-based (`from_signals()`) and stateful/dynamic (`from_order_func()`). **Phases 1-6 cover all position-based strategies** including single-asset, long/short, and pairs trading. Stateful strategies requiring portfolio state or dynamic sizing are deferred to Phase 7.
> - **Execution Modes**: The `direction` parameter filters position targets: `longonly` clamps to [0,1], `shortonly` clamps to [-1,0], `both` allows full [-1,0,1] range. Multiple modes can be run simultaneously for comparison.
> - **Risk Management**: Built-in parameters `stop_loss`, `take_profit`, `trailing_stop`, and `slippage` are passed directly to vectorbt's `from_signals()`. Stop loss and take profit are percentage-based (e.g., 0.05 = 5%). Slippage is also percentage-based (e.g., 0.001 = 0.1% = 10 bps). Set to `null` to disable.
> - **Provider Selection**: Alpaca for OHLCV, FMP for fundamentals. No fallback providers - fail fast if primary fails.
> - **Signal Execution Timing**: Signals at time `t` execute on bar `t+1` to prevent lookahead bias. Orchestrator shifts signals before passing to vectorbt. Codex must NOT generate shifted signals - it returns raw signals, and the orchestrator applies `.shift(1)` uniformly. This keeps strategy code simple and ensures consistent lookahead prevention.
> - **Fundamental Data Timestamps**: All fundamental data (earnings, balance sheets) must be indexed by **filing/release date**, NOT period end date, to prevent lookahead bias.
> - **Timezone Handling**: All timestamps are normalized to America/New_York internally (US market hours). DATA_SCHEMA includes optional `timestamp_tz` field to specify source timezone for conversion.
> - **Resampling Rules**: OHLCV downsampling uses proper aggregation (O=first, H=max, L=min, C=last, V=sum). Weekly = W-FRI (week ending Friday), Monthly = BME (business month end).
> - **Resampling Behavior**: All data sources are resampled to `execution_frequency`. Higher-frequency data (e.g., hourly→daily) is downsampled with aggregation. Lower-frequency data (e.g., quarterly→daily) is forward-filled, representing "last known value" until the next data point. This enables combining price data with fundamental data (e.g., PE ratio = daily close / quarterly EPS).
> - **Resampling Constraint**: `execution_frequency` must be >= the lowest frequency in DATA_SCHEMA. Upsampling price data (e.g., daily→hourly) is not allowed as it creates artificial data points. Fundamentals can always be forward-filled to higher frequencies.
> - **Two-Level LLM Processing**: ChatGPT (outer LLM) handles natural language processing - extracting symbols, parameter values, and execution settings from user prompts. Codex (inner LLM) handles code generation only - producing DATA_SCHEMA, strategy-specific PARAM_SCHEMA, and code. ChatGPT passes extracted values via the `params` dict; Codex never sees built-in parameter values.
> - **Built-in Parameter Injection**: Orchestrator owns and injects the built-in parameters (`execution_frequency`, `execution_price`, `direction`, `stop_loss`, `take_profit`, `trailing_stop`, `slippage`). Codex only generates strategy-specific parameters (e.g., `fast_window`, `rsi_threshold`). This keeps Codex focused, avoids wasted tokens, and ensures built-in definitions are always correct.
> - **Position Logic Pattern**: Strategies use explicit state machine with forward-fill. Set position to `np.nan` by default, assign target values (+1/0/-1) only when conditions trigger, then `.ffill().fillna(0)` to maintain position until next signal. This prevents ambiguous position assignments.

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
| "Add a 5% stop-loss" | ❌ No | Only params (built-in) |
| "Run weekly instead of daily" | ❌ No | Only params (built-in) |
| "Switch to RSI strategy" | ✅ Yes | New strategy |
| "Add a momentum filter" | ✅ Yes | Logic change |

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

The `execution_frequency` parameter (in PARAM_SCHEMA) controls how often the strategy runs. **All data sources are resampled to this single frequency**, ensuring the strategy receives aligned DataFrames with identical indices.

**Key concepts:**
- `DATA_SCHEMA.frequency` - The native frequency to fetch from the provider (e.g., "1Day" for prices, "quarterly" for earnings)
- `PARAM_SCHEMA.execution_frequency` - The common frequency for strategy execution (explicitly specified by user or defaulted)

**Resampling rules:**
| Source Frequency | Target Frequency | Method | Example |
|------------------|------------------|--------|---------|
| Higher (1Hour) | Lower (1Day) | Downsample with aggregation | Hourly bars → Daily OHLCV (O=first, H=max, L=min, C=last, V=sum) |
| Lower (quarterly) | Higher (1Day) | Forward-fill | Q1 EPS known on filing date → same value every day until Q2 filing |
| Same | Same | No change | Daily prices with daily execution |

**Why forward-fill makes sense for fundamentals:**
- Quarterly EPS of $2.50 reported on April 15 represents the "known" value
- This value remains the best available information until the next quarterly report
- Forward-filling to daily allows calculating metrics like PE ratio: `daily_close / eps`

The orchestrator:
1. Fetches each data series at its native `frequency`
2. Resamples ALL data to `execution_frequency`:
   - Downsampling: Uses proper aggregation (OHLCV rules for prices, last value for fundamentals)
   - Upsampling: Forward-fills (value remains constant until next data point)
3. Aligns indices across all DataFrames
4. Passes aligned data dict to `generate_signals()`

**Result:** Strategy code never needs to handle resampling or index alignment - all `data[slot]` DataFrames have identical indices at `execution_frequency`.

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

### Execution Semantics and Lookahead Prevention

**CRITICAL:** Proper handling of timing is essential to avoid lookahead bias (using future information).

#### Signal Execution Timing

Signals generated at time `t` execute on bar `t+1`:

```
Bar:        t-1     t       t+1     t+2
            │       │       │       │
Signal:     ────────●───────────────────  (signal generated using data up to t)
                            │
Trade:      ────────────────●───────────  (trade executes on t+1 open or close)
```

**vectorbt configuration:**
```python
pf = vbt.Portfolio.from_signals(
    close,
    entries,
    exits,
    upon_long_conflict='ignore',   # don't act on same-bar conflicts
    upon_short_conflict='ignore',
    accumulate=False,
)
# Note: vectorbt's default behavior executes on the SAME bar as signal.
# To enforce t+1 execution, shift signals: entries.shift(1), exits.shift(1)
```

#### Fundamental Data: Filing Date vs Period End

Fundamental data (earnings, balance sheets) has two relevant dates:
- **Period end date:** When the fiscal period ends (e.g., Dec 31)
- **Filing/release date:** When the data becomes publicly available (e.g., Feb 15)

**Using period end date causes lookahead bias** - the strategy would "know" Q4 earnings on Dec 31 when they weren't released until February.

```
Period End:     Dec 31 ─────────────────────────────────
                        │
Filing Date:    ────────────────── Feb 15 ──────────────
                                    │
Data Available: ════════════════════●═══════════════════
                                    │
Forward-fill:   ════════════════════●●●●●●●●●●●●●●●●●●●●
```

**DATA_SCHEMA must specify timestamp semantics:**
```python
DATA_SCHEMA = {
    "earnings": {
        "frequency": "quarterly",
        "columns": ["eps"],
        "timestamp_type": "filing_date",  # NOT period_end
        "description": "Quarterly EPS, indexed by filing/release date",
    },
}
```

#### Trading Calendar Considerations

Resampling to weekly/monthly frequencies requires explicit calendar rules:

| Frequency | Pandas Alias | Meaning |
|-----------|--------------|---------|
| `1Week` | `W-FRI` | Week ending Friday (standard US trading week) |
| `1Month` | `BME` | Business month end (last trading day) |

**Holidays and weekends:**
- Price data naturally excludes non-trading days
- When upsampling fundamentals to daily, forward-fill only to trading days
- Use `pandas_market_calendars` for accurate trading calendars if needed

#### OHLCV Downsampling Rules

When downsampling price data (e.g., daily → weekly):

| Column | Aggregation | Rationale |
|--------|-------------|-----------|
| `open` | `first` | Opening price of the period |
| `high` | `max` | Highest price during period |
| `low` | `min` | Lowest price during period |
| `close` | `last` | Closing price of the period |
| `volume` | `sum` | Total volume over period |

```python
df.resample('W-FRI').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
})
```

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
│  Alpaca (prices), FMP (fundamentals)                                    │
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
        "data_type": "ohlcv",      # for proper resampling aggregation
        "timestamp_tz": "America/New_York",  # source timezone (Alpaca returns ET)
        "description": "Daily OHLCV price data",
    },
    "earnings": {
        "frequency": "quarterly",  # raw data frequency to fetch (will be resampled)
        "columns": ["eps"],
        "data_type": "fundamental",
        "timestamp_type": "filing_date",  # CRITICAL: use release date, not period end
        "timestamp_tz": "America/New_York",  # normalized to ET
        "description": "Quarterly earnings per share, indexed by filing date",
    },
}
```

**Fields:**
- `frequency`: What frequency to fetch from the data provider
- `columns`: Required columns in the DataFrame
- `data_type`: Type of data - `"ohlcv"` or `"fundamental"` (affects resampling aggregation)
- `timestamp_type`: For fundamental data only - `"filing_date"` (required) to prevent lookahead bias
- `timestamp_tz`: Source timezone. All timestamps are normalized to America/New_York internally.
- `description`: Human-readable description

**Note:** All data sources are resampled to `execution_frequency` (from PARAM_SCHEMA) before being passed to the strategy. Lower-frequency data (e.g., quarterly) is forward-filled; higher-frequency data is downsampled.

### 2. PARAM_SCHEMA (Parameter Constraints)

Defines what parameters the strategy accepts. **Every strategy automatically includes built-in parameters** for execution control and risk management:

```python
PARAM_SCHEMA = {
    # ===== BUILT-IN PARAMETERS (always present, injected by orchestrator) =====
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
    "direction": {
        "type": "enum",
        "values": ["longonly", "shortonly", "both"],
        "default": "longonly",
        "description": "Position direction: longonly (buy/sell to cash), shortonly (short/cover), both (long and short positions).",
    },
    "stop_loss": {
        "type": "float",
        "min": 0.001,
        "max": 0.5,
        "default": null,
        "description": "Stop loss as fraction of entry price (e.g., 0.05 = 5%). Null to disable.",
    },
    "take_profit": {
        "type": "float",
        "min": 0.001,
        "max": 2.0,
        "default": null,
        "description": "Take profit as fraction of entry price (e.g., 0.10 = 10%). Null to disable.",
    },
    "trailing_stop": {
        "type": "bool",
        "default": false,
        "description": "If true, stop_loss trails the highest price since entry (only applies if stop_loss is set).",
    },
    "slippage": {
        "type": "float",
        "min": 0,
        "max": 0.05,
        "default": 0.0,
        "description": "Slippage as fraction of price per trade (e.g., 0.001 = 0.1% = 10 bps). Applied to both entries and exits.",
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

**Built-in Parameters (injected by orchestrator, not generated by LLM):**
- `execution_frequency`: Controls how often signals are generated. All data sources are resampled to this frequency.
- `execution_price`: Which price column vectorbt uses for trade execution (`open` or `close`).
- `direction`: Position direction mode - `longonly` exits to cash, `both` allows reversing to short positions.
- `stop_loss`: Percentage-based stop loss (e.g., 0.05 = exit if price drops 5% from entry). Null to disable.
- `take_profit`: Percentage-based take profit (e.g., 0.10 = exit if price rises 10% from entry). Null to disable.
- `trailing_stop`: If true, stop loss trails the highest price since entry instead of being fixed.
- `slippage`: Percentage-based slippage per trade (e.g., 0.001 = 0.1% = 10 bps). Applied to both entries and exits.

### 3. STRATEGY_CODE (Logic)

The function that generates **position targets** for each data slot:

```python
def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """
    Args:
        data: Dict[str, pd.DataFrame] keyed by DATA_SCHEMA keys.
              All DataFrames are pre-resampled to the same frequency.
        params: Strategy parameters, validated against PARAM_SCHEMA.

    Returns:
        Dict mapping data slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short).
        The orchestrator converts position changes to entry/exit signals.

    Uses explicit state machine with forward-fill:
    - Set position to np.nan by default (no change)
    - Assign target values only when conditions trigger
    - Forward-fill to maintain position until next signal
    """
    prices = data['prices']      # guaranteed: ohlcv columns, resampled frequency
    earnings = data['earnings']  # guaranteed: eps column, same frequency (pre-resampled)

    # No manual reindexing needed - orchestrator already aligned everything
    pe_ratio = prices['close'] / earnings['eps']

    # Explicit state machine: nan = no change, then ffill
    position = pd.Series(np.nan, index=prices.index)
    position[pe_ratio < params['pe_threshold']] = 1   # Go long when PE is low
    position[pe_ratio > params['pe_exit']] = 0        # Go flat when PE is high
    position = position.ffill().fillna(0)             # Maintain position until next signal

    return {"prices": position}
```

**Position Target Format:**
- `+1` = long position (buy and hold)
- `0` = flat (no position, in cash)
- `-1` = short position (sell short and hold)

**Orchestrator Conversion:** The orchestrator converts position changes to vectorbt signals:
```python
# Position changes trigger entries/exits
long_entries = (position == 1) & (position.shift(1) != 1)
long_exits = (position != 1) & (position.shift(1) == 1)
short_entries = (position == -1) & (position.shift(1) != -1)
short_exits = (position != -1) & (position.shift(1) == -1)
```

**Note:** The strategy code no longer needs to handle resampling/reindexing or signal shifting. The orchestrator aligns all data to the common `execution_frequency` and applies `.shift(1)` for lookahead prevention.

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
        "data_type": "ohlcv",
        "description": "Daily OHLCV price data",
    },
    "earnings": {
        "frequency": "quarterly",
        "columns": ["eps"],
        "data_type": "fundamental",
        "timestamp_type": "filing_date",  # prevents lookahead bias
        "description": "Quarterly EPS, indexed by filing date",
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
    "direction": {
        "type": "enum",
        "values": ["longonly", "shortonly", "both"],
        "default": "longonly",
        "description": "Position direction mode",
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

def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """Returns position targets at execution_frequency.

    Uses explicit state machine with forward-fill:
    - Set position to np.nan by default
    - Assign target values only when conditions trigger
    - Forward-fill to maintain position until next signal

    Orchestrator converts position targets to entry/exit signals for vectorbt's from_signals().
    """
    prices = data['prices']
    earnings = data['earnings']

    # No reindexing needed - all data already aligned by orchestrator
    pe_ratio = prices['close'] / earnings['eps']

    # Explicit state machine: nan = no change, then ffill
    position = pd.Series(np.nan, index=prices.index)
    position[pe_ratio < params['pe_threshold']] = 1  # Go long when PE is low
    position[pe_ratio > params['pe_exit']] = 0       # Go flat when PE is high
    position = position.ffill().fillna(0)            # Maintain position until next signal

    return {"prices": position}
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

    # Runtime parameters - symbols is a dict mapping DATA_SCHEMA slots to ticker symbols
    # e.g., {"prices": "SPY"} for single-asset strategies
    # e.g., {"asset_a": "GLD", "asset_b": "SLV"} for pairs trading
    symbols: Dict[str, str]  # Maps slot names to ticker symbols
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

        # 6. Run backtest with symbol mappings
        return self._run_backtest(code, data_schema, params, payload, exec_freq)

    def _run_backtest(self, code, data_schema, params, payload, exec_freq):
        # payload.symbols is a dict mapping slot names to symbols
        # e.g., {"prices": "SPY"} or {"asset_a": "GLD", "asset_b": "SLV"}

        # Fetch and resample data for all slots
        data = self._fetch_and_resample(data_schema, payload.symbols, payload.start, payload.end, exec_freq)

        # Execute strategy to get position targets
        positions = self.executor.execute(code, data, params)
        # positions: dict[str, pd.Series] e.g., {"prices": Series of +1/0/-1}

        # Apply direction filtering
        direction = params.get('direction', 'longonly')
        positions = self._apply_direction_filter(positions, direction)

        # Convert positions to vectorbt signals
        signals = self._positions_to_signals(positions)

        # Build price DataFrames for vectorbt
        close_prices = self._build_price_df(data, data_schema, payload.symbols, params.get('execution_price', 'close'))

        # Run vectorbt backtest
        result = self._run_vectorbt(close_prices, signals, payload.init_cash)
        return result

    def _apply_direction_filter(self, positions: dict, direction: str) -> dict:
        """Filter position targets based on direction parameter."""
        filtered = {}
        for slot, pos in positions.items():
            if direction == 'longonly':
                # Clamp to [0, 1]: shorts become flat
                filtered[slot] = pos.clip(lower=0)
            elif direction == 'shortonly':
                # Clamp to [-1, 0]: longs become flat
                filtered[slot] = pos.clip(upper=0)
            else:  # 'both'
                filtered[slot] = pos
        return filtered

    def _positions_to_signals(self, positions: dict) -> dict:
        """Convert position targets to entry/exit signals for vectorbt.

        Args:
            positions: Dict mapping slot names to position Series (+1/0/-1)

        Returns:
            Dict with 'long_entries', 'long_exits', 'short_entries', 'short_exits'
            as DataFrames (columns = slot names)
        """
        slots = list(positions.keys())
        index = positions[slots[0]].index

        long_entries = pd.DataFrame(index=index)
        long_exits = pd.DataFrame(index=index)
        short_entries = pd.DataFrame(index=index)
        short_exits = pd.DataFrame(index=index)

        for slot, pos in positions.items():
            prev_pos = pos.shift(1).fillna(0)

            # Long entry: transition to +1 from non-+1
            long_entries[slot] = (pos == 1) & (prev_pos != 1)
            # Long exit: transition from +1 to non-+1
            long_exits[slot] = (pos != 1) & (prev_pos == 1)
            # Short entry: transition to -1 from non--1
            short_entries[slot] = (pos == -1) & (prev_pos != -1)
            # Short exit: transition from -1 to non--1
            short_exits[slot] = (pos != -1) & (prev_pos == -1)

        # Apply t+1 execution shift (signals at t execute on t+1)
        return {
            'long_entries': long_entries.shift(1).fillna(False),
            'long_exits': long_exits.shift(1).fillna(False),
            'short_entries': short_entries.shift(1).fillna(False),
            'short_exits': short_exits.shift(1).fillna(False),
        }

    def _run_vectorbt(self, close: pd.DataFrame, signals: dict, init_cash: float) -> dict:
        """Run vectorbt backtest with converted signals."""
        pf = vbt.Portfolio.from_signals(
            close,
            entries=signals['long_entries'],
            exits=signals['long_exits'],
            short_entries=signals['short_entries'],
            short_exits=signals['short_exits'],
            init_cash=init_cash,
            freq='1D',
        )
        # Extract results...
        return {
            'equity_curve': pf.value().tolist(),
            'total_return': pf.total_return(),
            'sharpe_ratio': pf.sharpe_ratio(),
            # ... other metrics
        }

    def _fetch_and_resample(self, data_schema, symbols, start, end, exec_freq):
        """
        Args:
            data_schema: Dict of slot definitions (e.g., {"prices": {...}, "earnings": {...}})
            symbols: Dict mapping slot names to symbols (e.g., {"prices": "SPY", "earnings": "SPY"})
        """
        data = {}
        for slot_name, schema in data_schema.items():
            # Look up symbol for this slot from runtime symbols dict
            symbol = symbols.get(slot_name)
            if symbol is None:
                raise ValueError(f"Missing symbol mapping for slot '{slot_name}'")

            # Fetch at source frequency
            provider = self.providers.infer_provider(schema)
            df = self.cache.get_or_fetch(
                provider=provider,
                symbol=symbol,  # symbol comes from runtime, NOT from schema
                frequency=schema['frequency'],
                start=start,
                end=end,
            )

            # Resample to execution_frequency
            if schema['frequency'] != exec_freq:
                df = self._resample_to_exec_freq(df, schema['frequency'], exec_freq)

            data[slot_name] = df
        return data

    def _resample_to_exec_freq(self, df, source_freq, exec_freq, data_type):
        """Resample DataFrame to execution_frequency.

        Args:
            df: DataFrame with DatetimeIndex
            source_freq: Original frequency (e.g., "1Day", "quarterly")
            exec_freq: Target frequency (e.g., "1Week")
            data_type: Type of data for proper aggregation ("ohlcv" or "fundamental")
        """
        # Map frequency strings to pandas offset aliases
        freq_map = {
            "1Min": "1min", "5Min": "5min", "15Min": "15min",
            "1Hour": "1h", "1Day": "1D",
            "1Week": "W-FRI",  # Week ending Friday (standard trading week)
            "1Month": "BME",   # Business month end
            "quarterly": "QE", # Quarter end
        }
        target = freq_map.get(exec_freq, exec_freq)

        if is_higher_frequency(source_freq, exec_freq):
            # Downsample: use proper OHLCV aggregation
            if data_type == "ohlcv":
                return df.resample(target).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                })
            else:
                # For fundamental data, take last known value
                return df.resample(target).last()
        else:
            # Upsample: forward-fill (value known until next update)
            return df.resample(target).ffill()
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
1. Provider name is valid (alpaca, fmp)
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

> **Note:** Phases 1-6 cover **all position-based strategies** using `Portfolio.from_signals()` with the unified position-target return format. This includes single-asset, long/short, and pairs trading strategies. Phase 7 is reserved for stateful strategies requiring `from_order_func()`.

### Phase 1: Parameter Separation & Position Targets

**Goal:** Decouple strategy logic from parameter values; establish position-target return format

- [ ] Update function signature: `generate_signals(data, params) -> dict[str, pd.Series]`
- [ ] Position target format: +1 (long), 0 (flat), -1 (short)
- [ ] Add `params: Optional[Dict]` to BacktestInput
- [ ] Update system prompt to generate parameterized code with position targets
- [ ] Orchestrator converts positions to entry/exit signals for vectorbt
- [ ] `base_code` support for iterative modifications

**Outcome:** Can change MA window, RSI threshold without regenerating code. Unified return format established.

### Phase 2: Structured LLM Output

**Goal:** LLM returns JSON with separate schema and code

- [ ] Update system prompt for JSON output format
- [ ] Parse `{"param_schema": {...}, "code": "..."}`
- [ ] Validate param_schema structure
- [ ] Extract default values from param_schema
- [ ] Merge user-provided params with defaults

**Outcome:** Clean separation of schema and code; auto-generated defaults.

### Phase 3: DATA_SCHEMA & Symbol Injection

**Goal:** Strategies define data slots, symbols injected at runtime

- [ ] Add `data_schema` to LLM JSON output
- [ ] Update signature: `generate_signals(data: dict, params: dict) -> dict[str, pd.Series]`
- [ ] Symbol mapping: `{"prices": "SPY"}` → `data["prices"]`
- [ ] Validate symbols against data_schema slots
- [ ] Strategy reusable across any symbol without code change

**Outcome:** Same MA crossover strategy works for SPY, QQQ, AAPL.

### Phase 4: Built-in Parameters, Direction & Risk Management

**Goal:** Standard execution controls and risk management for all strategies

- [ ] `execution_frequency` - resample data before execution (1Day, 1Week, etc.)
- [ ] `execution_price` - open vs close for trade execution
- [ ] `direction` - longonly, shortonly, both (filters position targets)
- [ ] `stop_loss` - percentage-based stop loss passed to vectorbt `sl_stop`
- [ ] `take_profit` - percentage-based take profit passed to vectorbt `tp_stop`
- [ ] `trailing_stop` - enables trailing stop via vectorbt `sl_trail`
- [ ] Orchestrator injects built-ins (Codex only generates strategy-specific params)
- [ ] Direction filtering: `longonly` clamps to [0,1], `shortonly` clamps to [-1,0], `both` allows [-1,0,1]
- [ ] Orchestrator handles resampling logic with constraint validation

**Outcome:** Any strategy supports daily/weekly execution, long-only/long-short modes, and optional stop loss/take profit.

### Phase 5: Pairs Trading & Multi-Asset Strategies

**Goal:** Support strategies that trade multiple correlated assets

- [ ] Multi-slot DATA_SCHEMA: `{"asset_a": {...}, "asset_b": {...}}`
- [ ] Position targets per asset: `{"asset_a": positions_a, "asset_b": positions_b}`
- [ ] Orchestrator builds multi-column DataFrames for vectorbt
- [ ] Combined equity curve across all positions
- [ ] Per-asset and combined metrics

**Outcome:** Pairs trading (long A, short B) works with same position-target format. Same strategy works for GLD/SLV, SPY/QQQ, etc.

### Phase 6: Multi-Execution Comparison & Fundamental Data

**Goal:** Run same strategy with multiple modes; add fundamental data support

- [ ] Support array values: `direction: ["longonly", "both"]`
- [ ] Run backtest for each direction value
- [ ] Return multiple equity curves in response
- [ ] Widget displays curves with legend (Long-Only vs Long/Short)
- [ ] Per-mode metrics comparison table
- [ ] FMP provider implementation (Financial Modeling Prep)
- [ ] Provider registry pattern
- [ ] Resampling: quarterly → daily (forward-fill with filing dates)

**Outcome:** Single request shows long-only vs long/short performance. PE ratio strategy combining daily prices + quarterly EPS.

### Phase 7: Stateful & Dynamic Strategies *(deferred)*

**Goal:** Support strategies requiring portfolio state or dynamic position sizing

- [ ] `strategy_type: "stateful"` in LLM output
- [ ] `Portfolio.from_order_func()` execution
- [ ] Access to portfolio state (current positions, cash, equity)
- [ ] Dynamic position sizing (volatility-weighted, risk parity)
- [ ] Numba-compatible code generation patterns
- [ ] Advanced order types (limit orders, time-based exits, conditional orders)

**Outcome:** Strategies that need to know current portfolio state to make decisions. Advanced position sizing beyond simple +1/0/-1.

### Phase Summary

| Phase | Focus | Execution API | Status |
|-------|-------|---------------|--------|
| 1 | Parameter separation & position targets | `from_signals()` | Pending |
| 2 | Structured LLM output | `from_signals()` | Pending |
| 3 | DATA_SCHEMA & symbol injection | `from_signals()` | Pending |
| 4 | Built-in parameters & direction filtering | `from_signals()` | Pending |
| 5 | Pairs trading & multi-asset | `from_signals()` | Pending |
| 6 | Multi-execution comparison & fundamentals | `from_signals()` | Pending |
| 7 | Stateful & dynamic strategies | `from_order_func()` | Deferred |

## Open Questions

### Data & Providers

1. ~~**Which fundamental data provider?**~~ **DECIDED:** Financial Modeling Prep (FMP) for earnings, balance sheet, and other fundamental data. See Decision section.

2. ~~**How to handle timezone alignment?**~~ **DECIDED:** All timestamps normalized to America/New_York internally. DATA_SCHEMA includes optional `timestamp_tz` field for source timezone conversion.

3. **Should we support custom/user-provided data?** e.g., user uploads a CSV of proprietary signals. Options:
   - Not supported (v1)
   - File upload endpoint
   - Reference external URLs

4. **What about alternative data?** Sentiment, news, satellite imagery, etc. Same provider pattern or separate?

### Execution & Backtesting

5. **Execution price column?** Currently assumes `close`. See Decision #7.

6. ~~**Slippage and transaction costs?**~~ **DECIDED:** Slippage added as built-in parameter (percentage-based, e.g., 0.001 = 10 bps). Fees already supported. Bid-ask spread can be approximated via slippage.

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
        "data_type": "ohlcv",
        "description": "Daily OHLCV price data",
    },
}

# Note: Codex only generates strategy-specific params.
# Built-in params (execution_frequency, execution_price, direction) are injected by orchestrator.
PARAM_SCHEMA = {
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
        "description": "RSI level considered oversold (go long)",
    },
    "overbought": {
        "type": "float",
        "min": 50,
        "max": 100,
        "default": 70,
        "description": "RSI level considered overbought (go flat)",
    },
}

def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """Returns position targets for each data slot.

    Uses explicit state machine with forward-fill:
    - np.nan = no change to position
    - ffill maintains position until next signal

    Orchestrator handles direction filtering and signal conversion.
    """
    prices = data['prices']
    rsi = vbt.RSI.run(prices['close'], window=params['rsi_window']).rsi

    # Explicit state machine: nan = no change, then ffill
    position = pd.Series(np.nan, index=prices.index)
    position[rsi < params['oversold']] = 1      # Go long when oversold
    position[rsi > params['overbought']] = 0    # Go flat when overbought
    position = position.ffill().fillna(0)       # Maintain position until next signal

    return {"prices": position}
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
        "data_type": "ohlcv",
        "description": "Daily OHLCV price data",
    },
    "earnings": {
        "frequency": "quarterly",
        "columns": ["eps"],
        "data_type": "fundamental",
        "timestamp_type": "filing_date",  # prevents lookahead bias
        "description": "Quarterly EPS, indexed by filing date",
    },
}

# Note: Codex only generates strategy-specific params.
# Built-in params (execution_frequency, execution_price, direction) are injected by orchestrator.
PARAM_SCHEMA = {
    "pe_buy_threshold": {
        "type": "float",
        "min": 0,
        "max": 50,
        "default": 15,
        "description": "PE ratio below which to go long",
    },
    "pe_sell_threshold": {
        "type": "float",
        "min": 10,
        "max": 100,
        "default": 25,
        "description": "PE ratio above which to go flat",
    },
}

def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """Returns position targets for each data slot.

    Uses explicit state machine with forward-fill.
    """
    prices = data['prices']
    earnings = data['earnings']

    # No manual reindexing - orchestrator already aligned to execution_frequency
    pe_ratio = prices['close'] / earnings['eps']

    # Explicit state machine: nan = no change, then ffill
    position = pd.Series(np.nan, index=prices.index)
    position[pe_ratio < params['pe_buy_threshold']] = 1   # Go long when PE is low
    position[pe_ratio > params['pe_sell_threshold']] = 0  # Go flat when PE is high
    position = position.ffill().fillna(0)                 # Maintain position until next signal

    return {"prices": position}
```

**Runtime Input** (for "Run on AAPL"):
```python
symbols = {"prices": "AAPL", "earnings": "AAPL"}
```

---

### Pairs Trading Strategy (Multi-Asset)

> **Note:** Pairs trading uses the same position-target format with multiple slots. The strategy returns positions for each asset (+1/-1), and the orchestrator handles simultaneous execution via `from_signals()` with multi-column DataFrames. This is implemented in **Phase 5**.

```python
# ============== STRATEGY DEFINITION ==============

DATA_SCHEMA = {
    "asset_a": {
        "frequency": "1Day",
        "columns": ["open", "high", "low", "close", "volume"],
        "data_type": "ohlcv",
        "description": "First asset in the pair",
    },
    "asset_b": {
        "frequency": "1Day",
        "columns": ["open", "high", "low", "close", "volume"],
        "data_type": "ohlcv",
        "description": "Second asset in the pair",
    },
}

# Note: Codex only generates strategy-specific params.
# Built-in params are injected by orchestrator. For pairs, direction='both' is typical.
PARAM_SCHEMA = {
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

def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    """Returns position targets for each asset in the pair.

    Uses explicit state machine with forward-fill.
    When spread is low (z < -threshold): long A, short B
    When spread is high (z > threshold): short A, long B
    When spread normalizes: flat both
    """
    a = data['asset_a']['close']
    b = data['asset_b']['close']

    # Calculate spread z-score
    spread = a / b
    spread_mean = spread.rolling(params['lookback']).mean()
    spread_std = spread.rolling(params['lookback']).std()
    zscore = (spread - spread_mean) / spread_std

    # Explicit state machine: nan = no change, then ffill
    pos_a = pd.Series(np.nan, index=a.index)
    pos_b = pd.Series(np.nan, index=b.index)

    # Spread too low: long A, short B (expect spread to increase)
    low_spread = zscore < -params['zscore_entry']
    pos_a[low_spread] = 1   # Long asset A
    pos_b[low_spread] = -1  # Short asset B

    # Spread too high: short A, long B (expect spread to decrease)
    high_spread = zscore > params['zscore_entry']
    pos_a[high_spread] = -1  # Short asset A
    pos_b[high_spread] = 1   # Long asset B

    # Spread normalized: flat both
    normalized = abs(zscore) < params['zscore_exit']
    pos_a[normalized] = 0
    pos_b[normalized] = 0

    # Maintain positions until next signal
    pos_a = pos_a.ffill().fillna(0)
    pos_b = pos_b.ffill().fillna(0)

    return {"asset_a": pos_a, "asset_b": pos_b}
```

**Runtime Input** (symbols injected, strategy is reusable):
```python
# User: "Run pairs trading on SPY vs QQQ"
symbols = {"asset_a": "SPY", "asset_b": "QQQ"}

# User: "Run pairs trading on GLD vs SLV"
symbols = {"asset_a": "GLD", "asset_b": "SLV"}

# Same strategy logic works for any pair!
```

**Note:** The strategy defines abstract slots (`asset_a`, `asset_b`). Symbols are mapped at runtime, making the strategy reusable across any pair of assets. The orchestrator builds multi-column DataFrames for vectorbt's `from_signals()` to handle both positions simultaneously.
