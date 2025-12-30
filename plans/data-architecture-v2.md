# Data Architecture v2: Unified Time-Series Data Model

> **Status:** DRAFT - All decisions pending review

## Decision Summary

| # | Decision | Options | Status |
|---|----------|---------|--------|
| 1 | Symbol Handling | A: Placeholder, B: Explicit, C: Injected, D: Hybrid | UNDECIDED |
| 2 | Primary Data / Signal Alignment | A: Flag in requirements, B: STRATEGY_CONFIG, C: Convention, D: Infer | UNDECIDED |
| 3 | Parsing Generated Code | A: Execute, B: AST, C: Regex, D: Structured LLM output | UNDECIDED |
| 4 | Multi-Symbol Strategy Support | A: Single only, B: Multi mode, C: Always multi | UNDECIDED |
| 5 | Parameter Schema | A: Simple dict, B: Schema with bounds, C: Pydantic | UNDECIDED |
| 6 | Error Handling (Data Fetch) | A: Fail fast, B: Skip/warn, C: Codex feedback, D: Fallback | UNDECIDED |
| 7 | Caching Strategy | A: None, B: In-memory, C: Persistent, D: Hybrid | UNDECIDED |
| 8 | Fundamental Data Provider | A: Alpha Vantage, B: Polygon, C: FMP, D: Yahoo, E: Multiple | UNDECIDED |
| 9 | Execution Price | A: Always close, B: Configurable, C: Next open | UNDECIDED |

## Overview

This document outlines the architecture for a flexible, agentic backtesting system that treats all data uniformly as time-series and allows AI-generated strategies to declare their data dependencies.

**Note:** The "Reference Implementation" section below shows ONE possible approach combining several options. It is provided for illustration only and does not represent decided choices.

## Core Principles

1. **Data is data** - No distinction between "price data" and "fundamental data". Everything is a time-indexed DataFrame.
2. **Declarative dependencies** - Generated strategy code declares what data it needs via `DATA_REQUIREMENTS`.
3. **Agentic orchestration** - The orchestrator parses requirements and dispatches to appropriate data providers.
4. **Separation of concerns** - Logic, parameters, and data requirements are all independently modifiable.

## Architecture Flow

```
┌─────────────────┐
│  User Prompt    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  CodexAgent     │  → Generates code with DATA_REQUIREMENTS + PARAMS + logic
└────────┬────────┘
         ▼
┌─────────────────┐
│  Orchestrator   │  → Parses DATA_REQUIREMENTS from generated code (via AST)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Data Agents    │  → Alpaca, Alpha Vantage, etc. fetch required data
└────────┬────────┘
         ▼
┌─────────────────┐
│  Executor       │  → Passes data dict + params into generate_signals()
└────────┬────────┘
         ▼
┌─────────────────┐
│  vectorbt       │  → Runs backtest with signals
└─────────────────┘
```

## Generated Code Structure

```python
# Strategy configuration
STRATEGY_CONFIG = {
    "mode": "single_symbol",  # or "multi_symbol"
    "primary_data": "prices",  # which data series signals align with
}

# Data requirements - declarative dependencies
DATA_REQUIREMENTS = [
    {
        "name": "prices",
        "provider": "alpaca",
        "type": "ohlcv",
        "timeframe": "1Day",
    },
    {
        "name": "earnings",
        "provider": "alpha_vantage",
        "type": "quarterly_earnings",
        "fields": ["eps"],
    },
]

# Default parameters (overridable at runtime)
DEFAULT_PARAMS = {
    "pe_threshold": 15,
    "pe_exit": 25,
}

def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Args:
        data: Dict[str, pd.DataFrame] keyed by DATA_REQUIREMENTS names
        params: Strategy parameters (DEFAULT_PARAMS merged with overrides)

    Returns:
        (entries, exits) - boolean Series aligned with primary data index
    """
    prices = data['prices']
    earnings = data['earnings']

    eps = earnings['eps'].reindex(prices.index, method='ffill')
    pe_ratio = prices['close'] / eps

    entries = (pe_ratio < params['pe_threshold']).fillna(False).astype(bool)
    exits = (pe_ratio > params['pe_exit']).fillna(False).astype(bool)

    return entries, exits
```

## Design Decisions (Options)

### 1. Symbol Handling

How does `DATA_REQUIREMENTS` reference symbols?

#### Option A: Placeholder Substitution
```python
DATA_REQUIREMENTS = [
    {"name": "prices", "symbol": "{symbol}", "provider": "alpaca", ...},
]
# Orchestrator substitutes {symbol} → "SPY" at runtime
```
| Pros | Cons |
|------|------|
| Explicit about symbol substitution | Requires string parsing/substitution |
| Works for both single and multi-symbol | Magic placeholder syntax |

#### Option B: Explicit Symbols in Requirements
```python
DATA_REQUIREMENTS = [
    {"name": "spy_prices", "symbol": "SPY", ...},
    {"name": "qqq_prices", "symbol": "QQQ", ...},
]
```
| Pros | Cons |
|------|------|
| Clear what data is fetched | Strategy is tied to specific symbols |
| Good for cross-symbol strategies | Can't reuse strategy for different symbols |

#### Option C: Symbol-Agnostic (Injected at Runtime)
```python
DATA_REQUIREMENTS = [
    {"name": "prices", "provider": "alpaca", ...},  # no symbol field
]
# Orchestrator runs for each symbol in payload.symbols
```
| Pros | Cons |
|------|------|
| Strategy is reusable across symbols | Doesn't support cross-symbol strategies |
| Simple requirements structure | Need separate mode for pairs trading |

#### Option D: Hybrid (Mode-Based)
- `single_symbol` mode: Symbol injected at runtime (Option C)
- `multi_symbol` mode: Symbols explicit in requirements (Option B)

| Pros | Cons |
|------|------|
| Supports both use cases | More complex |
| Flexible | Two different patterns to understand |

**Status:** UNDECIDED

---

### 2. Primary Data / Signal Alignment

Signals must align with one time index. How do we specify which?

#### Option A: Flag in DATA_REQUIREMENTS
```python
DATA_REQUIREMENTS = [
    {"name": "prices", "provider": "alpaca", ..., "primary": True},
    {"name": "earnings", "provider": "alpha_vantage", ...},
]
```
| Pros | Cons |
|------|------|
| Co-located with data definition | Only one can be primary |
| Clear which is primary | Clutters requirements |

#### Option B: Separate STRATEGY_CONFIG
```python
STRATEGY_CONFIG = {
    "primary_data": "prices",
}
```
| Pros | Cons |
|------|------|
| Clean separation | Another top-level constant |
| Can include other config | More to parse |

#### Option C: Convention (First is Primary)
```python
DATA_REQUIREMENTS = [
    {"name": "prices", ...},  # First = primary by convention
    {"name": "earnings", ...},
]
```
| Pros | Cons |
|------|------|
| Simple, no extra syntax | Implicit, easy to forget |
| Less to configure | Order matters (fragile) |

#### Option D: Infer from Function
The orchestrator inspects which data series the returned signals align with (by index comparison).

| Pros | Cons |
|------|------|
| No configuration needed | Requires execution to determine |
| Always correct | Can't validate before running |

**Status:** UNDECIDED

---

### 3. Parsing Generated Code

How do we extract `DATA_REQUIREMENTS`, `DEFAULT_PARAMS`, etc. from generated code?

#### Option A: Execute and Read
```python
exec(code, sandbox_globals, sandbox_locals)
requirements = sandbox_locals.get('DATA_REQUIREMENTS', [])
```
| Pros | Cons |
|------|------|
| Simple | Security risk (code execution) |
| Handles complex expressions | Can have side effects |

#### Option B: AST Parsing
```python
import ast
tree = ast.parse(code)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        if target.id == 'DATA_REQUIREMENTS':
            return ast.literal_eval(node.value)
```
| Pros | Cons |
|------|------|
| Safe (no execution) | Only works for literal values |
| No side effects | More complex implementation |

#### Option C: Regex Extraction
```python
import re
match = re.search(r'DATA_REQUIREMENTS\s*=\s*(\[.*?\])', code, re.DOTALL)
requirements = json.loads(match.group(1))
```
| Pros | Cons |
|------|------|
| Simple | Fragile, breaks on edge cases |
| Fast | Doesn't handle Python syntax well |

#### Option D: Structured Output from LLM
Ask Codex to return JSON separately from code:
```json
{
  "code": "def generate_signals...",
  "data_requirements": [...],
  "default_params": {...}
}
```
| Pros | Cons |
|------|------|
| Clean separation | Changes LLM interface |
| Easy to parse | Two things to validate |
| Type-safe | Code and requirements can diverge |

**Status:** UNDECIDED

---

### 4. Multi-Symbol Strategy Support

How do we handle strategies that need multiple symbols (pairs trading, relative value)?

#### Option A: Single Mode Only
All strategies run independently per symbol. No cross-symbol support.

| Pros | Cons |
|------|------|
| Simple | Can't do pairs trading |
| Easy to parallelize | Limited strategy types |

#### Option B: Multi-Symbol Mode
Add `mode: "multi_symbol"` that passes all data in one call:
```python
data = {
    'spy': spy_prices_df,
    'qqq': qqq_prices_df,
}
signals = generate_signals(data, params)
```
| Pros | Cons |
|------|------|
| Supports pairs trading | More complex orchestration |
| Flexible | Signal alignment is trickier |

#### Option C: Always Multi-Symbol
Always pass data as dict keyed by symbol, even for single-symbol:
```python
# Single symbol
data = {'SPY': {'prices': df}}
# Multi symbol
data = {'SPY': {'prices': df}, 'QQQ': {'prices': df}}
```
| Pros | Cons |
|------|------|
| Consistent interface | More verbose for simple cases |
| No mode switching | Strategy must handle both |

**Status:** UNDECIDED

---

### 5. Parameter Schema / Validation

Should we have typed parameter schemas beyond just `DEFAULT_PARAMS`?

#### Option A: Simple Dict Only
```python
DEFAULT_PARAMS = {
    "pe_threshold": 15,
    "ma_window": 20,
}
```
| Pros | Cons |
|------|------|
| Simple | No validation |
| Easy to generate | No bounds for optimization |

#### Option B: Schema with Types and Bounds
```python
PARAM_SCHEMA = {
    "pe_threshold": {"type": "float", "min": 0, "max": 100, "default": 15},
    "ma_window": {"type": "int", "min": 1, "max": 500, "default": 20},
}
```
| Pros | Cons |
|------|------|
| Validation possible | More complex to generate |
| UI can auto-generate inputs | More to parse |
| Optimization knows bounds | |

#### Option C: Pydantic Model
```python
class StrategyParams(BaseModel):
    pe_threshold: float = Field(default=15, ge=0, le=100)
    ma_window: int = Field(default=20, ge=1, le=500)
```
| Pros | Cons |
|------|------|
| Full validation | Hard for LLM to generate |
| IDE autocomplete | Requires class definition |

**Status:** UNDECIDED

---

### 6. Error Handling for Data Fetching

What happens when a data provider fails or returns no data?

#### Option A: Fail Fast
Any data fetch error fails the entire backtest.

| Pros | Cons |
|------|------|
| Simple | One bad symbol kills everything |
| Clear failure | No partial results |

#### Option B: Skip and Warn
Skip symbols with errors, return partial results with warnings.

| Pros | Cons |
|------|------|
| Resilient | Partial results may confuse |
| Gets some results | Silent failures possible |

#### Option C: Feed Back to Codex
Feed error back to Codex agent to fix `DATA_REQUIREMENTS`.

| Pros | Cons |
|------|------|
| Self-healing | Uses more API calls |
| Handles typos | May not be fixable |

#### Option D: Fallback Providers
Try alternative providers if primary fails (e.g., Yahoo if Alpaca fails).

| Pros | Cons |
|------|------|
| Resilient | Data may differ between providers |
| Automatic recovery | Complex configuration |

**Status:** UNDECIDED

---

### 7. Caching Strategy

How do we cache fetched data to avoid redundant API calls?

#### Option A: No Caching
Fetch fresh data every time.

| Pros | Cons |
|------|------|
| Simple | Slow, wasteful |
| Always fresh | May hit rate limits |

#### Option B: In-Memory Session Cache
Cache data for the duration of the session/request.

| Pros | Cons |
|------|------|
| Simple implementation | Lost between requests |
| No persistence concerns | Re-fetches on restart |

#### Option C: Persistent Cache (SQLite/Redis)
Store fetched data in database with TTL.

| Pros | Cons |
|------|------|
| Fast subsequent requests | More infrastructure |
| Survives restarts | Cache invalidation complexity |

#### Option D: Hybrid (Memory + Disk)
In-memory for hot data, disk for historical.

| Pros | Cons |
|------|------|
| Best performance | Most complex |
| Handles both cases | Two systems to manage |

**Status:** UNDECIDED

---

### 8. Fundamental Data Provider

Which provider to use for fundamental data (earnings, balance sheet, etc.)?

#### Option A: Alpha Vantage
| Pros | Cons |
|------|------|
| Free tier available | 5 calls/min rate limit (free) |
| Good coverage | Limited historical depth |

#### Option B: Polygon.io
| Pros | Cons |
|------|------|
| High quality data | Paid only |
| Good API | Cost |

#### Option C: Financial Modeling Prep
| Pros | Cons |
|------|------|
| Comprehensive | Paid for full access |
| Good free tier | |

#### Option D: Yahoo Finance (yfinance)
| Pros | Cons |
|------|------|
| Free | Unofficial, may break |
| No API key needed | Limited/inconsistent data |

#### Option E: Multiple (with fallback)
Support multiple providers, use as fallbacks.

| Pros | Cons |
|------|------|
| Resilient | More implementation work |
| Flexibility | Data consistency issues |

**Status:** UNDECIDED

---

### 9. Execution Price Configuration

Which price column should vectorbt use for trade execution?

#### Option A: Always Close
Assume trades execute at close price.

| Pros | Cons |
|------|------|
| Simple | Not realistic for some strategies |
| Standard practice | |

#### Option B: Configurable in STRATEGY_CONFIG
```python
STRATEGY_CONFIG = {
    "execution_price": "open",  # or "close", "vwap"
}
```
| Pros | Cons |
|------|------|
| Flexible | More configuration |
| More realistic | VWAP needs additional data |

#### Option C: Signal-Based (Next Open)
Entry signals execute at next bar's open (more realistic).

| Pros | Cons |
|------|------|
| Avoids lookahead bias | More complex |
| Realistic | Slightly different semantics |

**Status:** UNDECIDED

---

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
        """Provider identifier used in DATA_REQUIREMENTS."""
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
PROVIDERS.register(AlphaVantageProvider())
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
    params: Optional[Dict[str, Any]] = None  # Override DEFAULT_PARAMS

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

        # 3. Parse declarative sections (safe AST parsing)
        config = parse_strategy_config(code)
        requirements = parse_requirements(code)
        default_params = parse_default_params(code)

        # 4. Merge params (user overrides defaults)
        params = {**default_params, **(payload.params or {})}

        # 5. Determine execution mode
        if config.get("mode") == "multi_symbol":
            return self._run_multi_symbol(code, requirements, params, payload)
        else:
            return self._run_single_symbol(code, requirements, params, payload)

    def _run_single_symbol(self, code, requirements, params, payload):
        results = {}
        for symbol in payload.symbols:
            # Fetch data for this symbol
            data = self._fetch_data(requirements, symbol, payload.start, payload.end)

            # Execute strategy
            result = self.executor.execute(code, data, params)
            results[symbol] = result

        return results

    def _fetch_data(self, requirements, symbol, start, end):
        data = {}
        for req in requirements:
            provider = self.providers.get(req['provider'])
            df = self.cache.get_or_fetch(
                provider=provider,
                symbol=symbol,
                data_type=req['type'],
                start=start,
                end=end,
                **{k: v for k, v in req.items() if k not in ['name', 'provider', 'type']}
            )
            data[req['name']] = df
        return data
```

## System Prompt Updates

The CodexAgent system prompt must document available providers:

```
AVAILABLE DATA PROVIDERS:

alpaca:
  - type: "ohlcv"
  - timeframes: "1Min", "5Min", "15Min", "1Hour", "1Day"
  - columns returned: open, high, low, close, volume

alpha_vantage:
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
1. Provider name is valid (alpaca, alpha_vantage, yahoo)
2. Data type is supported by the provider
3. Required fields are available

Output the corrected code with fixed DATA_REQUIREMENTS."""

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
- [ ] AST parsing for DATA_REQUIREMENTS, DEFAULT_PARAMS, STRATEGY_CONFIG
- [ ] Updated system prompt with provider documentation
- [ ] Updated `generate_signals(data, params)` signature
- [ ] Orchestrator with single-symbol mode

### Phase 2: Enhanced Data Support (Priority: Medium)

- [ ] Alpha Vantage provider (or alternative fundamental data source)
- [ ] In-memory caching
- [ ] Error handling with Codex feedback for data issues
- [ ] Yahoo Finance provider (fallback/alternative)

### Phase 3: Advanced Features (Priority: Low)

- [ ] Multi-symbol mode for cross-asset strategies
- [ ] Parameter schema with validation and bounds
- [ ] Persistent caching (SQLite or Redis)
- [ ] Strategy metadata (name, description, version)
- [ ] Parameter optimization integration

## Open Questions

### Data & Providers

1. **Which fundamental data provider?** Alpha Vantage has a free tier but rate limits. Polygon.io and Financial Modeling Prep are alternatives. See Decision #8.

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

5. **Execution price column?** Currently assumes `close`. See Decision #9.

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
STRATEGY_CONFIG = {
    "mode": "single_symbol",
    "primary_data": "prices",
}

DATA_REQUIREMENTS = [
    {"name": "prices", "provider": "alpaca", "type": "ohlcv", "timeframe": "1Day"},
]

DEFAULT_PARAMS = {
    "rsi_window": 14,
    "oversold": 30,
    "overbought": 70,
}

def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    prices = data['prices']
    rsi = vbt.RSI.run(prices['close'], window=params['rsi_window']).rsi

    entries = (rsi < params['oversold']).fillna(False).astype(bool)
    exits = (rsi > params['overbought']).fillna(False).astype(bool)

    return entries, exits
```

### PE Ratio Strategy (Price + Fundamentals)

```python
STRATEGY_CONFIG = {
    "mode": "single_symbol",
    "primary_data": "prices",
}

DATA_REQUIREMENTS = [
    {"name": "prices", "provider": "alpaca", "type": "ohlcv", "timeframe": "1Day"},
    {"name": "earnings", "provider": "alpha_vantage", "type": "quarterly_earnings", "fields": ["eps"]},
]

DEFAULT_PARAMS = {
    "pe_buy_threshold": 15,
    "pe_sell_threshold": 25,
}

def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    prices = data['prices']
    earnings = data['earnings']

    # Forward-fill quarterly EPS to daily
    eps = earnings['eps'].reindex(prices.index, method='ffill')
    pe_ratio = prices['close'] / eps

    entries = (pe_ratio < params['pe_buy_threshold']).fillna(False).astype(bool)
    exits = (pe_ratio > params['pe_sell_threshold']).fillna(False).astype(bool)

    return entries, exits
```

### Pairs Trading Strategy (Multi-Symbol)

```python
STRATEGY_CONFIG = {
    "mode": "multi_symbol",
    "primary_data": "spy",
}

DATA_REQUIREMENTS = [
    {"name": "spy", "symbol": "SPY", "provider": "alpaca", "type": "ohlcv", "timeframe": "1Day"},
    {"name": "qqq", "symbol": "QQQ", "provider": "alpaca", "type": "ohlcv", "timeframe": "1Day"},
]

DEFAULT_PARAMS = {
    "zscore_entry": 2.0,
    "zscore_exit": 0.5,
    "lookback": 20,
}

def generate_signals(data: dict, params: dict) -> tuple[pd.Series, pd.Series]:
    spy = data['spy']['close']
    qqq = data['qqq']['close']

    # Calculate spread
    spread = spy / qqq
    spread_mean = spread.rolling(params['lookback']).mean()
    spread_std = spread.rolling(params['lookback']).std()
    zscore = (spread - spread_mean) / spread_std

    # Mean reversion signals
    entries = (zscore < -params['zscore_entry']).fillna(False).astype(bool)
    exits = (abs(zscore) < params['zscore_exit']).fillna(False).astype(bool)

    return entries, exits
```
