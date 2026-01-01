# Multi-Strategy Backtest Architecture

> **Status:** DRAFT v4 - Removed role assignments (all strategies are equal)

## Core Insight

**Everything is a strategy.** Buy-and-hold is just a trivial strategy where position = 1 always. All strategies are equal peers - there's no special "benchmark" or "primary" distinction.

From a single user prompt, the system should:
1. **Parse intent** → Determine what strategies are needed
2. **Generate strategies** → Create code for each (MA crossover, buy-and-hold, etc.)
3. **Run backtests** → Execute each strategy independently
4. **Combine results** → Merge into unified output for visualization

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Prompt                               │
│  "RSI long-only and long/short on SPY with buy-and-hold"        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Strategy Planner (LLM)                        │
│                                                                  │
│  Determines what strategies are needed:                          │
│    1. RSI on SPY, directions=["longonly", "both"]               │
│    2. Buy-and-Hold on SPY, directions=["longonly"]              │
│                                                                  │
│  Output: List[StrategySpec]  (2 strategies)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Strategy Generator (LLM)                       │
│                                                                  │
│  For each StrategySpec:                                          │
│    - Generate data_schema, param_schema, code via LLM            │
│    - All strategies go through same generation path              │
│                                                                  │
│  Output: List[GeneratedStrategy]  (2 strategies)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backtest Engine                               │
│                                                                  │
│  For each GeneratedStrategy × each direction:                    │
│    1. RSI + longonly  → BacktestRun                             │
│    2. RSI + both      → BacktestRun                             │
│    3. B&H + longonly  → BacktestRun                             │
│                                                                  │
│  Output: List[BacktestRun]  (3 runs from 2 strategies)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Result Combiner                               │
│                                                                  │
│  Merge all BacktestRuns into unified structure:                  │
│    - Align timestamps (inner join)                               │
│    - Build combined metrics table (3 rows)                       │
│    - Name each run: "SPY (RSI Long)", "SPY (RSI L/S)", "SPY (B&H)" │
│                                                                  │
│  Output: CombinedBacktestResult                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Widget Renderer                            │
│                                                                  │
│  Render 3 equity curves, 3-row metrics table                     │
└─────────────────────────────────────────────────────────────────┘
```

## Data Structures

### StrategySpec (Planner Output)

```python
@dataclass
class StrategySpec:
    """Specification for a strategy to generate and run."""

    name: str                       # Display name: "MA Crossover", "Buy & Hold SPY"
    description: str                # Natural language description for LLM
    symbols: List[str]              # Symbols to run on: ["AAPL"] or ["AAPL", "MSFT"]

    # Execution parameters - directions is a LIST for multiple runs
    directions: List[str]           # ["longonly"] or ["longonly", "both"] for comparison
    execution_price: str = "open"
    slippage: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    init_cash: float = 100.0        # Initial cash allocation (default $100)
```

**Key Insight**: One strategy can produce multiple backtest runs. The `directions` list determines how many times the same code is executed with different direction parameters.

Example: "RSI long-only and long/short with SPY buy-and-hold"
- Planner outputs 2 strategies: RSI (directions=["longonly", "both"]), Buy-and-hold SPY (directions=["longonly"])
- Backtest engine runs 3 times: RSI(longonly), RSI(both), Buy-and-hold(longonly)
- Result Combiner merges all 3 into unified display

### GeneratedStrategy

```python
@dataclass
class GeneratedStrategy:
    """A fully specified strategy ready to backtest."""

    spec: StrategySpec
    code: str                       # generate_signals() function
    data_schema: Dict[str, Any]
    param_schema: Dict[str, Any]
    params: Dict[str, Any]          # Actual parameter values
```

### BacktestResult (Per-Strategy)

```python
@dataclass
class BacktestResult:
    """Result of running a single strategy."""

    strategy_name: str

    # Per-symbol results (strategy may run on multiple symbols)
    results_by_symbol: Dict[str, SymbolResult]

    # Metadata
    execution: ExecutionParams
    meta: BacktestMeta

@dataclass
class SymbolResult:
    symbol: str
    timestamps: List[str]
    equity_curve: List[float]
    positions: List[float]
    metrics: Metrics
    trades: List[Trade]

@dataclass
class Metrics:
    total_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    profit_factor: float
```

### CombinedBacktestResult (Final Output)

```python
@dataclass
class CombinedBacktestResult:
    """Combined results from all strategies, ready for display."""

    success: bool

    # All series to display on chart
    series: List[ChartSeries]

    # Metrics table rows
    metrics_table: List[MetricsRow]

    # Strategy details (for reproducibility section)
    strategies: List[StrategyDetails]

    # Common metadata
    meta: CombinedMeta

@dataclass
class ChartSeries:
    name: str                       # "AAPL (MA Cross)", "SPY (B&H)"
    strategy_name: str              # "MA Crossover", "Buy & Hold"
    symbol: str                     # "AAPL", "SPY"
    data: List[Dict]                # [{"time": "2024-01-01", "value": 10000}, ...]
    trades: List[Dict] | None       # Optional trade markers

@dataclass
class MetricsRow:
    name: str                       # "AAPL (MA Cross)"
    strategy_name: str
    symbol: str
    metrics: Dict[str, float]
```

## Strategy Planner

The planner interprets user intent and produces a list of strategies to run.

### Input Examples → Output StrategySpecs

**Example 1: Simple single strategy**
```
User: "Backtest RSI mean reversion on AAPL"

Output:
  strategies: [
    StrategySpec(
      name="RSI Mean Reversion",
      description="RSI mean reversion: buy when RSI < 30, sell when RSI > 70",
      symbols=["AAPL"],
      directions=["longonly"]
    )
  ]
```

**Example 2: Strategy with buy-and-hold comparison**
```
User: "Backtest MA crossover on AAPL, compare to buy-and-hold SPY"

Output:
  strategies: [
    StrategySpec(
      name="MA Crossover",
      description="MA crossover strategy on AAPL",
      symbols=["AAPL"],
      directions=["longonly"],
      slippage=0.0005
    ),
    StrategySpec(
      name="Buy & Hold SPY",
      description="Buy and hold SPY - always long",
      symbols=["SPY"],
      directions=["longonly"],
      slippage=0.0  # Theoretical, no trading costs
    )
  ]
```

**Example 3: Multi-symbol strategy with comparison**
```
User: "Backtest momentum on AAPL, MSFT, GOOGL vs SPY"

Output:
  strategies: [
    StrategySpec(
      name="Momentum",
      description="Momentum strategy",
      symbols=["AAPL", "MSFT", "GOOGL"],
      directions=["longonly"]
    ),
    StrategySpec(
      name="Buy & Hold SPY",
      description="Buy and hold SPY - always long",
      symbols=["SPY"],
      directions=["longonly"],
      slippage=0.0
    )
  ]
```

**Example 4: Strategy comparison**
```
User: "Compare RSI vs MACD strategies on SPY"

Output:
  strategies: [
    StrategySpec(
      name="RSI Strategy",
      description="RSI mean reversion",
      symbols=["SPY"],
      directions=["longonly"]
    ),
    StrategySpec(
      name="MACD Strategy",
      description="MACD crossover",
      symbols=["SPY"],
      directions=["longonly"]
    )
  ]
```

**Example 5: Long vs Short comparison**
```
User: "Backtest RSI on SPY, show long-only vs long-short"

Output:
  strategies: [
    StrategySpec(
      name="RSI",
      description="RSI mean reversion strategy",
      symbols=["SPY"],
      directions=["longonly", "both"]  # ONE strategy, TWO runs
    )
  ]

Result: 2 equity curves from 1 strategy code
  - "RSI (Long-Only)"
  - "RSI (Long/Short)"
```

### Planner Prompt

```python
PLANNER_SYSTEM_PROMPT = """You are a trading strategy planner. Your job is to analyze user requests and determine what strategies need to be backtested.

OUTPUT FORMAT - You MUST output valid JSON:
{
    "strategies": [
        {
            "name": "Display name for this strategy",
            "description": "Natural language description for the strategy generator",
            "symbols": ["AAPL", "MSFT"],
            "directions": ["longonly"],
            "execution_price": "open|close",
            "slippage": 0.0005,
            "stop_loss": null,
            "take_profit": null,
            "init_cash": 100.0
        }
    ]
}

IMPORTANT: `directions` is a LIST. One strategy can be run multiple times with different directions.
- The same strategy CODE is generated once
- But executed N times, once per direction in the list
- Each execution produces a separate equity curve in the output

RULES:

1. EVERY STRATEGY IS EQUAL:
   - There is no special "benchmark" or "primary" - all strategies are treated the same
   - Buy-and-hold is just a strategy where position = 1 always
   - The widget displays all strategies as equal peers

2. BUY-AND-HOLD DETECTION:
   - If user mentions "vs SPY", "compare to SPY", "SPY benchmark" → add buy-and-hold SPY strategy
   - If user mentions "vs buy-and-hold", "compare to holding" → add buy-and-hold on same symbol
   - Buy-and-hold strategies typically have slippage=0 (theoretical, no trading)

3. SLIPPAGE:
   - Active strategies: use user-specified or default 0.0005 (5 bps)
   - Buy-and-hold strategies: typically 0 (theoretical, single entry)

4. DIRECTIONS (plural):
   - Default to ["longonly"] unless user specifies otherwise
   - If user says "long/short" or "both directions" → ["both"]
   - If user wants to COMPARE long-only vs long-short → ["longonly", "both"]
     This generates ONE strategy but runs it TWICE with different direction params

5. MULTI-SYMBOL STRATEGIES:
   - If user lists multiple symbols for ONE strategy (e.g., "momentum on AAPL, MSFT, GOOGL")
     → single strategy with symbols=["AAPL", "MSFT", "GOOGL"]
   - If user wants SEPARATE strategies per symbol (e.g., "compare RSI on AAPL vs MSFT")
     → multiple strategies, each with one symbol

EXAMPLES:

User: "Backtest RSI mean reversion on AAPL"
→ strategies: [{name: "RSI", symbols: ["AAPL"], directions: ["longonly"]}]
→ Total runs: 1

User: "Backtest MA crossover on AAPL vs SPY"
→ strategies: [
    {name: "MA Crossover", symbols: ["AAPL"], directions: ["longonly"]},
    {name: "Buy & Hold SPY", description: "Buy and hold SPY - always long", symbols: ["SPY"], directions: ["longonly"], slippage: 0}
  ]
→ Total runs: 2

User: "Test RSI on SPY, show long-only vs long-short"
→ strategies: [
    {name: "RSI", symbols: ["SPY"], directions: ["longonly", "both"]}
  ]
→ Total runs: 2 (same code, different direction params)

User: "RSI long-only and long-short on SPY with buy-and-hold"
→ strategies: [
    {name: "RSI", symbols: ["SPY"], directions: ["longonly", "both"]},
    {name: "Buy & Hold SPY", symbols: ["SPY"], directions: ["longonly"], slippage: 0}
  ]
→ Total runs: 3 (RSI×2 + B&H×1)
"""
```

## Result Combiner

Merges multiple BacktestResults into a single CombinedBacktestResult.

### Timestamp Alignment (Inner Join)

All strategies must be aligned to the **same date range** for fair comparison:

```python
def align_timestamps(results: List[BacktestResult]) -> Tuple[pd.DatetimeIndex, List[BacktestResult]]:
    """Align all results to common timestamp range (inner join).

    This ensures apples-to-apples comparison:
    - If Strategy A has 200-day MA warmup, first 200 days are NaN
    - If Strategy B starts trading from day 1
    - Result: Both clipped to start from day 200

    Returns:
        common_index: The shared timestamp index
        aligned_results: Results with equity curves clipped to common range
    """
    # Find common date range across all strategies
    all_indices = []
    for result in results:
        for sym_result in result.results_by_symbol.values():
            all_indices.append(pd.DatetimeIndex(sym_result.timestamps))

    # Inner join: intersection of all date ranges
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.intersection(idx)

    # Clip all results to common range
    aligned_results = []
    for result in results:
        aligned = clip_result_to_index(result, common_index)
        aligned_results.append(aligned)

    return common_index, aligned_results
```

### Combine Function

```python
def combine_results(
    results: List[BacktestResult],
    display_config: DisplayConfig,
) -> CombinedBacktestResult:
    """Combine multiple strategy results into unified output."""

    # Step 1: Align all timestamps (inner join)
    common_index, aligned_results = align_timestamps(results)

    series = []
    metrics_table = []

    for result in aligned_results:
        strategy_name = result.strategy_name

        # Add series for each symbol in the strategy
        for symbol, sym_result in result.results_by_symbol.items():
            curve = sym_result.equity_curve
            if display_config.normalize:
                curve = normalize_to_100(curve)

            series.append(ChartSeries(
                name=f"{symbol} ({strategy_name})",
                strategy_name=strategy_name,
                symbol=symbol,
                data=build_timeseries(sym_result.timestamps, curve),
                trades=sym_result.trades if display_config.show_trades else None,
            ))

            metrics_table.append(MetricsRow(
                name=f"{symbol} ({strategy_name})",
                strategy_name=strategy_name,
                symbol=symbol,
                metrics=sym_result.metrics,
            ))

    return CombinedBacktestResult(
        success=True,
        series=series,
        metrics_table=metrics_table,
        strategies=[build_strategy_details(r) for r in results],
        meta=build_combined_meta(results),
    )
```

## Display Configuration

```python
@dataclass
class DisplayConfig:
    normalize: bool = False         # Normalize all curves to start at 100
    show_trades: bool = False       # Show trade entry/exit markers
    chart_type: str = "equity"      # "equity" | "returns" | "drawdown"
    metrics: List[str] = field(default_factory=lambda: [
        "total_return", "sharpe_ratio", "max_drawdown", "num_trades"
    ])
```

## Widget Data Structure

```javascript
{
    "success": true,

    "series": [
        {
            "name": "AAPL (MA Crossover)",
            "strategy_name": "MA Crossover",
            "symbol": "AAPL",
            "data": [
                {"time": "2016-01-04", "value": 10000},
                {"time": "2016-01-05", "value": 10125},
                ...
            ],
            "trades": [
                {"time": "2016-02-10", "type": "entry", "direction": "long"},
                {"time": "2016-03-15", "type": "exit", "direction": "long"},
                ...
            ]
        },
        {
            "name": "SPY (Buy & Hold)",
            "strategy_name": "Buy & Hold",
            "symbol": "SPY",
            "data": [...]
        }
    ],

    "metrics_table": [
        {
            "name": "AAPL (MA Crossover)",
            "strategy_name": "MA Crossover",
            "symbol": "AAPL",
            "metrics": {
                "total_return": 0.4532,
                "cagr": 0.0891,
                "sharpe_ratio": 0.87,
                "max_drawdown": -0.2341,
                "num_trades": 42
            }
        },
        {
            "name": "SPY (Buy & Hold)",
            "strategy_name": "Buy & Hold",
            "symbol": "SPY",
            "metrics": {
                "total_return": 0.6872,
                "cagr": 0.1023,
                "sharpe_ratio": 0.64,
                "max_drawdown": -0.1940,
                "num_trades": 1
            }
        }
    ],

    "strategies": [
        {
            "name": "MA Crossover",
            "prompt": "MA crossover strategy on AAPL",
            "code": "def generate_signals(data, params):\n    ...",
            "data_schema": {...},
            "param_schema": {...},
            "params_used": {"fast_window": 10, "slow_window": 30},
            "execution": {
                "direction": "longonly",
                "execution_price": "open",
                "slippage": 0.0005
            }
        },
        {
            "name": "Buy & Hold",
            "code": "def generate_signals(data, params):\n    position = pd.Series(1.0, ...)",
            "execution": {
                "direction": "longonly",
                "slippage": 0.0
            }
        }
    ],

    "meta": {
        "timeframe": "1Day",
        "start_date": "2016-01-04",
        "end_date": "2025-12-31",
        "total_bars": 2513
    }
}
```

## Implementation Plan

### Phase 1: Data Structures
- [ ] Define `StrategySpec`, `GeneratedStrategy`, `BacktestResult`, `CombinedBacktestResult`
- [ ] Create Pydantic models for validation

### Phase 2: Strategy Planner
- [ ] Create planner prompt that outputs `List[StrategySpec]`
- [ ] Handle common patterns: single strategy, with benchmark, comparisons
- [ ] Planner determines slippage (0 for benchmarks, user-specified for primary)

### Phase 3: Refactor Strategy Generator
- [ ] Accept `StrategySpec` as input instead of raw prompt
- [ ] Generate all strategies via LLM
- [ ] Return `GeneratedStrategy`

### Phase 4: Refactor Backtest Engine
- [ ] Accept `GeneratedStrategy` as input
- [ ] Return structured `BacktestResult`
- [ ] Add enhanced metrics (CAGR, Calmar, etc.)
- [ ] Store trade records

### Phase 5: Result Combiner
- [ ] Create `combine_results()` function
- [ ] Align timestamps across strategies
- [ ] Compute relative metrics vs benchmark
- [ ] Handle normalization

### Phase 6: Tool Integration
- [ ] Update backtest tool to use new pipeline
- [ ] Parse display preferences from user prompt or defaults
- [ ] Wire everything together

### Phase 7: Widget Updates
- [ ] Update widget for new data structure
- [ ] Color-code by strategy order (1st = blue, 2nd = green, etc.)
- [ ] Show strategy name in legend
- [ ] Expandable strategy details section

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Everything is a strategy | Yes | Buy-and-hold is just `position = 1`. No special "benchmark" concept. |
| Planner as separate step | Yes | Explicit strategy enumeration, easier to debug |
| All strategies LLM-generated | Yes | Uniform pipeline, no special cases. Even trivial strategies like buy-and-hold go through LLM. |
| No role field | Removed | All strategies are equal peers. Widget colors by order, not by role. |
| Per-strategy slippage | Yes | Each strategy specifies its own slippage. Buy-and-hold typically 0, active strategies use realistic values. |
| Default init_cash | $100 | Small default, user can override per strategy |
| Timestamp alignment | Inner join | All strategies must share identical date ranges for apples-to-apples comparison. If indicators require warmup periods (e.g., 200-day MA), clip all series to the common range where all strategies have valid data. |
| Partial failures | Fail entire request | If any strategy fails to generate or execute, the whole backtest fails. No partial results. Simpler error handling, clearer UX. |

## Future Enhancements

1. **Parameter sweeps** - Run same strategy with different params, compare
2. **Walk-forward analysis** - Train/test splits, out-of-sample validation
3. **Factor attribution** - Decompose returns by factors
4. **Risk parity portfolios** - Weight by inverse volatility
5. **Correlation matrix** - Show correlation between strategy returns
