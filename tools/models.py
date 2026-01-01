"""Data models for the multi-strategy backtest architecture.

This module defines the core data structures used throughout the backtesting
pipeline, following the "everything is a strategy" philosophy where buy-and-hold
is just a trivial strategy and all strategies are equal peers.

Architecture Flow:
    User Prompt -> Strategy Planner -> Strategy Generator -> Backtest Engine -> Result Combiner
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Strategy Specification (Planner Output)
# =============================================================================


class StrategySpec(BaseModel):
    """Specification for a strategy to generate and run.

    Key Insight: One strategy can produce multiple backtest runs.
    The `directions` list determines how many times the same code
    is executed with different direction parameters.

    Example: "RSI long-only and long/short with SPY buy-and-hold"
    - Planner outputs 2 strategies:
      - RSI (directions=["longonly", "both"])
      - Buy-and-hold SPY (directions=["longonly"])
    - Backtest engine runs 3 times:
      - RSI(longonly), RSI(both), Buy-and-hold(longonly)
    """

    name: str = Field(
        ...,
        description="Display name: 'MA Crossover', 'Buy & Hold SPY'",
    )
    description: str = Field(
        ...,
        description="Natural language description for LLM code generation",
    )
    symbols: List[str] = Field(
        ...,
        description="Symbols to run on: ['AAPL'] or ['AAPL', 'MSFT']",
    )

    # Execution parameters - directions is a LIST for multiple runs
    directions: List[str] = Field(
        default=["longonly"],
        description="List of directions to run: ['longonly'] or ['longonly', 'both'] for comparison",
    )
    execution_price: str = Field(
        default="open",
        description="Price column for execution: 'open' or 'close'",
    )
    slippage: float = Field(
        default=0.0,
        ge=0,
        le=0.05,
        description="Slippage as fraction (0.0005 = 5 bps). Use 0 for buy-and-hold.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=0.5,
        description="Stop loss as fraction of entry price",
    )
    take_profit: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=2.0,
        description="Take profit as fraction of entry price",
    )
    init_cash: float = Field(
        default=100.0,
        ge=1,
        description="Initial cash allocation (default $100)",
    )

    model_config = ConfigDict(extra="forbid")


class PlannerOutput(BaseModel):
    """Output from the Strategy Planner LLM."""

    strategies: List[StrategySpec] = Field(
        ...,
        description="List of strategies to generate and run",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Generated Strategy (Generator Output)
# =============================================================================


class GeneratedStrategy(BaseModel):
    """A fully specified strategy ready to backtest.

    Contains the original spec, plus the LLM-generated code and schemas.
    """

    spec: StrategySpec = Field(
        ...,
        description="Original strategy specification",
    )
    code: str = Field(
        ...,
        description="Generated generate_signals() function",
    )
    data_schema: Dict[str, Any] = Field(
        ...,
        description="Data requirements (slots, frequencies, columns)",
    )
    param_schema: Dict[str, Any] = Field(
        ...,
        description="Strategy parameter definitions",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Actual parameter values (defaults merged with overrides)",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Execution Parameters
# =============================================================================


class ExecutionParams(BaseModel):
    """Execution parameters for a backtest run."""

    direction: str = Field(
        default="longonly",
        description="Position direction: 'longonly', 'shortonly', or 'both'",
    )
    execution_price: str = Field(
        default="open",
        description="Price column for execution: 'open' or 'close'",
    )
    slippage: float = Field(
        default=0.0,
        description="Slippage as fraction",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Stop loss as fraction",
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="Take profit as fraction",
    )
    trailing_stop: bool = Field(
        default=False,
        description="Whether stop loss trails",
    )
    init_cash: float = Field(
        default=100.0,
        description="Initial cash",
    )
    fees: float = Field(
        default=0.001,
        description="Trading fees as fraction",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Trade Records
# =============================================================================


@dataclass
class Trade:
    """Record of a single trade."""

    entry_time: str
    exit_time: str
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float


# =============================================================================
# Performance Metrics
# =============================================================================


@dataclass
class Metrics:
    """Performance metrics for a backtest run."""

    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0  # in days


# =============================================================================
# Symbol-level Results
# =============================================================================


@dataclass
class SymbolResult:
    """Result of running a strategy on a single symbol."""

    symbol: str
    timestamps: List[str] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    trades: List[Trade] = field(default_factory=list)


# =============================================================================
# Backtest Metadata
# =============================================================================


@dataclass
class BacktestMeta:
    """Metadata for a backtest run."""

    timeframe: str = "1Day"
    start_date: str = ""
    end_date: str = ""
    total_bars: int = 0
    warmup_bars: int = 0


# =============================================================================
# Per-Strategy Backtest Result
# =============================================================================


@dataclass
class BacktestRunResult:
    """Result of running a single strategy with a specific direction.

    This represents ONE backtest run (e.g., "RSI Long-Only on SPY").
    A GeneratedStrategy with directions=["longonly", "both"] produces
    TWO BacktestRunResult objects.
    """

    strategy_name: str
    direction: str  # The specific direction for this run

    # Per-symbol results (strategy may run on multiple symbols)
    results_by_symbol: Dict[str, SymbolResult] = field(default_factory=dict)

    # Execution parameters used
    execution: ExecutionParams = field(default_factory=ExecutionParams)

    # Metadata
    meta: BacktestMeta = field(default_factory=BacktestMeta)

    # For error handling
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Chart Series (for widget)
# =============================================================================


class ChartSeries(BaseModel):
    """A single equity curve series for the chart."""

    name: str = Field(
        ...,
        description="Display name: 'AAPL (MA Cross)', 'SPY (B&H)'",
    )
    strategy_name: str = Field(
        ...,
        description="Strategy name: 'MA Crossover', 'Buy & Hold'",
    )
    symbol: str = Field(
        ...,
        description="Symbol: 'AAPL', 'SPY'",
    )
    direction: str = Field(
        default="longonly",
        description="Direction for this run",
    )
    data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Time series data: [{'time': '2024-01-01', 'value': 100}, ...]",
    )
    trades: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional trade markers for the chart",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Metrics Table Row (for widget)
# =============================================================================


class MetricsRow(BaseModel):
    """A row in the metrics comparison table."""

    name: str = Field(
        ...,
        description="Display name: 'AAPL (MA Cross)'",
    )
    strategy_name: str = Field(
        ...,
        description="Strategy name",
    )
    symbol: str = Field(
        ...,
        description="Symbol",
    )
    direction: str = Field(
        default="longonly",
        description="Direction for this run",
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metrics dict: {'total_return': 0.45, 'sharpe_ratio': 1.2, ...}",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Strategy Details (for reproducibility)
# =============================================================================


class StrategyDetails(BaseModel):
    """Details about a strategy for reproducibility."""

    name: str
    description: str
    code: str
    data_schema: Dict[str, Any]
    param_schema: Dict[str, Any]
    params_used: Dict[str, Any]
    execution: Dict[str, Any]

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Combined Metadata
# =============================================================================


class CombinedMeta(BaseModel):
    """Metadata for the combined result."""

    timeframe: str = "1Day"
    start_date: str = ""
    end_date: str = ""
    total_bars: int = 0
    num_strategies: int = 0
    num_runs: int = 0

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Combined Backtest Result (Final Output)
# =============================================================================


class CombinedBacktestResult(BaseModel):
    """Combined results from all strategies, ready for display.

    This is the final output structure sent to the widget.
    All strategies are equal peers - there's no special "benchmark" concept.
    """

    success: bool = True
    error: Optional[str] = None

    # All series to display on chart
    series: List[ChartSeries] = Field(
        default_factory=list,
        description="All equity curve series to display",
    )

    # Metrics table rows
    metrics_table: List[MetricsRow] = Field(
        default_factory=list,
        description="Rows for the metrics comparison table",
    )

    # Strategy details (for reproducibility section)
    strategies: List[StrategyDetails] = Field(
        default_factory=list,
        description="Details for each strategy",
    )

    # Common metadata
    meta: CombinedMeta = Field(
        default_factory=CombinedMeta,
        description="Combined metadata",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Display Configuration
# =============================================================================


class DisplayConfig(BaseModel):
    """Configuration for how results are displayed."""

    normalize: bool = Field(
        default=False,
        description="Normalize all curves to start at 100",
    )
    show_trades: bool = Field(
        default=False,
        description="Show trade entry/exit markers",
    )
    chart_type: str = Field(
        default="equity",
        description="Chart type: 'equity', 'returns', 'drawdown'",
    )
    metrics: List[str] = Field(
        default_factory=lambda: [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "num_trades",
        ],
        description="Metrics to show in the table",
    )

    model_config = ConfigDict(extra="forbid")
