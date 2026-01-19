"""
Backtest Result Container
=========================

Standardized result structure for all backtest runners.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd


@dataclass
class BacktestResult:
    """
    Standardized backtest result container.

    Provides consistent metrics across all strategy complexity levels
    for comparative analysis.
    """
    # Execution status
    success: bool = False
    error: Optional[str] = None

    # Core performance metrics
    total_return: Optional[float] = None          # Total return (decimal, e.g., 0.15 = 15%)
    annualized_return: Optional[float] = None     # CAGR
    sharpe_ratio: Optional[float] = None          # Annualized Sharpe ratio
    sortino_ratio: Optional[float] = None         # Annualized Sortino ratio
    max_drawdown: Optional[float] = None          # Maximum drawdown (decimal, negative)

    # Trade statistics
    total_trades: Optional[int] = None            # Number of completed trades
    win_rate: Optional[float] = None              # Winning trades / total trades
    profit_factor: Optional[float] = None         # Gross profit / gross loss
    avg_trade_return: Optional[float] = None      # Average return per trade

    # Risk metrics
    volatility: Optional[float] = None            # Annualized volatility
    calmar_ratio: Optional[float] = None          # CAGR / max drawdown

    # Time in market
    exposure_time: Optional[float] = None         # Fraction of time with position

    # Benchmark comparison (if provided)
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    # Raw data for further analysis
    equity_curve: Optional[pd.Series] = None      # Portfolio value over time
    returns: Optional[pd.Series] = None           # Daily returns
    positions: Optional[pd.Series] = None         # Position history
    trades: Optional[pd.DataFrame] = None         # Trade log

    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    num_bars: Optional[int] = None

    # Additional details
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "success": self.success,
            "error": self.error,
            "metrics": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "volatility": self.volatility,
                "calmar_ratio": self.calmar_ratio,
            },
            "trades": {
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "avg_trade_return": self.avg_trade_return,
            },
            "exposure": {
                "exposure_time": self.exposure_time,
            },
            "benchmark": {
                "benchmark_return": self.benchmark_return,
                "alpha": self.alpha,
                "beta": self.beta,
            },
            "period": {
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "num_bars": self.num_bars,
            },
            "details": self.details,
        }
        return result
