"""
Backtests Module
================

Functions to run vectorbt backtests for generated strategy code.
Each complexity level has its own backtest runner that returns
consistent results for analysis.
"""

from backtests.shared import BacktestResult, load_sample_data
from backtests import simple, medium, complex

__all__ = [
    "BacktestResult",
    "load_sample_data",
]

# Strategy complexity to backtest function mapping
BACKTEST_FUNCTIONS = {
    "simple": simple.run_backtest,
    "medium": medium.run_backtest,
    "complex": complex.run_backtest,
}
