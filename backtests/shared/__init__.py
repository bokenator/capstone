"""
Shared Backtest Utilities
=========================

Common utilities and data structures for backtesting.
"""

from .result import BacktestResult
from .data import load_sample_data
from .metrics import extract_metrics_from_portfolio

__all__ = [
    "BacktestResult",
    "load_sample_data",
    "extract_metrics_from_portfolio",
]
