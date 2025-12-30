"""Sandboxed strategy executor for running generated trading strategies.

This module provides a secure execution environment for AI-generated
trading strategy code, using vectorbt for backtesting. It returns detailed
error information that can be fed back to Codex for iterative fixing.
"""

from __future__ import annotations

import signal
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

from .strategy_generator import ExecutionResult


class ExecutionTimeout(Exception):
    """Raised when strategy execution exceeds time limit."""
    pass


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    profit_factor: float = 0.0


@dataclass
class BacktestResult:
    """Container for backtest results."""
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None

    # Equity curve data
    timestamps: List[str] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    # Performance metrics
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)

    # Signal counts
    entry_signals: int = 0
    exit_signals: int = 0

    # Strategy metadata
    strategy_code: str = ""
    symbols: List[str] = field(default_factory=list)
    timeframe: str = ""

    def to_execution_result(self) -> ExecutionResult:
        """Convert to ExecutionResult for Codex feedback."""
        return ExecutionResult(
            success=self.success,
            error=self.error,
            error_type=self.error_type,
            traceback=self.error_traceback,
            result=self.to_dict() if self.success else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "error": self.error,
            "data": [
                {
                    "timestamp": ts,
                    "open": eq,
                    "high": eq,
                    "low": eq,
                    "close": eq,
                    "volume": 0.0,
                }
                for ts, eq in zip(self.timestamps, self.equity_curve)
            ],
            "metrics": {
                "total_return": self.metrics.total_return,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "sortino_ratio": self.metrics.sortino_ratio,
                "max_drawdown": self.metrics.max_drawdown,
                "win_rate": self.metrics.win_rate,
                "num_trades": self.metrics.num_trades,
                "profit_factor": self.metrics.profit_factor,
                "entry_signals": self.entry_signals,
                "exit_signals": self.exit_signals,
            },
            "meta": {
                "symbols": self.symbols,
                "timeframe": self.timeframe,
                "strategy_code": self.strategy_code,
            },
        }


@contextmanager
def timeout(seconds: int):
    """Context manager for execution timeout."""
    def timeout_handler(signum, frame):
        raise ExecutionTimeout(f"Execution exceeded {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class StrategyExecutor:
    """Executes generated trading strategies in a sandboxed environment."""

    # Restricted builtins for sandbox
    SAFE_BUILTINS = {
        'True': True,
        'False': False,
        'None': None,
        'abs': abs,
        'all': all,
        'any': any,
        'bool': bool,
        'dict': dict,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'int': int,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'pow': pow,
        'range': range,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
        'isinstance': isinstance,
        'type': type,
        'print': print,  # Allow print for debugging
    }

    def __init__(
        self,
        timeout_seconds: int = 60,
        init_cash: float = 10000.0,
        fees: float = 0.001,
    ):
        """Initialize the executor.

        Args:
            timeout_seconds: Maximum execution time in seconds.
            init_cash: Initial portfolio cash.
            fees: Trading fees as a fraction (0.001 = 0.1%).
        """
        self.timeout_seconds = timeout_seconds
        self.init_cash = init_cash
        self.fees = fees

    def execute(
        self,
        code: str,
        prices: pd.DataFrame,
        symbols: List[str],
        timeframe: str,
    ) -> BacktestResult:
        """Execute a strategy and return backtest results.

        Args:
            code: The strategy code containing generate_signals function.
            prices: OHLCV DataFrame with lowercase column names.
            symbols: List of symbols being tested.
            timeframe: Data timeframe string.

        Returns:
            BacktestResult with equity curve, metrics, and detailed error info.
        """
        base_result = BacktestResult(
            success=False,
            strategy_code=code,
            symbols=symbols,
            timeframe=timeframe,
        )

        # Create sandboxed execution environment
        sandbox_globals = {
            '__builtins__': self.SAFE_BUILTINS,
            'pd': pd,
            'np': np,
            'vbt': vbt,
        }
        sandbox_locals: Dict[str, Any] = {}

        # Step 1: Execute the strategy code to define the function
        try:
            with timeout(self.timeout_seconds):
                exec(code, sandbox_globals, sandbox_locals)
        except ExecutionTimeout as e:
            base_result.error = str(e)
            base_result.error_type = "TimeoutError"
            return base_result
        except SyntaxError as e:
            base_result.error = f"Syntax error on line {e.lineno}: {e.msg}"
            base_result.error_type = "SyntaxError"
            base_result.error_traceback = traceback.format_exc()
            return base_result
        except Exception as e:
            base_result.error = str(e)
            base_result.error_type = type(e).__name__
            base_result.error_traceback = traceback.format_exc()
            return base_result

        # Step 2: Get the generate_signals function
        if 'generate_signals' not in sandbox_locals:
            base_result.error = "Code must define a function named 'generate_signals'"
            base_result.error_type = "MissingFunctionError"
            return base_result

        generate_signals = sandbox_locals['generate_signals']

        # Step 3: Execute the strategy function
        try:
            with timeout(self.timeout_seconds):
                result = generate_signals(prices)
        except ExecutionTimeout as e:
            base_result.error = str(e)
            base_result.error_type = "TimeoutError"
            return base_result
        except Exception as e:
            base_result.error = f"Error in generate_signals(): {str(e)}"
            base_result.error_type = type(e).__name__
            base_result.error_traceback = traceback.format_exc()
            return base_result

        # Step 4: Validate the result format
        if not isinstance(result, tuple):
            base_result.error = f"generate_signals() must return a tuple, got {type(result).__name__}"
            base_result.error_type = "ReturnTypeError"
            return base_result

        if len(result) != 2:
            base_result.error = f"generate_signals() must return tuple of 2 elements (entries, exits), got {len(result)}"
            base_result.error_type = "ReturnTypeError"
            return base_result

        entries, exits = result

        if not isinstance(entries, pd.Series):
            base_result.error = f"entries must be a pandas Series, got {type(entries).__name__}"
            base_result.error_type = "ReturnTypeError"
            return base_result

        if not isinstance(exits, pd.Series):
            base_result.error = f"exits must be a pandas Series, got {type(exits).__name__}"
            base_result.error_type = "ReturnTypeError"
            return base_result

        # Step 5: Run the backtest with vectorbt
        try:
            close = prices['close']
            entries_bool = entries.fillna(False).astype(bool)
            exits_bool = exits.fillna(False).astype(bool)

            # Log signal counts for debugging
            entry_count = entries_bool.sum()
            exit_count = exits_bool.sum()
            print(f"[Backtest] Entry signals: {entry_count}, Exit signals: {exit_count}")

            pf = vbt.Portfolio.from_signals(
                close,
                entries=entries_bool,
                exits=exits_bool,
                init_cash=self.init_cash,
                fees=self.fees,
                freq='1D',
            )
        except Exception as e:
            base_result.error = f"Error running vectorbt backtest: {str(e)}"
            base_result.error_type = type(e).__name__
            base_result.error_traceback = traceback.format_exc()
            return base_result

        # Step 6: Extract results
        try:
            equity = pf.value()
            timestamps = []
            for idx in equity.index:
                if isinstance(idx, pd.Timestamp):
                    ts = idx.to_pydatetime()
                elif isinstance(idx, datetime):
                    ts = idx
                else:
                    ts = pd.to_datetime(str(idx)).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                timestamps.append(ts.isoformat())

            # Extract metrics safely
            def safe_metric(func, default=0.0):
                try:
                    val = func()
                    if pd.isna(val) or np.isinf(val):
                        return default
                    return float(val)
                except Exception:
                    return default

            trades = pf.trades
            num_trades = len(trades.records) if hasattr(trades, 'records') else 0

            base_result.success = True
            base_result.timestamps = timestamps
            base_result.equity_curve = [float(v) for v in equity.values]
            base_result.metrics = BacktestMetrics(
                total_return=safe_metric(pf.total_return),
                sharpe_ratio=safe_metric(pf.sharpe_ratio),
                sortino_ratio=safe_metric(pf.sortino_ratio),
                max_drawdown=safe_metric(pf.max_drawdown),
                win_rate=safe_metric(trades.win_rate) if num_trades > 0 else 0.0,
                num_trades=num_trades,
                profit_factor=safe_metric(trades.profit_factor) if num_trades > 0 else 0.0,
            )
            base_result.entry_signals = int(entry_count)
            base_result.exit_signals = int(exit_count)
            return base_result

        except Exception as e:
            base_result.error = f"Error extracting backtest results: {str(e)}"
            base_result.error_type = type(e).__name__
            base_result.error_traceback = traceback.format_exc()
            return base_result


def get_strategy_executor(
    timeout_seconds: int = 60,
    init_cash: float = 10000.0,
    fees: float = 0.001,
) -> StrategyExecutor:
    """Factory function to create a StrategyExecutor instance."""
    return StrategyExecutor(
        timeout_seconds=timeout_seconds,
        init_cash=init_cash,
        fees=fees,
    )
