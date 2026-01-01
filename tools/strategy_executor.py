"""Sandboxed strategy executor for running generated trading strategies.

This module provides a secure execution environment for AI-generated
trading strategy code, using vectorbt for backtesting. It returns detailed
error information that can be fed back to Codex for iterative fixing.

Updated for v2 architecture:
- Accepts position targets (dict[str, pd.Series]) with +1/0/-1 values
- Converts position changes to entry/exit signals
- Applies direction filtering (longonly/shortonly/both)
"""

from __future__ import annotations

import logging
import signal
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

from .strategy_generator import ExecutionResult
from .schemas import apply_direction_filter, positions_to_signals

if TYPE_CHECKING:
    from .models import BacktestRunResult, GeneratedStrategy

# Configure logging
logger = logging.getLogger(__name__)


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

    # Position/signal counts (v2 format)
    position_changes: int = 0
    long_entries: int = 0
    long_exits: int = 0
    short_entries: int = 0
    short_exits: int = 0

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
                "position_changes": self.position_changes,
                "long_entries": self.long_entries,
                "long_exits": self.long_exits,
                "short_entries": self.short_entries,
                "short_exits": self.short_exits,
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
    """Executes generated trading strategies in a sandboxed environment.

    Updated for v2 architecture:
    - Strategies return position targets as dict[str, pd.Series]
    - Position values: +1 (long), 0 (flat), -1 (short)
    - Executor converts positions to entry/exit signals for vectorbt
    """

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

    def _extract_builtin_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract built-in parameter values from params dict.

        Args:
            params: Full params dict including built-in and strategy params.

        Returns:
            Dict with built-in parameter values with defaults applied.
        """
        return {
            "execution_price": params.get("execution_price", "close"),
            "stop_loss": params.get("stop_loss", None),
            "take_profit": params.get("take_profit", None),
            "trailing_stop": params.get("trailing_stop", False),
            "slippage": params.get("slippage", 0.0),
        }

    def execute(
        self,
        code: str,
        prices: pd.DataFrame,
        symbols: List[str],
        timeframe: str,
        params: Optional[Dict[str, Any]] = None,
        direction: str = "longonly",
        data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> BacktestResult:
        """Execute a strategy and return backtest results.

        Args:
            code: The strategy code containing generate_signals function.
            prices: OHLCV DataFrame with lowercase column names (for single-asset).
            symbols: List of symbols being tested.
            timeframe: Data timeframe string.
            params: Strategy parameters (e.g., {'fast_window': 10}).
            direction: Position direction: 'longonly', 'shortonly', or 'both'.
            data: Optional dict of DataFrames for multi-asset strategies.
                  Keys are slot names (e.g., 'asset_a', 'asset_b').
                  If not provided, uses {"prices": prices} for single-asset.

        Returns:
            BacktestResult with equity curve, metrics, and detailed error info.
        """
        if params is None:
            params = {}

        logger.info("=" * 60)
        logger.info("EXECUTING STRATEGY")
        logger.info("=" * 60)
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Direction: {direction}")
        logger.info(f"Data slots provided: {list(data.keys()) if data else ['prices (default)']}")
        logger.info(f"Params: {params}")

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
        logger.info("-" * 40)
        logger.info("Step 1: Executing code to define generate_signals function")
        try:
            with timeout(self.timeout_seconds):
                exec(code, sandbox_globals, sandbox_locals)
            logger.info("Step 1 SUCCESS: Code executed, function defined")
        except ExecutionTimeout as e:
            logger.error(f"Step 1 FAILED: Timeout - {e}")
            base_result.error = str(e)
            base_result.error_type = "TimeoutError"
            return base_result
        except SyntaxError as e:
            logger.error(f"Step 1 FAILED: Syntax error on line {e.lineno}: {e.msg}")
            base_result.error = f"Syntax error on line {e.lineno}: {e.msg}"
            base_result.error_type = "SyntaxError"
            base_result.error_traceback = traceback.format_exc()
            return base_result
        except Exception as e:
            logger.error(f"Step 1 FAILED: {type(e).__name__}: {e}")
            base_result.error = str(e)
            base_result.error_type = type(e).__name__
            base_result.error_traceback = traceback.format_exc()
            return base_result

        # Step 2: Get the generate_signals function
        logger.info("-" * 40)
        logger.info("Step 2: Getting generate_signals function")
        if 'generate_signals' not in sandbox_locals:
            logger.error("Step 2 FAILED: Function 'generate_signals' not found")
            base_result.error = "Code must define a function named 'generate_signals'"
            base_result.error_type = "MissingFunctionError"
            return base_result
        logger.info("Step 2 SUCCESS: Function found")

        generate_signals = sandbox_locals['generate_signals']

        # Step 3: Build data dict for the strategy
        # Use provided data dict for multi-asset, or wrap single prices DataFrame
        logger.info("-" * 40)
        logger.info("Step 3: Building data dict")
        if data is not None:
            strategy_data = data
            logger.info(f"Using provided data dict with {len(data)} slots")
        else:
            strategy_data = {"prices": prices}
            logger.info(f"Using single prices DataFrame with shape {prices.shape}")

        # Step 4: Execute the strategy function with data and params
        logger.info("-" * 40)
        logger.info("Step 4: Executing generate_signals(data, params)")
        try:
            with timeout(self.timeout_seconds):
                result = generate_signals(strategy_data, params)
            logger.info("Step 4 SUCCESS: generate_signals() returned")
        except ExecutionTimeout as e:
            logger.error(f"Step 4 FAILED: Timeout - {e}")
            base_result.error = str(e)
            base_result.error_type = "TimeoutError"
            return base_result
        except Exception as e:
            logger.error(f"Step 4 FAILED: {type(e).__name__}: {e}")
            base_result.error = f"Error in generate_signals(): {str(e)}"
            base_result.error_type = type(e).__name__
            base_result.error_traceback = traceback.format_exc()
            return base_result

        # Step 5: Validate the result format (must be dict[str, pd.Series])
        logger.info("-" * 40)
        logger.info("Step 5: Validating result format")
        if not isinstance(result, dict):
            logger.error(f"Step 5 FAILED: Expected dict, got {type(result).__name__}")
            base_result.error = f"generate_signals() must return a dict, got {type(result).__name__}"
            base_result.error_type = "ReturnTypeError"
            return base_result

        if not result:
            logger.error("Step 5 FAILED: Empty dict returned")
            base_result.error = "generate_signals() returned an empty dict"
            base_result.error_type = "ReturnTypeError"
            return base_result

        logger.info(f"Result contains {len(result)} slot(s): {list(result.keys())}")

        # Validate each position series
        for slot_name, pos_series in result.items():
            if not isinstance(pos_series, pd.Series):
                logger.error(f"Step 5 FAILED: Position for '{slot_name}' is {type(pos_series).__name__}, not Series")
                base_result.error = f"Position for '{slot_name}' must be a pandas Series, got {type(pos_series).__name__}"
                base_result.error_type = "ReturnTypeError"
                return base_result
            logger.info(f"  Slot '{slot_name}': Series with {len(pos_series)} values, unique positions: {pos_series.dropna().unique().tolist()}")

        logger.info("Step 5 SUCCESS: Result format is valid")

        # Step 6: Apply direction filter
        logger.info("-" * 40)
        logger.info(f"Step 6: Applying direction filter (direction={direction})")
        positions = apply_direction_filter(result, direction)
        logger.info("Step 6 SUCCESS: Direction filter applied")

        # Step 7: Count position changes for feedback
        logger.info("-" * 40)
        logger.info("Step 7: Counting position changes")
        total_position_changes = 0
        for slot_name, pos in positions.items():
            if isinstance(pos, pd.Series):
                changes = (pos != pos.shift(1)).sum()
                total_position_changes += int(changes)
                logger.info(f"  Slot '{slot_name}': {int(changes)} position changes")

        logger.info(f"Step 7 RESULT: Total position changes = {total_position_changes}")

        # Step 8: Convert positions to entry/exit signals
        logger.info("-" * 40)
        logger.info("Step 8: Converting positions to entry/exit signals")
        signals = positions_to_signals(positions)
        logger.info("Step 8 SUCCESS: Signals generated")

        # Step 9: Run the backtest with vectorbt
        logger.info("-" * 40)
        logger.info("Step 9: Running vectorbt backtest")
        try:
            # Extract built-in params for execution
            builtin_params = self._extract_builtin_params(params or {})
            exec_price = builtin_params["execution_price"]
            stop_loss = builtin_params["stop_loss"]
            take_profit = builtin_params["take_profit"]
            trailing_stop = builtin_params["trailing_stop"]
            slippage = builtin_params["slippage"]

            logger.info(f"  Execution price: {exec_price}")
            logger.info(f"  Stop loss: {stop_loss}")
            logger.info(f"  Take profit: {take_profit}")
            logger.info(f"  Trailing stop: {trailing_stop}")
            logger.info(f"  Slippage: {slippage}")

            # Get signal counts
            long_entry_count = signals['long_entries'].sum().sum() if not signals['long_entries'].empty else 0
            long_exit_count = signals['long_exits'].sum().sum() if not signals['long_exits'].empty else 0
            short_entry_count = signals['short_entries'].sum().sum() if not signals['short_entries'].empty else 0
            short_exit_count = signals['short_exits'].sum().sum() if not signals['short_exits'].empty else 0

            logger.info(f"  Long entries: {long_entry_count}")
            logger.info(f"  Long exits: {long_exit_count}")
            logger.info(f"  Short entries: {short_entry_count}")
            logger.info(f"  Short exits: {short_exit_count}")

            # Build common kwargs for from_signals
            common_kwargs = {
                "init_cash": self.init_cash,
                "fees": self.fees,
                "freq": "1D",
            }

            # Add stop loss if specified
            if stop_loss is not None:
                common_kwargs["sl_stop"] = stop_loss
                if trailing_stop:
                    common_kwargs["sl_trail"] = True

            # Add take profit if specified
            if take_profit is not None:
                common_kwargs["tp_stop"] = take_profit

            # Add slippage if specified (vectorbt uses slippage as fraction)
            if slippage and slippage > 0:
                common_kwargs["slippage"] = slippage

            # Determine if this is a multi-asset strategy
            slots = list(positions.keys())
            is_multi_asset = len(slots) > 1 or (data is not None and len(data) > 1)

            if is_multi_asset:
                # Multi-asset strategy: build combined price and signal DataFrames
                # Each column represents a different asset/slot
                print(f"[Backtest] Multi-asset strategy with {len(slots)} slots: {slots}")

                # Build combined price DataFrame (each column = asset)
                price_cols = pd.DataFrame(index=signals['long_entries'].index)
                for slot in slots:
                    if slot in strategy_data and exec_price in strategy_data[slot].columns:
                        price_cols[slot] = strategy_data[slot][exec_price]
                    elif slot in strategy_data:
                        price_cols[slot] = strategy_data[slot]['close']
                    else:
                        # Fallback to prices DataFrame for backward compatibility
                        price_cols[slot] = prices[exec_price] if exec_price in prices.columns else prices['close']

                # Use signals DataFrames directly (already boolean from positions_to_signals)
                long_entries = signals['long_entries']
                long_exits = signals['long_exits']

                if direction == "both" or direction == "shortonly":
                    short_entries = signals['short_entries']
                    short_exits = signals['short_exits']

                    pf = vbt.Portfolio.from_signals(
                        price_cols,
                        entries=long_entries,
                        exits=long_exits,
                        short_entries=short_entries,
                        short_exits=short_exits,
                        **common_kwargs,
                    )
                else:
                    pf = vbt.Portfolio.from_signals(
                        price_cols,
                        entries=long_entries,
                        exits=long_exits,
                        **common_kwargs,
                    )
            else:
                # Single-asset strategy
                first_slot = slots[0]

                # Select price column based on execution_price param
                if data is not None and first_slot in strategy_data:
                    asset_prices = strategy_data[first_slot]
                    price_col = asset_prices[exec_price] if exec_price in asset_prices.columns else asset_prices['close']
                else:
                    price_col = prices[exec_price] if exec_price in prices.columns else prices['close']

                # Signals are already boolean from positions_to_signals
                long_entries = signals['long_entries'][first_slot]
                long_exits = signals['long_exits'][first_slot]

                if direction == "both" or direction == "shortonly":
                    short_entries = signals['short_entries'][first_slot]
                    short_exits = signals['short_exits'][first_slot]

                    pf = vbt.Portfolio.from_signals(
                        price_col,
                        entries=long_entries,
                        exits=long_exits,
                        short_entries=short_entries,
                        short_exits=short_exits,
                        **common_kwargs,
                    )
                else:
                    pf = vbt.Portfolio.from_signals(
                        price_col,
                        entries=long_entries,
                        exits=long_exits,
                        **common_kwargs,
                    )
            logger.info("Step 9 SUCCESS: vectorbt backtest completed")
        except Exception as e:
            logger.error(f"Step 9 FAILED: {type(e).__name__}: {e}")
            base_result.error = f"Error running vectorbt backtest: {str(e)}"
            base_result.error_type = type(e).__name__
            base_result.error_traceback = traceback.format_exc()
            return base_result

        # Step 10: Extract results
        logger.info("-" * 40)
        logger.info("Step 10: Extracting backtest results")
        try:
            equity = pf.value()

            # Handle multi-asset portfolios: equity is a DataFrame, sum across columns for total
            if isinstance(equity, pd.DataFrame):
                total_equity = equity.sum(axis=1)
            else:
                total_equity = equity

            timestamps = []
            for idx in total_equity.index:
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
                    # Handle multi-asset returns (may be Series)
                    if isinstance(val, pd.Series):
                        val = val.sum() if len(val) > 1 else val.iloc[0]
                    if pd.isna(val) or np.isinf(val):
                        return default
                    return float(val)
                except Exception:
                    return default

            trades = pf.trades
            num_trades = len(trades.records) if hasattr(trades, 'records') else 0

            base_result.success = True
            base_result.timestamps = timestamps
            base_result.equity_curve = [float(v) for v in total_equity.values]
            base_result.metrics = BacktestMetrics(
                total_return=safe_metric(pf.total_return),
                sharpe_ratio=safe_metric(pf.sharpe_ratio),
                sortino_ratio=safe_metric(pf.sortino_ratio),
                max_drawdown=safe_metric(pf.max_drawdown),
                win_rate=safe_metric(trades.win_rate) if num_trades > 0 else 0.0,
                num_trades=num_trades,
                profit_factor=safe_metric(trades.profit_factor) if num_trades > 0 else 0.0,
            )
            # Helper to safely convert counts that might be arrays or Series
            def safe_int(val):
                if isinstance(val, (pd.Series, np.ndarray)):
                    return int(val.sum())
                return int(val)

            base_result.position_changes = total_position_changes
            base_result.long_entries = safe_int(long_entry_count)
            base_result.long_exits = safe_int(long_exit_count)
            base_result.short_entries = safe_int(short_entry_count)
            base_result.short_exits = safe_int(short_exit_count)

            # Log final results
            logger.info("=" * 60)
            logger.info("BACKTEST COMPLETE - RESULTS")
            logger.info("=" * 60)
            logger.info(f"  Total Return: {base_result.metrics.total_return:.2%}")
            logger.info(f"  Sharpe Ratio: {base_result.metrics.sharpe_ratio:.2f}")
            logger.info(f"  Sortino Ratio: {base_result.metrics.sortino_ratio:.2f}")
            logger.info(f"  Max Drawdown: {base_result.metrics.max_drawdown:.2%}")
            logger.info(f"  Win Rate: {base_result.metrics.win_rate:.1%}")
            logger.info(f"  Num Trades: {base_result.metrics.num_trades}")
            logger.info(f"  Profit Factor: {base_result.metrics.profit_factor:.2f}")
            logger.info(f"  Position Changes: {base_result.position_changes}")
            logger.info(f"  Equity Curve Length: {len(base_result.equity_curve)}")
            if base_result.equity_curve:
                logger.info(f"  Initial Equity: ${base_result.equity_curve[0]:,.2f}")
                logger.info(f"  Final Equity: ${base_result.equity_curve[-1]:,.2f}")
            logger.info("=" * 60)

            return base_result

        except Exception as e:
            logger.error(f"Step 10 FAILED: {type(e).__name__}: {e}")
            base_result.error = f"Error extracting backtest results: {str(e)}"
            base_result.error_type = type(e).__name__
            base_result.error_traceback = traceback.format_exc()
            return base_result


    def execute_generated_strategy(
        self,
        strategy: "GeneratedStrategy",
        data_dict: Dict[str, pd.DataFrame],
        direction: str = "longonly",
    ) -> "BacktestRunResult":
        """Execute a GeneratedStrategy and return a BacktestRunResult.

        This is the new v3 interface that accepts the structured data types
        from the multi-strategy architecture.

        Args:
            strategy: GeneratedStrategy with code, schemas, and spec.
            data_dict: Dict mapping slot names to price DataFrames.
            direction: Specific direction for this run.

        Returns:
            BacktestRunResult with structured results.
        """
        # Import here to avoid circular import
        from .models import (
            BacktestMeta,
            BacktestRunResult,
            ExecutionParams,
            Metrics,
            SymbolResult,
            Trade,
        )

        spec = strategy.spec
        logger.info("=" * 60)
        logger.info("EXECUTING GENERATED STRATEGY")
        logger.info("=" * 60)
        logger.info(f"Strategy: {spec.name}")
        logger.info(f"Direction: {direction}")
        logger.info(f"Symbols: {spec.symbols}")
        logger.info(f"Data slots: {list(data_dict.keys())}")

        # Build execution params
        exec_params = ExecutionParams(
            direction=direction,
            execution_price=spec.execution_price,
            slippage=spec.slippage,
            stop_loss=spec.stop_loss,
            take_profit=spec.take_profit,
            trailing_stop=False,
            init_cash=spec.init_cash,
            fees=self.fees,
        )

        # Get first DataFrame to determine timeframe and dates
        first_slot = list(data_dict.keys())[0]
        first_df = data_dict[first_slot]

        # Build result structure
        result = BacktestRunResult(
            strategy_name=spec.name,
            direction=direction,
            execution=exec_params,
            meta=BacktestMeta(
                timeframe="1Day",  # Will be populated from data
                start_date=str(first_df.index[0]),
                end_date=str(first_df.index[-1]),
                total_bars=len(first_df),
            ),
        )

        # Prepare params for execution
        params = strategy.params.copy()
        params["execution_price"] = spec.execution_price
        params["slippage"] = spec.slippage
        params["stop_loss"] = spec.stop_loss
        params["take_profit"] = spec.take_profit

        # Execute using existing method
        backtest_result = self.execute(
            code=strategy.code,
            prices=first_df,  # Fallback for single-asset
            symbols=spec.symbols,
            timeframe="1Day",
            params=params,
            direction=direction,
            data=data_dict,
        )

        if not backtest_result.success:
            result.success = False
            result.error = backtest_result.error
            return result

        # Convert BacktestResult to BacktestRunResult
        # For now, create a single SymbolResult with combined data
        # In future, we can expand to per-symbol results
        combined_key = "_".join(spec.symbols) if len(spec.symbols) > 1 else spec.symbols[0]

        # Extract trades from vectorbt portfolio
        trades_list: List[Trade] = []
        # TODO: Extract individual trade records from vectorbt

        result.results_by_symbol[combined_key] = SymbolResult(
            symbol=combined_key,
            timestamps=backtest_result.timestamps,
            equity_curve=backtest_result.equity_curve,
            positions=[],  # TODO: Extract position series
            metrics=Metrics(
                total_return=backtest_result.metrics.total_return,
                sharpe_ratio=backtest_result.metrics.sharpe_ratio,
                sortino_ratio=backtest_result.metrics.sortino_ratio,
                max_drawdown=backtest_result.metrics.max_drawdown,
                win_rate=backtest_result.metrics.win_rate,
                num_trades=backtest_result.metrics.num_trades,
                profit_factor=backtest_result.metrics.profit_factor,
                # Enhanced metrics - calculate from equity curve
                cagr=self._calculate_cagr(backtest_result.equity_curve),
                volatility=self._calculate_volatility(backtest_result.equity_curve),
                calmar_ratio=self._calculate_calmar(
                    backtest_result.equity_curve,
                    backtest_result.metrics.max_drawdown,
                ),
            ),
            trades=trades_list,
        )

        return result

    def _calculate_cagr(self, equity_curve: List[float]) -> float:
        """Calculate Compound Annual Growth Rate."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        try:
            start_val = equity_curve[0]
            end_val = equity_curve[-1]
            if start_val <= 0:
                return 0.0
            n_years = len(equity_curve) / 252  # Assume 252 trading days
            if n_years <= 0:
                return 0.0
            return (end_val / start_val) ** (1 / n_years) - 1
        except Exception:
            return 0.0

    def _calculate_volatility(self, equity_curve: List[float]) -> float:
        """Calculate annualized volatility of returns."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        try:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            return float(np.std(returns) * np.sqrt(252))
        except Exception:
            return 0.0

    def _calculate_calmar(self, equity_curve: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        cagr = self._calculate_cagr(equity_curve)
        if abs(max_drawdown) < 0.0001:
            return 0.0
        return cagr / abs(max_drawdown)


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
