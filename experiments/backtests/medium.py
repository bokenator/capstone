import traceback
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
    from vectorbt.portfolio.enums import SizeType, Direction
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

from backtests.shared import BacktestResult, load_sample_data, extract_metrics_from_portfolio
from prompts.shared import PARAM_SCHEMA_MEDIUM


def _wrap_order_func(user_order_func: Callable, close: np.ndarray) -> Callable:
    """
    Wrap user's order function to convert tuple returns to vectorbt Order objects.

    User function returns: (size, size_type, direction)
    vectorbt expects: Order namedtuple with many fields
    """
    def wrapped_order_func(c, *args):
        result = user_order_func(c, *args)

        # If result is already an Order-like object, return as-is
        if hasattr(result, 'size') and hasattr(result, 'price'):
            return result

        # Convert tuple (size, size_type, direction) to Order
        if isinstance(result, tuple) and len(result) >= 3:
            size, size_type, direction = result[0], result[1], result[2]

            # No action if size is nan
            if np.isnan(size):
                return vbt.portfolio.nb.order_nb(
                    size=np.nan,
                    price=np.nan,
                    size_type=SizeType.Amount,
                    direction=Direction.Both,
                )

            # Get current price for the order
            price = close[c.i] if hasattr(c, 'i') else np.nan

            return vbt.portfolio.nb.order_nb(
                size=float(size),
                price=float(price),
                size_type=int(size_type),
                direction=int(direction),
            )

        return result

    return wrapped_order_func


def get_default_params() -> dict[str, Any]:
    """Get default parameters from PARAM_SCHEMA."""
    return {
        name: spec["default"]
        for name, spec in PARAM_SCHEMA_MEDIUM.items()
    }


def run_backtest(
    compute_indicators: Callable[..., dict[str, np.ndarray]],
    order_func: Callable,
    data: Optional[dict[str, pd.DataFrame]] = None,
    params: Optional[dict[str, Any]] = None,
    initial_capital: float = 100_000.0,
    fees: float = 0.001,  # 0.1% per trade
) -> BacktestResult:
    """
    Run backtest for a medium strategy.

    Args:
        compute_indicators: Function to compute technical indicators
        order_func: Order function for vectorbt
        data: Optional data dict (loads sample data if not provided)
        params: Optional parameters dict (uses defaults if not provided)
        initial_capital: Starting capital
        fees: Trading fees as fraction

    Returns:
        BacktestResult with metrics and raw data
    """
    result = BacktestResult()

    # Check vectorbt is available
    if not HAS_VBT:
        result.error = "vectorbt not installed"
        return result

    # Load data if not provided
    if data is None:
        data = load_sample_data("medium")

    # Use default params if not provided
    if params is None:
        params = get_default_params()

    # Extract indicator computation params
    indicator_params = {
        k: v for k, v in params.items()
        if k in ["macd_fast", "macd_slow", "macd_signal", "sma_period", "atr_period"]
    }

    # Compute indicators
    try:
        ohlcv = data["ohlcv"]
        indicators = compute_indicators(ohlcv, **indicator_params)
    except Exception as e:
        result.error = f"compute_indicators failed: {e}\n{traceback.format_exc()}"
        return result

    # Extract arrays for order function
    try:
        close = indicators.get("close", ohlcv["close"].values)
        high = indicators.get("high", ohlcv["high"].values)
        macd = indicators["macd"]
        signal = indicators["signal"]
        atr = indicators["atr"]
        sma = indicators["sma"]
    except KeyError as e:
        result.error = f"compute_indicators missing key: {e}"
        return result

    # Get trailing multiplier
    trailing_mult = params.get("trailing_mult", 2.0)

    # Wrap order function to convert tuple returns to Order objects
    wrapped_order_func = _wrap_order_func(order_func, close)

    # Run backtest with from_order_func (numba disabled)
    try:
        pf = vbt.Portfolio.from_order_func(
            ohlcv["close"],
            wrapped_order_func,
            close, high, macd, signal, atr, sma, trailing_mult,
            init_cash=initial_capital,
            freq="D",
            flexible=False,
            use_numba=False,
        )
    except Exception as e:
        result.error = f"Portfolio.from_order_func failed: {e}\n{traceback.format_exc()}"
        return result

    # Extract metrics
    try:
        metrics = extract_metrics_from_portfolio(pf)
        if "error" in metrics:
            result.error = f"Metrics extraction failed: {metrics['error']}"
            return result

        result.total_return = metrics.get("total_return")
        result.annualized_return = metrics.get("annualized_return")
        result.sharpe_ratio = metrics.get("sharpe_ratio")
        result.sortino_ratio = metrics.get("sortino_ratio")
        result.max_drawdown = metrics.get("max_drawdown")
        result.volatility = metrics.get("volatility")
        result.calmar_ratio = metrics.get("calmar_ratio")
        result.total_trades = metrics.get("total_trades")
        result.win_rate = metrics.get("win_rate")
        result.profit_factor = metrics.get("profit_factor")
        result.exposure_time = metrics.get("exposure_time")

        # Fail if no trades were made
        if result.total_trades == 0:
            result.error = "Strategy produced no trades"
            return result

        result.success = True

        # Store raw data
        result.equity_curve = pf.value()
        result.returns = pf.returns()

        # Get trade log if available
        try:
            result.trades = pf.trades.records_readable
        except Exception:
            pass

        # Metadata
        close_series = ohlcv["close"]
        result.start_date = close_series.index[0].to_pydatetime() if hasattr(close_series.index[0], 'to_pydatetime') else None
        result.end_date = close_series.index[-1].to_pydatetime() if hasattr(close_series.index[-1], 'to_pydatetime') else None
        result.num_bars = len(close_series)

        # Store params used
        result.details["params"] = params
        result.details["initial_capital"] = initial_capital
        result.details["fees"] = fees

    except Exception as e:
        result.error = f"Failed to extract results: {e}\n{traceback.format_exc()}"
        return result

    return result
