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
from prompts.shared import PARAM_SCHEMA_COMPLEX


class _FlexOrderContext:
    """Simulated OrderContext for flexible mode that provides col attribute."""
    def __init__(self, i, col, position_now, cash_now):
        self.i = i
        self.col = col
        self.position_now = position_now
        self.cash_now = cash_now


def _wrap_flex_order_func(user_order_func: Callable, close_a: np.ndarray, close_b: np.ndarray, num_cols: int = 2) -> Callable:
    """
    Wrap user's order function for flexible multi-asset mode.

    In flexible mode, vectorbt calls the function repeatedly for each bar until
    we return NoOrder. We track which columns have already been processed for
    the current bar and return orders one at a time, then NoOrder when done.

    User function signature: order_func(c, close_a, close_b, ...) -> (size, size_type, direction)
    where c.col indicates which asset (0 or 1)
    """
    # Track state: which bar we're on and which columns we've processed
    state = {"last_bar": -1, "pending_orders": []}

    def wrapped_flex_order_func(c, close_a, close_b, zscore, hedge_ratio,
                                entry_threshold, exit_threshold, stop_threshold, notional_per_leg):
        i = c.i

        # If we're on a new bar, compute orders for all columns
        if i != state["last_bar"]:
            state["last_bar"] = i
            state["pending_orders"] = []

            # FlexOrderContext has different attributes than regular OrderContext
            cash = getattr(c, 'cash_now', getattr(c, 'value_now', 100000.0))

            # Get positions for both assets from last_position array
            last_pos = getattr(c, 'last_position', np.zeros(num_cols))
            pos_a = last_pos[0] if len(last_pos) > 0 else 0.0
            pos_b = last_pos[1] if len(last_pos) > 1 else 0.0

            # Call user function for each asset
            for col in range(num_cols):
                pos = pos_a if col == 0 else pos_b
                sim_ctx = _FlexOrderContext(i=i, col=col, position_now=pos, cash_now=cash)

                result = user_order_func(
                    sim_ctx, close_a, close_b, zscore, hedge_ratio,
                    entry_threshold, exit_threshold, stop_threshold, notional_per_leg
                )

                # Convert tuple to Order
                if isinstance(result, tuple) and len(result) >= 3:
                    size, size_type, direction = result[0], result[1], result[2]

                    if not np.isnan(size):
                        price = close_a[i] if col == 0 else close_b[i]
                        order = vbt.portfolio.nb.order_nb(
                            size=float(size),
                            price=float(price),
                            size_type=int(size_type),
                            direction=int(direction),
                        )
                        state["pending_orders"].append((col, order))

        # Return next pending order, or NoOrder if none left
        if state["pending_orders"]:
            return state["pending_orders"].pop(0)

        # Return NoOrder - this tells vectorbt to advance to next bar
        return (-1, vbt.portfolio.nb.order_nb(size=np.nan, price=np.nan, size_type=0, direction=0))

    return wrapped_flex_order_func


def get_default_params() -> dict[str, Any]:
    """Get default parameters from PARAM_SCHEMA."""
    return {
        name: spec["default"]
        for name, spec in PARAM_SCHEMA_COMPLEX.items()
    }


def run_backtest(
    compute_spread_indicators: Callable[..., dict[str, np.ndarray]],
    order_func: Callable,
    data: Optional[dict[str, pd.DataFrame]] = None,
    params: Optional[dict[str, Any]] = None,
    initial_capital: float = 100_000.0,
    fees: float = 0.001,  # 0.1% per trade
) -> BacktestResult:
    """
    Run backtest for a complex (pairs trading) strategy.

    Args:
        compute_spread_indicators: Function to compute spread indicators
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
        data = load_sample_data("complex")

    # Use default params if not provided
    if params is None:
        params = get_default_params()

    # Extract close prices
    close_a = data["asset_a"]["close"].values
    close_b = data["asset_b"]["close"].values

    # Extract indicator computation params
    indicator_params = {
        k: v for k, v in params.items()
        if k in ["hedge_lookback", "zscore_lookback"]
    }

    # Compute spread indicators
    try:
        indicators = compute_spread_indicators(close_a, close_b, **indicator_params)
    except Exception as e:
        result.error = f"compute_spread_indicators failed: {e}\n{traceback.format_exc()}"
        return result

    # Extract arrays for order function
    try:
        zscore = indicators["zscore"]
        hedge_ratio = indicators["hedge_ratio"]
    except KeyError as e:
        result.error = f"compute_spread_indicators missing key: {e}"
        return result

    # Get trading thresholds and sizing
    entry_threshold = params.get("entry_threshold", 2.0)
    exit_threshold = params.get("exit_threshold", 0.0)
    stop_threshold = params.get("stop_threshold", 3.0)
    notional_per_leg = params.get("notional_per_leg", 10000.0)

    # Create combined close DataFrame for vectorbt
    close_df = pd.DataFrame({
        "asset_a": data["asset_a"]["close"],
        "asset_b": data["asset_b"]["close"],
    })

    # Wrap order function for flexible multi-asset mode
    wrapped_order_func = _wrap_flex_order_func(order_func, close_a, close_b)

    # Run backtest with from_order_func (flexible=True for multi-asset, numba disabled)
    try:
        pf = vbt.Portfolio.from_order_func(
            close_df,
            wrapped_order_func,
            close_a, close_b, zscore, hedge_ratio, entry_threshold, exit_threshold, stop_threshold, notional_per_leg,
            init_cash=initial_capital,
            freq="D",
            flexible=True,
            group_by=True,  # Treat as single group for cash sharing
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
        result.start_date = close_df.index[0].to_pydatetime() if hasattr(close_df.index[0], 'to_pydatetime') else None
        result.end_date = close_df.index[-1].to_pydatetime() if hasattr(close_df.index[-1], 'to_pydatetime') else None
        result.num_bars = len(close_df)

        # Store params used
        result.details["params"] = params
        result.details["initial_capital"] = initial_capital
        result.details["fees"] = fees

    except Exception as e:
        result.error = f"Failed to extract results: {e}\n{traceback.format_exc()}"
        return result

    return result
