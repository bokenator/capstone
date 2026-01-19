import traceback
from typing import Any, Callable, Optional

import pandas as pd

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

from backtests.shared import BacktestResult, load_sample_data, extract_metrics_from_portfolio
from prompts.shared import PARAM_SCHEMA_SIMPLE


def get_default_params() -> dict[str, Any]:
    """Get default parameters from PARAM_SCHEMA."""
    return {
        name: spec["default"]
        for name, spec in PARAM_SCHEMA_SIMPLE.items()
    }


def run_backtest(
    generate_signals: Callable[[dict, dict], dict[str, pd.Series]],
    data: Optional[dict[str, pd.DataFrame]] = None,
    params: Optional[dict[str, Any]] = None,
    initial_capital: float = 100_000.0,
    fees: float = 0.001,  # 0.1% per trade
) -> BacktestResult:
    """
    Run backtest for a simple strategy.

    Args:
        generate_signals: The strategy's signal generation function
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
        data = load_sample_data("simple")

    # Use default params if not provided
    if params is None:
        params = get_default_params()

    # Generate signals
    try:
        signals = generate_signals(data, params)
    except Exception as e:
        result.error = f"generate_signals failed: {e}\n{traceback.format_exc()}"
        return result

    # Extract position series
    if "ohlcv" not in signals:
        result.error = "generate_signals did not return 'ohlcv' key"
        return result

    position = signals["ohlcv"]
    close = data["ohlcv"]["close"]

    # Convert position targets to entries/exits for vectorbt
    # Position: +1 = long, 0 = flat, -1 = short
    # Entry: position goes from 0 to +1
    # Exit: position goes from +1 to 0
    position_diff = position.diff().fillna(0)
    entries = position_diff > 0
    exits = position_diff < 0

    # Run backtest with vectorbt
    try:
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=fees,
            freq="D",
        )
    except Exception as e:
        result.error = f"Portfolio.from_signals failed: {e}\n{traceback.format_exc()}"
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
        result.positions = position

        # Get trade log if available
        try:
            result.trades = pf.trades.records_readable
        except Exception:
            pass

        # Metadata
        result.start_date = close.index[0].to_pydatetime() if hasattr(close.index[0], 'to_pydatetime') else None
        result.end_date = close.index[-1].to_pydatetime() if hasattr(close.index[-1], 'to_pydatetime') else None
        result.num_bars = len(close)

        # Store params used
        result.details["params"] = params
        result.details["initial_capital"] = initial_capital
        result.details["fees"] = fees

    except Exception as e:
        result.error = f"Failed to extract results: {e}\n{traceback.format_exc()}"
        return result

    return result
