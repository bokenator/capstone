import numpy as np
import pandas as pd
import vectorbt as vbt
import vectorbt.portfolio.enums as enums
from typing import Any, Dict, Optional

# Module-level state variables to track entry/highest price across calls within a run.
_IN_POSITION: bool = False
_ENTRY_IDX: int = -1
_HIGHEST: float = np.nan
_LAST_RUN_LENGTH: Optional[int] = None


def _reset_state(length: Optional[int] = None) -> None:
    """Reset the module-level state for a new backtest run."""
    global _IN_POSITION, _ENTRY_IDX, _HIGHEST, _LAST_RUN_LENGTH
    _IN_POSITION = False
    _ENTRY_IDX = -1
    _HIGHEST = np.nan
    _LAST_RUN_LENGTH = length


def _make_order_record(idx: int, size: float, price: float = np.inf, fees: float = 0.0, side: int = 1) -> np.ndarray:
    """Create a numpy record matching vectorbt.portfolio.enums.order_dt."""
    rec = np.zeros((), dtype=enums.order_dt)
    # Fill required fields. Use -1/defaults for id/col when not applicable.
    rec['id'] = -1
    rec['col'] = 0
    rec['idx'] = int(idx)
    rec['size'] = float(size) if not np.isnan(size) else np.nan
    rec['price'] = float(price) if not np.isnan(price) else np.nan
    rec['fees'] = float(fees)
    rec['side'] = int(side)
    return rec


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> np.ndarray:
    """
    Generate an order record for vectorbt simulation.

    Returns a numpy record of dtype vectorbt.portfolio.enums.order_dt.
    """
    global _IN_POSITION, _ENTRY_IDX, _HIGHEST, _LAST_RUN_LENGTH

    i = int(c.i)

    # Detect new run and reset state
    try:
        arr_len = len(close)
    except Exception:
        arr_len = None

    if i == 0 or _LAST_RUN_LENGTH != arr_len:
        _reset_state(arr_len)

    pos = float(c.position_now) if hasattr(c, "position_now") else 0.0

    # Safety checks for index bounds
    if i < 0 or i >= (len(close) if close is not None else 0):
        # No action
        return _make_order_record(i, np.nan, np.nan)

    # Helper: safe value retrieval
    def _safe_get(arr: np.ndarray, idx: int):
        try:
            return float(arr[idx])
        except Exception:
            return np.nan

    cur_close = _safe_get(close, i)
    cur_high = _safe_get(high, i)
    cur_macd = _safe_get(macd, i)
    cur_signal = _safe_get(signal, i)
    cur_atr = _safe_get(atr, i)
    cur_sma = _safe_get(sma, i)

    # Compute MACD cross signals using previous bar
    cross_up = False
    cross_down = False
    if i > 0:
        prev_macd = _safe_get(macd, i - 1)
        prev_signal = _safe_get(signal, i - 1)
        # Only evaluate crossover if both previous and current values are finite
        if (
            not np.isnan(prev_macd)
            and not np.isnan(prev_signal)
            and not np.isnan(cur_macd)
            and not np.isnan(cur_signal)
        ):
            cross_up = (prev_macd <= prev_signal) and (cur_macd > cur_signal)
            cross_down = (prev_macd >= prev_signal) and (cur_macd < cur_signal)

    # If currently flat - consider entry
    if pos == 0.0:
        # Ensure state cleared when flat
        if _IN_POSITION:
            _reset_state(arr_len)

        # Entry conditions: MACD cross up and price above SMA
        if (
            cross_up
            and not np.isnan(cur_sma)
            and not np.isnan(cur_close)
            and cur_close > cur_sma
        ):
            # Determine number of shares to buy: use 50% of available cash
            cash_now = float(c.cash_now) if hasattr(c, "cash_now") and c.cash_now is not None else 0.0
            if cash_now <= 0 or np.isnan(cash_now) or np.isnan(cur_close) or cur_close <= 0:
                return _make_order_record(i, np.nan, np.nan)

            size_shares = (0.5 * cash_now) / cur_close
            # Record entry state
            _IN_POSITION = True
            _ENTRY_IDX = i
            _HIGHEST = cur_high if not np.isnan(cur_high) else cur_close

            # Use market order: price = inf to indicate market execution
            return _make_order_record(i, float(size_shares), price=np.inf, fees=0.0, side=1)

        # No order
        return _make_order_record(i, np.nan, np.nan)

    # If in a position - check for exits and update highest
    else:
        # Ensure we have a sensible highest value
        if np.isnan(_HIGHEST):
            _HIGHEST = cur_high if not np.isnan(cur_high) else cur_close

        # Update highest since entry using current high
        if not np.isnan(cur_high) and cur_high > _HIGHEST:
            _HIGHEST = cur_high

        # Trailing stop (price falls below highest_since_entry - trailing_mult * ATR)
        if (
            not np.isnan(_HIGHEST)
            and not np.isnan(cur_atr)
            and not np.isnan(cur_close)
        ):
            trailing_level = _HIGHEST - trailing_mult * cur_atr
            # Exit if price breaches trailing stop
            if cur_close < trailing_level:
                size_to_sell = -pos
                # Reset state - it will also be reset on next bar when pos==0
                _IN_POSITION = False
                _ENTRY_IDX = -1
                _HIGHEST = np.nan

                return _make_order_record(i, float(size_to_sell), price=np.inf, fees=0.0, side=1)

        # Exit on MACD bearish cross
        if cross_down:
            size_to_sell = -pos
            _IN_POSITION = False
            _ENTRY_IDX = -1
            _HIGHEST = np.nan

            return _make_order_record(i, float(size_to_sell), price=np.inf, fees=0.0, side=1)

        # Otherwise, no action
        return _make_order_record(i, np.nan, np.nan)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute required indicators using vectorbt utilities.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'. All values
    are numpy arrays aligned with the input DataFrame index.
    """
    # Validate input columns per DATA_SCHEMA
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv must contain '{col}' column")

    close_series = ohlcv["close"]
    high_series = ohlcv["high"]
    low_series = ohlcv["low"]

    # MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA (trend filter)
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        "close": close_series.values,
        "high": high_series.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }