"""
MACD + ATR Trailing Stop Strategy for vectorbt

Exports:
- compute_indicators
- order_func

Notes:
- No numba, no vbt.portfolio.nb or vbt.portfolio.enums usage here.
- order_func returns simple Python tuples (size, size_type, direction).
  We use SizeType.Amount == 0 and directions Long == 1, Short == 2.

Strategy:
- MACD(12,26,9) crossover for entries/exits
- Trend filter: 50-period SMA (price must be above SMA to enter)
- ATR(14) used for trailing stop at highest_since_entry - trailing_mult * ATR

"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Strategy state maintained across order_func calls (single-asset)
# Using module-level state to persist between calls made by Portfolio.from_order_func
_STRATEGY_STATE: Dict[str, Any] = {
    "is_long": False,
    "entry_index": None,
    "entry_price": np.nan,
    "highest_price": -np.inf,
}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute technical indicators required by the strategy.

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close'] (volume optional)
        macd_fast: Fast EMA period for MACD
        macd_slow: Slow EMA period for MACD
        macd_signal: Signal EMA period for MACD
        sma_period: Period for trend SMA
        atr_period: Period for ATR

    Returns:
        Dict with numpy arrays for keys: 'macd', 'signal', 'atr', 'sma', 'close', 'high'

    Notes:
        - Uses pandas ewm for MACD and signal lines.
        - ATR uses True Range and a simple rolling mean so initial atr_period bars are NaN.
        - SMA is computed with min_periods=sma_period so it will be NaN until enough bars.
    """
    # Validate input
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise ValueError(f"ohlcv must contain columns: {required_cols}")

    # Ensure float dtype
    high = ohlcv["high"].astype(float).copy()
    low = ohlcv["low"].astype(float).copy()
    close = ohlcv["close"].astype(float).copy()

    # MACD: EMA(fast) - EMA(slow)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter) - require full period to avoid premature signals
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR computation (True Range and rolling mean)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    return {
        "macd": macd_line.to_numpy(dtype=float),
        "signal": signal_line.to_numpy(dtype=float),
        "atr": atr.to_numpy(dtype=float),
        "sma": sma.to_numpy(dtype=float),
        "close": close.to_numpy(dtype=float),
        "high": high.to_numpy(dtype=float),
    }


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function to be used with vectorbt.Portfolio.from_order_func

    Returns a tuple (size, size_type, direction).
    - size: float (use np.nan to indicate no order)
    - size_type: int (we use 0 -> Amount)
    - direction: int (1 -> Long, 2 -> Short)

    Strategy logic (long-only):
    - Entry: MACD crosses above Signal AND close > SMA
    - Exit: MACD crosses below Signal OR close < highest_since_entry - trailing_mult * ATR

    Notes:
    - Maintains state in module-level _STRATEGY_STATE dict for tracking open position and highest price since entry.
    - Avoids placing orders when indicators are NaN or during warmup.
    """
    # Access index from context
    # c.i is provided by vectorbt and indicates the current bar index
    i = int(getattr(c, "i", 0))

    # Safety bounds check
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Current values
    close_i = float(close[i]) if not np.isnan(close[i]) else np.nan
    high_i = float(high[i]) if not np.isnan(high[i]) else np.nan
    macd_i = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    signal_i = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    atr_i = float(atr[i]) if not np.isnan(atr[i]) else np.nan
    sma_i = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # If any primary indicator is NaN, do nothing
    if np.isnan(macd_i) or np.isnan(signal_i) or np.isnan(sma_i):
        return (np.nan, 0, 0)

    # Helper for previous values (for cross detection)
    prev_ok = i - 1 >= 0
    prev_macd = float(macd[i - 1]) if prev_ok and not np.isnan(macd[i - 1]) else np.nan
    prev_signal = float(signal[i - 1]) if prev_ok and not np.isnan(signal[i - 1]) else np.nan

    # Local references to state
    global _STRATEGY_STATE

    is_long = bool(_STRATEGY_STATE.get("is_long", False))

    # ENTRY: MACD cross up + price above SMA
    entered_now = False
    if not is_long:
        cross_up = False
        if prev_ok and not np.isnan(prev_macd) and not np.isnan(prev_signal):
            cross_up = (prev_macd <= prev_signal) and (macd_i > signal_i)

        price_above_sma = (not np.isnan(close_i)) and (not np.isnan(sma_i)) and (close_i > sma_i)

        if cross_up and price_above_sma:
            # Place a buy order: 1 unit (Amount), Long
            size = 1.0
            size_type = 0  # Amount
            direction = 1  # Long

            # Update state optimistically (pre-fill)
            _STRATEGY_STATE["is_long"] = True
            _STRATEGY_STATE["entry_index"] = i
            _STRATEGY_STATE["entry_price"] = close_i
            _STRATEGY_STATE["highest_price"] = high_i if not np.isnan(high_i) else close_i

            entered_now = True
            return (size, size_type, direction)

    # If we're long (either from earlier or just entered), check exit conditions
    is_long = bool(_STRATEGY_STATE.get("is_long", False))

    if is_long and not entered_now:
        # Update highest price since entry
        hp = _STRATEGY_STATE.get("highest_price", -np.inf)
        if not np.isnan(high_i):
            if hp is None or np.isnan(hp) or high_i > hp:
                _STRATEGY_STATE["highest_price"] = high_i
                hp = high_i

        # Exit condition 1: MACD cross down
        cross_down = False
        if prev_ok and not np.isnan(prev_macd) and not np.isnan(prev_signal):
            cross_down = (prev_macd >= prev_signal) and (macd_i < signal_i)

        # Exit condition 2: Trailing stop based on highest price since entry
        trail_trigger = False
        if not np.isnan(hp) and hp != -np.inf and not np.isnan(atr_i):
            threshold = hp - float(trailing_mult) * atr_i
            if not np.isnan(close_i) and close_i < threshold:
                trail_trigger = True

        if cross_down or trail_trigger:
            # Place a sell order to close the long: 1 unit (Amount), Short direction to offset
            size = 1.0
            size_type = 0  # Amount
            direction = 2  # Short

            # Reset state optimistically
            _STRATEGY_STATE["is_long"] = False
            _STRATEGY_STATE["entry_index"] = None
            _STRATEGY_STATE["entry_price"] = np.nan
            _STRATEGY_STATE["highest_price"] = -np.inf

            return (size, size_type, direction)

    # Default: no action
    return (np.nan, 0, 0)
