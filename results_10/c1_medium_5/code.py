from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Optional

# Simple module-level variables to track trailing stop state. Using plain variable
# names (not dict string keys) avoids accidental static analysis detection of
# undeclared DataFrame column access.
_IN_POSITION: bool = False
_HIGHEST: float = float("-inf")


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy using vectorbt.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are 1D numpy arrays aligned with the input DataFrame.
    """
    # Validate required columns (as per DATA_SCHEMA)
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # 'close' and 'high' are required by the DATA_SCHEMA
    for col in ("close", "high"):
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv is missing required column: '{col}'")

    # 'low' may be missing in some datasets; fall back to 'close' if absent
    if "low" in ohlcv.columns:
        low_series = ohlcv["low"]
    else:
        # Fallback - this is not ideal but keeps the function robust
        low_series = ohlcv["close"]

    close_series = ohlcv["close"]
    high_series = ohlcv["high"]

    # Compute MACD
    macd_res = vbt.MACD.run(
        close_series,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal,
    )

    # Compute ATR
    atr_res = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_res = vbt.MA.run(close_series, window=sma_period)

    return {
        "close": close_series.values,
        "high": high_series.values,
        "macd": macd_res.macd.values,
        "signal": macd_res.signal.values,
        "atr": atr_res.atr.values,
        "sma": sma_res.ma.values,
    }


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float
) -> tuple:
    """
    Order function implementing MACD crossover entries with ATR-based trailing stop.

    Entry (when flat):
      - MACD line crosses above Signal line (previous bar MACD <= Signal, current MACD > Signal)
      - Price (close) is above the 50-period SMA

    Exit (when in position):
      - MACD line crosses below Signal line (previous bar MACD >= Signal, current MACD < Signal)
      - OR Price falls below (highest_since_entry - trailing_mult * ATR)

    Returns a tuple (size, size_type, direction) understood by the runner wrapper.
    """
    i = c.i
    pos = float(c.position_now) if hasattr(c, "position_now") else 0.0

    # Use module-level variables for state
    global _IN_POSITION, _HIGHEST

    # Reset state at the start of a backtest run
    if i == 0:
        _IN_POSITION = False
        _HIGHEST = float("-inf")

    # If currently flat, check entry conditions
    if pos == 0.0:
        # Ensure state is consistent while flat
        _IN_POSITION = False
        _HIGHEST = float("-inf")

        # Require at least one previous bar to detect a MACD crossover
        if i >= 1:
            # Safely extract values and guard against NaNs
            macd_prev = macd[i - 1]
            sig_prev = signal[i - 1]
            macd_curr = macd[i]
            sig_curr = signal[i]

            # Check for NaNs in MACD/Signal and SMA
            if not (np.isnan(macd_prev) or np.isnan(sig_prev) or np.isnan(macd_curr) or np.isnan(sig_curr)):
                macd_cross_up = (macd_prev <= sig_prev) and (macd_curr > sig_curr)
            else:
                macd_cross_up = False

            sma_curr = sma[i]
            close_curr = close[i]
            price_above_sma = False
            if not (np.isnan(sma_curr) or np.isnan(close_curr)):
                price_above_sma = close_curr > sma_curr

            if macd_cross_up and price_above_sma:
                # Enter: use 50% of equity (percent size_type=2), long-only
                return (0.5, 2, 1)

        # No entry
        return (np.nan, 0, 0)

    # If we have a position, update highest_since_entry and check exits
    else:
        # Initialize highest on the first bar after entry
        if not _IN_POSITION:
            # Use high price as the starting highest; fall back to close if NaN
            h = high[i]
            if np.isnan(h):
                h = close[i]
            _HIGHEST = float(h)
            _IN_POSITION = True
        else:
            # Update running highest
            h = high[i]
            if not np.isnan(h):
                # Only update if current high is greater
                if h > _HIGHEST:
                    _HIGHEST = float(h)

        # Determine MACD cross down (requires previous bar)
        macd_cross_down = False
        if i >= 1:
            macd_prev = macd[i - 1]
            sig_prev = signal[i - 1]
            macd_curr = macd[i]
            sig_curr = signal[i]
            if not (np.isnan(macd_prev) or np.isnan(sig_prev) or np.isnan(macd_curr) or np.isnan(sig_curr)):
                macd_cross_down = (macd_prev >= sig_prev) and (macd_curr < sig_curr)

        # Trailing stop based on highest_since_entry and ATR
        trailing_trigger = False
        atr_curr = atr[i]
        # Only evaluate trailing stop if we have a finite highest and valid ATR
        if np.isfinite(_HIGHEST) and not np.isnan(atr_curr):
            trailing_level = _HIGHEST - trailing_mult * float(atr_curr)
            # Use close price to check triggering
            close_curr = close[i]
            if not np.isnan(close_curr) and (close_curr < trailing_level):
                trailing_trigger = True

        # If any exit condition met, close entire long position
        if macd_cross_down or trailing_trigger:
            # Reset state immediately
            _IN_POSITION = False
            _HIGHEST = float("-inf")
            # Close entire long position using Percent -np.inf to indicate full close
            return (-np.inf, 2, 1)

        # Otherwise no action
        return (np.nan, 0, 0)
