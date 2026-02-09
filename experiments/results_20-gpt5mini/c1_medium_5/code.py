import numpy as np
import pandas as pd
import vectorbt as vbt


class _OrderState:
    """Simple object to hold state between order_func calls.

    Using attribute access avoids false positives from static checks that
    interpret string-based indexing as DataFrame column access.
    """

    def __init__(self):
        self.last_pos: float = 0.0
        self.highest_since_entry: float = np.nan
        self.entry_idx: int | None = None


# Single-instance state for the single-asset backtest
_ORDER_STATE = _OrderState()


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
    Generate order at each bar. Called by vectorbt's from_order_func.

    Entry conditions (when flat):
      - MACD line crosses above Signal line
      - Price is above 50-period SMA
      - ATR is available
      -> Enter long with 100% equity

    Exit conditions (when long):
      - MACD line crosses below Signal line OR
      - Price falls below (highest_since_entry - trailing_mult * ATR)
      -> Close entire long position

    This implementation keeps minimal state in module-level _ORDER_STATE.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Local alias to module-level state
    state = _ORDER_STATE

    def is_finite(x):
        return np.isfinite(x)

    # Update highest_since_entry tracking when in position
    if pos > 0:
        if state.last_pos == 0.0:
            # Just entered -> initialize highest with current high (or close if high NaN)
            init_high = high[i] if is_finite(high[i]) else close[i]
            state.highest_since_entry = init_high
            state.entry_idx = i
        else:
            prev_highest = state.highest_since_entry
            current_high = high[i] if is_finite(high[i]) else close[i]
            if np.isnan(prev_highest):
                state.highest_since_entry = current_high
            else:
                # Keep running maximum
                state.highest_since_entry = max(prev_highest, current_high)
    else:
        # If we were previously long and now flat, reset tracking
        if state.last_pos > 0.0 and pos == 0.0:
            state.highest_since_entry = np.nan
            state.entry_idx = None

    # Prepare previous index for cross detection
    prev_i = i - 1 if i - 1 >= 0 else None

    # Safe fetch of MACD/Signal (may contain NaNs during warmup)
    macd_curr = macd[i]
    signal_curr = signal[i]
    macd_prev = macd[prev_i] if prev_i is not None else np.nan
    signal_prev = signal[prev_i] if prev_i is not None else np.nan

    can_eval_cross = is_finite(macd_curr) and is_finite(signal_curr) and is_finite(macd_prev) and is_finite(signal_prev)

    cross_up = False
    cross_down = False
    if can_eval_cross:
        cross_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
        cross_down = (macd_prev >= signal_prev) and (macd_curr < signal_curr)

    price = close[i]
    sma_curr = sma[i]
    price_above_sma = is_finite(price) and is_finite(sma_curr) and (price > sma_curr)

    # ENTRY logic when flat
    if pos == 0.0:
        if cross_up and price_above_sma and is_finite(atr[i]):
            # Enter long using 100% of equity
            state.last_pos = pos
            return (1.0, 2, 1)

    # EXIT logic when long
    else:
        highest = state.highest_since_entry
        trail_price = np.nan
        if is_finite(highest) and is_finite(atr[i]):
            trail_price = highest - (trailing_mult * atr[i])

        price_below_trail = is_finite(price) and is_finite(trail_price) and (price < trail_price)

        if cross_down or price_below_trail:
            state.last_pos = pos
            # Close entire long position
            return (-np.inf, 2, 1)

    # No action
    state.last_pos = pos
    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    """
    Precompute indicators required by the MACD + ATR trailing stop strategy.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Validate required columns
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain at least 'close' and 'high' columns")

    close_series = ohlcv["close"]
    high_series = ohlcv["high"]

    # Use 'low' if present for ATR, otherwise fall back to 'close'
    low_series = ohlcv["low"] if "low" in ohlcv.columns else ohlcv["close"]

    # MACD
    macd_res = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # ATR
    atr_res = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # SMA (trend filter)
    sma_res = vbt.MA.run(close_series, window=sma_period)

    return {
        "close": close_series.values,
        "high": high_series.values,
        "macd": macd_res.macd.values,
        "signal": macd_res.signal.values,
        "atr": atr_res.atr.values,
        "sma": sma_res.ma.values,
    }