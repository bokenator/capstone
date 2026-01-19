import numpy as np
import pandas as pd
import vectorbt as vbt


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
    Order function for vectorbt.from_order_func.

    Notes:
    - `high` is expected to be the precomputed "highest price since entry" array
      produced by `compute_indicators` (NaN when not in a simulated entry period).

    Returns:
        vbt.portfolio.nb.order_nb(...) object for an order or vbt.portfolio.enums.NoOrder
        for no action. This is done to match vectorbt's internal expectations.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Basic safety checks: need at least one previous bar to detect crosses
    if i <= 0:
        return vbt.portfolio.enums.NoOrder

    # Guard against invalid data at current/previous bar
    if not (np.isfinite(macd[i]) and np.isfinite(signal[i]) and np.isfinite(macd[i - 1]) and np.isfinite(signal[i - 1])):
        return vbt.portfolio.enums.NoOrder

    price = close[i]

    # Compute cross signals at this bar
    cross_up = (macd[i] > signal[i]) and (macd[i - 1] <= signal[i - 1])
    cross_down = (macd[i] < signal[i]) and (macd[i - 1] >= signal[i - 1])

    # ENTRY: flat and MACD cross up + price above SMA
    if pos == 0.0:
        if cross_up and np.isfinite(sma[i]) and np.isfinite(price) and (price > sma[i]):
            # Enter long with 50% of equity
            try:
                return vbt.portfolio.nb.order_nb(0.5, 2, 1)
            except Exception:
                # Fallback to tuple form if order_nb isn't usable
                return (0.5, 2, 1)
        return vbt.portfolio.enums.NoOrder

    # EXIT: have position - exit on MACD cross down or ATR-based trailing stop
    # Exit on MACD cross down
    if cross_down:
        try:
            return vbt.portfolio.nb.order_nb(-np.inf, 2, 1)
        except Exception:
            return (-np.inf, 2, 1)

    # Trailing stop: high is highest_since_entry (precomputed). Ensure values are finite
    highest_since_entry = high[i]
    atr_val = atr[i]
    if np.isfinite(highest_since_entry) and np.isfinite(atr_val) and np.isfinite(price):
        trailing_level = highest_since_entry - trailing_mult * atr_val
        if price < trailing_level:
            try:
                return vbt.portfolio.nb.order_nb(-np.inf, 2, 1)
            except Exception:
                return (-np.inf, 2, 1)

    return vbt.portfolio.enums.NoOrder


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    Note: 'high' is repurposed to contain the "highest price since entry"
    (NaN when no entry has yet occurred or after a simulated MACD exit).
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_s = ohlcv['close'].astype(float)
    high_s = ohlcv['high'].astype(float)
    # low may be missing; fallback to close
    low_s = ohlcv['low'].astype(float) if 'low' in ohlcv.columns else close_s

    # Compute MACD (returns object with .macd and .signal as Series)
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_s = macd_ind.macd
    signal_s = macd_ind.signal

    # Compute SMA trend filter
    sma_s = vbt.MA.run(close_s, window=sma_period).ma

    # Compute ATR
    atr_s = vbt.ATR.run(high_s, low_s, close_s, window=atr_period).atr

    # Compute MACD crosses
    prev_macd = macd_s.shift(1)
    prev_signal = signal_s.shift(1)
    cross_up = (macd_s > signal_s) & (prev_macd <= prev_signal)
    cross_down = (macd_s < signal_s) & (prev_macd >= prev_signal)

    # Entry condition (used to start a highest_since_entry segment): MACD cross up + price > SMA
    entry_cond = cross_up & (close_s > sma_s)

    # Simulate simple state machine to compute highest_since_entry per bar.
    # We reset on entry_cond and on MACD cross down.
    n = len(close_s)
    highest_since_entry = np.full(n, np.nan, dtype=float)

    active = False
    current_max = np.nan

    # Work with raw numpy values for speed
    high_vals = high_s.values
    entry_vals = entry_cond.values
    cross_down_vals = cross_down.values

    for i in range(n):
        if entry_vals[i]:
            # Start a new simulated position
            active = True
            current_max = float(high_vals[i]) if np.isfinite(high_vals[i]) else np.nan
        elif active and cross_down_vals[i]:
            # End simulated position on MACD cross down
            active = False
            current_max = np.nan

        if active:
            # Update running maximum
            hi = high_vals[i]
            if np.isfinite(hi):
                if not np.isfinite(current_max):
                    current_max = float(hi)
                else:
                    # Use Python max to avoid np issues
                    current_max = float(max(current_max, float(hi)))
            # Store the current highest since entry
            highest_since_entry[i] = current_max

    return {
        'close': close_s.values,
        # We intentionally pass the highest_since_entry array as 'high' so that order_func
        # receives it as the `high` parameter (precomputed highest price since entry).
        'high': highest_since_entry,
        'macd': macd_s.values,
        'signal': signal_s.values,
        'atr': atr_s.values,
        'sma': sma_s.values,
    }
