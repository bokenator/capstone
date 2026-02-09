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
    Generate order at each bar. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.position_now: current position size (float, 0.0 if flat)
           - c.cash_now: current cash balance (float)
        close: Close prices array (use close[c.i] for current price)
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent (of equity)
        - direction: int, 0=Both, 1=LongOnly, 2=ShortOnly

    Notes:
        - Uses function attributes to track the highest price since entry for the
          trailing stop: order_func._highest
        - Uses percent sizing for entries (50% of equity) and closes entire
          long position using (-np.inf, 2, 1).
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize persistent attributes on the function object if not present
    if not hasattr(order_func, "_entry_idx"):
        order_func._entry_idx = None
    if not hasattr(order_func, "_highest"):
        order_func._highest = None

    # Safety checks for current values
    # Use local variables to avoid repeated indexing
    curr_close = float(close[i]) if not np.isnan(close[i]) else np.nan
    curr_high = float(high[i]) if not np.isnan(high[i]) else np.nan
    curr_macd = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    curr_signal = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    curr_atr = float(atr[i]) if not np.isnan(atr[i]) else np.nan
    curr_sma = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # If flat, clear any stored entry state (stale state protection)
    if pos == 0.0:
        order_func._entry_idx = None
        order_func._highest = None

        # Check entry conditions
        # Need at least one previous bar to detect MACD crossover
        if i == 0:
            return (np.nan, 0, 0)

        prev_macd = macd[i - 1]
        prev_signal = signal[i - 1]

        # Ensure we have necessary non-NaN values
        if (
            not np.isnan(prev_macd)
            and not np.isnan(prev_signal)
            and not np.isnan(curr_macd)
            and not np.isnan(curr_signal)
            and not np.isnan(curr_close)
            and not np.isnan(curr_sma)
        ):
            macd_cross_up = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
            price_above_sma = curr_close > curr_sma

            if macd_cross_up and price_above_sma:
                # Register entry index and highest price since entry
                order_func._entry_idx = i
                order_func._highest = curr_high if not np.isnan(curr_high) else curr_close

                # Enter with 50% of equity (percent size type = 2), long only
                return (0.5, 2, 1)

        return (np.nan, 0, 0)

    else:
        # We have a position - update highest since entry
        if order_func._highest is None:
            # If for some reason it's not set (e.g., resumed state), initialize
            order_func._highest = curr_high if not np.isnan(curr_high) else curr_close
        else:
            if not np.isnan(curr_high):
                # Update highest observed price since entry
                order_func._highest = max(order_func._highest, curr_high)

        # Check exit conditions
        # 1) MACD line crosses below Signal line
        macd_cross_down = False
        if i > 0:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            if (
                not np.isnan(prev_macd)
                and not np.isnan(prev_signal)
                and not np.isnan(curr_macd)
                and not np.isnan(curr_signal)
            ):
                macd_cross_down = (prev_macd >= prev_signal) and (curr_macd < curr_signal)

        # 2) Trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        trailing_hit = False
        if order_func._highest is not None and not np.isnan(curr_atr):
            trailing_level = order_func._highest - (trailing_mult * curr_atr)
            if not np.isnan(curr_close) and curr_close < trailing_level:
                trailing_hit = True

        if macd_cross_down or trailing_hit:
            # Reset stored state and close position (close all longs)
            order_func._entry_idx = None
            order_func._highest = None
            # Close entire long position using percent -inf
            return (-np.inf, 2, 1)

        # No action
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
    Precompute all indicators. Use vectorbt indicator classes.

    Args:
        ohlcv: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        sma_period: SMA trend filter period
        atr_period: ATR period

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
        All values are np.ndarray of same length as input.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    # Prepare series (use low fallback if not available)
    close_sr = ohlcv['close']
    high_sr = ohlcv['high']
    low_sr = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_res = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_res = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)

    # Compute SMA
    sma_res = vbt.MA.run(close_sr, window=sma_period)

    return {
        'close': close_sr.values,
        'high': high_sr.values,
        'macd': macd_res.macd.values,
        'signal': macd_res.signal.values,
        'atr': atr_res.atr.values,
        'sma': sma_res.ma.values,
    }