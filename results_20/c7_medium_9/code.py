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
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Convert inputs to numpy arrays
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    def is_finite_at(arr, idx):
        return 0 <= idx < arr.shape[0] and np.isfinite(arr[idx])

    # ENTRY: when flat
    if pos == 0.0:
        if i > 0 and is_finite_at(macd, i - 1) and is_finite_at(signal, i - 1) and is_finite_at(macd, i) and is_finite_at(signal, i) and is_finite_at(sma, i) and is_finite_at(close, i):
            macd_prev = macd[i - 1]
            signal_prev = signal[i - 1]
            macd_curr = macd[i]
            signal_curr = signal[i]
            price = close[i]
            sma_curr = sma[i]

            macd_cross_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
            price_above_sma = price > sma_curr

            if macd_cross_up and price_above_sma:
                # Enter: use 50% of equity
                return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # If in position, check exits
    # Determine entry index for current position using pos_record_now (if available)
    entry_idx = None
    try:
        rec = getattr(c, 'pos_record_now', None)
        if rec is not None:
            # Try attribute access first
            for key in ('init_i', 'init_idx', 'init_index', 'entry_i', 'entry_idx', 'entry_index'):
                try:
                    val = getattr(rec, key)
                    if isinstance(val, (int, np.integer)) and val >= 0:
                        entry_idx = int(val)
                        break
                except Exception:
                    pass

            # If not found, try structured dtype names
            if entry_idx is None:
                try:
                    names = getattr(rec.dtype, 'names', None)
                    if names:
                        for name in names:
                            lname = name.lower()
                            if lname in ('init_i', 'init_idx', 'init_index', 'entry_i', 'entry_idx', 'entry_index', 'i'):
                                try:
                                    val = rec[name]
                                    if np.isfinite(val):
                                        entry_idx = int(val)
                                        break
                                except Exception:
                                    pass
                except Exception:
                    pass
    except Exception:
        entry_idx = None

    # Compute highest since entry
    if entry_idx is None or entry_idx < 0 or entry_idx > i:
        # Fallback: use the max since the beginning of data up to current bar
        try:
            highest_since_entry = np.nanmax(high[: i + 1])
        except ValueError:
            highest_since_entry = np.nan
    else:
        try:
            highest_since_entry = np.nanmax(high[entry_idx: i + 1])
        except ValueError:
            highest_since_entry = np.nan

    price = close[i] if is_finite_at(close, i) else np.nan

    # MACD bearish cross check
    macd_bearish = False
    if i > 0 and is_finite_at(macd, i - 1) and is_finite_at(signal, i - 1) and is_finite_at(macd, i) and is_finite_at(signal, i):
        macd_prev = macd[i - 1]
        signal_prev = signal[i - 1]
        macd_curr = macd[i]
        signal_curr = signal[i]
        macd_bearish = (macd_prev >= signal_prev) and (macd_curr < signal_curr)

    # Trailing stop check
    trailing_hit = False
    if np.isfinite(highest_since_entry) and is_finite_at(atr, i) and np.isfinite(price):
        stop_price = highest_since_entry - (trailing_mult * atr[i])
        if np.isfinite(stop_price) and price < stop_price:
            trailing_hit = True

    if macd_bearish or trailing_hit:
        # Close entire long position
        return (-np.inf, 2, 1)

    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict:
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
        raise ValueError("ohlcv DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' column")

    close_series = ohlcv['close'].astype(float)
    high_series = ohlcv['high'].astype(float)

    # low may be missing - fallback to close
    if 'low' in ohlcv.columns:
        low_series = ohlcv['low'].astype(float)
    else:
        low_series = close_series.copy()

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)

    return {
        'close': close_series.values.astype(float),
        'high': high_series.values.astype(float),
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }