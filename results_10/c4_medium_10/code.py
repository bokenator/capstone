import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


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

    # Basic bounds/sanity checks
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Helper to safely extract scalar index from OrderContext attributes
    def _scalar_index(val):
        """Try to convert various possible types to a Python int index.
        Returns -1 on failure.
        """
        if val is None:
            return -1
        try:
            # numpy scalar
            if hasattr(val, 'item'):
                return int(val.item())
            # numpy array or list-like
            if isinstance(val, (list, tuple, np.ndarray)):
                # try first element
                arr = np.asarray(val)
                if arr.size == 0:
                    return -1
                return int(arr.ravel()[0])
            return int(val)
        except Exception:
            return -1

    # Read current values (guard against NaNs)
    close_i = float(close[i]) if not np.isnan(close[i]) else np.nan
    macd_i = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    signal_i = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    sma_i = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # ENTRY: no open position
    if pos == 0.0:
        # Need previous bar to detect crossover
        if i == 0:
            return (np.nan, 0, 0)
        prev_macd = macd[i - 1]
        prev_signal = signal[i - 1]

        # If any required value is NaN, skip
        if (
            np.isnan(prev_macd) or np.isnan(prev_signal) or
            np.isnan(macd_i) or np.isnan(signal_i) or
            np.isnan(sma_i) or np.isnan(close_i)
        ):
            return (np.nan, 0, 0)

        macd_cross_up = (prev_macd <= prev_signal) and (macd_i > signal_i)
        price_above_sma = close_i > sma_i

        if macd_cross_up and price_above_sma:
            # Enter long with 50% of equity
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # EXIT: have an open long position
    else:
        # 1) MACD cross below
        macd_cross_down = False
        if i > 0:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            if not (np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(macd_i) or np.isnan(signal_i)):
                macd_cross_down = (prev_macd >= prev_signal) and (macd_i < signal_i)

        # 2) ATR-based trailing stop
        trailing_hit = False

        # Prefer last long index; fall back to last order index
        last_lidx = _scalar_index(getattr(c, 'last_lidx', None))
        if last_lidx < 0:
            last_lidx = _scalar_index(getattr(c, 'last_oidx', None))

        if last_lidx >= 0:
            # clamp
            start = max(0, last_lidx)
            if start <= i:
                seg = high[start:i + 1]
                # If segment is all NaN, skip
                if not np.all(np.isnan(seg)):
                    highest = float(np.nanmax(seg))
                    atr_i = float(atr[i]) if not np.isnan(atr[i]) else np.nan
                    if not np.isnan(highest) and not np.isnan(atr_i) and not np.isnan(close_i):
                        threshold = highest - float(trailing_mult) * atr_i
                        if close_i < threshold:
                            trailing_hit = True

        if macd_cross_down or trailing_hit:
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

    close_s = ohlcv['close']
    high_s = ohlcv['high']

    # Low is optional in DATA_SCHEMA; fall back to close if missing
    low_s = ohlcv['low'] if 'low' in ohlcv.columns else close_s

    # Compute MACD
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast,
                             slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_s, window=sma_period)

    return {
        'close': close_s.values.astype(float),
        'high': high_s.values.astype(float),
        'macd': macd_ind.macd.values.astype(float),
        'signal': macd_ind.signal.values.astype(float),
        'atr': atr_ind.atr.values.astype(float),
        'sma': sma_ind.ma.values.astype(float),
    }