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
    trailing_mult: float,
) -> tuple:
    """
    Stateless order function that reconstructs entry points from indicator history.

    Returns a 4-tuple (size, price, size_type, direction). price=np.inf indicates market order.
    Use (0.0, np.inf, 0, 0) to indicate no action.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Safe getter
    def safe_get(arr, idx):
        if idx < 0 or idx >= arr.shape[0]:
            return np.nan
        v = arr[idx]
        try:
            return float(v)
        except Exception:
            return np.nan

    if i == 0:
        return (0.0, np.inf, 0, 0)

    macd_curr = safe_get(macd, i)
    macd_prev = safe_get(macd, i - 1)
    sig_curr = safe_get(signal, i)
    sig_prev = safe_get(signal, i - 1)
    close_curr = safe_get(close, i)
    sma_curr = safe_get(sma, i)

    # Entry condition
    if (
        not np.isnan(macd_curr)
        and not np.isnan(macd_prev)
        and not np.isnan(sig_curr)
        and not np.isnan(sig_prev)
        and not np.isnan(close_curr)
        and not np.isnan(sma_curr)
    ):
        macd_cross_up_now = (macd_curr > sig_curr) and (macd_prev <= sig_prev)
        price_above_sma = close_curr > sma_curr
        if pos == 0.0 and macd_cross_up_now and price_above_sma:
            return (0.5, np.inf, 2, 1)  # buy 50% equity (size_type=2=percent)

    # Detect last entry index based on MACD cross up and last cross down
    last_cross_up_idx = None
    for j in range(i, 0, -1):
        m_curr = safe_get(macd, j)
        m_prev = safe_get(macd, j - 1)
        s_curr = safe_get(signal, j)
        s_prev = safe_get(signal, j - 1)
        c_price = safe_get(close, j)
        s_ma = safe_get(sma, j)
        if (
            not np.isnan(m_curr)
            and not np.isnan(m_prev)
            and not np.isnan(s_curr)
            and not np.isnan(s_prev)
            and not np.isnan(c_price)
            and not np.isnan(s_ma)
        ):
            if (m_curr > s_curr) and (m_prev <= s_prev) and (c_price > s_ma):
                last_cross_up_idx = j
                break

    last_cross_down_idx = None
    for j in range(i, 0, -1):
        m_curr = safe_get(macd, j)
        m_prev = safe_get(macd, j - 1)
        s_curr = safe_get(signal, j)
        s_prev = safe_get(signal, j - 1)
        if (
            not np.isnan(m_curr)
            and not np.isnan(m_prev)
            and not np.isnan(s_curr)
            and not np.isnan(s_prev)
        ):
            if (m_curr < s_curr) and (m_prev >= s_prev):
                last_cross_down_idx = j
                break

    in_trade = False
    if last_cross_up_idx is not None:
        if (last_cross_down_idx is None) or (last_cross_up_idx > last_cross_down_idx):
            in_trade = True

    if pos > 0.0:
        in_trade = True

    if in_trade and pos > 0.0:
        # Exit on MACD cross down
        if (
            not np.isnan(macd_curr)
            and not np.isnan(macd_prev)
            and not np.isnan(sig_curr)
            and not np.isnan(sig_prev)
        ):
            macd_cross_down_now = (macd_curr < sig_curr) and (macd_prev >= sig_prev)
            if macd_cross_down_now:
                return (-pos, np.inf, 0, 1)

        # ATR trailing stop
        if last_cross_up_idx is not None:
            highest = -np.inf
            for j in range(last_cross_up_idx, i + 1):
                h = safe_get(high, j)
                if not np.isnan(h):
                    if highest == -np.inf:
                        highest = h
                    else:
                        highest = max(highest, h)
            atr_curr = safe_get(atr, i)
            if highest != -np.inf and not np.isnan(atr_curr) and not np.isnan(close_curr):
                stop_level = highest - trailing_mult * atr_curr
                if close_curr < stop_level:
                    return (-pos, np.inf, 0, 1)

    return (0.0, np.inf, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
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
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns:
        raise ValueError("Input ohlcv must contain 'close' and 'high' columns")

    close_ser = ohlcv["close"].astype(float)
    high_ser = ohlcv["high"].astype(float)

    # 'low' is optional in the DATA_SCHEMA; fall back to 'close' if missing
    if "low" in ohlcv.columns:
        low_ser = ohlcv["low"].astype(float)
    else:
        low_ser = close_ser

    # Compute MACD
    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)

    # Compute SMA (trend filter)
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    return {
        "close": close_ser.values,
        "high": high_ser.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }
