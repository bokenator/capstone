import numpy as np
import pandas as pd
import vectorbt as vbt

# Module-level state variables (avoid string-bracket access to satisfy static schema checks)
_IN_POSITION = False
_PENDING_ENTRY_IDX = None
_ENTRY_IDX = None
_HIGHEST = np.nan
_HIGHEST_HISTORY = []


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
    Generate order at each bar. Called by vectorbt's from_order_func.

    This function is intentionally a plain Python function (NO NUMBA).

    Args:
        c: vectorbt OrderContext with attributes c.i, c.position_now, c.cash_now
        close: Close prices array
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        Tuple (size, size_type, direction) as described in prompt.
    """
    global _IN_POSITION, _PENDING_ENTRY_IDX, _ENTRY_IDX, _HIGHEST, _HIGHEST_HISTORY

    i = int(c.i)
    pos = float(c.position_now)

    # Reset state on the first bar of a run to avoid cross-run leakage
    if i == 0:
        _IN_POSITION = False
        _PENDING_ENTRY_IDX = None
        _ENTRY_IDX = None
        _HIGHEST = np.nan
        _HIGHEST_HISTORY = []

    # Helper to safely check for NaN
    def _is_num(x):
        return not (x is None or (isinstance(x, float) and np.isnan(x)))

    # If we detect that a position exists now but our state says we are not in a position,
    # it means an entry order was executed on a previous bar. Record the entry index and
    # initialize the highest price since entry.
    if pos > 0 and not _IN_POSITION:
        entry_idx = _PENDING_ENTRY_IDX if _PENDING_ENTRY_IDX is not None else i
        entry_idx = int(entry_idx)
        _ENTRY_IDX = entry_idx

        try:
            if entry_idx <= i:
                seg = np.asarray(high[entry_idx : i + 1])
                if seg.size:
                    _HIGHEST = float(np.nanmax(seg))
                else:
                    _HIGHEST = float(high[i]) if _is_num(high[i]) else float(close[i])
            else:
                _HIGHEST = float(high[i]) if _is_num(high[i]) else float(close[i])
        except Exception:
            _HIGHEST = float(high[i]) if _is_num(high[i]) else float(close[i])

        _IN_POSITION = True
        _PENDING_ENTRY_IDX = None

    # If we were in a position but now flat, clear state
    if pos == 0 and _IN_POSITION:
        _IN_POSITION = False
        _PENDING_ENTRY_IDX = None
        _ENTRY_IDX = None
        _HIGHEST = np.nan

    # ENTRY LOGIC (only when flat)
    if pos == 0:
        should_enter = False
        if i > 0:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            cur_macd = macd[i]
            cur_signal = signal[i]
            if _is_num(prev_macd) and _is_num(prev_signal) and _is_num(cur_macd) and _is_num(cur_signal):
                if (cur_macd > cur_signal) and (prev_macd <= prev_signal):
                    should_enter = True

        price_above_sma = False
        if _is_num(close[i]) and _is_num(sma[i]):
            price_above_sma = close[i] > sma[i]

        if should_enter and price_above_sma:
            _PENDING_ENTRY_IDX = i
            return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # EXIT LOGIC (when in position)
    if pos > 0 or _IN_POSITION:
        if _is_num(high[i]):
            if not _is_num(_HIGHEST):
                _HIGHEST = float(high[i])
            else:
                _HIGHEST = float(max(_HIGHEST, float(high[i])))

        _HIGHEST_HISTORY.append(_HIGHEST)

        macd_bearish = False
        if i > 0:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            cur_macd = macd[i]
            cur_signal = signal[i]
            if _is_num(prev_macd) and _is_num(prev_signal) and _is_num(cur_macd) and _is_num(cur_signal):
                if (cur_macd < cur_signal) and (prev_macd >= prev_signal):
                    macd_bearish = True

        trailing_trigger = False
        if _is_num(_HIGHEST) and _is_num(atr[i]):
            stop_price = _HIGHEST - float(trailing_mult) * float(atr[i])
            if _is_num(close[i]):
                if close[i] < stop_price:
                    trailing_trigger = True

        if macd_bearish or trailing_trigger:
            return (-np.inf, 2, 1)

        return (np.nan, 0, 0)

    return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
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

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"] if "low" in ohlcv.columns else ohlcv["close"]

    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)
    sma_ind = vbt.MA.run(close, window=sma_period)

    return {
        "close": close.values,
        "high": high.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }