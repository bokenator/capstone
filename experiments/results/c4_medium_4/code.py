import numpy as np
import pandas as pd
import vectorbt as vbt


def order_func(
    c: object,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> object:
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    This implementation returns vbt order objects (created with
    vbt.portfolio.nb.order_nb) and uses vbt.portfolio.enums.NoOrder for
    no-action. This ensures compatibility with vectorbt's internal
    simulator.
    """
    i = int(c.i)
    pos = float(c.position_now)

    def _get(arr: np.ndarray, idx: int) -> float:
        if idx < 0 or idx >= arr.shape[0]:
            return np.nan
        return float(arr[idx])

    close_i = _get(close, i)
    high_i = _get(high, i)
    macd_i = _get(macd, i)
    signal_i = _get(signal, i)
    atr_i = _get(atr, i)
    sma_i = _get(sma, i)

    def _is_finite(x: float) -> bool:
        return np.isfinite(x)

    # ENTRY
    if pos == 0.0:
        cross_up = False
        if i > 0:
            prev_macd = _get(macd, i - 1)
            prev_signal = _get(signal, i - 1)
            if _is_finite(prev_macd) and _is_finite(prev_signal) and _is_finite(macd_i) and _is_finite(signal_i):
                cross_up = (prev_macd <= prev_signal) and (macd_i > signal_i)

        trend_ok = _is_finite(sma_i) and _is_finite(close_i) and (close_i > sma_i)

        if cross_up and trend_ok:
            # Initialize tracking attributes on the context
            try:
                setattr(c, "entry_idx", i)
                entry_high = high_i if _is_finite(high_i) else close_i
                setattr(c, "entry_highest", float(entry_high) if _is_finite(entry_high) else np.nan)
            except Exception:
                pass

            # Use 100% of equity to enter (size_type=2 -> Percent, direction=1 -> LongOnly)
            return vbt.portfolio.nb.order_nb(1.0, 2, 1)

    # EXIT (when in position)
    if pos > 0.0:
        # Retrieve and maintain highest since entry
        try:
            entry_highest = getattr(c, "entry_highest", np.nan)
        except Exception:
            entry_highest = np.nan

        if not _is_finite(entry_highest):
            entry_highest = high_i if _is_finite(high_i) else close_i

        if _is_finite(high_i) and _is_finite(entry_highest):
            entry_highest = float(np.maximum(entry_highest, high_i))
        elif _is_finite(high_i):
            entry_highest = float(high_i)

        try:
            setattr(c, "entry_highest", entry_highest)
        except Exception:
            pass

        # Trailing stop level
        trailing_stop = np.nan
        if _is_finite(entry_highest) and _is_finite(atr_i):
            trailing_stop = entry_highest - (trailing_mult * atr_i)

        # MACD cross down
        cross_down = False
        if i > 0:
            prev_macd = _get(macd, i - 1)
            prev_signal = _get(signal, i - 1)
            if _is_finite(prev_macd) and _is_finite(prev_signal) and _is_finite(macd_i) and _is_finite(signal_i):
                cross_down = (prev_macd >= prev_signal) and (macd_i < signal_i)

        # Price below trailing stop
        price_below_trail = False
        if _is_finite(close_i) and _is_finite(trailing_stop):
            price_below_trail = close_i < trailing_stop

        if cross_down or price_below_trail:
            try:
                setattr(c, "entry_highest", np.nan)
                setattr(c, "entry_idx", None)
            except Exception:
                pass

            # Close entire long position (size=-inf, size_type=2 -> Percent)
            return vbt.portfolio.nb.order_nb(-np.inf, 2, 1)

    # No action
    return vbt.portfolio.enums.NoOrder


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict:
    """
    Precompute indicators required by the strategy using vectorbt indicator classes.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    if 'close' not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise KeyError("Input ohlcv must contain 'high' column")

    close_series = ohlcv['close']
    high_series = ohlcv['high']
    low_series = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_line = macd_ind.macd.values
    signal_line = macd_ind.signal.values

    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)
    atr_line = atr_ind.atr.values

    sma_ind = vbt.MA.run(close_series, window=sma_period)
    sma_line = sma_ind.ma.values

    return {
        'close': close_series.values,
        'high': high_series.values,
        'macd': macd_line,
        'signal': signal_line,
        'atr': atr_line,
        'sma': sma_line,
    }
