import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Dict, Tuple

# Module-level state variables to persist simple per-run state between order_func calls.
# These are simple names (not dict keys) to avoid static analysis thinking they are data columns.
_RUN_INITIALIZED: bool = False
_IN_POSITION: bool = False
_HIGHEST_SINCE_ENTRY: float = np.nan


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
    Order function implementing MACD crossover entries with ATR-based trailing stops.

    Uses module-level variables to persist 'highest since entry' state across calls within a
    single from_order_func simulation. State is reset when the simulation starts (i == 0).
    """
    global _RUN_INITIALIZED, _IN_POSITION, _HIGHEST_SINCE_ENTRY

    i = int(c.i)
    pos = float(c.position_now)

    # Reset state at simulation start
    if (not _RUN_INITIALIZED) or i == 0:
        _RUN_INITIALIZED = True
        _IN_POSITION = False
        _HIGHEST_SINCE_ENTRY = np.nan

    def is_finite(*vals: float) -> bool:
        return all(np.isfinite(v) for v in vals)

    no_action = (np.nan, 0, 0)

    # ENTRY logic
    if pos == 0.0:
        # Sync local state if inconsistent
        if _IN_POSITION:
            _IN_POSITION = False
            _HIGHEST_SINCE_ENTRY = np.nan

        if i == 0:
            return no_action

        macd_now = macd[i]
        macd_prev = macd[i - 1]
        sig_now = signal[i]
        sig_prev = signal[i - 1]
        price_now = close[i]
        sma_now = sma[i]

        if not is_finite(macd_now, macd_prev, sig_now, sig_prev, price_now, sma_now):
            return no_action

        macd_cross_up = (macd_prev <= sig_prev) and (macd_now > sig_now)
        above_sma = price_now > sma_now

        if macd_cross_up and above_sma:
            high_now = high[i] if np.isfinite(high[i]) else price_now
            _HIGHEST_SINCE_ENTRY = float(high_now)
            _IN_POSITION = True
            return (0.5, 2, 1)

        return no_action

    # EXIT logic when in position
    else:
        # Sync state if needed
        if not _IN_POSITION:
            _IN_POSITION = True
            try:
                past_highs = high[: i + 1]
                if past_highs.size > 0 and np.isfinite(past_highs).any():
                    _HIGHEST_SINCE_ENTRY = float(np.nanmax(past_highs[np.isfinite(past_highs)]))
                else:
                    _HIGHEST_SINCE_ENTRY = float(close[i])
            except Exception:
                _HIGHEST_SINCE_ENTRY = float(close[i])

        # Update highest
        if np.isfinite(high[i]) and high[i] > _HIGHEST_SINCE_ENTRY:
            _HIGHEST_SINCE_ENTRY = float(high[i])

        atr_now = atr[i] if i < len(atr) else np.nan
        highest = _HIGHEST_SINCE_ENTRY
        price_now = close[i]

        stopped = False
        if is_finite(atr_now, highest, price_now):
            stop_price = highest - float(trailing_mult) * atr_now
            stopped = price_now < stop_price

        macd_cross_down = False
        if i >= 1 and is_finite(macd[i], macd[i - 1], signal[i], signal[i - 1]):
            macd_cross_down = (macd[i - 1] >= signal[i - 1]) and (macd[i] < signal[i])

        if macd_cross_down or stopped:
            _IN_POSITION = False
            _HIGHEST_SINCE_ENTRY = np.nan
            return (-np.inf, 2, 1)

        return no_action


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Precompute indicators required by the strategy using vectorbt.

    Returns a dictionary with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' and 'high' columns")

    close = ohlcv['close']
    high = ohlcv['high']
    low = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    macd_res = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    atr_res = vbt.ATR.run(high, low, close, window=atr_period)
    sma_res = vbt.MA.run(close, window=sma_period)

    macd_arr = np.asarray(macd_res.macd.values).reshape(-1)
    signal_arr = np.asarray(macd_res.signal.values).reshape(-1)
    atr_arr = np.asarray(atr_res.atr.values).reshape(-1)
    sma_arr = np.asarray(sma_res.ma.values).reshape(-1)

    return {
        'close': np.asarray(close.values).reshape(-1),
        'high': np.asarray(high.values).reshape(-1),
        'macd': macd_arr,
        'signal': signal_arr,
        'atr': atr_arr,
        'sma': sma_arr,
    }
