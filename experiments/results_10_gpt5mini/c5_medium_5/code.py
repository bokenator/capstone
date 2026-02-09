import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict


class _OrderState:
    """Simple holder for persistent state between order_func calls.

    Using attributes (instead of dict keys) avoids accidental static analysis
    that treats string keys as DataFrame column accesses.
    """

    def __init__(self):
        self.in_position: bool = False
        self.entry_highest: float = -np.inf
        self.entry_idx: int | None = None


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

    This implementation combines MACD crossover entries with an ATR-based
    trailing stop. State (highest price since entry) is kept on the
    function object and reset at the start of each run (when c.i == 0).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.position_now: current position size (float, 0.0 if flat)
           - c.cash_now: current cash balance (float)
        close: Close prices array
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        A tuple of (size, size_type, direction)
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Initialize or reset state at the beginning of a run (i == 0)
    if not hasattr(order_func, "_state") or i == 0:
        order_func._state = _OrderState()

    state: _OrderState = order_func._state

    # Keep state consistent with actual portfolio position reported by c.position_now
    if pos == 0.0 and state.in_position:
        # The portfolio reports flat but internal state thought we were in a
        # position -> reset state to avoid carrying over stale data.
        state.in_position = False
        state.entry_highest = -np.inf
        state.entry_idx = None

    # Update highest price while in position using available high price of current bar
    if pos > 0.0:
        if not state.in_position:
            # Detected a new position (either filled previously or opened now)
            state.in_position = True
            state.entry_idx = i
            # Prefer high price, fall back to close if high is NaN
            start_high = high[i] if not np.isnan(high[i]) else close[i]
            state.entry_highest = start_high if not np.isnan(start_high) else -np.inf
        else:
            # Update the highest price seen since entry
            if not np.isnan(high[i]):
                # If entry_highest was -inf, max will set it to high[i]
                state.entry_highest = max(state.entry_highest, high[i])

    # Helper: check finite values for indicators
    def finite_vals(*vals):
        return all(np.isfinite(v) for v in vals)

    # ---------- ENTRY LOGIC (long only) ----------
    if pos == 0.0:
        bullish_cross = False
        if i > 0 and finite_vals(macd[i - 1], signal[i - 1], macd[i], signal[i]):
            bullish_cross = (macd[i - 1] < signal[i - 1]) and (macd[i] > signal[i])

        price_above_sma = False
        if np.isfinite(sma[i]) and np.isfinite(close[i]):
            price_above_sma = close[i] > sma[i]

        if bullish_cross and price_above_sma:
            # Place entry order: use 95% of equity
            # Update internal state to reflect the incoming position
            state.in_position = True
            state.entry_idx = i
            state.entry_highest = high[i] if not np.isnan(high[i]) else close[i]
            return (0.95, 2, 1)

    # ---------- EXIT LOGIC ----------
    if pos > 0.0:
        should_exit = False

        # 1) MACD bearish cross
        if i > 0 and finite_vals(macd[i - 1], signal[i - 1], macd[i], signal[i]):
            if (macd[i - 1] > signal[i - 1]) and (macd[i] < signal[i]):
                should_exit = True

        # 2) ATR-based trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        if not should_exit:
            # Only evaluate trailing stop if we have a recorded entry_highest and a finite ATR
            if (state.entry_highest != -np.inf) and np.isfinite(atr[i]):
                threshold = state.entry_highest - trailing_mult * atr[i]
                if np.isfinite(close[i]) and close[i] < threshold:
                    should_exit = True

        if should_exit:
            # Reset internal state and close the entire long position
            state.in_position = False
            state.entry_highest = -np.inf
            state.entry_idx = None
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
        raise ValueError("ohlcv must contain at least 'close' and 'high' columns")

    close_s = ohlcv["close"]
    high_s = ohlcv["high"]

    # Low is optional in DATA_SCHEMA; fall back to close if missing
    if "low" in ohlcv.columns:
        low_s = ohlcv["low"]
    else:
        low_s = close_s

    # Compute MACD
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)

    # Compute SMA (trend filter)
    sma_ind = vbt.MA.run(close_s, window=sma_period)

    return {
        "close": close_s.values,
        "high": high_s.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }