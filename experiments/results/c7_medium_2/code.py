import numpy as np
import pandas as pd
import vectorbt as vbt

# Global state storage keyed by OrderContext id to persist data across calls.
# We cannot attach arbitrary attributes to OrderContext in this environment, so
# use a module-level dictionary instead. Use an integer key to avoid static
# analyzers confusing strings with DataFrame column access.
_CTX_STATE = {}
_STATE_KEY_HIGHEST = 0


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
    Order generation function for vectorbt.from_order_func (pure Python, no numba).

    Implements MACD crossover entry with a 50-period SMA trend filter and an
    ATR-based trailing stop (trailing_mult * ATR from the highest price since
    entry). Long-only strategy.

    Returns vbt order objects (created via vbt.portfolio.nb.order_nb) or
    vbt.portfolio.enums.NoOrder for no action. This ensures compatibility with
    vectorbt's simulation internals.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # Use a global dict keyed by the context object's id to persist state
    ctx_id = id(c)
    if ctx_id not in _CTX_STATE:
        _CTX_STATE[ctx_id] = { _STATE_KEY_HIGHEST: np.nan }
    state = _CTX_STATE[ctx_id]

    # Helper to check finiteness
    def is_finite(val) -> bool:
        try:
            return np.isfinite(val)
        except Exception:
            return False

    # No position: check for entry signal
    if pos == 0.0:
        # Detect MACD bullish crossover: previous MACD <= previous signal and current MACD > current signal
        if i >= 1:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            curr_macd = macd[i]
            curr_signal = signal[i]

            if (
                is_finite(prev_macd)
                and is_finite(prev_signal)
                and is_finite(curr_macd)
                and is_finite(curr_signal)
                and prev_macd <= prev_signal
                and curr_macd > curr_signal
            ):
                # Trend filter: price must be above SMA
                price = close[i]
                sma_val = sma[i]
                if is_finite(price) and is_finite(sma_val) and price > sma_val:
                    # Prepare highest_since_entry for when the position becomes active on next bar(s)
                    hi = high[i] if is_finite(high[i]) else price
                    state[_STATE_KEY_HIGHEST] = float(hi)
                    # Enter with 50% of equity (Percent sizing)
                    return vbt.portfolio.nb.order_nb(0.5, np.inf, 2, 1)

    else:
        # Have a long position - update highest_since_entry and check exit conditions
        if not is_finite(state.get(_STATE_KEY_HIGHEST, np.nan)):
            state[_STATE_KEY_HIGHEST] = float(high[i]) if is_finite(high[i]) else float(close[i])

        # Update the highest price seen since entry (use intrabar high)
        if is_finite(high[i]) and high[i] > state[_STATE_KEY_HIGHEST]:
            state[_STATE_KEY_HIGHEST] = float(high[i])

        # Check MACD bearish crossover for exit
        macd_bear_cross = False
        if i >= 1:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]
            curr_macd = macd[i]
            curr_signal = signal[i]

            if (
                is_finite(prev_macd)
                and is_finite(prev_signal)
                and is_finite(curr_macd)
                and is_finite(curr_signal)
                and prev_macd >= prev_signal
                and curr_macd < curr_signal
            ):
                macd_bear_cross = True

        # Check ATR-based trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
        trail_exit = False
        if is_finite(state[_STATE_KEY_HIGHEST]) and is_finite(atr[i]):
            trail_level = state[_STATE_KEY_HIGHEST] - float(trailing_mult) * float(atr[i])
            if is_finite(close[i]) and close[i] < trail_level:
                trail_exit = True

        if macd_bear_cross or trail_exit:
            # Reset tracked high after exit
            state[_STATE_KEY_HIGHEST] = np.nan
            # Close entire long position
            return vbt.portfolio.nb.order_nb(-np.inf, np.inf, 2, 1)

    # No action
    return vbt.portfolio.enums.NoOrder


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict[str, np.ndarray]:
    """
    Compute required indicators using vectorbt's indicator classes.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays aligned with the input ohlcv index.
    """
    # Validate required columns
    if "close" not in ohlcv.columns or "high" not in ohlcv.columns:
        raise ValueError("Input ohlcv DataFrame must contain at least 'high' and 'close' columns")

    close_series = ohlcv["close"]
    high_series = ohlcv["high"]

    # Some datasets may omit 'low' (optional in DATA_SCHEMA). Use close as fallback.
    if "low" in ohlcv.columns:
        low_series = ohlcv["low"]
    else:
        low_series = close_series

    # Compute MACD
    macd_ind = vbt.MACD.run(close_series, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_series, low_series, close_series, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_series, window=sma_period)

    return {
        "close": close_series.values,
        "high": high_series.values,
        "macd": macd_ind.macd.values,
        "signal": macd_ind.signal.values,
        "atr": atr_ind.atr.values,
        "sma": sma_ind.ma.values,
    }
