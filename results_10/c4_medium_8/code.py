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
    pos_now = float(c.position_now)

    # Initialize persistent state on the function object. This keeps track of
    # the tentative/pending entry index when we signal an entry (since the
    # actual fill/position record may only be available in subsequent calls).
    if not hasattr(order_func, "_state"):
        order_func._state = {
            "pending_entry_idx": None,  # index where we placed an entry order
            "entry_idx": None,  # confirmed entry index (from pos_record if available)
            "last_pos": 0.0,
        }

    state = order_func._state

    # Helpers to safely access previous values
    def _safe_prev(arr: np.ndarray, idx: int):
        if idx <= 0 or arr is None:
            return np.nan
        return arr[idx - 1]

    # Try to extract confirmed entry index from OrderContext.pos_record_now
    def _extract_entry_idx_from_context():
        try:
            pr = c.pos_record_now
            # pos_record_now is a numpy.void with named fields. Try common access patterns.
            if pr is None:
                return None
            # Try dict-like access
            try:
                val = pr['init_i']
                if np.isfinite(val):
                    return int(val)
            except Exception:
                pass
            # Try attribute access
            try:
                val = getattr(pr, 'init_i')
                if np.isfinite(val):
                    return int(val)
            except Exception:
                pass
        except Exception:
            pass
        # Fallbacks using other possible OrderContext attributes
        try:
            ll = getattr(c, 'last_lidx', None)
            if ll is not None:
                # ll can be an array or scalar
                if np.isscalar(ll):
                    if np.isfinite(ll) and int(ll) >= 0:
                        return int(ll)
                else:
                    # Try to get element corresponding to current column if available
                    try:
                        col = int(getattr(c, 'col', 0))
                        val = ll[col]
                        if np.isfinite(val) and int(val) >= 0:
                            return int(val)
                    except Exception:
                        # As a last resort, take the last element
                        try:
                            val = ll[-1]
                            if np.isfinite(val) and int(val) >= 0:
                                return int(val)
                        except Exception:
                            pass
        except Exception:
            pass

        return None

    # Update confirmed entry index if we are in position but don't have it yet
    if pos_now > 0 and state["entry_idx"] is None:
        entry_idx_ctx = _extract_entry_idx_from_context()
        if entry_idx_ctx is not None:
            state["entry_idx"] = entry_idx_ctx
        elif state["pending_entry_idx"] is not None:
            # If we placed an entry order recently, assume it was filled at that index
            state["entry_idx"] = int(state["pending_entry_idx"])

    # Clear pending/confirmed entry idx if we are flat
    if pos_now == 0:
        state["pending_entry_idx"] = None
        state["entry_idx"] = None

    # Avoid acting on the very first bar
    if i <= 0:
        state["last_pos"] = pos_now
        return (np.nan, 0, 0)

    # Read current and previous indicator values safely
    macd_curr = macd[i] if macd is not None else np.nan
    macd_prev = _safe_prev(macd, i)
    signal_curr = signal[i] if signal is not None else np.nan
    signal_prev = _safe_prev(signal, i)
    sma_curr = sma[i] if sma is not None else np.nan

    # ENTRY: no position -> look for MACD crossover up and price above SMA
    if pos_now == 0.0:
        # Reset entry_idx state when flat
        # (already done above but keep for safety)
        state["entry_idx"] = None

        # Check for valid indicator values
        if (
            not np.isnan(macd_prev)
            and not np.isnan(signal_prev)
            and not np.isnan(macd_curr)
            and not np.isnan(signal_curr)
            and not np.isnan(sma_curr)
            and not np.isnan(close[i])
        ):
            crossed_up = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
            price_above_sma = close[i] > sma_curr

            if crossed_up and price_above_sma:
                # Place entry order: use 50% of equity by default
                state["pending_entry_idx"] = int(i)
                # Return buy with 50% of equity, long only (size_type=2, direction=1)
                state["last_pos"] = pos_now
                return (0.5, 2, 1)

    # EXIT: have a position -> check MACD cross down or ATR trailing stop breach
    else:
        should_exit = False

        # 1) MACD cross below signal
        if (
            not np.isnan(macd_prev)
            and not np.isnan(signal_prev)
            and not np.isnan(macd_curr)
            and not np.isnan(signal_curr)
        ):
            crossed_down = (macd_prev >= signal_prev) and (macd_curr < signal_curr)
            if crossed_down:
                should_exit = True

        # 2) ATR-based trailing stop: price < (highest_since_entry - trailing_mult * ATR)
        # Only evaluate if ATR is available
        atr_curr = atr[i] if atr is not None else np.nan
        if not should_exit and not np.isnan(atr_curr) and atr_curr > 0:
            # Determine entry index for calculating the highest price since entry
            entry_idx = state.get("entry_idx")
            if entry_idx is None:
                entry_idx = _extract_entry_idx_from_context()
            if entry_idx is None:
                entry_idx = state.get("pending_entry_idx")

            # Fallback: if we still don't know entry index, assume the run-up started at current position
            if entry_idx is None or entry_idx < 0 or entry_idx > i:
                entry_idx = i

            # Compute highest high since entry (inclusive)
            try:
                if entry_idx <= i:
                    highest = np.nanmax(high[int(entry_idx) : i + 1])
                else:
                    highest = high[i]
            except Exception:
                highest = high[i]

            if np.isnan(highest) or np.isnan(atr_curr):
                trailing_level = np.nan
            else:
                trailing_level = float(highest) - float(trailing_mult) * float(atr_curr)

            if not np.isnan(trailing_level) and not np.isnan(close[i]):
                if close[i] < trailing_level:
                    should_exit = True

        if should_exit:
            # Clear state and close entire long position
            state["pending_entry_idx"] = None
            state["entry_idx"] = None
            state["last_pos"] = pos_now
            return (-np.inf, 2, 1)

    # Update last_pos and do nothing
    state["last_pos"] = pos_now
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
    # Validate required columns
    required_cols = ["high", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise ValueError(f"compute_indicators: required column '{col}' not found in ohlcv")

    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)

    # low is optional in DATA_SCHEMA; ATR needs low - fallback to close if missing
    if "low" in ohlcv.columns:
        low_s = ohlcv["low"].astype(float)
    else:
        # Fallback: use close as proxy for low (conservative when low missing)
        low_s = close_s

    # Compute MACD
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_s, window=sma_period)

    # Ensure outputs are 1-d numpy arrays
    macd_arr = np.asarray(macd_ind.macd.ravel()).astype(float)
    signal_arr = np.asarray(macd_ind.signal.ravel()).astype(float)
    atr_arr = np.asarray(atr_ind.atr.ravel()).astype(float)
    sma_arr = np.asarray(sma_ind.ma.ravel()).astype(float)

    return {
        "close": np.asarray(close_s.ravel()).astype(float),
        "high": np.asarray(high_s.ravel()).astype(float),
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }