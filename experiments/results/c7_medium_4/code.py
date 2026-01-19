import numpy as np
import pandas as pd
import vectorbt as vbt


# Simple object to hold order/trailing state without using attribute keys
class _OrderState:
    def __init__(self):
        self.in_position = False
        self.highest = None
        self.entry_idx = None


_ORDER_STATE = _OrderState()


def _reset_order_state() -> None:
    """Reset module-level and function-level order state.

    This ensures that multiple backtest runs / unit tests start from a clean state.
    """
    global _ORDER_STATE
    _ORDER_STATE.in_position = False
    _ORDER_STATE.highest = None
    _ORDER_STATE.entry_idx = None

    # Also try to reset attributes on order_func if available
    try:
        order_func._in_position = False
        order_func._highest = None
        order_func._entry_idx = None
    except Exception:
        # If order_func not yet defined, ignore
        pass


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

    This implementation combines MACD crossover entries (MACD crossing above Signal)
    with a 50-period SMA trend filter, and ATR-based trailing stops.

    Entry rules (all must be true):
    - MACD crosses above Signal (current bar)
    - Price (close) is above SMA

    Exit rules (any):
    - MACD crosses below Signal
    - Price falls below (highest_since_entry - trailing_mult * ATR)

    Notes:
    - Long-only. Entries use 50% of equity.
    - Exits close the entire long position using percent -inf sentinel.

    Returns vbt.portfolio.nb.order_nb objects for compatibility with vectorbt internals.
    When creating market orders, price is set to np.inf (vectorbt sentinel for market).
    """
    i = int(c.i)

    # Current position size (number of shares). 0.0 if flat.
    pos = float(c.position_now) if c.position_now is not None else 0.0

    # Protect against index errors / invalid values
    n = len(close)
    if i < 0 or i >= n:
        return vbt.portfolio.enums.NoOrder

    # Read current values safely
    price = float(close[i]) if np.isfinite(close[i]) else np.nan
    high_price = float(high[i]) if np.isfinite(high[i]) else np.nan
    macd_i = float(macd[i]) if np.isfinite(macd[i]) else np.nan
    signal_i = float(signal[i]) if np.isfinite(signal[i]) else np.nan
    atr_i = float(atr[i]) if np.isfinite(atr[i]) else np.nan
    sma_i = float(sma[i]) if np.isfinite(sma[i]) else np.nan

    # Helper to get previous bar values for cross detection
    if i > 0:
        macd_prev = macd[i - 1] if np.isfinite(macd[i - 1]) else np.nan
        signal_prev = signal[i - 1] if np.isfinite(signal[i - 1]) else np.nan
    else:
        macd_prev = np.nan
        signal_prev = np.nan

    # Use module-level state and also store on function object for test introspection.
    global _ORDER_STATE
    # Initialize attributes on first use if missing
    if not hasattr(order_func, "_in_position"):
        order_func._in_position = _ORDER_STATE.in_position
        order_func._highest = _ORDER_STATE.highest
        order_func._entry_idx = _ORDER_STATE.entry_idx

    # If flat (no position), check for entry
    if pos == 0.0:
        # Ensure we mark state as flat
        order_func._in_position = False
        order_func._highest = None
        order_func._entry_idx = None
        _ORDER_STATE.in_position = False
        _ORDER_STATE.highest = None
        _ORDER_STATE.entry_idx = None

        # Entry condition: MACD crosses above signal AND price > SMA
        macd_cross_up = False
        if i > 0 and np.isfinite(macd_prev) and np.isfinite(signal_prev) and np.isfinite(macd_i) and np.isfinite(signal_i):
            # previous MACD <= previous Signal and current MACD > current Signal
            if (macd_prev <= signal_prev) and (macd_i > signal_i):
                macd_cross_up = True

        price_above_sma = False
        if np.isfinite(price) and np.isfinite(sma_i):
            price_above_sma = price > sma_i

        if macd_cross_up and price_above_sma:
            # Enter long with 50% of equity as a market order
            try:
                return vbt.portfolio.nb.order_nb(size=0.5, size_type=2, direction=1, price=np.inf)
            except Exception:
                # If nb.order_nb signature is different, fall back to tuple that vectorbt may accept
                return (0.5, 2, 1)

        # Otherwise no action
        try:
            return vbt.portfolio.enums.NoOrder
        except Exception:
            return (np.nan, 0, 0)

    # If we reach here, we have a long position (pos > 0)
    # Initialize tracking on the first bar we detect a position
    if not order_func._in_position:
        order_func._in_position = True
        order_func._entry_idx = i
        # Start highest since entry with the current high if available, else current price
        if np.isfinite(high_price):
            order_func._highest = high_price
        elif np.isfinite(price):
            order_func._highest = price
        else:
            order_func._highest = None

        _ORDER_STATE.in_position = True
        _ORDER_STATE.entry_idx = order_func._entry_idx
        _ORDER_STATE.highest = order_func._highest

    else:
        # Update highest_since_entry if we have a valid high price
        if np.isfinite(high_price):
            if order_func._highest is None:
                order_func._highest = high_price
            else:
                # Use numpy.maximum semantics
                order_func._highest = float(np.maximum(order_func._highest, high_price))

            _ORDER_STATE.highest = order_func._highest

    # Exit checks
    # 1) MACD bearish cross: previous MACD >= previous Signal and current MACD < current Signal
    macd_cross_down = False
    if i > 0 and np.isfinite(macd_prev) and np.isfinite(signal_prev) and np.isfinite(macd_i) and np.isfinite(signal_i):
        if (macd_prev >= signal_prev) and (macd_i < signal_i):
            macd_cross_down = True

    # 2) Trailing stop: price falls below (highest_since_entry - trailing_mult * ATR)
    trailing_stop_trigger = False
    if order_func._highest is not None and np.isfinite(atr_i) and np.isfinite(price):
        trailing_level = order_func._highest - (trailing_mult * atr_i)
        if price < trailing_level:
            trailing_stop_trigger = True

    if macd_cross_down or trailing_stop_trigger:
        # Reset tracking state anticipating the position will be closed
        order_func._in_position = False
        order_func._highest = None
        order_func._entry_idx = None

        _ORDER_STATE.in_position = False
        _ORDER_STATE.highest = None
        _ORDER_STATE.entry_idx = None

        # Close entire long position as a market order
        try:
            return vbt.portfolio.nb.order_nb(size=-np.inf, size_type=2, direction=1, price=np.inf)
        except Exception:
            return (-np.inf, 2, 1)

    # Otherwise hold position
    try:
        return vbt.portfolio.enums.NoOrder
    except Exception:
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
    Precompute indicators required by the strategy using vectorbt indicator classes.

    Returns a dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    All values are numpy arrays aligned with the input ohlcv index.
    """
    # Basic validation
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")
    if "high" not in ohlcv.columns:
        raise ValueError("ohlcv DataFrame must contain 'high' column")

    # Prepare series (use provided low if available, otherwise fall back to close)
    close_s = ohlcv["close"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float) if "low" in ohlcv.columns else ohlcv["close"].astype(float)

    # Reset order state before each run to avoid leaking state between backtests/tests
    _reset_order_state()

    # Compute MACD
    macd_ind = vbt.MACD.run(close_s, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_line = macd_ind.macd
    signal_line = macd_ind.signal

    # Compute ATR
    atr_ind = vbt.ATR.run(high_s, low_s, close_s, window=atr_period)
    atr_line = atr_ind.atr

    # Compute SMA (trend filter)
    sma_ind = vbt.MA.run(close_s, window=sma_period)
    sma_line = sma_ind.ma

    # Convert to numpy arrays; ensure shapes match input
    close_arr = close_s.values
    high_arr = high_s.values
    macd_arr = macd_line.values
    signal_arr = signal_line.values
    atr_arr = atr_line.values
    sma_arr = sma_line.values

    # Sanity check lengths
    length = len(ohlcv)
    arrays = [close_arr, high_arr, macd_arr, signal_arr, atr_arr, sma_arr]
    for arr in arrays:
        if len(arr) != length:
            raise ValueError("Indicator arrays must match input length")

    return {
        "close": close_arr,
        "high": high_arr,
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }