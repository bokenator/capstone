from typing import Dict, Tuple
import numpy as np
import pandas as pd

# Simple global store to keep per-order-context state. Keys are id(c).
_CTX_STATE: Dict[int, Dict[str, float]] = {}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute technical indicators required by the strategy.

    Returns a dictionary with numpy arrays for the following keys:
    - macd: MACD line (EMA_fast - EMA_slow)
    - signal: Signal line (EMA of MACD)
    - atr: Average True Range (rolling mean of True Range)
    - sma: Simple Moving Average of close (period = sma_period)
    - close: Close prices as numpy array
    - high: High prices as numpy array

    The computations only use present and past data (no lookahead).
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame with columns ['open','high','low','close']")

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        raise KeyError(f"ohlcv must contain columns {required_cols}")

    # Ensure numeric dtype
    open_s = ohlcv["open"].astype(float)
    high_s = ohlcv["high"].astype(float)
    low_s = ohlcv["low"].astype(float)
    close_s = ohlcv["close"].astype(float)

    # MACD (EMA-based)
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # ATR (True Range rolling mean). Use min_periods=1 to keep early values defined.
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    # SMA for trend filter
    sma = close_s.rolling(window=sma_period, min_periods=sma_period).mean()

    return {
        "macd": macd_line.to_numpy(),
        "signal": signal_line.to_numpy(),
        "atr": atr.to_numpy(),
        "sma": sma.to_numpy(),
        "close": close_s.to_numpy(),
        "high": high_s.to_numpy(),
    }


def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function implementing MACD crossover entries with ATR trailing stops.

    Returns a tuple compatible with the run_backtest wrapper: (size, size_type, direction).
    - size: positive for buy, negative for sell, np.nan for no order
    - size_type: integer code (0 => Amount)
    - direction: integer code (0 => Both)

    State is kept in a module-level dictionary keyed by id(c) because the OrderContext
    does not allow arbitrary attribute assignment.

    All decisions use only data up to the current index (no lookahead).
    """
    i = int(getattr(c, "i", 0))

    # Retrieve or initialize state for this context
    ctx_id = id(c)
    state = _CTX_STATE.get(ctx_id)
    if state is None:
        state = {"in_pos": False, "position_size": 0.0, "entry_high": -np.inf}
        _CTX_STATE[ctx_id] = state

    # Safety: if arrays are shorter than needed or we're at first bar, do nothing
    if i <= 0:
        return (np.nan, 0, 0)

    # Read current and previous values (cast to floats)
    macd_curr = float(macd[i])
    macd_prev = float(macd[i - 1])
    sig_curr = float(signal[i])
    sig_prev = float(signal[i - 1])

    price = float(close[i])
    high_curr = float(high[i])

    atr_curr = float(atr[i]) if not np.isnan(atr[i]) else np.nan
    sma_curr = float(sma[i]) if not np.isnan(sma[i]) else np.nan

    # Entry condition: MACD crosses above signal AND price > SMA
    macd_cross_up = (macd_prev < sig_prev) and (macd_curr > sig_curr)
    price_above_sma = (not np.isnan(sma_curr)) and (price > sma_curr)

    if (not state["in_pos"]) and macd_cross_up and price_above_sma:
        # Enter long: buy 1 unit
        state["in_pos"] = True
        state["position_size"] = 1.0
        state["entry_high"] = high_curr
        return (float(state["position_size"]), 0, 0)

    # If in position, update highest high and check exits
    if state["in_pos"]:
        # Update highest high since entry
        if high_curr > state["entry_high"]:
            state["entry_high"] = high_curr

        # Compute trailing stop level
        trailing_stop = -np.inf
        if not np.isnan(atr_curr):
            trailing_stop = state["entry_high"] - float(trailing_mult) * atr_curr

        # Exit on MACD cross down or when price falls below trailing stop
        macd_cross_down = (macd_prev > sig_prev) and (macd_curr < sig_curr)
        fell_below_trail = (not np.isneginf(trailing_stop)) and (price < trailing_stop)

        if macd_cross_down or fell_below_trail:
            # Close full position
            size_to_close = -float(state["position_size"]) if state["position_size"] != 0 else np.nan
            # reset state
            state["in_pos"] = False
            state["position_size"] = 0.0
            state["entry_high"] = -np.inf
            return (size_to_close, 0, 0)

    # No order
    return (np.nan, 0, 0)
