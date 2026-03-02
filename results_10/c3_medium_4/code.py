import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy.

    Returns a dictionary with numpy arrays for keys:
    - 'close', 'high', 'macd', 'signal', 'atr', 'sma'

    All computations are causal (no lookahead): EMAs, rolling SMA, ATR.
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")

    # Require core columns
    required_cols = {"close", "high", "low"}
    missing = required_cols - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing required columns: {missing}")

    close = ohlcv["close"].astype(float).copy()
    high = ohlcv["high"].astype(float).copy()
    low = ohlcv["low"].astype(float).copy()

    # MACD (EMA fast - EMA slow) and signal line (EMA of MACD)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA trend filter
    # Use min_periods = sma_period to produce NaN until the SMA window is full
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # ATR (Wilder's moving average via ewm with alpha=1/period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    return {
        "close": close.values,
        "high": high.values,
        "macd": macd.values,
        "signal": signal.values,
        "atr": atr.values,
        "sma": sma.values,
    }


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
    Order function compatible with vectorbt.Portfolio.from_order_func (use_numba=False).

    Strategy logic (long-only):
    - Entry: MACD crosses above Signal AND price > 50-period SMA
    - Exit: MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Implementation notes:
    - This function simulates the strategy from the start up to the current index c.i using only
      past data (no lookahead). Based on that simulation it determines whether an order
      (entry/exit) should be placed on the current bar.
    - Returns a tuple (size, size_type, direction). When no action, returns (np.nan, 0, 0)
      which the backtest wrapper interprets as no order.

    We avoid external state to ensure determinism and no lookahead.
    """
    # Convert inputs to numpy arrays and ensure float dtype
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    # Extract current index from context
    i = getattr(c, "i", None)
    if i is None:
        # No context provided -> no action
        return (np.nan, 0, 0)
    i = int(i)

    n = len(close)
    if n == 0 or i < 0 or i >= n:
        return (np.nan, 0, 0)

    # Simulate strategy up to and including index i using only past data
    pos = False
    entry_index = None
    highest = np.nan
    order_to_place = None  # 'entry' or 'exit' at the current bar

    # Start from 1 because cross detection requires previous bar
    for j in range(1, i + 1):
        macd_j, macd_j1 = macd[j], macd[j - 1]
        sig_j, sig_j1 = signal[j], signal[j - 1]
        sma_j = sma[j]
        close_j = close[j]
        high_j = high[j]
        atr_j = atr[j]

        # Bullish MACD cross at j
        bullish_cross = False
        if (
            not np.isnan(macd_j)
            and not np.isnan(macd_j1)
            and not np.isnan(sig_j)
            and not np.isnan(sig_j1)
        ):
            if (macd_j1 <= sig_j1) and (macd_j > sig_j):
                bullish_cross = True

        price_above_sma = False
        if (not np.isnan(sma_j)) and (not np.isnan(close_j)):
            price_above_sma = close_j > sma_j

        if (not pos) and bullish_cross and price_above_sma:
            # Enter long
            pos = True
            entry_index = j
            highest = high_j if not np.isnan(high_j) else np.nan
            if j == i:
                order_to_place = "entry"

        if pos:
            # Update highest high since entry
            if not np.isnan(high_j):
                if np.isnan(highest):
                    highest = high_j
                else:
                    highest = max(highest, high_j)

            # Bearish MACD cross at j
            bearish_cross = False
            if (
                not np.isnan(macd_j)
                and not np.isnan(macd_j1)
                and not np.isnan(sig_j)
                and not np.isnan(sig_j1)
            ):
                if (macd_j1 >= sig_j1) and (macd_j < sig_j):
                    bearish_cross = True

            # Trailing stop breach at j
            trailing_breach = False
            if (not np.isnan(highest)) and (not np.isnan(atr_j)) and (not np.isnan(close_j)):
                threshold = highest - trailing_mult * atr_j
                if close_j < threshold:
                    trailing_breach = True

            if bearish_cross or trailing_breach:
                pos = False
                if j == i:
                    order_to_place = "exit"

    # Map actions to (size, size_type, direction)
    # We use the following integer codes (compatible with vectorbt):
    #  - size_type = 0 : Amount
    #  - direction = 1 : Buy / Long
    #  - direction = 2 : Sell / Short (used to close long)
    if order_to_place == "entry":
        return (1.0, 0, 1)
    elif order_to_place == "exit":
        return (1.0, 0, 2)
    else:
        return (np.nan, 0, 0)
