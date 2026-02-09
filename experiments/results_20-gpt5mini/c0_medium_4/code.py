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

    Returns a dict containing numpy arrays for keys:
    - 'close', 'high', 'macd', 'signal', 'atr', 'sma'

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', ...]
        macd_fast, macd_slow, macd_signal: MACD parameters
        sma_period: period for the trend SMA
        atr_period: period for ATR

    Notes:
        - Uses EMA for MACD and signal line (pandas ewm)
        - ATR is computed as simple moving average of True Range (can have NaNs in warmup)
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    for col in ["high", "low", "close"]:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv missing required column: {col}")

    # Ensure floats
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)

    # MACD (EMA fast - EMA slow) and Signal (EMA of MACD)
    # Use adjust=False for standard EMA behaviour
    fast_ema = close.ewm(span=macd_fast, adjust=False).mean()
    slow_ema = close.ewm(span=macd_slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # 50-period SMA trend filter
    sma = close.rolling(window=sma_period, min_periods=1).mean()

    # ATR (True Range then moving average)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

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
    Order function for vectorbt.Portfolio.from_order_func (use_numba=False).

    Returns a tuple: (size, size_type, direction)
      - size: positive for buy, negative for sell, np.nan for no action
      - size_type: integer code (0 used for Amount)
      - direction: integer code (0 used for Both)

    Logic:
      - Enter long when MACD crosses above Signal AND price > SMA(50)
      - Exit when MACD crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)
      - Trailing stop is based on highest high since entry minus trailing_mult * ATR

    Important:
      - This function keeps minimal internal state on the function object (in_position, entry_high)
      - It avoids using numba or vectorbt.nb functions per instructions
    """
    # Initialize persistent state on the function object
    if not hasattr(order_func, "in_position"):
        order_func.in_position = False  # type: ignore
        order_func.entry_high = -np.inf  # type: ignore
        order_func.entry_index = None  # type: ignore

    i = int(getattr(c, "i", 0))

    # Safe guards: if index out of bounds, do nothing
    if i < 0 or i >= len(close):
        return (np.nan, 0, 0)

    # Read current and previous values
    macd_curr = float(macd[i]) if not np.isnan(macd[i]) else np.nan
    signal_curr = float(signal[i]) if not np.isnan(signal[i]) else np.nan
    sma_curr = float(sma[i]) if not np.isnan(sma[i]) else np.nan
    close_curr = float(close[i]) if not np.isnan(close[i]) else np.nan
    high_curr = float(high[i]) if not np.isnan(high[i]) else np.nan
    atr_curr = float(atr[i]) if not np.isnan(atr[i]) else np.nan

    prev_macd = float(macd[i - 1]) if i > 0 and not np.isnan(macd[i - 1]) else np.nan
    prev_signal = float(signal[i - 1]) if i > 0 and not np.isnan(signal[i - 1]) else np.nan

    # Determine crossover events (require non-NaN values)
    macd_cross_up = False
    macd_cross_down = False
    if not (np.isnan(prev_macd) or np.isnan(prev_signal) or np.isnan(macd_curr) or np.isnan(signal_curr)):
        macd_cross_up = (prev_macd <= prev_signal) and (macd_curr > signal_curr)
        macd_cross_down = (prev_macd >= prev_signal) and (macd_curr < signal_curr)

    # No action placeholder: (np.nan, size_type=0 (Amount), direction=0 (Both))
    NO_ORDER: Tuple[float, int, int] = (np.nan, 0, 0)

    # ENTRY: Only consider entering if not already in a position
    if not order_func.in_position:
        # require MACD crossover up and price above SMA
        if macd_cross_up and (not np.isnan(sma_curr)) and (not np.isnan(close_curr)) and (close_curr > sma_curr):
            # Enter long with 1 unit (amount). Using size_type=0 (Amount) and direction=0 (Both)
            order_func.in_position = True  # type: ignore
            order_func.entry_high = high_curr if not np.isnan(high_curr) else -np.inf  # type: ignore
            order_func.entry_index = i  # type: ignore
            return (1.0, 0, 0)
        return NO_ORDER

    # If we are in a position, update highest high since entry
    if not np.isnan(high_curr) and high_curr > order_func.entry_high:
        order_func.entry_high = high_curr  # type: ignore

    # Evaluate trailing stop condition if ATR is available
    trailing_trigger = False
    if (not np.isnan(order_func.entry_high)) and (not np.isnan(atr_curr)):
        stop_price = order_func.entry_high - float(trailing_mult) * atr_curr  # type: ignore
        if not np.isnan(close_curr) and close_curr < stop_price:
            trailing_trigger = True

    # EXIT conditions: MACD cross down OR trailing stop hit
    if macd_cross_down or trailing_trigger:
        # Exit full position by selling 1 unit (match entry size)
        order_func.in_position = False  # type: ignore
        order_func.entry_high = -np.inf  # type: ignore
        order_func.entry_index = None  # type: ignore
        return (-1.0, 0, 0)

    # Otherwise, hold
    return NO_ORDER
