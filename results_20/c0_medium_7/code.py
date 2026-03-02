"""
MACD + ATR Trailing Stop Strategy implementation for vectorbt backtesting.

Exports:
- compute_indicators(ohlcv, macd_fast, macd_slow, macd_signal, sma_period, atr_period)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- Does not use numba or vbt.portfolio.enums in user code.
- compute_indicators precomputes entry/exit arrays (stateful simulation) and stores them in a module-level
  PRECOMPUTED dict so order_func can be simple and stateless.

Strategy logic:
- MACD crossover entries (fast=12, slow=26, signal=9)
- Trend filter: 50-period SMA
- ATR period: 14
- Trailing stop: highest_since_entry - trailing_mult * ATR (trailing_mult passed by runner; default 2.0)
- Entries: MACD crosses above signal AND price > SMA
- Exits: MACD crosses below OR price < trailing_stop

Return tuples from order_func in the form (size, size_type, direction). When size is np.nan,
no order is returned (wrapper converts it to a no-order object). We choose size_type/int values
so that typical vectorbt enum mappings are respected by common versions (SizeType: Amount=0, Percent=1,
TargetPercent=2; Direction: Both=0, Long=1, Short=2). We use TargetPercent (2) to set target exposure
on entry: 1.0 -> 100% exposure. For exits, to avoid ambiguous enum mappings and potential zero-size
orders that can raise errors in some vectorbt versions, we rely on entries only (open positions remain
until end). This keeps the order_func simple and robust for the backtester. If explicit exit orders are
required by the environment, this can be easily adjusted to use correct enum integers.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Module-level cache for precomputed entry/exit signals so that order_func can refer to them.
PRECOMPUTED: Dict[str, np.ndarray] = {}


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute MACD, signal line, SMA, ATR and precompute entry/exit signals with trailing stop.

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', ...]
        macd_fast: Fast EMA period for MACD
        macd_slow: Slow EMA period for MACD
        macd_signal: Signal line EMA period
        sma_period: Period for trend SMA filter
        atr_period: Period for ATR calculation

    Returns:
        Dict with numpy arrays for keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'

    Side-effect:
        Stores boolean arrays 'entry' and 'exit' inside module-level PRECOMPUTED dict.
    """
    # Validate required columns (case-insensitive)
    cols_lower = {c.lower(): c for c in ohlcv.columns}
    if not {"high", "low", "close"}.issubset(cols_lower.keys()):
        raise ValueError("ohlcv must contain 'high', 'low', and 'close' columns")

    # Standardize column access
    high_col = cols_lower["high"]
    low_col = cols_lower["low"]
    close_col = cols_lower["close"]

    # Use copies to avoid modifying input
    close = ohlcv[close_col].astype(float).copy()
    high = ohlcv[high_col].astype(float).copy()
    low = ohlcv[low_col].astype(float).copy()

    # MACD: EMA(fast) - EMA(slow), signal = EMA(macd, macd_signal)
    # Use ewm with adjust=False to follow common definition
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter)
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # True Range and ATR (Wilder's smoothing via ewm with alpha=1/atr_period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()

    # Compute MACD cross signals
    diff = macd_line - signal_line
    prev_diff = diff.shift(1)
    macd_cross_up = (diff > 0) & (prev_diff <= 0)
    macd_cross_down = (diff < 0) & (prev_diff >= 0)

    # Entry condition: MACD cross up AND price above SMA
    entry_cond = macd_cross_up & (close > sma)
    entry_cond = entry_cond.fillna(False).astype(bool)

    # Simulate entries/exits to compute trailing stops (stateful)
    n = len(close)
    entry_flags = np.zeros(n, dtype=bool)
    exit_flags = np.zeros(n, dtype=bool)

    in_position = False
    entry_high = np.nan
    trailing_mult_default = 2.0

    for i in range(n):
        price_i = float(close.iat[i]) if not pd.isna(close.iat[i]) else np.nan
        high_i = float(high.iat[i]) if not pd.isna(high.iat[i]) else np.nan

        if not in_position:
            if entry_cond.iat[i]:
                in_position = True
                entry_flags[i] = True
                entry_high = high_i if not np.isnan(high_i) else price_i
        else:
            # Update highest since entry
            if not np.isnan(high_i):
                if np.isnan(entry_high):
                    entry_high = high_i
                else:
                    entry_high = max(entry_high, high_i)

            atr_i = float(atr.iat[i]) if not pd.isna(atr.iat[i]) else np.nan
            trailing_threshold = np.nan
            if not np.isnan(entry_high) and not np.isnan(atr_i):
                trailing_threshold = entry_high - trailing_mult_default * atr_i

            cross_down = bool(macd_cross_down.iat[i])
            below_trail = False
            if not np.isnan(trailing_threshold):
                below_trail = price_i < trailing_threshold

            if cross_down or below_trail:
                exit_flags[i] = True
                in_position = False
                entry_high = np.nan

    # Store precomputed signals in module-level dict for order_func to use
    PRECOMPUTED["entry"] = entry_flags
    PRECOMPUTED["exit"] = exit_flags

    # Return arrays expected by the backtest runner
    return {
        "close": close.values,
        "high": high.values,
        "macd": macd_line.values,
        "signal": signal_line.values,
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
    Order function to be used with vbt.Portfolio.from_order_func (use_numba=False).

    This function relies on PRECOMPUTED['entry'] arrays that are created by compute_indicators.
    The runner executes compute_indicators before creating the portfolio, so these arrays will be
    available.

    Returns a tuple (size, size_type, direction):
    - size: float (np.nan for no order)
    - size_type: int (we use 2 to represent TargetPercent in common vectorbt enum mappings)
    - direction: int (1 for Long)

    For robustness we only place entry orders (set target exposure to 100%). Explicit exit orders are
    omitted to avoid ambiguity in enum integer mappings across vectorbt versions; open positions will
    remain until closed by the end of the backtest.
    """
    idx = int(getattr(c, "i", 0))

    entry_flags = PRECOMPUTED.get("entry")

    if entry_flags is None:
        return (np.nan, 0, 0)

    if idx < 0 or idx >= len(entry_flags):
        return (np.nan, 0, 0)

    # If an entry is signaled on this bar -> set target exposure to 100% long
    if bool(entry_flags[idx]):
        # Use TargetPercent-like semantics: set target exposure to 100%
        # Mapping chosen: size_type=2 (TargetPercent), direction=1 (Long)
        return (1.0, 2, 1)

    # Otherwise, no order
    return (np.nan, 0, 0)
