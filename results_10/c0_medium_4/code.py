"""
MACD + ATR Trailing Stop Strategy

Exports:
- compute_indicators(ohlcv, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50, atr_period=14)
- order_func(c, close, high, macd, signal, atr, sma, trailing_mult)

Notes:
- Long-only single-asset strategy
- Trailing stop is highest price since entry minus trailing_mult * ATR
- No numba usage

This module is intended to be used with the provided `run_backtest` harness.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


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

    Returns a dict containing numpy arrays for the following keys:
    - close: close prices
    - high: high prices
    - macd: MACD line
    - signal: MACD signal line
    - atr: Average True Range (Wilder's EMA)
    - sma: Simple moving average of close over sma_period

    All returned arrays are the same length as the input DataFrame index.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    # Helper to fetch column case-insensitively
    def _get_col(df: pd.DataFrame, name: str) -> pd.Series:
        if name in df.columns:
            return df[name].astype(float)
        # try case-insensitive match
        lower_cols = {c.lower(): c for c in df.columns}
        if name.lower() in lower_cols:
            return df[lower_cols[name.lower()]].astype(float)
        raise KeyError(f"Column '{name}' not found in ohlcv DataFrame")

    close_s = _get_col(ohlcv, "close").copy()
    high_s = _get_col(ohlcv, "high").copy()
    low_s = _get_col(ohlcv, "low").copy()

    # MACD: EMA(fast) - EMA(slow)
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

    # ATR (Wilder's smoothing): TR then EMA with alpha=1/atr_period
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    # SMA
    sma = close_s.rolling(window=sma_period, min_periods=1).mean()

    # Return numpy arrays (preserve NaNs where applicable)
    return {
        "close": close_s.values,
        "high": high_s.values,
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
    Order function for vbt.Portfolio.from_order_func (use_numba=False).

    Returns a simple tuple (size, size_type, direction):
    - size: float (np.nan for no action)
    - size_type: int (interpreted by the wrapper / vbt)
    - direction: int (interpreted by the wrapper / vbt)

    This implementation uses absolute asset amounts (size_type=0) to avoid
    ambiguity with percent/target enums. It tries to read available cash from
    the context `c` when possible; otherwise it falls back to a default initial
    capital of 100000.0 to compute how many units to buy on entry.
    """
    # Initialize persistent state on the function object
    if not hasattr(order_func, "_state"):
        order_func._state = {
            "in_pos": False,
            "entry_idx": None,
            "max_high": -np.inf,
            "units": 0,
        }

    state = order_func._state

    # Safe access to index
    try:
        i = int(c.i)
    except Exception:
        return (float("nan"), 0, 0)

    # Bounds check
    n = len(close)
    if i < 0 or i >= n:
        return (float("nan"), 0, 0)

    # Helper guards for NaNs
    def _is_valid_idx(arr: np.ndarray, idx: int) -> bool:
        try:
            return not np.isnan(arr[idx])
        except Exception:
            return False

    # Detect MACD crosses (use previous bar and current bar)
    cross_up = False
    cross_down = False
    if i > 0 and _is_valid_idx(macd, i) and _is_valid_idx(signal, i) and _is_valid_idx(macd, i - 1) and _is_valid_idx(signal, i - 1):
        cross_up = (macd[i - 1] < signal[i - 1]) and (macd[i] > signal[i])
        cross_down = (macd[i - 1] > signal[i - 1]) and (macd[i] < signal[i])

    price_above_sma = False
    if _is_valid_idx(close, i) and _is_valid_idx(sma, i):
        price_above_sma = close[i] > sma[i]

    current_high = float(high[i]) if _is_valid_idx(high, i) else np.nan
    current_close = float(close[i]) if _is_valid_idx(close, i) else np.nan

    # Helper: try to infer available cash from context
    cash_avail = None
    for attr in ("cash", "cash_value", "balance", "wallet", "value"):
        if hasattr(c, attr):
            try:
                val = getattr(c, attr)
                # Try to coerce to float (handles numpy types)
                cash_avail = float(val)
                break
            except Exception:
                continue
    # Try portfolio init_cash if available
    if cash_avail is None and hasattr(c, "portfolio"):
        try:
            cash_avail = float(getattr(c.portfolio, "init_cash", np.nan))
        except Exception:
            cash_avail = None
    # Final fallback
    if cash_avail is None or np.isnan(cash_avail):
        cash_avail = 100000.0

    # ----- ENTRY -----
    if not state["in_pos"]:
        if cross_up and price_above_sma and not np.isnan(current_close) and current_close > 0:
            # Compute number of units to buy (use whole capital approximation)
            units = max(1, int(np.floor(cash_avail / current_close)))
            state["in_pos"] = True
            state["entry_idx"] = i
            state["max_high"] = current_high if not np.isnan(current_high) else -np.inf
            state["units"] = units
            # size_type=0 -> Amount, direction=0 -> Both (buy/sell allowed)
            return (float(units), 0, 0)
        return (float("nan"), 0, 0)

    # ----- POSITION MANAGEMENT / EXIT -----
    # Update highest high since entry
    if not np.isnan(current_high) and current_high > state["max_high"]:
        state["max_high"] = current_high

    # Compute trailing stop level if possible
    trailing_level = np.nan
    if state["max_high"] != -np.inf and _is_valid_idx(atr, i) and not np.isnan(atr[i]):
        trailing_level = state["max_high"] - float(trailing_mult) * float(atr[i])

    # Exit on MACD cross down
    if cross_down:
        units = state.get("units", 0) or 0
        state["in_pos"] = False
        state["entry_idx"] = None
        state["max_high"] = -np.inf
        state["units"] = 0
        # Sell the units we previously bought: negative amount to reduce position
        return (-float(units), 0, 0)

    # Exit on trailing stop hit
    if not np.isnan(trailing_level) and _is_valid_idx(close, i) and current_close < trailing_level:
        units = state.get("units", 0) or 0
        state["in_pos"] = False
        state["entry_idx"] = None
        state["max_high"] = -np.inf
        state["units"] = 0
        return (-float(units), 0, 0)

    # Otherwise no action
    return (float("nan"), 0, 0)
