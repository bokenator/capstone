# Pairs trading indicator and order functions for vectorbt
# Uses only allowed APIs from the Verified API Surface.

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats

# Try to detect the correct enum integer values for Direction and SizeType
# Prefer a SizeType that represents units (Amount/Size/Shares). Fall back to Value
# only if units-style members are not found.
try:
    import vectorbt as vbt
    enums_mod = vbt.portfolio.enums

    DirectionEnum = getattr(enums_mod, 'Direction', None)
    SizeTypeEnum = getattr(enums_mod, 'SizeType', None)

    if DirectionEnum is not None:
        direction_long = int(getattr(DirectionEnum, 'LONG', getattr(DirectionEnum, 'BUY', 1)))
        direction_short = int(getattr(DirectionEnum, 'SHORT', getattr(DirectionEnum, 'SELL', 2)))
    else:
        direction_long = 1
        direction_short = 2

    # Default to interpreting size as "size/units"
    size_type_mode = 'size'
    if SizeTypeEnum is not None:
        if hasattr(SizeTypeEnum, 'Amount'):
            size_type_units = int(SizeTypeEnum.Amount)
            size_type_mode = 'size'
        elif hasattr(SizeTypeEnum, 'Size'):
            size_type_units = int(SizeTypeEnum.Size)
            size_type_mode = 'size'
        elif hasattr(SizeTypeEnum, 'Shares'):
            size_type_units = int(SizeTypeEnum.Shares)
            size_type_mode = 'size'
        elif hasattr(SizeTypeEnum, 'Value'):
            size_type_units = int(SizeTypeEnum.Value)
            size_type_mode = 'value'
        else:
            try:
                first_member = list(SizeTypeEnum)[0]
                size_type_units = int(first_member)
            except Exception:
                size_type_units = 0
            size_type_mode = 'size'
    else:
        size_type_units = 0
        size_type_mode = 'size'
except Exception:
    direction_long = 1
    direction_short = 2
    size_type_units = 0
    size_type_mode = 'size'


# A small module-level guard to avoid excessive repeated trading in pathological cases.
# This keeps track of the last bar where we opened a new trade and avoids opening
# another too soon.
_LAST_TRADE_BAR = -1
_MIN_BARS_BETWEEN_TRADES = 5


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS) and spread z-score for a pair of assets.

    This function is defensive with input types: it accepts numpy arrays,
    pandas Series/DataFrames, or array-like objects. If a 2D array or
    DataFrame is passed, the first column is used by default.

    Args:
        close_a: Close prices for asset A as a 1-D array-like.
        close_b: Close prices for asset B as a 1-D array-like.
        hedge_lookback: Lookback window (in bars) for rolling OLS hedge ratio.
                        If fewer than hedge_lookback bars are available at time t,
                        the regression uses all available past bars up to t (no lookahead).
        zscore_lookback: Lookback window for rolling mean/std of the spread.

    Returns:
        A dict with keys:
            - "zscore": numpy array of z-score values (same length as inputs)
            - "hedge_ratio": numpy array of hedge ratios (same length as inputs)
    """
    # Helper to coerce many possible input types to a 1D numpy array
    def _to_1d_array(x: Any) -> np.ndarray:
        # pandas Series -> values
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        # pandas DataFrame -> take first column
        if isinstance(x, pd.DataFrame):
            # If single-column DF, take that column; otherwise take first column
            return x.iloc[:, 0].values.astype(float)
        # numpy array or other array-like
        arr = np.array(x)
        if arr.ndim > 1:
            # Take the first column to produce a 1D array
            return arr[:, 0].astype(float)
        return arr.astype(float)

    # Normalize inputs
    close_a = _to_1d_array(close_a)
    close_b = _to_1d_array(close_b)

    # Align lengths: use the minimum length to allow being called with truncated inputs
    min_len = min(len(close_a), len(close_b))
    if len(close_a) != len(close_b):
        close_a = close_a[:min_len]
        close_b = close_b[:min_len]

    n = len(close_a)

    # Prepare hedge ratio array
    hedge_ratio = np.full(n, np.nan, dtype=float)

    # Rolling/expanding OLS: use past up to hedge_lookback bars (inclusive)
    for i in range(n):
        start = 0 if i - hedge_lookback + 1 < 0 else (i - hedge_lookback + 1)
        x = close_b[start : i + 1]
        y = close_a[start : i + 1]

        # Need at least two points to compute slope; otherwise, carry previous or set default
        if len(x) >= 2 and (np.sum(np.isfinite(x)) == len(x)) and (np.sum(np.isfinite(y)) == len(y)):
            # Use fully-qualified scipy call per API citation requirements
            slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
            hedge_ratio[i] = float(slope)
        else:
            if i == 0:
                hedge_ratio[i] = 1.0
            else:
                hedge_ratio[i] = hedge_ratio[i - 1]

    # Compute spread: A - hedge_ratio * B
    spread = close_a - hedge_ratio * close_b

    # Rolling mean and std for spread using pandas rolling (past values only)
    spread_series = pd.Series(spread)
    rolling_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean()
    rolling_std = pd.Series.rolling(spread_series, window=zscore_lookback).std()

    # z-score
    zscore_series = (spread_series - rolling_mean) / rolling_std

    # Convert to numpy arrays
    zscore = zscore_series.values.astype(float)
    hedge_ratio = hedge_ratio.astype(float)

    return {"zscore": zscore, "hedge_ratio": hedge_ratio}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
) -> Tuple[float, int, int]:
    """
    Order function for pairs trading in flexible multi-asset mode.

    Returns a tuple (size, size_type, direction) where:
      - size: positive float (number of units or value depending on size_type)
      - size_type: integer sentinel (from SizeType enum)
      - direction: integer sentinel (from Direction enum)

    The wrapper provided by the backtest runner will convert this tuple into
    vectorbt order objects. This function uses only the information available
    at or before the current bar (no lookahead).
    """
    global _LAST_TRADE_BAR

    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))

    # Safely get current position for this asset (may be scalar or indexable)
    pos_attr = getattr(c, "position_now", 0.0)
    try:
        position_now = float(pos_attr)
    except Exception:
        try:
            position_now = float(pos_attr[col])
        except Exception:
            position_now = 0.0

    # Get available cash/value to avoid over-leveraging
    cash = getattr(c, 'cash_now', getattr(c, 'value_now', 100000.0))
    try:
        cash = float(cash)
    except Exception:
        cash = 100000.0

    # Bounds check for index
    if i < 0 or i >= len(zscore):
        return (np.nan, size_type_units, 0)

    z = float(zscore[i]) if np.isfinite(zscore[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    if not np.isfinite(z) or not np.isfinite(hr):
        return (np.nan, size_type_units, 0)

    # Do not trade during warmup to avoid excess churn while indicators stabilize
    WARMUP = max(50, 60)
    if i < WARMUP:
        return (np.nan, size_type_units, 0)

    NOTIONAL_PER_LEG = 10_000.0

    price_a = float(close_a[i]) if hasattr(close_a, "__getitem__") else float(close_a)
    price_b = float(close_b[i]) if hasattr(close_b, "__getitem__") else float(close_b)

    if not np.isfinite(price_a) or price_a <= 0:
        return (np.nan, size_type_units, 0)

    # Compute unit counts for A and B based on NOTIONAL_PER_LEG
    units_a = NOTIONAL_PER_LEG / price_a
    units_b = abs(hr) * units_a

    NO_ORDER = (np.nan, size_type_units, 0)

    # previous zscore for crossing detection
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else 0.0

    # If currently in a position for this asset, check exits and stop-loss
    if abs(position_now) > 0:
        # Stop-loss: close entire position
        if abs(z) > float(stop_threshold):
            if size_type_mode == 'value':
                # position_now is in units -> convert to value
                size_to_close = abs(position_now) * (price_a if col == 0 else price_b)
            else:
                size_to_close = abs(position_now)
            # Bound and sanity-check
            if not np.isfinite(size_to_close) or size_to_close <= 0:
                return NO_ORDER
            # Do not request more than current value/cash if value-based
            if size_type_mode == 'value' and size_to_close > max(0.0, cash):
                size_to_close = max(0.0, cash)
            direction = direction_short if position_now > 0 else direction_long
            _LAST_TRADE_BAR = i
            return (float(size_to_close), size_type_units, int(direction))

        # Exit on zero crossing
        if i > 0 and np.isfinite(zscore[i - 1]):
            prev_z = float(zscore[i - 1])
            if prev_z * z < 0:
                if size_type_mode == 'value':
                    size_to_close = abs(position_now) * (price_a if col == 0 else price_b)
                else:
                    size_to_close = abs(position_now)
                if not np.isfinite(size_to_close) or size_to_close <= 0:
                    return NO_ORDER
                if size_type_mode == 'value' and size_to_close > max(0.0, cash):
                    size_to_close = max(0.0, cash)
                direction = direction_short if position_now > 0 else direction_long
                _LAST_TRADE_BAR = i
                return (float(size_to_close), size_type_units, int(direction))

    # Entry rules (only if not currently in position for this asset)
    if abs(position_now) < 1e-12:
        # Restrict minimum bars between trades to avoid pathological churn
        if (i - _LAST_TRADE_BAR) < _MIN_BARS_BETWEEN_TRADES:
            return NO_ORDER

        # z > entry_threshold and prev_z <= entry_threshold => SHORT A, LONG B
        if (z > float(entry_threshold)) and (prev_z <= float(entry_threshold)):
            if col == 0:
                # SHORT A
                if size_type_mode == 'value':
                    size = NOTIONAL_PER_LEG
                else:
                    size = units_a
                if size_type_mode == 'value' and size > max(0.0, cash):
                    size = max(0.0, cash)
                if not np.isfinite(size) or size <= 0:
                    return NO_ORDER
                _LAST_TRADE_BAR = i
                return (float(size), size_type_units, int(direction_short))
            else:
                # LONG B
                if size_type_mode == 'value':
                    # compute B dollar notional so that units_B = units_A * hr
                    size = units_b * price_b
                else:
                    size = units_b
                if size_type_mode == 'value' and size > max(0.0, cash):
                    size = max(0.0, cash)
                if not np.isfinite(size) or size <= 0:
                    return NO_ORDER
                _LAST_TRADE_BAR = i
                return (float(size), size_type_units, int(direction_long))

        # z < -entry_threshold and prev_z >= -entry_threshold => LONG A, SHORT B
        if (z < -float(entry_threshold)) and (prev_z >= -float(entry_threshold)):
            if col == 0:
                if size_type_mode == 'value':
                    size = NOTIONAL_PER_LEG
                else:
                    size = units_a
                if size_type_mode == 'value' and size > max(0.0, cash):
                    size = max(0.0, cash)
                if not np.isfinite(size) or size <= 0:
                    return NO_ORDER
                _LAST_TRADE_BAR = i
                return (float(size), size_type_units, int(direction_long))
            else:
                if size_type_mode == 'value':
                    size = units_b * price_b
                else:
                    size = units_b
                if size_type_mode == 'value' and size > max(0.0, cash):
                    size = max(0.0, cash)
                if not np.isfinite(size) or size <= 0:
                    return NO_ORDER
                _LAST_TRADE_BAR = i
                return (float(size), size_type_units, int(direction_short))

    return NO_ORDER