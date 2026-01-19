from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd

# Attempt to resolve vectorbt enums at import time to get correct integer codes
try:
    import vectorbt as vbt  # type: ignore
    _SIZE_TYPE_MAP = {}
    _DIRECTION_MAP = {}
    try:
        st_enum = vbt.portfolio.enums.SizeType
        for member in list(st_enum):
            _SIZE_TYPE_MAP[member.name.lower()] = int(member.value)
    except Exception:
        pass
    try:
        d_enum = vbt.portfolio.enums.Direction
        for member in list(d_enum):
            _DIRECTION_MAP[member.name.lower()] = int(member.value)
    except Exception:
        pass

    # Choose sensible defaults if available
    # Prefer shares/size for absolute shares, and buy/long and sell/short for directions
    SIZE_TYPE_SHARES = _SIZE_TYPE_MAP.get('shares',
                        _SIZE_TYPE_MAP.get('size',
                        _SIZE_TYPE_MAP.get('amount',
                        _SIZE_TYPE_MAP.get('value', 0))))
    DIR_BUY = _DIRECTION_MAP.get('buy',
              _DIRECTION_MAP.get('long',
              _DIRECTION_MAP.get('up', 1)))
    DIR_SELL = _DIRECTION_MAP.get('sell',
               _DIRECTION_MAP.get('short',
               _DIRECTION_MAP.get('down', 2)))
except Exception:
    # Fallback constants
    SIZE_TYPE_SHARES = 0
    DIR_BUY = 1
    DIR_SELL = 2


def _to_1d_array(x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
    """Convert input to 1D numpy array of floats.

    - If x is a Series, return its values.
    - If x is a 1D numpy array, return as float array.
    - If x is a DataFrame with one column, return that column.
    - If x is a DataFrame with a 'close' column, use it (if multiple, take first).
    - Otherwise, if it's a 2D array, take the first column.
    """
    if isinstance(x, pd.Series):
        return x.values.astype(float)
    if isinstance(x, pd.DataFrame):
        # Prefer explicit 'close' column if present
        if 'close' in x.columns:
            col = x['close']
            if isinstance(col, pd.DataFrame):
                return col.iloc[:, 0].values.astype(float)
            return col.values.astype(float)
        # If DataFrame has a single column, use it
        if x.shape[1] == 1:
            return x.iloc[:, 0].values.astype(float)
        # Fallback: take first column
        return x.iloc[:, 0].values.astype(float)
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        return arr[:, 0].astype(float)
    # Last resort: flatten
    return arr.ravel().astype(float)


def compute_spread_indicators(
    close_a: Union[np.ndarray, pd.Series, pd.DataFrame],
    close_b: Union[np.ndarray, pd.Series, pd.DataFrame],
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS slope) and z-score of the spread.

    Uses only information up to and including time t (no lookahead). The rolling
    windows use an expanding window until the requested lookback is available
    (i.e. window size = min(lookback, t+1)). This ensures outputs are defined
    early and avoids NaNs after modest warmup.

    The function is robust to inputs of different lengths: it will compute
    indicators up to the length of the first input (close_a). If close_b is
    longer, only its prefix of equal length is used (prevents lookahead). If
    close_b is shorter, indicators beyond its length are returned as NaN.

    Args:
        close_a: Prices for asset A (array-like)
        close_b: Prices for asset B (array-like)
        hedge_lookback: Lookback (in bars) for rolling OLS hedge ratio
        zscore_lookback: Lookback (in bars) for rolling mean/std of spread

    Returns:
        Dict with keys:
            - "zscore": np.ndarray, same length as close_a
            - "hedge_ratio": np.ndarray, same length as close_a
    """
    a = _to_1d_array(close_a)
    b = _to_1d_array(close_b)

    n_a = a.shape[0]
    n_b = b.shape[0]

    # Output arrays sized to length of close_a (first arg)
    hedge_ratio = np.full(n_a, np.nan, dtype=float)
    spread = np.full(n_a, np.nan, dtype=float)
    zscore = np.full(n_a, np.nan, dtype=float)

    # Compute up to the available overlap to avoid lookahead
    compute_len = min(n_a, n_b)

    for t in range(compute_len):
        # Determine rolling window for regression (use only past values up to t)
        start = max(0, t - hedge_lookback + 1) if hedge_lookback is not None else 0
        x = b[start : t + 1]
        y = a[start : t + 1]

        if x.size < 2:
            slope = 0.0
        else:
            x_mean = np.nanmean(x)
            y_mean = np.nanmean(y)
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0 or not np.isfinite(denom):
                slope = 0.0
            else:
                slope = ((x - x_mean) * (y - y_mean)).sum() / denom

        hedge_ratio[t] = float(slope)
        spread[t] = a[t] - hedge_ratio[t] * b[t]

    # Rolling z-score (expanding until lookback reached) for the computed region
    for t in range(compute_len):
        start = max(0, t - zscore_lookback + 1)
        w = spread[start : t + 1]
        mu = np.nanmean(w)
        sigma = np.nanstd(w)
        if not np.isfinite(sigma) or sigma == 0:
            z = 0.0
        else:
            z = (spread[t] - mu) / sigma
        zscore[t] = float(z)

    return {"zscore": zscore, "hedge_ratio": hedge_ratio}


def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
) -> Tuple[float, int, int]:
    """
    Order function for flexible multi-asset pairs trading.

    This function is called once per asset per bar. It computes the desired
    target position (in shares) for the current asset given the z-score and
    hedge ratio, and returns the order needed to move from the current
    position to the target.

    Position sizing:
      - Anchor on asset A: target_shares_a = round(fixed_notional / price_a)
      - Asset B shares are scaled by the hedge ratio: target_shares_b = hedge_ratio * target_shares_a

    We only place entry orders on threshold crossings (to avoid repeated orders
    while the z-score remains beyond the threshold). Exits occur when z-score
    crosses 0 or when stop-loss threshold is breached.

    This function keeps a lightweight internal state (reset each run at i==0)
    to ensure both legs use the same anchor sizing for the duration of the trade
    and to avoid micro-adjustments causing many small orders.

    Returns a tuple (size, size_type, direction):
      - size: absolute number of shares to trade (float). If NaN, no order.
      - size_type: integer indicating absolute share sizing (0 used here).
      - direction: integer code for direction as resolved from vectorbt enums.
    """
    # Basic safety and context extraction
    i = int(getattr(c, "i", 0))
    col = int(getattr(c, "col", 0))  # 0 -> asset A, 1 -> asset B

    # Cooldown period (bars) after a closed trade to avoid overtrading
    COOLDOWN_PERIOD = 50
    MAX_TRADES = 1

    # Initialize per-run state on the first bar
    if not hasattr(order_func, "_phase") or i == 0:
        order_func._phase = 'idle'  # one of 'idle','opening','open','closing'
        order_func._pending = set()
        order_func._target_a = 0.0
        order_func._target_b = 0.0
        order_func._entry_index = -1
        order_func._cooldown_until = -1
        order_func._trade_count = 0

    # Defensive access of arrays
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    zscore = np.asarray(zscore, dtype=float)
    hedge_ratio = np.asarray(hedge_ratio, dtype=float)

    n = len(zscore)
    if i < 0 or i >= n:
        # Invalid index: no order
        return (np.nan, 0, 0)

    price_a = float(close_a[i]) if i < len(close_a) else np.nan
    price_b = float(close_b[i]) if i < len(close_b) else np.nan
    z = float(zscore[i])
    hr = float(hedge_ratio[i])

    # If indicators are not finite, do nothing
    if not np.isfinite(z) or not np.isfinite(hr) or not np.isfinite(price_a) or not np.isfinite(price_b) or price_a <= 0 or price_b <= 0:
        return (np.nan, 0, 0)

    # Current position in shares for this asset (may be 0.0)
    pos_now = float(getattr(c, "position_now", 0.0) or 0.0)

    # Fixed notional per leg
    fixed_notional = 10_000.0

    # Determine previous z (for crossing detection)
    prev_z = float(zscore[i - 1]) if i > 0 and np.isfinite(zscore[i - 1]) else np.nan

    # Helper to produce an order for a pending column
    def _handle_pending(col_idx: int, tgt: float) -> Tuple[float, int, int]:
        nonlocal pos_now
        delta = tgt - pos_now
        # If already at target, mark as done
        if abs(delta) < 1e-6:
            # remove from pending
            if col_idx in order_func._pending:
                order_func._pending.discard(col_idx)
            # If no pending left, update phase and set cooldown if closing
            if len(order_func._pending) == 0:
                if order_func._phase == 'opening':
                    order_func._phase = 'open'
                elif order_func._phase == 'closing':
                    order_func._phase = 'idle'
                    # start cooldown
                    order_func._cooldown_until = i + COOLDOWN_PERIOD
            return (np.nan, 0, 0)
        # Otherwise place order to move to target
        size = float(abs(delta))
        size_type = int(SIZE_TYPE_SHARES)
        direction = int(DIR_BUY if delta > 0 else DIR_SELL)
        return (size, int(size_type), int(direction))

    # Process according to phase
    if order_func._phase in ('opening', 'closing'):
        # If this column is pending, handle it; otherwise no action
        if col in order_func._pending:
            tgt = order_func._target_a if col == 0 else order_func._target_b
            return _handle_pending(col, tgt)
        else:
            return (np.nan, 0, 0)

    if order_func._phase == 'idle':
        # Respect cooldown
        if i < order_func._cooldown_until:
            return (np.nan, 0, 0)

        # Check trade count
        if order_func._trade_count >= MAX_TRADES:
            return (np.nan, 0, 0)

        # Check for entry crossing
        if z > entry_threshold and (not np.isfinite(prev_z) or prev_z <= entry_threshold):
            # Prepare opening
            raw_anchor = fixed_notional / price_a if price_a > 0 else 0.0
            anchor_shares = max(1, int(round(raw_anchor)))
            order_func._target_a = -float(anchor_shares)
            order_func._target_b = float(hr * anchor_shares)
            order_func._phase = 'opening'
            order_func._pending = {0, 1}
            order_func._entry_index = i
            order_func._trade_count += 1
            # Now handle this column immediately
            if col in order_func._pending:
                return _handle_pending(col, order_func._target_a if col == 0 else order_func._target_b)
            return (np.nan, 0, 0)

        if z < -entry_threshold and (not np.isfinite(prev_z) or prev_z >= -entry_threshold):
            raw_anchor = fixed_notional / price_a if price_a > 0 else 0.0
            anchor_shares = max(1, int(round(raw_anchor)))
            order_func._target_a = float(anchor_shares)
            order_func._target_b = -float(hr * anchor_shares)
            order_func._phase = 'opening'
            order_func._pending = {0, 1}
            order_func._entry_index = i
            order_func._trade_count += 1
            if col in order_func._pending:
                return _handle_pending(col, order_func._target_a if col == 0 else order_func._target_b)
            return (np.nan, 0, 0)

        # No entry
        return (np.nan, 0, 0)

    if order_func._phase == 'open':
        # Check for exit: stop-loss or crossing 0
        if abs(z) > stop_threshold:
            order_func._phase = 'closing'
            order_func._pending = {0, 1}
            order_func._target_a = 0.0
            order_func._target_b = 0.0
            if col in order_func._pending:
                return _handle_pending(col, order_func._target_a if col == 0 else order_func._target_b)
            return (np.nan, 0, 0)

        crossed = False
        if np.isfinite(prev_z):
            if (prev_z > 0 and z <= exit_threshold) or (prev_z < 0 and z >= exit_threshold):
                crossed = True
        if crossed:
            order_func._phase = 'closing'
            order_func._pending = {0, 1}
            order_func._target_a = 0.0
            order_func._target_b = 0.0
            if col in order_func._pending:
                return _handle_pending(col, order_func._target_a if col == 0 else order_func._target_b)
            return (np.nan, 0, 0)

        return (np.nan, 0, 0)

    # Fallback: no order
    return (np.nan, 0, 0)
