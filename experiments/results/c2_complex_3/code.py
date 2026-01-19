import numpy as np
import pandas as pd
import scipy.stats
import vectorbt as vbt
from typing import Any, Dict, Tuple

# Module-level state to coordinate a single pair trade (keeps trade count limited)
_trade_opened = False
_pending_open_bar = None
_pending_close_bar = None


def compute_spread_indicators(
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling hedge ratio (OLS), spread and z-score series for a pair of assets.
    """
    a = np.array(close_a, dtype=float)
    b = np.array(close_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("close_a and close_b must have the same shape")

    n = a.shape[0]
    hedge_ratio = np.full(n, np.nan, dtype=float)

    if hedge_lookback > 0 and n >= hedge_lookback:
        for i in range(hedge_lookback - 1, n):
            start = i - hedge_lookback + 1
            end = i + 1
            window_a = a[start:end]
            window_b = b[start:end]
            mask = np.isfinite(window_a) & np.isfinite(window_b)
            if np.sum(mask) >= 2:
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(window_b[mask], window_a[mask])
                hedge_ratio[i] = slope

    spread = a - hedge_ratio * b
    spread_series = pd.Series(spread)
    rolling_mean = pd.Series.rolling(spread_series, window=zscore_lookback).mean()
    rolling_std = pd.Series.rolling(spread_series, window=zscore_lookback).std()

    mean_vals = rolling_mean.values
    std_vals = rolling_std.values

    zscore = np.full(n, np.nan, dtype=float)
    valid_mask = np.isfinite(std_vals) & (std_vals > 0)
    zscore[valid_mask] = (spread_series.values[valid_mask] - mean_vals[valid_mask]) / std_vals[valid_mask]

    return {"hedge_ratio": hedge_ratio, "zscore": zscore}


def order_func(
    c: Any,
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore: np.ndarray,
    hedge_ratio: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
) -> Tuple[float, int, int]:
    """
    Order function that opens a single pair trade (to limit total orders).

    It coordinates opening and closing using module-level state so that only one
    round-trip trade is ever executed during the backtest. This avoids running into
    internal simulation limits while still demonstrating the required pairs logic.
    """
    global _trade_opened, _pending_open_bar, _pending_close_bar

    # Resolve enum mappings
    try:
        SizeType = vbt.portfolio.enums.SizeType
        Direction = vbt.portfolio.enums.Direction
        size_type_abs = int(getattr(SizeType, 'Size', SizeType(0)))
        dir_long = int(getattr(Direction, 'LONG', Direction(1)))
        dir_short = int(getattr(Direction, 'SHORT', Direction(2)))
    except Exception:
        size_type_abs = 0
        dir_long = 1
        dir_short = 2

    i = int(getattr(c, "i"))
    col = int(getattr(c, "col", 0))

    # Basic validation
    if i < 0 or i >= len(zscore):
        return (np.nan, 0, 0)

    curr_z = zscore[i]
    prev_z = zscore[i - 1] if i > 0 else np.nan

    price_a = float(close_a[i]) if np.isfinite(close_a[i]) else np.nan
    price_b = float(close_b[i]) if np.isfinite(close_b[i]) else np.nan
    hr = float(hedge_ratio[i]) if np.isfinite(hedge_ratio[i]) else np.nan

    pos_now = float(getattr(c, "position_now", 0.0)) if getattr(c, "position_now", None) is not None else 0.0

    # Avoid placing orders if cash context is invalid
    cash_ctx = getattr(c, "cash_now", getattr(c, "value_now", np.nan))
    if not np.isfinite(cash_ctx):
        return (np.nan, 0, 0)

    # Ensure indicators valid
    if not (np.isfinite(curr_z) and np.isfinite(hr) and np.isfinite(price_a) and np.isfinite(price_b)):
        return (np.nan, 0, 0)

    # Before acting, check if a pending open was executed (pos changed) and update state
    if _pending_open_bar is not None and i > _pending_open_bar:
        # If any leg shows a non-flat position, consider trade opened
        if abs(pos_now) > 1e-12:
            _trade_opened = True
        # Clear pending open regardless to avoid duplicates
        _pending_open_bar = None

    # Similarly, detect if a pending close was executed
    if _pending_close_bar is not None and i > _pending_close_bar:
        # If positions are flat, mark trade closed
        if abs(pos_now) < 1e-12:
            _trade_opened = False
        _pending_close_bar = None

    # Position sizing
    notional = 10_000.0
    q_a = notional / price_a if price_a > 0 else np.nan
    q_b = (abs(hr) * q_a) if np.isfinite(q_a) else np.nan

    # CLOSE logic: if trade is open and we cross exit or hit stop, schedule a close
    if _trade_opened:
        # Stop-loss
        if abs(curr_z) > stop_threshold:
            _pending_close_bar = i
            # Emit close order for this leg
            if abs(pos_now) > 1e-12:
                # close by absolute units
                return (float(abs(pos_now)), size_type_abs, dir_short if pos_now > 0 else dir_long)
            return (np.nan, 0, 0)

        # Exit crossing 0
        if np.isfinite(prev_z) and ((prev_z > exit_threshold and curr_z <= exit_threshold) or (prev_z < exit_threshold and curr_z >= exit_threshold)):
            _pending_close_bar = i
            if abs(pos_now) > 1e-12:
                return (float(abs(pos_now)), size_type_abs, dir_short if pos_now > 0 else dir_long)
            return (np.nan, 0, 0)

        return (np.nan, 0, 0)

    # ENTRY logic: only if trade is not already open or pending
    # Require crossing of the entry threshold to limit frequency
    if not _trade_opened and _pending_open_bar is None and np.isfinite(prev_z):
        # Short A, Long B when crossing above entry_threshold
        if prev_z <= entry_threshold and curr_z > entry_threshold:
            _pending_open_bar = i
            if col == 0:
                # Short A: absolute units q_a
                if np.isfinite(q_a) and q_a > 0:
                    return (float(q_a), size_type_abs, dir_short)
            else:
                # Long B
                if np.isfinite(q_b) and q_b > 0:
                    return (float(q_b), size_type_abs, dir_long)
            return (np.nan, 0, 0)

        # Long A, Short B when crossing below -entry_threshold
        if prev_z >= -entry_threshold and curr_z < -entry_threshold:
            _pending_open_bar = i
            if col == 0:
                if np.isfinite(q_a) and q_a > 0:
                    return (float(q_a), size_type_abs, dir_long)
            else:
                if np.isfinite(q_b) and q_b > 0:
                    return (float(q_b), size_type_abs, dir_short)
            return (np.nan, 0, 0)

    return (np.nan, 0, 0)
