"""
RSI Mean Reversion Signal Generator

This module exports a single function `generate_signals` which computes entry/exit
position targets based on RSI mean-reversion logic:

- RSI period = 14
- Enter long when RSI crosses below 30 (oversold)
- Exit long when RSI crosses above 70 (overbought)
- Long-only, single-asset (but supports DataFrame of multiple assets)

The function returns a dict with key 'ohlcv' whose value is a pd.Series or
pd.DataFrame containing position targets (1 = long, 0 = flat).

The implementation handles NaNs/warmup periods by ignoring cross events when
previous or current RSI is NaN. It uses vectorbt's RSI implementation when
available (vectorbt is expected in the backtest runner environment).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except Exception:  # pragma: no cover - vectorbt should be present in runner
    vbt = None


def _compute_rsi(close: pd.Series | pd.DataFrame, window: int = 14) -> pd.Series | pd.DataFrame:
    """Compute RSI for a price series or DataFrame.

    Uses vectorbt's RSI if available, otherwise falls back to a pandas
    implementation (Wilder's smoothing via ewm).

    Args:
        close: Price series or DataFrame of close prices.
        window: RSI lookback period.

    Returns:
        RSI values aligned with `close`.
    """
    if vbt is not None:
        try:
            # vbt.RSI.run returns an object with `rsi` attribute
            rsi_res = vbt.RSI.run(close, window=window)
            rsi = rsi_res.rsi
            return rsi
        except Exception:
            # Fall back to manual implementation if vbt fails for some reason
            pass

    # Manual RSI implementation (Wilder's smoothing)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder smoothing: alpha = 1/window, adjust=False for exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _positions_from_entry_exit(entries: pd.Series | pd.DataFrame, exits: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Given boolean entries and exits (aligned), produce a position series (1/0).

    This function works for both Series and DataFrame. It iterates over time
    and ensures entries only take effect when flat and exits only when long.

    Args:
        entries: Boolean Series/DataFrame where True indicates an entry signal.
        exits: Boolean Series/DataFrame where True indicates an exit signal.

    Returns:
        Series/DataFrame of integer positions (1 for long, 0 for flat).
    """
    # Normalize types
    if isinstance(entries, pd.Series):
        # Series path
        entries_s = entries.fillna(False).astype(bool)
        exits_s = exits.fillna(False).astype(bool)

        pos = pd.Series(0, index=entries_s.index, dtype="int8")
        in_pos = False
        # Use integer indexing for speed
        for i in range(len(entries_s)):
            e = entries_s.iloc[i]
            ex = exits_s.iloc[i]
            if not in_pos:
                if bool(e):
                    in_pos = True
            else:
                if bool(ex):
                    in_pos = False
            pos.iloc[i] = 1 if in_pos else 0
        return pos

    # DataFrame path: process column by column
    entries_df = entries.fillna(False).astype(bool)
    exits_df = exits.fillna(False).astype(bool)

    pos_df = pd.DataFrame(0, index=entries_df.index, columns=entries_df.columns, dtype="int8")

    for col in entries_df.columns:
        in_pos = False
        for i in range(len(entries_df)):
            e = entries_df.iloc[i][col]
            ex = exits_df.iloc[i][col]
            if not in_pos:
                if bool(e):
                    in_pos = True
            else:
                if bool(ex):
                    in_pos = False
            pos_df.iloc[i, pos_df.columns.get_loc(col)] = 1 if in_pos else 0

    return pos_df


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Generate position targets based on RSI mean-reversion.

    Args:
        data: Dict containing at least the key 'ohlcv' mapping to a DataFrame
              with a 'close' column (pd.Series) or a DataFrame of closes.
        params: Strategy parameters (not used for thresholds here, kept for API
                compatibility).

    Returns:
        Dict with key 'ohlcv' mapping to a pd.Series or pd.DataFrame of
        position targets (1 for long, 0 for flat).

    Raises:
        ValueError: If the required data is not present or improperly formatted.
    """
    # Validate input
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing 'ohlcv' DataFrame/Series")

    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data['ohlcv']

    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame containing a 'close' column")

    if 'close' not in ohlcv:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close = ohlcv['close']

    # Compute RSI (period fixed at 14 as per spec)
    rsi = _compute_rsi(close, window=14)

    # Ensure alignment
    rsi = rsi.reindex_like(close)

    # Detect crosses: entry when RSI crosses below 30, exit when RSI crosses above 70
    prev = rsi.shift(1)

    # Create boolean entry/exit signals; ignore cross if prev or curr is NaN
    entry_mask = (prev > 30) & (rsi <= 30) & (~prev.isna()) & (~rsi.isna())
    exit_mask = (prev < 70) & (rsi >= 70) & (~prev.isna()) & (~rsi.isna())

    # Build positions (1 = long, 0 = flat)
    positions = _positions_from_entry_exit(entry_mask, exit_mask)

    # Ensure return type is aligned with ohlcv['close'] (Series or DataFrame)
    # Cast to same index/columns and consistent dtype
    if isinstance(close, pd.Series) and isinstance(positions, pd.DataFrame):
        # If positions came back as DataFrame with single column, convert to Series
        if positions.shape[1] == 1:
            positions = positions.iloc[:, 0]
        else:
            # collapse columns if names differ (should not happen here)
            positions = positions.iloc[:, 0]

    # Final safety: ensure positions index equals close index
    if not positions.index.equals(close.index):
        positions = positions.reindex(close.index).fillna(0).astype(int)

    return {"ohlcv": positions}
