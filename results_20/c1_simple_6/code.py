import pandas as pd
import numpy as np
from typing import Dict


def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    """
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
              - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series.
        This is a LONG-ONLY strategy, so position values are: 1 (long) or 0 (flat).
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}
    """

    # -------------------------
    # Input validation
    # -------------------------
    if not isinstance(data, dict):
        raise TypeError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise KeyError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv DataFrame must contain 'close' column")

    close: pd.Series = ohlcv["close"].astype(float)

    # Parameters with defaults from PARAM_SCHEMA
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Validate parameter ranges (as per PARAM_SCHEMA)
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # -------------------------
    # RSI calculation (Wilder's smoothing)
    # -------------------------
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        """Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

        Returns a Series aligned with the input. Initial warmup values will be NaN
        until enough data is available for the given period.
        """
        delta = series.diff()

        # Gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing: exponential moving average with alpha=1/period
        # Use min_periods=period to enforce warmup (NaN until enough obs)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        # Relative strength
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle edge cases:
        # - If avg_loss == 0 and avg_gain > 0 -> RSI = 100
        # - If avg_gain == 0 and avg_loss == 0 -> RSI = 50 (no movement)
        mask_loss_zero = (avg_loss == 0) & (~avg_gain.isna())
        mask_both_zero = (avg_loss == 0) & (avg_gain == 0)

        rsi = rsi.copy()
        rsi[mask_loss_zero] = 100.0
        rsi[mask_both_zero] = 50.0

        return rsi

    rsi = _rsi(close, rsi_period)

    # -------------------------
    # Entry / Exit signals
    # - Entry: RSI crosses BELOW oversold (previous >= oversold and current < oversold)
    # - Exit:  RSI crosses ABOVE overbought (previous <= overbought and current > overbought)
    # -------------------------
    prev_rsi = rsi.shift(1)

    entry_signal = (prev_rsi >= oversold) & (rsi < oversold)
    exit_signal = (prev_rsi <= overbought) & (rsi > overbought)

    # Replace NaNs with False so they don't create spurious signals
    entry_signal = entry_signal.fillna(False)
    exit_signal = exit_signal.fillna(False)

    # -------------------------
    # Build position series (long-only): 1 = long, 0 = flat
    # Use cumulative counts of entries and exits to maintain state.
    # -------------------------
    entries_cum = entry_signal.astype(int).cumsum()
    exits_cum = exit_signal.astype(int).cumsum()

    position = (entries_cum > exits_cum).astype(int)

    # Ensure alignment and no NaNs
    position = pd.Series(position, index=close.index, name="position").fillna(0).astype(int)

    return {"ohlcv": position}
