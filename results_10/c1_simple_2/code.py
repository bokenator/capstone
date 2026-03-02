import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_signals(
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, pd.Series]:
    """
    Generate position signals for an RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames. Must contain 'ohlcv' key
              with DataFrame having a 'close' column.
        params: Strategy parameters dict with keys:
            - rsi_period (int): RSI calculation period
            - oversold (float): RSI level that triggers entry when RSI CROSSES BELOW it
            - overbought (float): RSI level that triggers exit when RSI CROSSES ABOVE it

    Returns:
        Dict mapping slot names to position Series. LONG-ONLY strategy; values are 1 (long) or 0 (flat).
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")
    df = data["ohlcv"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if "close" not in df.columns:
        raise ValueError("data['ohlcv'] must contain a 'close' column")

    close: pd.Series = df["close"].astype(float).copy()

    # Read and validate params (only allowed params are used)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if oversold < 0.0 or oversold > 50.0:
        raise ValueError("oversold must be between 0.0 and 50.0")
    if overbought < 50.0 or overbought > 100.0:
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold level must be strictly less than overbought level")

    # Helper: compute RSI using Wilder's smoothing (EWMA with alpha = 1/period)
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0).fillna(0.0)
        loss = -delta.clip(upper=0.0).fillna(0.0)

        # Use exponential weighted mean with alpha = 1/period (Wilder's smoothing)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle edge cases explicitly
        # If both avg_gain and avg_loss are zero -> no movement -> RSI = 50
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        loss_zero_only = (avg_loss == 0) & (avg_gain != 0)
        gain_zero_only = (avg_gain == 0) & (avg_loss != 0)

        rsi = rsi.copy()
        rsi.loc[both_zero] = 50.0
        rsi.loc[loss_zero_only] = 100.0
        rsi.loc[gain_zero_only] = 0.0

        return rsi

    rsi = _rsi(close, rsi_period)

    # Entry: RSI crosses below oversold -> prev >= oversold and curr < oversold
    # Exit: RSI crosses above overbought -> prev <= overbought and curr > overbought
    prev_rsi = rsi.shift(1)

    # Use boolean masks; comparisons with NaN produce False (so no false triggers during warmup)
    entry_mask = (prev_rsi >= oversold) & (rsi < oversold)
    exit_mask = (prev_rsi <= overbought) & (rsi > overbought)

    # Build position series (long-only). Iterate to ensure proper handling of consecutive signals.
    pos_values = np.zeros(len(close), dtype=int)
    in_long = False

    for i in range(len(close)):
        # If RSI is NaN at this bar, remain flat
        if pd.isna(rsi.iloc[i]):
            pos_values[i] = 0
            continue

        if not in_long:
            if bool(entry_mask.iloc[i]):
                in_long = True
                pos_values[i] = 1
            else:
                pos_values[i] = 0
        else:
            # currently long
            if bool(exit_mask.iloc[i]):
                in_long = False
                pos_values[i] = 0
            else:
                pos_values[i] = 1

    positions = pd.Series(pos_values, index=close.index, name="position").astype(int)

    return {"ohlcv": positions}
