import pandas as pd
import numpy as np


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

    Notes:
        - RSI is computed using Wilder's smoothing (EWMA with alpha=1/period).
        - Entry: RSI crosses below `oversold` (prev >= oversold and current < oversold).
        - Exit: RSI crosses above `overbought` (prev <= overbought and current > overbought).
    """

    # Basic validations
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    df = data["ohlcv"]
    if "close" not in df.columns:
        raise ValueError("'ohlcv' DataFrame must contain 'close' column")

    close = df["close"].astype(float)

    # Extract and validate params (use defaults if missing)
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0 and 50")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50 and 100")
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")

    # Compute RSI using Wilder's smoothing (EWMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing: alpha = 1/period, adjust=False
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Compute RSI values safely (handle divide-by-zero)
    rs_values = avg_gain.values / avg_loss.values
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi_vals = 100.0 - (100.0 / (1.0 + rs_values))

    mask_loss_zero = (avg_loss.values == 0)
    mask_gain_zero = (avg_gain.values == 0)

    # avg_loss == 0 & avg_gain != 0 -> RSI = 100
    rsi_vals[mask_loss_zero & ~mask_gain_zero] = 100.0
    # avg_gain == 0 & avg_loss != 0 -> RSI = 0
    rsi_vals[mask_gain_zero & ~mask_loss_zero] = 0.0
    # both zero -> no movement -> neutral 50
    rsi_vals[mask_gain_zero & mask_loss_zero] = 50.0

    rsi = pd.Series(rsi_vals, index=close.index, name="rsi")

    # Preserve NaNs from price series
    rsi[close.isna()] = np.nan

    # Detect crosses: entry when RSI crosses below oversold, exit when crosses above overbought
    rsi_prev = rsi.shift(1)
    entry_signals = (rsi_prev >= oversold) & (rsi < oversold)
    exit_signals = (rsi_prev <= overbought) & (rsi > overbought)

    # Build position series (long-only). Iterate to properly handle stateful behavior.
    pos = np.zeros(len(rsi), dtype=int)
    in_pos = False
    entry_vals = entry_signals.fillna(False).values
    exit_vals = exit_signals.fillna(False).values

    for i in range(len(pos)):
        if entry_vals[i] and not in_pos:
            in_pos = True
        elif exit_vals[i] and in_pos:
            in_pos = False
        pos[i] = 1 if in_pos else 0

    position = pd.Series(pos, index=close.index, dtype=int, name="position")

    return {"ohlcv": position}
