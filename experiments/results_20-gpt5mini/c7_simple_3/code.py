import numpy as np
import pandas as pd
from typing import Dict, Any


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

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """
    # Basic validation
    if not isinstance(data, dict):
        raise ValueError("data must be a dict mapping slot names to DataFrames")
    if 'ohlcv' not in data:
        raise ValueError("data must contain 'ohlcv' key with a DataFrame")

    df = data['ohlcv']
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")
    if 'close' not in df.columns:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = df['close'].astype(float).copy()

    # Parameters with validation against PARAM_SCHEMA
    if params is None:
        params = {}

    try:
        rsi_period = int(params.get('rsi_period', 14))
    except Exception:
        raise ValueError("rsi_period must be an integer")
    if rsi_period < 2 or rsi_period > 100:
        raise ValueError("rsi_period must be between 2 and 100")

    try:
        oversold = float(params.get('oversold', 30.0))
        overbought = float(params.get('overbought', 70.0))
    except Exception:
        raise ValueError("oversold and overbought must be floats")

    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")

    # Compute RSI using Wilder's smoothing (EMA with alpha = 1/period, adjust=False)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use exponential weighted mean with alpha = 1 / period (Wilder's smoothing)
    # adjust=False ensures recursive (causal) calculation -> no lookahead
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    # Compute RS and RSI
    # Avoid division by zero: when avg_loss == 0 and avg_gain == 0 -> RSI = 50
    # when avg_loss == 0 but avg_gain > 0 -> RSI = 100
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases explicitly
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    loss_zero = (avg_loss == 0.0) & (~both_zero)
    rsi = rsi.copy()
    if both_zero.any():
        rsi.loc[both_zero] = 50.0
    if loss_zero.any():
        rsi.loc[loss_zero] = 100.0

    # For any remaining NaNs (e.g., first bar), fill with neutral 50
    rsi = rsi.fillna(50.0)

    # Generate crossing signals (uses only past and current RSI values)
    prev_rsi = rsi.shift(1)
    cross_below = (prev_rsi >= oversold) & (rsi < oversold)
    cross_above = (prev_rsi <= overbought) & (rsi > overbought)

    # If the first valid RSI is already below oversold, allow entry at first bar
    if len(rsi) > 0 and not pd.isna(rsi.iloc[0]):
        if rsi.iloc[0] < oversold:
            cross_below.iloc[0] = True
        # Do not force an exit on the first bar even if RSI > overbought

    # Build position series (0 = flat, 1 = long)
    n = len(rsi)
    pos_arr = np.zeros(n, dtype=int)
    in_position = False

    for i in range(n):
        # If RSI is NaN at this bar, remain flat
        if pd.isna(rsi.iloc[i]):
            pos_arr[i] = 0
            continue

        if not in_position:
            # Enter when RSI crosses below oversold
            if bool(cross_below.iloc[i]):
                pos_arr[i] = 1
                in_position = True
            else:
                pos_arr[i] = 0
        else:
            # Exit when RSI crosses above overbought
            if bool(cross_above.iloc[i]):
                pos_arr[i] = 0
                in_position = False
            else:
                pos_arr[i] = 1

    position = pd.Series(pos_arr, index=close.index, name='position', dtype=int)

    # Ensure there are no NaNs in the output (after warmup/warm periods)
    position = position.fillna(0).astype(int)

    return {'ohlcv': position}
