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

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    """
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("data must be a dict mapping slot names to DataFrames")

    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' key with OHLCV DataFrame")

    ohlcv = data["ohlcv"]
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("data['ohlcv'] must be a pandas DataFrame")

    if "close" not in ohlcv.columns:
        raise ValueError("data['ohlcv'] must contain 'close' column")

    close = ohlcv["close"].astype(float)

    # Read and validate parameters (only use allowed params)
    # Defaults mirror PARAM_SCHEMA defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))

    # Validate parameter ranges according to PARAM_SCHEMA
    if not (2 <= rsi_period <= 100):
        raise ValueError("rsi_period must be between 2 and 100")
    if not (0.0 <= oversold <= 50.0):
        raise ValueError("oversold must be between 0.0 and 50.0")
    if not (50.0 <= overbought <= 100.0):
        raise ValueError("overbought must be between 50.0 and 100.0")
    if oversold >= overbought:
        raise ValueError("oversold must be strictly less than overbought")

    # Helper: compute RSI using rolling mean (simple RSI) with min_periods = rsi_period
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        # Ensure float dtype
        s = series.astype(float)
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use simple moving average of gains/losses to produce NaN until warmup
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = pd.Series(rsi, index=series.index)

        # Handle special cases:
        # - If both avg_gain and avg_loss are exactly zero -> neutral RSI (50)
        # - If avg_loss == 0 and avg_gain > 0 -> RSI = 100
        # - If avg_gain == 0 and avg_loss > 0 -> RSI = 0
        mask_both_zero = (avg_gain == 0) & (avg_loss == 0)
        mask_loss_zero = (avg_loss == 0) & (avg_gain > 0)
        mask_gain_zero = (avg_gain == 0) & (avg_loss > 0)

        if mask_both_zero.any():
            rsi.loc[mask_both_zero.fillna(False)] = 50.0
        if mask_loss_zero.any():
            rsi.loc[mask_loss_zero.fillna(False)] = 100.0
        if mask_gain_zero.any():
            rsi.loc[mask_gain_zero.fillna(False)] = 0.0

        # Keep NaN for warmup periods where rolling window isn't full
        warmup_mask = avg_gain.isna() | avg_loss.isna()
        rsi.loc[warmup_mask] = np.nan

        return rsi

    rsi = _compute_rsi(close, rsi_period)

    # Define crossing conditions using previous bar to detect crossings
    prev_rsi = rsi.shift(1)

    # Ensure NA comparisons do not produce True
    entry_cond = (prev_rsi >= oversold) & (rsi < oversold) & rsi.notna() & prev_rsi.notna()
    exit_cond = (prev_rsi <= overbought) & (rsi > overbought) & rsi.notna() & prev_rsi.notna()

    # Convert to boolean Series aligned with close index
    entry_signals = entry_cond.astype(bool).reindex(close.index, fill_value=False)
    exit_signals = exit_cond.astype(bool).reindex(close.index, fill_value=False)

    # Build position series: long (1) when #entries > #exits, else flat (0)
    entries_cum = entry_signals.cumsum()
    exits_cum = exit_signals.cumsum()

    position = (entries_cum > exits_cum).astype(int)

    # Ensure no NaNs and correct dtype
    position = position.reindex(close.index, fill_value=0).astype(int)

    return {"ohlcv": position}
