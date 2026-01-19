"""
Function Signatures
===================

Type-annotated function signatures for all strategy complexity levels.
Used by: C1 (Schema), C4 (Schema+Docs), C5 (Schema+TDD), C7 (All)

The simple interface uses position targets {-1, 0, +1} which can be
converted to vectorbt signals via:
    entries = position.diff() > 0
    exits = position.diff() < 0
"""

SIGNATURE_SIMPLE = """
## Function Signature

```python
def generate_signals(
    data: dict[str, pd.DataFrame],
    params: dict
) -> dict[str, pd.Series]:
    \"\"\"
    Generate position signals for RSI mean reversion strategy.

    Args:
        data: Dict mapping slot names to DataFrames.
              Must contain 'ohlcv' key with DataFrame having 'close' column.
        params: Strategy parameters dict with keys:
              - rsi_period (int): RSI calculation period
              - oversold (float): RSI threshold for entry (go long)
              - overbought (float): RSI threshold for exit (go flat)

    Returns:
        Dict mapping slot names to position Series.
        Position values: +1 (long), 0 (flat), -1 (short)
        Example: {"ohlcv": pd.Series([0, 0, 1, 1, 0, ...], index=...)}

    Usage with vectorbt:
        signals = generate_signals(data, params)
        position = signals['ohlcv']
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        pf = vbt.Portfolio.from_signals(data['ohlcv']['close'], entries, exits)
    \"\"\"
```
"""

SIGNATURE_MEDIUM = """
## Function Signatures

```python
import numpy as np
import pandas as pd
import vectorbt as vbt

def order_func(
    c,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float
) -> tuple:
    \"\"\"
    Generate order at each bar. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.position_now: current position size (float, 0.0 if flat)
           - c.cash_now: current cash balance (float)
        close: Close prices array (use close[c.i] for current price)
        high: High prices array
        macd: MACD line array
        signal: Signal line array
        atr: ATR array
        sma: SMA array
        trailing_mult: ATR multiplier for trailing stop

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent (of equity)
        - direction: int, 0=Both, 1=LongOnly, 2=ShortOnly

    Return Examples:
        (100.0, 0, 1)     # Buy 100 shares, long only
        (0.5, 2, 1)       # Buy with 50% of equity, long only
        (-np.inf, 2, 1)   # Close entire long position (size=-inf with Percent)
        (np.nan, 0, 0)    # No action (size=nan means no order)
    \"\"\"
    i = c.i  # Current bar index
    pos = c.position_now  # Current position (0.0 if flat)

    # Example logic structure:
    if pos == 0:  # No position - check for entry
        # ... entry conditions ...
        if should_enter:
            return (0.5, 2, 1)  # Buy with 50% of equity
    else:  # Have position - check for exit
        # ... exit conditions ...
        if should_exit:
            return (-np.inf, 2, 1)  # Close position

    return (np.nan, 0, 0)  # No action


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    \"\"\"
    Precompute all indicators. Use vectorbt indicator classes.

    Args:
        ohlcv: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        sma_period: SMA trend filter period
        atr_period: ATR period

    Returns:
        Dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'
        All values are np.ndarray of same length as input.

    Example:
        macd_ind = vbt.MACD.run(ohlcv['close'], fast_window=macd_fast,
                                slow_window=macd_slow, signal_window=macd_signal)
        return {
            'close': ohlcv['close'].values,
            'high': ohlcv['high'].values,
            'macd': macd_ind.macd.values,
            'signal': macd_ind.signal.values,
            'atr': vbt.ATR.run(ohlcv['high'], ohlcv['low'], ohlcv['close'],
                               window=atr_period).atr.values,
            'sma': vbt.MA.run(ohlcv['close'], window=sma_period).ma.values,
        }
    \"\"\"
```
"""

SIGNATURE_COMPLEX = """
## Function Signatures

```python
import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats

def order_func(
    c,
    close_a: np.ndarray,
    close_b: np.ndarray,
    hedge_ratio: np.ndarray,
    zscore: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    notional_per_leg: float
) -> tuple:
    \"\"\"
    Generate orders for pairs trading. Called by vectorbt's from_order_func.

    This is a regular Python function (NO NUMBA). Uses flexible=True (multi-asset).

    Args:
        c: vectorbt OrderContext with these key attributes:
           - c.i: current bar index (int)
           - c.col: current asset column (0=Asset A, 1=Asset B)
           - c.position_now: current position size for this asset (float)
           - c.cash_now: current cash balance (float)
        close_a: Close prices for Asset A
        close_b: Close prices for Asset B
        hedge_ratio: Rolling hedge ratio array
        zscore: Z-score of spread array
        entry_threshold: Z-score level to enter (e.g., 2.0)
        exit_threshold: Z-score level to exit (e.g., 0.0)
        stop_threshold: Z-score level for stop-loss (e.g., 3.0)
        notional_per_leg: Fixed notional per leg (e.g., 10000.0)

    Returns:
        A tuple of (size, size_type, direction):
        - size: float, order size (positive=buy, negative=sell)
        - size_type: int, 0=Amount (shares), 1=Value ($), 2=Percent
        - direction: int, 0=Both (allows long and short)

    Return Examples:
        (100.0, 0, 0)     # Buy 100 shares
        (-50.0, 0, 0)     # Sell/short 50 shares
        (-np.inf, 2, 0)   # Close entire position
        (np.nan, 0, 0)    # No action
    \"\"\"
    i = c.i  # Current bar index
    col = c.col  # 0 = Asset A, 1 = Asset B
    pos = c.position_now  # Current position for this asset

    # Example structure for pairs trading:
    z = zscore[i]
    if np.isnan(z):
        return (np.nan, 0, 0)

    # Determine shares based on notional and price
    price_a, price_b = close_a[i], close_b[i]
    shares_a = notional_per_leg / price_a
    shares_b = notional_per_leg / price_b * hedge_ratio[i]

    # Entry/exit logic depends on which asset (col) we're processing
    # ... implement pairs logic ...

    return (np.nan, 0, 0)  # No action


def compute_spread_indicators(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20
) -> dict[str, np.ndarray]:
    \"\"\"
    Precompute all indicators for pairs strategy.

    Args:
        asset_a: DataFrame with 'close' column for Asset A
        asset_b: DataFrame with 'close' column for Asset B
        hedge_lookback: Lookback for rolling OLS hedge ratio
        zscore_lookback: Lookback for z-score calculation

    Returns:
        Dict with 'close_a', 'close_b', 'hedge_ratio', 'zscore' arrays

    Example:
        close_a = asset_a['close'].values
        close_b = asset_b['close'].values

        # Rolling hedge ratio using OLS
        hedge_ratio = np.full(len(close_a), np.nan)
        for i in range(hedge_lookback, len(close_a)):
            y = close_a[i-hedge_lookback:i]
            x = close_b[i-hedge_lookback:i]
            slope, _, _, _, _ = stats.linregress(x, y)
            hedge_ratio[i] = slope

        # Spread and z-score
        spread = close_a - hedge_ratio * close_b
        spread_mean = pd.Series(spread).rolling(zscore_lookback).mean().values
        spread_std = pd.Series(spread).rolling(zscore_lookback).std().values
        zscore = (spread - spread_mean) / spread_std

        return {
            'close_a': close_a,
            'close_b': close_b,
            'hedge_ratio': hedge_ratio,
            'zscore': zscore,
        }
    \"\"\"
```
"""
