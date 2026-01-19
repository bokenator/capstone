from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_signals(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Generate position signals for a single-asset RSI mean reversion strategy.

    Strategy rules:
    - Compute RSI with period=14 (can be overridden via params['rsi_period'])
    - Enter long when RSI crosses below entry_threshold (default 30)
    - Exit long when RSI crosses above exit_threshold (default 70)

    Args:
        data: Dict containing at least an "ohlcv" pd.DataFrame with a "close" column.
        params: Dict of parameters. Supported keys:
            - rsi_period: int (default 14)
            - entry_threshold: float (default 30.0)
            - exit_threshold: float (default 70.0)

    Returns:
        Dict with key "ohlcv" mapping to a pd.Series of position targets (+1 for long, 0 for flat).
    """

    # Validate inputs
    if "ohlcv" not in data:
        raise ValueError("data must contain 'ohlcv' DataFrame")
    ohlcv = data["ohlcv"]
    if "close" not in ohlcv:
        raise ValueError("ohlcv DataFrame must contain 'close' column")

    close = ohlcv["close"]

    # Parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    entry_threshold = float(params.get("entry_threshold", 30.0))
    exit_threshold = float(params.get("exit_threshold", 70.0))

    # Compute RSI using vectorbt's implementation
    # Fully-qualified call as required by VAS
    rsi = vbt.RSI.run(close, window=rsi_period).rsi

    # Prepare numpy arrays for fast processing
    n = len(rsi)
    rsi_prev = pd.Series.shift(rsi, 1)  # shift previous values

    rsi_vals = rsi.values
    rsi_prev_vals = rsi_prev.values

    # Initialize position array: 1 for long, 0 for flat
    pos = np.zeros(n, dtype=np.int8)

    in_pos = False
    for i in range(n):
        cur = rsi_vals[i]
        prev = rsi_prev_vals[i]

        entered = False
        exited = False

        # Detect crossing below entry threshold: prev >= entry and cur < entry
        if (np.isfinite(prev) and np.isfinite(cur)):
            if (prev >= entry_threshold) and (cur < entry_threshold) and (not in_pos):
                in_pos = True
                entered = True

            # Detect crossing above exit threshold: prev <= exit and cur > exit
            if in_pos and (prev <= exit_threshold) and (cur > exit_threshold):
                in_pos = False
                exited = True

        # Set position for this bar
        pos[i] = 1 if in_pos else 0

        # Note: entries/exits happen on the same bar index where crossing occurs

    position_series = pd.Series(pos, index=close.index, dtype=int)

    return {"ohlcv": position_series}
