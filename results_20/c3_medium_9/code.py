# Generated strategy: MACD crossover entries + ATR-based trailing stops
# Exports: compute_indicators, order_func

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Compute indicators required by the strategy.

    Returns a dictionary with keys:
    - macd: MACD line (np.ndarray)
    - signal: MACD signal line (np.ndarray)
    - atr: Average True Range (np.ndarray)
    - sma: Simple Moving Average (np.ndarray)
    - close: close prices (np.ndarray)
    - high: high prices (np.ndarray)
    - entries: boolean array where an entry was executed in a full simulation (np.ndarray)
    - exits: boolean array where an exit was executed in a full simulation (np.ndarray)
    - highest: highest price since latest entry (np.ndarray, NaN when not in position)
    - stop: trailing stop level (np.ndarray, NaN when not in position)

    Notes:
    - Uses causal calculations (no lookahead): pandas.ewm and rolling with past data.
    - Performs a forward simulation across the whole dataset to expose internal state
      such as highest and stop levels for testing purposes.
    """
    # Validate input
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in ohlcv.columns:
            raise KeyError(f"ohlcv missing required column: {col}")

    # Ensure float dtype and copy to avoid side effects
    close_s = ohlcv["close"].astype(float).copy()
    high_s = ohlcv["high"].astype(float).copy()
    low_s = ohlcv["low"].astype(float).copy()

    # MACD: EMA fast - EMA slow, signal is EMA of MACD
    ema_fast = close_s.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # SMA (trend filter)
    sma = close_s.rolling(window=sma_period, min_periods=1).mean()

    # ATR (Wilder smoothing using ewm)
    prev_close = close_s.shift(1)
    tr_components = pd.concat(
        [
            (high_s - low_s).abs(),
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    # Avoid NaNs creeping past warmup by forward-filling only (causal)
    # Do not use backward fill as it would introduce lookahead
    macd = macd.fillna(method="ffill").fillna(0.0)
    signal = signal.fillna(method="ffill").fillna(0.0)
    sma = sma.fillna(method="ffill").fillna(close_s)
    atr = atr.fillna(method="ffill").fillna(0.0)
    close_s = close_s.fillna(method="ffill").fillna(0.0)
    high_s = high_s.fillna(method="ffill").fillna(close_s)
    low_s = low_s.fillna(method="ffill").fillna(close_s)

    # Convert to numpy arrays for faster numeric operations below
    close = close_s.values
    high = high_s.values
    macd_arr = macd.values
    signal_arr = signal.values
    atr_arr = atr.values
    sma_arr = sma.values

    n = len(close)

    # Precompute discrete MACD crosses (causal)
    macd_cross_up = np.zeros(n, dtype=bool)
    macd_cross_down = np.zeros(n, dtype=bool)
    # Start from index 1 because cross requires previous value
    for i in range(1, n):
        if np.isfinite(macd_arr[i]) and np.isfinite(signal_arr[i]) and np.isfinite(macd_arr[i - 1]) and np.isfinite(signal_arr[i - 1]):
            macd_cross_up[i] = (macd_arr[i] > signal_arr[i]) and (macd_arr[i - 1] <= signal_arr[i - 1])
            macd_cross_down[i] = (macd_arr[i] < signal_arr[i]) and (macd_arr[i - 1] >= signal_arr[i - 1])

    # Simulate full-series trades to expose internal state (entries/exits/highest/stop)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    in_position = np.zeros(n, dtype=bool)
    highest = np.full(n, np.nan)
    stop = np.full(n, np.nan)

    i = 1
    while i < n:
        # Entry condition must satisfy MACD cross-up and price above SMA
        try_entry = False
        if macd_cross_up[i] and (close[i] > sma_arr[i]):
            try_entry = True

        if try_entry:
            entries[i] = True
            # Enter position at bar i
            cur_high = high[i]
            # mark position as active at entry bar
            in_position[i] = True
            highest[i] = cur_high
            stop[i] = cur_high - 2.0 * atr_arr[i] if np.isfinite(atr_arr[i]) else np.nan

            j = i + 1
            exited = False
            while j < n:
                # update highest using intrabar high
                cur_high = max(cur_high, high[j])
                in_position[j] = True
                highest[j] = cur_high
                stop[j] = cur_high - 2.0 * atr_arr[j] if np.isfinite(atr_arr[j]) else np.nan

                # Check MACD bearish cross exit
                if macd_cross_down[j]:
                    exits[j] = True
                    exited = True
                    i = j + 1
                    break

                # Check trailing stop exit: price falls below (highest_since_entry - 2*ATR)
                if np.isfinite(atr_arr[j]) and (close[j] < (cur_high - 2.0 * atr_arr[j])):
                    exits[j] = True
                    exited = True
                    i = j + 1
                    break

                j += 1

            if not exited:
                # Position remains open until end of data
                break
        else:
            i += 1

    result: Dict[str, np.ndarray] = {
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
        "close": close,
        "high": high,
        # extras for testing internal state
        "entries": entries,
        "exits": exits,
        "in_position": in_position,
        "highest": highest,
        "stop": stop,
    }

    return result


def order_func(
    c: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> Tuple[float, int, int]:
    """
    Order function to be used with vectorbt.Portfolio.from_order_func.

    Returns a tuple (size, size_type, direction) where:
    - size: number of units to trade (float), np.nan to indicate no action
    - size_type: integer corresponding to SizeType enum (use 0 == Amount)
    - direction: integer corresponding to Direction enum (1 == Long, 2 == Short)

    Implementation details:
    - Stateless: all decisions derive from provided arrays up to the current index c.i
    - Simulates strategy deterministically up to the current index to decide whether
      an entry or exit should be placed on this bar.
    - Long-only: Entries are long orders (direction=1), exits are short orders (direction=2)

    Important: Do NOT import or use vectorbt enums/nb functions here. Return simple tuples.
    """
    # Convert to numpy arrays if pandas passed in
    close = np.asarray(close)
    high = np.asarray(high)
    macd = np.asarray(macd)
    signal = np.asarray(signal)
    atr = np.asarray(atr)
    sma = np.asarray(sma)

    # Attempt to get current index from context
    if not hasattr(c, "i"):
        # If context does not provide index, no-op
        return (float("nan"), 0, 0)

    idx = int(c.i)
    n = len(close)

    # Safety checks
    if idx < 1 or idx >= n:
        return (float("nan"), 0, 0)

    # Build cross arrays up to current index (causal)
    # We only need to inspect values up to idx, so slice arrays
    end = idx + 1
    macd_seg = macd[:end]
    signal_seg = signal[:end]
    close_seg = close[:end]
    high_seg = high[:end]
    atr_seg = atr[:end]
    sma_seg = sma[:end]

    seg_len = len(close_seg)
    macd_cross_up = np.zeros(seg_len, dtype=bool)
    macd_cross_down = np.zeros(seg_len, dtype=bool)
    for i in range(1, seg_len):
        if np.isfinite(macd_seg[i]) and np.isfinite(signal_seg[i]) and np.isfinite(macd_seg[i - 1]) and np.isfinite(signal_seg[i - 1]):
            macd_cross_up[i] = (macd_seg[i] > signal_seg[i]) and (macd_seg[i - 1] <= signal_seg[i - 1])
            macd_cross_down[i] = (macd_seg[i] < signal_seg[i]) and (macd_seg[i - 1] >= signal_seg[i - 1])

    # Simulate trades forward up to current index. This determines whether an order
    # should be placed on the current bar based only on past and present data.
    entries = np.zeros(seg_len, dtype=bool)
    exits = np.zeros(seg_len, dtype=bool)

    i = 1
    while i < seg_len:
        entry_cond = False
        if macd_cross_up[i] and (close_seg[i] > sma_seg[i]):
            entry_cond = True

        if entry_cond:
            entries[i] = True
            cur_high = high_seg[i]
            j = i + 1
            exited = False
            while j < seg_len:
                cur_high = max(cur_high, high_seg[j])
                # MACD bearish cross triggers exit
                if macd_cross_down[j]:
                    exits[j] = True
                    exited = True
                    i = j + 1
                    break
                # Trailing stop check (use ATR at time j)
                if np.isfinite(atr_seg[j]) and (close_seg[j] < (cur_high - trailing_mult * atr_seg[j])):
                    exits[j] = True
                    exited = True
                    i = j + 1
                    break
                j += 1
            if not exited:
                # Open position persists until current bar; stop simulation
                break
        else:
            i += 1

    # Place order on the current index if simulation indicates entry or exit here
    if entries[idx]:
        # Enter long: size 1.0 (amount), direction=1 (Long)
        return (1.0, 0, 1)
    if exits[idx]:
        # Exit long: sell 1.0 (amount) using Short direction to close long
        return (1.0, 0, 2)

    # No action
    return (float("nan"), 0, 0)
