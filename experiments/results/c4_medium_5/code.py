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
    """
    Generate order at each bar. Called by vectorbt's from_order_func.

    Strategy:
      - Entry: MACD line crosses above Signal AND price > SMA
      - Exit: MACD line crosses below Signal OR price < (highest_since_entry - trailing_mult * ATR)

    Returns either an order object created by vbt.portfolio.nb.order_nb or vbt.portfolio.enums.NoOrder.
    """
    i = c.i
    pos = c.position_now

    # Require previous bar for crossover detection
    if i <= 0:
        return vbt.portfolio.enums.NoOrder

    # Ensure data is present and finite
    if not (np.isfinite(macd[i]) and np.isfinite(signal[i]) and np.isfinite(macd[i - 1]) and np.isfinite(signal[i - 1]) and np.isfinite(sma[i]) and np.isfinite(close[i]) and np.isfinite(high[i]) and np.isfinite(atr[i])):
        return vbt.portfolio.enums.NoOrder

    # Detect crossovers
    macd_cross_up = (macd[i - 1] < signal[i - 1]) and (macd[i] > signal[i])
    macd_cross_down = (macd[i - 1] > signal[i - 1]) and (macd[i] < signal[i])

    # No position: check entry
    if pos == 0.0:
        if macd_cross_up and (close[i] > sma[i]):
            # Initialize highest price since entry on context
            try:
                c.entry_highest = float(high[i]) if np.isfinite(high[i]) else float(close[i])
            except Exception:
                pass

            # Use 100% of equity for entry (size_type=2 -> Percent, direction=1 -> LongOnly)
            return vbt.portfolio.nb.order_nb(size=1.0, price=np.inf, size_type=2, direction=1)

        return vbt.portfolio.enums.NoOrder

    # In position: update highest and check exit conditions
    else:
        # Initialize or update highest since entry
        if not hasattr(c, 'entry_highest') or not np.isfinite(getattr(c, 'entry_highest', np.nan)):
            try:
                c.entry_highest = float(high[i]) if np.isfinite(high[i]) else float(close[i])
            except Exception:
                pass
        else:
            try:
                if np.isfinite(high[i]):
                    c.entry_highest = float(np.maximum(getattr(c, 'entry_highest', high[i]), high[i]))
            except Exception:
                pass

        entry_high = getattr(c, 'entry_highest', np.nan)
        threshold = np.nan
        if np.isfinite(entry_high) and np.isfinite(atr[i]):
            threshold = entry_high - trailing_mult * atr[i]

        # Exit if MACD crosses down
        if macd_cross_down:
            return vbt.portfolio.nb.order_nb(size=-np.inf, price=np.inf, size_type=2, direction=1)

        # Exit if close breaches trailing stop
        if np.isfinite(threshold) and close[i] < threshold:
            return vbt.portfolio.nb.order_nb(size=-np.inf, price=np.inf, size_type=2, direction=1)

        return vbt.portfolio.enums.NoOrder


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    """
    Precompute indicators required by the strategy using vectorbt.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("ohlcv must contain 'high' column")

    close_ser = ohlcv['close']
    high_ser = ohlcv['high']
    low_ser = ohlcv['low'] if 'low' in ohlcv.columns else ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close_ser, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high_ser, low_ser, close_ser, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close_ser, window=sma_period)

    return {
        'close': close_ser.values,
        'high': high_ser.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }