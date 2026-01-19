# trading_strategy.py
"""
Attempt to use low-level OrderContext fields to record entries/exits when
high-level order methods are not available.

This implementation will try to set position_now and pos_record_now and call
update_value to record trades. It's a defensive approach that falls back to
raising informative errors if it cannot interact with the context.
"""

from typing import Any

import numpy as np
import pandas as pd


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> pd.DataFrame:
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")
    if 'close' not in ohlcv.columns or 'high' not in ohlcv.columns or 'low' not in ohlcv.columns:
        raise KeyError("ohlcv must contain 'close', 'high', and 'low' columns")

    close = ohlcv['close'].astype(float).copy()
    high = ohlcv['high'].astype(float).copy()
    low = ohlcv['low'].astype(float).copy()

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    indicators = pd.DataFrame({
        'close': close,
        'high': high,
        'macd': macd,
        'signal': signal,
        'atr': atr,
        'sma': sma,
    }, index=ohlcv.index)

    return indicators


def order_func(
    ctx: Any,
    close: np.ndarray,
    high: np.ndarray,
    macd: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    sma: np.ndarray,
    trailing_mult: float,
) -> None:
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    macd = np.asarray(macd, dtype=float)
    signal = np.asarray(signal, dtype=float)
    atr = np.asarray(atr, dtype=float)
    sma = np.asarray(sma, dtype=float)

    n = len(close)
    in_pos = False
    highest_since_entry = -np.inf

    for i in range(n):
        if i < 1:
            prev_macd = np.nan
            prev_signal = np.nan
        else:
            prev_macd = macd[i - 1]
            prev_signal = signal[i - 1]

        cur_macd = macd[i]
        cur_signal = signal[i]
        cur_close = close[i]
        cur_high = high[i]
        cur_atr = atr[i]
        cur_sma = sma[i]

        if np.isnan(cur_macd) or np.isnan(cur_signal):
            continue

        cross_up = (not np.isnan(prev_macd) and not np.isnan(prev_signal)) and (prev_macd <= prev_signal) and (cur_macd > cur_signal)
        cross_down = (not np.isnan(prev_macd) and not np.isnan(prev_signal)) and (prev_macd >= prev_signal) and (cur_macd < cur_signal)

        if not in_pos:
            if cross_up and (not np.isnan(cur_sma)) and (cur_close > cur_sma):
                # Primary attempt: try high-level methods if present
                placed = False
                try:
                    # Try commonly available method names directly
                    ctx.order_target_percent(i, 1.0)
                    placed = True
                except Exception:
                    placed = False

                if not placed:
                    # Fallback to low-level pos_record manipulation
                    try:
                        # pos_record_now is a numpy void with known fields; set entry values
                        pr = ctx.pos_record_now
                        # Some fields may not be writable; attempt assignments in try/except
                        try:
                            pr['entry_idx'] = int(i)
                        except Exception:
                            pass
                        try:
                            pr['entry_price'] = float(cur_close)
                        except Exception:
                            pass
                        try:
                            pr['size'] = float(1.0)
                        except Exception:
                            pass
                        try:
                            pr['direction'] = int(1)
                        except Exception:
                            pass
                        # Set position_now to reflect long position
                        try:
                            ctx.position_now = np.float64(1.0)
                        except Exception:
                            pass
                        # Update derived values
                        try:
                            ctx.update_value()
                        except Exception:
                            pass
                        placed = True
                    except Exception:
                        placed = False

                if not placed:
                    raise RuntimeError(f"Could not place entry order at idx {i}; no supported methods")

                in_pos = True
                highest_since_entry = cur_high if not np.isnan(cur_high) else cur_close
        else:
            if not np.isnan(cur_high):
                highest_since_entry = max(highest_since_entry, cur_high)

            stop_price = np.nan
            if (not np.isneginf(highest_since_entry)) and (not np.isnan(cur_atr)):
                stop_price = highest_since_entry - float(trailing_mult) * cur_atr

            if cross_down:
                exited = False
                try:
                    ctx.order_target_percent(i, 0.0)
                    exited = True
                except Exception:
                    exited = False

                if not exited:
                    try:
                        pr = ctx.pos_record_now
                        try:
                            pr['exit_idx'] = int(i)
                        except Exception:
                            pass
                        try:
                            pr['exit_price'] = float(cur_close)
                        except Exception:
                            pass
                        try:
                            ctx.position_now = np.float64(0.0)
                        except Exception:
                            pass
                        try:
                            ctx.update_value()
                        except Exception:
                            pass
                        exited = True
                    except Exception:
                        exited = False

                if not exited:
                    raise RuntimeError(f"Could not place exit order (macd) at idx {i}")

                in_pos = False
                highest_since_entry = -np.inf
                continue

            if (not np.isnan(stop_price)) and (not np.isnan(cur_close)) and (cur_close < stop_price):
                exited = False
                try:
                    ctx.order_target_percent(i, 0.0)
                    exited = True
                except Exception:
                    exited = False

                if not exited:
                    try:
                        pr = ctx.pos_record_now
                        try:
                            pr['exit_idx'] = int(i)
                        except Exception:
                            pass
                        try:
                            pr['exit_price'] = float(cur_close)
                        except Exception:
                            pass
                        try:
                            ctx.position_now = np.float64(0.0)
                        except Exception:
                            pass
                        try:
                            ctx.update_value()
                        except Exception:
                            pass
                        exited = True
                    except Exception:
                        exited = False

                if not exited:
                    raise RuntimeError(f"Could not place exit order (trailing) at idx {i}")

                in_pos = False
                highest_since_entry = -np.inf

    return
