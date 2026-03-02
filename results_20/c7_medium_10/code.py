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
    trailing_mult: float,
) -> tuple:
    """
    Generate orders based on MACD crossover entries with ATR-based trailing stops.

    This implementation avoids attaching arbitrary attributes to the OrderContext
    (which is not allowed). Instead, it reads the position record available in
    `c.pos_record_now` (or falls back to `c.last_lidx`) to determine the entry
    index and computes the highest price since entry on the fly.

    Entry:
      - MACD crosses above signal (current > signal and previous <= previous signal)
      - Price (close) is above 50-period SMA

    Exit (either):
      - MACD crosses below signal
      - Price falls below (highest_since_entry - trailing_mult * ATR)

    Returns tuple (size, size_type, direction) as required by the runner.
    """
    i = int(c.i)
    pos = float(c.position_now)

    # No action on first bar
    if i <= 0:
        return (np.nan, 0, 0)

    # Helpers
    def valid(arr, idx):
        return 0 <= idx < len(arr) and not np.isnan(arr[idx])

    def _try_get_entry_idx_from_pos_record(ctx):
        """Try multiple ways to extract initial entry index from the position record."""
        pr = getattr(ctx, "pos_record_now", None)
        # Try structured field access if available
        try:
            if pr is not None:
                # If pr is a numpy.void with named fields
                dtype = getattr(pr, "dtype", None)
                names = getattr(dtype, "names", None) if dtype is not None else None
                if names:
                    # Common candidate field names
                    candidates = ("init_i", "init_idx", "entry_i", "entry_idx", "initIndex", "init")
                    for nm in candidates:
                        if nm in names:
                            try:
                                val = pr[nm]
                                if np.isfinite(val) and int(val) >= 0:
                                    return int(val)
                            except Exception:
                                pass
                    # Fallback: look for any field containing 'init' or 'entry'
                    for nm in names:
                        if "init" in nm.lower() or "entry" in nm.lower():
                            try:
                                val = pr[nm]
                                if np.isfinite(val) and int(val) >= 0:
                                    return int(val)
                            except Exception:
                                pass
                # Try attribute access
                for attr in ("init_i", "init_idx", "entry_i", "entry_idx"):
                    if hasattr(pr, attr):
                        try:
                            val = getattr(pr, attr)
                            if np.isfinite(val) and int(val) >= 0:
                                return int(val)
                        except Exception:
                            pass
        except Exception:
            pass

        # Fallback: try c.last_lidx (last long index)
        if hasattr(ctx, "last_lidx"):
            try:
                last_lidx = getattr(ctx, "last_lidx")
                col = getattr(ctx, "col", 0)
                if isinstance(last_lidx, np.ndarray):
                    # If 1D array, try indexing by column
                    try:
                        candidate = int(last_lidx[col])
                    except Exception:
                        # last_lidx might be a scalar array
                        candidate = int(np.asarray(last_lidx).item())
                else:
                    candidate = int(last_lidx)
                if candidate >= 0:
                    return candidate
            except Exception:
                pass

        return None

    # ENTRY: only when flat
    if pos == 0.0:
        # Need valid MACD/signal for current & previous, and SMA
        if (
            valid(macd, i)
            and valid(signal, i)
            and valid(macd, i - 1)
            and valid(signal, i - 1)
            and valid(sma, i)
            and valid(close, i)
        ):
            macd_curr = float(macd[i])
            sig_curr = float(signal[i])
            macd_prev = float(macd[i - 1])
            sig_prev = float(signal[i - 1])

            macd_cross_up = (macd_curr > sig_curr) and (macd_prev <= sig_prev)
            above_sma = float(close[i]) > float(sma[i])

            if macd_cross_up and above_sma:
                # Enter long with 50% of equity
                return (0.5, 2, 1)

        return (np.nan, 0, 0)

    # EXIT: when in a long position
    else:
        # Obtain entry index for this open position
        entry_idx = _try_get_entry_idx_from_pos_record(c)

        # If we have a valid entry index and valid ATR, compute trailing stop
        if entry_idx is not None and entry_idx <= i and valid(atr, i):
            # Compute highest price since entry (inclusive). Use nan-safe max.
            try:
                # Slice may be empty in edge cases; guard that
                slice_high = high[entry_idx : i + 1]
                if slice_high.size > 0:
                    highest_since_entry = np.nanmax(slice_high)
                else:
                    highest_since_entry = float(high[i]) if valid(high, i) else float(close[i])

                if not np.isnan(highest_since_entry):
                    trail_stop = float(highest_since_entry) - float(trailing_mult) * float(atr[i])
                    if valid(close, i) and not np.isnan(trail_stop) and float(close[i]) < trail_stop:
                        return (-np.inf, 2, 1)
            except Exception:
                # If anything goes wrong computing trailing stop, ignore and rely on MACD exit
                pass

        # MACD bearish crossover: exit
        if (
            valid(macd, i)
            and valid(signal, i)
            and valid(macd, i - 1)
            and valid(signal, i - 1)
        ):
            macd_curr = float(macd[i])
            sig_curr = float(signal[i])
            macd_prev = float(macd[i - 1])
            sig_prev = float(signal[i - 1])

            macd_cross_down = (macd_curr < sig_curr) and (macd_prev >= sig_prev)
            if macd_cross_down:
                return (-np.inf, 2, 1)

        return (np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14,
) -> dict[str, np.ndarray]:
    """
    Precompute indicators needed by the strategy using vectorbt's indicators.

    Returns dict with keys: 'close', 'high', 'macd', 'signal', 'atr', 'sma'.
    """
    # Validate required columns
    required = ["high", "close"]
    missing = [c for c in required if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {missing}")

    close_sr = ohlcv["close"].astype(float)
    high_sr = ohlcv["high"].astype(float)
    low_sr = ohlcv["low"].astype(float) if "low" in ohlcv.columns else ohlcv["close"].astype(float)

    # MACD
    macd_ind = vbt.MACD.run(close_sr, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    macd_arr = np.asarray(macd_ind.macd.values, dtype=float)
    signal_arr = np.asarray(macd_ind.signal.values, dtype=float)

    # ATR
    atr_ind = vbt.ATR.run(high_sr, low_sr, close_sr, window=atr_period)
    atr_arr = np.asarray(atr_ind.atr.values, dtype=float)

    # SMA
    sma_ind = vbt.MA.run(close_sr, window=sma_period)
    sma_arr = np.asarray(sma_ind.ma.values, dtype=float)

    return {
        "close": np.asarray(close_sr.values, dtype=float),
        "high": np.asarray(high_sr.values, dtype=float),
        "macd": macd_arr,
        "signal": signal_arr,
        "atr": atr_arr,
        "sma": sma_arr,
    }
