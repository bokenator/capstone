import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Callable, Optional


class TrailingState:
    __slots__ = ("in_position", "highest", "entry_index")

    def __init__(self) -> None:
        self.in_position: bool = False
        self.highest: float = np.nan
        self.entry_index: int | None = None


_TRAILING_STATE = TrailingState()


# Try to detect vectorbt's Order class and build a factory to instantiate it.
OrderFactoryType = Optional[Callable[[float, float, int, int], object]]
_ORDER_FACTORY: OrderFactoryType = None


def _build_order_factory() -> OrderFactoryType:
    candidates = []

    # candidate locations
    try:
        import vectorbt.portfolio.base as vbt_portfolio_base
    except Exception:
        vbt_portfolio_base = None

    try:
        import vectorbt.portfolio.order as vbt_portfolio_order
    except Exception:
        vbt_portfolio_order = None

    # Collect potential Order classes
    if vbt_portfolio_base is not None and hasattr(vbt_portfolio_base, 'Order'):
        candidates.append(getattr(vbt_portfolio_base, 'Order'))
    if vbt_portfolio_order is not None and hasattr(vbt_portfolio_order, 'Order'):
        cls_tmp = getattr(vbt_portfolio_order, 'Order')
        if cls_tmp not in candidates:
            candidates.append(cls_tmp)
    # Check attributes on vbt
    try:
        if hasattr(vbt, 'Order'):
            cls_tmp = getattr(vbt, 'Order')
            if cls_tmp not in candidates:
                candidates.append(cls_tmp)
    except Exception:
        pass
    try:
        cls_tmp = getattr(vbt.Portfolio, 'Order', None)
        if cls_tmp is not None and cls_tmp not in candidates:
            candidates.append(cls_tmp)
    except Exception:
        pass

    # Helper to attempt creating a factory for a given class
    def try_cls(cls) -> Optional[Callable[[float, float, int, int], object]]:
        # Try common kwarg patterns
        kw_patterns = [
            ('size', 'price', 'size_type', 'direction'),
            ('size', 'price', 'size_type', 'side'),
            ('size', 'size_type', 'price', 'direction'),
            ('price', 'size', 'size_type', 'direction'),
            ('size', 'price', 'side', 'size_type'),
        ]
        for pat in kw_patterns:
            def factory(size, price, size_type, direction, _cls=cls, _pat=pat):
                kwargs = {_pat[0]: size, _pat[1]: price, _pat[2]: size_type, _pat[3]: direction}
                return _cls(**kwargs)  # type: ignore
            try:
                o = factory(0.0, np.inf, 2, 1)
                # Accept if object has .price attribute
                if hasattr(o, 'price'):
                    return factory
            except Exception:
                continue

        # Try positional argument permutations (limited set)
        pos_perms = [
            (0, 1, 2, 3),
            (0, 2, 1, 3),
            (1, 0, 2, 3),
            (1, 2, 0, 3),
            (2, 0, 1, 3),
        ]
        for perm in pos_perms:
            def factory_pos(size, price, size_type, direction, _cls=cls, _perm=perm):
                args = [None, None, None, None]
                args[_perm[0]] = size
                args[_perm[1]] = price
                args[_perm[2]] = size_type
                args[_perm[3]] = direction
                return _cls(*args)  # type: ignore
            try:
                o = factory_pos(0.0, np.inf, 2, 1)
                if hasattr(o, 'price'):
                    return factory_pos
            except Exception:
                continue

        return None

    for cls in candidates:
        try:
            fac = try_cls(cls)
            if fac is not None:
                return fac
        except Exception:
            continue

    return None


# Build factory at import time
try:
    _ORDER_FACTORY = _build_order_factory()
except Exception:
    _ORDER_FACTORY = None


def _make_order(size: float, price: float, size_type: int, direction: int):
    """Create an order object using detected factory or fallback to tuple."""
    if _ORDER_FACTORY is not None:
        try:
            return _ORDER_FACTORY(size, price, size_type, direction)
        except Exception:
            pass
    # Fallback tuple format (size, price, size_type, direction)
    return (size, price, size_type, direction)


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
        An order-like object or tuple in the format (size, price, size_type, direction).
    """
    global _TRAILING_STATE

    i = int(c.i)  # Current bar index
    pos = float(c.position_now)  # Current position (0.0 if flat)

    # Reset state at the start of a run to avoid leakage between runs
    if i == 0:
        _TRAILING_STATE.in_position = False
        _TRAILING_STATE.highest = np.nan
        _TRAILING_STATE.entry_index = None
        try:
            order_func._state = _TRAILING_STATE
        except Exception:
            pass

    # Safely extract current values with NaN handling
    price = float(close[i]) if np.isfinite(close[i]) else np.nan
    high_p = float(high[i]) if np.isfinite(high[i]) else (price if np.isfinite(price) else np.nan)
    macd_i = macd[i]
    signal_i = signal[i]
    atr_i = atr[i]
    sma_i = sma[i]

    # Helper to detect cross up/down using previous bar
    def cross_up(arr1: np.ndarray, arr2: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = arr1[idx - 1], arr2[idx - 1]
        a_curr, b_curr = arr1[idx], arr2[idx]
        if not (np.isfinite(a_prev) and np.isfinite(b_prev) and np.isfinite(a_curr) and np.isfinite(b_curr)):
            return False
        return (a_prev <= b_prev) and (a_curr > b_curr)

    def cross_down(arr1: np.ndarray, arr2: np.ndarray, idx: int) -> bool:
        if idx <= 0:
            return False
        a_prev, b_prev = arr1[idx - 1], arr2[idx - 1]
        a_curr, b_curr = arr1[idx], arr2[idx]
        if not (np.isfinite(a_prev) and np.isfinite(b_prev) and np.isfinite(a_curr) and np.isfinite(b_curr)):
            return False
        return (a_prev >= b_prev) and (a_curr < b_curr)

    # No position: check entry conditions
    if pos == 0.0:
        enter = False
        if cross_up(macd, signal, i):
            # Trend filter: price above SMA
            if np.isfinite(sma_i) and np.isfinite(price) and (price > sma_i):
                enter = True

        if enter:
            # Initialize trailing state on entry
            _TRAILING_STATE.in_position = True
            _TRAILING_STATE.highest = high_p if np.isfinite(high_p) else price
            _TRAILING_STATE.entry_index = i
            try:
                order_func._state = _TRAILING_STATE
            except Exception:
                pass

            # Enter long using a percent of equity (use most capital but leave a tiny buffer)
            return _make_order(0.99, np.inf, 2, 1)

        # No entry -> no order
        return _make_order(np.nan, np.nan, 0, 0)

    # Have position: check exits and update trailing high
    else:
        # Update highest price since entry
        if np.isfinite(high_p):
            if (not np.isfinite(_TRAILING_STATE.highest)) or (high_p > float(_TRAILING_STATE.highest)):
                _TRAILING_STATE.highest = high_p
                try:
                    order_func._state = _TRAILING_STATE
                except Exception:
                    pass

        # Compute trailing stop level
        trailing_stop = np.nan
        highest = _TRAILING_STATE.highest
        if np.isfinite(highest) and np.isfinite(atr_i):
            trailing_stop = float(highest) - float(trailing_mult) * float(atr_i)

        # Exit conditions: MACD cross down OR price breaks trailing stop
        exit_by_macd = cross_down(macd, signal, i)
        exit_by_trail = np.isfinite(trailing_stop) and np.isfinite(price) and (price < trailing_stop)

        if exit_by_macd or exit_by_trail:
            # Reset state
            _TRAILING_STATE.in_position = False
            _TRAILING_STATE.highest = np.nan
            _TRAILING_STATE.entry_index = None
            try:
                order_func._state = _TRAILING_STATE
            except Exception:
                pass

            # Close entire long position using percent sentinel (-inf)
            return _make_order(-np.inf, np.inf, 2, 1)

        # Otherwise hold -> no order
        return _make_order(np.nan, np.nan, 0, 0)


def compute_indicators(
    ohlcv: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 50,
    atr_period: int = 14
) -> dict[str, np.ndarray]:
    """
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
    """
    # Validate required columns
    if 'close' not in ohlcv.columns:
        raise ValueError("Input DataFrame must contain 'close' column")
    if 'high' not in ohlcv.columns:
        raise ValueError("Input DataFrame must contain 'high' column")

    close = ohlcv['close']
    high = ohlcv['high']

    # Some datasets may lack 'low' - fall back to close if missing
    if 'low' in ohlcv.columns:
        low = ohlcv['low']
    else:
        # Use close as fallback; this will make ATR zero, but keeps the pipeline working
        low = ohlcv['close']

    # Compute MACD
    macd_ind = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)

    # Compute ATR
    atr_ind = vbt.ATR.run(high, low, close, window=atr_period)

    # Compute SMA
    sma_ind = vbt.MA.run(close, window=sma_period)

    return {
        'close': close.values,
        'high': high.values,
        'macd': macd_ind.macd.values,
        'signal': macd_ind.signal.values,
        'atr': atr_ind.atr.values,
        'sma': sma_ind.ma.values,
    }


# Expose state for tests
try:
    order_func._state = _TRAILING_STATE
except Exception:
    pass

# Expose order factory for tests/debugging
try:
    order_func._order_factory = _ORDER_FACTORY
except Exception:
    pass
