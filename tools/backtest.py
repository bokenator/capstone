"""Backtest tool - runs moving-average crossover backtests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from pydantic import BaseModel, ConfigDict, Field

from .common import get_alpaca_credentials, parse_dt, resolve_timeframe


BACKTEST_TOOL_NAME = "backtest-strategy"

BACKTEST_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of ticker symbols, e.g., ['AAPL','MSFT']",
        },
        "timeframe": {
            "type": "string",
            "description": "One of: 1Min,5Min,15Min,1Hour,1Day,1Week,1Month",
        },
        "start": {"type": "string", "description": "ISO8601 start datetime (UTC)"},
        "end": {"type": "string", "description": "ISO8601 end datetime (UTC)"},
        "fast_window": {"type": "integer", "minimum": 1, "description": "Fast MA window"},
        "slow_window": {"type": "integer", "minimum": 2, "description": "Slow MA window"},
        "limit": {"type": "integer", "minimum": 10, "maximum": 10000, "description": "Max bars to return per symbol"},
    },
    "required": ["symbols"],
    "additionalProperties": False,
}


class BacktestInput(BaseModel):
    """Schema for running a simple MA crossover backtest."""

    symbols: List[str] = Field(..., description="Ticker symbols, e.g., ['AAPL','MSFT']")
    timeframe: str = Field(
        default="1Day",
        description="One of: 1Min,5Min,15Min,1Hour,1Day,1Week,1Month",
    )
    start: Optional[str] = Field(default=None, description="ISO8601 start datetime (UTC)")
    end: Optional[str] = Field(default=None, description="ISO8601 end datetime (UTC)")
    fast_window: int = Field(default=10, ge=1, description="Fast MA window")
    slow_window: int = Field(default=30, ge=2, description="Slow MA window (must be > fast)")
    limit: int = Field(default=5000, ge=10, le=10000, description="Max bars per symbol")

    model_config = ConfigDict(extra="forbid")


def ma_crossover_backtest(payload: BacktestInput) -> Dict[str, Any]:
    """Run a moving-average crossover backtest."""
    if payload.fast_window >= payload.slow_window:
        raise ValueError("fast_window must be less than slow_window")

    api_key, api_secret = get_alpaca_credentials()

    timeframe = resolve_timeframe(payload.timeframe)
    client = StockHistoricalDataClient(api_key, api_secret)
    start_dt = parse_dt(payload.start)
    end_dt = parse_dt(payload.end)

    request = StockBarsRequest(
        symbol_or_symbols=payload.symbols,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        limit=payload.limit,
    )
    bars = client.get_stock_bars(request)
    df = cast(pd.DataFrame, getattr(bars, "df", None))
    if df is None or df.empty:
        raise RuntimeError("No bars returned for symbols")

    # Normalize to MultiIndex: symbol, time
    if not isinstance(df.index, pd.MultiIndex):
        raise RuntimeError("Expected MultiIndex bars from alpaca-py")

    equity_series_list: List[pd.Series] = []
    for sym in payload.symbols:
        try:
            sym_df = df.xs(sym, level=0, drop_level=True).sort_index()
        except Exception:
            continue
        prices = pd.Series(sym_df["close"])
        fast = prices.rolling(payload.fast_window, min_periods=payload.fast_window).mean()
        slow = prices.rolling(payload.slow_window, min_periods=payload.slow_window).mean()
        signal = pd.Series((fast > slow).astype(float), index=prices.index)
        returns = pd.Series(prices.pct_change().fillna(0.0), index=prices.index)
        signal_lag = signal.shift(1).fillna(0.0)
        strat_ret = pd.Series(signal_lag * returns, index=prices.index)
        equity = (1 + strat_ret).cumprod()
        equity = pd.Series(equity, index=prices.index, name=sym)
        equity_series_list.append(equity)

    if not equity_series_list:
        raise RuntimeError("No valid data to backtest")

    equity_df: pd.DataFrame = pd.concat(equity_series_list, axis=1).dropna()
    equity_curve: pd.Series = pd.Series(equity_df.mean(axis=1), index=equity_df.index)

    records = []
    for ts, val in equity_curve.items():
        ts_dt: datetime
        if isinstance(ts, pd.Timestamp):
            ts_dt = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            ts_dt = ts
        else:
            ts_ts = pd.Timestamp(str(ts))
            if ts_ts is pd.NaT or bool(pd.isna(ts_ts)):
                continue
            ts_dt = cast(datetime, ts_ts.to_pydatetime())
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        records.append(
            {
                "timestamp": ts_dt.isoformat(),
                "open": float(val),
                "high": float(val),
                "low": float(val),
                "close": float(val),
                "volume": 0.0,
            }
        )

    return {
        "symbol": "BACKTEST",
        "timeframe": payload.timeframe,
        "data": records,
        "meta": {
            "symbols": payload.symbols,
            "fast_window": payload.fast_window,
            "slow_window": payload.slow_window,
        },
    }
