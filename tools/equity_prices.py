"""Equity prices tool - fetches historical stock data from Alpaca."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, cast

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from pydantic import BaseModel, ConfigDict, Field

from .common import get_alpaca_credentials, parse_dt, resolve_timeframe


EQUITY_TOOL_NAME = "get-equity-prices"

EQUITY_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "description": "Ticker symbol, e.g., AAPL"},
        "timeframe": {
            "type": "string",
            "description": "One of: 1Min,5Min,15Min,1Hour,1Day,1Week,1Month",
        },
        "start": {"type": "string", "description": "ISO8601 start datetime (UTC)"},
        "end": {"type": "string", "description": "ISO8601 end datetime (UTC)"},
        "limit": {
            "type": "integer",
            "description": "Max bars to return (1-10000)",
            "minimum": 1,
            "maximum": 10000,
        },
    },
    "required": ["symbol"],
    "additionalProperties": False,
}


class EquityPricesInput(BaseModel):
    """Schema for fetching equity prices from Alpaca."""

    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")
    timeframe: str = Field(
        default="1Day",
        description="One of: 1Min,5Min,15Min,1Hour,1Day,1Week,1Month",
    )
    start: Optional[str] = Field(
        default=None, description="ISO8601 start datetime (UTC), e.g., 2024-01-01T00:00:00"
    )
    end: Optional[str] = Field(
        default=None, description="ISO8601 end datetime (UTC)"
    )
    limit: int = Field(
        default=100, ge=1, le=10000, description="Maximum number of bars to return"
    )

    model_config = ConfigDict(extra="forbid")


def fetch_equity_prices(payload: EquityPricesInput) -> Dict[str, Any]:
    """Fetch historical equity prices from Alpaca."""
    api_key, api_secret = get_alpaca_credentials()

    symbol = payload.symbol.upper()
    timeframe = resolve_timeframe(payload.timeframe)

    client = StockHistoricalDataClient(api_key, api_secret)
    start_dt = parse_dt(payload.start)
    end_dt = parse_dt(payload.end)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        limit=payload.limit,
    )

    bars = client.get_stock_bars(request)
    df = cast(pd.DataFrame, getattr(bars, "df", None))
    if df is None:
        return {"symbol": symbol, "timeframe": payload.timeframe, "data": []}
    if df.empty:
        return {"symbol": symbol, "timeframe": payload.timeframe, "data": []}

    if (hasattr(df.index, "names") and len(df.index.names) > 1) or hasattr(df.index, "levels"):
        df = df.xs(symbol, level=0, drop_level=True)

    df = df.sort_index()
    records = []
    for idx, row in df.iterrows():
        if isinstance(idx, pd.Timestamp):
            ts = idx.to_pydatetime()
        elif isinstance(idx, datetime):
            ts = idx
        else:
            ts = pd.to_datetime(str(idx)).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        records.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    return {
        "symbol": symbol,
        "timeframe": payload.timeframe,
        "data": records,
    }
