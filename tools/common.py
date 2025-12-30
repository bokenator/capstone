"""Shared utilities for MCP tools."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Optional

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


_TF_MINUTE: TimeFrameUnit = TimeFrameUnit("Min")
_TF_HOUR: TimeFrameUnit = TimeFrameUnit("Hour")
_TF_DAY: TimeFrameUnit = TimeFrameUnit("Day")
_TF_WEEK: TimeFrameUnit = TimeFrameUnit("Week")
_TF_MONTH: TimeFrameUnit = TimeFrameUnit("Month")

TIMEFRAME_PARAMS: Dict[str, tuple[int, TimeFrameUnit]] = {
    "1Min": (1, _TF_MINUTE),
    "5Min": (5, _TF_MINUTE),
    "15Min": (15, _TF_MINUTE),
    "1Hour": (1, _TF_HOUR),
    "1Day": (1, _TF_DAY),
    "1Week": (1, _TF_WEEK),
    "1Month": (1, _TF_MONTH),
}


def parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO8601 datetime string to a UTC datetime."""
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def resolve_timeframe(name: str) -> TimeFrame:
    """Convert a timeframe name to an Alpaca TimeFrame object."""
    params = TIMEFRAME_PARAMS.get(name)
    if params is None:
        raise ValueError(f"Unsupported timeframe: {name}")
    amount, unit = params
    return TimeFrame(amount=amount, unit=unit)


def get_alpaca_credentials() -> tuple[str, str]:
    """Get Alpaca API credentials from environment."""
    api_key = os.getenv("ALPACA_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_KEY_ID or ALPACA_SECRET_KEY in environment/.env")
    return api_key, api_secret
