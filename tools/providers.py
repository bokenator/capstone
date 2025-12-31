"""Data provider registry and implementations.

This module implements a provider registry pattern for fetching different
types of financial data (OHLCV, fundamentals) from various sources.

Phase 6: Provider registry with Alpaca for OHLCV. FMP for fundamentals
can be added later.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .common import get_alpaca_credentials, parse_dt, resolve_timeframe


class DataProvider(ABC):
    """Base interface for all data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier used in DATA_SCHEMA."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """List of data types this provider supports."""
        ...

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        data_type: str,
        start: Optional[datetime],
        end: Optional[datetime],
        frequency: str = "1Day",
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data for a symbol.

        Args:
            symbol: Ticker symbol
            data_type: Type of data (e.g., "ohlcv", "earnings")
            start: Start datetime
            end: End datetime
            frequency: Data frequency (e.g., "1Day", "1Hour")
            **kwargs: Additional parameters

        Returns:
            DataFrame with DatetimeIndex
        """
        ...


class AlpacaProvider(DataProvider):
    """Provider for OHLCV data from Alpaca Markets."""

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def supported_types(self) -> List[str]:
        return ["ohlcv"]

    def __init__(self):
        """Initialize Alpaca client."""
        api_key, api_secret = get_alpaca_credentials()
        self.client = StockHistoricalDataClient(api_key, api_secret)

    def fetch(
        self,
        symbol: str,
        data_type: str = "ohlcv",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        frequency: str = "1Day",
        limit: int = 5000,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpaca.

        Args:
            symbol: Ticker symbol
            data_type: Must be "ohlcv" for Alpaca
            start: Start datetime
            end: End datetime
            frequency: Data frequency (e.g., "1Day", "1Hour")
            limit: Maximum number of bars to fetch

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        if data_type != "ohlcv":
            raise ValueError(f"Alpaca provider only supports 'ohlcv', got '{data_type}'")

        timeframe = resolve_timeframe(frequency)
        start_dt = parse_dt(start) if isinstance(start, str) else start
        end_dt = parse_dt(end) if isinstance(end, str) else end

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
            limit=limit,
        )

        bars = self.client.get_stock_bars(request)
        df = getattr(bars, "df", None)

        if df is None or df.empty:
            raise RuntimeError(f"No data returned for symbol: {symbol}")

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0, drop_level=True)
        df = df.sort_index()
        df.columns = df.columns.str.lower()
        return df


class ProviderRegistry:
    """Registry for data providers."""

    def __init__(self):
        self._providers: Dict[str, DataProvider] = {}

    def register(self, provider: DataProvider) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider

    def get(self, name: str) -> DataProvider:
        """Get a provider by name."""
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}. Available: {list(self._providers.keys())}")
        return self._providers[name]

    def list_providers(self) -> Dict[str, Dict[str, Any]]:
        """Return provider documentation for system prompt."""
        return {
            name: {
                "types": provider.supported_types,
            }
            for name, provider in self._providers.items()
        }

    def infer_provider(self, data_schema: Dict[str, Any]) -> DataProvider:
        """Infer the appropriate provider from a data schema slot.

        Args:
            data_schema: A single slot from DATA_SCHEMA (e.g., {"frequency": "1Day", ...})

        Returns:
            The appropriate DataProvider instance
        """
        data_type = data_schema.get("data_type", "ohlcv")

        # For now, use simple type-based routing
        if data_type == "ohlcv":
            return self.get("alpaca")
        elif data_type == "fundamental":
            # FMP would go here in the future
            raise ValueError("Fundamental data provider (FMP) not yet implemented")
        else:
            raise ValueError(f"Unknown data_type: {data_type}")


# Global provider registry
PROVIDERS = ProviderRegistry()

# Register default providers
try:
    PROVIDERS.register(AlpacaProvider())
except Exception as e:
    print(f"Warning: Failed to initialize Alpaca provider: {e}")


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    return PROVIDERS


def resample_to_frequency(
    df: pd.DataFrame,
    source_freq: str,
    target_freq: str,
    data_type: str = "ohlcv",
) -> pd.DataFrame:
    """Resample DataFrame to target frequency.

    Args:
        df: DataFrame with DatetimeIndex
        source_freq: Original frequency (e.g., "1Day", "quarterly")
        target_freq: Target frequency (e.g., "1Week")
        data_type: Type of data for proper aggregation ("ohlcv" or "fundamental")

    Returns:
        Resampled DataFrame
    """
    # Map frequency strings to pandas offset aliases
    freq_map = {
        "1Min": "1min",
        "5Min": "5min",
        "15Min": "15min",
        "1Hour": "1h",
        "1Day": "1D",
        "1Week": "W-FRI",  # Week ending Friday (standard trading week)
        "1Month": "BME",   # Business month end
        "quarterly": "QE",  # Quarter end
    }

    # Frequency hierarchy for comparison (higher number = lower frequency)
    freq_order = {
        "1Min": 1,
        "5Min": 2,
        "15Min": 3,
        "1Hour": 4,
        "1Day": 5,
        "1Week": 6,
        "1Month": 7,
        "quarterly": 8,
    }

    target = freq_map.get(target_freq, target_freq)
    source_order = freq_order.get(source_freq, 5)
    target_order = freq_order.get(target_freq, 5)

    if source_order == target_order:
        # Same frequency, no resampling needed
        return df

    if source_order < target_order:
        # Downsampling: higher frequency to lower (e.g., hourly → daily)
        if data_type == "ohlcv":
            # Use proper OHLCV aggregation
            agg_dict = {}
            if 'open' in df.columns:
                agg_dict['open'] = 'first'
            if 'high' in df.columns:
                agg_dict['high'] = 'max'
            if 'low' in df.columns:
                agg_dict['low'] = 'min'
            if 'close' in df.columns:
                agg_dict['close'] = 'last'
            if 'volume' in df.columns:
                agg_dict['volume'] = 'sum'

            if agg_dict:
                return df.resample(target).agg(agg_dict).dropna()
            else:
                return df.resample(target).last().dropna()
        else:
            # For fundamental data, take last known value
            return df.resample(target).last().dropna()
    else:
        # Upsampling: lower frequency to higher (e.g., quarterly → daily)
        # Forward-fill represents "last known value until next update"
        return df.resample(target).ffill()
