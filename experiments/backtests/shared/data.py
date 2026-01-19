"""
Sample Data Generation
======================

Functions to generate synthetic data for backtesting.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def load_sample_data(
    complexity: str,
    num_bars: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Load or generate sample data for backtesting.

    Args:
        complexity: Strategy complexity level ("simple", "medium", "complex")
        num_bars: Number of bars to generate
        seed: Random seed for reproducibility

    Returns:
        Dict mapping slot names to DataFrames per DATA_SCHEMA
    """
    np.random.seed(seed)

    # Generate date index
    dates = pd.date_range(end=datetime.now(), periods=num_bars, freq="D")

    if complexity == "simple":
        # Simple: single OHLCV series
        return _generate_ohlcv_data(dates)

    elif complexity == "medium":
        # Medium: single OHLCV with high required for ATR
        return _generate_ohlcv_data(dates, include_high_low=True)

    elif complexity == "complex":
        # Complex: two correlated assets for pairs trading
        return _generate_pairs_data(dates)

    else:
        raise ValueError(f"Unknown complexity: {complexity}")


def _generate_ohlcv_data(
    dates: pd.DatetimeIndex,
    include_high_low: bool = False,
    start_price: float = 100.0,
    volatility: float = 0.02,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data."""
    n = len(dates)

    # Generate log returns with some autocorrelation
    returns = np.random.normal(0.0003, volatility, n)  # Slight positive drift

    # Generate close prices
    log_prices = np.log(start_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    # Generate OHLCV
    high_low_range = volatility * 0.5
    high = close * (1 + np.abs(np.random.normal(0, high_low_range, n)))
    low = close * (1 - np.abs(np.random.normal(0, high_low_range, n)))
    open_price = close * (1 + np.random.normal(0, high_low_range / 2, n))

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Volume with some randomness
    volume = np.random.lognormal(15, 0.5, n)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return {"ohlcv": df}


def _generate_pairs_data(
    dates: pd.DatetimeIndex,
    start_price_a: float = 100.0,
    start_price_b: float = 50.0,
    correlation: float = 0.85,
    volatility: float = 0.02,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic data for pairs trading (two correlated assets)."""
    n = len(dates)

    # Generate correlated returns
    # Asset A returns
    returns_a = np.random.normal(0.0003, volatility, n)

    # Asset B returns (correlated with A)
    independent = np.random.normal(0.0002, volatility, n)
    returns_b = correlation * returns_a + np.sqrt(1 - correlation**2) * independent

    # Add some mean-reversion in the spread (for pairs trading to work)
    spread = np.cumsum(returns_a - returns_b)
    mean_reversion = -0.05 * spread  # Pull spread back toward zero
    returns_b = returns_b + mean_reversion * 0.1

    # Generate prices
    close_a = start_price_a * np.exp(np.cumsum(returns_a))
    close_b = start_price_b * np.exp(np.cumsum(returns_b))

    df_a = pd.DataFrame({"close": close_a}, index=dates)
    df_b = pd.DataFrame({"close": close_b}, index=dates)

    return {
        "asset_a": df_a,
        "asset_b": df_b,
    }
