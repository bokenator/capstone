"""Backtest tool - AI-powered strategy generation and backtesting.

This module provides the MCP tool that uses Codex as an agentic coding tool
to iteratively generate and fix strategy code until it executes successfully.

Workflow (v2 - Single Strategy):
1. Codex generates strategy code from natural language prompt
2. Code is validated for safety
3. Code is executed in sandbox with vectorbt
4. If error, error details are fed back to Codex (same conversation context)
5. Codex fixes the code
6. Repeat until success or max attempts

Workflow (v3 - Multi-Strategy):
1. Strategy Planner parses user intent -> List[StrategySpec]
2. For each StrategySpec:
   a. Strategy Generator creates code -> GeneratedStrategy
   b. For each direction in spec.directions:
      - Backtest Engine runs strategy -> BacktestRunResult
3. Result Combiner merges all results -> CombinedBacktestResult
4. Widget renders unified chart with all equity curves
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from pydantic import BaseModel, ConfigDict, Field

from .common import get_alpaca_credentials, parse_dt, resolve_timeframe
from .schemas import extract_defaults_from_param_schema, merge_params
from .strategy_executor import BacktestResult, StrategyExecutor, get_strategy_executor
from .strategy_generator import CodexAgent, CodexSession, get_codex_agent
from .providers import get_provider_registry, resample_to_frequency

# v3 imports
from .models import (
    CombinedBacktestResult,
    DisplayConfig,
    GeneratedStrategy,
    PlannerOutput,
    StrategySpec,
)
from .strategy_planner import StrategyPlanner, get_strategy_planner
from .result_combiner import combine_results, downsample_combined_result

# Configure logging
logger = logging.getLogger(__name__)


BACKTEST_TOOL_NAME = "backtest"

# =============================================================================
# V3 Multi-Strategy Tool (preferred)
# =============================================================================

MULTI_BACKTEST_TOOL_NAME = "multi_backtest"

MULTI_BACKTEST_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": """CRITICAL: Pass the user's EXACT request verbatim. DO NOT rephrase, simplify, or split into multiple calls.

This tool handles complex multi-strategy requests automatically, including:
- Multiple strategies (e.g., "RSI vs MACD on SPY")
- Benchmark comparisons (e.g., "momentum strategy vs buy-and-hold SPY")
- Direction comparisons (e.g., "show long-only vs long-short")
- Multi-symbol strategies (e.g., "pairs trading on GLD and SLV")

The tool's internal planner will parse the user's intent and generate all strategies.

Examples of prompts to pass EXACTLY as the user said them:
- "Backtest RSI on AAPL with buy-and-hold SPY as benchmark"
- "Compare MA crossover vs MACD on SPY, show both long-only and long-short"
- "Valuation strategy on AAPL and MSFT, go long when PE < 50, also show buy-and-hold SPY"

DO NOT translate these into separate tool calls. Pass the full request.""",
        },
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Default symbols if not specified in prompt. The planner will extract symbols from the prompt if mentioned there.",
        },
        "timeframe": {
            "type": "string",
            "description": "Data timeframe: 1Day (default), 1Hour, 1Week, etc.",
            "default": "1Day",
        },
        "start": {
            "type": "string",
            "description": "ISO8601 start datetime (UTC), e.g., 2024-01-01",
        },
        "end": {
            "type": "string",
            "description": "ISO8601 end datetime (UTC)",
        },
        "init_cash": {
            "type": "number",
            "description": "Initial cash per strategy (default $100)",
            "default": 100,
        },
        "normalize": {
            "type": "boolean",
            "description": "Normalize all equity curves to start at 100 for fair comparison",
            "default": False,
        },
        "show_trades": {
            "type": "boolean",
            "description": "Show trade markers on the chart",
            "default": False,
        },
    },
    "required": ["prompt"],
    "additionalProperties": False,
}


class MultiBacktestInput(BaseModel):
    """Schema for the v3 multi-strategy backtest tool."""

    prompt: str = Field(
        ...,
        description="User's EXACT request - do not rephrase. Can include multiple strategies and benchmarks.",
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Default symbols if not in prompt",
    )
    timeframe: str = Field(
        default="1Day",
        description="Data timeframe",
    )
    start: Optional[str] = Field(
        default=None,
        description="ISO8601 start datetime",
    )
    end: Optional[str] = Field(
        default=None,
        description="ISO8601 end datetime",
    )
    init_cash: float = Field(
        default=100.0,
        ge=1,
        description="Initial cash per strategy",
    )
    normalize: bool = Field(
        default=False,
        description="Normalize curves to start at 100",
    )
    show_trades: bool = Field(
        default=False,
        description="Show trade markers",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# V2 Single-Strategy Tool (legacy, still supported)
# =============================================================================

BACKTEST_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "Natural language description of the trading strategy to generate and test. Example: 'Buy when RSI drops below 30, sell when it rises above 70'. When modifying an existing strategy via base_code, describe only the changes to make.",
        },
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of ticker symbols to backtest, e.g., ['AAPL', 'MSFT']. First symbol maps to 'prices' slot.",
        },
        "symbol_mapping": {
            "type": "object",
            "description": "Explicit mapping of data slot names to symbols (e.g., {'prices': 'SPY'} or {'asset_a': 'GLD', 'asset_b': 'SLV'}).",
        },
        "base_code": {
            "type": "string",
            "description": "IMPORTANT: When the user wants to modify, update, or build upon a previous strategy, you MUST pass the previous strategy's generated code here. Look for the code in the 'Parameters for Reproducibility' section of the previous backtest result. If the user says 'same strategy', 'modify', 'update', 'add shorts', 'change parameters', or references a previous backtest, extract and pass that code here. The prompt should then describe only the modifications.",
        },
        "params": {
            "type": "object",
            "description": "Optional strategy parameters to override defaults (e.g., {'fast_window': 10, 'slow_window': 30}).",
        },
        "direction": {
            "oneOf": [
                {
                    "type": "string",
                    "enum": ["longonly", "shortonly", "both"],
                },
                {
                    "type": "array",
                    "items": {"type": "string", "enum": ["longonly", "shortonly", "both"]},
                },
            ],
            "description": "Position direction mode(s): 'longonly' (default), 'shortonly', or 'both'. Can be an array for comparison runs (e.g., ['longonly', 'both']).",
            "default": "longonly",
        },
        "execution_price": {
            "type": "string",
            "description": "Price column to use for trade execution: 'close' (default) or 'open'.",
            "default": "close",
            "enum": ["open", "close"],
        },
        "stop_loss": {
            "type": "number",
            "description": "Stop loss as fraction of entry price (e.g., 0.05 = 5%). Set to null to disable.",
            "minimum": 0.001,
            "maximum": 0.5,
        },
        "take_profit": {
            "type": "number",
            "description": "Take profit as fraction of entry price (e.g., 0.10 = 10%). Set to null to disable.",
            "minimum": 0.001,
            "maximum": 2.0,
        },
        "trailing_stop": {
            "type": "boolean",
            "description": "If true, stop_loss trails the highest price since entry. Requires stop_loss to be set.",
            "default": False,
        },
        "slippage": {
            "type": "number",
            "description": "Slippage as fraction of price per trade (e.g., 0.001 = 0.1% = 10 bps).",
            "default": 0.0,
            "minimum": 0,
            "maximum": 0.05,
        },
        "timeframe": {
            "type": "string",
            "description": "Data timeframe: 1Min, 5Min, 15Min, 1Hour, 1Day, 1Week, 1Month",
            "default": "1Day",
        },
        "start": {
            "type": "string",
            "description": "ISO8601 start datetime (UTC), e.g., 2024-01-01",
        },
        "end": {
            "type": "string",
            "description": "ISO8601 end datetime (UTC)",
        },
        "init_cash": {
            "type": "number",
            "description": "Initial portfolio cash",
            "default": 10000,
        },
    },
    "required": ["prompt", "symbols"],
    "additionalProperties": False,
}


class BacktestInput(BaseModel):
    """Schema for the backtest tool."""

    prompt: str = Field(
        ...,
        description="Natural language description of the trading strategy, or modifications to make if base_code is provided",
    )
    symbols: List[str] = Field(
        ...,
        description="Ticker symbols to backtest. For single-asset strategies, first symbol maps to 'prices' slot.",
    )
    symbol_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Explicit mapping of data slot names to symbols (e.g., {'prices': 'SPY'} or {'asset_a': 'GLD', 'asset_b': 'SLV'}).",
    )
    base_code: Optional[str] = Field(
        default=None,
        description="Previous strategy code to modify. MUST be provided when user wants to update/modify a previous backtest.",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Strategy parameters to override defaults (e.g., {'fast_window': 10}).",
    )
    direction: Union[str, List[str]] = Field(
        default="longonly",
        description="Position direction mode(s): 'longonly', 'shortonly', or 'both'. Can be a list for comparison runs.",
    )
    execution_price: str = Field(
        default="close",
        description="Price column to use for trade execution: 'close' or 'open'.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=0.5,
        description="Stop loss as fraction of entry price (e.g., 0.05 = 5%). None to disable.",
    )
    take_profit: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=2.0,
        description="Take profit as fraction of entry price (e.g., 0.10 = 10%). None to disable.",
    )
    trailing_stop: bool = Field(
        default=False,
        description="If true, stop_loss trails the highest price since entry.",
    )
    slippage: float = Field(
        default=0.0,
        ge=0,
        le=0.05,
        description="Slippage as fraction of price per trade (e.g., 0.001 = 0.1% = 10 bps).",
    )
    timeframe: str = Field(
        default="1Day",
        description="Data timeframe",
    )
    start: Optional[str] = Field(
        default=None,
        description="ISO8601 start datetime (UTC)",
    )
    end: Optional[str] = Field(
        default=None,
        description="ISO8601 end datetime (UTC)",
    )
    init_cash: float = Field(
        default=10000.0,
        ge=100,
        le=10000000,
        description="Initial portfolio cash",
    )

    model_config = ConfigDict(extra="forbid")


def _downsample_equity_curve(
    timestamps: List[str],
    equity_curve: List[float],
    max_points: int = 500,
) -> List[Dict[str, Any]]:
    """Downsample equity curve for visualization.

    If the data has more than max_points, uniformly sample to reduce size.
    Always includes the first and last points.

    Args:
        timestamps: List of ISO8601 timestamp strings.
        equity_curve: List of equity values.
        max_points: Maximum number of points to return.

    Returns:
        List of {"timestamp": ..., "value": ...} dicts.
    """
    n = len(timestamps)
    if n <= max_points:
        # No downsampling needed
        return [
            {"timestamp": ts, "value": eq}
            for ts, eq in zip(timestamps, equity_curve)
        ]

    # Calculate step size to get approximately max_points
    step = max(1, n // max_points)
    indices = list(range(0, n, step))

    # Always include the last point
    if indices[-1] != n - 1:
        indices.append(n - 1)

    return [
        {"timestamp": timestamps[i], "value": equity_curve[i]}
        for i in indices
    ]


def _build_symbol_mapping(
    data_schema: Dict[str, Any],
    symbols: List[str],
    explicit_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build symbol mapping from data_schema slots and provided symbols.

    Args:
        data_schema: The data_schema from LLM output (e.g., {"prices": {...}})
        symbols: List of symbols from user
        explicit_mapping: Optional explicit slot->symbol mapping

    Returns:
        Dict mapping slot names to symbols (e.g., {"prices": "SPY"})
    """
    # If explicit mapping provided, use it
    if explicit_mapping:
        return explicit_mapping

    # Get slot names from data_schema
    slots = list(data_schema.keys())

    # For single-slot strategies, map first symbol to first (and only) slot
    if len(slots) == 1:
        # Map all symbols to the single slot for multi-symbol runs
        # For development, we return mapping for the first symbol
        return {slots[0]: symbols[0]}

    # For multi-slot strategies, map symbols in order
    mapping = {}
    for i, slot in enumerate(slots):
        if i < len(symbols):
            mapping[slot] = symbols[i]
        else:
            # Not enough symbols provided
            raise ValueError(f"Not enough symbols provided. Need {len(slots)} symbols for slots: {slots}")

    return mapping


def _fetch_prices_for_symbol(
    symbol: str,
    timeframe_str: str,
    start: Optional[str],
    end: Optional[str],
    limit: int = 5000,
) -> pd.DataFrame:
    """Fetch price data for a single symbol."""
    from datetime import datetime, timedelta, timezone

    api_key, api_secret = get_alpaca_credentials()
    client = StockHistoricalDataClient(api_key, api_secret)

    timeframe = resolve_timeframe(timeframe_str)

    # Default to 10 years of data if no start date provided
    if start:
        start_dt = parse_dt(start)
    else:
        start_dt = datetime.now(timezone.utc) - timedelta(days=365 * 10)

    end_dt = parse_dt(end)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        limit=limit,
        adjustment=Adjustment.SPLIT,  # Use split-adjusted prices for accurate backtesting
    )

    bars = client.get_stock_bars(request)
    df = cast(pd.DataFrame, getattr(bars, "df", None))

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for symbol: {symbol}")

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0, drop_level=True)
    df = df.sort_index()
    df.columns = df.columns.str.lower()
    return df


def _fetch_prices_for_symbols(
    symbols: List[str],
    timeframe_str: str,
    start: Optional[str],
    end: Optional[str],
    limit: int = 5000,
) -> Dict[str, pd.DataFrame]:
    """Fetch price data for multiple symbols, returning a dict."""
    result = {}
    for symbol in symbols:
        try:
            result[symbol] = _fetch_prices_for_symbol(
                symbol, timeframe_str, start, end, limit
            )
        except Exception as e:
            print(f"[Backtest] Failed to fetch {symbol}: {e}")
            continue
    return result


def _build_data_dict(
    data_schema: Dict[str, Any],
    symbol_mapping: Dict[str, str],
    prices_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Build data dict mapping slot names to DataFrames.

    Args:
        data_schema: The data_schema from LLM (defines slot names)
        symbol_mapping: Maps slot names to symbols (e.g., {"asset_a": "GLD", "asset_b": "SLV"})
        prices_by_symbol: Dict of fetched prices keyed by symbol

    Returns:
        Dict mapping slot names to DataFrames (e.g., {"asset_a": GLD_df, "asset_b": SLV_df})
    """
    data_dict = {}
    for slot_name in data_schema.keys():
        symbol = symbol_mapping.get(slot_name)
        if symbol and symbol in prices_by_symbol:
            data_dict[slot_name] = prices_by_symbol[symbol]
        elif symbol:
            raise ValueError(f"No price data for symbol '{symbol}' (slot: '{slot_name}')")
        else:
            raise ValueError(f"No symbol mapping for slot '{slot_name}'")
    return data_dict


def _fetch_data_for_slot(
    symbol: str,
    slot_schema: Dict[str, Any],
    start: Optional[str],
    end: Optional[str],
    target_frequency: str = "1Day",
) -> pd.DataFrame:
    """Fetch data for a single slot using the appropriate provider.

    This function uses the provider registry to determine which provider
    to use based on the data_type in the slot schema.

    Args:
        symbol: Ticker symbol to fetch.
        slot_schema: Schema for the slot (contains data_type, frequency).
        start: Start datetime (ISO8601 or datetime).
        end: End datetime.
        target_frequency: Target frequency for resampling.

    Returns:
        DataFrame with data for the slot.
    """
    from datetime import datetime, timedelta, timezone

    data_type = slot_schema.get("data_type", "ohlcv")
    source_freq = slot_schema.get("frequency", target_frequency)

    # Get the appropriate provider
    registry = get_provider_registry()
    provider = registry.infer_provider(slot_schema)

    logger.info(f"    Fetching {data_type} for {symbol} from {provider.name} provider")

    # Parse dates
    start_dt = parse_dt(start) if start else datetime.now(timezone.utc) - timedelta(days=365 * 10)
    end_dt = parse_dt(end) if end else None

    # Fetch data
    df = provider.fetch(
        symbol=symbol,
        data_type=data_type,
        start=start_dt,
        end=end_dt,
        frequency=source_freq,
    )

    # Resample to target frequency if needed
    if source_freq != target_frequency:
        df = resample_to_frequency(df, source_freq, target_frequency, data_type)
        logger.info(f"    Resampled from {source_freq} to {target_frequency}: {len(df)} rows")

    return df


def _infer_symbol_from_slot_name(slot_name: str, symbols: List[str]) -> Optional[str]:
    """Try to infer which symbol a slot refers to based on its name.

    Examples:
        - "aapl_prices" with symbols=["AAPL", "MSFT"] -> "AAPL"
        - "msft_pe" with symbols=["AAPL", "MSFT"] -> "MSFT"
        - "prices" with symbols=["SPY"] -> None (use positional)

    Args:
        slot_name: The slot name from data_schema.
        symbols: List of available symbols.

    Returns:
        Matched symbol or None if no match found.
    """
    slot_lower = slot_name.lower()
    for symbol in symbols:
        symbol_lower = symbol.lower()
        # Check if slot name starts with or contains the symbol
        if slot_lower.startswith(symbol_lower) or f"_{symbol_lower}" in slot_lower:
            return symbol
    return None


def _fetch_data_dict_from_schema(
    data_schema: Dict[str, Any],
    symbols: List[str],
    start: Optional[str],
    end: Optional[str],
    timeframe: str = "1Day",
    fallback_prices: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch all data required by a data_schema using appropriate providers.

    Args:
        data_schema: Dict mapping slot names to slot schemas.
        symbols: List of symbols (mapped to slots by position or inferred from name).
        start: Start date.
        end: End date.
        timeframe: Target timeframe.
        fallback_prices: Pre-fetched OHLCV data to use as fallback.

    Returns:
        Dict mapping slot names to DataFrames.
    """
    data_dict = {}
    slot_names = list(data_schema.keys())

    # Track which slots we've assigned to which symbols for positional fallback
    positional_index = 0

    for slot_name in slot_names:
        slot_schema = data_schema[slot_name]
        data_type = slot_schema.get("data_type", "ohlcv")

        # Try to infer symbol from slot name first (e.g., "aapl_prices" -> "AAPL")
        symbol = _infer_symbol_from_slot_name(slot_name, symbols)

        if symbol is None:
            # Fall back to positional mapping
            if positional_index < len(symbols):
                symbol = symbols[positional_index]
                positional_index += 1
            else:
                symbol = symbols[0]

        logger.info(f"    Slot '{slot_name}' -> symbol '{symbol}' (data_type: {data_type})")

        # For OHLCV data, try to use pre-fetched data first
        if data_type == "ohlcv" and fallback_prices and symbol in fallback_prices:
            logger.info(f"    Using pre-fetched OHLCV for {slot_name} ({symbol})")
            data_dict[slot_name] = fallback_prices[symbol]
        else:
            # Fetch from provider
            try:
                df = _fetch_data_for_slot(
                    symbol=symbol,
                    slot_schema=slot_schema,
                    start=start,
                    end=end,
                    target_frequency=timeframe,
                )
                data_dict[slot_name] = df
                logger.info(f"    Fetched {data_type} for {slot_name} ({symbol}): {len(df)} rows")
            except Exception as e:
                logger.error(f"    Failed to fetch {data_type} for {slot_name} ({symbol}): {e}")
                raise

    return data_dict


def _run_single_backtest(
    agent: CodexAgent,
    session: CodexSession,
    executor: StrategyExecutor,
    prices: pd.DataFrame,
    symbol: str,
    timeframe: str,
    code: str,
) -> BacktestResult:
    """Run backtest for a single symbol with given strategy code."""
    return executor.execute(
        code=code,
        prices=prices,
        symbols=[symbol],
        timeframe=timeframe,
    )


def run_backtest(payload: BacktestInput) -> Dict[str, Any]:
    """Execute the full backtest pipeline with iterative Codex agent.

    This function implements an agentic loop where:
    1. Codex generates strategy code
    2. Code is validated and executed on EACH symbol separately
    3. If errors occur, they are fed back to Codex (same conversation)
    4. Codex fixes the code
    5. Repeat until success or max attempts

    Args:
        payload: Input parameters for the backtest.

    Returns:
        Dictionary with backtest results suitable for JSON serialization.
        Includes separate equity curves for each symbol.
    """
    logger.info("=" * 80)
    logger.info("BACKTEST PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Prompt: {payload.prompt}")
    logger.info(f"Symbols: {payload.symbols}")
    logger.info(f"Timeframe: {payload.timeframe}")
    logger.info(f"Direction: {payload.direction}")
    logger.info(f"Start: {payload.start}")
    logger.info(f"End: {payload.end}")
    logger.info(f"Initial Cash: ${payload.init_cash:,.2f}")
    if payload.base_code:
        logger.info(f"Base code provided: {len(payload.base_code)} chars")

    # Initialize Codex agent and executor
    logger.info("-" * 40)
    logger.info("Initializing Codex agent and executor...")
    agent = get_codex_agent(model="gpt-5.1-codex-max", max_attempts=5)
    executor = get_strategy_executor(init_cash=payload.init_cash, timeout_seconds=60)
    logger.info("Codex agent and executor initialized")

    # Fetch price data for each symbol separately
    logger.info("-" * 40)
    logger.info(f"Fetching price data for {len(payload.symbols)} symbol(s)...")
    try:
        prices_by_symbol = _fetch_prices_for_symbols(
            symbols=payload.symbols,
            timeframe_str=payload.timeframe,
            start=payload.start,
            end=payload.end,
        )
        for symbol, df in prices_by_symbol.items():
            logger.info(f"  {symbol}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        return {
            "success": False,
            "error": f"Failed to fetch price data: {str(e)}",
            "symbol": ", ".join(payload.symbols),
            "timeframe": payload.timeframe,
            "data": [],
            "equity_curves": {},
        }

    if not prices_by_symbol:
        logger.error("No price data available for any symbol")
        return {
            "success": False,
            "error": "No price data available for any symbol",
            "symbol": ", ".join(payload.symbols),
            "timeframe": payload.timeframe,
            "data": [],
            "equity_curves": {},
        }

    logger.info(f"Price data fetched successfully for {len(prices_by_symbol)} symbol(s)")

    # Use first symbol's prices for strategy development
    first_symbol = list(prices_by_symbol.keys())[0]
    first_prices = prices_by_symbol[first_symbol]

    # Create Codex session (maintains conversation context)
    # If base_code is provided, we're modifying an existing strategy
    logger.info("-" * 40)
    logger.info("Creating Codex session...")
    session: CodexSession = agent.create_session(
        strategy_prompt=payload.prompt,
        base_code=payload.base_code,
    )

    # Iterative generate-execute-fix loop
    logger.info("-" * 40)
    logger.info("Starting iterative generate-execute-fix loop...")
    last_result: Optional[BacktestResult] = None
    all_attempts: List[Dict[str, Any]] = []
    successful_code: Optional[str] = None

    while session.attempts < session.max_attempts:
        logger.info("")
        logger.info("*" * 60)
        logger.info(f"ATTEMPT {session.attempts + 1}/{session.max_attempts}")
        logger.info("*" * 60)
        # Step 1: Generate code
        logger.info("Step 1: Generating code from Codex...")
        try:
            code = agent.generate(session)
            logger.info(f"Code generated: {len(code)} chars")
        except Exception as e:
            logger.error(f"Codex API error: {e}")
            return {
                "success": False,
                "error": f"Codex API error: {str(e)}",
                "symbol": ", ".join(payload.symbols),
                "timeframe": payload.timeframe,
                "data": [],
                "equity_curves": {},
                "attempts": session.attempts,
            }

        # Step 2: Validate code
        logger.info("Step 2: Validating generated code...")
        is_valid, validation_error = agent.validate_code(code)
        if not is_valid:
            logger.warning(f"Code validation FAILED: {validation_error}")
            all_attempts.append({
                "attempt": session.attempts,
                "code": code,
                "error": validation_error,
                "error_type": "ValidationError",
            })
            agent.feed_validation_error(session, validation_error or "Unknown validation error")
            continue
        logger.info("Code validation PASSED")

        # Step 2b: Validate data_schema
        logger.info("Step 2b: Validating data schema...")
        is_schema_valid, schema_error = agent.validate_data_schema(session.data_schema)
        if not is_schema_valid:
            logger.warning(f"Data schema validation FAILED: {schema_error}")
            all_attempts.append({
                "attempt": session.attempts,
                "code": code,
                "error": schema_error,
                "error_type": "DataSchemaError",
            })
            agent.feed_validation_error(session, schema_error or "Unknown schema error")
            continue
        logger.info("Data schema validation PASSED")

        # Step 3: Check if this is a multi-asset strategy and build test data dict
        logger.info("Step 3: Executing strategy code...")
        data_schema_test = session.data_schema
        is_multi_asset = len(data_schema_test) > 1
        logger.info(f"  Multi-asset strategy: {is_multi_asset}")
        logger.info(f"  Data schema slots: {list(data_schema_test.keys())}")

        if is_multi_asset:
            # Multi-asset strategy: need to build data dict with all slots
            # First build symbol mapping to get all required symbols
            logger.info("  Building data dict for multi-asset strategy...")
            try:
                test_symbol_mapping = _build_symbol_mapping(
                    data_schema_test, payload.symbols, payload.symbol_mapping
                )
                logger.info(f"  Symbol mapping: {test_symbol_mapping}")
                test_data_dict = _build_data_dict(
                    data_schema_test, test_symbol_mapping, prices_by_symbol
                )
                logger.info(f"  Data dict built with {len(test_data_dict)} slots")
            except ValueError as e:
                logger.error(f"Symbol mapping error: {e}")
                all_attempts.append({
                    "attempt": session.attempts,
                    "code": code,
                    "error": str(e),
                    "error_type": "SymbolMappingError",
                })
                agent.feed_validation_error(session, str(e))
                continue

            result: BacktestResult = executor.execute(
                code=code,
                prices=first_prices,  # Still needed for fallback
                symbols=payload.symbols,
                timeframe=payload.timeframe,
                params=payload.params,
                direction=payload.direction if isinstance(payload.direction, str) else payload.direction[0],
                data=test_data_dict,  # Pass full data dict for multi-asset
            )
        else:
            # Single-asset strategy: use first symbol's prices
            logger.info(f"  Single-asset strategy, using {first_symbol} data")
            result = executor.execute(
                code=code,
                prices=first_prices,
                symbols=[first_symbol],
                timeframe=payload.timeframe,
                params=payload.params,
                direction=payload.direction if isinstance(payload.direction, str) else payload.direction[0],
            )
        last_result = result

        all_attempts.append({
            "attempt": session.attempts,
            "code": code,
            "success": result.success,
            "error": result.error,
            "error_type": result.error_type,
        })

        # Step 4: Check if successful
        if result.success:
            logger.info(f"Execution SUCCEEDED with {result.position_changes} position changes")

            # Code works! Save it for running on all symbols
            logger.info("SUCCESS! Strategy code is working.")
            logger.info("=" * 60)
            logger.info("FINAL GENERATED CODE:")
            logger.info("=" * 60)
            logger.info(code)
            logger.info("=" * 60)
            successful_code = code
            break
        else:
            logger.warning(f"Execution FAILED: {result.error_type}: {result.error}")

        # Step 5: Feed error back to Codex for fixing
        execution_result = result.to_execution_result()
        agent.feed_error(session, execution_result)

    # If we didn't find working code, return failure
    if successful_code is None:
        logger.error("=" * 60)
        logger.error("BACKTEST FAILED - Max attempts reached")
        logger.error("=" * 60)
        # Include param_schema from last attempt if available
        param_schema = session.param_schema if hasattr(session, 'param_schema') else {}
        error_msg = "Max attempts reached without generating valid strategy"
        if last_result:
            error_msg = f"{error_msg}. Last error: {last_result.error}"
        logger.error(f"Final error: {error_msg}")
        logger.error(f"Total attempts: {session.attempts}")

        return {
            "success": False,
            "error": error_msg,
            "symbol": ", ".join(payload.symbols),
            "timeframe": payload.timeframe,
            "data": [],
            "equity_curves": {},
            "attempts": session.attempts,
            "attempt_history": all_attempts,
            "last_code": session.generated_code,
            "param_schema": param_schema,
        }

    logger.info("-" * 40)
    logger.info("Preparing final results...")

    # Extract schemas from successful generation
    data_schema = session.data_schema
    param_schema = session.param_schema
    logger.info(f"Data schema: {data_schema}")
    logger.info(f"Param schema: {param_schema}")

    # Build symbol mapping from data_schema
    logger.info("Building symbol mapping...")
    try:
        symbol_mapping = _build_symbol_mapping(
            data_schema, payload.symbols, payload.symbol_mapping
        )
        logger.info(f"Symbol mapping: {symbol_mapping}")
    except ValueError as e:
        logger.error(f"Symbol mapping error: {e}")
        return {
            "success": False,
            "error": str(e),
            "symbol": ", ".join(payload.symbols),
            "timeframe": payload.timeframe,
            "data": [],
            "equity_curves": {},
            "data_schema": data_schema,
            "param_schema": param_schema,
        }

    # Merge user params with defaults from param_schema
    logger.info("Merging parameters...")
    merged_params = merge_params(param_schema, payload.params)
    logger.info(f"User params: {payload.params}")

    # Inject built-in params from payload (these override any user params)
    # Note: direction is handled specially for multi-execution comparison
    merged_params["execution_price"] = payload.execution_price
    merged_params["stop_loss"] = payload.stop_loss
    merged_params["take_profit"] = payload.take_profit
    merged_params["trailing_stop"] = payload.trailing_stop
    merged_params["slippage"] = payload.slippage
    logger.info(f"Merged params (with built-ins): {merged_params}")

    # Normalize direction to a list for multi-execution comparison
    directions = payload.direction if isinstance(payload.direction, list) else [payload.direction]
    is_comparison_run = len(directions) > 1
    logger.info(f"Directions to run: {directions}")
    logger.info(f"Comparison run: {is_comparison_run}")

    # Determine if this is a multi-asset strategy
    is_multi_asset = len(data_schema) > 1
    logger.info(f"Multi-asset strategy: {is_multi_asset}")

    # Run the successful strategy (potentially multiple times for comparison)
    equity_curves: Dict[str, List[Dict[str, Any]]] = {}
    all_metrics: Dict[str, Dict[str, Any]] = {}
    comparison_results: Dict[str, Dict[str, Any]] = {}  # For multi-direction comparison
    combined_data: List[Dict[str, Any]] = []

    # Build data dict for multi-asset strategies
    data_dict = None
    if is_multi_asset:
        logger.info(f"Building data dict for multi-asset strategy...")
        logger.info(f"  Slots: {list(data_schema.keys())}")
        logger.info(f"  Symbol mapping: {symbol_mapping}")
        try:
            data_dict = _build_data_dict(data_schema, symbol_mapping, prices_by_symbol)
            logger.info(f"  Data dict built with {len(data_dict)} slots")
        except ValueError as e:
            logger.error(f"Failed to build data dict: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": ", ".join(payload.symbols),
                "timeframe": payload.timeframe,
                "data": [],
                "equity_curves": {},
                "data_schema": data_schema,
                "param_schema": param_schema,
            }

    # Run for each direction (supports comparison runs)
    logger.info("-" * 40)
    logger.info("EXECUTING FINAL BACKTESTS")
    first_result = None
    primary_direction = directions[0]

    for direction in directions:
        direction_key = f"dir_{direction}" if is_comparison_run else None
        logger.info(f"Running backtest with direction: {direction}")

        # Set direction for this run
        run_params = merged_params.copy()
        run_params["direction"] = direction

        if is_multi_asset:
            # Multi-asset strategy
            result = executor.execute(
                code=successful_code,
                prices=first_prices,
                symbols=payload.symbols,
                timeframe=payload.timeframe,
                params=run_params,
                direction=direction,
                data=data_dict,
            )

            if result.success:
                logger.info(f"  Multi-asset result: Return={result.metrics.total_return:.2%}, Sharpe={result.metrics.sharpe_ratio:.2f}, Trades={result.metrics.num_trades}")
                # Store under direction key for comparison, or "combined" for single direction
                key = direction_key or "combined"
                equity_curves[key] = _downsample_equity_curve(
                    result.timestamps, result.equity_curve
                )
                all_metrics[key] = {
                    "total_return": result.metrics.total_return,
                    "sharpe_ratio": result.metrics.sharpe_ratio,
                    "sortino_ratio": result.metrics.sortino_ratio,
                    "max_drawdown": result.metrics.max_drawdown,
                    "win_rate": result.metrics.win_rate,
                    "num_trades": result.metrics.num_trades,
                    "profit_factor": result.metrics.profit_factor,
                    "position_changes": result.position_changes,
                    "long_entries": result.long_entries,
                    "long_exits": result.long_exits,
                    "short_entries": result.short_entries,
                    "short_exits": result.short_exits,
                }

                if is_comparison_run:
                    comparison_results[direction] = {
                        "equity_curve": equity_curves[key],
                        "metrics": all_metrics[key],
                    }
            else:
                logger.warning(f"  Multi-asset result: FAILED - {result.error}")
        else:
            # Single-asset strategy: run on each symbol
            logger.info(f"  Single-asset strategy: running on {len(prices_by_symbol)} symbols")

            for symbol, prices in prices_by_symbol.items():
                logger.info(f"    Executing on {symbol}...")
                result = executor.execute(
                    code=successful_code,
                    prices=prices,
                    symbols=[symbol],
                    timeframe=payload.timeframe,
                    params=run_params,
                    direction=direction,
                )

                if result.success:
                    logger.info(f"      {symbol}: Return={result.metrics.total_return:.2%}, Sharpe={result.metrics.sharpe_ratio:.2f}, Trades={result.metrics.num_trades}")
                    # For comparison runs, key by direction_symbol
                    key = f"{direction}_{symbol}" if is_comparison_run else symbol
                    equity_curves[key] = _downsample_equity_curve(
                        result.timestamps, result.equity_curve
                    )
                    all_metrics[key] = {
                        "total_return": result.metrics.total_return,
                        "sharpe_ratio": result.metrics.sharpe_ratio,
                        "sortino_ratio": result.metrics.sortino_ratio,
                        "max_drawdown": result.metrics.max_drawdown,
                        "win_rate": result.metrics.win_rate,
                        "num_trades": result.metrics.num_trades,
                        "profit_factor": result.metrics.profit_factor,
                        "position_changes": result.position_changes,
                        "long_entries": result.long_entries,
                        "long_exits": result.long_exits,
                        "short_entries": result.short_entries,
                        "short_exits": result.short_exits,
                    }
                    # Capture first successful result for primary direction
                    if direction == primary_direction and first_result is None:
                        first_result = result
                else:
                    logger.warning(f"      {symbol}: FAILED - {result.error}")

        # For multi-asset, capture result from the loop
        if is_multi_asset and direction == primary_direction and first_result is None:
            first_result = result

    if first_result is None:
        return {
            "success": False,
            "error": "No successful backtest results",
            "symbol": ", ".join(payload.symbols),
            "timeframe": payload.timeframe,
            "data": [],
            "equity_curves": {},
        }

    result_dict = first_result.to_dict()

    # Downsample the data field to avoid MCP size limits
    if "data" in result_dict and len(result_dict["data"]) > 500:
        original_len = len(result_dict["data"])
        step = max(1, original_len // 500)
        indices = list(range(0, original_len, step))
        if indices[-1] != original_len - 1:
            indices.append(original_len - 1)
        result_dict["data"] = [result_dict["data"][i] for i in indices]
        logger.info(f"Downsampled data from {original_len} to {len(result_dict['data'])} points")

    result_dict["generated_code"] = successful_code
    result_dict["symbol"] = ", ".join(payload.symbols)
    result_dict["timeframe"] = payload.timeframe
    result_dict["prompt"] = payload.prompt
    result_dict["params"] = merged_params  # Use merged params
    result_dict["data_schema"] = data_schema  # Include data schema
    result_dict["param_schema"] = param_schema  # Include param schema
    result_dict["symbol_mapping"] = symbol_mapping  # Include symbol mapping
    result_dict["attempts"] = session.attempts
    # Note: Omitting attempt_history from widget response to reduce payload size
    # The full attempt_history is available in logs

    # Built-in execution parameters
    result_dict["direction"] = payload.direction
    result_dict["execution_price"] = payload.execution_price
    result_dict["stop_loss"] = payload.stop_loss
    result_dict["take_profit"] = payload.take_profit
    result_dict["trailing_stop"] = payload.trailing_stop
    result_dict["slippage"] = payload.slippage

    # Add multi-symbol data
    result_dict["equity_curves"] = equity_curves
    result_dict["metrics_by_symbol"] = all_metrics
    result_dict["symbols"] = list(prices_by_symbol.keys())

    # Add comparison data if multiple directions were run
    if is_comparison_run:
        result_dict["comparison"] = {
            "directions": directions,
            "results_by_direction": comparison_results,
        }

    # Log final summary
    logger.info("=" * 80)
    logger.info("BACKTEST PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Success: {result_dict.get('success', False)}")
    logger.info(f"Symbols: {result_dict.get('symbols', [])}")
    logger.info(f"Attempts: {result_dict.get('attempts', 0)}")
    if result_dict.get('success'):
        metrics = result_dict.get('metrics', {})
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
        logger.info(f"Number of Trades: {metrics.get('num_trades', 0)}")

    # Debug: Log equity_curves sizes
    logger.info("-" * 40)
    logger.info("DEBUG: Equity curves data sizes:")
    for key, curve in equity_curves.items():
        logger.info(f"  {key}: {len(curve)} data points")
        if curve:
            logger.info(f"    First: {curve[0]}")
            logger.info(f"    Last: {curve[-1]}")
    logger.info("=" * 80)

    return result_dict


def run_multi_backtest(payload: MultiBacktestInput) -> Dict[str, Any]:
    """Execute the v3 multi-strategy backtest from tool input.

    This is the entry point for the multi_backtest MCP tool.
    It calls run_multi_strategy_backtest and converts the result to a dict.

    Args:
        payload: MultiBacktestInput with prompt and options.

    Returns:
        Dictionary suitable for widget display.
    """
    result = run_multi_strategy_backtest(
        prompt=payload.prompt,
        symbols=payload.symbols,
        timeframe=payload.timeframe,
        start=payload.start,
        end=payload.end,
        init_cash=payload.init_cash,
        normalize=payload.normalize,
        show_trades=payload.show_trades,
    )
    return combined_result_to_dict(result)


def run_multi_strategy_backtest(
    prompt: str,
    symbols: Optional[List[str]] = None,
    timeframe: str = "1Day",
    start: Optional[str] = None,
    end: Optional[str] = None,
    init_cash: float = 100.0,
    normalize: bool = False,
    show_trades: bool = False,
) -> CombinedBacktestResult:
    """Run the v3 multi-strategy backtest pipeline.

    This function implements the new architecture where:
    1. Strategy Planner parses user intent into multiple strategies
    2. Each strategy is generated and run (potentially with multiple directions)
    3. Results are combined into a unified output

    Args:
        prompt: Natural language description (can include multiple strategies).
        symbols: Default symbols if not specified in prompt.
        timeframe: Data timeframe.
        start: Start date.
        end: End date.
        init_cash: Initial cash per strategy.
        normalize: Whether to normalize all curves to start at 100.
        show_trades: Whether to include trade markers.

    Returns:
        CombinedBacktestResult ready for widget display.
    """
    from .models import BacktestRunResult

    logger.info("=" * 80)
    logger.info("MULTI-STRATEGY BACKTEST PIPELINE (v3)")
    logger.info("=" * 80)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Default symbols: {symbols}")
    logger.info(f"Timeframe: {timeframe}")

    # Default symbols if not provided
    if not symbols:
        symbols = ["SPY"]

    # Step 1: Plan strategies
    logger.info("-" * 40)
    logger.info("Step 1: Planning strategies...")
    try:
        planner = get_strategy_planner()
        planner_output = planner.plan(
            prompt=prompt,
            default_symbols=symbols,
        )
        logger.info(f"Planner output: {len(planner_output.strategies)} strategies")
    except Exception as e:
        logger.error(f"Strategy planning failed: {e}")
        return CombinedBacktestResult(
            success=False,
            error=f"Strategy planning failed: {str(e)}",
        )

    # Collect all symbols needed across all strategies
    all_symbols = set()
    for spec in planner_output.strategies:
        all_symbols.update(spec.symbols)
    logger.info(f"Total symbols needed: {all_symbols}")

    # Step 2: Fetch price data for all symbols
    logger.info("-" * 40)
    logger.info("Step 2: Fetching price data...")
    try:
        prices_by_symbol = _fetch_prices_for_symbols(
            symbols=list(all_symbols),
            timeframe_str=timeframe,
            start=start,
            end=end,
        )
        for sym, df in prices_by_symbol.items():
            logger.info(f"  {sym}: {len(df)} bars")
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        return CombinedBacktestResult(
            success=False,
            error=f"Failed to fetch price data: {str(e)}",
        )

    # Step 3: Generate and run each strategy
    logger.info("-" * 40)
    logger.info("Step 3: Generating and running strategies...")

    agent = get_codex_agent(model="gpt-5.1-codex-max", max_attempts=5)
    executor = get_strategy_executor(init_cash=init_cash, timeout_seconds=60)

    generated_strategies: List[GeneratedStrategy] = []
    all_results: List[BacktestRunResult] = []

    for spec in planner_output.strategies:
        logger.info(f"Processing strategy: {spec.name}")
        logger.info(f"  Symbols: {spec.symbols}")
        logger.info(f"  Directions: {spec.directions}")

        # Generate the strategy code via LLM
        try:
            session = agent.create_session_from_spec(spec)

            # Iterative generation loop
            success = False
            for attempt in range(agent.max_attempts):
                code = agent.generate(session)

                # Validate
                is_valid, val_error = agent.validate_code(code)
                if not is_valid:
                    agent.feed_validation_error(session, val_error or "Unknown validation error")
                    continue

                is_schema_valid, schema_error = agent.validate_data_schema(session.data_schema)
                if not is_schema_valid:
                    agent.feed_validation_error(session, schema_error or "Unknown schema error")
                    continue

                # Build data dict for this strategy using appropriate providers
                logger.info(f"  Building data dict from schema: {session.data_schema}")
                try:
                    data_dict = _fetch_data_dict_from_schema(
                        data_schema=session.data_schema,
                        symbols=spec.symbols,
                        start=start,
                        end=end,
                        timeframe=timeframe,
                        fallback_prices=prices_by_symbol,  # Use pre-fetched OHLCV as fallback
                    )
                except Exception as e:
                    logger.error(f"  Failed to fetch data: {e}")
                    agent.feed_validation_error(session, f"Data fetch failed: {str(e)}")
                    continue

                if not data_dict:
                    agent.feed_validation_error(session, "No data available for strategy")
                    continue

                # Test execution with first direction
                first_direction = spec.directions[0]
                test_result = executor.execute(
                    code=code,
                    prices=list(data_dict.values())[0],
                    symbols=spec.symbols,
                    timeframe=timeframe,
                    params={},
                    direction=first_direction,
                    data=data_dict,
                )

                if test_result.success:
                    success = True
                    break
                else:
                    agent.feed_error(session, test_result.to_execution_result())

            if not success:
                logger.error(f"Failed to generate strategy: {spec.name}")
                continue

            # Create GeneratedStrategy
            gen_strategy = GeneratedStrategy(
                spec=spec,
                code=code,
                data_schema=session.data_schema,
                param_schema=session.param_schema,
                params=merge_params(session.param_schema, {}),
            )
            generated_strategies.append(gen_strategy)

            # Run for each direction
            for direction in spec.directions:
                logger.info(f"  Running with direction: {direction}")
                run_result = executor.execute_generated_strategy(
                    strategy=gen_strategy,
                    data_dict=data_dict,
                    direction=direction,
                )
                all_results.append(run_result)
                if run_result.success:
                    for sym, sym_result in run_result.results_by_symbol.items():
                        logger.info(
                            f"    {sym}: Return={sym_result.metrics.total_return:.2%}"
                        )
                else:
                    logger.warning(f"    Failed: {run_result.error}")

        except Exception as e:
            logger.error(f"Error processing strategy {spec.name}: {e}")
            continue

    if not all_results:
        return CombinedBacktestResult(
            success=False,
            error="No strategies executed successfully",
        )

    # Step 4: Combine results
    logger.info("-" * 40)
    logger.info("Step 4: Combining results...")

    display_config = DisplayConfig(
        normalize=normalize,
        show_trades=show_trades,
    )

    combined = combine_results(
        results=all_results,
        strategies=generated_strategies,
        display_config=display_config,
    )

    # Downsample for widget
    combined = downsample_combined_result(combined, max_points=500)

    logger.info("=" * 80)
    logger.info("MULTI-STRATEGY BACKTEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Success: {combined.success}")
    logger.info(f"Series count: {len(combined.series)}")
    logger.info(f"Metrics rows: {len(combined.metrics_table)}")

    return combined


def combined_result_to_dict(result: CombinedBacktestResult) -> Dict[str, Any]:
    """Convert CombinedBacktestResult to widget-compatible dict.

    Args:
        result: The combined result.

    Returns:
        Dict suitable for JSON serialization and widget consumption.
    """
    if not result.success:
        return {
            "success": False,
            "error": result.error,
            "series": [],
            "metrics_table": [],
            "strategies": [],
            "meta": {},
        }

    # Build equity_curves for backward compatibility with widget
    # Widget expects: {name: [{timestamp, value}, ...]}
    equity_curves = {}
    for s in result.series:
        # Convert {time, value} to {timestamp, value}
        equity_curves[s.name] = [
            {"timestamp": d["time"], "value": d["value"]}
            for d in s.data
        ]

    # Build metrics_by_symbol for backward compatibility
    metrics_by_symbol = {}
    for m in result.metrics_table:
        metrics_by_symbol[m.name] = m.metrics

    # Extract symbols list
    symbols = list({s.symbol for s in result.series})

    return {
        "success": True,
        # V3 format (new)
        "series": [
            {
                "name": s.name,
                "strategy_name": s.strategy_name,
                "symbol": s.symbol,
                "direction": s.direction,
                "data": s.data,
                "trades": s.trades,
            }
            for s in result.series
        ],
        "metrics_table": [
            {
                "name": m.name,
                "strategy_name": m.strategy_name,
                "symbol": m.symbol,
                "direction": m.direction,
                "metrics": m.metrics,
            }
            for m in result.metrics_table
        ],
        "strategies": [
            {
                "name": st.name,
                "description": st.description,
                "code": st.code,
                "data_schema": st.data_schema,
                "param_schema": st.param_schema,
                "params_used": st.params_used,
                "execution": st.execution,
            }
            for st in result.strategies
        ],
        "meta": {
            "timeframe": result.meta.timeframe,
            "start_date": result.meta.start_date,
            "end_date": result.meta.end_date,
            "total_bars": result.meta.total_bars,
            "num_strategies": result.meta.num_strategies,
            "num_runs": result.meta.num_runs,
        },
        # V2 format (backward compatibility for widget)
        "equity_curves": equity_curves,
        "metrics_by_symbol": metrics_by_symbol,
        "symbols": symbols,
        "timeframe": result.meta.timeframe,
    }


def format_result_text(result: Dict[str, Any]) -> str:
    """Format backtest result as human-readable text.

    Handles both v3 format (series, metrics_table) and legacy v2 format.
    """
    if not result.get("success", False):
        return f"Backtest failed: {result.get('error', 'Unknown error')}"

    meta = result.get("meta", {})
    metrics_table = result.get("metrics_table", [])
    strategies = result.get("strategies", [])
    series = result.get("series", [])

    # V3 format detection
    is_v3 = bool(metrics_table) or bool(series)

    if is_v3:
        # V3 multi-strategy format
        lines = [
            "BACKTEST COMPLETE",
            "",
            f"Period: {meta.get('start_date', 'N/A')[:10]} to {meta.get('end_date', 'N/A')[:10]}",
            f"Bars: {meta.get('total_bars', 'N/A')} | Strategies: {meta.get('num_strategies', len(strategies))} | Runs: {meta.get('num_runs', len(series))}",
            "",
        ]

        if metrics_table:
            lines.append("PERFORMANCE SUMMARY:")
            for row in metrics_table:
                m = row.get("metrics", {})
                ret = m.get("total_return", 0)
                sharpe = m.get("sharpe_ratio", 0)
                trades = m.get("num_trades", 0)
                max_dd = m.get("max_drawdown", 0)
                lines.append(
                    f"  {row.get('name', 'Unknown')}: "
                    f"Return={ret:+.2%} | Sharpe={sharpe:.2f} | "
                    f"Trades={trades} | MaxDD={max_dd:.2%}"
                )
            lines.append("")

        if strategies:
            lines.append("STRATEGIES:")
            for s in strategies:
                lines.append(f"  - {s.get('name', 'Unknown')}: {s.get('description', '')[:50]}")
            lines.append("")

        return "\n".join(lines)

    # Legacy V2 format fallback
    metrics = result.get("metrics", {})
    num_bars = len(result.get("data", []))
    attempts = result.get("attempts", 1)
    direction = result.get("direction", "longonly")

    symbols = result.get("symbols", []) or meta.get("symbols", [])
    symbols_str = ", ".join(symbols) if symbols else "unknown"

    metrics_by_symbol = result.get("metrics_by_symbol", {})

    lines = [
        "BACKTEST COMPLETE",
        "",
        f"Symbols: {symbols_str} | Timeframe: {result.get('timeframe', '1Day')} | Direction: {direction}",
        f"Data: {num_bars} bars | Generated in {attempts} attempt(s)",
        "",
    ]

    if metrics_by_symbol:
        lines.append("RESULTS BY SYMBOL:")
        for sym, m in metrics_by_symbol.items():
            ret = m.get('total_return', 0)
            sharpe = m.get('sharpe_ratio', 0)
            trades = m.get('num_trades', 0)
            max_dd = m.get('max_drawdown', 0)
            lines.append(f"  {sym}: Return={ret:.2%} | Sharpe={sharpe:.2f} | Trades={trades} | MaxDD={max_dd:.2%}")
        lines.append("")
    else:
        lines.extend([
            f"Return: {metrics.get('total_return', 0):.2%} | Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Trades: {metrics.get('num_trades', 0)} | Win Rate: {metrics.get('win_rate', 0):.1%} | Max DD: {metrics.get('max_drawdown', 0):.2%}",
            "",
        ])

    # Execution settings summary
    exec_price = result.get("execution_price", "close")
    slippage = result.get("slippage", 0.0)
    stop_loss = result.get("stop_loss")
    take_profit = result.get("take_profit")

    exec_parts = [f"Execution: {exec_price}"]
    if slippage and slippage > 0:
        exec_parts.append(f"Slippage: {slippage*10000:.0f}bps")
    if stop_loss is not None:
        exec_parts.append(f"SL: {stop_loss:.1%}")
    if take_profit is not None:
        exec_parts.append(f"TP: {take_profit:.1%}")
    lines.append(" | ".join(exec_parts))

    lines.extend([
        "",
        "Full results with equity curves are displayed in the widget above.",
        "Strategy parameters and generated code are available in the widget's 'Strategy Details' section.",
    ])

    return "\n".join(lines)
