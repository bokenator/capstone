"""Backtest tool - AI-powered strategy generation and backtesting.

This module provides the MCP tool that uses Codex as an agentic coding tool
to iteratively generate and fix strategy code until it executes successfully.

Workflow:
1. Codex generates strategy code from natural language prompt
2. Code is validated for safety
3. Code is executed in sandbox with vectorbt
4. If error, error details are fed back to Codex (same conversation context)
5. Codex fixes the code
6. Repeat until success or max attempts
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from pydantic import BaseModel, ConfigDict, Field

from .common import get_alpaca_credentials, parse_dt, resolve_timeframe
from .schemas import extract_defaults_from_param_schema, merge_params
from .strategy_executor import BacktestResult, StrategyExecutor, get_strategy_executor
from .strategy_generator import CodexAgent, CodexSession, get_codex_agent

# Configure logging
logger = logging.getLogger(__name__)


BACKTEST_TOOL_NAME = "backtest"

BACKTEST_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "Natural language description of the trading strategy to generate and test. Example: 'Buy when RSI drops below 30, sell when it rises above 70'. If base_code is provided, this describes the modifications to make.",
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
            "description": "Optional existing strategy code to modify. If provided, the prompt should describe the changes to make. If not provided, a new strategy is created from scratch.",
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
        description="Existing strategy code to modify. If provided, prompt describes changes to make.",
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
    api_key, api_secret = get_alpaca_credentials()
    client = StockHistoricalDataClient(api_key, api_secret)

    timeframe = resolve_timeframe(timeframe_str)
    start_dt = parse_dt(start)
    end_dt = parse_dt(end)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        limit=limit,
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
    agent = get_codex_agent(model="gpt-5.1", max_attempts=5)
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
            agent.feed_validation_error(session, validation_error)
            continue
        logger.info("Code validation PASSED")

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


def format_result_text(result: Dict[str, Any]) -> str:
    """Format backtest result as human-readable text."""
    if not result.get("success", False):
        attempts = result.get("attempts", 0)
        return f"Backtest failed after {attempts} attempts: {result.get('error', 'Unknown error')}"

    metrics = result.get("metrics", {})
    meta = result.get("meta", {})
    num_bars = len(result.get("data", []))
    attempts = result.get("attempts", 1)
    direction = result.get("direction", "longonly")

    # Get position/signal counts
    position_changes = metrics.get("position_changes", 0)
    long_entries = metrics.get("long_entries", 0)
    long_exits = metrics.get("long_exits", 0)
    short_entries = metrics.get("short_entries", 0)
    short_exits = metrics.get("short_exits", 0)

    # Get built-in execution params
    exec_price = result.get("execution_price", "close")
    stop_loss = result.get("stop_loss")
    take_profit = result.get("take_profit")
    trailing_stop = result.get("trailing_stop", False)
    slippage = result.get("slippage", 0.0)

    lines = [
        f"Backtest complete for {meta.get('symbols', [])} (attempt {attempts}, direction: {direction}).",
        f"Bars: {num_bars} | Position changes: {position_changes}",
        f"Long entries: {long_entries} | Long exits: {long_exits}",
    ]

    if direction in ("both", "shortonly"):
        lines.append(f"Short entries: {short_entries} | Short exits: {short_exits}")

    lines.extend([
        f"Trades: {metrics.get('num_trades', 0)} | Win rate: {metrics.get('win_rate', 0):.1%}",
        f"Return: {metrics.get('total_return', 0):.2%} | Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | Max DD: {metrics.get('max_drawdown', 0):.2%}",
    ])

    # Add execution params if non-default
    exec_info = [f"Execution: {exec_price}"]
    if stop_loss is not None:
        exec_info.append(f"SL: {stop_loss:.1%}")
        if trailing_stop:
            exec_info.append("(trailing)")
    if take_profit is not None:
        exec_info.append(f"TP: {take_profit:.1%}")
    if slippage and slippage > 0:
        exec_info.append(f"Slippage: {slippage:.2%}")
    if len(exec_info) > 1:  # Only show if there's more than just execution price
        lines.append(" | ".join(exec_info))

    # Include generated code snippet
    code = meta.get("strategy_code", "") or result.get("generated_code", "")
    if code:
        lines.append(f"\nGenerated strategy:\n```python\n{code}\n```")

    return "\n".join(lines)
