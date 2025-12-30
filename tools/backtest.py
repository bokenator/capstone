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

from typing import Any, Dict, List, Optional, cast

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from pydantic import BaseModel, ConfigDict, Field

from .common import get_alpaca_credentials, parse_dt, resolve_timeframe
from .strategy_executor import BacktestResult, StrategyExecutor, get_strategy_executor
from .strategy_generator import CodexAgent, CodexSession, get_codex_agent


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
            "description": "List of ticker symbols to backtest, e.g., ['AAPL', 'MSFT']",
        },
        "base_code": {
            "type": "string",
            "description": "Optional existing strategy code to modify. If provided, the prompt should describe the changes to make. If not provided, a new strategy is created from scratch.",
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
        description="Ticker symbols to backtest",
    )
    base_code: Optional[str] = Field(
        default=None,
        description="Existing strategy code to modify. If provided, prompt describes changes to make.",
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
    # Initialize Codex agent and executor
    agent = get_codex_agent(model="gpt-5.1", max_attempts=5)
    executor = get_strategy_executor(init_cash=payload.init_cash, timeout_seconds=60)

    # Fetch price data for each symbol separately
    try:
        prices_by_symbol = _fetch_prices_for_symbols(
            symbols=payload.symbols,
            timeframe_str=payload.timeframe,
            start=payload.start,
            end=payload.end,
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to fetch price data: {str(e)}",
            "symbol": ", ".join(payload.symbols),
            "timeframe": payload.timeframe,
            "data": [],
            "equity_curves": {},
        }

    if not prices_by_symbol:
        return {
            "success": False,
            "error": "No price data available for any symbol",
            "symbol": ", ".join(payload.symbols),
            "timeframe": payload.timeframe,
            "data": [],
            "equity_curves": {},
        }

    # Use first symbol's prices for strategy development
    first_symbol = list(prices_by_symbol.keys())[0]
    first_prices = prices_by_symbol[first_symbol]

    # Create Codex session (maintains conversation context)
    # If base_code is provided, we're modifying an existing strategy
    session: CodexSession = agent.create_session(
        strategy_prompt=payload.prompt,
        base_code=payload.base_code,
    )

    # Iterative generate-execute-fix loop
    last_result: Optional[BacktestResult] = None
    all_attempts: List[Dict[str, Any]] = []
    successful_code: Optional[str] = None

    while session.attempts < session.max_attempts:
        # Step 1: Generate code
        try:
            code = agent.generate(session)
        except Exception as e:
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
        is_valid, validation_error = agent.validate_code(code)
        if not is_valid:
            all_attempts.append({
                "attempt": session.attempts,
                "code": code,
                "error": validation_error,
                "error_type": "ValidationError",
            })
            agent.feed_validation_error(session, validation_error)
            continue

        # Step 3: Execute code on first symbol to test
        result: BacktestResult = executor.execute(
            code=code,
            prices=first_prices,
            symbols=[first_symbol],
            timeframe=payload.timeframe,
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
            # Check if strategy generated enough signals
            min_signals = 5  # Require at least 5 entry signals
            if result.entry_signals < min_signals:
                # Strategy is too sparse - ask Codex to fix
                all_attempts.append({
                    "attempt": session.attempts,
                    "code": code,
                    "success": False,
                    "error": f"Strategy only generated {result.entry_signals} entry signals (need at least {min_signals})",
                    "error_type": "InsufficientSignals",
                })
                agent.feed_signal_error(session, result.entry_signals, result.exit_signals)
                continue

            # Code works! Save it for running on all symbols
            successful_code = code
            break

        # Step 5: Feed error back to Codex for fixing
        execution_result = result.to_execution_result()
        agent.feed_error(session, execution_result)

    # If we didn't find working code, return failure
    if successful_code is None:
        error_msg = "Max attempts reached without generating valid strategy"
        if last_result:
            error_msg = f"{error_msg}. Last error: {last_result.error}"

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
        }

    # Run the successful strategy on ALL symbols
    equity_curves: Dict[str, List[Dict[str, Any]]] = {}
    all_metrics: Dict[str, Dict[str, Any]] = {}
    combined_data: List[Dict[str, Any]] = []

    print(f"[Backtest] Running strategy on {len(prices_by_symbol)} symbols: {list(prices_by_symbol.keys())}")

    for symbol, prices in prices_by_symbol.items():
        print(f"[Backtest] Executing for {symbol}, prices shape: {prices.shape}")
        result = executor.execute(
            code=successful_code,
            prices=prices,
            symbols=[symbol],
            timeframe=payload.timeframe,
        )
        print(f"[Backtest] {symbol} result: success={result.success}, equity_len={len(result.equity_curve)}, final_equity={result.equity_curve[-1] if result.equity_curve else 'N/A'}")

        if result.success:
            # Store equity curve for this symbol
            equity_curves[symbol] = [
                {"timestamp": ts, "value": eq}
                for ts, eq in zip(result.timestamps, result.equity_curve)
            ]
            all_metrics[symbol] = {
                "total_return": result.metrics.total_return,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "sortino_ratio": result.metrics.sortino_ratio,
                "max_drawdown": result.metrics.max_drawdown,
                "win_rate": result.metrics.win_rate,
                "num_trades": result.metrics.num_trades,
                "profit_factor": result.metrics.profit_factor,
                "entry_signals": result.entry_signals,
                "exit_signals": result.exit_signals,
            }

    # Use first symbol's data for backward compatibility
    first_result = executor.execute(
        code=successful_code,
        prices=first_prices,
        symbols=[first_symbol],
        timeframe=payload.timeframe,
    )

    result_dict = first_result.to_dict()
    result_dict["generated_code"] = successful_code
    result_dict["symbol"] = ", ".join(payload.symbols)
    result_dict["timeframe"] = payload.timeframe
    result_dict["prompt"] = payload.prompt
    result_dict["attempts"] = session.attempts
    result_dict["attempt_history"] = all_attempts

    # Add multi-symbol data
    result_dict["equity_curves"] = equity_curves
    result_dict["metrics_by_symbol"] = all_metrics
    result_dict["symbols"] = list(prices_by_symbol.keys())

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

    lines = [
        f"Backtest complete for {meta.get('symbols', [])} (attempt {attempts}).",
        f"Bars: {num_bars} | Entry signals: {metrics.get('entry_signals', 0)} | Exit signals: {metrics.get('exit_signals', 0)}",
        f"Trades: {metrics.get('num_trades', 0)} | Win rate: {metrics.get('win_rate', 0):.1%}",
        f"Return: {metrics.get('total_return', 0):.2%} | Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | Max DD: {metrics.get('max_drawdown', 0):.2%}",
    ]

    # Include generated code snippet
    code = meta.get("strategy_code", "") or result.get("generated_code", "")
    if code:
        lines.append(f"\nGenerated strategy:\n```python\n{code}\n```")

    return "\n".join(lines)
