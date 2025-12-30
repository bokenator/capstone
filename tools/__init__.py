"""MCP Tools package."""

from .common import (
    TIMEFRAME_PARAMS,
    get_alpaca_credentials,
    parse_dt,
    resolve_timeframe,
)
from .backtest import (
    BACKTEST_TOOL_NAME,
    BACKTEST_TOOL_SCHEMA,
    BacktestInput,
    format_result_text,
    run_backtest,
)
from .equity_prices import (
    EQUITY_TOOL_NAME,
    EQUITY_TOOL_SCHEMA,
    EquityPricesInput,
    fetch_equity_prices,
)
from .strategy_executor import (
    BacktestMetrics,
    BacktestResult,
    ExecutionTimeout,
    StrategyExecutor,
    get_strategy_executor,
)
from .strategy_generator import (
    CodexAgent,
    CodexSession,
    ExecutionResult,
    get_codex_agent,
)
from .widgets import (
    MIME_TYPE,
    WIDGET_TOOL_SCHEMA,
    Widget,
    WidgetInput,
    get_widgets,
    load_widget_html,
    resource_description,
    tool_invocation_meta,
    tool_meta,
)

__all__ = [
    # Common
    "TIMEFRAME_PARAMS",
    "get_alpaca_credentials",
    "parse_dt",
    "resolve_timeframe",
    # Equity prices
    "EQUITY_TOOL_NAME",
    "EQUITY_TOOL_SCHEMA",
    "EquityPricesInput",
    "fetch_equity_prices",
    # Backtest (AI-powered)
    "BACKTEST_TOOL_NAME",
    "BACKTEST_TOOL_SCHEMA",
    "BacktestInput",
    "run_backtest",
    "format_result_text",
    # Strategy generator (Codex agent)
    "CodexAgent",
    "CodexSession",
    "ExecutionResult",
    "get_codex_agent",
    # Strategy executor
    "BacktestMetrics",
    "BacktestResult",
    "ExecutionTimeout",
    "StrategyExecutor",
    "get_strategy_executor",
    # Widgets
    "MIME_TYPE",
    "WIDGET_TOOL_SCHEMA",
    "Widget",
    "WidgetInput",
    "get_widgets",
    "load_widget_html",
    "resource_description",
    "tool_invocation_meta",
    "tool_meta",
]
