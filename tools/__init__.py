"""MCP Tools package."""

from .backtest import (
    BACKTEST_TOOL_NAME,
    BACKTEST_TOOL_SCHEMA,
    BacktestInput,
    ma_crossover_backtest,
)
from .common import (
    TIMEFRAME_PARAMS,
    get_alpaca_credentials,
    parse_dt,
    resolve_timeframe,
)
from .equity_prices import (
    EQUITY_TOOL_NAME,
    EQUITY_TOOL_SCHEMA,
    EquityPricesInput,
    fetch_equity_prices,
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
    # Backtest
    "BACKTEST_TOOL_NAME",
    "BACKTEST_TOOL_SCHEMA",
    "BacktestInput",
    "ma_crossover_backtest",
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
