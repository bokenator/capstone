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
from .schemas import (
    BUILTIN_PARAM_SCHEMA,
    BacktestError,
    DataSlotSchema,
    DataType,
    Direction,
    ErrorType,
    ExecutionFrequency,
    ExecutionPrice,
    ParamDefinition,
    StrategyOutput,
    TimestampType,
    apply_direction_filter,
    extract_defaults_from_param_schema,
    merge_params,
    positions_to_signals,
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
from .providers import (
    AlpacaProvider,
    DataProvider,
    ProviderRegistry,
    get_provider_registry,
    resample_to_frequency,
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
    # Schemas (v2 architecture)
    "BUILTIN_PARAM_SCHEMA",
    "BacktestError",
    "DataSlotSchema",
    "DataType",
    "Direction",
    "ErrorType",
    "ExecutionFrequency",
    "ExecutionPrice",
    "ParamDefinition",
    "StrategyOutput",
    "TimestampType",
    "apply_direction_filter",
    "extract_defaults_from_param_schema",
    "merge_params",
    "positions_to_signals",
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
    # Providers
    "AlpacaProvider",
    "DataProvider",
    "ProviderRegistry",
    "get_provider_registry",
    "resample_to_frequency",
]
