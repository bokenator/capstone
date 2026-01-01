"""MCP Tools package."""

from .common import (
    TIMEFRAME_PARAMS,
    get_alpaca_credentials,
    parse_dt,
    resolve_timeframe,
)
from .backtest import (
    # v3 multi-strategy tool (the only backtest tool)
    MULTI_BACKTEST_TOOL_NAME as BACKTEST_TOOL_NAME,  # Alias for compatibility
    MULTI_BACKTEST_TOOL_SCHEMA as BACKTEST_TOOL_SCHEMA,  # Alias for compatibility
    MultiBacktestInput as BacktestInput,  # Alias for compatibility
    run_multi_backtest as run_backtest,  # Alias for compatibility
    run_multi_strategy_backtest,
    combined_result_to_dict,
    format_result_text,
)
from .models import (
    # v3 data structures
    BacktestMeta,
    BacktestRunResult,
    ChartSeries,
    CombinedBacktestResult,
    CombinedMeta,
    DisplayConfig,
    ExecutionParams,
    GeneratedStrategy,
    Metrics,
    MetricsRow,
    PlannerOutput,
    StrategyDetails,
    StrategySpec,
    SymbolResult,
    Trade,
)
from .strategy_planner import (
    StrategyPlanner,
    get_strategy_planner,
)
from .result_combiner import (
    align_timestamps,
    combine_results,
    downsample_combined_result,
    downsample_series,
    normalize_to_100,
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
    WIDGETS_NOT_AS_TOOLS,
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
    # Backtest tool (v3 multi-strategy)
    "BACKTEST_TOOL_NAME",
    "BACKTEST_TOOL_SCHEMA",
    "BacktestInput",
    "run_backtest",
    "format_result_text",
    "run_multi_strategy_backtest",
    "combined_result_to_dict",
    # v3 Data Models
    "BacktestMeta",
    "BacktestRunResult",
    "ChartSeries",
    "CombinedBacktestResult",
    "CombinedMeta",
    "DisplayConfig",
    "ExecutionParams",
    "GeneratedStrategy",
    "Metrics",
    "MetricsRow",
    "PlannerOutput",
    "StrategyDetails",
    "StrategySpec",
    "SymbolResult",
    "Trade",
    # Strategy Planner (v3)
    "StrategyPlanner",
    "get_strategy_planner",
    # Result Combiner (v3)
    "align_timestamps",
    "combine_results",
    "downsample_combined_result",
    "downsample_series",
    "normalize_to_100",
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
    "WIDGETS_NOT_AS_TOOLS",
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
