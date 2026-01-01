"""Result Combiner - Merges multiple backtest results into unified output.

This module combines multiple BacktestRunResults into a single
CombinedBacktestResult suitable for display in the widget.

Key Features:
- Timestamp alignment via inner join (all strategies share same date range)
- Optional normalization (all curves start at 100)
- Builds chart series and metrics table for the widget
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .models import (
    BacktestRunResult,
    ChartSeries,
    CombinedBacktestResult,
    CombinedMeta,
    DisplayConfig,
    GeneratedStrategy,
    MetricsRow,
    StrategyDetails,
)

logger = logging.getLogger(__name__)


def align_timestamps(
    results: List[BacktestRunResult],
) -> Tuple[pd.DatetimeIndex, List[BacktestRunResult]]:
    """Align all results to common timestamp range (inner join).

    This ensures apples-to-apples comparison:
    - If Strategy A has 200-day MA warmup, first 200 days are NaN
    - If Strategy B starts trading from day 1
    - Result: Both clipped to start from day 200

    Args:
        results: List of BacktestRunResults to align.

    Returns:
        Tuple of (common_index, aligned_results).
    """
    if not results:
        return pd.DatetimeIndex([]), []

    logger.info("=" * 60)
    logger.info("ALIGNING TIMESTAMPS")
    logger.info("=" * 60)

    # Collect all timestamp indices
    all_indices = []
    for result in results:
        for sym_result in result.results_by_symbol.values():
            if sym_result.timestamps:
                idx = pd.DatetimeIndex(sym_result.timestamps)
                all_indices.append(idx)
                logger.info(
                    f"  {result.strategy_name} ({sym_result.symbol}): "
                    f"{len(idx)} bars, {idx[0]} to {idx[-1]}"
                )

    if not all_indices:
        logger.warning("No timestamps found in any result")
        return pd.DatetimeIndex([]), results

    # Inner join: intersection of all date ranges
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.intersection(idx)

    logger.info(f"Common date range: {len(common_index)} bars")
    if len(common_index) > 0:
        logger.info(f"  Start: {common_index[0]}")
        logger.info(f"  End: {common_index[-1]}")

    # Clip all results to common range
    aligned_results = []
    for result in results:
        aligned_result = _clip_result_to_index(result, common_index)
        aligned_results.append(aligned_result)

    return common_index, aligned_results


def _clip_result_to_index(
    result: BacktestRunResult,
    common_index: pd.DatetimeIndex,
) -> BacktestRunResult:
    """Clip a BacktestRunResult's data to the common index.

    Args:
        result: The result to clip.
        common_index: The target timestamp index.

    Returns:
        New BacktestRunResult with clipped data.
    """
    from .models import BacktestRunResult, SymbolResult, Metrics

    # Create new results_by_symbol with clipped data
    new_results_by_symbol = {}
    for symbol, sym_result in result.results_by_symbol.items():
        if not sym_result.timestamps:
            new_results_by_symbol[symbol] = sym_result
            continue

        # Build Series with original timestamps
        orig_index = pd.DatetimeIndex(sym_result.timestamps)
        equity_series = pd.Series(sym_result.equity_curve, index=orig_index)

        # Reindex to common index
        clipped_equity = equity_series.reindex(common_index)

        # Build new SymbolResult
        new_results_by_symbol[symbol] = SymbolResult(
            symbol=sym_result.symbol,
            timestamps=[str(ts) for ts in common_index],
            equity_curve=clipped_equity.tolist(),
            positions=sym_result.positions,  # TODO: clip positions too
            metrics=sym_result.metrics,  # Metrics stay the same
            trades=sym_result.trades,
        )

    return BacktestRunResult(
        strategy_name=result.strategy_name,
        direction=result.direction,
        results_by_symbol=new_results_by_symbol,
        execution=result.execution,
        meta=result.meta,
        success=result.success,
        error=result.error,
    )


def normalize_to_100(equity_curve: List[float]) -> List[float]:
    """Normalize an equity curve to start at 100.

    Args:
        equity_curve: List of equity values.

    Returns:
        Normalized curve starting at 100.
    """
    if not equity_curve or equity_curve[0] == 0:
        return equity_curve

    factor = 100.0 / equity_curve[0]
    return [v * factor for v in equity_curve]


def build_timeseries(
    timestamps: List[str],
    values: List[float],
) -> List[Dict[str, Any]]:
    """Build timeseries data for chart.

    Args:
        timestamps: List of ISO timestamp strings.
        values: List of values.

    Returns:
        List of {"time": ..., "value": ...} dicts.
    """
    return [
        {"time": ts, "value": v}
        for ts, v in zip(timestamps, values)
    ]


def build_strategy_details(
    strategy: GeneratedStrategy,
    directions_run: List[str],
) -> StrategyDetails:
    """Build StrategyDetails for reproducibility section.

    Args:
        strategy: The GeneratedStrategy.
        directions_run: List of directions that were executed.

    Returns:
        StrategyDetails object.
    """
    return StrategyDetails(
        name=strategy.spec.name,
        description=strategy.spec.description,
        code=strategy.code,
        data_schema=strategy.data_schema,
        param_schema=strategy.param_schema,
        params_used=strategy.params,
        execution={
            "directions": directions_run,
            "execution_price": strategy.spec.execution_price,
            "slippage": strategy.spec.slippage,
            "stop_loss": strategy.spec.stop_loss,
            "take_profit": strategy.spec.take_profit,
            "init_cash": strategy.spec.init_cash,
        },
    )


def build_display_name(
    strategy_name: str,
    symbol: str,
    direction: str,
    include_direction: bool = True,
) -> str:
    """Build display name for chart legend and metrics table.

    Args:
        strategy_name: Name of the strategy.
        symbol: Symbol being traded.
        direction: Direction of this run.
        include_direction: Whether to include direction in name.

    Returns:
        Display name like "AAPL (RSI Long)" or "SPY (Buy & Hold)".
    """
    direction_label = {
        "longonly": "Long",
        "shortonly": "Short",
        "both": "L/S",
    }.get(direction, direction)

    if include_direction and direction != "longonly":
        return f"{symbol} ({strategy_name} {direction_label})"
    else:
        return f"{symbol} ({strategy_name})"


def combine_results(
    results: List[BacktestRunResult],
    strategies: List[GeneratedStrategy],
    display_config: Optional[DisplayConfig] = None,
) -> CombinedBacktestResult:
    """Combine multiple strategy results into unified output.

    Args:
        results: List of BacktestRunResults from executing strategies.
        strategies: List of GeneratedStrategies (for reproducibility info).
        display_config: Optional display configuration.

    Returns:
        CombinedBacktestResult ready for widget display.
    """
    if display_config is None:
        display_config = DisplayConfig()

    logger.info("=" * 60)
    logger.info("COMBINING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Number of results: {len(results)}")
    logger.info(f"Normalize: {display_config.normalize}")
    logger.info(f"Show trades: {display_config.show_trades}")

    # Check for failures
    failed_results = [r for r in results if not r.success]
    if failed_results:
        first_error = failed_results[0].error or "Unknown error"
        logger.error(f"Some strategies failed: {first_error}")
        return CombinedBacktestResult(
            success=False,
            error=f"Strategy execution failed: {first_error}",
        )

    if not results:
        return CombinedBacktestResult(
            success=False,
            error="No results to combine",
        )

    # Step 1: Align timestamps
    common_index, aligned_results = align_timestamps(results)

    if len(common_index) == 0:
        return CombinedBacktestResult(
            success=False,
            error="No common date range found across strategies",
        )

    # Build output structures
    series_list: List[ChartSeries] = []
    metrics_table: List[MetricsRow] = []
    strategy_details: List[StrategyDetails] = []

    # Track which strategies we've seen (for details)
    seen_strategies: Dict[str, List[str]] = {}  # name -> directions

    for result in aligned_results:
        strategy_name = result.strategy_name
        direction = result.direction

        # Track directions per strategy
        if strategy_name not in seen_strategies:
            seen_strategies[strategy_name] = []
        seen_strategies[strategy_name].append(direction)

        # Determine if we need to include direction in display name
        # (only if multiple directions for same strategy)
        include_direction = True  # We'll set this properly later

        for symbol, sym_result in result.results_by_symbol.items():
            if not sym_result.equity_curve:
                continue

            # Apply normalization if requested
            equity_data = sym_result.equity_curve
            if display_config.normalize:
                equity_data = normalize_to_100(equity_data)

            # Build display name
            display_name = build_display_name(
                strategy_name, symbol, direction, include_direction
            )

            # Build chart series
            series_list.append(
                ChartSeries(
                    name=display_name,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    direction=direction,
                    data=build_timeseries(sym_result.timestamps, equity_data),
                    trades=[asdict(t) for t in sym_result.trades]
                    if display_config.show_trades and sym_result.trades
                    else None,
                )
            )

            # Build metrics row
            metrics_dict = {
                "total_return": sym_result.metrics.total_return,
                "cagr": sym_result.metrics.cagr,
                "volatility": sym_result.metrics.volatility,
                "sharpe_ratio": sym_result.metrics.sharpe_ratio,
                "sortino_ratio": sym_result.metrics.sortino_ratio,
                "calmar_ratio": sym_result.metrics.calmar_ratio,
                "max_drawdown": sym_result.metrics.max_drawdown,
                "win_rate": sym_result.metrics.win_rate,
                "num_trades": sym_result.metrics.num_trades,
                "profit_factor": sym_result.metrics.profit_factor,
            }

            metrics_table.append(
                MetricsRow(
                    name=display_name,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    direction=direction,
                    metrics=metrics_dict,
                )
            )

    # Build strategy details from GeneratedStrategies
    for strategy in strategies:
        directions_for_this = seen_strategies.get(strategy.spec.name, [])
        if directions_for_this:
            strategy_details.append(
                build_strategy_details(strategy, directions_for_this)
            )

    # Build combined metadata
    combined_meta = CombinedMeta(
        timeframe="1Day",  # TODO: Get from data
        start_date=str(common_index[0]) if len(common_index) > 0 else "",
        end_date=str(common_index[-1]) if len(common_index) > 0 else "",
        total_bars=len(common_index),
        num_strategies=len(strategies),
        num_runs=len(results),
    )

    logger.info(f"Combined {len(series_list)} equity curves")
    logger.info(f"Combined {len(metrics_table)} metrics rows")
    logger.info(f"Date range: {combined_meta.start_date} to {combined_meta.end_date}")

    return CombinedBacktestResult(
        success=True,
        series=series_list,
        metrics_table=metrics_table,
        strategies=strategy_details,
        meta=combined_meta,
    )


def downsample_series(
    series: ChartSeries,
    max_points: int = 500,
) -> ChartSeries:
    """Downsample a chart series to reduce payload size.

    Args:
        series: The series to downsample.
        max_points: Maximum number of data points.

    Returns:
        Downsampled series.
    """
    if len(series.data) <= max_points:
        return series

    n = len(series.data)
    step = max(1, n // max_points)
    indices = list(range(0, n, step))

    # Always include last point
    if indices[-1] != n - 1:
        indices.append(n - 1)

    downsampled_data = [series.data[i] for i in indices]

    return ChartSeries(
        name=series.name,
        strategy_name=series.strategy_name,
        symbol=series.symbol,
        direction=series.direction,
        data=downsampled_data,
        trades=series.trades,
    )


def downsample_combined_result(
    result: CombinedBacktestResult,
    max_points: int = 500,
) -> CombinedBacktestResult:
    """Downsample all series in a combined result.

    Args:
        result: The combined result.
        max_points: Maximum points per series.

    Returns:
        Combined result with downsampled series.
    """
    if not result.success:
        return result

    downsampled_series = [
        downsample_series(s, max_points) for s in result.series
    ]

    return CombinedBacktestResult(
        success=result.success,
        error=result.error,
        series=downsampled_series,
        metrics_table=result.metrics_table,
        strategies=result.strategies,
        meta=result.meta,
    )
