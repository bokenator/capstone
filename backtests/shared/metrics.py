"""
Metrics Extraction
==================

Functions to extract standardized metrics from vectorbt Portfolio objects.
"""

from typing import Any

import numpy as np


def extract_metrics_from_portfolio(pf) -> dict[str, Any]:
    """
    Extract standard metrics from a vectorbt Portfolio object.

    Args:
        pf: vectorbt Portfolio object

    Returns:
        Dict of metrics
    """
    try:
        stats = pf.stats()

        metrics = {
            "total_return": float(pf.total_return()) if hasattr(pf, 'total_return') else None,
            "annualized_return": float(stats.get("Total Return [%]", 0)) / 100 if "Total Return [%]" in stats else None,
            "sharpe_ratio": float(stats.get("Sharpe Ratio", np.nan)),
            "sortino_ratio": float(stats.get("Sortino Ratio", np.nan)),
            "max_drawdown": float(stats.get("Max Drawdown [%]", 0)) / 100 if "Max Drawdown [%]" in stats else None,
            "volatility": float(stats.get("Annualized Volatility [%]", 0)) / 100 if "Annualized Volatility [%]" in stats else None,
            "calmar_ratio": float(stats.get("Calmar Ratio", np.nan)),
            "total_trades": int(stats.get("Total Trades", 0)),
            "win_rate": float(stats.get("Win Rate [%]", 0)) / 100 if "Win Rate [%]" in stats else None,
            "profit_factor": float(stats.get("Profit Factor", np.nan)),
            "exposure_time": float(stats.get("Exposure Time [%]", 0)) / 100 if "Exposure Time [%]" in stats else None,
        }

        # Handle NaN and Inf values
        for key, value in metrics.items():
            if value is not None:
                if np.isnan(value) or np.isinf(value):
                    metrics[key] = None

        return metrics

    except Exception as e:
        return {"error": str(e)}
