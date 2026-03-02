"""
Base Strategy Descriptions
==========================

Core strategy logic descriptions used across all conditions.
These describe WHAT the strategy does (the logic), not HOW to validate it.
"""

from pathlib import Path

# Load backtest files
_BACKTESTS_DIR = Path(__file__).parent.parent.parent / "backtests"
_BACKTEST_SIMPLE = (_BACKTESTS_DIR / "simple.py").read_text()
_BACKTEST_MEDIUM = (_BACKTESTS_DIR / "medium.py").read_text()
_BACKTEST_COMPLEX = (_BACKTESTS_DIR / "complex.py").read_text()

STRATEGY_BASE_SIMPLE = """
Generate a trading strategy that implements RSI mean reversion.

## Strategy Logic

- Calculate RSI with period = 14
- Go long when RSI crosses below 30 (oversold)
- Exit when RSI crosses above 70 (overbought)
- Long-only, single asset
"""

STRATEGY_BASE_MEDIUM = """
Generate a trading strategy that combines MACD crossover entries with ATR-based trailing stops.

## Strategy Logic

- MACD parameters: fast=12, slow=26, signal=9
- Trend filter: 50-period SMA
- ATR period: 14
- Trailing stop: 2.0 × ATR from highest price since entry

Entry conditions (all must be true):
- MACD line crosses above Signal line
- Price is above 50-period SMA

Exit conditions (any):
- MACD line crosses below Signal line
- Price falls below (highest_since_entry - 2.0 × ATR)

Long-only, single asset.
"""

STRATEGY_BASE_COMPLEX = """
Generate a statistical arbitrage pairs trading strategy.

## Strategy Logic

Spread calculation:
- Hedge ratio: rolling OLS regression, lookback = 60 periods
- Spread = Price_A - (hedge_ratio × Price_B)

Z-score calculation:
- Rolling mean of spread: 20 periods
- Rolling std of spread: 20 periods
- Z-score = (spread - rolling_mean) / rolling_std

Entry conditions:
- Z-score > 2.0: Short Asset A (1 unit), Long Asset B (hedge_ratio units)
- Z-score < -2.0: Long Asset A (1 unit), Short Asset B (hedge_ratio units)

Exit conditions:
- Z-score crosses 0.0: Close both positions
- |Z-score| > 3.0: Stop-loss, close both positions

Position sizing:
- Fixed notional: $10,000 per leg
- No volatility adjustment
"""

# Output sections for each strategy type
OUTPUT_SIMPLE = f"""
## Output

Export the the following function:
- `generate_signals`

The function will be passed into the `run_backtest` function to be run.

The following is the backtest runner that will use your generated code:

```python
{_BACKTEST_SIMPLE}
```
"""

OUTPUT_MEDIUM = f"""
## Output

Export the following functions:
- `compute_indicators`
- `order_func`

The functions will be passed into the `run_backtest` function to be run.

**CRITICAL: DO NOT USE NUMBA.** The backtest runs with `use_numba=False`.
- Do NOT use `@njit` or any numba decorators
- Do NOT use `vbt.portfolio.nb.*` functions (like `order_nb`, `no_order_nb`)
- Do NOT use `vbt.portfolio.enums.*`
- Return simple Python tuples, not numba objects

The following is the backtest runner that will use your generated code:

```python
{_BACKTEST_MEDIUM}
```
"""

OUTPUT_COMPLEX = f"""
## Output

Export the following functions:
- `compute_spread_indicators`
- `order_func`

The functions will be passed into the `run_backtest` function to be run.

**CRITICAL: DO NOT USE NUMBA.** The backtest runs with `use_numba=False`.
- Do NOT use `@njit` or any numba decorators
- Do NOT use `vbt.portfolio.nb.*` functions (like `order_nb`, `no_order_nb`)
- Do NOT use `vbt.portfolio.enums.*`
- Return simple Python tuples, not numba objects

The following is the backtest runner that will use your generated code:

```python
{_BACKTEST_COMPLEX}
```
"""

# Allowed libraries sections
ALLOWED_LIBS_SIMPLE = """
## Allowed Libraries

- pandas
- numpy
- vectorbt (vbt)
"""

ALLOWED_LIBS_MEDIUM = """
## Allowed Libraries

- pandas
- numpy
- vectorbt (vbt)
"""

ALLOWED_LIBS_COMPLEX = """
## Allowed Libraries

- pandas
- numpy
- vectorbt (vbt)
- scipy.stats (for regression)
"""
