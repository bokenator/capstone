"""
Data and Parameter Schemas
==========================

Schema definitions for all three strategy complexity levels.
Used by: C1 (Schema), C4 (Schema+Docs), C5 (Schema+TDD), C7 (All)

Provides both:
- Dict form (for programmatic validation)
- String form (for inclusion in prompts)
"""

# =============================================================================
# SIMPLE STRATEGY (RSI Mean Reversion)
# =============================================================================

DATA_SCHEMA_SIMPLE = {
    "ohlcv": {
        "columns": {
            "open": {"type": "float64", "required": False},
            "high": {"type": "float64", "required": False},
            "low": {"type": "float64", "required": False},
            "close": {"type": "float64", "required": True},
            "volume": {"type": "float64", "required": False},
        },
        "frequency": "1D",
        "description": "Daily OHLCV price data",
    }
}

PARAM_SCHEMA_SIMPLE = {
    "rsi_period": {
        "type": "int",
        "default": 14,
        "min": 2,
        "max": 100,
        "description": "RSI calculation period",
    },
    "oversold": {
        "type": "float",
        "default": 30.0,
        "min": 0.0,
        "max": 50.0,
        "description": "RSI threshold for entry (oversold)",
    },
    "overbought": {
        "type": "float",
        "default": 70.0,
        "min": 50.0,
        "max": 100.0,
        "description": "RSI threshold for exit (overbought)",
    },
}

DATA_SCHEMA_SIMPLE_STR = """
## DATA_SCHEMA

```json
{
  "ohlcv": {
    "columns": {
      "open": {"type": "float64", "required": false},
      "high": {"type": "float64", "required": false},
      "low": {"type": "float64", "required": false},
      "close": {"type": "float64", "required": true},
      "volume": {"type": "float64", "required": false}
    },
    "frequency": "1D",
    "description": "Daily OHLCV price data"
  }
}
```
"""

PARAM_SCHEMA_SIMPLE_STR = """
## PARAM_SCHEMA

```json
{
  "rsi_period": {
    "type": "int",
    "default": 14,
    "min": 2,
    "max": 100,
    "description": "RSI calculation period"
  },
  "oversold": {
    "type": "float",
    "default": 30.0,
    "min": 0.0,
    "max": 50.0,
    "description": "RSI threshold for entry (oversold)"
  },
  "overbought": {
    "type": "float",
    "default": 70.0,
    "min": 50.0,
    "max": 100.0,
    "description": "RSI threshold for exit (overbought)"
  }
}
```
"""

# =============================================================================
# MEDIUM STRATEGY (MACD + ATR Trailing Stop)
# =============================================================================

DATA_SCHEMA_MEDIUM = {
    "ohlcv": {
        "columns": {
            "open": {"type": "float64", "required": False},
            "high": {"type": "float64", "required": True},
            "low": {"type": "float64", "required": False},
            "close": {"type": "float64", "required": True},
            "volume": {"type": "float64", "required": False},
        },
        "frequency": "1D",
        "description": "Daily OHLCV price data",
    }
}

PARAM_SCHEMA_MEDIUM = {
    "macd_fast": {
        "type": "int",
        "default": 12,
        "min": 2,
        "max": 50,
        "description": "MACD fast EMA period",
    },
    "macd_slow": {
        "type": "int",
        "default": 26,
        "min": 10,
        "max": 100,
        "description": "MACD slow EMA period",
    },
    "macd_signal": {
        "type": "int",
        "default": 9,
        "min": 2,
        "max": 50,
        "description": "MACD signal line period",
    },
    "sma_period": {
        "type": "int",
        "default": 50,
        "min": 10,
        "max": 200,
        "description": "SMA trend filter period",
    },
    "atr_period": {
        "type": "int",
        "default": 14,
        "min": 2,
        "max": 50,
        "description": "ATR calculation period",
    },
    "trailing_mult": {
        "type": "float",
        "default": 2.0,
        "min": 0.5,
        "max": 5.0,
        "description": "ATR multiplier for trailing stop",
    },
}

DATA_SCHEMA_MEDIUM_STR = """
## DATA_SCHEMA

```json
{
  "ohlcv": {
    "columns": {
      "open": {"type": "float64", "required": false},
      "high": {"type": "float64", "required": true},
      "low": {"type": "float64", "required": false},
      "close": {"type": "float64", "required": true},
      "volume": {"type": "float64", "required": false}
    },
    "frequency": "1D",
    "description": "Daily OHLCV price data"
  }
}
```
"""

PARAM_SCHEMA_MEDIUM_STR = """
## PARAM_SCHEMA

```json
{
  "macd_fast": {
    "type": "int",
    "default": 12,
    "min": 2,
    "max": 50,
    "description": "MACD fast EMA period"
  },
  "macd_slow": {
    "type": "int",
    "default": 26,
    "min": 10,
    "max": 100,
    "description": "MACD slow EMA period"
  },
  "macd_signal": {
    "type": "int",
    "default": 9,
    "min": 2,
    "max": 50,
    "description": "MACD signal line period"
  },
  "sma_period": {
    "type": "int",
    "default": 50,
    "min": 10,
    "max": 200,
    "description": "SMA trend filter period"
  },
  "atr_period": {
    "type": "int",
    "default": 14,
    "min": 2,
    "max": 50,
    "description": "ATR calculation period"
  },
  "trailing_mult": {
    "type": "float",
    "default": 2.0,
    "min": 0.5,
    "max": 5.0,
    "description": "ATR multiplier for trailing stop"
  }
}
```

## ORDER_CONTEXT_SCHEMA

The `order_func` receives a vectorbt OrderContext `c` with these attributes:

| Attribute       | Type  | Description                                    |
|-----------------|-------|------------------------------------------------|
| `c.i`           | int   | Current bar index (0-indexed)                  |
| `c.position_now`| float | Current position size (0.0 if flat)            |
| `c.cash_now`    | float | Current available cash                         |

**Order Return Format:** Return a tuple `(size, size_type, direction)`:
- `size`: float - Number of shares (positive=buy, negative=sell)
- `size_type`: int - 0=Amount, 1=Value, 2=Percent
- `direction`: int - 0=Both, 1=LongOnly, 2=ShortOnly

**Examples:**
- Buy 100 shares: `return (100.0, 0, 1)`
- Sell/close position: `return (-c.position_now, 0, 1)`
- No action: `return (0.0, 0, 0)`
"""

# =============================================================================
# COMPLEX STRATEGY (Pairs Trading)
# =============================================================================

DATA_SCHEMA_COMPLEX = {
    "asset_a": {
        "columns": {
            "close": {"type": "float64", "required": True},
        },
        "frequency": "1D",
        "description": "Daily close prices for Asset A",
    },
    "asset_b": {
        "columns": {
            "close": {"type": "float64", "required": True},
        },
        "frequency": "1D",
        "description": "Daily close prices for Asset B",
    },
}

PARAM_SCHEMA_COMPLEX = {
    "hedge_lookback": {
        "type": "int",
        "default": 60,
        "min": 20,
        "max": 252,
        "description": "Lookback period for hedge ratio regression",
    },
    "zscore_lookback": {
        "type": "int",
        "default": 20,
        "min": 5,
        "max": 100,
        "description": "Lookback period for z-score calculation",
    },
    "entry_threshold": {
        "type": "float",
        "default": 2.0,
        "min": 1.0,
        "max": 4.0,
        "description": "Z-score threshold for entry",
    },
    "exit_threshold": {
        "type": "float",
        "default": 0.0,
        "min": -1.0,
        "max": 1.0,
        "description": "Z-score threshold for exit",
    },
    "stop_threshold": {
        "type": "float",
        "default": 3.0,
        "min": 2.0,
        "max": 5.0,
        "description": "Z-score threshold for stop-loss",
    },
    "notional_per_leg": {
        "type": "float",
        "default": 10000.0,
        "min": 1000.0,
        "max": 1000000.0,
        "description": "Fixed notional amount per leg in dollars",
    },
}

DATA_SCHEMA_COMPLEX_STR = """
## DATA_SCHEMA

```json
{
  "asset_a": {
    "columns": {
      "close": {"type": "float64", "required": true}
    },
    "frequency": "1D",
    "description": "Daily close prices for Asset A"
  },
  "asset_b": {
    "columns": {
      "close": {"type": "float64", "required": true}
    },
    "frequency": "1D",
    "description": "Daily close prices for Asset B"
  }
}
```
"""

PARAM_SCHEMA_COMPLEX_STR = """
## PARAM_SCHEMA

```json
{
  "hedge_lookback": {
    "type": "int",
    "default": 60,
    "min": 20,
    "max": 252,
    "description": "Lookback period for hedge ratio regression"
  },
  "zscore_lookback": {
    "type": "int",
    "default": 20,
    "min": 5,
    "max": 100,
    "description": "Lookback period for z-score calculation"
  },
  "entry_threshold": {
    "type": "float",
    "default": 2.0,
    "min": 1.0,
    "max": 4.0,
    "description": "Z-score threshold for entry"
  },
  "exit_threshold": {
    "type": "float",
    "default": 0.0,
    "min": -1.0,
    "max": 1.0,
    "description": "Z-score threshold for exit"
  },
  "stop_threshold": {
    "type": "float",
    "default": 3.0,
    "min": 2.0,
    "max": 5.0,
    "description": "Z-score threshold for stop-loss"
  },
  "notional_per_leg": {
    "type": "float",
    "default": 10000.0,
    "min": 1000.0,
    "max": 1000000.0,
    "description": "Fixed notional amount per leg in dollars"
  }
}
```

## ORDER_CONTEXT_SCHEMA

The `order_func` receives a vectorbt OrderContext `c` with these attributes:

| Attribute       | Type  | Description                                    |
|-----------------|-------|------------------------------------------------|
| `c.i`           | int   | Current bar index (0-indexed)                  |
| `c.col`         | int   | Current asset column (0=Asset A, 1=Asset B)    |
| `c.position_now`| float | Current position size for this asset           |
| `c.cash_now`    | float | Current available cash                         |

**Order Return Format:** Return a tuple `(size, size_type, direction)`:
- `size`: float - Number of shares (positive=buy, negative=sell)
- `size_type`: int - 0=Amount, 1=Value, 2=Percent
- `direction`: int - 0=Both, 1=LongOnly, 2=ShortOnly

**Examples:**
- Buy 50 shares: `return (50.0, 0, 0)`
- Sell 50 shares: `return (-50.0, 0, 0)`
- Close position: `return (-c.position_now, 0, 0)`
- No action: `return (0.0, 0, 0)`
"""
