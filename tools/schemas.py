"""Schemas and data models for the backtesting system.

This module defines the core data structures used throughout the backtesting
pipeline, including DATA_SCHEMA, PARAM_SCHEMA, and error handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Enums for standard values
# =============================================================================

class ExecutionFrequency(str, Enum):
    """Valid execution frequencies for strategy resampling."""
    MIN_1 = "1Min"
    MIN_5 = "5Min"
    MIN_15 = "15Min"
    HOUR_1 = "1Hour"
    DAY_1 = "1Day"
    WEEK_1 = "1Week"
    MONTH_1 = "1Month"


class ExecutionPrice(str, Enum):
    """Price column to use for trade execution."""
    OPEN = "open"
    CLOSE = "close"


class Direction(str, Enum):
    """Position direction mode."""
    LONG_ONLY = "longonly"
    SHORT_ONLY = "shortonly"
    BOTH = "both"


class DataType(str, Enum):
    """Type of data for proper resampling aggregation."""
    OHLCV = "ohlcv"
    FUNDAMENTAL = "fundamental"


class TimestampType(str, Enum):
    """Timestamp semantics for fundamental data."""
    FILING_DATE = "filing_date"
    PERIOD_END = "period_end"


# =============================================================================
# DATA_SCHEMA definitions
# =============================================================================

class DataSlotSchema(BaseModel):
    """Schema for a single data slot in DATA_SCHEMA."""

    frequency: str = Field(
        ...,
        description="Native frequency to fetch from provider (e.g., '1Day', 'quarterly')"
    )
    columns: List[str] = Field(
        ...,
        description="Required columns in the DataFrame"
    )
    data_type: DataType = Field(
        default=DataType.OHLCV,
        description="Type of data for proper resampling aggregation"
    )
    timestamp_type: Optional[TimestampType] = Field(
        default=None,
        description="For fundamental data: 'filing_date' prevents lookahead bias"
    )
    timestamp_tz: str = Field(
        default="America/New_York",
        description="Source timezone for normalization"
    )
    description: str = Field(
        default="",
        description="Human-readable description"
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# PARAM_SCHEMA definitions
# =============================================================================

class ParamDefinition(BaseModel):
    """Schema for a single parameter in PARAM_SCHEMA."""

    type: Literal["int", "float", "bool", "enum"] = Field(
        ...,
        description="Parameter type"
    )
    default: Any = Field(
        ...,
        description="Default value"
    )
    description: str = Field(
        default="",
        description="Human-readable description"
    )
    # For numeric types
    min: Optional[float] = Field(default=None, description="Minimum value")
    max: Optional[float] = Field(default=None, description="Maximum value")
    # For enum types
    values: Optional[List[Any]] = Field(default=None, description="Valid enum values")

    model_config = ConfigDict(extra="forbid")


# Built-in parameters that are always present (injected by orchestrator)
BUILTIN_PARAM_SCHEMA: Dict[str, Dict[str, Any]] = {
    "execution_frequency": {
        "type": "enum",
        "values": ["1Min", "5Min", "15Min", "1Hour", "1Day", "1Week", "1Month"],
        "default": "1Day",
        "description": "Frequency at which the strategy executes. All data is resampled to this frequency.",
    },
    "execution_price": {
        "type": "enum",
        "values": ["open", "close"],
        "default": "close",
        "description": "Price column used for trade execution.",
    },
    "direction": {
        "type": "enum",
        "values": ["longonly", "shortonly", "both"],
        "default": "longonly",
        "description": "Position direction: longonly (buy/sell to cash), shortonly (short/cover), both (long and short positions).",
    },
    "stop_loss": {
        "type": "float",
        "min": 0.001,
        "max": 0.5,
        "default": None,
        "description": "Stop loss as fraction of entry price (e.g., 0.05 = 5%). Null to disable.",
    },
    "take_profit": {
        "type": "float",
        "min": 0.001,
        "max": 2.0,
        "default": None,
        "description": "Take profit as fraction of entry price (e.g., 0.10 = 10%). Null to disable.",
    },
    "trailing_stop": {
        "type": "bool",
        "default": False,
        "description": "If true, stop_loss trails the highest price since entry.",
    },
    "slippage": {
        "type": "float",
        "min": 0,
        "max": 0.05,
        "default": 0.0,
        "description": "Slippage as fraction of price per trade (e.g., 0.001 = 0.1% = 10 bps).",
    },
}


# =============================================================================
# ERROR_SCHEMA definitions
# =============================================================================

class ErrorType(str, Enum):
    """Types of errors that can occur during backtesting."""
    DATA_FETCH_ERROR = "data_fetch_error"
    INVALID_SYMBOL_ERROR = "invalid_symbol_error"
    CODE_GENERATION_ERROR = "code_generation_error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    BACKTEST_ERROR = "backtest_error"
    SCHEMA_MISMATCH_ERROR = "schema_mismatch_error"


@dataclass
class BacktestError:
    """Structured error response."""
    type: ErrorType
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": False,
            "error": {
                "type": self.type.value,
                "message": self.message,
                "details": self.details,
            },
        }


# =============================================================================
# Strategy Output Schema
# =============================================================================

@dataclass
class StrategyOutput:
    """Output from LLM code generation (Phase 2+)."""
    data_schema: Dict[str, DataSlotSchema]
    param_schema: Dict[str, ParamDefinition]
    code: str


# =============================================================================
# Position Target Utilities
# =============================================================================

def apply_direction_filter(
    positions: Dict[str, Any],  # pd.Series values
    direction: str,
) -> Dict[str, Any]:
    """Filter position targets based on direction parameter.

    Args:
        positions: Dict mapping slot names to position Series (+1/0/-1)
        direction: One of 'longonly', 'shortonly', 'both'

    Returns:
        Filtered positions dict
    """
    import pandas as pd

    filtered = {}
    for slot, pos in positions.items():
        if not isinstance(pos, pd.Series):
            filtered[slot] = pos
            continue

        if direction == "longonly":
            # Clamp to [0, 1]: shorts become flat
            filtered[slot] = pos.clip(lower=0)
        elif direction == "shortonly":
            # Clamp to [-1, 0]: longs become flat
            filtered[slot] = pos.clip(upper=0)
        else:  # 'both'
            filtered[slot] = pos
    return filtered


def positions_to_signals(
    positions: Dict[str, Any],  # pd.Series values
) -> Dict[str, Any]:
    """Convert position targets to entry/exit signals for vectorbt.

    Args:
        positions: Dict mapping slot names to position Series (+1/0/-1)

    Returns:
        Dict with 'long_entries', 'long_exits', 'short_entries', 'short_exits'
        as DataFrames (columns = slot names)
    """
    import numpy as np
    import pandas as pd

    slots = list(positions.keys())
    if not slots:
        return {
            'long_entries': pd.DataFrame(),
            'long_exits': pd.DataFrame(),
            'short_entries': pd.DataFrame(),
            'short_exits': pd.DataFrame(),
        }

    index = positions[slots[0]].index

    long_entries = pd.DataFrame(index=index)
    long_exits = pd.DataFrame(index=index)
    short_entries = pd.DataFrame(index=index)
    short_exits = pd.DataFrame(index=index)

    for slot, pos in positions.items():
        if not isinstance(pos, pd.Series):
            continue

        # Use fill_value in shift() to avoid object dtype and FutureWarning
        prev_pos = pos.shift(1, fill_value=0.0)

        # Long entry: transition to +1 from non-+1
        long_entries[slot] = (pos == 1) & (prev_pos != 1)
        # Long exit: transition from +1 to non-+1
        long_exits[slot] = (pos != 1) & (prev_pos == 1)
        # Short entry: transition to -1 from non--1
        short_entries[slot] = (pos == -1) & (prev_pos != -1)
        # Short exit: transition from -1 to non--1
        short_exits[slot] = (pos != -1) & (prev_pos == -1)

    # Apply t+1 execution shift using fill_value to avoid FutureWarning
    return {
        'long_entries': long_entries.shift(1, fill_value=False),
        'long_exits': long_exits.shift(1, fill_value=False),
        'short_entries': short_entries.shift(1, fill_value=False),
        'short_exits': short_exits.shift(1, fill_value=False),
    }


def extract_defaults_from_param_schema(
    param_schema: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract default values from a PARAM_SCHEMA.

    Args:
        param_schema: Dict of parameter definitions

    Returns:
        Dict mapping parameter names to their default values
    """
    defaults = {}
    for name, definition in param_schema.items():
        if "default" in definition:
            defaults[name] = definition["default"]
    return defaults


def merge_params(
    param_schema: Dict[str, Dict[str, Any]],
    user_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge user-provided params with defaults from schema.

    Args:
        param_schema: Dict of parameter definitions with defaults
        user_params: Optional user overrides

    Returns:
        Complete params dict with all values
    """
    # Start with built-in defaults
    merged = extract_defaults_from_param_schema(BUILTIN_PARAM_SCHEMA)

    # Add strategy-specific defaults
    merged.update(extract_defaults_from_param_schema(param_schema))

    # Override with user values
    if user_params:
        merged.update(user_params)

    return merged
