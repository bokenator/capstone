"""
C1: Schema-Constrained Condition Prompts
========================================
S=on, D=off, T=off

Adds to control:
- DATA_SCHEMA: Required input data structure
- PARAM_SCHEMA: Parameters with types, ranges, defaults
- Function signatures with type hints
- Column access validation (code must only access declared columns)
"""

from .shared import (
    STRATEGY_BASE_SIMPLE,
    STRATEGY_BASE_MEDIUM,
    STRATEGY_BASE_COMPLEX,
    DATA_SCHEMA_SIMPLE_STR,
    DATA_SCHEMA_MEDIUM_STR,
    DATA_SCHEMA_COMPLEX_STR,
    PARAM_SCHEMA_SIMPLE_STR,
    PARAM_SCHEMA_MEDIUM_STR,
    PARAM_SCHEMA_COMPLEX_STR,
    SIGNATURE_SIMPLE,
    SIGNATURE_MEDIUM,
    SIGNATURE_COMPLEX,
    OUTPUT_SIMPLE,
    OUTPUT_MEDIUM,
    OUTPUT_COMPLEX,
    ALLOWED_LIBS_SIMPLE,
    ALLOWED_LIBS_MEDIUM,
    ALLOWED_LIBS_COMPLEX,
    VALIDATION_SCHEMA,
)

# =============================================================================
# STRATEGY 1: RSI Mean Reversion (Simple - from_signals)
# =============================================================================

STRATEGY_1_SIMPLE = (
    STRATEGY_BASE_SIMPLE
    + DATA_SCHEMA_SIMPLE_STR
    + PARAM_SCHEMA_SIMPLE_STR
    + SIGNATURE_SIMPLE
    + VALIDATION_SCHEMA
    + OUTPUT_SIMPLE
    + ALLOWED_LIBS_SIMPLE
    + "\nGenerate the complete implementation.\n"
)

# =============================================================================
# STRATEGY 2: MACD + ATR Trailing Stop (Medium - from_order_func flexible=False)
# =============================================================================

STRATEGY_2_MEDIUM = (
    STRATEGY_BASE_MEDIUM
    + DATA_SCHEMA_MEDIUM_STR
    + PARAM_SCHEMA_MEDIUM_STR
    + SIGNATURE_MEDIUM
    + VALIDATION_SCHEMA
    + OUTPUT_MEDIUM
    + ALLOWED_LIBS_MEDIUM
    + "\nGenerate the complete implementation.\n"
)

# =============================================================================
# STRATEGY 3: Pairs Trading (Complex - from_order_func flexible=True)
# =============================================================================

STRATEGY_3_COMPLEX = (
    STRATEGY_BASE_COMPLEX
    + DATA_SCHEMA_COMPLEX_STR
    + PARAM_SCHEMA_COMPLEX_STR
    + SIGNATURE_COMPLEX
    + VALIDATION_SCHEMA
    + OUTPUT_COMPLEX
    + ALLOWED_LIBS_COMPLEX
    + "\nGenerate the complete implementation.\n"
)

# =============================================================================
# Prompt Collection
# =============================================================================

PROMPTS = {
    "simple": {
        "name": "RSI Mean Reversion",
        "interface": "from_signals",
        "prompt": STRATEGY_1_SIMPLE,
    },
    "medium": {
        "name": "MACD + ATR Trailing Stop",
        "interface": "from_order_func(flexible=False)",
        "prompt": STRATEGY_2_MEDIUM,
    },
    "complex": {
        "name": "Pairs Trading",
        "interface": "from_order_func(flexible=True)",
        "prompt": STRATEGY_3_COMPLEX,
    },
}

# Condition metadata
CONDITION = {
    "id": "C1",
    "name": "Schema-Constrained",
    "schema_enabled": True,
    "doc_grounding_enabled": False,
    "tdd_enabled": False,
}
