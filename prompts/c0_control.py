"""
C0: Control Condition Prompts
=============================
No treatments applied (S=off, D=off, T=off)

The agent generates code with minimal constraints:
- Just the function name to export
- Allowed libraries
- Strategy description with ALL parameters specified
- No schema, no API verification, no tests
"""

from .shared import (
    STRATEGY_BASE_SIMPLE,
    STRATEGY_BASE_MEDIUM,
    STRATEGY_BASE_COMPLEX,
    OUTPUT_SIMPLE,
    OUTPUT_MEDIUM,
    OUTPUT_COMPLEX,
    ALLOWED_LIBS_SIMPLE,
    ALLOWED_LIBS_MEDIUM,
    ALLOWED_LIBS_COMPLEX,
)

# =============================================================================
# STRATEGY 1: RSI Mean Reversion (Simple - from_signals)
# =============================================================================

STRATEGY_1_SIMPLE = (
    STRATEGY_BASE_SIMPLE
    + OUTPUT_SIMPLE
    + ALLOWED_LIBS_SIMPLE
    + "\nGenerate the complete implementation.\n"
)

# =============================================================================
# STRATEGY 2: MACD + ATR Trailing Stop (Medium - from_order_func flexible=False)
# =============================================================================

STRATEGY_2_MEDIUM = (
    STRATEGY_BASE_MEDIUM
    + OUTPUT_MEDIUM
    + ALLOWED_LIBS_MEDIUM
    + "\nGenerate the complete implementation.\n"
)

# =============================================================================
# STRATEGY 3: Pairs Trading (Complex - from_order_func flexible=True)
# =============================================================================

STRATEGY_3_COMPLEX = (
    STRATEGY_BASE_COMPLEX
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
    "id": "C0",
    "name": "Control",
    "schema_enabled": False,
    "doc_grounding_enabled": False,
    "tdd_enabled": False,
}
