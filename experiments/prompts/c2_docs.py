"""
C2: Documentation-Grounded Condition Prompts
============================================
S=off, D=on, T=off

Adds to control:
- API citations required for third-party library calls
- Must use module-qualified API calls (e.g., pd.Series.rolling, not .rolling)
- VAS (Verified API Surface) verification
- No schema constraints, no tests
"""

from .shared import (
    STRATEGY_BASE_SIMPLE,
    STRATEGY_BASE_MEDIUM,
    STRATEGY_BASE_COMPLEX,
    VAS_DESCRIPTION,
    API_CITATION_SIMPLE,
    API_CITATION_MEDIUM,
    API_CITATION_COMPLEX,
    OUTPUT_SIMPLE,
    OUTPUT_MEDIUM,
    OUTPUT_COMPLEX,
    ALLOWED_LIBS_SIMPLE,
    ALLOWED_LIBS_MEDIUM,
    ALLOWED_LIBS_COMPLEX,
    VALIDATION_VAS,
)

# =============================================================================
# STRATEGY 1: RSI Mean Reversion (Simple - from_signals)
# =============================================================================

STRATEGY_1_SIMPLE = (
    STRATEGY_BASE_SIMPLE
    + VAS_DESCRIPTION
    + API_CITATION_SIMPLE
    + VALIDATION_VAS
    + OUTPUT_SIMPLE
    + ALLOWED_LIBS_SIMPLE
    + "\nGenerate the complete implementation using only VAS-approved APIs.\n"
)

# =============================================================================
# STRATEGY 2: MACD + ATR Trailing Stop (Medium - from_order_func flexible=False)
# =============================================================================

STRATEGY_2_MEDIUM = (
    STRATEGY_BASE_MEDIUM
    + VAS_DESCRIPTION
    + API_CITATION_MEDIUM
    + VALIDATION_VAS
    + OUTPUT_MEDIUM
    + ALLOWED_LIBS_MEDIUM
    + "\nGenerate the complete implementation using only VAS-approved APIs.\n"
)

# =============================================================================
# STRATEGY 3: Pairs Trading (Complex - from_order_func flexible=True)
# =============================================================================

STRATEGY_3_COMPLEX = (
    STRATEGY_BASE_COMPLEX
    + VAS_DESCRIPTION
    + API_CITATION_COMPLEX
    + VALIDATION_VAS
    + OUTPUT_COMPLEX
    + ALLOWED_LIBS_COMPLEX
    + "\nGenerate the complete implementation using only VAS-approved APIs.\n"
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
    "id": "C2",
    "name": "Documentation-Grounded",
    "schema_enabled": False,
    "doc_grounding_enabled": True,
    "tdd_enabled": False,
}
