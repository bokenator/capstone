"""
C6: Documentation + TDD Condition Prompts
=========================================
S=off, D=on, T=on

Combines:
- RAG-based documentation grounding via OpenSearch
- Invariant tests (properties that must always hold)
- Strategy-specific tests
"""

from .shared import (
    STRATEGY_BASE_SIMPLE,
    STRATEGY_BASE_MEDIUM,
    STRATEGY_BASE_COMPLEX,
    RAG_DESCRIPTION,
    API_CITATION_SIMPLE,
    API_CITATION_MEDIUM,
    API_CITATION_COMPLEX,
    INVARIANT_TESTS,
    STRATEGY_TESTS_SIMPLE,
    STRATEGY_TESTS_MEDIUM,
    STRATEGY_TESTS_COMPLEX,
    OUTPUT_SIMPLE,
    OUTPUT_MEDIUM,
    OUTPUT_COMPLEX,
    ALLOWED_LIBS_SIMPLE,
    ALLOWED_LIBS_MEDIUM,
    ALLOWED_LIBS_COMPLEX,
    VALIDATION_VAS_TESTS,
)

# =============================================================================
# STRATEGY 1: RSI Mean Reversion (Simple - from_signals)
# =============================================================================

STRATEGY_1_SIMPLE = (
    STRATEGY_BASE_SIMPLE
    + RAG_DESCRIPTION
    + API_CITATION_SIMPLE
    + INVARIANT_TESTS
    + STRATEGY_TESTS_SIMPLE
    + VALIDATION_VAS_TESTS
    + OUTPUT_SIMPLE
    + ALLOWED_LIBS_SIMPLE
    + "\nGenerate the complete implementation that passes all tests. Search docs to verify each API.\n"
)

# =============================================================================
# STRATEGY 2: MACD + ATR Trailing Stop (Medium - from_order_func flexible=False)
# =============================================================================

STRATEGY_2_MEDIUM = (
    STRATEGY_BASE_MEDIUM
    + RAG_DESCRIPTION
    + API_CITATION_MEDIUM
    + INVARIANT_TESTS
    + STRATEGY_TESTS_MEDIUM
    + VALIDATION_VAS_TESTS
    + OUTPUT_MEDIUM
    + ALLOWED_LIBS_MEDIUM
    + "\nGenerate the complete implementation that passes all tests. Search docs to verify each API.\n"
)

# =============================================================================
# STRATEGY 3: Pairs Trading (Complex - from_order_func flexible=True)
# =============================================================================

STRATEGY_3_COMPLEX = (
    STRATEGY_BASE_COMPLEX
    + RAG_DESCRIPTION
    + API_CITATION_COMPLEX
    + INVARIANT_TESTS
    + STRATEGY_TESTS_COMPLEX
    + VALIDATION_VAS_TESTS
    + OUTPUT_COMPLEX
    + ALLOWED_LIBS_COMPLEX
    + "\nGenerate the complete implementation that passes all tests. Search docs to verify each API.\n"
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
    "id": "C6",
    "name": "Documentation + TDD",
    "schema_enabled": False,
    "doc_grounding_enabled": True,
    "tdd_enabled": True,
}
