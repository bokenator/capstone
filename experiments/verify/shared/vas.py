"""
Documentation Grounding Verification (Passthrough)
===================================================

With RAG-based documentation grounding, verification is handled at the
agent level (the agent searches docs before using APIs). There is no
post-hoc AST enforcement.

All verify_vas_* functions return passed=True as passthroughs.

Used by conditions with D=on: C2, C4, C6, C7
"""

from typing import Any, Optional

from common import VerificationResult


# =============================================================================
# PASSTHROUGH VERIFICATION FUNCTIONS (D)
# =============================================================================


def verify_vas_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """
    Passthrough verification for simple strategy (RAG-based grounding).

    Args:
        generated_module: The imported module (unused)
        source_code: Source code string (unused)
        vas: Unused (kept for signature compatibility)

    Returns:
        VerificationResult with passed=True
    """
    result = VerificationResult(passed=True)
    result.details["reason"] = "RAG-based documentation grounding (no enforcement)"
    return result


def verify_vas_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """
    Passthrough verification for medium strategy (RAG-based grounding).

    Args:
        generated_module: The imported module (unused)
        source_code: Source code string (unused)
        vas: Unused (kept for signature compatibility)

    Returns:
        VerificationResult with passed=True
    """
    result = VerificationResult(passed=True)
    result.details["reason"] = "RAG-based documentation grounding (no enforcement)"
    return result


def verify_vas_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """
    Passthrough verification for complex strategy (RAG-based grounding).

    Args:
        generated_module: The imported module (unused)
        source_code: Source code string (unused)
        vas: Unused (kept for signature compatibility)

    Returns:
        VerificationResult with passed=True
    """
    result = VerificationResult(passed=True)
    result.details["reason"] = "RAG-based documentation grounding (no enforcement)"
    return result


# Strategy to verification function mapping
VERIFY_VAS_FUNCTIONS = {
    "simple": verify_vas_simple,
    "medium": verify_vas_medium,
    "complex": verify_vas_complex,
}
