"""
Passthrough Verification Functions
==================================

Functions that always pass - used when a constraint is disabled.
For example, C0 (control) has all constraints off, so all verifications pass.
"""

from typing import Any, Callable, Optional

from common import VerificationResult


# =============================================================================
# SCHEMA PASSTHROUGH (S=off)
# =============================================================================

def passthrough_schema(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
    reason: str = "Schema validation disabled (S=off)",
) -> VerificationResult:
    """Schema verification that always passes."""
    _ = generated_module, source_code, data_schema, param_schema
    result = VerificationResult(passed=True)
    result.details["reason"] = reason
    return result


# =============================================================================
# VAS PASSTHROUGH (D=off)
# =============================================================================

def passthrough_vas(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
    reason: str = "VAS validation disabled (D=off)",
) -> VerificationResult:
    """VAS verification that always passes."""
    _ = generated_module, source_code, vas
    result = VerificationResult(passed=True)
    result.details["reason"] = reason
    return result


# =============================================================================
# TEST PASSTHROUGH (T=off)
# =============================================================================

def passthrough_tests(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[Any] = None,
    invariant_tests: Optional[list[Callable]] = None,
    strategy_tests: Optional[list[Callable]] = None,
    reason: str = "Test validation disabled (T=off)",
) -> VerificationResult:
    """Test verification that always passes."""
    _ = generated_module, source_code, test_data, invariant_tests, strategy_tests
    result = VerificationResult(passed=True)
    result.details["reason"] = reason
    return result
