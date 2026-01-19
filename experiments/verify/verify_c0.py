"""
C0: Control Condition Verification
==================================
S=off, D=off, T=off

For C0 (control), all verifications pass since there are no constraints.
"""

from typing import Any, Callable, Optional

from common import VerificationResult
from .shared import (
    passthrough_schema,
    passthrough_vas,
    passthrough_tests,
)


# =============================================================================
# SCHEMA VERIFICATION (S=off)
# =============================================================================

def verify_s_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
) -> VerificationResult:
    """Schema verification for simple strategy. Always passes for C0."""
    return passthrough_schema(
        generated_module, source_code, data_schema, param_schema,
        reason="C0 control: no schema constraints (S=off)"
    )


def verify_s_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
) -> VerificationResult:
    """Schema verification for medium strategy. Always passes for C0."""
    return passthrough_schema(
        generated_module, source_code, data_schema, param_schema,
        reason="C0 control: no schema constraints (S=off)"
    )


def verify_s_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
) -> VerificationResult:
    """Schema verification for complex strategy. Always passes for C0."""
    return passthrough_schema(
        generated_module, source_code, data_schema, param_schema,
        reason="C0 control: no schema constraints (S=off)"
    )


# =============================================================================
# DOCUMENTATION/VAS VERIFICATION (D=off)
# =============================================================================

def verify_d_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """VAS verification for simple strategy. Always passes for C0."""
    return passthrough_vas(
        generated_module, source_code, vas,
        reason="C0 control: no VAS constraints (D=off)"
    )


def verify_d_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """VAS verification for medium strategy. Always passes for C0."""
    return passthrough_vas(
        generated_module, source_code, vas,
        reason="C0 control: no VAS constraints (D=off)"
    )


def verify_d_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """VAS verification for complex strategy. Always passes for C0."""
    return passthrough_vas(
        generated_module, source_code, vas,
        reason="C0 control: no VAS constraints (D=off)"
    )


# =============================================================================
# TEST-DRIVEN VERIFICATION (T=off)
# =============================================================================

def verify_t_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[Any] = None,
    invariant_tests: Optional[list[Callable]] = None,
    strategy_tests: Optional[list[Callable]] = None,
) -> VerificationResult:
    """Test verification for simple strategy. Always passes for C0."""
    return passthrough_tests(
        generated_module, source_code, test_data, invariant_tests, strategy_tests,
        reason="C0 control: no test constraints (T=off)"
    )


def verify_t_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[Any] = None,
    invariant_tests: Optional[list[Callable]] = None,
    strategy_tests: Optional[list[Callable]] = None,
) -> VerificationResult:
    """Test verification for medium strategy. Always passes for C0."""
    return passthrough_tests(
        generated_module, source_code, test_data, invariant_tests, strategy_tests,
        reason="C0 control: no test constraints (T=off)"
    )


def verify_t_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[tuple[Any, Any]] = None,
    invariant_tests: Optional[list[Callable]] = None,
    strategy_tests: Optional[list[Callable]] = None,
) -> VerificationResult:
    """Test verification for complex strategy. Always passes for C0."""
    return passthrough_tests(
        generated_module, source_code, test_data, invariant_tests, strategy_tests,
        reason="C0 control: no test constraints (T=off)"
    )


# =============================================================================
# COMBINED VERIFICATION
# =============================================================================

def verify_all_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for simple strategy."""
    return {
        "schema": verify_s_simple(generated_module, source_code),
        "documentation": verify_d_simple(generated_module, source_code),
        "tests": verify_t_simple(generated_module, source_code),
    }


def verify_all_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for medium strategy."""
    return {
        "schema": verify_s_medium(generated_module, source_code),
        "documentation": verify_d_medium(generated_module, source_code),
        "tests": verify_t_medium(generated_module, source_code),
    }


def verify_all_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for complex strategy."""
    return {
        "schema": verify_s_complex(generated_module, source_code),
        "documentation": verify_d_complex(generated_module, source_code),
        "tests": verify_t_complex(generated_module, source_code),
    }


# =============================================================================
# CONDITION METADATA
# =============================================================================

CONDITION_ID = "C0"
CONDITION_NAME = "Control"

VERIFY_FUNCTIONS = {
    "simple": {
        "schema": verify_s_simple,
        "documentation": verify_d_simple,
        "tests": verify_t_simple,
        "all": verify_all_simple,
    },
    "medium": {
        "schema": verify_s_medium,
        "documentation": verify_d_medium,
        "tests": verify_t_medium,
        "all": verify_all_medium,
    },
    "complex": {
        "schema": verify_s_complex,
        "documentation": verify_d_complex,
        "tests": verify_t_complex,
        "all": verify_all_complex,
    },
}


def run_verification(
    strategy: str,
    generated_module: Any,
    source_code: Optional[str] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all C0 verifications for a given strategy. Extra kwargs are ignored."""
    if strategy not in VERIFY_FUNCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")

    return VERIFY_FUNCTIONS[strategy]["all"](generated_module, source_code)
