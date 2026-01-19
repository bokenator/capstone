"""
C2: Documentation-Grounded Condition Verification
==================================================
S=off, D=on, T=off

- verify_s: Always passes (no schema constraints)
- verify_d: Validates VAS compliance
- verify_t: Always passes (no test constraints)
"""

from typing import Any, Callable, Optional

from common import VerificationResult
from .shared import (
    # VAS validation (D=on)
    verify_vas_simple,
    verify_vas_medium,
    verify_vas_complex,
    # Passthrough for disabled constraints
    passthrough_schema,
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
    """Schema verification for simple strategy. Always passes for C2."""
    return passthrough_schema(
        generated_module, source_code, data_schema, param_schema,
        reason="C2: no schema constraints (S=off)"
    )


def verify_s_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
) -> VerificationResult:
    """Schema verification for medium strategy. Always passes for C2."""
    return passthrough_schema(
        generated_module, source_code, data_schema, param_schema,
        reason="C2: no schema constraints (S=off)"
    )


def verify_s_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
) -> VerificationResult:
    """Schema verification for complex strategy. Always passes for C2."""
    return passthrough_schema(
        generated_module, source_code, data_schema, param_schema,
        reason="C2: no schema constraints (S=off)"
    )


# =============================================================================
# DOCUMENTATION/VAS VERIFICATION (D=on)
# =============================================================================

verify_d_simple = verify_vas_simple
verify_d_medium = verify_vas_medium
verify_d_complex = verify_vas_complex


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
    """Test verification for simple strategy. Always passes for C2."""
    return passthrough_tests(
        generated_module, source_code, test_data, invariant_tests, strategy_tests,
        reason="C2: no test constraints (T=off)"
    )


def verify_t_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[Any] = None,
    invariant_tests: Optional[list[Callable]] = None,
    strategy_tests: Optional[list[Callable]] = None,
) -> VerificationResult:
    """Test verification for medium strategy. Always passes for C2."""
    return passthrough_tests(
        generated_module, source_code, test_data, invariant_tests, strategy_tests,
        reason="C2: no test constraints (T=off)"
    )


def verify_t_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[tuple[Any, Any]] = None,
    invariant_tests: Optional[list[Callable]] = None,
    strategy_tests: Optional[list[Callable]] = None,
) -> VerificationResult:
    """Test verification for complex strategy. Always passes for C2."""
    return passthrough_tests(
        generated_module, source_code, test_data, invariant_tests, strategy_tests,
        reason="C2: no test constraints (T=off)"
    )


# =============================================================================
# COMBINED VERIFICATION
# =============================================================================

def verify_all_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for simple strategy."""
    return {
        "schema": verify_s_simple(generated_module, source_code),
        "documentation": verify_d_simple(generated_module, source_code, vas),
        "tests": verify_t_simple(generated_module, source_code),
    }


def verify_all_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for medium strategy."""
    return {
        "schema": verify_s_medium(generated_module, source_code),
        "documentation": verify_d_medium(generated_module, source_code, vas),
        "tests": verify_t_medium(generated_module, source_code),
    }


def verify_all_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for complex strategy."""
    return {
        "schema": verify_s_complex(generated_module, source_code),
        "documentation": verify_d_complex(generated_module, source_code, vas),
        "tests": verify_t_complex(generated_module, source_code),
    }


# =============================================================================
# CONDITION METADATA
# =============================================================================

CONDITION_ID = "C2"
CONDITION_NAME = "Documentation-Grounded"

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
    """Run all C2 verifications for a given strategy. Extra kwargs are ignored."""
    if strategy not in VERIFY_FUNCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")

    return VERIFY_FUNCTIONS[strategy]["all"](generated_module, source_code)
