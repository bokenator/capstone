"""
C7: All Treatments Condition Verification
=========================================
S=on, D=on, T=on

- verify_s: Validates schema compliance
- verify_d: Validates VAS compliance
- verify_t: Validates test compliance
"""

from typing import Any, Callable, Optional

from common import VerificationResult
from .shared import (
    # Schema validation (S=on)
    verify_schema_simple,
    verify_schema_medium,
    verify_schema_complex,
    DEFAULT_SCHEMAS,
    # VAS validation (D=on)
    verify_vas_simple,
    verify_vas_medium,
    verify_vas_complex,
    # Test validation (T=on)
    verify_tests_simple,
    verify_tests_medium,
    verify_tests_complex,
)


# =============================================================================
# SCHEMA VERIFICATION (S=on)
# =============================================================================

verify_s_simple = verify_schema_simple
verify_s_medium = verify_schema_medium
verify_s_complex = verify_schema_complex


# =============================================================================
# DOCUMENTATION/VAS VERIFICATION (D=on)
# =============================================================================

verify_d_simple = verify_vas_simple
verify_d_medium = verify_vas_medium
verify_d_complex = verify_vas_complex


# =============================================================================
# TEST-DRIVEN VERIFICATION (T=on)
# =============================================================================

verify_t_simple = verify_tests_simple
verify_t_medium = verify_tests_medium
verify_t_complex = verify_tests_complex


# =============================================================================
# COMBINED VERIFICATION
# =============================================================================

def verify_all_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
    vas: Optional[set[str]] = None,
    test_data: Optional[Any] = None,
    invariant_tests: Optional[list] = None,
    strategy_tests: Optional[list[Callable]] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for simple strategy."""
    schemas = DEFAULT_SCHEMAS["simple"]
    return {
        "schema": verify_s_simple(
            generated_module, source_code,
            data_schema or schemas["data_schema"],
            param_schema or schemas["param_schema"]
        ),
        "documentation": verify_d_simple(generated_module, source_code, vas),
        "tests": verify_t_simple(generated_module, source_code, test_data, invariant_tests, strategy_tests),
    }


def verify_all_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
    vas: Optional[set[str]] = None,
    test_data: Optional[Any] = None,
    invariant_tests: Optional[list] = None,
    strategy_tests: Optional[list[Callable]] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for medium strategy."""
    schemas = DEFAULT_SCHEMAS["medium"]
    return {
        "schema": verify_s_medium(
            generated_module, source_code,
            data_schema or schemas["data_schema"],
            param_schema or schemas["param_schema"]
        ),
        "documentation": verify_d_medium(generated_module, source_code, vas),
        "tests": verify_t_medium(generated_module, source_code, test_data, invariant_tests, strategy_tests),
    }


def verify_all_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
    vas: Optional[set[str]] = None,
    test_data: Optional[tuple[Any, Any]] = None,
    invariant_tests: Optional[list] = None,
    strategy_tests: Optional[list[Callable]] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all verifications for complex strategy."""
    schemas = DEFAULT_SCHEMAS["complex"]
    return {
        "schema": verify_s_complex(
            generated_module, source_code,
            data_schema or schemas["data_schema"],
            param_schema or schemas["param_schema"]
        ),
        "documentation": verify_d_complex(generated_module, source_code, vas),
        "tests": verify_t_complex(generated_module, source_code, test_data, invariant_tests, strategy_tests),
    }


# =============================================================================
# CONDITION METADATA
# =============================================================================

CONDITION_ID = "C7"
CONDITION_NAME = "All Treatments"

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
    test_data: Optional[Any] = None,
    **kwargs,
) -> dict[str, VerificationResult]:
    """Run all C7 verifications for a given strategy."""
    if strategy not in VERIFY_FUNCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")

    return VERIFY_FUNCTIONS[strategy]["all"](generated_module, source_code, test_data=test_data)
