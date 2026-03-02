"""
Shared Verification Components
==============================

Common verification logic used across experimental conditions.
"""

from .schema import (
    # AST helpers
    ColumnAccessVisitor,
    FunctionSignatureVisitor,
    extract_column_accesses,
    extract_function_signatures,
    extract_string_literals,
    # Validation logic
    validate_column_access,
    validate_function_signature,
    # Verification functions
    verify_schema_simple,
    verify_schema_medium,
    verify_schema_complex,
    VERIFY_SCHEMA_FUNCTIONS,
    DEFAULT_SCHEMAS,
)

from .vas import (
    # Verification functions (passthrough for RAG-based grounding)
    verify_vas_simple,
    verify_vas_medium,
    verify_vas_complex,
    VERIFY_VAS_FUNCTIONS,
)

from .tests import (
    # Invariant tests
    test_output_shape,
    test_no_lookahead,
    test_deterministic,
    test_no_nan_after_warmup,
    DEFAULT_INVARIANT_TESTS,
    # Test runners
    run_invariant_tests,
    run_strategy_tests,
    # Verification functions
    verify_tests_simple,
    verify_tests_medium,
    verify_tests_complex,
    VERIFY_TESTS_FUNCTIONS,
)

from .passthrough import (
    passthrough_schema,
    passthrough_vas,
    passthrough_tests,
)

__all__ = [
    # Schema validation
    "ColumnAccessVisitor",
    "FunctionSignatureVisitor",
    "extract_column_accesses",
    "extract_function_signatures",
    "extract_string_literals",
    "validate_column_access",
    "validate_function_signature",
    "verify_schema_simple",
    "verify_schema_medium",
    "verify_schema_complex",
    "VERIFY_SCHEMA_FUNCTIONS",
    "DEFAULT_SCHEMAS",
    # VAS validation (passthrough for RAG)
    "verify_vas_simple",
    "verify_vas_medium",
    "verify_vas_complex",
    "VERIFY_VAS_FUNCTIONS",
    # Test validation
    "test_output_shape",
    "test_no_lookahead",
    "test_deterministic",
    "test_no_nan_after_warmup",
    "DEFAULT_INVARIANT_TESTS",
    "run_invariant_tests",
    "run_strategy_tests",
    "verify_tests_simple",
    "verify_tests_medium",
    "verify_tests_complex",
    "VERIFY_TESTS_FUNCTIONS",
    # Passthrough
    "passthrough_schema",
    "passthrough_vas",
    "passthrough_tests",
]
