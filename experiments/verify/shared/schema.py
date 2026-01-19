"""
Schema Validation Logic
=======================

Validates schema compliance (DATA_SCHEMA and PARAM_SCHEMA).
Used by conditions with S=on: C1, C4, C5, C7
"""

import ast
import re
import traceback
from typing import Any, Optional

from common import VerificationResult
from prompts.shared import (
    DATA_SCHEMA_SIMPLE,
    DATA_SCHEMA_MEDIUM,
    DATA_SCHEMA_COMPLEX,
    PARAM_SCHEMA_SIMPLE,
    PARAM_SCHEMA_MEDIUM,
    PARAM_SCHEMA_COMPLEX,
)


# =============================================================================
# AST ANALYSIS HELPERS
# =============================================================================

class ColumnAccessVisitor(ast.NodeVisitor):
    """AST visitor to extract column access patterns from code."""

    def __init__(self):
        self.accessed_columns: set[str] = set()
        self.subscript_accesses: list[str] = []

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Capture df['column'] or df["column"] patterns."""
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            self.accessed_columns.add(node.slice.value)
            self.subscript_accesses.append(node.slice.value)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Capture df.column patterns (less reliable, may have false positives)."""
        # Only capture if it looks like a column access on a dataframe-like object
        if isinstance(node.value, ast.Name):
            if node.value.id in ('df', 'data', 'ohlcv', 'prices', 'asset_a', 'asset_b'):
                self.accessed_columns.add(node.attr)
        self.generic_visit(node)


class FunctionSignatureVisitor(ast.NodeVisitor):
    """AST visitor to extract function signatures."""

    def __init__(self):
        self.functions: dict[str, dict] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function name and parameters."""
        params = []
        defaults_offset = len(node.args.args) - len(node.args.defaults)

        for i, arg in enumerate(node.args.args):
            param_info = {"name": arg.arg, "has_default": i >= defaults_offset}
            if arg.annotation:
                param_info["annotation"] = ast.unparse(arg.annotation)
            params.append(param_info)

        self.functions[node.name] = {
            "params": params,
            "has_return_annotation": node.returns is not None,
        }
        self.generic_visit(node)


def extract_column_accesses(source_code: str) -> tuple[set[str], Optional[str]]:
    """Extract all column names accessed in the source code.

    Returns:
        Tuple of (accessed_columns, error_message). error_message is None on success.
    """
    try:
        tree = ast.parse(source_code)
        visitor = ColumnAccessVisitor()
        visitor.visit(tree)
        return visitor.accessed_columns, None
    except SyntaxError as e:
        return set(), f"Syntax error parsing code for column access: {e}\n{traceback.format_exc()}"


def extract_function_signatures(source_code: str) -> tuple[dict[str, dict], Optional[str]]:
    """Extract function signatures from source code.

    Returns:
        Tuple of (functions_dict, error_message). error_message is None on success.
    """
    try:
        tree = ast.parse(source_code)
        visitor = FunctionSignatureVisitor()
        visitor.visit(tree)
        return visitor.functions, None
    except SyntaxError as e:
        return {}, f"Syntax error parsing code for function signatures: {e}\n{traceback.format_exc()}"


def extract_string_literals(source_code: str) -> set[str]:
    """Extract all string literals that might be column names."""
    pattern = r'["\']([a-z_][a-z0-9_]*)["\']'
    matches = re.findall(pattern, source_code, re.IGNORECASE)
    return set(matches)


# =============================================================================
# SCHEMA VALIDATION LOGIC
# =============================================================================

def validate_column_access(
    source_code: str,
    data_schema: dict,
    param_schema: dict,
    result: VerificationResult,
) -> None:
    """Validate that code only accesses columns declared in schema."""
    # Get all declared slots and columns from schema
    declared_slots: set[str] = set()
    declared_columns: set[str] = set()
    required_columns: set[str] = set()

    for slot_name, slot_spec in data_schema.items():
        declared_slots.add(slot_name)
        for col_name, col_spec in slot_spec.get("columns", {}).items():
            declared_columns.add(col_name)
            if col_spec.get("required", False):
                required_columns.add(col_name)

    # Get all declared parameter names from param_schema
    declared_params: set[str] = set(param_schema.keys())

    # Extract accessed columns from code
    accessed_columns, error = extract_column_accesses(source_code)
    if error:
        result.add_error(error)
        return

    # Also check string literals for potential column access
    string_literals = extract_string_literals(source_code)

    # Common OHLCV column names to check
    ohlcv_columns = {"open", "high", "low", "close", "volume"}
    potential_column_accesses = accessed_columns | (string_literals & ohlcv_columns)

    # Allowed names: declared slots + declared columns + declared params + common false positives
    allowed_names = declared_slots | declared_columns | declared_params
    # DataFrame/Series attributes and common variable names that aren't column accesses
    false_positives = {
        "index", "values", "name", "dtype", "shape", "size",  # pandas attributes
        "columns", "loc", "iloc", "at", "iat",  # DataFrame attributes
        "data", "params", "df", "result", "output",  # common variable names
    }
    allowed_names = allowed_names | false_positives

    # Check for undeclared column access
    undeclared = potential_column_accesses - allowed_names

    for col in undeclared:
        result.add_error(f"Undeclared column access: '{col}' not in DATA_SCHEMA")

    # Check that required columns are accessed
    for col in required_columns:
        if col not in potential_column_accesses:
            result.add_warning(f"Required column '{col}' may not be accessed in code")

    result.details["declared_slots"] = list(declared_slots)
    result.details["declared_columns"] = list(declared_columns)
    result.details["accessed_columns"] = list(potential_column_accesses)


def validate_function_signature(
    source_code: str,
    expected_functions: list[str],
    param_schema: dict,
    result: VerificationResult,
    expected_params: Optional[list[str]] = None,
) -> None:
    """Validate that expected functions exist with correct parameters."""
    signatures, error = extract_function_signatures(source_code)
    if error:
        result.add_error(error)
        return

    for func_name in expected_functions:
        if func_name not in signatures:
            result.add_error(f"Missing required function: '{func_name}'")
            continue

        func_info = signatures[func_name]
        actual_params = [p["name"] for p in func_info["params"]]
        result.details[f"{func_name}_params"] = actual_params

        # Check expected parameters if provided
        if expected_params:
            for expected in expected_params:
                if expected not in actual_params:
                    result.add_error(
                        f"Function '{func_name}' missing required parameter: '{expected}'. "
                        f"Expected: {expected_params}, got: {actual_params}"
                    )

    # Check that param_schema parameters appear somewhere in the code
    # (either as function params or accessed from params dict)
    schema_params = set(param_schema.keys())
    for param in schema_params:
        # Check if param is accessed via params dict (e.g., params['rsi_period'] or params.get('rsi_period'))
        param_access_patterns = [
            f"params['{param}']",
            f'params["{param}"]',
            f"params.get('{param}'",
            f'params.get("{param}"',
        ]
        found = any(pattern in source_code for pattern in param_access_patterns)
        if not found:
            result.add_warning(f"Parameter '{param}' from PARAM_SCHEMA may not be accessed in code")


# =============================================================================
# SCHEMA VERIFICATION FUNCTIONS (S)
# =============================================================================

def verify_schema_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
) -> VerificationResult:
    """
    Verify schema compliance for simple strategy (from_signals).

    Expected interface:
        def generate_signals(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]

    Args:
        generated_module: The imported module containing generate_signals
        source_code: Source code string (required for AST analysis)
        data_schema: Expected DATA_SCHEMA (uses default if None)
        param_schema: Expected PARAM_SCHEMA (uses default if None)

    Returns:
        VerificationResult with pass/fail status and details
    """
    result = VerificationResult(passed=True)

    # Use default schemas if not provided
    data_schema = data_schema or DATA_SCHEMA_SIMPLE
    param_schema = param_schema or PARAM_SCHEMA_SIMPLE

    # Need source code for AST analysis
    if source_code is None:
        result.add_error("Source code required for schema validation")
        return result

    # Check that generate_signals function exists
    if not hasattr(generated_module, "generate_signals"):
        result.add_error("Missing required function: 'generate_signals'")
        return result

    # Validate column access
    validate_column_access(source_code, data_schema, param_schema, result)

    # Validate function signature - must have (data, params) parameters
    validate_function_signature(
        source_code,
        expected_functions=["generate_signals"],
        param_schema=param_schema,
        result=result,
        expected_params=["data", "params"],
    )

    return result


def verify_schema_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
) -> VerificationResult:
    """
    Verify schema compliance for medium strategy (from_order_func flexible=False).

    Args:
        generated_module: The imported module containing compute_indicators and order_func
        source_code: Source code string (required for AST analysis)
        data_schema: Expected DATA_SCHEMA (uses default if None)
        param_schema: Expected PARAM_SCHEMA (uses default if None)

    Returns:
        VerificationResult with pass/fail status and details
    """
    result = VerificationResult(passed=True)

    data_schema = data_schema or DATA_SCHEMA_MEDIUM
    param_schema = param_schema or PARAM_SCHEMA_MEDIUM

    if source_code is None:
        result.add_error("Source code required for schema validation")
        return result

    # Check required functions exist
    required_funcs = ["compute_indicators", "order_func"]
    for func_name in required_funcs:
        if not hasattr(generated_module, func_name):
            result.add_error(f"Missing required function: '{func_name}'")

    if not result.passed:
        return result

    # Validate column access
    validate_column_access(source_code, data_schema, param_schema, result)

    # Validate function signatures
    validate_function_signature(
        source_code,
        expected_functions=required_funcs,
        param_schema=param_schema,
        result=result,
    )

    return result


def verify_schema_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    data_schema: Optional[dict] = None,
    param_schema: Optional[dict] = None,
) -> VerificationResult:
    """
    Verify schema compliance for complex strategy (from_order_func flexible=True).

    Args:
        generated_module: The imported module containing compute_spread_indicators, order_func
        source_code: Source code string (required for AST analysis)
        data_schema: Expected DATA_SCHEMA (uses default if None)
        param_schema: Expected PARAM_SCHEMA (uses default if None)

    Returns:
        VerificationResult with pass/fail status and details
    """
    result = VerificationResult(passed=True)

    data_schema = data_schema or DATA_SCHEMA_COMPLEX
    param_schema = param_schema or PARAM_SCHEMA_COMPLEX

    if source_code is None:
        result.add_error("Source code required for schema validation")
        return result

    # Check required functions exist
    # Complex strategies use compute_spread_indicators (not compute_indicators)
    required_funcs = ["compute_spread_indicators", "order_func"]
    for func_name in required_funcs:
        if not hasattr(generated_module, func_name):
            result.add_error(f"Missing required function: '{func_name}'")

    if not result.passed:
        return result

    # Validate column access
    validate_column_access(source_code, data_schema, param_schema, result)

    # Validate function signatures
    validate_function_signature(
        source_code,
        expected_functions=required_funcs,
        param_schema=param_schema,
        result=result,
    )

    return result


# =============================================================================
# DEFAULT SCHEMAS
# =============================================================================

DEFAULT_SCHEMAS = {
    "simple": {
        "data_schema": DATA_SCHEMA_SIMPLE,
        "param_schema": PARAM_SCHEMA_SIMPLE,
    },
    "medium": {
        "data_schema": DATA_SCHEMA_MEDIUM,
        "param_schema": PARAM_SCHEMA_MEDIUM,
    },
    "complex": {
        "data_schema": DATA_SCHEMA_COMPLEX,
        "param_schema": PARAM_SCHEMA_COMPLEX,
    },
}

# Strategy to verification function mapping
VERIFY_SCHEMA_FUNCTIONS = {
    "simple": verify_schema_simple,
    "medium": verify_schema_medium,
    "complex": verify_schema_complex,
}
