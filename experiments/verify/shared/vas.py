"""
VAS (Verified API Surface) Validation Logic
============================================

Validates documentation/API compliance.
Used by conditions with D=on: C2, C4, C6, C7
"""

import ast
import traceback
from typing import Any, Optional

from common import VerificationResult
from prompts.shared import VAS_SET


# =============================================================================
# AST ANALYSIS FOR API CALLS
# =============================================================================

class APICallVisitor(ast.NodeVisitor):
    """AST visitor to extract API calls from code."""

    def __init__(self):
        self.api_calls: set[str] = set()
        self.unqualified_calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Extract function/method calls."""
        call_name = self._get_call_name(node.func)
        if call_name:
            self.api_calls.add(call_name)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Track attribute access for API verification."""
        full_name = self._get_attribute_chain(node)
        if full_name:
            self.api_calls.add(full_name)
        self.generic_visit(node)

    def _get_call_name(self, node: ast.expr) -> Optional[str]:
        """Get the full name of a call expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_chain(node)
        return None

    def _get_attribute_chain(self, node: ast.Attribute) -> Optional[str]:
        """Get the full attribute chain (e.g., pd.Series.rolling)."""
        parts = []
        current = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))

        return None


def extract_api_calls(source_code: str) -> tuple[set[str], Optional[str]]:
    """Extract all API calls from source code.

    Returns:
        Tuple of (api_calls, error_message). error_message is None on success.
    """
    try:
        tree = ast.parse(source_code)
        visitor = APICallVisitor()
        visitor.visit(tree)
        return visitor.api_calls, None
    except SyntaxError as e:
        return set(), f"Syntax error parsing code for API calls: {e}\n{traceback.format_exc()}"


# =============================================================================
# VAS VALIDATION LOGIC
# =============================================================================

def validate_api_calls(
    source_code: str,
    vas: set[str],
    result: VerificationResult,
) -> None:
    """Validate that code only uses APIs from the Verified API Surface."""
    api_calls, error = extract_api_calls(source_code)
    if error:
        result.add_error(error)
        return

    # Known library prefixes that need VAS validation
    library_prefixes = {"pd", "np", "vbt", "ta", "scipy", "numba"}

    # Check each API call
    invalid_apis: list[str] = []
    valid_apis: list[str] = []

    for api_call in api_calls:
        # Only validate calls that start with known library prefixes
        prefix = api_call.split(".")[0] if "." in api_call else api_call

        if prefix in library_prefixes:
            # Check if this API or a parent is in VAS
            is_valid = False
            for vas_api in vas:
                if api_call.startswith(vas_api) or vas_api.startswith(api_call):
                    is_valid = True
                    break

            if is_valid:
                valid_apis.append(api_call)
            else:
                invalid_apis.append(api_call)

    for api in invalid_apis:
        result.add_error(f"API not in VAS: '{api}'")

    result.details["valid_apis"] = valid_apis
    result.details["invalid_apis"] = invalid_apis


# =============================================================================
# VAS VERIFICATION FUNCTIONS (D)
# =============================================================================

def verify_vas_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """
    Verify VAS compliance for simple strategy.

    Args:
        generated_module: The imported module
        source_code: Source code string (required for AST analysis)
        vas: Verified API Surface (uses default if None)

    Returns:
        VerificationResult with pass/fail status and details
    """
    _ = generated_module  # Not needed for VAS validation

    result = VerificationResult(passed=True)

    vas = vas or VAS_SET

    if source_code is None:
        result.add_error("Source code required for VAS validation")
        return result

    validate_api_calls(source_code, vas, result)

    return result


def verify_vas_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """
    Verify VAS compliance for medium strategy.

    Args:
        generated_module: The imported module
        source_code: Source code string (required for AST analysis)
        vas: Verified API Surface (uses default if None)

    Returns:
        VerificationResult with pass/fail status and details
    """
    _ = generated_module

    result = VerificationResult(passed=True)

    vas = vas or VAS_SET

    if source_code is None:
        result.add_error("Source code required for VAS validation")
        return result

    validate_api_calls(source_code, vas, result)

    return result


def verify_vas_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    vas: Optional[set[str]] = None,
) -> VerificationResult:
    """
    Verify VAS compliance for complex strategy.

    Args:
        generated_module: The imported module
        source_code: Source code string (required for AST analysis)
        vas: Verified API Surface (uses default if None)

    Returns:
        VerificationResult with pass/fail status and details
    """
    _ = generated_module

    result = VerificationResult(passed=True)

    vas = vas or VAS_SET

    if source_code is None:
        result.add_error("Source code required for VAS validation")
        return result

    validate_api_calls(source_code, vas, result)

    return result


# Strategy to verification function mapping
VERIFY_VAS_FUNCTIONS = {
    "simple": verify_vas_simple,
    "medium": verify_vas_medium,
    "complex": verify_vas_complex,
}
