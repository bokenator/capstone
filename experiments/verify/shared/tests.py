"""
Test Validation Logic
=====================

Validates TDD compliance (invariant tests and strategy-specific tests).
Used by conditions with T=on: C3, C5, C6, C7
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Optional, TYPE_CHECKING

try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY_PANDAS = True
except ImportError:
    HAS_NUMPY_PANDAS = False
    np = None  # type: ignore
    pd = None  # type: ignore

from common import VerificationResult


# =============================================================================
# INVARIANT TEST IMPLEMENTATIONS
# =============================================================================

def test_output_shape(
    generate_func: Callable,
    input_data: "pd.DataFrame",
) -> tuple[bool, str]:
    """Output must have same length as input."""
    try:
        output = generate_func(input_data)

        # Handle tuple output (entries, exits)
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if len(out) != len(input_data):
                    return False, f"Output[{i}] length {len(out)} != input length {len(input_data)}"
        else:
            if len(output) != len(input_data):
                return False, f"Output length {len(output)} != input length {len(input_data)}"

        return True, "Output shape matches input"
    except Exception as e:
        return False, f"Error testing output shape: {e}\n{traceback.format_exc()}"


def test_no_lookahead(
    generate_func: Callable,
    input_data: "pd.DataFrame",
    test_points: Optional[list[int]] = None,
) -> tuple[bool, str]:
    """Signal at time t must not depend on data after time t."""
    test_points = test_points or [50, 100, 150]

    try:
        full_output = generate_func(input_data)

        # Handle tuple output
        if isinstance(full_output, tuple):
            full_outputs = full_output
        else:
            full_outputs = (full_output,)

        for t in test_points:
            if t >= len(input_data):
                continue

            truncated_data = input_data.iloc[:t].copy()
            truncated_output = generate_func(truncated_data)

            if isinstance(truncated_output, tuple):
                truncated_outputs = truncated_output
            else:
                truncated_outputs = (truncated_output,)

            for i, (full, trunc) in enumerate(zip(full_outputs, truncated_outputs)):
                full_slice = full.iloc[:t]
                if not full_slice.equals(trunc):
                    # Check for NaN-aware comparison
                    if not np.allclose(
                        full_slice.fillna(-999).values,
                        trunc.fillna(-999).values,
                        equal_nan=True
                    ):
                        return False, f"Lookahead bias detected at t={t} in output[{i}]"

        return True, "No lookahead bias detected"
    except Exception as e:
        return False, f"Error testing lookahead: {e}\n{traceback.format_exc()}"


def test_deterministic(
    generate_func: Callable,
    input_data: "pd.DataFrame",
) -> tuple[bool, str]:
    """Same input must produce same output."""
    try:
        output1 = generate_func(input_data)
        output2 = generate_func(input_data)

        # Handle tuple output
        if isinstance(output1, tuple):
            for i, (o1, o2) in enumerate(zip(output1, output2)):
                if not o1.equals(o2):
                    return False, f"Non-deterministic behavior in output[{i}]"
        else:
            if not output1.equals(output2):
                return False, "Non-deterministic behavior"

        return True, "Function is deterministic"
    except Exception as e:
        return False, f"Error testing determinism: {e}\n{traceback.format_exc()}"


def test_no_nan_after_warmup(
    generate_func: Callable,
    input_data: "pd.DataFrame",
    warmup_period: int = 50,
) -> tuple[bool, str]:
    """No NaN values after warmup period."""
    try:
        output = generate_func(input_data)

        # Handle tuple output
        if isinstance(output, tuple):
            outputs = output
        else:
            outputs = (output,)

        for i, out in enumerate(outputs):
            after_warmup = out.iloc[warmup_period:]
            nan_count = after_warmup.isna().sum()
            if nan_count > 0:
                return False, f"Output[{i}] contains {nan_count} NaN values after warmup"

        return True, "No NaN after warmup"
    except Exception as e:
        return False, f"Error testing NaN: {e}\n{traceback.format_exc()}"


# Default invariant tests
DEFAULT_INVARIANT_TESTS = [
    ("output_shape", test_output_shape),
    ("no_lookahead", test_no_lookahead),
    ("deterministic", test_deterministic),
    ("no_nan_after_warmup", test_no_nan_after_warmup),
]


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_invariant_tests(
    generate_func: Callable,
    test_data: "pd.DataFrame",
    result: VerificationResult,
    invariant_tests: Optional[list[tuple[str, Callable]]] = None,
) -> None:
    """Run all invariant tests and record results."""
    invariant_tests = invariant_tests or DEFAULT_INVARIANT_TESTS

    test_results = {}
    for test_name, test_func in invariant_tests:
        passed, message = test_func(generate_func, test_data)
        test_results[test_name] = {"passed": passed, "message": message}

        if not passed:
            result.add_error(f"Invariant test '{test_name}' failed: {message}")

    result.details["invariant_tests"] = test_results


def run_strategy_tests(
    generated_module: Any,
    test_data: Any,
    result: VerificationResult,
    strategy_tests: Optional[list[Callable]] = None,
) -> None:
    """Run strategy-specific tests and record results."""
    if strategy_tests is None:
        result.details["strategy_tests"] = "No strategy tests provided"
        return

    test_results = {}
    for test_func in strategy_tests:
        test_name = test_func.__name__
        try:
            test_func(generated_module, test_data)
            test_results[test_name] = {"passed": True, "message": "Passed"}
        except AssertionError as e:
            test_results[test_name] = {"passed": False, "message": str(e)}
            result.add_error(f"Strategy test '{test_name}' failed: {e}")
        except Exception as e:
            test_results[test_name] = {"passed": False, "message": f"Error: {e}\n{traceback.format_exc()}"}
            result.add_error(f"Strategy test '{test_name}' error: {e}")

    result.details["strategy_tests"] = test_results


# =============================================================================
# TEST VERIFICATION FUNCTIONS (T)
# =============================================================================

def verify_tests_simple(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[pd.DataFrame] = None,
    invariant_tests: Optional[list[tuple[str, Callable]]] = None,
    strategy_tests: Optional[list[Callable]] = None,
) -> VerificationResult:
    """
    Verify TDD compliance for simple strategy.

    Args:
        generated_module: The imported module containing generate_signals
        source_code: Source code string (not used for test validation)
        test_data: Test data DataFrame to run tests against
        invariant_tests: List of (name, test_func) tuples for invariant tests
        strategy_tests: List of strategy-specific test functions

    Returns:
        VerificationResult with pass/fail status and details
    """
    _ = source_code  # Not needed for test validation

    result = VerificationResult(passed=True)

    if test_data is None:
        result.add_error("Test data required for TDD validation")
        return result

    if not hasattr(generated_module, "generate_signals"):
        result.add_error("Missing required function: 'generate_signals'")
        return result

    # Default params for simple strategy (RSI Mean Reversion)
    default_params = {
        "rsi_period": 14,
        "oversold": 30.0,
        "overbought": 70.0,
    }

    # Create wrapper that calls generate_signals with proper interface:
    # generate_signals(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]
    def generate_func_wrapper(df: pd.DataFrame) -> pd.Series:
        data = {"ohlcv": df}
        output = generated_module.generate_signals(data, default_params)
        # Output is dict[str, pd.Series] with "entries" and "exits" keys
        # Return entries for invariant tests (or combine as needed)
        if isinstance(output, dict):
            return output.get("entries", pd.Series(dtype=bool, index=df.index))
        return output

    # Run invariant tests
    run_invariant_tests(
        generate_func=generate_func_wrapper,
        test_data=test_data,
        result=result,
        invariant_tests=invariant_tests,
    )

    # Run strategy tests
    run_strategy_tests(
        generated_module=generated_module,
        test_data=test_data,
        result=result,
        strategy_tests=strategy_tests,
    )

    return result


def verify_tests_medium(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[pd.DataFrame] = None,
    invariant_tests: Optional[list[tuple[str, Callable]]] = None,
    strategy_tests: Optional[list[Callable]] = None,
) -> VerificationResult:
    """
    Verify TDD compliance for medium strategy.

    Args:
        generated_module: The imported module containing compute_indicators and order_func
        source_code: Source code string (not used)
        test_data: Test data DataFrame
        invariant_tests: List of invariant tests
        strategy_tests: List of strategy-specific tests

    Returns:
        VerificationResult with pass/fail status and details
    """
    _ = source_code

    result = VerificationResult(passed=True)

    if test_data is None:
        result.add_error("Test data required for TDD validation")
        return result

    # Check required functions
    required_funcs = ["compute_indicators", "order_func"]
    for func_name in required_funcs:
        if not hasattr(generated_module, func_name):
            result.add_error(f"Missing required function: '{func_name}'")

    if not result.passed:
        return result

    # For medium strategy, we test compute_indicators output
    def generate_func(data: "pd.DataFrame"):
        indicators = generated_module.compute_indicators(data)
        # Return a series for shape testing
        return pd.Series(indicators.get("close", data["close"].values), index=data.index)

    run_invariant_tests(
        generate_func=generate_func,
        test_data=test_data,
        result=result,
        invariant_tests=invariant_tests,
    )

    run_strategy_tests(
        generated_module=generated_module,
        test_data=test_data,
        result=result,
        strategy_tests=strategy_tests,
    )

    return result


def verify_tests_complex(
    generated_module: Any,
    source_code: Optional[str] = None,
    test_data: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
    invariant_tests: Optional[list[tuple[str, Callable]]] = None,
    strategy_tests: Optional[list[Callable]] = None,
) -> VerificationResult:
    """
    Verify TDD compliance for complex strategy.

    Args:
        generated_module: The imported module containing compute_spread_indicators and order_func
        source_code: Source code string (not used)
        test_data: Tuple of (asset_a_data, asset_b_data) DataFrames
        invariant_tests: List of invariant tests
        strategy_tests: List of strategy-specific tests

    Returns:
        VerificationResult with pass/fail status and details
    """
    _ = source_code

    result = VerificationResult(passed=True)

    if test_data is None:
        result.add_error("Test data required for TDD validation")
        return result

    # Check required functions
    required_funcs = ["compute_spread_indicators", "order_func"]
    for func_name in required_funcs:
        if not hasattr(generated_module, func_name):
            result.add_error(f"Missing required function: '{func_name}'")

    if not result.passed:
        return result

    asset_a, asset_b = test_data

    # For complex strategy, we test compute_spread_indicators with both assets.
    # The wrapper must truncate asset_b to match whenever the test harness
    # truncates asset_a (e.g., in the no-lookahead test).
    def generate_func(data: "pd.DataFrame"):
        b = asset_b.iloc[:len(data)]
        indicators = generated_module.compute_spread_indicators(data, b)
        return pd.Series(indicators.get("zscore", np.zeros(len(data))), index=data.index)

    run_invariant_tests(
        generate_func=generate_func,
        test_data=asset_a,
        result=result,
        invariant_tests=invariant_tests,
    )

    run_strategy_tests(
        generated_module=generated_module,
        test_data=test_data,
        result=result,
        strategy_tests=strategy_tests,
    )

    return result


# Strategy to verification function mapping
VERIFY_TESTS_FUNCTIONS = {
    "simple": verify_tests_simple,
    "medium": verify_tests_medium,
    "complex": verify_tests_complex,
}
