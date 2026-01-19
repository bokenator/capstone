"""
Invariant Tests
===============

Tests that apply to all strategies regardless of complexity.
Used by: C3 (TDD), C5 (Schema+TDD), C6 (Docs+TDD), C7 (All)
"""

INVARIANT_TESTS = """
## Invariant Tests

Your generated code must pass ALL of these invariant tests. The test harness will run these automatically.

### Invariant 1: Output Shape
```python
def test_output_shape(output, input_data):
    \"\"\"Output must have same length as input.\"\"\"
    assert len(output) == len(input_data), "Output length must match input length"
```

### Invariant 2: No Future Data (Lookahead Bias)
```python
def test_no_lookahead(generate_func, input_data):
    \"\"\"Signal at time t must not depend on data after time t.\"\"\"
    full_output = generate_func(input_data)

    # Test with truncated data
    for t in [50, 100, 150]:
        truncated_data = input_data.iloc[:t]
        truncated_output = generate_func(truncated_data)

        # Signals up to t must be identical
        assert all(truncated_output == full_output.iloc[:t]), \\
            f"Signal at t<{t} changed when future data removed - lookahead bias detected"
```

### Invariant 3: Determinism
```python
def test_deterministic(generate_func, input_data):
    \"\"\"Same input must produce same output.\"\"\"
    output1 = generate_func(input_data)
    output2 = generate_func(input_data)
    assert all(output1 == output2), "Function must be deterministic"
```

### Invariant 4: No NaN in Output (after warmup)
```python
def test_no_nan_after_warmup(output, warmup_period=50):
    \"\"\"No NaN values after warmup period.\"\"\"
    assert not output.iloc[warmup_period:].isna().any(), \\
        "Output contains NaN after warmup period"
```
"""
