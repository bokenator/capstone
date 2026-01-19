"""
Strategy-Specific Tests
=======================

Tests specific to each strategy complexity level.
Used by: C3 (TDD), C5 (Schema+TDD), C6 (Docs+TDD), C7 (All)
"""

STRATEGY_TESTS_SIMPLE = """
## Strategy-Specific Tests

Your code must also pass these strategy-specific tests:

### Test 1: Entry Signal on Oversold
```python
def test_entry_on_oversold(generate_signals):
    \"\"\"Must generate entry when RSI crosses below 30.\"\"\"
    # Create synthetic data where RSI will cross below 30
    prices = create_declining_prices(n=100, start=100, end=70)
    entries, exits = generate_signals(prices)

    # Must have at least one entry signal
    assert entries.any(), "No entry signal generated on oversold condition"
```

### Test 2: Exit Signal on Overbought
```python
def test_exit_on_overbought(generate_signals):
    \"\"\"Must generate exit when RSI crosses above 70.\"\"\"
    # Create synthetic data where RSI will cross above 70
    prices = create_rising_prices(n=100, start=100, end=150)
    entries, exits = generate_signals(prices)

    # Must have at least one exit signal
    assert exits.any(), "No exit signal generated on overbought condition"
```

### Test 3: No Entry When Already Long
```python
def test_no_double_entry(generate_signals):
    \"\"\"Should not signal entry when already in position.\"\"\"
    prices = create_volatile_prices(n=200)
    entries, exits = generate_signals(prices)

    # Check that we don't have consecutive entries without an exit
    in_position = False
    for i in range(len(entries)):
        if entries.iloc[i]:
            assert not in_position, "Entry signal while already in position"
            in_position = True
        if exits.iloc[i]:
            in_position = False
```
"""

STRATEGY_TESTS_MEDIUM = """
## Strategy-Specific Tests

Your code must also pass these strategy-specific tests:

### Test 1: Entry Requires Trend Filter
```python
def test_entry_requires_uptrend(order_func, compute_indicators):
    \"\"\"Entry must only occur when price > SMA.\"\"\"
    # Create data where MACD crosses up but price < SMA
    prices = create_prices_below_sma(n=200)
    indicators = compute_indicators(prices)

    # Simulate and check no entries when below SMA
    # ... (test harness will run simulation)
    assert no_entries_below_sma, "Entry occurred when price below SMA"
```

### Test 2: Trailing Stop Triggers Exit
```python
def test_trailing_stop_exit(order_func, compute_indicators):
    \"\"\"Position must close when price drops 2*ATR from high.\"\"\"
    # Create data with entry then sharp decline
    prices = create_entry_then_crash(n=200, crash_pct=0.15)
    indicators = compute_indicators(prices)

    # Must exit before reaching bottom
    assert exited_on_trailing_stop, "Trailing stop did not trigger"
```

### Test 3: Trailing Stop Updates
```python
def test_trailing_stop_updates(order_func):
    \"\"\"Trailing stop must update as price makes new highs.\"\"\"
    # This tests that highest_since_entry is tracked correctly
    # ... (test harness validates internal state)
```

### Test 4: MACD Cross Exit
```python
def test_macd_cross_exit(order_func, compute_indicators):
    \"\"\"Position must close on MACD bearish cross.\"\"\"
    prices = create_macd_bearish_cross(n=200)
    indicators = compute_indicators(prices)

    assert exited_on_macd_cross, "MACD bearish cross did not trigger exit"
```
"""

STRATEGY_TESTS_COMPLEX = """
## Strategy-Specific Tests

Your code must also pass these strategy-specific tests:

### Test 1: Opposite Positions
```python
def test_opposite_positions(order_func, compute_indicators):
    \"\"\"When in trade, positions must be opposite (long/short or short/long).\"\"\"
    prices_a, prices_b = create_cointegrated_pair(n=300)
    indicators = compute_indicators(prices_a, prices_b)

    # Check that when position_a > 0, position_b < 0 (and vice versa)
    assert positions_are_opposite, "Pairs positions must be opposite"
```

### Test 2: Entry on Z-Score Threshold
```python
def test_entry_on_zscore(order_func, compute_indicators):
    \"\"\"Must enter when |z-score| > 2.\"\"\"
    prices_a, prices_b = create_diverging_pair(n=300, divergence=3.0)
    indicators = compute_indicators(prices_a, prices_b)

    assert entered_on_zscore_breach, "Did not enter when z-score > 2"
```

### Test 3: Exit on Mean Reversion
```python
def test_exit_on_reversion(order_func, compute_indicators):
    \"\"\"Must exit when z-score crosses 0.\"\"\"
    prices_a, prices_b = create_reverting_pair(n=300)
    indicators = compute_indicators(prices_a, prices_b)

    assert exited_on_reversion, "Did not exit when z-score crossed 0"
```

### Test 4: Stop Loss Triggers
```python
def test_stop_loss(order_func, compute_indicators):
    \"\"\"Must exit when |z-score| > 3.\"\"\"
    prices_a, prices_b = create_diverging_pair(n=300, divergence=4.0)
    indicators = compute_indicators(prices_a, prices_b)

    assert exited_on_stop, "Stop loss did not trigger at z-score > 3"
```

### Test 5: Hedge Ratio Applied
```python
def test_hedge_ratio_sizing(order_func, compute_indicators):
    \"\"\"Asset B position must be scaled by hedge ratio.\"\"\"
    prices_a, prices_b = create_cointegrated_pair(n=300, hedge_ratio=1.5)
    indicators = compute_indicators(prices_a, prices_b)

    # position_b should be approximately hedge_ratio * position_a (opposite sign)
    assert hedge_ratio_applied, "Hedge ratio not applied to position sizing"
```
"""
