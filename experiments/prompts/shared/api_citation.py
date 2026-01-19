"""
API Citation Requirements
=========================

Instructions for documentation-grounded conditions requiring VAS compliance.
Used by: C2 (Docs), C4 (Schema+Docs), C6 (Docs+TDD), C7 (All)
"""

API_CITATION_SIMPLE = """
## API Citation Requirements

For EVERY third-party API call, you must:
1. Use fully-qualified module paths (e.g., `pd.Series.rolling()`, not `.rolling()`)
2. Only use APIs listed in the Verified API Surface above
3. Any unlisted API will cause validation failure

Example of CORRECT usage:
```python
# Correct: module-qualified call
rsi = vbt.RSI.run(close, window=14).rsi

# Correct: explicit pandas call
entries = pd.Series(np.zeros(len(close)), index=close.index, dtype=bool)
```

Example of INCORRECT usage:
```python
# Wrong: unqualified method call
rsi = close.rolling(14).apply(calc_rsi)  # .rolling() not qualified

# Wrong: API not in VAS
result = pd.Series.interpolate(data)  # interpolate not in VAS
```
"""

API_CITATION_MEDIUM = """
## API Citation Requirements

For EVERY third-party API call, you must:
1. Use fully-qualified module paths (e.g., `vbt.MACD.run()`, not unqualified calls)
2. Only use APIs listed in the Verified API Surface above
3. Any unlisted API will cause validation failure

Example of CORRECT usage:
```python
# Correct: module-qualified calls
macd_indicator = vbt.MACD.run(close, fast_window=12, slow_window=26, signal_window=9)
macd_line = macd_indicator.macd
signal_line = macd_indicator.signal

atr = vbt.ATR.run(high, low, close, window=14).atr
sma = vbt.MA.run(close, window=50).ma
```
"""

API_CITATION_COMPLEX = """
## API Citation Requirements

For EVERY third-party API call, you must:
1. Use fully-qualified module paths
2. Only use APIs listed in the Verified API Surface above
3. Any unlisted API will cause validation failure

Example of CORRECT usage:
```python
# Correct: scipy regression
from scipy.stats import linregress
slope, intercept, _, _, _ = scipy.stats.linregress(x, y)

# Correct: pandas rolling with qualified calls
spread = pd.Series(price_a.values - hedge_ratio * price_b.values, index=price_a.index)
rolling_mean = pd.Series.rolling(spread, window=20).mean()
rolling_std = pd.Series.rolling(spread, window=20).std()
```
"""
