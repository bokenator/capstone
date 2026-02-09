# Semantic Equivalence Analysis - Medium Strategies (20-turn)

## Summary
- Identical: 0/40
- Partial: 33/40
- Different: 5/40
- Failed: 2/40

## Key Differences from Reference

The reference implementation uses:
1. **MACD**: `vbt.MACD.run()` with parameters (12, 26, 9)
2. **ATR**: `vbt.ATR.run()` with period 14
3. **SMA**: `vbt.MA.run()` with period 50
4. **Entry**: MACD crosses above signal AND price > SMA, enter with 95% equity
5. **Exit**: MACD crosses below signal OR price < (highest_since_entry - trailing_mult * ATR)
6. **Position sizing**: Returns `(0.95, 2, 1)` for entry, `(-np.inf, 2, 1)` for exit

Most generated implementations differ in:
- Using custom pandas-based MACD/ATR instead of vbt functions (C0, C3 conditions)
- Different position sizing (50% vs 95% of equity)
- Entry conditions: MACD line crosses above signal vs histogram > 0

## Detailed Results

### C0 (Control: ---)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses custom pandas EMA for MACD (not vbt.MACD.run()), uses Wilder's ATR (ewm alpha=1/period), entry returns 100% equity not 95% |
| 2 | different | Uses custom pandas EMA for MACD, uses simple rolling ATR, entry returns 100% equity |
| 3 | different | Uses custom pandas EMA for MACD, uses EWM ATR (span=period), returns (1.0, 0, 1) for entry instead of percent sizing |
| 4 | different | Uses custom pandas EMA for MACD, uses simple rolling ATR, entry returns (1.0, 0, 0) with Amount sizing |
| 5 | different | Uses custom pandas EMA for MACD, uses simple rolling ATR, entry returns (1.0, 1, 1) with Percent sizing |

### C1 (RAG: +++)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), but entry uses 50% equity instead of 95% |
| 2 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 100% equity, same logic |
| 3 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 4 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 5 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 100% equity |

### C2 (Semantic Linting: +--)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), but returns (1.0, 5, 0) - uses TargetPercent=5 instead of Percent=2 |
| 2 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), imports SizeType/Direction enums, uses TargetPercent sizing |
| 3 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), uses TargetPercent=5 with Direction.Both=2 |
| 4 | failed | Generation failed: Max turns (20) exceeded |
| 5 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), uses pandas rolling for SMA (not vbt.MA), TargetPercent sizing |

### C3 (Exec Feedback: --+)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses custom pandas EMA for MACD, simple rolling ATR, returns (1.0, 0, 1) Amount sizing |
| 2 | different | Uses custom pandas EMA for MACD, Wilder's ATR (alpha=1/period), returns (1.0, 0, 1) |
| 3 | partial | Uses vbt.MACD.run() but imports SizeType/Direction enums, simulates bar-by-bar (O(n^2) complexity) |
| 4 | different | Uses custom pandas EMA for MACD, Wilder's ATR, simulates bar-by-bar (O(n^2)), returns (-1.0, 0, 0) for exit |
| 5 | different | Uses custom pandas EMA for MACD, simple rolling ATR, returns (-1.0, 0, 0) for exit |

### C4 (RAG + Semantic Linting: ++-)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 2 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 3 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 4 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 5 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity, strict crossover (macd[i-1] < signal) |

### C5 (RAG + Exec Feedback: +-+)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity, uses strict crossover |
| 2 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity, ffills NaN indicators |
| 3 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 4 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity, strict crossover |
| 5 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |

### C6 (Semantic Linting + Exec Feedback: -++)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), imports SizeType/Direction enums, uses TargetPercent sizing |
| 2 | failed | Generation failed: Max turns (20) exceeded |
| 3 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), imports SizeType/Direction enums, Percent sizing |
| 4 | partial | Uses custom pandas EMA for MACD, Wilder's ATR, TargetPercent=5 sizing |
| 5 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), TargetPercent=5 sizing |

### C7 (All Treatments: +++)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 2 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity, simulates bar-by-bar for trailing stop |
| 3 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 4 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity |
| 5 | partial | Uses vbt.MACD.run() and vbt.ATR.run(), entry uses 50% equity, ffills NaN indicators |

## Analysis by Condition

### Indicator Calculation Method
| Condition | Uses vbt.MACD.run() | Uses custom pandas EMA |
|-----------|---------------------|------------------------|
| C0 (Control) | 0/5 | 5/5 |
| C1 (RAG) | 5/5 | 0/5 |
| C2 (SemLint) | 4/4* | 0/4* |
| C3 (ExecFB) | 1/5 | 4/5 |
| C4 (RAG+SemLint) | 5/5 | 0/5 |
| C5 (RAG+ExecFB) | 5/5 | 0/5 |
| C6 (SemLint+ExecFB) | 4/4* | 0/4* |
| C7 (All) | 5/5 | 0/5 |

*Note: C2 run 4 and C6 run 2 failed to generate code

### Position Sizing (Entry)
| Condition | 95% equity | 100% equity | 50% equity | Other |
|-----------|------------|-------------|------------|-------|
| C0 (Control) | 0 | 3 | 0 | 2 |
| C1 (RAG) | 0 | 2 | 3 | 0 |
| C2 (SemLint) | 0 | 4 | 0 | 0 |
| C3 (ExecFB) | 0 | 2 | 0 | 3 |
| C4 (RAG+SemLint) | 0 | 0 | 5 | 0 |
| C5 (RAG+ExecFB) | 0 | 0 | 5 | 0 |
| C6 (SemLint+ExecFB) | 0 | 4 | 0 | 0 |
| C7 (All) | 0 | 0 | 5 | 0 |

### Key Observations

1. **RAG treatment strongly influences indicator implementation**: Conditions with RAG (C1, C4, C5, C7) consistently use `vbt.MACD.run()` and `vbt.ATR.run()`, while C0 (control) and C3 (exec feedback only) predominantly use custom pandas implementations.

2. **Position sizing varies significantly**: The reference uses 95% equity, but no generated implementation matches this exactly. Most RAG-enabled conditions use 50% or 100% equity.

3. **All implementations are semantically different from reference**: While the core strategy logic (MACD crossover + SMA filter + ATR trailing stop) is preserved, the specific implementations differ in:
   - Indicator calculation methods
   - Position sizing
   - Size type constants (Amount=0, Value=1, Percent=2, TargetPercent=5)
   - Direction constants

4. **Failed generations**: Two runs (C2_4, C6_2) exceeded the 20-turn limit without producing valid code.

5. **Entry condition consistency**: All successful implementations correctly identify MACD crossing above signal AND price > SMA as the entry condition.

6. **Exit condition consistency**: All successful implementations correctly implement both exit conditions (MACD cross down OR trailing stop breach).
