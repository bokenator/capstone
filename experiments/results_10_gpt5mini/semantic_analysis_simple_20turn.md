# Semantic Equivalence Analysis - Simple Strategies (20-turn)

## Summary
- Identical: 5/40
- Partial: 27/40
- Different: 8/40

## Key Differences Explained

**Identical**: Uses vbt.RSI.run() exactly like reference, same crossing logic, same position building approach.

**Partial**: Uses custom Wilder EWM RSI (alpha=1/period) instead of vbt.RSI.run(). This is mathematically equivalent but implementation differs. Same entry/exit logic.

**Different**: Uses SMA-based RSI (rolling mean) instead of EWM/Wilder smoothing, OR has different crossing condition logic (e.g., <= instead of <), OR has other semantic differences.

---

## Detailed Results

### C0 (Control: ---)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses custom Wilder EWM RSI (alpha=1/period) instead of vbt.RSI.run(). Also fills early RSI with 50 during warmup. |
| 2 | different | Uses custom Wilder EWM RSI (alpha=1/period) instead of vbt.RSI.run(). Same crossing logic. |
| 3 | different | Uses custom Wilder EWM RSI (alpha=1/period) instead of vbt.RSI.run(). Same crossing logic. |
| 4 | different | Uses custom Wilder EWM RSI (alpha=1/period) instead of vbt.RSI.run(). Same crossing logic. |
| 5 | different | Uses custom Wilder EWM RSI with min_periods but no vbt. Also manually sets warmup NaNs. |

### C1 (S--)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Fills NaN with 50. Same crossing logic. |
| 2 | partial | Uses custom Wilder EWM RSI (alpha=1/period). Has edge case handling for zero gain/loss. Same crossing logic. |
| 3 | partial | Uses custom Wilder EWM RSI (alpha=1/period). Uses ffill approach for position building instead of iterative loop - may differ semantically. |
| 4 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Same crossing logic. |
| 5 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Same crossing logic. |

### C2 (E--)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | identical | Uses vbt.RSI.run(close, window=rsi_period).rsi. Same crossing logic (prev >= 30 and curr < 30). Same iterative position building. |
| 2 | identical | Uses vbt.RSI.run(close, window=rsi_window).rsi. Same crossing logic. Same iterative position building. |
| 3 | identical | Uses vbt.RSI.run(close, window=rsi_period).rsi. Same crossing logic. Same iterative position building. |
| 4 | partial | Uses vbt.RSI.run(). But uses cumsum approach for position building instead of iterative loop - may differ on edge cases. |
| 5 | identical | Uses vbt.RSI.run(close, window=rsi_period).rsi. Same crossing logic. Same iterative position building. |

### C3 (-S-)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Fills early RSI with 50. Same crossing logic. |
| 2 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Same crossing logic. |
| 3 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Same crossing logic. |
| 4 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Fills early RSI with 50 and uses ffill for prev_rsi. Same crossing logic. |
| 5 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Same crossing logic. |

### C4 (ES-)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses SMA rolling mean for RSI instead of Wilder EWM. Also uses different crossing logic (> and <= instead of >= and <). |
| 2 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Uses ffill approach for position building. |
| 3 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Uses cumsum approach for position building. |
| 4 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Same crossing logic. |
| 5 | different | Uses SMA rolling mean for initial RSI then Wilder smoothing (hybrid approach). Same crossing logic. |

### C5 (-SE)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Alternative RSI formula using (avg_gain / (avg_gain + avg_loss)). Same crossing logic. |
| 2 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Different crossing detection using below/above masks instead of prev comparison. |
| 3 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Same crossing logic. |
| 4 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Also allows entry on first bar if RSI < oversold. |
| 5 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Same crossing logic. |

### C6 (--E)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses SMA rolling mean for initial RSI then Wilder smoothing (hybrid approach). Same crossing logic. |
| 2 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Different crossing logic (> and < instead of >= and <=). |
| 3 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Allows entry on first valid RSI if below oversold. |
| 4 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Fills early NaNs with 50. Same crossing logic. |
| 5 | different | Uses SMA rolling mean for initial RSI then Wilder smoothing (hybrid approach). Same crossing logic. |

### C7 (ESE)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Same crossing logic. |
| 2 | different | Uses SMA rolling mean for RSI instead of Wilder EWM. Same crossing logic. |
| 3 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Fills early NaNs with 50. Allows entry on first bar if RSI < oversold. |
| 4 | partial | Uses custom Wilder EWM RSI (alpha=1/period, no min_periods). Fills early NaNs with 50. Same crossing logic. |
| 5 | identical | Uses custom Wilder EWM RSI (alpha=1/period) with min_periods. Same crossing logic. Same iterative position building. |

---

## Summary by Condition

| Condition | Identical | Partial | Different |
|-----------|-----------|---------|-----------|
| C0 (Control: ---) | 0 | 0 | 5 |
| C1 (S--) | 0 | 5 | 0 |
| C2 (E--) | 4 | 1 | 0 |
| C3 (-S-) | 0 | 5 | 0 |
| C4 (ES-) | 0 | 3 | 2 |
| C5 (-SE) | 0 | 5 | 0 |
| C6 (--E) | 0 | 3 | 2 |
| C7 (ESE) | 1 | 3 | 1 |

## Key Observations

1. **C0 (Control)**: All 5 runs classified as "different" because they use custom Wilder EWM RSI instead of vbt.RSI.run(). However, Wilder EWM is mathematically similar to vbt's RSI implementation.

2. **C2 (Examples only)**: Best performance with 4 identical implementations. The examples condition appears to effectively guide the model to use vbt.RSI.run() as specified.

3. **C1, C3, C5 (Schema conditions)**: All partial - use custom RSI but with correct logic.

4. **C4, C6 (Mixed conditions)**: Some implementations use SMA-based RSI which is semantically different from the reference Wilder/EWM approach.

5. **Most common difference**: Using custom Wilder EWM RSI (alpha=1/period) instead of vbt.RSI.run(). This is considered "partial" match because:
   - The mathematical formula is equivalent (both use exponential smoothing)
   - The entry/exit logic is identical
   - Position building is the same
   - Only the API call differs

6. **True semantic differences** (classified as "different"):
   - SMA-based RSI (rolling mean) instead of EWM
   - Different threshold comparison operators (> vs >=, < vs <=)
   - Hybrid SMA-then-Wilder approaches

## Note on Classification Criteria

- **Identical**: Uses vbt.RSI.run() API, same crossing logic, same position building
- **Partial**: Different RSI implementation (typically Wilder EWM) but semantically equivalent logic
- **Different**: Uses fundamentally different RSI calculation (SMA) or different entry/exit logic
