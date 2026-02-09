# Semantic Equivalence Analysis - Complex Strategies (20-turn)

## Summary
- Identical: 0/40
- Partial: 21/40
- Different: 19/40

## Key Differences from Reference

The reference implementation has these key characteristics:
1. **compute_spread_indicators()**: Uses scipy.stats.linregress, lookback=60 for hedge ratio, lookback=20 for z-score, uses vbt.MA.run() for rolling mean
2. **order_func()**:
   - Entry: z > 2 -> short A, long B; z < -2 -> long A, short B
   - Exit: z-score CROSSES exit_threshold (0.0) - requires checking previous bar's value AND current
   - Stop-loss: |z| > 3
   - Position sizing: shares_a = notional/price_a, shares_b = (notional/price_b) * abs(hr)
   - Uses module-level _state to track position_type across bars
   - Returns (-size, 0, 0) format with signed size

Most generated implementations differ in:
- No module-level state tracking (they rely on c.position_now)
- Different position sizing for B: some use hr*shares_a instead of (notional/price_b)*abs(hr)
- Exit logic variations: some check only sign change, others check crossing through threshold
- Return format variations: some use different size_type/direction values

## Detailed Results

### C0 (Control: ---)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | order_func uses value-based sizing (size_type=1), direction encoding (1=LONG, 2=SHORT), exit on sign flip only, position sizing different |
| 2 | different | Uses scipy linregress (correct), but position sizing uses hr*units_a for B instead of (notional/price_b)*hr, exit checks sign flip |
| 3 | different | Uses numpy.polyfit instead of linregress, position sizing uses hr*units_a for B, uses rolling mean/std differently |
| 4 | partial | Uses numpy.linalg.lstsq for OLS, exit crossing logic correct, position sizing close but not identical |
| 5 | partial | Uses scipy linregress, exit crossing logic present, sizing uses abs(hr)*units_a for B |

### C1 (Condition: E--)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses scipy linregress, target-based ordering with delta from current, exit crossing detection present |
| 2 | partial | Uses scipy linregress, shares_b = (notional/price_b)*abs(hr) matches reference, crossing detection present |
| 3 | partial | Uses scipy linregress, shares_b = hedge*shares_a (different from reference), exit logic correct |
| 4 | different | Uses scipy linregress, expanding window carries forward previous hedge ratio, shares_b = hr*shares_a |
| 5 | partial | Uses scipy linregress, shares_b = hedge*shares_a, exit crossing logic present |

### C2 (Condition: -S-)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses vectorbt enums directly, uses SizeType.Amount/Direction.ShortOnly etc., exit logic checks sign change only |
| 2 | different | Uses scipy linregress, returns (units, size_type, direction) with integer-floored units, different sizing |
| 3 | different | Uses scipy linregress and vectorbt enums, returns (notional, SizeType.Value, direction), notional-based not share-based |
| 4 | different | Uses analytic OLS formula, returns delta in shares, but sizing uses hr*base_units for B |
| 5 | partial | Uses scipy linregress, exit crossing logic present, sizing uses hr*units_a for B |

### C3 (Condition: ES-)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses scipy linregress, forward-fills NaN hedge ratios, fills zscore NaNs with 0, exit logic present |
| 2 | different | Uses scipy linregress, expanding window with fallback, returns (size, 1, direction) value-based sizing |
| 3 | partial | Uses manual OLS, target_b = -hr*target_a (different relationship), exit crossing logic present |
| 4 | partial | Uses scipy linregress, sizing: size_a=notional/price_a, size_b=abs(hr)*size_a, exit logic correct |
| 5 | partial | Uses manual OLS for hedge ratio, sizing uses abs(hr)*units_a for B, exit crossing logic present |

### C4 (Condition: --T)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses scipy linregress, shares_b=(notional/price_b)*hr (matches reference), exit crossing logic present |
| 2 | partial | Uses scipy linregress, shares_b=(notional/price_b)*abs(hr) (matches reference), exit on sign change |
| 3 | partial | Uses scipy linregress, shares_b=(notional/price_b)*hr, exit crossing detection present |
| 4 | different | Uses scipy linregress, shares_b=(notional/price_b)*hr, handles position reversal logic |
| 5 | partial | Uses scipy linregress, shares_b=(notional/price_b)*hr, exit crossing logic present |

### C5 (Condition: E-T)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses scipy linregress with expanding window, shares_b=hr*shares_a, also exits when |z|<=exit_threshold (incorrect) |
| 2 | different | Uses scipy linregress with expanding window, desired_b=-hr*desired_a (different relationship), min_periods=1 |
| 3 | different | Uses scipy linregress with expanding window, desired_b=-hr*desired_a, exit crossing logic present |
| 4 | different | Uses scipy linregress with expanding window, entry uses sign_z for direction, different from reference |
| 5 | partial | Uses scipy linregress, shares_b=shares_a*hedge, exit crossing logic present |

### C6 (Condition: -ST)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | different | Uses scipy linregress, uses vectorbt enums (SizeType.Amount, Direction.ShortOnly etc.), expanding window |
| 2 | different | Uses scipy linregress, uses TargetAmount to close, Direction enums, expanding window with fallback |
| 3 | different | Uses scipy linregress, uses SizeType/Direction enums, expanding window, sizing uses abs(hr)*units_a |
| 4 | different | Uses numpy.polyfit, expanding window, sizing uses abs(hr)*units_a, exit with closing order using Both direction |
| 5 | partial | Uses manual OLS, sizing uses hr*base_units for B (desired_b=-hr*desired_a relationship) |

### C7 (Condition: EST)
| Run | Result | Reason |
|-----|--------|--------|
| 1 | partial | Uses scipy linregress, expanding window, shares_b=shares_a*hr, exit crossing logic present |
| 2 | partial | Uses scipy linregress, expanding window, target_b=-hr*target_a, exit crossing logic present |
| 3 | different | Uses scipy linregress with STRICT no-lookahead (window ends at i-1), shares_b=shares_a*hr |
| 4 | different | Uses scipy linregress with STRICT no-lookahead (window ends at i-1), target_b=-hr*target_a, z-score uses shift(1) |
| 5 | partial | Uses scipy linregress with smaller-than-lookback windows at start, target calculation matches reference pattern |

## Notes on Classification Criteria

**Identical**: Same logic as reference in all critical aspects:
- Rolling OLS using scipy.stats.linregress with lookback=60
- Z-score calculation with lookback=20 using same formula
- Exit on z-score crossing through exit_threshold (not just sign change)
- Stop-loss when |z| > 3
- Position sizing: shares_a = notional/price_a, shares_b = (notional/price_b) * abs(hr)
- Module-level state tracking for position type

**Partial**: Acceptable implementation with minor differences:
- Slightly different OLS implementation (polyfit, manual, lstsq) that produces equivalent results
- Minor differences in edge case handling
- Uses c.position_now instead of module-level state
- Sizing relationship B = hr * units_A (equivalent but different formula)
- Exit checks sign change rather than formal crossing

**Different**: Semantically different behavior:
- Different sizing logic (value-based instead of share-based)
- Different exit conditions (wrong threshold interpretation)
- Wrong position sizing relationship
- Uses expanding window instead of fixed lookback
- Returns different tuple format incompatible with reference
- Forward-fills NaN values affecting calculations
