# Semantic Equivalence Analysis Summary (20-turn)

*Generated from thorough code-level comparison against reference implementations.*

## Classification Definitions

| Category | Definition |
|----------|------------|
| **Identical** | Uses same API calls (vbt functions), same logic, same input produces same output |
| **Partial** | Different implementation but mathematically equivalent (e.g., custom Wilder EWM vs vbt.RSI), core strategy logic preserved |
| **Different** | Fundamentally different calculation (e.g., SMA vs EWM), wrong thresholds, or incompatible output format |

## Results by Complexity

### Simple Strategies (RSI Mean Reversion)

| Condition | Identical | Partial | Different | Total |
|-----------|-----------|---------|-----------|-------|
| C0 (---) | 0 | 0 | 5 | 5 |
| C1 (S--) | 0 | 5 | 0 | 5 |
| C2 (-D-) | 4 | 1 | 0 | 5 |
| C3 (--T) | 0 | 5 | 0 | 5 |
| C4 (SD-) | 0 | 3 | 2 | 5 |
| C5 (S-T) | 0 | 5 | 0 | 5 |
| C6 (-DT) | 0 | 3 | 2 | 5 |
| C7 (SDT) | 1 | 3 | 1 | 5 |
| **Total** | **5** | **27** | **8** | **40** |

**Key finding**: Only C2 (D-only) consistently produces identical implementations using `vbt.RSI.run()`.

### Medium Strategies (MACD + ATR Trailing Stop)

| Condition | Identical | Partial | Different | Failed | Total |
|-----------|-----------|---------|-----------|--------|-------|
| C0 (---) | 0 | 0 | 5 | 0 | 5 |
| C1 (S--) | 0 | 5 | 0 | 0 | 5 |
| C2 (-D-) | 0 | 4 | 0 | 1 | 5 |
| C3 (--T) | 0 | 1 | 4 | 0 | 5 |
| C4 (SD-) | 0 | 5 | 0 | 0 | 5 |
| C5 (S-T) | 0 | 5 | 0 | 0 | 5 |
| C6 (-DT) | 0 | 4 | 0 | 1 | 5 |
| C7 (SDT) | 0 | 5 | 0 | 0 | 5 |
| **Total** | **0** | **33** | **5** | **2** | **40** |

**Key finding**: RAG-enabled conditions (C1, C4, C5, C7) use `vbt.MACD.run()` and `vbt.ATR.run()` → partial. C0 and C3 use custom pandas → different.

### Complex Strategies (Pairs Trading)

| Condition | Identical | Partial | Different | Total |
|-----------|-----------|---------|-----------|-------|
| C0 (---) | 0 | 2 | 3 | 5 |
| C1 (S--) | 0 | 4 | 1 | 5 |
| C2 (-D-) | 0 | 1 | 4 | 5 |
| C3 (--T) | 0 | 3 | 2 | 5 |
| C4 (SD-) | 0 | 4 | 1 | 5 |
| C5 (S-T) | 0 | 1 | 4 | 5 |
| C6 (-DT) | 0 | 1 | 4 | 5 |
| C7 (SDT) | 0 | 4 | 1 | 5 |
| **Total** | **0** | **21** | **19** | **40** |

**Key finding**: No identical matches. Complex strategies have highest "different" rate due to order_func variations.

## Aggregate by Condition (All Complexities)

| Condition | Identical | Partial | Different | Failed | Total | Equiv Rate |
|-----------|-----------|---------|-----------|--------|-------|------------|
| C0 (---) | 0 | 2 | 13 | 0 | 15 | 13% |
| C1 (S--) | 0 | 14 | 1 | 0 | 15 | 93% |
| C2 (-D-) | 4 | 6 | 4 | 1 | 15 | 67% |
| C3 (--T) | 0 | 9 | 6 | 0 | 15 | 60% |
| C4 (SD-) | 0 | 12 | 3 | 0 | 15 | 80% |
| C5 (S-T) | 0 | 11 | 4 | 0 | 15 | 73% |
| C6 (-DT) | 0 | 8 | 6 | 1 | 15 | 53% |
| C7 (SDT) | 1 | 12 | 2 | 0 | 15 | 87% |
| **Total** | **5** | **81** | **32** | **2** | **120** | **72%** |

*Equiv Rate = (Identical + Partial) / (Total - Failed)*

## Treatment Effects on Semantic Equivalence

### Combined (Identical + Partial) Rate

| Treatment | Enabled | Disabled | Difference |
|-----------|---------|----------|------------|
| Schema (S) | 50/60 (83%) | 36/60 (60%) | +23pp |
| Docs (D) | 44/60 (73%) | 42/60 (70%) | +3pp |
| Tests (T) | 41/60 (68%) | 45/60 (75%) | -7pp |

### Key Observations

1. **Schema (S) has the strongest positive effect** on semantic equivalence (+23pp). S-enabled conditions produce more consistent implementations.

2. **Docs (D) effect is modest** (+3pp overall). However, D is critical for "identical" matches on simple strategies (4/5 identical in C2).

3. **Tests (T) has slightly negative effect** (-7pp). This may be because test-driven generation focuses on passing tests rather than matching reference implementation.

4. **C0 (Control) performs worst** with only 13% equivalence rate. Without any treatment, the LLM uses custom implementations that differ significantly.

5. **C7 (SDT) and C1 (S--) perform best** with 87% and 93% equivalence respectively. Schema alone is highly effective.

## Notes on Correctness vs Semantic Equivalence

- **Pathological returns still fail**: C0 and C3 complex runs produce 10^15%+ returns → marked as "different" regardless of indicator logic
- **Partial ≠ Incorrect**: Most "partial" implementations produce numerically similar results; they just use different code paths
- **Different ≠ Failed**: "Different" implementations may still pass backtests and produce reasonable returns, but use fundamentally different logic
