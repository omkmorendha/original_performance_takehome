# Optimization 011: Depth-0 Zero-Index Update

**Date**: 2026-02-06
**Status**: ✓ Completed
**Result**: 2037 cycles (72.53x speedup, Tier 2)
**Improvement**: -16 cycles from 2053

## Summary

Depth-0 rounds always start with `idx = 0` for all elements. We exploit this invariant to
replace the generic `idx = 2*idx + 1 + (val&1)` update with a cheaper `idx = 1 + (val&1)`
sequence.

## Hypothesis

At depth 0, `idx` is always 0 (initially, and after depth-10 wrap). The update can be
reduced to a parity check plus add, saving VALU cycles in broadcast rounds.

## Implementation

### Specialized update for depth-0

**Before**
- Generic update: `multiply_add` + `&` + `+` (5 VALU cycles, no wrap path).

**After**
- Specialized path: `&` + `+` only.
- Packed with the final hash op to minimize cycle count.

## Results

```
$ python perf_takehome.py Tests.test_kernel_cycles
CYCLES:  2037
Speedup over baseline:  72.53x
```

## Analysis

- Depth-0 rounds: 8 iterations total (2 rounds × 4 iters).
- Savings: 2 cycles/iteration → 16 cycles overall.
- No changes to multicore behavior (still single-core).

## Files Changed

- `perf_takehome.py`

