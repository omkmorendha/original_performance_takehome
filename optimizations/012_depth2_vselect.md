# Optimization 012: Depth-2 vselect Broadcast (Experiment)

**Date**: 2026-02-06
**Status**: ✗ Failed (slower)
**Result**: 2243 cycles (65.86x speedup)
**Change vs best**: +206 cycles from 2037

## Summary

Attempted to replace depth-2 scattered loads with a scratch-resident broadcast
that selects among tree[3..6] using lane-wise `vselect` based on index bits.

## Approach

1. Preload tree[3..6] into scratch and broadcast into vectors.
2. Compute bit masks:
   - `b0 = idx & 1`
   - `b1 = (idx >> 1) & 1`
3. Use a 2-level vselect tree per group to select the correct node value:
   - b1=0 → {tree[4], tree[5]} via b0
   - b1=1 → {tree[6], tree[3]} via b0

## Outcome

```
$ python perf_takehome.py Tests.test_kernel_cycles
CYCLES:  2243
```

## Why It Lost

- **Flow engine bottleneck**: 3 vselects × 8 groups = 24 flow cycles per iteration.
- **Extra VALU masks**: 4 VALU cycles for b0/b1 computation.
- Net result: ~50 cycles/iter vs 33 cycles for pipelined scattered loads.

## Notes

The code path remains available behind `USE_D2_VSELECT` (default off) for future
experiments, but it is not enabled because it regresses Tier 2 performance.

## Files Changed

- `perf_takehome.py` (experiment, gated)

