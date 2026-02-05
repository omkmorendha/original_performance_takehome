# Iteration 002: Software Pipelining

**Date:** 2026-02-05
**Status:** ✓ Success
**Cycles:** 6173
**Speedup:** 23.9x (1.51x over previous)
**Previous Best:** 9293 cycles (15.9x)

## Overview

Implement software pipelining to overlap memory loads with computation. The key insight is that while we're computing the hash for the current iteration, we can be loading data for the next iteration.

## Hypothesis

Current bottleneck analysis (per dual-group, 16 elements):
- Load phase: ~13 cycles (addresses, vloads, scattered node loads)
- Compute phase: ~23 cycles (XOR, hash, index calc, stores)

By overlapping loads with computation:
- Hash takes 12 cycles - plenty of time to load next iteration's data
- Load slots (2/cycle) are unused during hash computation
- Could potentially reduce effective cycles per iteration by ~10 cycles

Expected improvement: ~30% reduction (9293 → ~6500 cycles)

## Implementation Details

### Approach
1. Double buffering: 2 complete sets of vector registers
2. Prologue: Load first iteration data (no pipelining)
3. Steady state: Compute current buffer while loading into next buffer
4. Epilogue: Compute final buffer (no more loads)

### Key Pipelining Points
- Hash stage 0: Load vload for indices/values
- Hash stage 1: Compute node addresses (ALU)
- Hash stages 2-5: Load scattered node values (2 per cycle)
- Index calc phase: Load remaining node values for group B

### Technical Details
- Double buffered: v_idx[2][2], v_val[2][2], v_node_val[2][2], etc.
- Scratch usage: 429 words (was 240)
- Each pipelined iteration overlaps ~13 cycles of loads with compute

### Cycle Analysis (per dual-group)
Without pipelining: ~36 cycles (13 load + 23 compute)
With pipelining: ~23 cycles (loads hidden behind compute)
Saving: ~13 cycles per iteration

Total: 16 rounds × 16 iterations × 23 cycles ≈ 5888 cycles (close to actual 6173)

## Results

### Performance
```
Cycles: 6173
Speedup vs baseline: 23.9x
Speedup vs previous: 1.51x
```

### Correctness
```bash
python perf_takehome.py Tests.test_kernel_cycles  # PASS
python tests/submission_tests.py  # Passes Tier 1
```

## Lessons Learned

1. **Buffer switching is tricky**: Initial implementation had wrong buffer logic - was computing buffer before loading it
2. **Prologue/epilogue pattern**: Need separate handling for first and last iterations
3. **Load/compute interleaving**: Can fit 2 loads per cycle during hash stages
4. **ALU is underutilized**: During hash stages, ALU slots are mostly free - used for address calculation

## Next Steps

To reach Tier 2 (<2,164 cycles, need ~2.85x more):
1. Process more groups in parallel (3+ groups using 6 VALU slots)
2. Further reduce flow bottleneck (vselect serialization)
3. Optimize index calculation phase
4. Consider loop restructuring for better pipelining
