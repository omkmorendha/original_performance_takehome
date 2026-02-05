# Iteration 001: SIMD Vectorization with VLIW Packing

**Date:** 2026-02-05
**Status:** ✓ Success
**Cycles:** 9293
**Speedup:** 15.9x
**Previous Best:** 147734 (baseline)

## Overview

Transform the scalar implementation to process 16 elements simultaneously (two groups of 8) using SIMD vector operations (VLEN=8), combined with VLIW packing to execute multiple operations per cycle.

## Hypothesis

The baseline processes 256 batch elements one at a time with scalar operations. By using SIMD:
- Process 8 elements per iteration instead of 1 (32x fewer iterations)
- Use vload/vstore to load/store 8 elements per instruction vs 1
- Use valu operations for element-wise computations
- Pack independent operations into single instruction bundles
- Process 2 groups in parallel to maximize VALU utilization (6 slots)

Expected speedup: 16-32x minimum from SIMD alone, potentially more with VLIW packing.

## Implementation Details

### Approach
1. Allocate vector scratch space for two groups (A and B)
2. Process 2 groups of 8 elements per iteration (16 total)
3. Use vload/vstore for vector memory operations
4. Vectorize hash function with parallel operations for both groups
5. Pack operations across engines (ALU, VALU, Load, Store)

### Key Optimizations Applied
1. **Dual group processing**: Process groups A and B simultaneously
2. **Hash parallelization**: Both groups hash in same cycles (4 VALU ops per cycle)
3. **Address calculation packing**: All 4 addresses in one ALU cycle
4. **Node address pipelining**: 8 addresses per group in one ALU cycle
5. **Parallel stores**: Both groups store in same cycle

### Cycle Breakdown (per dual-group iteration)
- 1 cycle: address calculation (4 ALU)
- 2 cycles: vload indices/values (2×2 vloads)
- 2 cycles: node address calculation (16 ALU in 2 cycles)
- 8 cycles: node value loads (16 loads, 2 per cycle)
- 1 cycle: XOR (2 VALU)
- 12 cycles: hash (6 stages × 2, both groups parallel)
- 1 cycle: mod/mul
- 1 cycle: equality
- 2 cycles: vselect (flow limit 1)
- 1 cycle: add
- 1 cycle: compare
- 2 cycles: vselect
- 2 cycles: vstore

Total: ~36 cycles per 16 elements → 256 iterations × 36 = 9216 cycles (close to actual 9293)

## Results

### Performance
```
Cycles: 9293
Speedup vs baseline: 15.9x
```

### Correctness
```bash
python perf_takehome.py Tests.test_kernel_cycles  # PASS
python tests/submission_tests.py  # Passes Tier 1 (< 18532)
```

Result: ✓ Passes correctness and Tier 1

## Lessons Learned

1. **Vector constant initialization timing**: Must pre-allocate all constants BEFORE the main loop to ensure vbroadcast happens before use
2. **Scattered loads are bottleneck**: 8 node loads per group = 4 cycles each due to 2 load slots/cycle
3. **Flow engine limitation**: vselect requires flow slot, only 1 per cycle - this serializes conditionals
4. **VALU utilization**: Using 4 of 6 VALU slots for dual-group hash, could potentially do 3 groups

## Next Steps

To reach target of <1,487 cycles (need ~6x more improvement):
1. Software pipelining: Overlap loads with computation
2. Process more groups in parallel (3 groups = 6 VALU slots)
3. Reduce scattered load overhead
4. Optimize flow operations (vselect bottleneck)
