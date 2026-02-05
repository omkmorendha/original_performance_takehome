# Iteration 000: Baseline Implementation

**Date:** 2026-02-05
**Status:** ✓ Baseline
**Cycles:** 147734
**Speedup:** 1.00x

## Overview

This is the original naive scalar implementation that serves as our baseline. It processes each element sequentially using only scalar ALU operations.

## Implementation Details

### Approach
- Fully scalar processing (one element at a time)
- Sequential execution through all batch elements
- Nested loops: rounds (16) × batch_size (256)
- No SIMD vectorization
- No VLIW instruction packing
- No optimization of instruction scheduling

### Key Characteristics

**Scratch Space Usage:**
- 7 initial variables (rounds, n_nodes, batch_size, forest_height, pointers)
- 3 temporary registers (tmp1, tmp2, tmp3)
- Per-iteration scratch: tmp_idx, tmp_val, tmp_node_val, tmp_addr
- Constants: 0, 1, 2, and batch element indices

**Instruction Pattern (per element):**
1. Load index from memory
2. Load value from memory
3. Load node value from tree
4. XOR value with node_value
5. Hash computation (6 stages × 3-4 operations each)
6. Compute next index (modulo, multiply, add)
7. Wrap index if >= n_nodes
8. Store index back to memory
9. Store value back to memory

**Operations per Element:**
- ~30-35 ALU operations
- 3 loads
- 2 stores
- Multiple flow control (select operations)
- Debug/compare instructions (ignored in final submission)

### Bottlenecks Identified

1. **No SIMD:** Processing one element at a time despite VLEN=8 capability
2. **Poor VLIW utilization:** One engine per cycle, not leveraging parallel execution
3. **Hash function overhead:** 6 stages with serial dependencies
4. **Memory access pattern:** Sequential loads/stores not batched
5. **Control flow:** Select operations for conditionals add overhead
6. **Scratch space:** Suboptimal register allocation and reuse

### Resource Utilization

**Engine Usage per Element:**
- ALU: ~30-35 slots (could use up to 12/cycle)
- Load: 3 slots (can do 2/cycle)
- Store: 2 slots (can do 2/cycle)
- Flow: ~3-4 slots (1/cycle limit)
- VALU: 0 slots (6/cycle available, unused!)

**Theoretical Minimum:**
- With perfect SIMD: ~32× reduction (256 elements / 8 per vector)
- With VLIW packing: Additional 2-10× reduction
- Combined potential: 64-320× speedup possible

### Cycle Breakdown

Total operations for 16 rounds × 256 elements:
- 4096 iterations
- ~150 operations per iteration
- ~614,400 total operations
- 147734 cycles
- Average ~4.2 operations per cycle (very poor for VLIW architecture)

## Next Steps

**Priority 1: SIMD Vectorization**
- Replace scalar loads with `vload`
- Replace scalar stores with `vstore`
- Replace ALU operations with `valu` vector operations
- Process 8 elements per iteration instead of 1

**Priority 2: VLIW Instruction Packing**
- Identify independent operations that can run in parallel
- Pack multiple operations into single instruction bundles
- Maximize slot utilization per cycle

**Priority 3: Hash Function Optimization**
- Vectorize the 6-stage hash computation
- Explore instruction reordering to improve dependencies

## Correctness Verification

```bash
python perf_takehome.py Tests.test_kernel_cycles
```

Result: ✓ All tests pass

## Files Modified

- `perf_takehome.py` - Original baseline implementation
