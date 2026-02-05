# Iteration 004: Eliminate Wrap vselect + 4-Group Processing

**Date:** 2026-02-05
**Status:** ✓ Success
**Cycles:** 4525
**Speedup:** 32.6x (1.25x over previous)
**Previous Best:** 5661 cycles (26.1x)

## Overview

Two optimizations combined:
1. Eliminate remaining vselect operations for index wrap using multiplication
2. Process 4 groups (32 elements) per iteration instead of 2 groups (16 elements)

## Hypothesis

### Part 1: Eliminate Wrap vselect

Current code still uses vselect for wrap check:
```python
cond = idx < n_nodes
idx = vselect(cond, idx, 0)  # FLOW - 1 slot/cycle limit
```

Mathematical transformation:
```python
cond = idx < n_nodes     # 0 or 1
idx = idx * cond         # idx*1=idx if in bounds, idx*0=0 if out of bounds
```

Savings: 2 vselect × 16 iters × 16 rounds = 512 cycles saved

### Part 2: 4-Group Processing

Current: 2 groups (16 elements) per iteration, using ~4 VALU slots
VALU limit: 6 slots/cycle

With 4 groups:
- Process 32 elements per iteration
- 8 iterations per round instead of 16
- Better amortization of loop overhead
- 32 scattered loads per iter but can overlap with compute

For batch_size=256 with VLEN=8:
- 32 total groups / 4 per iter = 8 iterations per round
- Hash: 6 stages × 3 cycles = 18 cycles compute
- 32 scattered loads at 2/cycle = 16 cycles load (can be overlapped)

## Implementation Details

### Changes
1. Replace vselect with multiply for wrap (in VALU)
2. Increase to 4 groups per iteration (32 elements)
3. Double buffering with 4 groups per buffer
4. Aggressive pipelining: overlap 32 scattered loads with hash computation
5. Hash stage 1 interleaved with node address calculation (32 ALU ops)

### Scratch Space
- v_idx, v_val, v_node, v_tmp1, v_tmp2: 2 buffers × 4 groups × 8 words = 320 words
- Node addresses: 2 buffers × 4 groups × 8 addresses = 64 words
- idx_addr, val_addr: 2 × 4 × 2 = 16 words
- Total: ~500+ words (well under 1536 limit)

## Results

### Performance
```
Cycles: 4525
Speedup vs baseline: 32.6x
Speedup vs previous: 1.25x (1136 cycles saved)
```

### Correctness
```bash
python perf_takehome.py Tests.test_kernel_cycles  # PASS
python tests/submission_tests.py  # Passes correctness + Tier 1
```

## Lessons Learned

1. **4 groups is optimal for this architecture**: Better than 2 groups (more throughput) and better than processing all 32 groups at once (too much memory pressure)
2. **Multiplication replaces vselect elegantly**: `idx = idx * (idx < n_nodes)` is cleaner than conditional selection
3. **Hash stages provide good overlap window**: 18 cycles of hash computation can hide most of the 16 cycles of scattered loads
4. **Loop overhead reduction matters**: Going from 16 to 8 iterations per round reduces control flow overhead

## Analysis

Cycle breakdown per 4-group iteration (32 elements):
- XOR: 1 cycle
- Hash: ~18 cycles (6 stages × 3 cycles)
- Index calc: ~5 cycles
- Wrap: 2 cycles
- Store: 4 cycles
- Total compute: ~30 cycles

With pipelining, most loads are hidden. Effective cycles: ~30-32 per iteration.
8 iterations × 16 rounds × 30 cycles ≈ 3840 cycles (close to actual 4525)
Overhead from prologue/epilogue per round: ~40 cycles × 16 rounds = 640 cycles

## Next Steps

To reach target (< 1487 cycles):
- Need ~3x more improvement
- Options:
  1. Deeper pipelining (triple buffering)
  2. Round-level fusion
  3. Further instruction scheduling optimization
  4. Investigate if hash can be shortened or parallelized more
