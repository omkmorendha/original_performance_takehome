# Optimization 009: Micro-optimizations (Init Overlap + Store Prep)

**Date**: 2026-02-06
**Status**: ✓ Completed
**Result**: 2097 cycles (70.5x speedup, Tier 2)
**Improvement**: -4 cycles from 2101

## Summary

Applied two micro-optimizations to reduce overhead in init and store phases:
1. **Init overlap**: Overlapped vbroadcast operations (VALU) with offset constant creation (LOAD)
2. **Store prep**: Pre-computed first store addresses during last compute iteration's idle ALU cycles

## Hypothesis

The init and store phases have single-engine instructions that waste cycles. By better utilizing idle execution engines during these phases, we can reduce the total cycle count without affecting correctness.

## Implementation

### 1. Init Overlap (-3 cycles)

**Before**: Sequential execution
```python
# Emit all vbroadcasts (VALU only)
for i in range(0, len(all_vbc), 6):
    self.instrs.append({"valu": all_vbc[i:i+6]})

# Create offset constants (LOAD only)
for i in range(0, len(new_consts), 2):
    self.instrs.append({"load": [const ops]})
```

**After**: Interleaved execution
```python
# Interleave vbroadcast (VALU) with offset constant creation (LOAD)
while vbc_idx < len(all_vbc) or const_idx < len(new_offset_consts):
    instr = {}
    if vbc_idx < len(all_vbc):
        instr["valu"] = all_vbc[vbc_idx:vbc_idx+6]
    if const_idx < len(new_offset_consts):
        instr["load"] = [up to 2 const ops]
    self.instrs.append(instr)
```

**Savings**: 3 cycles (from better engine utilization)

### 2. Store Prep (-1 cycle)

**Before**: Compute first store addresses at start of store epilogue
```python
# Store epilogue begins
self.instrs.append({"alu": [compute addr1, addr2]})  # 1 cycle
self.instrs.append({"store": [...], "alu": [...]})    # interleaved stores
```

**After**: Pre-compute during last compute iteration
```python
# Last compute iteration (24 VALU cycles, ALU idle)
def emit_compute_iter_g8(it, prep_store=False):
    instr1 = {"valu": [...]}
    if prep_store:
        instr1["alu"] = [compute store_addr1, store_addr2]
    self.instrs.append(instr1)

# Store epilogue uses pre-computed addresses
self.instrs.append({"store": [use store_addr1/2], "alu": [next addrs]})
```

**Savings**: 1 cycle (eliminated initial ALU-only cycle in store epilogue)

## Results

### Performance
- **Before**: 2101 cycles
- **After**: 2097 cycles
- **Improvement**: -4 cycles (0.19% reduction)
- **Speedup**: 70.5x over baseline (147734 cycles)

### Validation
```bash
$ python perf_takehome.py Tests.test_kernel_cycles
CYCLES:  2097

$ python tests/submission_tests.py
✓ All correctness tests pass
✓ Tier 1 (<18,532): PASS
✓ Tier 2 (<2,164): PASS
✗ Tier 3 (<1,790): FAIL (need -307 cycles)
```

## Analysis

### Why These Optimizations Work

Both optimizations exploit **idle execution engines**:
- Init phase: VALU busy with vbroadcast → LOAD idle → overlap const creation
- Last compute: VALU busy with hash → ALU idle → pre-compute store addresses

This is the essence of VLIW optimization: pack independent operations into single cycles.

### Why Impact is Small

The small 4-cycle improvement reveals we're near architectural limits:
- **Perfect VLIW packing**: 2097 instructions = 2097 cycles (1:1 ratio)
- **Single-engine instructions**: 881 (42%) - mostly unavoidable
  - 407 VALU-only (broadcast rounds, inherently sequential)
  - 468 LOAD-only (scattered loads, data dependencies)
- **Already optimal**: Most opportunities already exploited in earlier iterations

### Remaining Opportunities (Estimated)

1. **Init/store micro-opts**: ~5-10 cycles remaining
2. **Pipelined iteration tightening**: 33cy → 32cy theoretical = ~48 cycles
3. **Other micro-opts**: ~5-10 cycles

**Total realistic potential**: ~60 cycles → 2037 cycles (still 247 above Tier 3)

## Key Learnings

### 1. Architectural Limits Are Real
- VALU floor: ~1536 cycles (hash operations / 6 per cycle)
- LOAD floor: ~1536 cycles (loads / 2 per cycle)
- Current overhead: 561 cycles (2097 - 1536)
- To reach Tier 3 (1790): overhead must be only 254 cycles

### 2. Broadcast Extension is Counterproductive
Analysis of depth-2 broadcast:
- Selection overhead: ~14 VALU cycles per iteration
- Full round: 4 × (14 + 24) = 152 cycles
- Current scattered (with d1 pipelining): 123 cycles
- **Net effect**: +29 cycles per round (SLOWER!)

Depth 3-4 would be exponentially worse.

### 3. Near-Optimal Performance Achieved
Current 2097 cycles represents **near-optimal** performance for this architecture:
- Pipelined scattered: 33 cy (vs 32 cy theoretical, only 1 cy overhead)
- Broadcast rounds: 24-27 cy (optimal)
- Init/store: highly optimized with interleaving

## Conclusion

**Tier 2 (70.5x) achieved and sustained.**

**Tier 3 (<1790) assessment**: Not achievable through instruction-level optimization alone. Would require:
1. **Algorithmic changes**: Different hash implementation
2. **Architectural breakthrough**: Undiscovered simulator optimization
3. **Problem redefinition**: Modified traversal or hash algorithm

The gap of 307 cycles (15%) cannot be closed with conventional VLIW optimization techniques.

## Files Changed
- `perf_takehome.py`: Init overlap + store prep implementation

## Next Steps
- Document final architecture analysis
- Explore any remaining architectural features
- Consider if alternate algorithmic approaches exist
