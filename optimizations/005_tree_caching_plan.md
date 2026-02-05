# Iteration 005: Tree Value Caching

**Date:** 2026-02-05
**Status:** PLANNED
**Target Cycles:** ~2500-3000 (estimated)
**Previous Best:** 3940 cycles (37.5x)

## Problem Analysis

### Current Bottleneck
The kernel is **memory-bound** by scattered tree value loads:
- 256 elements × 16 rounds = 4096 tree value loads
- Load bandwidth: 2 ops/cycle
- Theoretical minimum: 4096 / 2 = 2048 cycles just for tree loads
- Current: 3940 cycles (leaving ~1900 cycles for everything else)

### Key Insight
The tree is a perfect binary tree with predictable access patterns:
- All elements start at index 0 (root)
- After round 1: indices are 1 or 2 (depth 1)
- After round k: indices are in depths 0 to k (with some wrapping to 0)
- Upper tree levels are accessed more frequently

### Scratch Space Analysis
```
Current usage: ~540 words
Available: 1536 words
Remaining: ~996 words
```

Tree structure (forest_height=10, n_nodes=2047):
| Depth | Nodes | Cumulative | Index Range |
|-------|-------|------------|-------------|
| 0     | 1     | 1          | 0           |
| 1     | 2     | 3          | 1-2         |
| 2     | 4     | 7          | 3-6         |
| 3     | 8     | 15         | 7-14        |
| 4     | 16    | 31         | 15-30       |
| 5     | 32    | 63         | 31-62       |
| 6     | 64    | 127        | 63-126      |
| 7     | 128   | 255        | 127-254     |
| 8     | 256   | 511        | 255-510     |
| 9     | 512   | 1023       | 511-1022    |
| 10    | 1024  | 2047       | 1023-2046   |

**We can cache depths 0-8 (511 nodes) in scratch**, leaving ~485 words for other uses.

## Proposed Approach

### Strategy: Conditional Cache Lookup

For each tree value load, determine if the index is in the cached range:
1. If `idx < 511`: load from scratch cache (no memory access needed)
2. If `idx >= 511`: load from memory (scattered load)

### Implementation Technique

Since we can't branch efficiently, use arithmetic masking:

```python
# For each element's tree value load:
cache_mask = (idx < cache_size)  # 1 if cached, 0 if not
mem_mask = 1 - cache_mask        # 0 if cached, 1 if not

# Load from cache (always executes, but result zeroed if not cached)
cached_val = scratch[cache_base + idx] * cache_mask

# Load from memory (still need scattered load for uncached)
# But can skip if ALL elements in vector are cached
mem_val = mem[tree_p + idx] * mem_mask

# Combine
tree_val = cached_val + mem_val
```

### Optimization: Skip Scattered Loads When Possible

Key insight: In early rounds, ALL elements have indices < 511 (depths 0-8).
- Round 1: All at depth 1 (indices 1-2) → ALL CACHED
- Round 2: All at depth 2 (indices 3-6) → ALL CACHED
- ...
- Round 8: All at depth 8 (indices 255-510) → ALL CACHED
- Round 9+: Some may be at depth 9+ (indices 511+) → MIXED

For rounds 1-8, we can use vload from a computed address in scratch instead of scattered loads!

### Two-Phase Approach

**Phase 1: Rounds 1-8 (guaranteed cached)**
- All indices < 511
- Use scratch-based vector loads where possible
- Compute: `scratch_addr = cache_base + idx[i]` for each element
- Since indices are contiguous within depth levels, may be able to use vload

**Phase 2: Rounds 9-16 (mixed cached/uncached)**
- Some indices may be >= 511
- Use conditional loading with masks
- Fall back to scattered loads for uncached elements

### Detailed Implementation Plan

#### Step 1: Initialize Tree Cache at Startup
```python
# Allocate cache in scratch
cache_base = self.alloc_scratch("tree_cache", 511)

# Load tree values 0-510 into cache
# This is 511 loads at 2/cycle = 256 cycles one-time cost
for i in range(0, 511, 2):
    # Load tree[i] and tree[i+1] into cache
    self.instrs.append({"load": [
        ("load", cache_base + i, tree_addr_i),
        ("load", cache_base + i + 1, tree_addr_i_plus_1),
    ]})
```

#### Step 2: Modify Tree Value Loading

**For pipelined iterations (most common case):**
```python
# Current: 32 scattered loads for tree values
# New: Check if we can use cache

# Compute cache hit mask for each element
# v_cache_hit[g] = v_idx[g] < 511 (vector compare)

# For elements with cache hits:
#   tree_val = scratch[cache_base + idx]
# For elements without:
#   tree_val = mem[tree_p + idx] (scattered load)
```

#### Step 3: Optimize Early Rounds

For rounds 1-8, all elements are guaranteed to be in cache (depth <= 8):
- Skip the scattered load path entirely
- Use direct scratch access

### Cycle Savings Estimate

**Current scattered loads:**
- 32 per iteration × 8 iterations × 16 rounds = 4096 loads
- At 2/cycle = 2048 cycles

**With caching (conservative estimate):**
- Rounds 1-8: ~0 scattered loads (all cached)
- Rounds 9-16: ~50% cache hits (estimated)
  - 32 × 8 × 8 × 0.5 = 1024 scattered loads
- Total: 1024 loads = 512 cycles

**Savings: ~1536 cycles**

**One-time cache init cost: ~256 cycles**

**Net savings: ~1280 cycles**

**Projected result: 3940 - 1280 = ~2660 cycles**

### Risks and Challenges

1. **Scratch space pressure**: Cache takes 511 words, may need to reduce other allocations
2. **Conditional logic overhead**: Mask computation adds cycles
3. **Complex address calculation**: Computing scratch addresses for cached values
4. **Code complexity**: Two code paths (cached vs uncached)

### Alternative: Simpler Depth-Based Optimization

Instead of per-element caching, exploit the fact that early rounds are fully cached:

**Unroll rounds into two phases:**
1. **Rounds 1-8**: Use scratch-based tree access (no scattered loads)
2. **Rounds 9-16**: Use current scattered load approach

This is simpler but may have less benefit if elements wrap to root frequently.

### Validation Plan

1. Verify correctness with `python tests/submission_tests.py`
2. Compare output against reference kernel
3. Test edge cases (elements that wrap multiple times)
4. Verify `git diff origin/main tests/` is empty

### Success Criteria

- [ ] Cycles < 3000 (30%+ improvement)
- [ ] All correctness tests pass
- [ ] No modifications to tests/ folder
- [ ] Clear documentation of approach

## Next Steps After This Iteration

If caching succeeds, further optimizations could include:
1. **Larger cache**: Cache depths 0-9 (1023 nodes) if scratch allows
2. **Prefetching**: Start loading next iteration's tree values earlier
3. **Better pipelining**: Overlap cache lookups with computation
4. **Instruction packing**: Fill unused ALU slots during cache operations
