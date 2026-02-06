# Optimization 006: Depth-Aware Broadcast Implementation Plan

## Overview

**Current**: 3664 cycles (40.3x speedup)
**Target**: <1487 cycles (99x speedup)
**Gap**: ~2.5x improvement needed

## Critical Insight

The target of <1487 cycles is **below** the theoretical minimum of 2048 cycles achievable with scattered loads:
- 256 elements × 16 rounds = 4096 tree accesses
- 2 loads/cycle maximum → 2048 cycles minimum

**Conclusion**: The target can ONLY be achieved by eliminating most scattered tree value loads.

## The Solution: Depth-Aware Broadcast

### Tree Access Pattern Analysis

All 256 batch elements start at index 0 (root). The tree traversal follows a predictable depth pattern:

| Round | Depth | Tree Indices | # Values | Broadcast Viable? |
|-------|-------|--------------|----------|-------------------|
| 0 | 0 | 0 | 1 | ✓ Trivial |
| 1 | 1 | 1-2 | 2 | ✓ Easy |
| 2 | 2 | 3-6 | 4 | ✓ Easy |
| 3 | 3 | 7-14 | 8 | ✓ Moderate |
| 4 | 4 | 15-30 | 16 | ✓ Moderate |
| 5 | 5 | 31-62 | 32 | ✗ Too many |
| 6 | 6 | 63-126 | 64 | ✗ Too many |
| 7 | 7 | 127-254 | 128 | ✗ Too many |
| 8 | 8 | 255-510 | 256 | ✗ Too many |
| 9 | 9 | 511-1022 | 512 | ✗ Too many |
| 10 | 10 | 1023-2046 | 1024 | ✗ Too many |
| 11 | 0 | 0 (wrap) | 1 | ✓ Trivial |
| 12 | 1 | 1-2 | 2 | ✓ Easy |
| 13 | 2 | 3-6 | 4 | ✓ Easy |
| 14 | 3 | 7-14 | 8 | ✓ Moderate |
| 15 | 4 | 15-30 | 16 | ✓ Moderate |

**Summary**: 10 of 16 rounds can use broadcast (depths 0-4)

### Cycle Savings Estimate

**Scattered load rounds (6 rounds: depths 5-10)**:
- 8 iterations × 16 scattered loads × 6 rounds = 768 load cycles
- Plus compute overlap = ~900 cycles total

**Broadcast rounds (10 rounds: depths 0-4)**:
- No scattered loads needed
- Preload overhead: ~20 cycles (31 values)
- Selection overhead per round: ~5-15 cycles depending on depth
- Estimated: ~200-300 cycles for all 10 rounds

**Projected total**: 900 + 300 = ~1200 cycles (achieves target!)

## Implementation Strategy

### Phase 1: Preload Tree Values (One-time cost)

At kernel start, preload tree values for depths 0-4:
```
Depth 0: tree[0]        → 1 value   (tree_d0)
Depth 1: tree[1:3]      → 2 values  (tree_d1_0, tree_d1_1)
Depth 2: tree[3:7]      → 4 values  (tree_d2_0..3)
Depth 3: tree[7:15]     → 8 values  (tree_d3_0..7)
Depth 4: tree[15:31]    → 16 values (tree_d4_0..15)
Total: 31 scalar values
```

**Optimized loading** (batch address computation + loads):
```python
# Compute all 31 addresses in batches of 12 ALU ops
for batch in range(0, 31, 12):
    alu_ops = [("+", addr[i], forest_values_p, const_i) for i in batch]
    emit({"alu": alu_ops})
    # Load in pairs
    for j in range(0, len(batch), 2):
        emit({"load": [("load", tree_cache[i], addr[j]), ...]})
```
Cost: ~3 ALU cycles + 16 load cycles = ~19 cycles

### Phase 2: Arithmetic Selection (NOT vselect!)

**Critical**: `vselect` uses the flow engine (1 slot/cycle limit). Use VALU arithmetic instead:

```
# Instead of vselect(mask, a, b):
result = (a - b) * mask + b   # 3 VALU ops, 6 slots/cycle
# Or:
result = b ^ ((a ^ b) & mask) # 3 VALU ops if mask is 0/1
```

### Phase 3: Depth-Specific Selection Functions

**Depth 0** (1 value - tree[0]):
```python
def get_tree_value_d0(v_idx):
    # All elements access tree[0]
    return vbroadcast(tree_d0)  # 1 VALU op
```

**Depth 1** (2 values - tree[1], tree[2]):
```python
def get_tree_value_d1(v_idx):
    # idx is 1 or 2
    # tree[1] if idx=1, tree[2] if idx=2
    # Selection: idx & 1 == 1 means tree[1], else tree[2]
    v_a = vbroadcast(tree_d1_0)  # tree[1]
    v_b = vbroadcast(tree_d1_1)  # tree[2]
    mask = v_idx & v_one         # bit 0
    mask = 1 - mask              # invert: 1→0, 0→1
    return (v_a - v_b) * mask + v_b  # 5 VALU ops
```

**Depth 2** (4 values - tree[3:7]):
```python
def get_tree_value_d2(v_idx):
    # idx is in range [3, 6]
    # tree index = idx (directly)
    # Need to select from 4 options based on bits 0-1 of idx

    # First level: select based on bit 0
    mask0 = v_idx & v_one
    pair_01 = (v_d2_0 - v_d2_1) * mask0 + v_d2_1  # tree[3] or tree[4]
    pair_23 = (v_d2_2 - v_d2_3) * mask0 + v_d2_3  # tree[5] or tree[6]

    # Second level: select based on bit 1
    mask1 = (v_idx >> 1) & v_one
    return (pair_01 - pair_23) * mask1 + pair_23
    # ~10 VALU ops
```

**General pattern for depth D**:
- D levels of selection
- Each level: ~3 VALU ops
- Total: ~3D VALU ops

### Phase 4: Round-Specific Code Paths

```python
def get_round_depth(round_idx, forest_height):
    return round_idx % (forest_height + 1)

# In main loop:
for round_idx in range(rounds):
    depth = get_round_depth(round_idx, forest_height)

    if depth <= 4:
        # Broadcast round - no scattered loads
        emit_broadcast_round(depth)
    else:
        # Scattered load round - use existing pipelined code
        emit_scattered_round()
```

### Phase 5: Broadcast Round Implementation

```python
def emit_broadcast_round(depth, v_tree_getter):
    """
    Process a round using broadcast tree values instead of scattered loads.

    Key difference from scattered round:
    - No node address computation needed
    - No scattered loads needed
    - Use v_tree_getter to get tree values via arithmetic selection
    """
    for iteration in range(n_iters):
        # Load v_idx and v_val from memory (still needed)
        emit_vload_idx_val(...)

        # Get tree value via broadcast + selection (NO scattered loads!)
        v_node = v_tree_getter(v_idx, depth)

        # XOR, hash, index update, wrap, store (same as before)
        emit_compute_and_store(...)
```

## Scratch Space Budget

Current usage with G=4: ~540 words

Additional for broadcast:
- tree_cache: 31 words
- temp addresses: 16 words
- v_tree broadcasts (reusable): 8 words per depth level = ~40 words

Total: ~627 words (well under 1536 limit)

## Implementation Order

### Step 1: Infrastructure (est. 50 cycles overhead)
1. Allocate tree_cache scratch space
2. Implement optimized tree value preloading
3. Create depth-to-tree-cache index mapping

### Step 2: Depth 0 Broadcast (simplest case)
1. Create `emit_broadcast_load_phase_d0()` - no scattered loads
2. Create `emit_broadcast_compute_d0()` - use vbroadcast(tree[0])
3. Test with rounds 0 and 11

### Step 3: Depth 1 Broadcast
1. Create arithmetic selection for 2 values
2. Test with rounds 1 and 12

### Step 4: Depths 2-4 Broadcast
1. Generalize selection for 4, 8, 16 values
2. Test with all broadcast rounds

### Step 5: Integration
1. Create round dispatcher that chooses broadcast vs scattered
2. Handle round transitions (broadcast→scattered, scattered→broadcast)
3. Full integration test

### Step 6: Optimization
1. Pack selection operations into VLIW bundles
2. Overlap selection with other operations
3. Fine-tune cycle count

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Selection overhead exceeds savings | High | Benchmark each depth level separately |
| Buffer management bugs | High | Keep G=4, don't change buffer structure |
| Incorrect index-to-depth mapping | High | Add debug asserts, compare with reference |
| Scratch space overflow | Low | Budget is well under limit |

## Success Criteria

- [ ] Correctness: All submission tests pass
- [ ] tests/ folder unchanged: `git diff origin/main tests/` is empty
- [ ] Cycles < 2000 (below scattered-load minimum)
- [ ] Cycles < 1487 (target achieved)

## Files to Modify

Only `perf_takehome.py`:
- Add tree cache allocation and loading (~lines 175-195)
- Add depth-specific tree value getter functions
- Add `emit_broadcast_*` functions
- Modify main loop to dispatch based on depth

## Testing Strategy

1. After each step, run: `python perf_takehome.py Tests.test_kernel_cycles`
2. Periodically run: `python tests/submission_tests.py`
3. Always verify: `git diff origin/main tests/` is empty
