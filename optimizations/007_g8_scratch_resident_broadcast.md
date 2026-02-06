# Iteration 007: G=8 Scratch-Resident + Broadcast d0-d1

**Date:** 2026-02-06
**Status:** ✓ Success
**Cycles:** 2321
**Speedup:** 63.7x (1.58x over previous)
**Previous Best:** 3664 cycles (40.3x)

## Overview

Major architectural overhaul combining three techniques:
1. **G=8 processing**: 64 elements (8 VLEN-8 groups) per iteration → 4 iters/round
2. **Scratch-resident storage**: All 256 idx and 256 val values live in scratch across all rounds (no intermediate memory loads/stores between rounds)
3. **Depth-aware broadcast**: For depths 0-1, tree values are computed via broadcast/arithmetic selection instead of scattered loads
4. **multiply_add hash collapse**: Stages 0, 2, 4 collapsed from 3 ops to 1 multiply_add each
5. **Interleaved hash stages**: Non-collapsible stages overlap op2 with next stage start

## Hypothesis

### G=8 (up from G=4)
With G=4, pipelined iterations took ~16 VALU cycles but 16 load cycles — loads were barely hidden. By doubling to G=8, we get 24 VALU cycles of compute which provides a longer window for the 32 load cycles. Though loads overshoot by ~9 cycles, the 4x reduction in iterations per round (8→4) eliminates per-iteration overhead.

### Scratch-Resident
Previous iterations loaded idx/val from memory at the start of each round and stored at the end. With 256 elements × 2 arrays × 16 rounds = 8192 memory accesses eliminated. All idx/val live in scratch (512 words of 1536 total).

### Depth-Aware Broadcast
Rounds at depth 0 access tree[0] (1 value) — can use vbroadcast.
Rounds at depth 1 access tree[1] or tree[2] — can use multiply_add selection.
This eliminates 64 scattered loads per iteration for 4 of 16 rounds.

### multiply_add Hash Collapse
Hash stages where op1="+", op2="+", op3="<<":
- Stage 0: `val + 0x7ED55D16 + (val << 12)` = `multiply_add(val, 4097, 0x7ED55D16)`
- Stage 2: `val + 0x165667B1 + (val << 5)` = `multiply_add(val, 33, 0x165667B1)`
- Stage 4: `val + 0xFD7046C5 + (val << 3)` = `multiply_add(val, 9, 0xFD7046C5)`

Each reduces from 3 VALU ops to 1 multiply_add (saves 2 ops × 8 groups = 16 VALU ops per stage).

## Implementation Details

### Scratch Space Layout (~896 words of 1536)
```
all_idx[32]:         256 words  (32 groups × VLEN)
all_val[32]:         256 words
v_node[8]:            64 words  (working tree values)
v_tmp1[8]:            64 words  (hash/idx temporaries)
v_tmp2[8]:            64 words
node_addrs[8][8]:     64 words  (current iter load addrs)
next_node_addrs[8][8]: 64 words  (next iter load addrs)
tree_cache[3]:         3 words  (scalar tree[0..2])
v_tree[3]:            24 words  (vector broadcasts)
v_tree_d1_diff:        8 words  (tree[1]-tree[2] vector)
Constants/temps:      ~50 words
Total:               ~917 words
```

### VALU Ops Per Iteration (G=8)
```
XOR:             8 ops  →  2 cycles (ceil(8/6))
Hash Stage 0:    8 madd →  2 cycles
Hash Stage 1:   24 ops  →  4 cycles (interleaved)
Hash Stage 2:    8 madd →  2 cycles
Hash Stage 3:   24 ops  →  4 cycles (interleaved)
Hash Stage 4:    8 madd →  2 cycles
Hash Stage 5:   24 ops  →  4 cycles (interleaved)
Idx update:
  madd+and:     16 ops  →  3 cycles
  add:           8 ops  →  2 cycles
  cmp:           8 ops  →  2 cycles
  wrap(mul):     8 ops  →  2 cycles
─────────────────────────────
Total:         144 ops  → 24 cycles (6 ops/cycle throughput)
                           +5 cycles dependency stalls in idx
                         = ~27 cycles non-pipelined
                           24 cycles pipelined (interleaved)
```

### Pipelined Iteration Structure (24 VALU cycles + loads)
```
Cy1:  XOR g0-5                    + addr calc (12 ALU)
Cy2:  XOR g6-7, H0 madd g0-3     + addr calc + first loads
Cy3:  H0 madd g4-7, H1 op13 g0   + addr calc + loads
Cy4:  H1 op13 g1-3               + addr calc + loads
Cy5:  H1 op13 g4-6               + addr calc + loads
Cy6:  H1 op13 g7, H1 op2 g0-3   + remaining addr + loads
Cy7:  H1 op2 g4-7, H2 madd g0-1 + loads
Cy8:  H2 madd g2-7               + loads
Cy9:  H3 op13 g0-2               + loads
Cy10: H3 op13 g3-5               + loads
Cy11: H3 op13 g6-7, H3 op2 g0-1 + loads
Cy12: H3 op2 g2-7                + loads
Cy13: H4 madd g0-5               + loads
Cy14: H4 madd g6-7, H5 op13 g0-1 + loads
Cy15: H5 op13 g2-4               + loads
Cy16: H5 op13 g5-7               + loads
Cy17: H5 op2 g0-5                + loads
Cy18: H5 op2 g6-7, idx madd+& g0-1 + loads
Cy19: idx madd+& g2-4            + loads
Cy20: idx madd+& g5-6, add g0-1  + loads
Cy21: idx madd+& g7, add g2-5    + loads
Cy22: add g6-7, cmp g0-3         + loads
Cy23: cmp g4-7, wrap g0-1        + loads
Cy24: wrap g2-7                   + remaining loads
Extra: ~9 remaining load cycles (load overshoot)
```

### Main Loop Structure (16 rounds × 4 iters)
```
Round 0  (d0, broadcast): 4× broadcast_d0 + compute_iter → transition load
Round 1  (d1, broadcast): 4× broadcast_d1 + compute_iter → transition load
Round 2  (d2, scattered): prologue load + compute + 2× pipelined + last compute
Round 3  (d3, scattered): prologue + 2× pipelined + last
...
Round 10 (d10, scattered): prologue + 2× pipelined + last
Round 11 (d0, broadcast): 4× broadcast → transition load
Round 12 (d1, broadcast): 4× broadcast → transition load
Round 13 (d2, scattered): prologue + 2× pipelined + last
Round 14 (d3, scattered): prologue + 2× pipelined + last
Round 15 (d4, scattered): prologue + 2× pipelined + final (no transition)
```

### Cycle Breakdown Estimate
```
Broadcast rounds (4 rounds × 4 iters = 16 iters):
  Selection: ~2-4 cycles per iter
  Compute (non-pipelined): ~27 cycles per iter
  Transition loads (end of broadcast): ~38 cycles × 3 transitions
  Subtotal: ~16 × 29 + 3 × 38 ≈ 578 cycles

Scattered rounds (12 rounds × 4 iters = 48 iters):
  Prologue (addr calc + loads + compute): ~65 cycles × 12 rounds
  Pipelined iters: 33 cycles × 24 iters
  Last iters (compute only): 27 cycles × 12
  Subtotal: 12 × 65 + 24 × 33 + 12 × 27 ≈ 1896 cycles

Init + final stores: ~100 cycles

Estimated total: 578 + 1896 + 100 ≈ 2574 (actual: 2321)
(Estimate is conservative; real code has some optimizations not captured)
```

## Bugs Found and Fixed

### Bug 1: Wrong idx update formula
- **Wrong**: `new_idx = 2*idx + 2 - (val&1)` (swapped left/right child)
- **Correct**: `new_idx = 2*idx + 1 + (val&1)` (from reference: even→left(+1), odd→right(+2))
- **Fix**: Changed `multiply_add(idx, 2, 2)` to `multiply_add(idx, 2, 1)` and `-` to `+`
- Had to fix in both `emit_idx_update_g8` and `emit_pipelined_iter_g8`

### Bug 2: Missing address calculations
- 64 addr calcs needed, only 5 × 12 = 60 emitted
- g7[4..7] addresses never computed, causing loads from stale addresses
- Fix: Added remaining 4 addr calcs in Cy6

### Bug 3: v_node stale for first pipelined iteration
- Prologue loads v_node for it0 and computes it0
- First pipelined iter (it1) XORs with v_node still containing it0's tree values
- Previous G=4 code used double buffering; G=8 scratch-resident has single v_node
- Fix: Added transition load after prologue to scatter-load v_node for it1

## Results

### Performance
```
Cycles: 2321
Speedup vs baseline: 63.7x
Speedup vs previous (3664): 1.58x
```

### Correctness
```bash
python perf_takehome.py Tests.test_kernel_cycles    # PASS
python tests/submission_tests.py                     # Passes correctness + speed through Tier 1
git diff origin/main tests/                          # Empty (tests unchanged)
```

## Analysis

### What's Working Well
- Scratch-resident eliminates all inter-round memory traffic (huge win)
- multiply_add hash collapse reduces VALU ops by 48 per iteration
- Interleaved hash keeps VALU pipeline full during pipelined iterations
- Broadcast for d0-d1 saves ~128 scattered loads (4 rounds × 4 iters × 32 loads... wait, no loads at all for broadcast)

### Remaining Bottlenecks
1. **VALU throughput floor**: 144 VALU ops/iter × 64 iters = 9216 ops / 6 per cycle = 1536 cycles minimum
2. **Load overshoot**: Pipelined iters take 33 cycles (24 VALU + 9 load overshoot)
3. **Non-pipelined compute**: 27 cycles vs 24 pipelined (dependency stalls in idx update)
4. **Prologue overhead**: Separate addr calc + loads for first iter of each scattered round
5. **Only depths 0-1 broadcast**: Depths 2-4 selection overhead exceeds load savings

### Theoretical Limits
```
VALU floor: 1536 cycles (all 9216 VALU ops at 6/cycle)
Load floor:  scattered only → 6 rounds × 4 iters × 32 loads / 2 per cycle = 384 cycles
             But loads overlap with VALU, so additional load cost = max(0, loads - VALU)
Practical floor: ~1536 + overhead ≈ ~1600-1700 cycles
```

## Lessons Learned

1. **G=8 is VALU-bound, not load-bound**: 24 VALU cycles vs 32 load cycles means 9 cycles of load overshoot per pipelined iter. G=4 (15 VALU, 16 loads) would be perfectly balanced.
2. **Broadcast selection is cheap for d0-1 but expensive for d2+**: d2 needs binary tree selection (2 levels × 3 ops × 4 groups), quickly exceeding the 9-cycle load savings.
3. **Buffer management complexity**: Single v_node with pipelining requires careful transition loads between phases.
4. **multiply_add is powerful**: Collapsing 3 ops to 1 for half the hash stages saves 48 VALU ops per iteration.

## Next Steps

To reach < 1487 cycles (need ~36% improvement from 2321):

1. **G=4 with fully interleaved hash**: 15 VALU cycles/iter, loads perfectly hidden in 16 cycles. 128 iters × 16 cycles = 2048 + overhead. Could be competitive.
2. **Deeper broadcast (depths 2-4)**: For G=4, selection overhead is smaller (4 groups vs 8). May be viable for d2 (4 values, 2-level selection in ~6 ops).
3. **Interleaved compute for non-pipelined iters**: Current emit_compute_iter_g8 takes 27 cycles due to dependency stalls. Interleaving XOR→H0→...→H5→idx could get to 24 cycles even without pipelining.
4. **Cross-iteration pipelining**: Pipeline across round boundaries to eliminate prologue overhead.
5. **Reduce idx update cost**: Currently 7 cycles; could try to overlap cmp/wrap with hash tail.

## References

- Previous: 004 (4-group), 005 (tree caching plan), 006 (broadcast plan)
- Key commits: `09fce79` (G=8 base, 2426cy), `87effc6` (+ broadcast, 2321cy)
