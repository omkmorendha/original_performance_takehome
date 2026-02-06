# Iteration 008: Deep Broadcast Interleave + Init/Store Overlap

**Date:** 2026-02-06
**Status:** ✓ Success
**Cycles:** 2101
**Speedup:** 70.3x (1.08x over previous)
**Previous Best:** 2267 cycles (65.2x) — interleaved emit_compute_iter_g8

## Overview

Five targeted micro-optimizations that collectively save 166 cycles by eliminating redundant work and overlapping independent operations:

1. **Skip d0 vbroadcasts** (-16 cycles): XOR directly with pre-existing `v_tree[0]`
2. **Overlap final stores** (-31 cycles): Interleave ALU addr calc with vstore ops
3. **Batch init constants + interleave initial loads** (-45 cycles): Pre-batch offset constants; overlap vload with ALU addr calc
4. **Fuse d1 selection into compute** (-8 cycles): Overlap mask+madd tail with XOR+hash head
5. **Pipeline d1 transition loads** (-66 cycles): Overlap addr calc + scattered loads with d1 compute

## Hypothesis

The 2267-cycle kernel had multiple sections with idle engine slots:
- **d0 broadcast**: Wasted 2 cycles doing `vbroadcast` into `v_node[g]` when `v_tree[0]` already held the value
- **Final stores**: ALU and store engines alternated (never ran together)
- **Init loads**: Same ALU/load alternation pattern
- **d1 compute**: 4-cycle selection ran before 24-cycle compute, with no overlap
- **d1→scattered transitions**: 27-cycle d1 compute left ALU+load slots completely idle while addr calc + 64 scattered loads ran separately afterward

## Implementation Details

### Step 1: emit_compute_iter_d0_g8

New function identical to `emit_compute_iter_g8` but replaces `v_node[g]` with `v_tree[0]` in the XOR lines (Cy1-2). Main loop dispatches to this for `depth == 0`, skipping `emit_broadcast_tree_value` entirely.

### Step 2: Interleaved Final Stores

Replaced the sequential pattern:
```
[ALU pair0], [store pair0], [ALU pair1], [store pair1], ...  (64 cycles)
```
With overlapped pattern:
```
[ALU pair0], [store0 + ALU pair1], [store1 + ALU pair2], ..., [store last]  (33 cycles)
```

Uses the memory model guarantee (all reads before writes per cycle) to safely overwrite `tmp1/tmp2` with next addresses while the store reads current addresses.

### Step 3: Batched Init Constants + Interleaved Initial Loads

- Pre-created all 33 offset constants (0, 8, 16, ..., 256) in batches of 2 per load instruction (17 cycles instead of 33)
- Applied same interleave pattern as final stores to the 32 initial vload operations

### Step 4: emit_compute_iter_d1_g8

Fused function that overlaps d1 selection tail with compute head:
```
Cy1: mask g0-5 (& idx with v_one)
Cy2: mask g6-7 + madd g0-3 (selection)
Cy3: madd g4-7 + XOR g0-1 (overlap: selection finishes, compute starts)
Cy4: XOR g2-5 + H0 madd g0-1
Cy5: XOR g6-7 + H0 madd g2-5
Cy6-27: remaining hash + idx update
```
Total: 27 cycles (vs 4 + 24 = 28 separate)

### Step 5: emit_compute_iter_d1_pipelined_g8

Combines d1 selection + compute with addr calc + scattered loads for next round's first iteration. Same structure as Step 4 but adds `next_addr_calcs()` on ALU and `next_loads()` on load engine to every cycle. The 27-cycle compute window absorbs all 64 addr calcs (5.3 cycles of ALU) and most of the 64 scattered loads (32 cycles at 2/cycle).

Used for the 2 d1→scattered transitions (rounds 1→2 and 12→13).

## Results

### Performance
```
Cycles: 2101
Speedup vs baseline: 70.3x
Speedup vs previous (2267): 1.08x
```

### Step-by-Step Breakdown
| Step | Change | Savings | Running Cycles |
|------|--------|---------|----------------|
| Start | Interleaved compute (uncommitted) | - | 2267 |
| Step 1 | Skip d0 vbroadcasts | 16 | 2251 |
| Step 2 | Overlap final stores | 31 | 2220 |
| Step 3 | Batch init + interleave loads | 45 | 2175 |
| Step 4 | Fuse d1 selection+compute | 8 | 2167 |
| Step 5 | Pipeline d1 transition loads | 66 | 2101 |
| **Total** | | **166** | **2101** |

### Correctness
```bash
python perf_takehome.py Tests.test_kernel_cycles    # PASS (2101 cycles)
python tests/submission_tests.py                     # Passes correctness + Tier 2
git diff origin/main tests/                          # Empty (tests unchanged)
```

Result: ✓ Pass — achieves Tier 2 (< 2164 cycles)

### Analysis

**What worked well:**
- All 5 savings predictions were accurate (within 6 cycles total)
- The interleave pattern (store+ALU / load+ALU overlap) is a reliable technique
- Fusing d1 selection with compute was clean — no dependency issues
- Pipelined d1 transition was the biggest single win (66 cycles from just 2 transitions)

**Remaining bottlenecks:**
1. VALU throughput floor: ~1536 cycles (9216 ops ÷ 6/cycle)
2. Load overshoot in pipelined iters: 9 extra cycles per iter (32 loads - 24 VALU cycles)
3. Scattered round prologue: addr calc + loads before first iter ~38 cycles × 12 rounds
4. Non-pipelined compute: 24 cycles (d0) or 27 cycles (d1) vs 24 pipelined

## Lessons Learned

1. **Engine overlap is cumulative**: Small per-cycle savings compound across many repetitions (31 cycles from just adding 2 ALU ops per store cycle)
2. **Transition points are high-value targets**: The d1→scattered transitions were expensive (67 cycles each) with most engine slots idle — perfect for pipelining
3. **Predictions matched reality**: Careful cycle counting during planning yielded accurate estimates, enabling prioritization without trial-and-error
4. **Batching constants matters**: 33 individual const instructions → 17 paired ones saved 16 cycles in init alone

## Next Steps

To reach Tier 3 (< 1790 cycles, need ~15% improvement from 2101):

1. **Pipeline d0→d1 transition**: d0 last iter (24cy) has idle ALU+load — could precompute d1 masks
2. **Overlap scattered prologue with previous round's last iter**: Current last iter of each round uses only VALU; could start next round's addr calcs on ALU
3. **G=4 alternative**: 15 VALU cycles/iter perfectly balances with 16 load cycles, eliminating 9-cycle load overshoot per pipelined iter. 128 iters × 16 ≈ 2048 base + broadcast savings
4. **Reduce idx update stalls**: Currently 7 VALU cycles in idx update; overlapping cmp/wrap with hash tail could save 2-3 cycles
5. **Further d1 fusion**: emit_compute_iter_d1_g8 Cy11 has only 4 VALU ops — could pack more

## References

- Previous: 007 (G=8 scratch-resident + broadcast, 2321cy)
- Key commits: `3421169` (007 docs), this commit (008 implementation)
- Branch: `opt/008-g4-deep-broadcast`
