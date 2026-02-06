# Optimization Plan: 2101 → Target <1487 cycles

## Current Cycle Breakdown (2101 total)
- Init: 79 cycles
- d0 broadcast (rounds 0,11): 192 cycles (8 iters × 24 cy)
- d1 broadcast (rounds 1,12): 230 cycles (6×27 + 2×34 pipelined)
- Pipelined scattered (46 iters): 1518 cycles (46 × 33 cy)
- Compute-only scattered (2 iters): 48 cycles (2 × 24 cy)
- Store epilogue: 34 cycles

## Fundamental Limits
- VALU floor: 9216 ops / 6 per cycle = 1536 cycles
- LOAD floor (scattered): 3072 loads / 2 per cycle = 1536 cycles
- These overlap during pipelined iterations

## Phase 1: Pre-load tree values during broadcast rounds (est. -72 cycles)

**Key insight**: During d1 rounds (108 VALU cycles), the LOAD engine is completely idle. We can:
1. As each group's idx update completes during d1 VALU, compute next round's tree addresses on ALU
2. Start scattered loads on the LOAD engine (which is idle during broadcast)
3. By the time d1 ends, ~166 of 256 tree values are pre-loaded
4. The next scattered round only needs 90 more loads (45 cy), fully hidden within its 96 VALU cycles

**Savings**: Each d1→scattered transition saves ~36 cycles (132→96). Two transitions = 72 cycles.

**Implementation**: Modify `emit_compute_iter_d1_g8` and `emit_compute_iter_d1_pipelined_g8` to emit ALU addr calcs + loads in the latter iterations of d1 rounds.

## Phase 2: Pre-load during d0 (est. -8 cycles)

During d0 rounds, the load engine is also idle. We can pre-compute d1's broadcast tree values (just 3 loads: tree[0], tree[1], tree[2]) and do other preparatory work.

## Phase 3: Init optimization (est. -10 cycles)

Batch scalar constant creation (11 separate loads → 6 paired loads), overlap vbroadcasts with offset creation, merge const loads with header loads.

## Phase 4: Store overlap (est. -8 cycles)

Begin computing store addresses and executing vstores during the last compute iteration's idx update phase (VALU busy, but ALU and STORE idle).

## Phase 5: Pause elimination (est. -2 cycles)

Replace pause instructions with useful work or remove them. Submission tests set `enable_pause=False`.

## Phase 6: Further transition optimization (est. -20 cycles)

Better overlap at scattered→broadcast and broadcast→scattered transitions.

## Expected Result
2101 - 72 - 8 - 10 - 8 - 2 - 20 = ~1981 cycles (Tier 2, approaching Tier 3)

## Path to Tier 3+ (<1790)
Requires more aggressive approaches:
- d2 broadcast with VALU selection (7 extra ops/group, eliminates 2 scattered rounds)
- Deeper cross-round pipelining
- Hybrid ALU/VALU processing

## Implementation Order
1. Phase 1 (pre-load during d1) - highest impact
2. Phase 3 (init optimization) - low risk
3. Phase 5 (pause elimination) - trivial
4. Phase 4 (store overlap) - moderate complexity
5. Phase 2 (pre-load during d0) - moderate impact
6. Phase 6 (transition optimization) - moderate complexity
