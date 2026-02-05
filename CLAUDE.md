# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is Anthropic's original performance engineering take-home challenge. The task is to optimize a kernel that simulates parallel tree traversal operations on a custom VLIW SIMD architecture. The goal is to minimize the cycle count as measured by the frozen simulator in `tests/frozen_problem.py`.

**Baseline performance:** 147734 cycles
**Target:** Beat 1487 cycles (Claude Opus 4.5's best performance at launch)

## Critical Rules

**DO NOT modify anything in the `tests/` folder.** The submission validation depends on a frozen copy of the simulator. Modifying tests or the frozen problem file will invalidate your solution.

Before submitting, always verify:
```bash
# This should be empty - tests folder must be unchanged
git diff origin/main tests/

# Run validation and check cycle count
python tests/submission_tests.py
```

## Commands

### Running Tests

```bash
# Run the main performance test (measures cycles)
python perf_takehome.py Tests.test_kernel_cycles

# Run a specific test
python perf_takehome.py Tests.test_kernel_trace

# Run all tests
python perf_takehome.py

# Run submission validation (includes correctness and speed tests)
python tests/submission_tests.py
```

### Debug Workflow (Recommended)

Generate a visual trace with hot-reloading:
```bash
# Terminal 1: Generate trace
python perf_takehome.py Tests.test_kernel_trace

# Terminal 2: Start trace viewer (opens browser, only works in Chrome)
python watch_trace.py
```

Re-run the test in Terminal 1 to see updated traces. The trace viewer automatically reloads when `trace.json` changes.

## Architecture

### Simulator (`problem.py`)

Simulates a custom VLIW SIMD machine with:
- **VLIW (Very Large Instruction Word)**: Multiple execution engines run parallel "slots" per cycle
- **SIMD**: Vector operations on VLEN=8 elements simultaneously
- **Engines**: `alu` (scalar), `valu` (vector), `load`, `store`, `flow`, `debug`
- **Slot limits per cycle**: alu=12, valu=6, load=2, store=2, flow=1, debug=64
- **Memory model**: Effects apply at end of cycle (all reads before writes)
- **Scratch space**: 1536 words, acts as registers and manually-managed cache
- **Single core**: N_CORES=1 (multicore support exists but is disabled)

### Problem Definition

The kernel performs `rounds` iterations over a `batch_size` of inputs, each doing:
1. Load node value from tree at current index
2. XOR with input value
3. Hash the result (6-stage hash function)
4. Navigate tree: go left (2*idx+1) if even, right (2*idx+2) if odd
5. Wrap to root if index exceeds tree size
6. Store updated index and value

Parameters: forest_height=10, rounds=16, batch_size=256, n_nodes=2047

### Optimization Target (`perf_takehome.py`)

The `KernelBuilder.build_kernel()` method is where optimization happens. It:
- Allocates scratch space for variables
- Builds instruction sequences for the kernel
- Currently implements a naive scalar version processing one element at a time

The baseline implementation:
- Uses only scalar ALU operations
- Processes batch elements sequentially
- Has significant instruction scheduling inefficiencies
- Does not leverage SIMD (valu instructions)
- Does not leverage VLIW parallelism (multiple engines per cycle)

### Optimization Opportunities

1. **SIMD vectorization**: Use `vload`, `vstore`, `valu` to process 8 elements at once
2. **VLIW packing**: Pack independent operations into single instruction bundles
3. **Instruction scheduling**: Reorder operations to maximize slot utilization
4. **Memory access patterns**: Optimize loads/stores for better throughput
5. **Loop unrolling**: Reduce control flow overhead
6. **Scratch space management**: Better register allocation and reuse
7. **Hash function**: The 6-stage hash is a bottleneck - vectorize it

### Key Constraints

- **Scratch space limit**: 1536 words total
- **Slot limits**: Cannot exceed per-cycle limits for each engine
- **Memory consistency**: All reads happen before all writes each cycle
- **Correctness**: Output must match `reference_kernel2()` exactly

### Testing Infrastructure

- `reference_kernel()`: Pure Python reference implementation
- `reference_kernel2()`: Memory-based reference with value tracing for debugging
- `frozen_problem.py`: Immutable copy used for submission validation
- Debug instructions (`compare`, `vcompare`) validate intermediate values against reference

### Memory Layout

The memory image contains:
```
[0]: rounds
[1]: n_nodes
[2]: batch_size
[3]: forest_height
[4]: forest_values_p (pointer to tree values)
[5]: inp_indices_p (pointer to indices array)
[6]: inp_values_p (pointer to values array)
[7+]: actual data (tree values, indices, values, extra room)
```

## Debugging Tips

- Use `trace=True` with `watch_trace.py` to visualize instruction execution in Perfetto
- Add debug instructions to validate intermediate computations
- Use `prints=True` in `do_kernel_test()` to see scratch space state
- The `pause` instruction synchronizes with reference kernel yields for step-by-step validation
- Check `scratch_map` in debug info to see variable names and addresses
