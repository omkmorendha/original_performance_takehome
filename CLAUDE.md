# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```
EDIT: perf_takehome.py → KernelBuilder.build_kernel() (lines 88-174)
TEST: python perf_takehome.py Tests.test_kernel_cycles
VALIDATE: python tests/submission_tests.py

Baseline: 147734 cycles → Target: <1487 cycles (99% reduction)
```

**Golden Rule:** NEVER modify `tests/` folder. Verify with `git diff origin/main tests/`

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

## ISA Quick Reference

### Instruction Bundle Format
```python
{"alu": [(op, dest, a1, a2), ...], "valu": [...], "load": [...], "store": [...], "flow": [...]}
```

### ALU Operations (12 slots/cycle)
```
("+", dest, a, b)   # dest = a + b
("-", dest, a, b)   # dest = a - b
("*", dest, a, b)   # dest = a * b
("//", dest, a, b)  # dest = a // b (floor div)
("%", dest, a, b)   # dest = a % b
("^", dest, a, b)   # dest = a XOR b
("&", dest, a, b)   # dest = a AND b
("|", dest, a, b)   # dest = a OR b
("<<", dest, a, b)  # dest = a << b
(">>", dest, a, b)  # dest = a >> b
("<", dest, a, b)   # dest = 1 if a < b else 0
("==", dest, a, b)  # dest = 1 if a == b else 0
```

### VALU Operations (6 slots/cycle) - operates on VLEN=8 elements
```
("vbroadcast", dest, src)        # dest[0:8] = src (scalar to vector)
("+", dest, a, b)                # dest[i] = a[i] + b[i] for i in 0..7
("multiply_add", dest, a, b, c)  # dest[i] = a[i] * b[i] + c[i]
# All ALU ops work element-wise on vectors
```

### Load Operations (2 slots/cycle)
```
("const", dest, value)           # dest = immediate value
("load", dest, addr_scratch)     # dest = mem[scratch[addr_scratch]]
("vload", dest, addr_scratch)    # dest[0:8] = mem[scratch[addr]:scratch[addr]+8]
```

### Store Operations (2 slots/cycle)
```
("store", addr_scratch, src)     # mem[scratch[addr]] = scratch[src]
("vstore", addr_scratch, src)    # mem[scratch[addr]:+8] = scratch[src:src+8]
```

### Flow Operations (1 slot/cycle)
```
("select", dest, cond, a, b)     # dest = a if cond != 0 else b
("vselect", dest, cond, a, b)    # vector select, element-wise
("jump", addr)                   # pc = addr (immediate)
("cond_jump", cond, addr)        # if cond != 0: pc = addr
("halt",)                        # stop execution
("pause",)                       # pause for debug sync (ignored in submission)
```

## Debugging Tips

- Use `trace=True` with `watch_trace.py` to visualize instruction execution in Perfetto
- Add debug instructions to validate intermediate computations
- Use `prints=True` in `do_kernel_test()` to see scratch space state
- The `pause` instruction synchronizes with reference kernel yields for step-by-step validation
- Check `scratch_map` in debug info to see variable names and addresses

## Optimization Tracking

All optimization iterations are documented in the `optimizations/` directory:

- **`optimizations/README.md`**: Index of all iterations with performance metrics
- **`optimizations/XXX_name.md`**: Individual iteration documentation
- **`optimizations/TEMPLATE.md`**: Template for new iterations

### Workflow for New Optimizations

#### Using Skills (Recommended)
1. `/new-iteration <description>` - Creates iteration file with auto-setup
2. Document your hypothesis and approach in the file
3. Implement changes in `perf_takehome.py`
4. `/perf-test` - Run performance test with formatted results
5. `/analyze-trace` - (Optional) Analyze bottlenecks if needed
6. `/record-result` - Update documentation with results
7. `/benchmark` - Run full validation when ready
8. Commit with descriptive message

#### Manual Workflow
1. Copy `TEMPLATE.md` to `XXX_description.md` (use next number in sequence)
2. Document your hypothesis and approach
3. Implement changes in `perf_takehome.py`
4. Run tests and record results
5. Update the iteration file with findings
6. Update `README.md` index table
7. Commit with descriptive message

See `.claude/skills/README.md` for detailed skill documentation.

### Current Status

- **Baseline:** 147734 cycles
- **Target:** < 1487 cycles (99% speedup needed)
- **Best:** Check `optimizations/README.md` for current best result

## Scratch Space Planning

Total: 1536 words. Plan allocation carefully for SIMD.

```
Scalar variables: 1 word each
Vector variables: 8 words each (VLEN=8)
Constants: Can be shared via scratch_const()

Example SIMD allocation:
- 7 init vars (rounds, n_nodes, etc.): 7 words
- 3 scalar temps: 3 words
- Vector indices (32 elements): 32 words
- Vector values (32 elements): 32 words
- Vector temps for hash (8 per stage × 6): 48 words
- Constants: ~20 words
Total: ~142 words (plenty of room)
```

## Common VLIW Packing Patterns

```python
# BAD: One instruction per bundle (baseline)
self.add("alu", ("+", a, b, c))
self.add("alu", ("*", d, e, f))  # 2 cycles

# GOOD: Pack independent ops into one bundle
self.instrs.append({
    "alu": [("+", a, b, c), ("*", d, e, f)],
    "load": [("load", x, y)]
})  # 1 cycle

# GOOD: Mix engines in one bundle
self.instrs.append({
    "alu": [("+", a, b, c)],
    "valu": [("vbroadcast", v, s)],
    "load": [("vload", data, addr)]
})  # 1 cycle, using 3 engines
```
