---
name: record-result
description: Record the results of an optimization iteration
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit
---

Record and document the results of the current optimization.

1. Run: `python perf_takehome.py Tests.test_kernel_cycles` to get current cycle count
2. Parse the CYCLES output
3. Find the latest iteration file in `optimizations/` with status "In Progress" or "TBD"
   - Look for the highest numbered XXX_*.md file
4. Update that iteration file:
   - Set Cycles field to the result
   - Calculate and set Speedup vs baseline (147734 ÷ cycles)
   - Set Status to "✓ Success"
   - Set Date to today if not already set
5. Update the corresponding row in `optimizations/README.md`:
   - Set Cycles to the result
   - Calculate Speedup (147734 ÷ cycles, format as X.XXx)
   - Change Status from "In Progress" to "✓ Success"
   - If this is better than "Current Best", update that field too
6. Display summary with:
   - Iteration number and name
   - Cycle count
   - Speedup vs baseline
   - Whether it's a new personal best
   - Progress toward target (1487 cycles)
