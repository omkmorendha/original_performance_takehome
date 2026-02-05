---
name: perf-test
description: Run performance test and display cycle count with speedup calculations
disable-model-invocation: true
allowed-tools: Bash, Read
---

Run the performance test and display formatted results.

1. Execute: `python perf_takehome.py Tests.test_kernel_cycles`
2. Parse the CYCLES output from the results
3. Read `optimizations/README.md` to get the previous best cycle count
4. Calculate speedups versus baseline (147734 cycles) and previous best
5. Display formatted results with progress toward target (1487 cycles)

Show results in this format:
```
=== Performance Test Results ===
Cycles: [number]
Speedup vs baseline (147734): [number]x
Speedup vs previous best: [number]x
Target: 1487 cycles ([number]x more speedup needed)
Status: [✓ Tests passed / ✗ Tests failed]
```
