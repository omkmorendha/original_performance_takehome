# Optimization Iterations

This directory tracks all optimization attempts for the performance engineering challenge.

## Directory Structure

Each optimization iteration has:
- A numbered markdown file (e.g., `001_baseline.md`) documenting the approach, results, and learnings
- Code snapshots or diffs if needed
- Performance metrics and analysis

## Iteration Index

| Iteration | Description | Cycles | Speedup | Status |
|-----------|-------------|--------|---------|--------|
| 000 | Baseline (original scalar implementation) | 147734 | 1.00x | ✓ Baseline |

## Target Metrics

- **Baseline:** 147734 cycles
- **Target:** < 1487 cycles (Claude Opus 4.5 benchmark)
- **Current Best:** TBD

## Optimization Strategy

Planned optimization phases:
1. **Baseline Analysis** - Understand bottlenecks in current implementation
2. **SIMD Vectorization** - Replace scalar operations with vector operations
3. **VLIW Packing** - Pack independent operations into single cycles
4. **Instruction Scheduling** - Optimize instruction order for maximum throughput
5. **Loop Unrolling** - Reduce control flow overhead
6. **Memory Optimization** - Improve scratch space usage and access patterns
7. **Hash Function Optimization** - Vectorize the 6-stage hash computation

## Guidelines

- Each iteration should be self-contained and documented
- Include cycle count, correctness verification, and key learnings
- Reference previous iterations when building on them
- Track what worked and what didn't for future reference
