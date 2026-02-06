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
| 001 | SIMD vectorization + VLIW packing (dual groups) | 9293 | 15.9x | ✓ Tier 1 |
| 002 | Software pipelining (overlap loads with compute) | 6173 | 23.9x | ✓ Tier 1 |
| 003 | Eliminate vselect with math (idx = 2*idx+1+val&1) | 5661 | 26.1x | ✓ Tier 1 |
| 004 | Eliminate wrap vselect + 4-group + round fusion | 3940 | 37.5x | ✓ Tier 1 |
| 005 | Store overlap + init batching | 3664 | 40.3x | ✓ Tier 1 |
| 006 | Depth-aware broadcast plan | N/A | N/A | 📋 Plan |
| 007 | G=8 scratch-resident + multiply_add + broadcast d0-d1 | 2321 | 63.7x | ✓ Tier 1 |
| 008 | Deep broadcast interleave + init/store overlap | 2101 | 70.3x | ✓ Tier 2 |
| 009 | Micro-optimizations (init overlap + store prep) | 2097 | 70.5x | ✓ Tier 2 |
| 010 | Skip index loads + no-wrap broadcast (d0/d1) | 2053 | 72.0x | ✓ Tier 2 |

## Target Metrics

- **Baseline:** 147734 cycles
- **Target:** < 1487 cycles (Claude Opus 4.5 benchmark)
- **Current Best:** 2053 cycles (72.0x speedup, Tier 2)

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

## Pre-Submission Checklist

Before claiming any result, verify:

```bash
# 1. Tests folder unchanged (MUST be empty output)
git diff origin/main tests/

# 2. Run submission validation
python tests/submission_tests.py

# 3. Record cycles and which tiers pass
```

## Performance Tiers

| Tier | Cycles | Speedup | Milestone |
|------|--------|---------|-----------|
| Start | < 147,734 | > 1× | Beat baseline |
| Tier 1 | < 18,532 | > 8× | Updated starting point |
| Tier 2 | < 2,164 | > 68× | Opus 4 many hours |
| Tier 3 | < 1,790 | > 83× | Opus 4.5 casual |
| Tier 4 | < 1,579 | > 94× | Opus 4.5 2hr |
| Tier 5 | < 1,548 | > 95× | Sonnet 4.5 many hours |
| **TARGET** | < **1,487** | > **99×** | **Opus 4.5 11hr** |
| Ultimate | < 1,363 | > 108× | Opus 4.5 improved |
