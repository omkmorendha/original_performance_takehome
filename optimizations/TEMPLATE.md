# Iteration XXX: [Optimization Name]

**Date:** YYYY-MM-DD
**Status:** [In Progress / ✓ Success / ✗ Failed / ⚠ Partial]
**Cycles:** [Number or N/A]
**Speedup:** [X.XXx or N/A]
**Previous Best:** [Previous cycle count for comparison]

## Overview

Brief description of what this optimization attempts to achieve.

## Hypothesis

What do we expect this optimization to improve and why?

## Implementation Details

### Approach
- Key changes made
- Techniques applied
- Design decisions

### Code Changes
- Files modified
- Key functions/methods changed
- Algorithm modifications

### Technical Details
- Instruction patterns
- Resource utilization (ALU, VALU, Load, Store, Flow slots)
- Scratch space usage
- Memory access patterns

## Results

### Performance
```
Cycles: XXXXX
Speedup vs baseline: X.XXx
Speedup vs previous: X.XXx
```

### Correctness
```bash
# Commands run
python perf_takehome.py Tests.test_kernel_cycles
python tests/submission_tests.py
```

Result: [✓ Pass / ✗ Fail with details]

### Analysis
- What worked well
- What didn't work as expected
- Bottlenecks remaining
- Unexpected findings

## Debugging Notes

- Issues encountered
- How they were resolved
- Trace analysis insights
- Verification steps taken

## Lessons Learned

- Key insights from this iteration
- What to avoid in future attempts
- Patterns that proved effective
- Constraints discovered

## Next Steps

Based on this iteration's results:
1. [Next optimization priority]
2. [Alternative approach to try]
3. [Follow-up experiments]

## References

- Previous iterations: [links to related iterations]
- Code snapshots: [if applicable]
- Trace files: [if saved]
