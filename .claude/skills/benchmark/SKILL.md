---
name: benchmark
description: Run comprehensive submission validation with correctness and speed tests
disable-model-invocation: true
allowed-tools: Bash, Read
---

Run the complete submission validation suite.

1. Check that tests folder is unchanged: `git diff origin/main tests/`
   - If there are any changes, show ERROR and explain that the submission will be invalid
   - If no changes, show ✓ Tests folder unchanged

2. Run: `python tests/submission_tests.py`

3. Parse the output and display formatted results:
   - Overall correctness: ✓ Pass / ✗ Fail
   - Cycle count from the output
   - Which performance tiers achieved:
     - Tier 1: < 100000 cycles
     - Tier 2: < 10000 cycles
     - Tier 3: < 2000 cycles
     - Target: < 1500 cycles (Claude Opus 4.5 benchmark)
     - Ultimate: < 1487 cycles (beat the benchmark)

Display in format:
```
=== Submission Benchmark ===

✓ Tests folder unchanged
✓ Correctness tests passed

Performance:
Cycles: [number]

Thresholds:
[✓/✗] < 100000 cycles (Tier 1)
[✓/✗] < 10000 cycles (Tier 2)
[✓/✗] < 2000 cycles (Tier 3)
[✓/✗] < 1500 cycles (Target)
[✓/✗] < 1487 cycles (Beat Claude Opus 4.5)

Status: [Highest tier achieved]
```
