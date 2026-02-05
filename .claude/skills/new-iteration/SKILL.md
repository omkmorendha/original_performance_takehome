---
name: new-iteration
description: Create a new optimization iteration with automated setup
argument-hint: description
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit
---

Create a new optimization iteration file from template.

For argument "$ARGUMENTS":

1. Read `optimizations/README.md` to find the highest iteration number in the table
2. Increment it to get the next number (format as 001, 002, 003, etc.)
3. Read the previous best cycle count from the README
4. Read `optimizations/TEMPLATE.md`
5. Create `optimizations/NNN_$ARGUMENTS.md` with these replacements:
   - `XXX` → the iteration number (e.g., 001)
   - `[Optimization Name]` → $ARGUMENTS
   - `YYYY-MM-DD` → today's date (2026-02-05)
   - `[Previous cycle count for comparison]` → the actual previous best
6. Add a new row to the iteration table in `optimizations/README.md`:
   - Iteration: NNN
   - Description: $ARGUMENTS
   - Cycles: "TBD"
   - Speedup: "TBD"
   - Status: "In Progress"
7. Display summary and next steps

Example: `/new-iteration simd-vectorization` creates `optimizations/001_simd-vectorization.md`
