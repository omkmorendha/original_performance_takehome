---
name: analyze-trace
description: Generate and analyze execution trace to identify bottlenecks and optimization opportunities
disable-model-invocation: true
allowed-tools: Bash, Read
---

Generate a detailed execution trace and analyze performance characteristics.

1. Run: `python perf_takehome.py Tests.test_kernel_trace` to generate trace.json

2. Inform user: "Starting trace viewer... Run: python watch_trace.py"
   - Mention this opens in browser and only works in Chrome
   - They can also drag trace.json to https://ui.perfetto.dev/

3. Read and analyze trace.json for:
   - Total cycles
   - Engine utilization patterns
   - Average operations per cycle
   - Slot utilization vs theoretical limits for each engine:
     - ALU: 12 slots per cycle
     - VALU: 6 slots per cycle
     - Load: 2 slots per cycle
     - Store: 2 slots per cycle
     - Flow: 1 slot per cycle

4. Identify bottlenecks:
   - Which engines are underutilized
   - Serial dependencies causing stalls
   - Memory access patterns
   - SIMD opportunities (VALU usage)

5. Suggest optimization priorities based on analysis:
   - Rank by potential impact (high/medium/low)
   - Give specific recommendations

Display analysis in format:
```
=== Trace Analysis ===

Performance Metrics:
- Total cycles: [number]
- Average ops/cycle: [number] (theoretical max: ~20)

Engine Utilization:
- ALU: [percentage]% ([avg] / 12 slots)
- VALU: [percentage]% ([avg] / 6 slots) [⚠️ if 0%]
- Load: [percentage]% ([avg] / 2 slots)
- Store: [percentage]% ([avg] / 2 slots)
- Flow: [percentage]% ([avg] / 1 slot)

Bottlenecks:
1. [Issue] - [Impact level]
2. [Issue] - [Impact level]

Optimization Priorities:
1. [High] [Recommendation]
2. [Medium] [Recommendation]

Trace viewer: python watch_trace.py
```
