# Performance Take-Home Solution

## Approach

The goal was to optimize a kernel that traverses a random forest and hashes values. The baseline implementation was scalar and extremely slow (~147,734 cycles). The target was to beat Claude Opus 4.5 (~1363 cycles).

### Optimizations Implemented

1.  **Instruction Scheduling**:
    -   Replaced the default scheduler with a greedy list scheduler.
    -   Implemented correct dependency tracking for RAW and WAW hazards to allow safe VLIW packing.
    -   Optimized packing respecting `SLOT_LIMITS` (e.g., 2 loads, 6 valus per cycle).

2.  **Vectorization**:
    -   Rewrote the kernel to use vector instructions (`valu`, `vload`, `vstore`) with `VLEN=8`.
    -   Processed the batch of 256 items in 32 vector chunks.

3.  **Variable Hoisting & Parallelism**:
    -   Hoisted `vload` and `vstore` of indices/values outside the main loop.
    -   Allocated **unique temporary registers** for each of the 32 chunks (using a pool of 3 sets to fit in scratch) to eliminate register contention.
    -   This allows the scheduler to interleave execution of multiple chunks, saturating the VLIW functional units (e.g., processing loads for Chunk 1 while hashing Chunk 0).

4.  **Cached Forest (Disabled)**:
    -   Implemented a strategy to cache top-level forest nodes in scratch and use `vselect` trees to gather. However, this introduced regression/correctness issues in the final tuning, so it was disabled (`CACHE_LEVELS=0`), relying purely on highly parallel gather loads.

### Results

-   **Baseline**: 147,734 cycles
-   **My Solution**: 3,098 cycles
-   **Speedup**: ~47.7x

### Analysis

-   **Parallelism is Key**: The massive speedup comes from exposing instruction-level parallelism across the 256 items. By unrolling and using distinct registers, the scheduler can hide the latency of memory loads (gather) by executing ALU operations for other vectors.
-   **Bottleneck**: The solution is likely limited by the `load` engine throughput (2 loads/cycle). Achieving < 2000 cycles would require successfully implementing the Cached Forest strategy to bypass memory loads for the top of the tree.
