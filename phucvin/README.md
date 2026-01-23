# Performance Take-Home Solution

## Approach

The goal was to optimize a kernel that traverses a random forest and hashes values. The baseline implementation was scalar and extremely slow (~147,734 cycles).

### Optimizations Implemented

1.  **Instruction Scheduling**:
    -   Replaced the default scheduler with a greedy list scheduler.
    -   Implemented correct dependency tracking for RAW and WAW hazards to allow safe VLIW packing.
    -   Optimized packing respecting `SLOT_LIMITS` (e.g., 2 loads, 6 valus per cycle).

2.  **Vectorization**:
    -   Rewrote the kernel to use vector instructions (`valu`, `vload`, `vstore`) with `VLEN=8`.
    -   Processed the batch of 256 items using vectorized operations.

3.  **Tiling & Register Management**:
    -   Implemented **Tiling**: Instead of loading the entire batch (256 items) into scratch memory at once, the batch is processed in tiles of 128 items (`TILE_VECS=16`).
    -   This reduces register pressure and allows for a higher degree of instruction-level parallelism within the tile.
    -   Allocated **multiple sets of temporary registers** (`NUM_SETS=8`) to pipeline execution across vectors within a tile, hiding memory latency.

4.  **Algorithmic Improvements**:
    -   Optimized the tree traversal logic using vector masks and select operations to avoid branching.
    -   Experimented with caching top tree levels in scratch memory (`CACHE_LEVELS`), though the final submitted version uses `CACHE_LEVELS=0` as it proved most robust given scratch space constraints.

### Results

-   **Baseline**: 147,734 cycles
-   **My Solution**: ~2484 cycles
-   **Speedup**: ~59.5x

### Analysis

-   **Parallelism vs Resource Constraints**: The primary bottleneck was the limited scratch space (1536 words). This constrained the ability to unroll loops further or cache more of the tree. Tiling was a crucial strategy to balance register usage with parallelism.
-   **Load Throughput**: The dual load units were heavily utilized. Further optimization would likely require more aggressive caching strategies if scratch space permitted.
