# Data

Dataset generation is independent of models and training. The frozen split sizes are prediction `8192`, sensitivity `512`, validation `1024`, test `2048`, and probe `64`; train-only statistics are reused for every downstream split and stress set.

## Component contract

1. **Purpose:** generate, split, standardize, and schema-check benchmark data.
2. **Mathematical definition:** each sample is a physical pair `(q_phys, z_phys)` satisfying `F(z,q)=0`, with `x_std/y_std` derived from train-only statistics.
3. **Terminology:** prediction, sensitivity, validation, test, and representation-probe splits are disjoint frozen populations.
4. **Inputs:** benchmark system, counts, random seed, and optional frozen case data.
5. **Outputs:** `DatasetBundle`, split records, standardization statistics, and sample IDs.
6. **Shapes:** split arrays are `(N, d_q)` and `(N, d_z)`; IDs are `(N,)`.
7. **Physical units:** benchmark-native units before standardization; dimensionless after standardization.
8. **Coordinate system:** public physical arrays use the system's declared state/parameter ordering.
9. **Numerical precision:** generation and statistics use `float64`; model tensors may later use `float32`.
10. **Public API:** `DatasetProvider`, `Standardizer`, split/schema datatypes, and benchmark providers.
11. **Dependencies:** NumPy, SciPy/PYPOWER through the owning system, and core random/artifact helpers.
12. **Invariants:** statistics are fitted on the permitted training population and reused unchanged downstream.
13. **Numerical verification:** residual checks, split hashes, and degenerate-coordinate checks are persisted.
14. **Failure conditions:** zero-variance retained coordinates or failed solves produce `FAILED_PRECONDITION`.
15. **No-fallback rule:** no coordinate deletion, synthetic jitter, or alternate scaler is introduced after freezing.
16. **Replacement interface:** a provider must return the same `DatasetBundle` fields and sample-ID semantics.
