# Local Nonlinearity

Builds a full-space four-direction local sensitivity score and the bounded target `e/(e+mean(e))`. It never reads the dominant subspace; split-half Spearman reliability is a separate gate.

## Component contract

1. **Purpose:** construct an independent local nonlinearity supervision target.
2. **Mathematical definition:** per-center energy is aggregated across four directional finite-scale variations and normalized as `e/(e+mean(e))`.
3. **Terminology:** activity target is full-space; reliability is a split-half Spearman gate.
4. **Inputs:** four-direction energy matrix and common-valid-pair mask.
5. **Outputs:** target vector, active indices, reliability/status diagnostics.
6. **Shapes:** energies `(N, 4)`; target and active mask `(N,)`.
7. **Physical units:** energy is dimensionless in standardized derivative coordinates; target is dimensionless.
8. **Coordinate system:** no dominant-subspace coordinate is used.
9. **Numerical precision:** `float64` for aggregation and rank statistics.
10. **Public API:** `compute_local_nonlinearity`, reliability primitives, and result datatypes.
11. **Dependencies:** NumPy and SciPy rank correlation.
12. **Invariants:** activity computation is independent of `P_R` and uses the frozen four directions.
13. **Numerical verification:** bounded-target and split-half reliability tests.
14. **Failure conditions:** invalid shape, nonfinite energy, or too few valid pairs yields an explicit unresolved status.
15. **No-fallback rule:** no activity target is fabricated when its reliability gate fails.
16. **Replacement interface:** a target backend must return target, active indices, mask, status, and diagnostics.
