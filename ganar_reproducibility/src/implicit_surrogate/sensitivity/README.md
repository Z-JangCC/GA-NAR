# Sensitivity

Computes finite-scale directional Jacobian variation with shared Rademacher probes, common-valid-pair filtering, normalized and centered second moments, reliability-gated scale selection, simplex-projected spectra, and the dominant subspace. The estimator is operationally conditional when branch filtering is present.

## Component contract

1. **Purpose:** estimate finite-scale directional Jacobian variation and the dominant sensitivity subspace.
2. **Mathematical definition:** shared-probe Hutchinson energies are aggregated into normalized/centered matrices, projected to a simplex, and eigendecomposed.
3. **Terminology:** scale `h`, pair budget, common-valid-pair mask, reliability gate, and `P_R` have protocol-defined meanings.
4. **Inputs:** centers, solved states/parameters, standardized scales, unit directions, Rademacher probes, and a JVP solver.
5. **Outputs:** directional variation, scale diagnostics, projected spectrum, and subspace basis.
6. **Shapes:** pairs `(N_centers * N_directions,)`; matrices `(d_q, d_q)`; basis `(d_q, R)`.
7. **Physical units:** sensitivity matrices use standardized parameter/state coordinates and are dimensionless.
8. **Coordinate system:** columns follow retained standardized input coordinates.
9. **Numerical precision:** estimator and eigendecomposition use `float64`.
10. **Public API:** direction generators, variation estimator, matrix builder, scale selector, projection, and subspace finder.
11. **Dependencies:** NumPy, SciPy linear algebra, derivative and solver components.
12. **Invariants:** shared pairs/probes and common-valid masks are used for every compared scale and half.
13. **Numerical verification:** Hutchinson-vs-explicit tiny reference, symmetry/trace, spectrum, and projector tests.
14. **Failure conditions:** invalid pairs, unstable scales, or failed split reliability yield `UNRESOLVED`.
15. **No-fallback rule:** no scale, rank, direction count, or seed is changed after a failed gate.
16. **Replacement interface:** an estimator must return energies, masks, diagnostics, and status without mutating inputs.
