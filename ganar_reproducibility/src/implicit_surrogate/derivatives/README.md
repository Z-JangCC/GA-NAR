# Derivatives

Implements the standardized implicit solution-map JVP through a linear solve and validates it with central Richardson extrapolation at five frozen centers and directions.

## Component contract

1. **Purpose:** compute and verify derivatives of the implicit solution map.
2. **Mathematical definition:** `dz/dx = -F_z^{-1} F_q D_q D_z^{-1}` applied to a direction, without forming an inverse.
3. **Terminology:** JVP means Jacobian-vector product; Richardson checks compare analytic and central finite differences.
4. **Inputs:** system state/parameter, standardized direction, scaling vectors, and retained input indices.
5. **Outputs:** standardized JVP vectors and numerical verification records.
6. **Shapes:** direction `(d_q,)`; JVP `(d_z,)`; batched callers may stack rows.
7. **Physical units:** output follows standardized state coordinates; system derivatives use native units internally.
8. **Coordinate system:** state and parameter ordering is supplied by the system interface.
9. **Numerical precision:** linear solves and validation use `float64`.
10. **Public API:** `implicit_solution_jvp`, `representation`-independent neural JVP wrapper, and validation helpers.
11. **Dependencies:** NumPy and the `LinearSystemSolver`/system derivative interfaces.
12. **Invariants:** no explicit state-matrix inverse and no hidden solver call in the model forward path.
13. **Numerical verification:** five-center Richardson extrapolation and analytic-vs-reference tests.
14. **Failure conditions:** singular factorization, nonfinite derivatives, or tolerance violations raise/report failure.
15. **No-fallback rule:** a failed derivative check blocks downstream formal sensitivity analysis.
16. **Replacement interface:** a derivative backend must expose the same standardized JVP signature.
