# Solvers

Contains LU reuse, fixed-equation Newton correction, tangent predictor/adaptive parameter continuation, and pseudo-arclength CPF. A failed continuation or fold certificate is returned as an explicit failure and is never replaced with the last converged point.

## Component contract

1. **Purpose:** solve regular implicit systems and continuation paths used by data and stress evaluation.
2. **Mathematical definition:** Newton solves `F(z,q)=0`; continuation advances parameters with tangent or pseudo-arclength predictors.
3. **Terminology:** factorization, correction, branch, ray, fold certificate, and continuation failure are explicit statuses.
4. **Inputs:** system residual/Jacobian interfaces, initial states, target parameters, tolerances, and step bounds.
5. **Outputs:** solved states, factorization-backed JVP results, continuation traces, and failure certificates.
6. **Shapes:** state `(d_z,)`, parameter `(d_q,)`, Jacobian `(d_z,d_z)`.
7. **Physical units:** native benchmark units; standardized conversion is owned by data/derivative layers.
8. **Coordinate system:** system-declared state/parameter ordering is preserved.
9. **Numerical precision:** solver arithmetic is `float64` with configured tolerances.
10. **Public API:** linear solver, nonlinear solver, parameter continuation, and CPF solver classes.
11. **Dependencies:** NumPy/SciPy and system interfaces.
12. **Invariants:** no explicit inverse, no flat-start branch rescue, and no last-point-as-fold substitution.
13. **Numerical verification:** residual, Jacobian, continuation, and CPF certificate tests.
14. **Failure conditions:** nonconvergence, singular Jacobian, or absent fold certificate is returned explicitly.
15. **No-fallback rule:** failed branch evidence cannot be replaced with an unrelated converged point.
16. **Replacement interface:** solver implementations must preserve result/status and trace fields.
