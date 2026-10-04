# Systems

Defines the square implicit interface `F(z_phys, q_phys)=0`, the controlled synthetic system, fixed-bus-type IEEE-118 AC power flow, rich-control IEEE-118, coupled Duffing, 2D Allen-Cahn, and 2D shallow-water equilibria. Public state and parameter arrays use physical coordinates; derivatives are analytic and explicit. Branch certificates use continuation and never fall back to an unrelated flat-start solve.

## Component contract

1. **Purpose:** expose regular implicit nonlinear benchmarks through one square-system interface.
2. **Mathematical definition:** each system implements `F(z_phys,q_phys)=0`, `F_z`, `F_q`, and a canonical solution method.
3. **Terminology:** state, parameter, residual, fixed-bus-type, and branch certificate follow the benchmark definitions.
4. **Inputs:** physical parameter vectors and optional physical initial states.
5. **Outputs:** physical states, residuals, analytic Jacobians, and benchmark metadata.
6. **Shapes:** `F` and state `(d_z,)`; parameter `(d_q,)`; Jacobians `(d_z,d_z)` and `(d_z,d_q)`.
7. **Physical units:** benchmark-native units; power flow uses per-unit values and radians internally as documented.
8. **Coordinate system:** ordering is explicit in each system README and case metadata.
9. **Numerical precision:** system equations and derivatives use `float64`.
10. **Public API:** `ImplicitSystem`, `SyntheticImplicitSystem`, and `ACPowerFlowSystem` methods.
11. **Dependencies:** NumPy/SciPy and PYPOWER for canonical case data/reference checks.
12. **Invariants:** square residual, analytic derivatives, canonical case hash, and fixed bus types.
13. **Numerical verification:** residual/Jacobian/JVP checks and independent PYPOWER comparison.
14. **Failure conditions:** nonconvergence, degenerate output coordinates, or invalid case metadata.
15. **No-fallback rule:** no bus-type switch, coordinate deletion, or unrelated branch rescue.
16. **Replacement interface:** new systems implement the base residual/Jacobian/solve contract and metadata.
