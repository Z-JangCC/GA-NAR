# Fixed-Bus-Type AC Power Flow

Loads are the only parameters and bus types are frozen from canonical PYPOWER `case118`; Q-limit PV/PQ switching is disabled. The state is `[theta_nonref, vm_pq]`, residuals are per-unit `P_calc-P_spec` and `Q_calc-Q_spec`, and `F_z/F_q` are analytic reduced derivatives.

The allocation-only study additionally exposes `RichIEEE118PowerFlowSystem`:
the same fixed-bus-type 180-dimensional reduced state with 36 operating
parameters (regional load multipliers, generator redispatch, voltage setpoints,
and shunt controls) plus heterogeneous ZIP-like load dependence. It is a
separate benchmark identity and does not overwrite the historical load-only
V17.1 results.

## Component contract

1. **Purpose:** expose the canonical fixed-bus-type IEEE-118 AC power-flow implicit system.
2. **Mathematical definition:** reduced AC equations use `F(z_phys,q_phys)=0` with analytic reduced `F_z` and `F_q`.
3. **Terminology:** non-reference angles and PQ voltage magnitudes form the state; loads are the parameters.
4. **Inputs:** load-scaling parameter vector in canonical load ordering and optional continuation state.
5. **Outputs:** state, residual, derivatives, case metadata, and continuation-compatible branch operations.
6. **Shapes:** state dimension follows the frozen case; Jacobians are square/reduced and parameter columns match load count.
7. **Physical units:** power and voltage are per-unit; angles are radians in the public state.
8. **Coordinate system:** bus numbers, reference bus, PV/PQ masks, and load ordering are frozen in case metadata.
9. **Numerical precision:** `float64` equations, derivatives, and continuation.
10. **Public API:** `ACPowerFlowSystem`, case-data loader, equations, and derivative helpers.
11. **Dependencies:** NumPy, SciPy, and PYPOWER `case118` for the independent reference.
12. **Invariants:** bus types and canonical case hash are fixed; Q-limit switching is disabled.
13. **Numerical verification:** residual/Jacobian checks, continuation checks, and PYPOWER state comparison.
14. **Failure conditions:** canonical zero-variance state coordinates are `FAILED_PRECONDITION`; failed continuation is explicit.
15. **No-fallback rule:** no coordinate deletion, artificial jitter, bus-type switching, or flat-start rescue.
16. **Replacement interface:** compatible power-flow systems must preserve reduced ordering, derivatives, and case metadata.
