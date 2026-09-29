# Verification

Verification checks mathematical invariants and numerical references. A protocol-defined unresolved result is recorded; a violated implementation invariant raises an exception.

## Component contract

1. **Purpose:** independently verify systems, derivatives, sensitivity estimators, models, and pipeline contracts.
2. **Mathematical definition:** checks compare implementation identities against analytic or reference calculations.
3. **Terminology:** PASS is an implementation check; `UNRESOLVED`/`FAILED_PRECONDITION` are legal scientific statuses.
4. **Inputs:** systems, arrays, models, artifacts, and configured tolerances.
5. **Outputs:** structured check records, markdown reports, and summary JSON.
6. **Shapes:** checks preserve the shapes of the component under test and report dimensions explicitly.
7. **Physical units:** checks state whether values are physical, standardized, or dimensionless.
8. **Coordinate system:** every report names the tested coordinate convention.
9. **Numerical precision:** tolerances and dtype are recorded with each report.
10. **Public API:** verification functions and pipeline audit/report writers.
11. **Dependencies:** component APIs, NumPy/SciPy, and serialized formal artifacts.
12. **Invariants:** verification never modifies frozen data, seeds, thresholds, or formal results.
13. **Numerical verification:** Richardson JVP, explicit Jacobian/Hutchinson, residual, and PYPOWER checks.
14. **Failure conditions:** implementation mismatch raises/fails; protocol gate failure is recorded with reason.
15. **No-fallback rule:** verification cannot rescue a failed scientific outcome.
16. **Replacement interface:** new checks return serializable status, metric, tolerance, and reason fields.
