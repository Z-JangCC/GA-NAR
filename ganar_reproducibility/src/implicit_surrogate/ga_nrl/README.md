# Legacy compatibility allocation module

The formal name of this component is **NRGD (Nonlinear Response Geometry
Decomposition)**. The `ga_nrl` directory is retained only as a compatibility
namespace for archived runs; new code should import `implicit_surrogate.nrgd`.

## Component contract

**Purpose.** Construct the system-side response geometry, local response
intensity, and dominant/complement subspace-response allocation.

**Mathematical definition.** From local finite-scale Jacobian variation, `B_i = mean_k(D_ik.T D_ik)`, mode strengths are `b_ij=u_j.T B_i u_j`, and `w_i=[1-alpha_i, alpha_i*pi_i]` with `alpha_i=e_i/(e_i+mean(e))`.

**Inputs.** Frozen local energies, directions, selected sensitivity basis, and optionally eigenvalues.

**Outputs.** `ModeCapacityTarget` with mode strengths, complement strength, fractions, conserved simplex weights, and diagnostics.

**Public API.** `build_local_mode_geometry`, `build_mode_capacity_targets`, `GANRLAllocationRouter`, and `fit_allocation_router`.

**Invariants.** No implicit solver is called by the router; valid target rows sum to one and are non-negative; the complement explicitly accounts for truncated physical modes.

**Failure conditions.** Incompatible geometry/basis dimensions, non-finite inputs, or empty valid populations raise a clear error or unresolved status.

**No-fallback rule.** Invalid geometry is recorded as `UNRESOLVED`; fabricated allocation targets are never emitted.

**Replacement interface.** A future allocator may implement the same target fields and simplex output contract without changing downstream evaluation.
