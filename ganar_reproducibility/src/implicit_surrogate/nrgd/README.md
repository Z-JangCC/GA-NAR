# NRGD

`nrgd` is the canonical system-side namespace for **Nonlinear Response Geometry Decomposition**.
It constructs finite-scale response geometry, dominant-response subspaces,
local response-intensity targets, and subspace-response allocation targets.

The historical `ga_nrl` namespace remains import-compatible for archived
experiments, but it is not the formal method name in the current GA-NAR
framework.

## Component contract

**Purpose.** Construct and validate system-side nonlinear-response geometry
before a GANR representation consumes it.

**Mathematical definition.** The descriptor is
\(\mathcal D_G(x)=(P_R,\rho_G^\star(x),\pi_G^\star(x))\), with an explicit
dominant-response projector, local response intensity, and dominant/complement
subspace allocation.

**Inputs.** Frozen standardized implicit-system centers, finite-scale response
energies, directions, and a training-derived DRS projector.

**Outputs.** Local nonlinear-response intensity, dominant/complement
subspace-response allocation, and conserved simplex targets for allocation-
fidelity evaluation.

**Invariants.** Geometry construction is train-only and auditable; valid
targets are nonnegative and sum to one; the deployable predictor performs no
implicit solve at inference time.

**Public API.** `build_mode_capacity_targets`, `build_local_mode_geometry`,
`NRGDAllocationRouter`, and the training/evaluation helpers re-exported by this
namespace.

**Failure conditions.** Non-finite geometry, incompatible basis dimensions,
or an empty valid-center set produce a clear error or unresolved status.

**No-fallback rule.** Invalid geometry is reported as unresolved; fabricated
response descriptors are never emitted.

**Replacement interface.** A future NRGD implementation may replace the
historical backend if it preserves the descriptor fields, simplex target, and
evaluation contract above.
