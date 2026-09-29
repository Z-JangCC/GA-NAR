# GA-NAR result traceability

This document maps each published numerical result family to its canonical
package namespace. No manuscript archive or rendered figure is required.

| Experiment family | Canonical package namespace | Source project role |
|---|---|---|
| NRGD five-benchmark descriptor | `results/nrgd_five_benchmark_descriptor` | implicit-system equations, solvers, finite-scale geometry and allocation router |
| GANR shared-physics factorial | `results/ganr_shared_physics_factorial` | response-aware representation comparison under matched physics conditions |
| GA-NAR-LW activation matrix | `results/ga_nar_lw_activation_matrix` | three-system activation-matched end-to-end evaluation |
| GA-NAR-LW controlled ablation | `results/ga_nar_lw_controlled_ablation` | capacity, generic geometry, and structured modulation attribution |

The implementation packages are under `src/implicit_surrogate`,
`src/nonlinear_geometry`, `src/ganar`, and `src/ganar_lw_independent`.
All distributed result rows are hashed by `manifests/package_manifest.json` and
audited by `verification/audit_result_records.py`.
