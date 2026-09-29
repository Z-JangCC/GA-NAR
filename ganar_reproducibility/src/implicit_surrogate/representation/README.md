# Representation Sensitivity

Extracts Early/Middle/Late activations from ordinary ReLU/SwiGLU networks, fits ridge-regularized covariance whitening, computes neural JVP finite-scale sensitivity on an independent probe population, and reports centered-matrix cosine distance only when every reliability gate passes.

## Component contract

1. **Purpose:** quantify hidden-representation sensitivity and its alignment with system sensitivity.
2. **Mathematical definition:** whitened representation JVP finite-scale energies produce a centered sensitivity matrix compared by cosine distance.
3. **Terminology:** Early/Middle/Late are frozen layer checkpoints; alignment is conditional on all reliability gates.
4. **Inputs:** ordinary model, independent probe population, layer activations, whitening fit, directions, and system target matrix.
5. **Outputs:** representation sensitivity result, split reliability diagnostics, and alignment rows.
6. **Shapes:** activations `(N, d_layer)`; sensitivity matrices are square in the probe-coordinate output space.
7. **Physical units:** standardized input and whitened activation coordinates are dimensionless.
8. **Coordinate system:** whitening is fitted on frozen reference activations and reused unchanged.
9. **Numerical precision:** JVP/whitening calculations use `float64` arrays around FP32 model evaluation.
10. **Public API:** activation extraction, ridge whitening, representation JVP, sensitivity estimation, and alignment helpers.
11. **Dependencies:** PyTorch, NumPy, SciPy, and sensitivity matrix utilities.
12. **Invariants:** probe population is independent of training; invalid pairs are masked consistently across splits.
13. **Numerical verification:** whitening residuals, finite-difference JVP, and split-half reliability checks.
14. **Failure conditions:** nonfinite activations, singular whitening, or failed system probe reliability produce N/A alignment.
15. **No-fallback rule:** no alignment is reported from an unresolved system geometry gate.
16. **Replacement interface:** a representation backend must expose layer activations and standardized JVPs.
