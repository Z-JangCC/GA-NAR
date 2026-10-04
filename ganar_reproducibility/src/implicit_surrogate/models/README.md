# Models

Implements the six frozen registry configurations: ReLU, SwiGLU, parameter-matched SwiGLU, random-subspace sensitivity, subspace-only sensitivity, and full sensitivity-informed surrogate. Inference contains no physics solver or oracle.

## Component contract

1. **Purpose:** define ordinary and sensitivity-informed neural surrogate architectures.
2. **Mathematical definition:** models map standardized inputs to standardized states; informed models project through a frozen sensitivity basis.
3. **Terminology:** registry name is software identity; display name is paper-facing metadata.
4. **Inputs:** `(N, d_q)` standardized tensors and optional frozen basis/adapter parameters.
5. **Outputs:** `(N, d_z)` standardized predictions and optional detached activity outputs during training.
6. **Shapes:** dimensions are explicit in each `ModelSpec`; batch dimension is preserved.
7. **Physical units:** model tensors are dimensionless standardized coordinates.
8. **Coordinate system:** basis and adapters use the standardized input coordinate ordering.
9. **Numerical precision:** training is FP32 without AMP; physics evaluation de-standardizes to `float64`.
10. **Public API:** `MODEL_SPECS`, `build_model`, model classes, and parameter-matching helpers.
11. **Dependencies:** PyTorch and NumPy for basis construction.
12. **Invariants:** no model forward path invokes a system solver or oracle.
13. **Numerical verification:** parameter counts, projector identities, shape checks, and deterministic initialization.
14. **Failure conditions:** missing basis for an informed model or incompatible dimensions raises `ValueError`.
15. **No-fallback rule:** unresolved geometry prevents informed model registration; ordinary baselines remain explicit.
16. **Replacement interface:** new models register a `ModelSpec` and implement the common tensor forward contract.
