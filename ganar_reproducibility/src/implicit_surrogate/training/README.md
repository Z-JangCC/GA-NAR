# Training

Uses AdamW, fixed learning rate and weight decay, FP32 without AMP, validation-MSE-only early stopping, and formal seeds `0,1,2`. Activity supervision is detached and its lambda is selected only on the tuning seed `999`.

## Component contract

1. **Purpose:** train registered surrogate models under the frozen optimizer, seed, and stopping protocol.
2. **Mathematical definition:** optimize prediction MSE plus optional detached activity supervision; select checkpoints by validation MSE only.
3. **Terminology:** formal seed, tuning seed, best checkpoint, initial/mid trajectory, and detached target are explicit.
4. **Inputs:** model, train/validation standardized arrays, optional activity target, seed, and training configuration.
5. **Outputs:** training result, immutable checkpoints, loss trajectory, and model records.
6. **Shapes:** feature `(N,d_q)`, target `(N,d_z)`, activity target `(N_active,)`.
7. **Physical units:** optimization uses dimensionless standardized coordinates.
8. **Coordinate system:** input/output ordering follows the dataset standardizer.
9. **Numerical precision:** FP32 model training on CPU; metadata records optimizer/environment.
10. **Public API:** `Trainer`, `EarlyStoppingState`, checkpoint helpers, losses, and hyperparameter selection.
11. **Dependencies:** PyTorch, NumPy, and model registry.
12. **Invariants:** validation-MSE-only early stopping, formal seeds `{0,1,2}`, tuning seed `999`, no AMP.
13. **Numerical verification:** deterministic seed, checkpoint-load, parameter-count, and trajectory tests.
14. **Failure conditions:** missing data, incompatible dimensions, or failed checkpoint metadata raises an error.
15. **No-fallback rule:** unresolved geometry/activity gates do not fabricate informed training targets.
16. **Replacement interface:** trainers must return best epoch, validation metric, checkpoint path, and trajectory.
