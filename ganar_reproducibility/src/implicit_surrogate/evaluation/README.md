# Evaluation

Reports standardized-state MSE, de-standardized physical equation residual, parameter/MAC counts, batch-1 latency, CPF stress metrics, and paired mean/sample-standard-deviation summaries. No p-values or aggregate superiority score is generated.

## Component contract

1. **Purpose:** evaluate predictions, physical residuals, stress behavior, and computational cost.
2. **Mathematical definition:** metrics are computed on frozen sample IDs and the frozen train-only standardizer.
3. **Terminology:** nominal means the held-out test split; stress means CPF continuation populations.
4. **Inputs:** model predictions, dataset splits, system equations, standardization, and timing settings.
5. **Outputs:** scalar metric dictionaries, CSV rows, and paired summaries.
6. **Shapes:** predictions match `(N, d_z)`; stress tables have one row per model/ray/fraction as applicable.
7. **Physical units:** MSE is standardized; residuals are native equation units; latency is microseconds.
8. **Coordinate system:** residuals are evaluated after de-standardizing state predictions.
9. **Numerical precision:** metric accumulation uses `float64`; timing reports observed CPU values.
10. **Public API:** prediction, residual, stress, cost, and paired-statistics helpers.
11. **Dependencies:** NumPy, PyTorch model interfaces, and benchmark systems.
12. **Invariants:** test-time evaluation never calls an oracle or physics solver from the model forward graph.
13. **Numerical verification:** residual identity and deterministic paired-summary tests are included.
14. **Failure conditions:** shape mismatch, nonfinite prediction, or missing frozen scaling raises an error.
15. **No-fallback rule:** no metric is replaced by a proxy or aggregate superiority score.
16. **Replacement interface:** evaluators must preserve metric names, coordinate declarations, and status fields.
