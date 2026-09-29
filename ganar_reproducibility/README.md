# GA-NAR reproduction package

This package contains the implementation, benchmark data, experiment
protocols, and numerical records for the GA-NAR ICLR 2027 experiments. It is
organized by scientific experiment rather than by search stage or temporary
run directory.

## Included experiment families

1. `nrgd_five_benchmark_descriptor` — Jacobian-informed geometry extraction
   and routing across five implicit systems.
2. `ganr_shared_physics_factorial` — matched representation and physics
   comparisons across the shared benchmark suite.
3. `ga_nar_lw_activation_matrix` — activation-matched GA-NAR-LW experiments
   on IEEE-118, Kuramoto-64, and nonlinear heat FEM.
4. `ga_nar_lw_controlled_ablation` — capacity, generic geometry, and
   structured-modulation controls, including physical residual evaluation.

The central equations are distributed as pre-rendered SVG assets so they are
visible without a platform-specific mathematics renderer.

<p align="center"><img src="assets/equations/implicit_jacobian.svg" width="360" alt="Implicit system and implicit Jacobian"></p>

<p align="center"><img src="assets/equations/response_operator.svg" width="470" alt="Response operator and dominant response subspace"></p>

<p align="center"><img src="assets/equations/geometry_descriptor.svg" width="420" alt="NRGD geometry descriptor"></p>

For each hidden layer, GA-NAR-LW forms an additive geometry-conditioned gate:

<p align="center"><img src="assets/equations/layerwise_modulation.svg" width="560" alt="Layer-wise geometry modulation"></p>

Here e<sub>R</sub> and e<sub>⊥</sub> are dominant and complementary geometry
embeddings, β<sub>ℓ</sub>(x) allocates modulation energy across depth, and RMS
normalization matches the gate scale. For SwiGLU, only the gate branch is
modulated. The depth weights are a softmax and sum to one.

The physical consistency metric is

<p align="center"><img src="assets/equations/physics_residual.svg" width="180" alt="Physics residual"></p>

Here d<sub>z</sub> is the state dimension.

## GA-NAR architecture

The architecture follows a system-to-network pathway. NRGD extracts the global
DRS projector P<sub>R</sub>, local response intensity
ρ<sub>G</sub><sup>⋆</sup>(x), and dominant/complement allocation
π<sub>G</sub><sup>⋆</sup>(x). The input is split into
x<sub>R</sub> = P<sub>R</sub>x and
x<sub>⊥</sub> = (I<sub>d</sub> − P<sub>R</sub>)x; separate encoders preserve both
channels, and a shared controller predicts neural modulation strength,
composition, and depth allocation.

Each nonlinear block projects the two encoded channels into its preactivation
coordinates, mixes them according to the allocation, scales them by local
intensity and depth allocation, and RMS-matches the result to the backbone.
The additive modulation is injected before the selected activation; for SwiGLU
it acts only on the gate branch. Physical geometry construction is train-only,
so deployment uses the frozen descriptor and neural forward path.

<p align="center"><img src="assets/Fig.2.svg" width="900" alt="GA-NAR architecture"></p>

<p align="center"><em>GA-NAR architecture: NRGD extracts physical response
geometry, and GANR converts it into layer-wise nonlinear modulation through
dual-subspace encoding, shared control, scale matching, normalization, and
activation blocks.</em></p>

## Quick start

```bash
python -m pip install -e .

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python verification/verify_final_reproduction.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python verification/audit_result_records.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q
```

Run a small CPU geometry example:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python experiments/run_nrgd_five_benchmark.py --mode smoke \
  --benchmark synthetic --centers 8 --directions-per-center 2 \
  --output-root local_reproduction_results/nrgd_smoke
```

Run the complete registered protocol:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python experiments/reproduce_all.py --stage all
```

Generated training outputs are written to local ignored directories and do not
replace the numerical records distributed with this package.

## Contents

- `configs/`: final experiment protocols and hyperparameters.
- `data_store/`: frozen datasets, probes, and geometry artifacts.
- `experiments/`: training, evaluation, and report entry points.
- `manifests/`: data lineage and numerical traceability records.
- `reports/`: compact machine-readable summaries.
- `results/`: numerical records used by the submission.
- `src/`: NRGD, GANR, GA-NAR-LW, and physics implementations.
- `verification/`: consistency checks for the package and result records.

Manuscript files and exploratory search outputs are not part of the source
package. The trained models used for the reported runs are provided as a
separate [companion model archive](https://github.com/Z-JangCC/GA-NAR/releases/tag/final-submission-reproducibility)
with a machine-readable manifest and per-file checksums.
