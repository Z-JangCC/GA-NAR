# GA-NAR

**Geometry-Adaptive Nonlinearity Allocation and Representation for Complex Physical Systems**

GA-NAR learns surrogates for solution maps defined by implicit physical equations.
It combines Nonlinear Response Geometry Decomposition (NRGD), which measures
finite-scale variation of the physical solution Jacobian, with Geometry-Adaptive
Nonlinear Representation (GANR), which conditions a residual network on that
geometry. This repository provides the method implementation, benchmark data,
experiment protocols, numerical records, and verification tools accompanying
the manuscript.

## Method

NRGD constructs a global Dominant Nonlinear-Response Subspace (DRS) projector
from training-only directional response measurements. It also estimates two
operating-point descriptors: bounded local response intensity and the allocation
of response between the DRS and its orthogonal complement. The projector acts on
standardized inputs; both subspaces remain available to the model.

GANR encodes the two projected inputs separately. A shared controller predicts
intensity, two-way subspace allocation, and a softmax allocation across network
depth. At each residual block, separate fixed-highway plus trainable low-rank
projections map the branch codes into gate coordinates. The resulting direction
is RMS-matched to the block's unshifted gate and added before its activation.
The intensity and subspace allocation are supervised by NRGD descriptors;
depth allocation is learned from prediction loss. SwiGLU modifies only its gate
branch. Setting the adapter scale to zero exactly recovers the same backbone.

The [architecture diagram](ganar_reproducibility/assets/Fig.2.svg) summarizes
the physical-geometry-to-network path. The implementation is in
[`ga_nar.model`](ganar_reproducibility/src/ga_nar/model.py); the data,
training, and command-line interfaces are in the same package and
[`experiments/run_ga_nar.py`](ganar_reproducibility/experiments/run_ga_nar.py).
For a new physical system, [`build_response_geometry`](ganar_reproducibility/src/ga_nar/geometry.py)
accepts training-only standardized operating points and a callable returning
the physical solution Jacobian applied to a vector. It estimates the
finite-scale response, DRS projector, intensity, and two-way allocation. The
caller supplies a Jacobian action valid on the selected regular solution branch.
The trained model's inference path does not call the physical solver or
Jacobian action.

## Install

Python 3.10 or newer is required. From the repository root:

```bash
cd ganar_reproducibility
python -m pip install -e .
```

The packaged benchmark data contain fixed train, validation, and held-out test
splits, together with geometry descriptors for IEEE-118 AC power flow,
Kuramoto-64, and nonlinear heat FEM. Their checksums are recorded with each
trained checkpoint. Use the same standardized input coordinates when supplying
a custom projector or descriptor set.

## Train and evaluate

The following example trains the full layer-wise method on one system and
activation. Output files are written to an ignored local directory.

```bash
python experiments/run_ga_nar.py train \
  --system ieee118_acpf --activation relu --seed 0 \
  --epochs 400 --device cpu \
  --output local_training_results/ga_nar/ieee118_relu

python experiments/run_ga_nar.py evaluate \
  --checkpoint local_training_results/ga_nar/ieee118_relu/model.pt
```

The `train` command fits the prediction loss and NRGD intensity/allocation
losses on training data, selects an epoch by validation MSE, then reports the
held-out standardized MSE. `evaluate` recomputes that metric and rejects
changed source data or geometry. To predict physical states from a NumPy array
of unstandardized physical inputs with shape `(samples, input_dim)`:

```bash
python experiments/run_ga_nar.py predict \
  --checkpoint local_training_results/ga_nar/ieee118_relu/model.pt \
  --input inputs.npy --output predictions.npy
```

The checkpoint stores training normalization and the frozen projector, so the
prediction command does not require a physical solver or geometry target files.
Reproducing the full manuscript tables requires the registered experiment
protocols and substantially more computation than this single-run example.

## Repository layout

```text
ganar_reproducibility/
  src/ga_nar/       layer-wise method, data contract, training
  src/             NRGD, auxiliary studies, and benchmark implementations
  experiments/     method and registered experiment entry points
  configs/         experiment settings
  data_store/      fixed benchmark and geometry artifacts
  results/         archived numerical records
  verification/    package and result-record checks
```

The archived experiment records preserve their original model and result
identifiers for traceability. The `ga_nar` package and command above are the
public entry point for the manuscript's complete layer-wise adapter. Archived
records should not be interpreted as output from a fresh run of that entry
point. Large trained weights are available separately in the
[repository releases](https://github.com/Z-JangCC/GA-NAR/releases).

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q tests/test_ga_nar.py
PYTHONDONTWRITEBYTECODE=1 python verification/verify_clean_package.py
PYTHONDONTWRITEBYTECODE=1 python verification/audit_result_records.py
```

The verification scripts check method behavior, distributed-file checksums,
and numerical-record consistency. Full benchmark retraining is separate from
these checks.

## Citation

```bibtex
@misc{ganar,
  title = {GA-NAR: Geometry-Adaptive Nonlinearity Allocation and Representation for Complex Physical Systems},
  note  = {Manuscript and accompanying code}
}
```

Please include the system, command, software environment, and relevant output
when reporting a reproducibility issue.
