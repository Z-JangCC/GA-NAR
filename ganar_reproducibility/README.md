# GA-NAR software package

This directory contains the GA-NAR implementation, fixed benchmark data,
geometry descriptors, experiment protocols, and numerical records. The
[repository README](../README.md) describes the method and gives complete
installation, training, evaluation, and inference commands.

The public method lives in `src/ga_nar/`. Its `geometry.py` constructs NRGD
descriptors from physical Jacobian actions; `model.py` implements
the dual-subspace controller and additive preactivation modulation; `data.py`
loads fixed benchmark splits and NRGD descriptors; `training.py` applies the
prediction and geometry-target objectives. `experiments/run_ga_nar.py` provides
the `train`, `evaluate`, and `predict` commands.

Existing experiment scripts and numerical records retain their original
identifiers for traceability. They are separate from newly trained checkpoints,
which are written under ignored local output directories.
