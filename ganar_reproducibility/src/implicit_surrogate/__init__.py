"""Modular implicit-system surrogate learning package."""

import os

# Keep dense numerical kernels reproducible and avoid machine-wide BLAS
# oversubscription during continuation experiments.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

__version__ = "0.1.0"
# This constant remains the V17.0 compatibility default.  Artifact metadata
# uses ``core.protocol.active_protocol_id`` so a V17.1 run can be isolated by
# setting IMPLICIT_SURROGATE_PROTOCOL_ID before process startup.
STUDY_PROTOCOL_ID = "GA-NRL-17.0-FINAL-FROZEN"
