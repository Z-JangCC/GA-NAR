from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...core.hashing import sha256_array, sha256_file
from ...core.exceptions import ArtifactMismatchError


@dataclass(frozen=True)
class PowerFlowCase:
    base_mva: float
    bus: np.ndarray
    generator: np.ndarray
    branch: np.ndarray
    source: str
    source_version: str
    original_file_sha256: str
    internal_case_sha256: str


def _internal_hash(base_mva: float, bus: np.ndarray, generator: np.ndarray, branch: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.asarray([base_mva], dtype=np.float64).tobytes())
    for array in (bus, generator, branch):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def load_case118(frozen_path: str | Path | None = None) -> PowerFlowCase:
    """Load canonical PYPOWER case118 or a previously frozen npz artifact."""
    if frozen_path is not None and Path(frozen_path).exists():
        path = Path(frozen_path)
        archive = np.load(path, allow_pickle=False)
        base_mva = float(archive["base_mva"])
        bus = archive["bus"].astype(np.float64)
        generator = archive["generator"].astype(np.float64)
        branch = archive["branch"].astype(np.float64)
        source_hash = str(archive["source_sha256"]) if "source_sha256" in archive else sha256_file(path)
        source_version = str(archive["source_version"]) if "source_version" in archive else "frozen"
        source = str(archive["source"]) if "source" in archive else "frozen artifact"
        return PowerFlowCase(base_mva, bus, generator, branch, source, source_version, source_hash, _internal_hash(base_mva, bus, generator, branch))
    try:
        from pypower.case118 import case118
        import inspect
    except ImportError as exc:
        raise RuntimeError("PYPOWER is required to load canonical case118") from exc
    payload = case118()
    base_mva = float(payload["baseMVA"])
    bus = np.asarray(payload["bus"], dtype=np.float64)
    generator = np.asarray(payload["gen"], dtype=np.float64)
    branch = np.asarray(payload["branch"], dtype=np.float64)
    # PYPOWER distributes case data as Python source; hash its canonical numeric content.
    source_path = Path(inspect.getsourcefile(case118) or "")
    source_hash = sha256_file(source_path) if source_path.exists() else _internal_hash(base_mva, bus, generator, branch)
    return PowerFlowCase(base_mva, bus, generator, branch, str(source_path) if source_path else "PYPOWER case118", str(payload.get("version", "")), source_hash, _internal_hash(base_mva, bus, generator, branch))


def freeze_case118(path: str | Path) -> PowerFlowCase:
    case = load_case118()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_case118(path)
        if existing.internal_case_sha256 != case.internal_case_sha256:
            raise ArtifactMismatchError(f"frozen case artifact mismatch: {path}")
        return existing
    np.savez_compressed(
        path,
        base_mva=np.asarray(case.base_mva),
        bus=case.bus,
        generator=case.generator,
        branch=case.branch,
        source=np.asarray(case.source),
        source_version=np.asarray(case.source_version),
        source_sha256=np.asarray(case.original_file_sha256),
    )
    return load_case118(path)
