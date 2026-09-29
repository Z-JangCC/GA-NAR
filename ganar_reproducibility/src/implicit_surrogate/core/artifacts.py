from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ArtifactMismatchError
from .hashing import sha256_file, stable_json_hash
from .. import __version__
from .protocol import active_protocol_id


class ArtifactStore:
    """Immutable artifact writer with explicit smoke/formal separation."""

    def __init__(self, root: str | Path = "artifacts", *, mode: str = "smoke") -> None:
        if mode not in {"smoke", "formal"}:
            raise ValueError("mode must be smoke or formal")
        self.root = Path(root)
        self.mode = mode
        self.base = self.root / ("formal_runs" if mode == "formal" else "smoke_runs")
        self.base.mkdir(parents=True, exist_ok=True)

    def metadata(self, *, benchmark_name: str, pipeline_stage: str, random_seed: int | None, configuration: Any, input_artifact_hashes: list[str] | None = None, status: str = "CREATED") -> dict[str, Any]:
        git_commit = os.environ.get("IMPLICIT_SURROGATE_GIT_COMMIT")
        if not git_commit:
            try:
                git_commit = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                git_commit = "unavailable"
        package_versions = {}
        for package_name in ("numpy", "scipy", "torch", "PyYAML", "PYPOWER"):
            try:
                package_versions[package_name] = version(package_name)
            except PackageNotFoundError:
                package_versions[package_name] = "unavailable"
        return {
            "software_version": __version__,
            "study_protocol_id": active_protocol_id(),
            "git_commit": git_commit,
            "configuration_hash": stable_json_hash(configuration),
            "environment_hash": stable_json_hash({"python": os.sys.version, "platform": os.name, "packages": package_versions}),
            "environment": {"python": os.sys.version, "platform": os.name, "packages": package_versions},
            "benchmark_name": benchmark_name,
            "pipeline_stage": pipeline_stage,
            "random_seed": random_seed,
            "formal_flag": self.mode == "formal",
            "input_artifact_hashes": input_artifact_hashes or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }

    @staticmethod
    def write_metadata_sidecar(path: str | Path, metadata: dict[str, Any], *, resume: bool = False) -> Path:
        """Register an existing artifact without changing its payload.

        Checkpoint files are written by PyTorch rather than ``ArtifactStore``;
        this method gives them the same immutable, auditable metadata contract.
        Existing sidecars can only be reused when their configuration and
        payload hash agree.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        payload = dict(metadata)
        payload.setdefault("artifact_sha256", sha256_file(path))
        if sidecar.exists():
            existing = json.loads(sidecar.read_text())
            for key in ("configuration_hash", "formal_flag", "artifact_sha256"):
                if key in existing and existing.get(key) != payload.get(key):
                    raise ArtifactMismatchError(f"immutable artifact metadata mismatch: {path}")
            if resume:
                return sidecar
            raise FileExistsError(f"artifact metadata already exists; use an explicit resume policy: {sidecar}")
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return sidecar

    def _check_path(self, path: Path, metadata: dict[str, Any], *, resume: bool = False) -> bool:
        if path.exists():
            meta_path = path.with_suffix(path.suffix + ".metadata.json")
            if meta_path.exists():
                existing = json.loads(meta_path.read_text())
                if existing.get("configuration_hash") != metadata.get("configuration_hash") or existing.get("formal_flag") != metadata.get("formal_flag"):
                    raise ArtifactMismatchError(f"immutable artifact mismatch: {path}")
                recorded_hash = existing.get("artifact_sha256")
                if recorded_hash is not None and recorded_hash != sha256_file(path):
                    raise ArtifactMismatchError(f"artifact payload hash mismatch: {path}")
                if resume:
                    return True
            raise FileExistsError(f"artifact already exists; use an explicit resume policy: {path}")
        return False

    def write_json(self, relative_path: str, payload: Any, metadata: dict[str, Any], *, resume: bool = False) -> Path:
        path = self.base / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._check_path(path, metadata, resume=resume):
            return path
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        sidecar_metadata = dict(metadata)
        sidecar_metadata["artifact_sha256"] = sha256_file(path)
        path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(sidecar_metadata, indent=2, sort_keys=True))
        return path

    def write_npz(self, relative_path: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any], *, resume: bool = False) -> Path:
        path = self.base / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._check_path(path, metadata, resume=resume):
            return path
        np.savez_compressed(path, **arrays)
        sidecar_metadata = dict(metadata)
        sidecar_metadata["artifact_sha256"] = sha256_file(path)
        path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(sidecar_metadata, indent=2, sort_keys=True))
        return path
