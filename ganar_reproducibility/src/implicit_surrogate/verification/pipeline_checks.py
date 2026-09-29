from __future__ import annotations

from pathlib import Path


def formal_artifact_is_separate(path: str | Path) -> bool:
    return "formal_runs" in Path(path).parts and "smoke_runs" not in Path(path).parts

