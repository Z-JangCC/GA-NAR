from __future__ import annotations

import hashlib
from pathlib import Path


def protocol_hash(path: str | Path | None = None) -> str:
    path = Path(path) if path else Path(__file__).with_name("protocol.yaml")
    return hashlib.sha256(path.read_bytes()).hexdigest()
