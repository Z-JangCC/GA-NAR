from __future__ import annotations

import json
from pathlib import Path

from .physics_satisfaction import evaluate


def main() -> None:
    path = Path("local_training_results/physics_satisfaction.json")
    rows = json.loads(path.read_text())
    keys = {(r.get("system"), r.get("model"), r.get("seed")) for r in rows}
    for system in ("ieee118_acpf", "kuramoto64", "nonlinear_heat_fem"):
        for seed in (0, 1, 2):
            key = (system, "ganar_lw", seed)
            if key not in keys:
                rows.append(evaluate(system, "ganar_lw", seed))
    path.write_text(json.dumps(rows, indent=2))
    print(json.dumps([r for r in rows if r.get("model") == "ganar_lw"], indent=2))


if __name__ == "__main__":
    main()
