from __future__ import annotations

import numpy as np


def check_square_system(system) -> dict[str, object]:
    return {"state_dimension": int(system.state_dimension), "parameter_dimension": int(system.parameter_dimension), "square": int(system.state_dimension) == int(system.residual(system.base_state, system.base_parameter).size)}


def check_residual(system, state, parameter, tolerance: float = 1e-10) -> bool:
    return bool(np.max(np.abs(system.residual(state, parameter))) <= tolerance)

