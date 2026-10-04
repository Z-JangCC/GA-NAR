from __future__ import annotations

import inspect


def inference_has_no_solver_dependency(model) -> bool:
    source = inspect.getsource(model.forward)
    forbidden = ("solve_state", "Continuation", "Jacobian", "residual", "cpf")
    return not any(token in source for token in forbidden)


def paired_parameter_equal(first, second) -> bool:
    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    return first_parameters.keys() == second_parameters.keys() and all((first_parameters[name] == second_parameters[name]).all().item() for name in first_parameters)

