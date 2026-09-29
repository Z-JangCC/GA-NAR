"""Deterministic nonlinear systems used by the geometry experiments.

All systems expose the same small API:

``sample_inputs(n, seed=..., margin=...)``
    Draw normalized controls in ``[-1 + margin, 1 - margin]``.
``solve(z) -> (y, diagnostics)``
    Solve one input ``(d,)`` or a batch ``(n, d)``.  The leading shape of
    ``y`` follows ``z``; diagnostic values are always one-dimensional arrays
    of length ``n`` so callers can concatenate batches without special cases.
``residual(y, z)``
    Evaluate the defining equations without solving them.

The implementations intentionally avoid optional simulation dependencies.  In
particular, :class:`ACPowerFlowSystem` contains the public MATPOWER/PYPOWER
IEEE-9 data directly and only uses NumPy for its Newton solve.  This makes data
generation reproducible across workers and machines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Diagnostics = dict[str, NDArray[Any]]


class SystemSolveError(RuntimeError):
    """Raised when one or more samples in a nonlinear batch do not converge."""

    def __init__(
        self,
        system_name: str,
        failed_indices: NDArray[np.int64],
        diagnostics: Diagnostics,
    ) -> None:
        self.system_name = system_name
        self.failed_indices = np.asarray(failed_indices, dtype=np.int64)
        self.diagnostics = diagnostics
        indices = ", ".join(str(int(index)) for index in self.failed_indices[:10])
        if self.failed_indices.size > 10:
            indices += ", ..."
        super().__init__(
            f"{system_name} failed to solve {self.failed_indices.size} sample(s) "
            f"at batch indices [{indices}]; call with raise_on_failure=False "
            "to inspect all returned iterates and diagnostics"
        )


def _as_batch(values: ArrayLike, width: int, label: str) -> tuple[FloatArray, bool]:
    array = np.asarray(values, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{label} must have shape ({width},) or (n, {width}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return np.ascontiguousarray(array), single


def _paired_batches(
    y: ArrayLike,
    output_dim: int,
    z: ArrayLike,
    input_dim: int,
) -> tuple[FloatArray, FloatArray, bool]:
    y_batch, y_single = _as_batch(y, output_dim, "y")
    z_batch, z_single = _as_batch(z, input_dim, "z")
    if y_batch.shape[0] != z_batch.shape[0]:
        raise ValueError(
            "y and z must contain the same number of samples, got "
            f"{y_batch.shape[0]} and {z_batch.shape[0]}"
        )
    if y_single != z_single:
        raise ValueError("y and z must either both be single samples or both be batches")
    return y_batch, z_batch, y_single


def _finish(values: FloatArray, single: bool) -> FloatArray:
    return values[0] if single else values


def _make_rng(
    seed: int | np.random.Generator | None,
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        if rng is not None:
            raise ValueError("pass either seed or rng, not both")
        return seed
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        return rng
    return np.random.default_rng(seed)


def _diagnostics(
    success: NDArray[np.bool_],
    iterations: NDArray[np.int64],
    residual_norm: FloatArray,
    messages: list[str],
    **extra: ArrayLike,
) -> Diagnostics:
    result: Diagnostics = {
        "success": np.asarray(success, dtype=np.bool_),
        # ``converged`` is a readable alias retained alongside ``success``.
        "converged": np.asarray(success, dtype=np.bool_).copy(),
        "iterations": np.asarray(iterations, dtype=np.int64),
        "residual_norm": np.asarray(residual_norm, dtype=np.float64),
        "message": np.asarray(messages, dtype=object),
    }
    for key, value in extra.items():
        result[key] = np.asarray(value)
    return result


def _raise_failed(name: str, diagnostics: Diagnostics, enabled: bool) -> None:
    failed = np.flatnonzero(~np.asarray(diagnostics["success"], dtype=bool))
    if enabled and failed.size:
        raise SystemSolveError(name, failed.astype(np.int64), diagnostics)


class BaseSystem(ABC):
    """Shared interface for normalized-input implicit systems."""

    name: str
    input_dim: int
    output_dim: int

    def sample_inputs(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        *,
        rng: np.random.Generator | None = None,
        margin: float = 0.05,
    ) -> FloatArray:
        """Sample normalized controls, reproducibly when ``seed`` is supplied."""

        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if not np.isfinite(margin) or not 0.0 <= margin < 1.0:
            raise ValueError("margin must satisfy 0 <= margin < 1")
        generator = _make_rng(seed, rng)
        bound = 1.0 - float(margin)
        return generator.uniform(-bound, bound, size=(int(n), self.input_dim))

    @abstractmethod
    def solve(
        self,
        z: ArrayLike,
        *,
        tol: float = 1e-10,
        max_iter: int = 50,
        raise_on_failure: bool = True,
    ) -> tuple[FloatArray, Diagnostics]:
        """Solve the system at one or more normalized inputs."""

    @abstractmethod
    def residual(self, y: ArrayLike, z: ArrayLike) -> FloatArray:
        """Evaluate the independent equations defining the returned state."""

    def residual_norm(self, y: ArrayLike, z: ArrayLike) -> FloatArray | float:
        """Return the per-sample infinity norm of :meth:`residual`."""

        values = np.asarray(self.residual(y, z), dtype=np.float64)
        if values.ndim == 1:
            return float(np.linalg.norm(values, ord=np.inf))
        return np.linalg.norm(values, ord=np.inf, axis=1)

    def local_linearization(self, step: float = 1e-4) -> tuple[FloatArray, FloatArray]:
        """Reference equilibrium and finite-difference input sensitivity.

        This common implementation makes physics-skip comparisons available
        for every benchmark without exposing test targets: only the governing
        solver at the nominal input and symmetric perturbations are used.
        """
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive")
        zero = np.zeros(self.input_dim, dtype=np.float64)
        base, _ = self.solve(zero)
        sensitivity = np.empty((self.output_dim, self.input_dim), dtype=np.float64)
        for column in range(self.input_dim):
            plus = zero.copy(); minus = zero.copy()
            plus[column] = step; minus[column] = -step
            y_plus, _ = self.solve(plus); y_minus, _ = self.solve(minus)
            sensitivity[:, column] = (y_plus - y_minus) / (2.0 * step)
        return np.asarray(base, dtype=np.float32), sensitivity.astype(np.float32)


class ControlledImplicitSystem(BaseSystem):
    r"""Controlled implicit benchmark with optional input interactions.

    The equation is

    ``A x + alpha*tanh(l*(B x+C mu)) + D mu``
    ``+ beta*E[(P mu) * (Q mu)] = 0``,

    where ``*`` is elementwise multiplication and ``beta`` is exposed as the
    ``interaction`` constructor argument (``beta`` is accepted as an alias).

    ``coupling`` rotates a fixed-rank diagonal nonlinear operator into dense
    input/state directions without changing its spectral norm.  Thus coupling,
    rank, strength, conditioning and localization can be swept one at a time.
    The matrices are built so the exact global fixed-point bound is
    ``alpha * localization``; values at or above one are rejected.
    """

    name = "controlled_implicit"

    def __init__(
        self,
        state_dim: int = 8,
        input_dim: int = 5,
        *,
        alpha: float = 0.45,
        coupling: float = 0.5,
        rank: int | None = None,
        condition: float = 5.0,
        localization: float = 1.0,
        interaction: float = 0.5,
        beta: float | None = None,
        matrix_seed: int = 1729,
    ) -> None:
        if state_dim < 1 or input_dim < 1:
            raise ValueError("state_dim and input_dim must be positive")
        if rank is None:
            rank = state_dim
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
            raise TypeError("rank must be an integer")
        if not 1 <= int(rank) <= state_dim:
            raise ValueError("rank must satisfy 1 <= rank <= state_dim")
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not np.isfinite(coupling) or not 0.0 <= coupling <= 1.0:
            raise ValueError("coupling must lie in [0, 1]")
        if not np.isfinite(condition) or condition < 1.0:
            raise ValueError("condition must be finite and at least one")
        if not np.isfinite(localization) or localization <= 0.0:
            raise ValueError("localization must be finite and positive")
        if beta is not None:
            if interaction != 0.5 and not np.isclose(interaction, beta):
                raise ValueError("interaction and beta aliases disagree")
            interaction = float(beta)
        if not np.isfinite(interaction) or not 0.0 <= interaction <= 1.0:
            raise ValueError("interaction must lie in [0, 1]")

        self.input_dim = int(input_dim)
        self.output_dim = int(state_dim)
        self.state_dim = int(state_dim)
        self.alpha = float(alpha)
        self.coupling = float(coupling)
        self.rank = int(rank)
        self.condition = float(condition)
        self.localization = float(localization)
        self.interaction = float(interaction)
        self.beta = self.interaction
        self.matrix_seed = int(matrix_seed)

        generator = np.random.default_rng(self.matrix_seed)
        identity_basis = np.eye(self.state_dim)
        random_u, _ = np.linalg.qr(generator.normal(size=(self.state_dim, self.state_dim)))
        random_v, _ = np.linalg.qr(generator.normal(size=(self.state_dim, self.state_dim)))

        # Continuous paths from coordinate directions to dense directions.
        raw_u = (
            (1.0 - self.coupling) * identity_basis[:, : self.rank]
            + self.coupling * random_u[:, : self.rank]
        )
        raw_v = (
            (1.0 - self.coupling) * identity_basis[:, : self.rank]
            + self.coupling * random_v[:, : self.rank]
        )
        basis_u, _ = np.linalg.qr(raw_u, mode="reduced")
        basis_v, _ = np.linalg.qr(raw_v, mode="reduced")
        singular_values = np.linspace(1.0, 0.4, self.rank)
        self.nonlinear_operator = (
            basis_u * singular_values[None, :]
        ) @ basis_v.T

        # A is diagonal so A^{-1} diag(tanh') A commutes exactly.  This makes
        # alpha*localization*||nonlinear_operator|| the rigorous contraction
        # bound even as condition(A) is varied.
        diagonal = np.geomspace(1.0, self.condition, self.state_dim)
        self.A = np.diag(diagonal)
        self.B = self.A @ self.nonlinear_operator
        c_base = generator.normal(size=(self.state_dim, self.input_dim))
        c_base *= 0.65 / np.sqrt(self.input_dim)
        d_base = generator.normal(size=(self.state_dim, self.input_dim))
        d_base *= 0.35 / np.sqrt(self.input_dim)
        self.C = self.A @ c_base
        self.D = self.A @ d_base

        # Fixed normalized factors for the explicit multiplicative-input
        # intervention.  Folding A into E keeps its state-space scale comparable
        # as condition(A) changes, while the term remains independent of x.
        self.P = generator.normal(size=(self.state_dim, self.input_dim))
        self.Q = generator.normal(size=(self.state_dim, self.input_dim))
        self.P /= np.maximum(np.linalg.norm(self.P, axis=1, keepdims=True), 1e-12)
        self.Q /= np.maximum(np.linalg.norm(self.Q, axis=1, keepdims=True), 1e-12)
        interaction_mix = generator.normal(size=(self.state_dim, self.state_dim))
        interaction_mix *= 0.25 / np.linalg.norm(interaction_mix, ord=2)
        self.E = self.A @ interaction_mix

        self.contraction_bound = float(
            self.alpha
            * self.localization
            * np.linalg.norm(self.nonlinear_operator, ord=2)
        )
        if self.contraction_bound >= 1.0 - 1e-12:
            raise ValueError(
                "global uniqueness is not certified: require "
                "alpha * localization * ||A^-1 B||_2 < 1; got "
                f"bound={self.contraction_bound:.6g}"
            )

    def _interaction_one(self, mu: FloatArray) -> FloatArray:
        product = (self.P @ mu) * (self.Q @ mu)
        return self.interaction * (self.E @ product)

    def _residual_one(self, x: FloatArray, mu: FloatArray) -> FloatArray:
        argument = self.localization * (self.B @ x + self.C @ mu)
        return (
            self.A @ x
            + self.alpha * np.tanh(argument)
            + self.D @ mu
            + self._interaction_one(mu)
        )

    def residual(self, y: ArrayLike, z: ArrayLike) -> FloatArray:
        y_batch, z_batch, single = _paired_batches(
            y, self.output_dim, z, self.input_dim
        )
        linear = y_batch @ self.A.T + z_batch @ self.D.T
        arguments = self.localization * (
            y_batch @ self.B.T + z_batch @ self.C.T
        )
        products = (z_batch @ self.P.T) * (z_batch @ self.Q.T)
        interaction_values = self.interaction * (products @ self.E.T)
        values = linear + self.alpha * np.tanh(arguments) + interaction_values
        return _finish(values, single)

    def solve(
        self,
        z: ArrayLike,
        *,
        tol: float = 1e-10,
        max_iter: int = 50,
        raise_on_failure: bool = True,
    ) -> tuple[FloatArray, Diagnostics]:
        if tol <= 0.0 or not np.isfinite(tol):
            raise ValueError("tol must be finite and positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        z_batch, single = _as_batch(z, self.input_dim, "z")
        n = z_batch.shape[0]
        solutions = np.empty((n, self.output_dim), dtype=np.float64)
        success = np.zeros(n, dtype=np.bool_)
        iterations = np.zeros(n, dtype=np.int64)
        norms = np.full(n, np.inf, dtype=np.float64)
        jacobian_condition = np.full(n, np.nan, dtype=np.float64)
        messages: list[str] = []

        for sample, mu in enumerate(z_batch):
            # Exact solution of the alpha=0 system is a reliable deterministic
            # initialization for every factor sweep.
            x = -np.linalg.solve(
                self.A, self.D @ mu + self._interaction_one(mu)
            )
            message = "maximum iterations reached"
            for iteration in range(max_iter + 1):
                value = self._residual_one(x, mu)
                norm = float(np.linalg.norm(value, ord=np.inf))
                if not np.isfinite(norm):
                    message = "non-finite residual"
                    break
                if norm <= tol:
                    success[sample] = True
                    iterations[sample] = iteration
                    norms[sample] = norm
                    message = "converged"
                    break
                if iteration == max_iter:
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break

                argument = self.localization * (self.B @ x + self.C @ mu)
                sech_squared = 1.0 - np.tanh(argument) ** 2
                jacobian = self.A + (
                    self.alpha
                    * self.localization
                    * sech_squared[:, None]
                    * self.B
                )
                try:
                    step = np.linalg.solve(jacobian, -value)
                except np.linalg.LinAlgError:
                    message = "singular Newton Jacobian"
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break
                jacobian_condition[sample] = np.linalg.cond(jacobian)

                accepted = False
                damping = 1.0
                for _ in range(16):
                    candidate = x + damping * step
                    candidate_norm = float(
                        np.linalg.norm(self._residual_one(candidate, mu), ord=np.inf)
                    )
                    if np.isfinite(candidate_norm) and candidate_norm < norm:
                        x = candidate
                        accepted = True
                        break
                    damping *= 0.5
                if not accepted:
                    message = "Newton line search failed"
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break
            solutions[sample] = x
            messages.append(message)

        diagnostics = _diagnostics(
            success,
            iterations,
            norms,
            messages,
            jacobian_condition=jacobian_condition,
            contraction_bound=np.full(n, self.contraction_bound),
        )
        _raise_failed(self.name, diagnostics, raise_on_failure)
        return _finish(solutions, single), diagnostics


class CoupledDuffingSystem(BaseSystem):
    r"""Static equilibrium of a coupled Duffing oscillator chain.

    The physical balance law is

    ``K x + beta * x**3 = F u``

    where ``K`` contains local linear stiffness and nearest-neighbour coupling.
    This is the zero-velocity, zero-acceleration equilibrium of mechanically
    coupled Duffing oscillators.  Positive stiffness and ``beta`` make the
    operator strongly monotone, so every sampled force has a unique stable
    equilibrium while the cubic term creates a controllable high-curvature
    solution map.  The input is the normalized external-force vector ``u``.
    """

    name = "coupled_duffing"

    def __init__(
        self,
        n_oscillators: int = 8,
        input_dim: int | None = None,
        *,
        stiffness: float = 1.0,
        coupling: float = 0.35,
        cubic_stiffness: float = 1.8,
        force_scale: float = 1.0,
        matrix_seed: int = 2718,
    ) -> None:
        if n_oscillators < 2:
            raise ValueError("n_oscillators must be at least two")
        if input_dim is None:
            input_dim = n_oscillators
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        for name, value in (("stiffness", stiffness), ("coupling", coupling),
                            ("cubic_stiffness", cubic_stiffness),
                            ("force_scale", force_scale)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.state_dim = int(n_oscillators)
        self.input_dim = int(input_dim)
        self.output_dim = self.state_dim
        self.stiffness = float(stiffness)
        self.coupling = float(coupling)
        self.cubic_stiffness = float(cubic_stiffness)
        self.force_scale = float(force_scale)
        self.matrix_seed = int(matrix_seed)

        generator = np.random.default_rng(self.matrix_seed)
        force = generator.normal(size=(self.state_dim, self.input_dim))
        force /= np.maximum(np.linalg.norm(force, axis=0, keepdims=True), 1e-12)
        self.force_matrix = self.force_scale * force
        laplacian = 2.0 * np.eye(self.state_dim)
        laplacian -= np.eye(self.state_dim, k=1) + np.eye(self.state_dim, k=-1)
        laplacian[0, 0] = laplacian[-1, -1] = 1.0
        self.K = self.stiffness * np.eye(self.state_dim) + self.coupling * laplacian

    def residual(self, y: ArrayLike, z: ArrayLike) -> FloatArray:
        y_batch, z_batch, single = _paired_batches(
            y, self.output_dim, z, self.input_dim
        )
        values = y_batch @ self.K.T + self.cubic_stiffness * y_batch**3
        values -= z_batch @ self.force_matrix.T
        return _finish(values, single)

    def solve(
        self,
        z: ArrayLike,
        *,
        tol: float = 1e-10,
        max_iter: int = 50,
        raise_on_failure: bool = True,
    ) -> tuple[FloatArray, Diagnostics]:
        if tol <= 0.0 or not np.isfinite(tol):
            raise ValueError("tol must be finite and positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        z_batch, single = _as_batch(z, self.input_dim, "z")
        n = len(z_batch)
        solutions = np.empty((n, self.output_dim), dtype=np.float64)
        success = np.zeros(n, dtype=bool)
        iterations = np.zeros(n, dtype=np.int64)
        norms = np.full(n, np.inf)
        jacobian_condition = np.full(n, np.nan)
        messages: list[str] = []
        for index, force_input in enumerate(z_batch):
            x = np.linalg.solve(self.K, self.force_matrix @ force_input)
            message = "maximum iterations reached"
            for iteration in range(max_iter + 1):
                value = self.residual(x, force_input)
                norm = float(np.linalg.norm(value, ord=np.inf))
                if not np.isfinite(norm):
                    message = "non-finite residual"
                    break
                if norm <= tol:
                    success[index] = True
                    iterations[index] = iteration
                    norms[index] = norm
                    message = "converged"
                    break
                if iteration == max_iter:
                    iterations[index] = iteration
                    norms[index] = norm
                    break
                jacobian = self.K + np.diag(3.0 * self.cubic_stiffness * x**2)
                jacobian_condition[index] = np.linalg.cond(jacobian)
                step = np.linalg.solve(jacobian, -value)
                damping = 1.0
                accepted = False
                for _ in range(16):
                    candidate = x + damping * step
                    candidate_norm = float(np.linalg.norm(self.residual(candidate, force_input), ord=np.inf))
                    if np.isfinite(candidate_norm) and candidate_norm < norm:
                        x = candidate
                        accepted = True
                        break
                    damping *= 0.5
                if not accepted:
                    message = "Duffing Newton line search failed"
                    iterations[index] = iteration
                    norms[index] = norm
                    break
            solutions[index] = x
            messages.append(message)
        diagnostics = _diagnostics(
            success, iterations, norms, messages,
            jacobian_condition=jacobian_condition,
            max_displacement=np.max(np.abs(solutions), axis=1),
        )
        _raise_failed(self.name, diagnostics, raise_on_failure)
        return _finish(solutions, single), diagnostics


class IEEE118PowerFlowSystem(BaseSystem):
    """Large IEEE-118-bus nonlinear AC phase-balance benchmark.

    The network topology is loaded from the standard pandapower IEEE-118 case.
    We retain the full 118-bus graph and solve the nonlinear active-power
    balance ``sum_j b_ij sin(theta_i-theta_j)=p_i(u)`` with one slack angle.
    This is the angle-only AC power-flow manifold; unlike a toy small system,
    its state dimension is 117 and its sparse graph/topology are fixed by the
    published IEEE-118 benchmark.
    """

    name = "ac_power_flow_ieee118"
    input_dim = 16

    def __init__(self, *, load_variation: float = 0.20, seed: int = 118) -> None:
        import pandapower.networks as networks
        net = networks.case118()
        self.n_bus = int(len(net.bus))
        self.output_dim = self.n_bus - 1
        self.input_dim = 16
        self.load_variation = float(load_variation)
        self.ref = 0
        rng = np.random.default_rng(seed)
        # Positive susceptance graph from line reactances; transformers are
        # folded into the same sparse weighted adjacency.
        weights = np.zeros((self.n_bus, self.n_bus), dtype=np.float64)
        for _, row in net.line.iterrows():
            i, j = int(row.from_bus), int(row.to_bus)
            b = 1.0 / max(abs(float(row.x_ohm_per_km * row.length_km)), 1e-3)
            weights[i, j] = weights[j, i] = b
        for _, row in net.trafo.iterrows():
            i, j = int(row.hv_bus), int(row.lv_bus)
            b = 1.0 / max(abs(float(row.vk_percent)), 1e-3)
            weights[i, j] = weights[j, i] = b
        weights /= max(np.median(weights[weights > 0]), 1e-12)
        self.weights = weights
        self.base_injection = np.zeros(self.n_bus, dtype=np.float64)
        for _, row in net.load.iterrows():
            self.base_injection[int(row.bus)] -= (float(row.p_mw) / 10000.0)
        for _, row in net.gen.iterrows():
            self.base_injection[int(row.bus)] += (float(row.p_mw) / 10000.0)
        for _, row in net.sgen.iterrows():
            self.base_injection[int(row.bus)] += (float(row.p_mw) / 10000.0)
        self.control_map = rng.normal(size=(self.n_bus, self.input_dim))
        self.control_map[0] = 0.0
        self.control_map /= np.maximum(np.linalg.norm(self.control_map, axis=0, keepdims=True), 1e-12)
        self.control_map *= 0.03

    def _full_residual(self, theta: FloatArray, z: FloatArray) -> FloatArray:
        diff = theta[:, None] - theta[None, :]
        flow = np.sum(self.weights * np.sin(diff), axis=1)
        injection = self.base_injection + self.load_variation * (self.control_map @ z)
        injection[0] = -np.sum(injection[1:])
        return flow - injection

    def residual(self, y: ArrayLike, z: ArrayLike) -> FloatArray:
        yb, zb, single = _paired_batches(y, self.output_dim, z, self.input_dim)
        theta = np.zeros((len(yb), self.n_bus), dtype=np.float64); theta[:, 1:] = yb
        result = np.stack([self._full_residual(t, u)[1:] for t, u in zip(theta, zb)])
        return _finish(result, single)

    def solve(self, z: ArrayLike, *, tol: float = 1e-9, max_iter: int = 40,
              raise_on_failure: bool = True) -> tuple[FloatArray, Diagnostics]:
        zb, single = _as_batch(z, self.input_dim, "z"); n = len(zb)
        sol = np.zeros((n, self.output_dim)); success = np.zeros(n, dtype=bool)
        iterations = np.zeros(n, dtype=np.int64); norms = np.full(n, np.inf)
        for k, u in enumerate(zb):
            theta = np.zeros(self.n_bus)
            for it in range(max_iter + 1):
                value = self._full_residual(theta, u)[1:]; norm = float(np.max(np.abs(value)))
                if norm <= tol:
                    success[k] = True; iterations[k] = it; norms[k] = norm; break
                if it == max_iter: iterations[k] = it; norms[k] = norm; break
                diff = theta[:, None] - theta[None, :]
                c = self.weights * np.cos(diff)
                jac = -c; jac[np.diag_indices(self.n_bus)] = -np.sum(jac, axis=1)
                step = np.linalg.solve(jac[1:, 1:], -value); damping = 1.0
                for _ in range(12):
                    candidate = theta.copy(); candidate[1:] += damping * step
                    if np.max(np.abs(self._full_residual(candidate, u)[1:])) < norm:
                        theta = candidate; break
                    damping *= 0.5
            sol[k] = theta[1:]
        diagnostics = _diagnostics(success, iterations, norms, ["converged" if x else "maximum iterations reached" for x in success])
        _raise_failed(self.name, diagnostics, raise_on_failure)
        return _finish(sol, single), diagnostics

    def local_linearization(self, step: float = 1e-4) -> tuple[FloatArray, FloatArray]:
        zero = np.zeros(self.input_dim, dtype=np.float64)
        base, _ = self.solve(zero)
        sensitivity = np.empty((self.output_dim, self.input_dim), dtype=np.float64)
        for j in range(self.input_dim):
            plus = zero.copy(); minus = zero.copy(); plus[j] = step; minus[j] = -step
            yp, _ = self.solve(plus); ym, _ = self.solve(minus)
            sensitivity[:, j] = (yp - ym) / (2.0 * step)
        return np.asarray(base, dtype=np.float32), sensitivity.astype(np.float32)


class NonlinearGridPhysicsSystem(BaseSystem):
    """Large non-power nonlinear PDE equilibrium on a regular 2-D grid."""

    def __init__(self, kind: str, *, grid_size: int = 8, input_dim: int = 12,
                 diffusion: float = 0.12, nonlinearity: float = 0.8) -> None:
        if kind not in {"allen_cahn", "shallow_water"}:
            raise ValueError("kind must be allen_cahn or shallow_water")
        self.kind = kind; self.name = kind + "_grid"; self.grid_size = int(grid_size)
        self.input_dim = int(input_dim); self.output_dim = grid_size * grid_size
        self.diffusion = float(diffusion); self.nonlinearity = float(nonlinearity)
        n = self.output_dim; lap = np.zeros((n, n));
        for i in range(grid_size):
            for j in range(grid_size):
                q = i * grid_size + j; lap[q, q] = 4.0
                for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    ni, nj = i+di, j+dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size: lap[q, ni*grid_size+nj] = -1.0
        self.laplacian = lap
        rng = np.random.default_rng(2026 + n); self.forcing = rng.normal(size=(n, input_dim))
        self.forcing /= np.maximum(np.linalg.norm(self.forcing, axis=0, keepdims=True), 1e-12)
        self.forcing *= 0.35

    def residual(self, y: ArrayLike, z: ArrayLike) -> FloatArray:
        yb, zb, single = _paired_batches(y, self.output_dim, z, self.input_dim)
        linear = yb + self.diffusion * (yb @ self.laplacian.T)
        if self.kind == "allen_cahn": nonlinear = self.nonlinearity * (yb**3 - yb)
        else: nonlinear = self.nonlinearity * (yb**2)
        return _finish(linear + nonlinear - zb @ self.forcing.T, single)

    def solve(self, z: ArrayLike, *, tol: float = 1e-9, max_iter: int = 40,
              raise_on_failure: bool = True) -> tuple[FloatArray, Diagnostics]:
        zb, single = _as_batch(z, self.input_dim, "z"); n = len(zb); sol = np.zeros((n, self.output_dim))
        success = np.zeros(n, dtype=bool); iterations = np.zeros(n, dtype=np.int64); norms = np.full(n, np.inf)
        for k, u in enumerate(zb):
            x = np.linalg.solve(np.eye(self.output_dim) + self.diffusion*self.laplacian, self.forcing @ u)
            for it in range(max_iter + 1):
                value = self.residual(x, u); norm = float(np.max(np.abs(value)))
                if norm <= tol: success[k]=True; iterations[k]=it; norms[k]=norm; break
                if it == max_iter: iterations[k]=it; norms[k]=norm; break
                diag = 1.0 + self.diffusion*np.diag(self.laplacian) + (self.nonlinearity*(3*x*x-1) if self.kind == "allen_cahn" else self.nonlinearity*2*x)
                jac = np.eye(self.output_dim) + self.diffusion*self.laplacian + np.diag(diag - 1.0 - self.diffusion*np.diag(self.laplacian))
                step = np.linalg.solve(jac, -value); x = x + step
            sol[k] = x
        diagnostics = _diagnostics(success, iterations, norms, ["converged" if x else "maximum iterations reached" for x in success])
        _raise_failed(self.name, diagnostics, raise_on_failure); return _finish(sol, single), diagnostics

    def local_linearization(self, step: float = 1e-4) -> tuple[FloatArray, FloatArray]:
        zero = np.zeros(self.input_dim, dtype=np.float64)
        base, _ = self.solve(zero)
        sensitivity = np.empty((self.output_dim, self.input_dim), dtype=np.float64)
        for j in range(self.input_dim):
            plus = zero.copy(); minus = zero.copy(); plus[j] = step; minus[j] = -step
            yp, _ = self.solve(plus); ym, _ = self.solve(minus)
            sensitivity[:, j] = (yp - ym) / (2.0 * step)
        return np.asarray(base, dtype=np.float32), sensitivity.astype(np.float32)


class ACPowerFlowSystem(BaseSystem):
    """IEEE case9 AC power flow with normalized load/generator controls.

    The 11 controls are, in order, independent P-load scales at buses 5/7/9,
    Q-load scales at buses 5/7/9, active-generation scales at PV buses 2/3,
    and voltage-setpoint offsets at buses 1/2/3.  A normalized value ``z`` maps
    to ``base * (1 + variation*z)`` for powers and ``1 + variation*z`` for
    voltage.  The output contains eight non-reference angles (radians), then
    six PQ-bus voltage magnitudes.  The residual is the corresponding P/Q
    mismatch vector.
    """

    name = "ac_power_flow_case9"
    input_dim = 11
    output_dim = 14

    control_names = (
        "p_load_bus5",
        "p_load_bus7",
        "p_load_bus9",
        "q_load_bus5",
        "q_load_bus7",
        "q_load_bus9",
        "p_gen_bus2",
        "p_gen_bus3",
        "v_set_bus1",
        "v_set_bus2",
        "v_set_bus3",
    )

    def __init__(
        self,
        *,
        load_variation: float = 0.25,
        generation_variation: float = 0.15,
        voltage_variation: float = 0.025,
    ) -> None:
        for label, value in (
            ("load_variation", load_variation),
            ("generation_variation", generation_variation),
            ("voltage_variation", voltage_variation),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        self.load_variation = float(load_variation)
        self.generation_variation = float(generation_variation)
        self.voltage_variation = float(voltage_variation)
        self.base_mva = 100.0
        self.n_bus = 9
        self.ref = np.array([0], dtype=np.int64)
        self.pv = np.array([1, 2], dtype=np.int64)
        self.pq = np.arange(3, 9, dtype=np.int64)
        self.nonref = np.arange(1, 9, dtype=np.int64)
        self.load_buses = np.array([4, 6, 8], dtype=np.int64)

        self.base_p_load = np.array([90.0, 100.0, 125.0]) / self.base_mva
        self.base_q_load = np.array([30.0, 35.0, 50.0]) / self.base_mva
        self.base_p_gen_pv = np.array([163.0, 85.0]) / self.base_mva
        self.base_voltage_setpoint = np.ones(3, dtype=np.float64)
        self.q_min = np.full(3, -300.0 / self.base_mva)
        self.q_max = np.full(3, 300.0 / self.base_mva)
        self.p_slack_min = 10.0 / self.base_mva
        self.p_slack_max = 250.0 / self.base_mva

        # MATPOWER case9 branches: from, to, r, x, total line charging b.
        branches = np.array(
            [
                [1, 4, 0.0000, 0.0576, 0.0000],
                [4, 5, 0.0170, 0.0920, 0.1580],
                [5, 6, 0.0390, 0.1700, 0.3580],
                [3, 6, 0.0000, 0.0586, 0.0000],
                [6, 7, 0.0119, 0.1008, 0.2090],
                [7, 8, 0.0085, 0.0720, 0.1490],
                [8, 2, 0.0000, 0.0625, 0.0000],
                [8, 9, 0.0320, 0.1610, 0.3060],
                [9, 4, 0.0100, 0.0850, 0.1760],
            ],
            dtype=np.float64,
        )
        self.branches = branches.copy()
        self.branches[:, :2] -= 1.0
        self.ybus = self._build_ybus(self.branches)
        self.g = self.ybus.real
        self.b = self.ybus.imag

        # Initial voltage angles provided with the standard case (degrees).
        self._initial_angle = np.deg2rad(
            np.array([0.0, 9.3, 4.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

    def _build_ybus(self, branches: FloatArray) -> NDArray[np.complex128]:
        ybus = np.zeros((self.n_bus, self.n_bus), dtype=np.complex128)
        for from_bus, to_bus, resistance, reactance, charging in branches:
            i, j = int(from_bus), int(to_bus)
            admittance = 1.0 / complex(resistance, reactance)
            shunt = 0.5j * charging
            ybus[i, i] += admittance + shunt
            ybus[j, j] += admittance + shunt
            ybus[i, j] -= admittance
            ybus[j, i] -= admittance
        return ybus

    def physical_controls(self, z: ArrayLike) -> dict[str, FloatArray]:
        """Convert normalized controls to per-unit specifications."""

        z_batch, single = _as_batch(z, self.input_dim, "z")
        p_load = self.base_p_load[None, :] * (
            1.0 + self.load_variation * z_batch[:, 0:3]
        )
        q_load = self.base_q_load[None, :] * (
            1.0 + self.load_variation * z_batch[:, 3:6]
        )
        p_gen_pv = self.base_p_gen_pv[None, :] * (
            1.0 + self.generation_variation * z_batch[:, 6:8]
        )
        voltage = self.base_voltage_setpoint[None, :] + (
            self.voltage_variation * z_batch[:, 8:11]
        )
        return {
            "p_load": _finish(p_load, single),
            "q_load": _finish(q_load, single),
            "p_gen_pv": _finish(p_gen_pv, single),
            "voltage_setpoint": _finish(voltage, single),
        }

    def local_linearization(self, step: float = 1e-4) -> tuple[FloatArray, FloatArray]:
        """Return the base solution and d solution / d normalized controls.

        The Jacobian is estimated by symmetric solves rather than by exposing
        solver internals.  This gives the AC-specific residual branch a stable
        angle/voltage operating-point prior while preserving a purely
        differentiable trainable correction.
        """
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive")
        zero = np.zeros(self.input_dim, dtype=np.float64)
        base, diagnostics = self.solve(zero)
        if not bool(np.asarray(diagnostics["success"])[0]):
            raise RuntimeError("base AC power-flow solve failed")
        sensitivity = np.empty((self.output_dim, self.input_dim), dtype=np.float64)
        for column in range(self.input_dim):
            plus = zero.copy(); minus = zero.copy()
            plus[column] = step; minus[column] = -step
            y_plus, d_plus = self.solve(plus)
            y_minus, d_minus = self.solve(minus)
            if not (bool(np.asarray(d_plus["success"])[0]) and bool(np.asarray(d_minus["success"])[0])):
                raise RuntimeError(f"AC linearization failed for control {column}")
            sensitivity[:, column] = (y_plus - y_minus) / (2.0 * step)
        return np.asarray(base, dtype=np.float32), sensitivity.astype(np.float32)

    def _specifications(
        self, z: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        p_load_values = self.base_p_load * (1.0 + self.load_variation * z[0:3])
        q_load_values = self.base_q_load * (1.0 + self.load_variation * z[3:6])
        p_gen_values = self.base_p_gen_pv * (
            1.0 + self.generation_variation * z[6:8]
        )
        voltage_setpoints = self.base_voltage_setpoint + self.voltage_variation * z[8:11]

        p_load = np.zeros(self.n_bus, dtype=np.float64)
        q_load = np.zeros(self.n_bus, dtype=np.float64)
        p_load[self.load_buses] = p_load_values
        q_load[self.load_buses] = q_load_values
        p_spec = -p_load
        p_spec[self.pv] += p_gen_values
        q_spec = -q_load
        return p_spec, q_spec, voltage_setpoints, p_load, q_load

    def _unpack_state(
        self, y: FloatArray, voltage_setpoints: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        angle = np.zeros(self.n_bus, dtype=np.float64)
        magnitude = np.ones(self.n_bus, dtype=np.float64)
        angle[self.nonref] = y[: self.nonref.size]
        magnitude[np.r_[self.ref, self.pv]] = voltage_setpoints
        magnitude[self.pq] = y[self.nonref.size :]
        return angle, magnitude

    def _injections(
        self, angle: FloatArray, magnitude: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        voltage = magnitude * np.exp(1j * angle)
        power = voltage * np.conjugate(self.ybus @ voltage)
        return power.real, power.imag

    def _mismatch_one(self, y: FloatArray, z: FloatArray) -> FloatArray:
        p_spec, q_spec, voltage_setpoints, _, _ = self._specifications(z)
        angle, magnitude = self._unpack_state(y, voltage_setpoints)
        p_calc, q_calc = self._injections(angle, magnitude)
        return np.concatenate(
            (p_spec[self.nonref] - p_calc[self.nonref], q_spec[self.pq] - q_calc[self.pq])
        )

    def residual(self, y: ArrayLike, z: ArrayLike) -> FloatArray:
        y_batch, z_batch, single = _paired_batches(
            y, self.output_dim, z, self.input_dim
        )
        result = np.stack(
            [self._mismatch_one(state, control) for state, control in zip(y_batch, z_batch)]
        )
        return _finish(result, single)

    def _jacobian(
        self,
        angle: FloatArray,
        magnitude: FloatArray,
        p_calc: FloatArray,
        q_calc: FloatArray,
    ) -> FloatArray:
        n = self.n_bus
        d_p_angle = np.empty((n, n), dtype=np.float64)
        d_q_angle = np.empty((n, n), dtype=np.float64)
        d_p_voltage = np.empty((n, n), dtype=np.float64)
        d_q_voltage = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    d_p_angle[i, i] = -q_calc[i] - self.b[i, i] * magnitude[i] ** 2
                    d_q_angle[i, i] = p_calc[i] - self.g[i, i] * magnitude[i] ** 2
                    d_p_voltage[i, i] = p_calc[i] / magnitude[i] + self.g[i, i] * magnitude[i]
                    d_q_voltage[i, i] = q_calc[i] / magnitude[i] - self.b[i, i] * magnitude[i]
                else:
                    difference = angle[i] - angle[j]
                    sine = np.sin(difference)
                    cosine = np.cos(difference)
                    d_p_angle[i, j] = magnitude[i] * magnitude[j] * (
                        self.g[i, j] * sine - self.b[i, j] * cosine
                    )
                    d_q_angle[i, j] = -magnitude[i] * magnitude[j] * (
                        self.g[i, j] * cosine + self.b[i, j] * sine
                    )
                    d_p_voltage[i, j] = magnitude[i] * (
                        self.g[i, j] * cosine + self.b[i, j] * sine
                    )
                    d_q_voltage[i, j] = magnitude[i] * (
                        self.g[i, j] * sine - self.b[i, j] * cosine
                    )
        top = np.hstack(
            (
                d_p_angle[np.ix_(self.nonref, self.nonref)],
                d_p_voltage[np.ix_(self.nonref, self.pq)],
            )
        )
        bottom = np.hstack(
            (
                d_q_angle[np.ix_(self.pq, self.nonref)],
                d_q_voltage[np.ix_(self.pq, self.pq)],
            )
        )
        return np.vstack((top, bottom))

    def solve(
        self,
        z: ArrayLike,
        *,
        tol: float = 1e-10,
        max_iter: int = 30,
        raise_on_failure: bool = True,
    ) -> tuple[FloatArray, Diagnostics]:
        if tol <= 0.0 or not np.isfinite(tol):
            raise ValueError("tol must be finite and positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        z_batch, single = _as_batch(z, self.input_dim, "z")
        n = z_batch.shape[0]
        solutions = np.empty((n, self.output_dim), dtype=np.float64)
        success = np.zeros(n, dtype=np.bool_)
        iterations = np.zeros(n, dtype=np.int64)
        norms = np.full(n, np.inf, dtype=np.float64)
        jacobian_condition = np.full(n, np.nan, dtype=np.float64)
        pv_voltage_error = np.full(n, np.nan, dtype=np.float64)
        q_limit_violation = np.full(n, np.nan, dtype=np.float64)
        slack_p = np.full(n, np.nan, dtype=np.float64)
        slack_p_limit_violation = np.full(n, np.nan, dtype=np.float64)
        messages: list[str] = []

        for sample, control in enumerate(z_batch):
            p_spec, q_spec, setpoints, p_load, q_load = self._specifications(control)
            state = np.concatenate(
                (self._initial_angle[self.nonref], np.ones(self.pq.size))
            )
            message = "maximum iterations reached"
            for iteration in range(max_iter + 1):
                angle, magnitude = self._unpack_state(state, setpoints)
                p_calc, q_calc = self._injections(angle, magnitude)
                mismatch = np.concatenate(
                    (
                        p_spec[self.nonref] - p_calc[self.nonref],
                        q_spec[self.pq] - q_calc[self.pq],
                    )
                )
                norm = float(np.linalg.norm(mismatch, ord=np.inf))
                if not np.isfinite(norm):
                    message = "non-finite power mismatch"
                    break
                if norm <= tol:
                    success[sample] = True
                    iterations[sample] = iteration
                    norms[sample] = norm
                    message = "converged"
                    break
                if iteration == max_iter:
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break
                jacobian = self._jacobian(angle, magnitude, p_calc, q_calc)
                jacobian_condition[sample] = np.linalg.cond(jacobian)
                try:
                    step = np.linalg.solve(jacobian, mismatch)
                except np.linalg.LinAlgError:
                    message = "singular power-flow Jacobian"
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break

                accepted = False
                damping = 1.0
                for _ in range(16):
                    candidate = state + damping * step
                    pq_voltage = candidate[self.nonref.size :]
                    if np.any((pq_voltage <= 0.5) | (pq_voltage >= 1.5)):
                        damping *= 0.5
                        continue
                    candidate_norm = float(
                        np.linalg.norm(
                            self._mismatch_one(candidate, control), ord=np.inf
                        )
                    )
                    if np.isfinite(candidate_norm) and candidate_norm < norm:
                        state = candidate
                        accepted = True
                        break
                    damping *= 0.5
                if not accepted:
                    message = "power-flow Newton line search failed"
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break

            solutions[sample] = state
            final_angle, final_magnitude = self._unpack_state(state, setpoints)
            final_p, final_q = self._injections(final_angle, final_magnitude)
            controlled_buses = np.r_[self.ref, self.pv]
            q_generation = final_q[controlled_buses] + q_load[controlled_buses]
            q_violation = np.maximum(
                np.maximum(self.q_min - q_generation, q_generation - self.q_max), 0.0
            )
            q_limit_violation[sample] = float(np.max(q_violation))
            slack_p[sample] = final_p[self.ref[0]] + p_load[self.ref[0]]
            slack_p_limit_violation[sample] = float(
                max(
                    self.p_slack_min - slack_p[sample],
                    slack_p[sample] - self.p_slack_max,
                    0.0,
                )
            )
            pv_voltage_error[sample] = float(
                np.max(np.abs(final_magnitude[controlled_buses] - setpoints))
            )
            if success[sample] and q_limit_violation[sample] > 1e-8:
                success[sample] = False
                message = "generator reactive-power limit violated"
            messages.append(message)

        diagnostics = _diagnostics(
            success,
            iterations,
            norms,
            messages,
            jacobian_condition=jacobian_condition,
            pv_voltage_error=pv_voltage_error,
            q_limit_violation=q_limit_violation,
            slack_p=slack_p,
            slack_p_limit_violation=slack_p_limit_violation,
        )
        _raise_failed(self.name, diagnostics, raise_on_failure)
        return _finish(solutions, single), diagnostics


class KuramotoSystem(BaseSystem):
    r"""Fixed-graph phase equilibrium with a slack phase and balanced injections.

    Inputs specify normalized injections at every non-slack node.  The slack
    injection is set to the negative sum, so total injection is exactly zero.
    Outputs are all non-slack phases; the slack phase is fixed to zero.
    """

    name = "kuramoto"

    def __init__(
        self,
        n_nodes: int = 8,
        *,
        injection_scale: float = 0.45,
        adjacency: ArrayLike | None = None,
        phase_limit: float = 0.5 * np.pi,
    ) -> None:
        if n_nodes < 3:
            raise ValueError("n_nodes must be at least three")
        if not np.isfinite(injection_scale) or injection_scale <= 0.0:
            raise ValueError("injection_scale must be finite and positive")
        if not np.isfinite(phase_limit) or not 0.0 < phase_limit <= np.pi:
            raise ValueError("phase_limit must lie in (0, pi]")
        self.n_nodes = int(n_nodes)
        self.input_dim = self.n_nodes - 1
        self.output_dim = self.n_nodes - 1
        self.slack = 0
        self.non_slack = np.arange(1, self.n_nodes, dtype=np.int64)
        self.injection_scale = float(injection_scale)
        self.phase_limit = float(phase_limit)

        if adjacency is None:
            weights = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
            for node in range(self.n_nodes):
                neighbor = (node + 1) % self.n_nodes
                weights[node, neighbor] = weights[neighbor, node] = 2.0
            # A second-neighbor chord makes the benchmark non-tree-like while
            # retaining a deterministic topology for every n_nodes >= 5.
            if self.n_nodes >= 5:
                for node in range(self.n_nodes):
                    neighbor = (node + 2) % self.n_nodes
                    weights[node, neighbor] = weights[neighbor, node] = 0.75
        else:
            weights = np.asarray(adjacency, dtype=np.float64)
            if weights.shape != (self.n_nodes, self.n_nodes):
                raise ValueError(
                    f"adjacency must have shape ({self.n_nodes}, {self.n_nodes})"
                )
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError("adjacency must be finite and non-negative")
            if not np.allclose(weights, weights.T, atol=1e-12):
                raise ValueError("adjacency must be symmetric")
            if not np.allclose(np.diag(weights), 0.0, atol=1e-12):
                raise ValueError("adjacency diagonal must be zero")
            weights = weights.copy()
        self.adjacency = weights
        self.laplacian = np.diag(weights.sum(axis=1)) - weights
        reduced = self.laplacian[np.ix_(self.non_slack, self.non_slack)]
        if np.linalg.matrix_rank(reduced) != self.output_dim:
            raise ValueError("adjacency graph must be connected")
        self._reduced_laplacian = reduced

    def physical_injections(self, z: ArrayLike) -> FloatArray:
        """Return balanced physical injections, including the slack node."""

        z_batch, single = _as_batch(z, self.input_dim, "z")
        injections = np.empty((z_batch.shape[0], self.n_nodes), dtype=np.float64)
        injections[:, 1:] = self.injection_scale * z_batch
        injections[:, 0] = -np.sum(injections[:, 1:], axis=1)
        return _finish(injections, single)

    def _flows(self, angle: FloatArray) -> FloatArray:
        differences = angle[:, None] - angle[None, :]
        return np.sum(self.adjacency * np.sin(differences), axis=1)

    def _full_residual_one(self, y: FloatArray, z: FloatArray) -> FloatArray:
        angle = np.zeros(self.n_nodes, dtype=np.float64)
        angle[self.non_slack] = y
        injection = np.empty(self.n_nodes, dtype=np.float64)
        injection[self.non_slack] = self.injection_scale * z
        injection[self.slack] = -np.sum(injection[self.non_slack])
        return self._flows(angle) - injection

    def full_residual(self, y: ArrayLike, z: ArrayLike) -> FloatArray:
        """Return all node balances, including the dependent slack equation."""

        y_batch, z_batch, single = _paired_batches(
            y, self.output_dim, z, self.input_dim
        )
        values = np.stack(
            [self._full_residual_one(state, control) for state, control in zip(y_batch, z_batch)]
        )
        return _finish(values, single)

    def residual(self, y: ArrayLike, z: ArrayLike) -> FloatArray:
        values = self.full_residual(y, z)
        if values.ndim == 1:
            return values[self.non_slack]
        return values[:, self.non_slack]

    def _weighted_laplacian(self, angle: FloatArray) -> FloatArray:
        differences = angle[:, None] - angle[None, :]
        weights = self.adjacency * np.cos(differences)
        return np.diag(weights.sum(axis=1)) - weights

    def _max_edge_difference(self, angle: FloatArray) -> float:
        edges = np.triu(self.adjacency > 0.0, k=1)
        differences = np.abs(angle[:, None] - angle[None, :])
        return float(np.max(differences[edges]))

    def solve(
        self,
        z: ArrayLike,
        *,
        tol: float = 1e-10,
        max_iter: int = 40,
        raise_on_failure: bool = True,
    ) -> tuple[FloatArray, Diagnostics]:
        if tol <= 0.0 or not np.isfinite(tol):
            raise ValueError("tol must be finite and positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        z_batch, single = _as_batch(z, self.input_dim, "z")
        n = z_batch.shape[0]
        solutions = np.empty((n, self.output_dim), dtype=np.float64)
        success = np.zeros(n, dtype=np.bool_)
        iterations = np.zeros(n, dtype=np.int64)
        norms = np.full(n, np.inf, dtype=np.float64)
        full_norms = np.full(n, np.inf, dtype=np.float64)
        balance_error = np.full(n, np.nan, dtype=np.float64)
        max_edge_angle = np.full(n, np.nan, dtype=np.float64)
        jacobian_condition = np.full(n, np.nan, dtype=np.float64)
        messages: list[str] = []

        for sample, control in enumerate(z_batch):
            injection = np.empty(self.n_nodes, dtype=np.float64)
            injection[self.non_slack] = self.injection_scale * control
            injection[self.slack] = -np.sum(injection[self.non_slack])
            # DC/linearized equilibrium is deterministic and already close in
            # the certified small-angle operating regime.
            state = np.linalg.solve(
                self._reduced_laplacian, injection[self.non_slack]
            )
            message = "maximum iterations reached"
            for iteration in range(max_iter + 1):
                angle = np.zeros(self.n_nodes, dtype=np.float64)
                angle[self.non_slack] = state
                mismatch = injection[self.non_slack] - self._flows(angle)[self.non_slack]
                norm = float(np.linalg.norm(mismatch, ord=np.inf))
                if not np.isfinite(norm):
                    message = "non-finite phase-balance residual"
                    break
                if norm <= tol:
                    success[sample] = True
                    iterations[sample] = iteration
                    norms[sample] = norm
                    message = "converged"
                    break
                if iteration == max_iter:
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break
                jacobian = self._weighted_laplacian(angle)[
                    np.ix_(self.non_slack, self.non_slack)
                ]
                jacobian_condition[sample] = np.linalg.cond(jacobian)
                try:
                    step = np.linalg.solve(jacobian, mismatch)
                except np.linalg.LinAlgError:
                    message = "singular Kuramoto Jacobian"
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break

                accepted = False
                damping = 1.0
                for _ in range(16):
                    candidate = state + damping * step
                    candidate_angle = np.zeros(self.n_nodes, dtype=np.float64)
                    candidate_angle[self.non_slack] = candidate
                    candidate_mismatch = (
                        injection[self.non_slack]
                        - self._flows(candidate_angle)[self.non_slack]
                    )
                    candidate_norm = float(
                        np.linalg.norm(candidate_mismatch, ord=np.inf)
                    )
                    if np.isfinite(candidate_norm) and candidate_norm < norm:
                        state = candidate
                        accepted = True
                        break
                    damping *= 0.5
                if not accepted:
                    message = "Kuramoto Newton line search failed"
                    iterations[sample] = iteration
                    norms[sample] = norm
                    break

            solutions[sample] = state
            final_angle = np.zeros(self.n_nodes, dtype=np.float64)
            final_angle[self.non_slack] = state
            full_residual = self._flows(final_angle) - injection
            full_norms[sample] = float(np.linalg.norm(full_residual, ord=np.inf))
            balance_error[sample] = float(abs(np.sum(injection)))
            max_edge_angle[sample] = self._max_edge_difference(final_angle)
            if success[sample] and max_edge_angle[sample] >= self.phase_limit:
                success[sample] = False
                message = (
                    "equilibrium lies outside the requested stable phase-difference regime"
                )
            messages.append(message)

        diagnostics = _diagnostics(
            success,
            iterations,
            norms,
            messages,
            full_residual_norm=full_norms,
            injection_balance_error=balance_error,
            max_edge_angle=max_edge_angle,
            jacobian_condition=jacobian_condition,
        )
        _raise_failed(self.name, diagnostics, raise_on_failure)
        return _finish(solutions, single), diagnostics


SYSTEM_REGISTRY: dict[str, type[BaseSystem]] = {
    "controlled": ControlledImplicitSystem,
    "controlled_implicit": ControlledImplicitSystem,
    "powerflow": ACPowerFlowSystem,
    "ac_power": ACPowerFlowSystem,
    "ac_power_flow": ACPowerFlowSystem,
    "ieee118": IEEE118PowerFlowSystem,
    "ac_power_118": IEEE118PowerFlowSystem,
    "allen_cahn": lambda **kwargs: NonlinearGridPhysicsSystem("allen_cahn", **kwargs),
    "shallow_water": lambda **kwargs: NonlinearGridPhysicsSystem("shallow_water", **kwargs),
    "case9": ACPowerFlowSystem,
    "kuramoto": KuramotoSystem,
    "duffing": CoupledDuffingSystem,
    "coupled_duffing": CoupledDuffingSystem,
}


def build_system(name: str, **kwargs: Any) -> BaseSystem:
    """Construct a registered benchmark system by a concise configuration name."""

    try:
        system_type = SYSTEM_REGISTRY[name.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(SYSTEM_REGISTRY))
        raise ValueError(f"unknown system {name!r}; choose one of {choices}") from exc
    return system_type(**kwargs)


__all__ = [
    "ACPowerFlowSystem",
    "IEEE118PowerFlowSystem",
    "NonlinearGridPhysicsSystem",
    "BaseSystem",
    "ControlledImplicitSystem",
    "CoupledDuffingSystem",
    "Diagnostics",
    "KuramotoSystem",
    "SYSTEM_REGISTRY",
    "SystemSolveError",
    "build_system",
]
