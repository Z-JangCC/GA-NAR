class ProtocolViolationError(RuntimeError):
    """Raised when an implementation would change a frozen scientific protocol."""


class NumericalValidityError(RuntimeError):
    """Raised for an implementation-level numerical invariant violation."""


class ArtifactMismatchError(RuntimeError):
    """Raised when an immutable formal artifact is requested with new inputs."""


class BranchContinuationError(RuntimeError):
    """Raised when a requested target cannot be certified on the target branch."""


class DegenerateStateCoordinateError(ValueError):
    """Raised when a frozen state coordinate has zero training-set variance."""
