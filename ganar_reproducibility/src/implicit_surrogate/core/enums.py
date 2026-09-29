from enum import Enum


class ScientificStatus(str, Enum):
    """Protocol-level status, intentionally distinct from software exceptions."""

    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"

