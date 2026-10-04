# Core

Shared dataclasses, protocol status values, deterministic random streams, hashes, and immutable artifact storage. Physics arrays use `float64`; a formal artifact records protocol, configuration, environment, input hashes, seed, and formal flag. `UNRESOLVED` is a scientific status; implementation invariant failures raise exceptions.

## Component contract

1. **Purpose:** provide shared types, status values, randomness, hashing, and artifact boundaries.
2. **Mathematical definition:** an artifact is a payload plus immutable metadata and a SHA-256 digest.
3. **Terminology:** physical coordinates are `q_phys/z_phys`; standardized coordinates are `x_std/y_std`.
4. **Inputs:** JSON-like metadata, NumPy arrays, seeds, and filesystem paths.
5. **Outputs:** dataclasses, deterministic streams, hashes, and persisted artifacts.
6. **Shapes:** arrays retain their caller-provided shape; metadata is scalar or JSON-compatible.
7. **Physical units:** no units are imposed by core; units belong to the owning system.
8. **Coordinate system:** core never silently converts coordinates.
9. **Numerical precision:** hashes cover exact bytes; physics payloads are normally `float64`.
10. **Public API:** `ArtifactStore`, `RandomManager`, protocol enums, datatypes, and hash helpers.
11. **Dependencies:** Python standard library and NumPy where array hashing is required.
12. **Invariants:** formal payloads have protocol, configuration, environment, and artifact hashes.
13. **Numerical verification:** hash round trips and deterministic random-stream tests live in `tests/`.
14. **Failure conditions:** mismatched metadata or payload hashes raise `ArtifactMismatchError`.
15. **No-fallback rule:** an existing formal artifact is never replaced implicitly.
16. **Replacement interface:** replacements must preserve the public dataclasses and artifact metadata contract.
