class RVEError(RuntimeError):
    """Base package exception."""


class ConfigError(RVEError):
    """Raised when configuration data is invalid."""


class MeshingError(RVEError):
    """Raised when meshing cannot be completed."""


class SolverError(RVEError):
    """Raised when the downstream FEM solve cannot be completed."""
