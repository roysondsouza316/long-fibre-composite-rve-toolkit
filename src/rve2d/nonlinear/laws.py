"""Build diffcohesive traction-separation laws from the rve2d interface configuration.

The laws are evaluated in units of their own onset opening ``s = strength / K``: the wrapped
diffcohesive law is constructed with ``K' = K s`` and ``G' = G / s`` (tractions keep their
units) and receives separations ``delta / s``. The bilinear and shape laws are exactly
covariant under this scaling, so results are unchanged -- but the small absolute regularisation
constants inside the laws (e.g. ``sqrt(shear^2 + 1e-12)``) then act on O(1) numbers. Without
it, realistic stiffnesses in mm/MPa (K = 1e8 N/mm^3, onset opening 5e-7 mm) or SI units give a
non-zero damage at zero separation.
"""

from __future__ import annotations

from typing import Any

import torch

from rve2d.exceptions import SolverError

SHAPE_LAWS = ("bilinear", "linear-parabolic", "exponential", "trapezoidal")


class ScaledTractionLaw(torch.nn.Module):
    """Evaluate ``inner`` on separations divided by ``length_scale``.

    ``relaxation_time`` > 0 enables Duvaut-Lions viscous regularisation of the damage with a
    relaxation time measured in units of the load path (pseudo-time 0..1). diffcohesive applies
    its relaxation once per load step, so the per-step coefficient is set to
    ``relaxation_time / dt`` before every increment (``set_time_step``); the regularisation is
    then independent of how the increments are cut.
    """

    def __init__(
        self, inner: torch.nn.Module, length_scale: float, relaxation_time: float = 0.0
    ) -> None:
        super().__init__()
        self.inner = inner
        self.length_scale = float(length_scale)
        self.relaxation_time = float(relaxation_time)
        self.state_dim = int(getattr(inner, "state_dim", 1))

    def set_time_step(self, dt: float) -> None:
        if self.relaxation_time > 0.0:
            setattr(self.inner, "viscosity", self.relaxation_time / dt)  # noqa: B010

    def forward(self, delta: torch.Tensor, history: torch.Tensor) -> Any:
        return self.inner(delta / self.length_scale, history)


def build_traction_law(
    law: str,
    penalty_stiffness: float,
    normal_strength: float,
    shear_strength: float | None,
    mode_i_toughness: float,
    mode_ii_toughness: float | None,
    bk_exponent: float = 1.45,
    mixed_mode_criterion: str = "bk",
    viscosity: float = 0.0,
    shear_penalty_stiffness: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> ScaledTractionLaw:
    """Unit-safe diffcohesive law; shear strength and mode II toughness default to mode I."""
    try:
        from diffcohesive.laws import SHAPE_LAWS as DIFFCOHESIVE_SHAPES
        from diffcohesive.laws import BilinearMixedModeTSL
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SolverError(
            "Cohesive interfaces need the diffcohesive package: "
            "pip install 'rve2d-fibre[nonlinear]'."
        ) from exc

    if shear_strength is None:
        shear_strength = normal_strength
    if mode_ii_toughness is None:
        mode_ii_toughness = mode_i_toughness
    scale = normal_strength / penalty_stiffness
    inner: torch.nn.Module
    if law == "bilinear_mixed_mode":
        inner = BilinearMixedModeTSL(
            T_max_n=normal_strength,
            T_max_s=shear_strength,
            G_c1=mode_i_toughness / scale,
            G_c2=mode_ii_toughness / scale,
            eta=bk_exponent,
            K=penalty_stiffness * scale,
            K_s=None if shear_penalty_stiffness is None else shear_penalty_stiffness * scale,
            mixed_mode_criterion=mixed_mode_criterion,
            viscosity=viscosity,  # any positive value selects the [kappa, D_v] history layout
            dtype=dtype,
        )
    elif law in SHAPE_LAWS:
        if viscosity > 0.0:
            raise SolverError("Viscous regularisation is available for bilinear_mixed_mode only.")
        inner = DIFFCOHESIVE_SHAPES[law](
            K0=penalty_stiffness * scale, sigma0=normal_strength, Gc=mode_i_toughness / scale
        )
    else:
        raise SolverError(f"Unknown cohesive law {law!r}.")
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    return ScaledTractionLaw(inner, scale, relaxation_time=viscosity).to(dtype)
