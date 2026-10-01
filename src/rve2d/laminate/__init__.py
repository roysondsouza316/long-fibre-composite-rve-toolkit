"""Laminates from RVE results: ply properties, stacking sequences, laminate stiffness, ply
damage models and the laminate coupon test in tension and compression (stress, strain,
elongation, force).

Engine independent: the plies come from the RVE solves of either engine (``tensormesh`` or
``julia``); everything here is closed-form or a small nonlinear solve in NumPy.
"""

from rve2d.laminate.clt import (
    Laminate,
    abd_matrix,
    effective_3d_constants,
    effective_3d_stiffness,
    laminate_constants,
)
from rve2d.laminate.coupon import CouponResult, CouponSettings, coupon_test
from rve2d.laminate.damage import (
    MODELS,
    ContinuumDamageModel,
    CurveModel,
    FractureEnergies,
    HashinModel,
    MaxStressModel,
    PlyElasticity,
    PlyModel,
    PlyStrengths,
    build_ply_model,
)
from rve2d.laminate.ply import (
    PlyCurve,
    PlyProperties,
    curve_from_response,
    ply_from_homogenization,
    ply_stiffness_from_rve,
)
from rve2d.laminate.stacking import format_stacking_sequence, parse_stacking_sequence

__all__ = [
    "MODELS",
    "ContinuumDamageModel",
    "CouponResult",
    "CouponSettings",
    "CurveModel",
    "FractureEnergies",
    "HashinModel",
    "Laminate",
    "MaxStressModel",
    "PlyCurve",
    "PlyElasticity",
    "PlyModel",
    "PlyProperties",
    "PlyStrengths",
    "abd_matrix",
    "build_ply_model",
    "coupon_test",
    "curve_from_response",
    "effective_3d_constants",
    "effective_3d_stiffness",
    "format_stacking_sequence",
    "laminate_constants",
    "parse_stacking_sequence",
    "ply_from_homogenization",
    "ply_stiffness_from_rve",
]
