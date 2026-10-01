"""Regenerate the DiffCohesive.jl parity table from the Python diffcohesive package.

Run from the repository root with the rve2d ``nonlinear`` extra installed:

    python tools/make_diffcohesive_reference.py \
        src/rve2d/engines/julia/DiffCohesive/test/reference_diffcohesive.csv

The laws are evaluated through rve2d's unit-safe wrapper (``build_traction_law``), the form the
Python solver uses; lengths in mm, stresses in MPa. Each row: law settings, separation
(zero-padded to 3), previous state (kappa, relaxed damage), traction, new state, damage and the
3x3 zero-padded tangent (row-major), with kappa in physical units.
"""

from __future__ import annotations

import csv
import sys
import warnings

import numpy as np
import torch

from rve2d.engines.tensormesh.nonlinear.laws import build_traction_law

K, TN, TS, GIC, GIIC = 1.0e8, 50.0, 75.0, 0.002, 0.006
ONSET = TN / K
LAWS = [  # (law, keyword arguments, shear stiffness or None, viscosity)
    ("bilinear_mixed_mode", {"bk_exponent": 1.45, "mixed_mode_criterion": "bk"}, None, 0.0),
    ("bilinear_mixed_mode", {"bk_exponent": 2.0, "mixed_mode_criterion": "power"}, None, 0.0),
    ("bilinear_mixed_mode", {"bk_exponent": 1.45, "mixed_mode_criterion": "bk"}, 0.5e8, 0.0),
    ("bilinear_mixed_mode", {"bk_exponent": 1.45, "mixed_mode_criterion": "bk"}, None, 0.7),
    ("bilinear", {}, None, 0.0),
    ("linear-parabolic", {}, None, 0.0),
    ("exponential", {}, None, 0.0),
    ("trapezoidal", {}, None, 0.0),
]
# (separation magnitude, previous history) in onset openings: elastic, onset, softening,
# near failure, compression
REGIMES = [(0.3, 0.0), (1.5, 0.5), (6.0, 2.0), (40.0, 10.0), (3.0, 20.0)]


def pad(values: list[float], size: int) -> list[float]:
    return list(values) + [0.0] * (size - len(values))


def main(path: str) -> None:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(1234)
    header = (
        ["law", "criterion", "shear_stiffness", "dim", "viscosity", "dt"]
        + [f"delta{i}" for i in range(3)]
        + ["kappa_prev", "dv_prev"]
        + [f"t{i}" for i in range(3)]
        + ["kappa_new", "dv_new", "damage"]
        + [f"J{i}{j}" for i in range(3) for j in range(3)]
    )
    rows = []
    for name, kwargs, shear_stiffness, viscosity in LAWS:
        law = build_traction_law(
            name,
            K,
            TN,
            TS,
            GIC,
            GIIC,
            viscosity=viscosity,
            shear_penalty_stiffness=shear_stiffness,
            **kwargs,
        )
        scale = law.length_scale
        dt = 0.5 if viscosity > 0 else 1.0
        law.set_time_step(dt)
        viscous = law.state_dim == 2
        for dim in (2, 3):
            for k in range(10):
                magnitude, history = REGIMES[k % 5]
                delta = rng.normal(size=dim) * magnitude * ONSET
                if k % 5 == 4:
                    delta[0] = -abs(delta[0])
                kappa = history * ONSET * rng.uniform(0.5, 1.5)
                relaxed = rng.uniform(0.0, 0.3) if viscous else 0.0
                state = torch.tensor(
                    [kappa / scale, relaxed] if viscous else [kappa / scale], dtype=torch.float64
                )
                h = state if viscous else state[0]
                d = torch.tensor(delta, dtype=torch.float64)
                traction, new_state, damage = law(d[None], h[None])
                jacobian = torch.func.jacfwd(lambda x, h=h, law=law: law(x[None], h[None])[0][0])(d)
                new = new_state[0].reshape(-1).tolist()
                tangent = np.zeros((3, 3))
                tangent[:dim, :dim] = jacobian.numpy()
                rows.append(
                    [name, kwargs.get("mixed_mode_criterion", ""), shear_stiffness or 0.0, dim]
                    + [viscosity, dt]
                    + pad(delta.tolist(), 3)
                    + [kappa, relaxed]
                    + pad(traction[0].tolist(), 3)
                    + [new[0] * scale, new[1] if viscous else 0.0, float(damage)]
                    + tangent.reshape(-1).tolist()
                )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([v if isinstance(v, str | int) else repr(float(v)) for v in row])
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main(sys.argv[1])
