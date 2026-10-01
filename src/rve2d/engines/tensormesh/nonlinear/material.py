"""Small-strain plasticity and ductile damage of the RVE phases, vectorised in torch.

Voigt order is ``(xx, yy, zz, yz, xz, xy)``. Strains carry engineering shear components
(``gamma = 2 * eps``); stresses carry tensor components. The same 6-component formulation
serves 3D solids and 2D plane strain / generalized plane strain (where ``yz = xz = 0``).

Two return mappings, both with their consistent (algorithmic) tangents:

* **J2 (von Mises)** with linear isotropic hardening (Simo & Hughes, *Computational
  Inelasticity*, 1998, Box 3.1/3.2): yield function ``||s|| - sqrt(2/3) (sigma_y0 + H alpha)``
  with ``alpha`` the equivalent (uniaxial) plastic strain. In uniaxial stress it gives the
  bilinear curve with elastic modulus E and post-yield tangent ``E H / (E + H)``.
* **Pressure-dependent (paraboloidal)** plasticity for polymer matrices (Tschoegl 1971; Melro
  et al., Int. J. Solids Struct. 50, 2013): yield function
  ``f = q^2 + 3 p (sigma_c - sigma_t) - sigma_c sigma_t`` with von Mises stress ``q``, mean
  stress ``p`` and the current tensile and compressive yield stresses (linear hardening, the
  compressive one scaled with the tensile one), and the non-associative flow potential
  ``g = q^2 + alpha_g p^2``, ``alpha_g = 9/2 (1 - 2 nu_p) / (1 + nu_p)`` for the plastic
  Poisson ratio ``nu_p``. With equal yield stresses and ``nu_p = 1/2`` it is J2. The
  equivalent plastic strain rate ``sqrt(eps_p : eps_p / (1 + 2 nu_p^2))`` equals the axial
  plastic strain rate in uniaxial tension.

Ductile damage (optional): once the equivalent plastic strain exceeds ``damage_onset`` the
stress is ``(1 - d)`` times the effective (plastic) stress, with ``d`` growing linearly with
the plastic displacement ``l_e (eps_p - onset)`` of a crack band one element wide; ``d``
reaches 1 when the dissipated energy per unit crack area is about the fracture energy (it is
capped at ``MAX_DAMAGE``). ``d`` is a function of the equivalent plastic strain, so it needs
no history variable of its own. Phases without a yield stress are linear elastic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rve2d.exceptions import SolverError

_SQRT_2_3 = math.sqrt(2.0 / 3.0)


MAX_DAMAGE = 0.99  # residual stiffness of fully damaged elements: 1 %


@dataclass(frozen=True)
class ElementMaterial:
    """Per-element material parameters as tensors of shape ``(n_elements,)``."""

    shear_modulus: torch.Tensor
    bulk_modulus: torch.Tensor
    yield_stress: torch.Tensor  # +inf for elastic phases
    hardening_modulus: torch.Tensor
    compressive_yield_stress: torch.Tensor  # = yield_stress for a pressure-insensitive phase
    plastic_poisson_ratio: torch.Tensor  # 0.5: isochoric plastic flow
    damage_onset: torch.Tensor  # equivalent plastic strain at damage onset (+inf: no damage)
    damage_rate: torch.Tensor  # d(damage) / d(equivalent plastic strain) after onset

    @property
    def general(self) -> torch.Tensor:
        """Elements that need the paraboloidal return mapping (otherwise J2 or elastic)."""
        plastic = torch.isfinite(self.yield_stress)
        return plastic & (
            (self.compressive_yield_stress != self.yield_stress)
            | (self.plastic_poisson_ratio != 0.5)
            | torch.isfinite(self.damage_onset)
        )


@dataclass(frozen=True)
class PlasticState:
    plastic_strain: torch.Tensor  # (n, 6), engineering shear
    equivalent_plastic_strain: torch.Tensor  # (n,)


def element_material(
    youngs_modulus: torch.Tensor,
    poisson_ratio: torch.Tensor,
    yield_stress: torch.Tensor,
    hardening_modulus: torch.Tensor,
    compressive_yield_stress: torch.Tensor | None = None,
    plastic_poisson_ratio: torch.Tensor | None = None,
    damage_onset: torch.Tensor | None = None,
    damage_rate: torch.Tensor | None = None,
) -> ElementMaterial:
    shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    bulk = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    return ElementMaterial(
        shear,
        bulk,
        yield_stress,
        hardening_modulus,
        yield_stress if compressive_yield_stress is None else compressive_yield_stress,
        torch.full_like(yield_stress, 0.5)
        if plastic_poisson_ratio is None
        else plastic_poisson_ratio,
        torch.full_like(yield_stress, math.inf) if damage_onset is None else damage_onset,
        torch.zeros_like(yield_stress) if damage_rate is None else damage_rate,
    )


def damage_rate(
    characteristic_length: torch.Tensor,
    yield_stress: torch.Tensor,
    hardening_modulus: torch.Tensor,
    damage_onset: torch.Tensor,
    fracture_energy: torch.Tensor,
) -> torch.Tensor:
    """Damage per unit equivalent plastic strain for linear softening in the plastic
    displacement over a crack band of width ``characteristic_length``: ``d`` reaches 1 at the
    plastic displacement ``2 G_f / sigma_onset`` (the tensile yield stress at onset)."""
    onset_stress = yield_stress + hardening_modulus * damage_onset
    return characteristic_length * onset_stress / (2.0 * fracture_energy)


def bulk_damage(equivalent_plastic_strain: torch.Tensor, material: ElementMaterial) -> torch.Tensor:
    """Damage of every element for its equivalent plastic strain."""
    enabled = torch.isfinite(material.damage_onset)
    onset = torch.where(enabled, material.damage_onset, torch.zeros_like(material.damage_onset))
    raw = torch.where(
        enabled,
        material.damage_rate * (equivalent_plastic_strain - onset),
        torch.zeros_like(onset),
    )
    return raw.clamp(0.0, MAX_DAMAGE)


def initial_state(n: int, dtype: torch.dtype, device: torch.device) -> PlasticState:
    return PlasticState(
        plastic_strain=torch.zeros(n, 6, dtype=dtype, device=device),
        equivalent_plastic_strain=torch.zeros(n, dtype=dtype, device=device),
    )


def j2_return_mapping(
    strain: torch.Tensor,
    state: PlasticState,
    material: ElementMaterial,
) -> tuple[torch.Tensor, torch.Tensor, PlasticState, torch.Tensor]:
    """Return ``(stress, consistent_tangent, new_state, yielding)`` for strains ``(n, 6)``."""
    mu = material.shear_modulus
    kappa = material.bulk_modulus
    hardening = material.hardening_modulus

    elastic_strain = strain - state.plastic_strain
    volumetric = elastic_strain[:, :3].sum(dim=1)
    dev_normal = elastic_strain[:, :3] - volumetric[:, None] / 3.0
    dev_shear = 0.5 * elastic_strain[:, 3:]  # tensor shear components
    s_trial = 2.0 * mu[:, None] * torch.cat([dev_normal, dev_shear], dim=1)
    s_norm = torch.sqrt((s_trial[:, :3] ** 2).sum(dim=1) + 2.0 * (s_trial[:, 3:] ** 2).sum(dim=1))

    yield_radius = _SQRT_2_3 * (material.yield_stress + hardening * state.equivalent_plastic_strain)
    f_trial = s_norm - yield_radius
    yielding = f_trial > 0.0

    safe_norm = torch.where(s_norm > 0.0, s_norm, torch.ones_like(s_norm))
    direction = s_trial / safe_norm[:, None]
    dgamma = torch.where(
        yielding, f_trial / (2.0 * mu * (1.0 + hardening / (3.0 * mu))), torch.zeros_like(f_trial)
    )

    deviatoric = s_trial - (2.0 * mu * dgamma)[:, None] * direction
    pressure = kappa * volumetric
    stress = deviatoric.clone()
    stress[:, :3] = stress[:, :3] + pressure[:, None]

    engineering_direction = torch.cat([direction[:, :3], 2.0 * direction[:, 3:]], dim=1)
    new_state = PlasticState(
        plastic_strain=state.plastic_strain + dgamma[:, None] * engineering_direction,
        equivalent_plastic_strain=state.equivalent_plastic_strain + _SQRT_2_3 * dgamma,
    )

    theta = torch.where(yielding, 1.0 - 2.0 * mu * dgamma / safe_norm, torch.ones_like(s_norm))
    theta_bar = torch.where(
        yielding, 1.0 / (1.0 + hardening / (3.0 * mu)) - (1.0 - theta), torch.zeros_like(s_norm)
    )
    tangent = _isotropic_tangent(kappa, mu * theta) - (2.0 * mu * theta_bar)[:, None, None] * (
        direction[:, :, None] * direction[:, None, :]
    )
    return stress, tangent, new_state, yielding


def return_mapping(
    strain: torch.Tensor,
    state: PlasticState,
    material: ElementMaterial,
) -> tuple[torch.Tensor, torch.Tensor, PlasticState, torch.Tensor]:
    """``(stress, consistent_tangent, new_state, yielding)``: J2 (or elastic) for most
    elements, the paraboloidal return mapping with damage where the material needs it."""
    stress, tangent, new_state, yielding = j2_return_mapping(strain, state, material)
    general = material.general
    if not bool(general.any()):
        return stress, tangent, new_state, yielding
    idx = torch.nonzero(general).squeeze(1)
    sub = ElementMaterial(*(getattr(material, name)[idx] for name in _MATERIAL_FIELDS))
    sub_state = PlasticState(state.plastic_strain[idx], state.equivalent_plastic_strain[idx])
    effective, effective_tangent, plastic_strain, eqps, sub_yielding, deqps = (
        paraboloid_return_mapping(strain[idx], sub_state, sub)
    )
    damage = bulk_damage(eqps, sub)
    raw = sub.damage_rate * (
        eqps - torch.where(torch.isfinite(sub.damage_onset), sub.damage_onset, eqps)
    )
    growing = sub_yielding & (raw > 0.0) & (raw < MAX_DAMAGE)
    rate = torch.where(growing, sub.damage_rate, torch.zeros_like(raw))
    sub_stress = (1.0 - damage)[:, None] * effective
    sub_tangent = (1.0 - damage)[:, None, None] * effective_tangent - rate[:, None, None] * (
        effective[:, :, None] * deqps[:, None, :]
    )
    stress = stress.index_copy(0, idx, sub_stress)
    tangent = tangent.index_copy(0, idx, sub_tangent)
    yielding = yielding.index_copy(0, idx, sub_yielding)
    new_state = PlasticState(
        new_state.plastic_strain.index_copy(0, idx, plastic_strain),
        new_state.equivalent_plastic_strain.index_copy(0, idx, eqps),
    )
    return stress, tangent, new_state, yielding


_MATERIAL_FIELDS = (
    "shear_modulus",
    "bulk_modulus",
    "yield_stress",
    "hardening_modulus",
    "compressive_yield_stress",
    "plastic_poisson_ratio",
    "damage_onset",
    "damage_rate",
)
_ROOT_TOLERANCE = 1e-14  # |f| relative to sigma_c * sigma_t
_ROOT_ITERATIONS = 100


def paraboloid_return_mapping(
    strain: torch.Tensor,
    state: PlasticState,
    material: ElementMaterial,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paraboloidal return mapping (no damage): ``(effective_stress, tangent, plastic_strain,
    equivalent_plastic_strain, yielding, d_eqps_d_strain)``; the last is the derivative of the
    equivalent plastic strain with respect to the (engineering) strain, for the damage tangent.
    """
    mu, kappa = material.shear_modulus, material.bulk_modulus
    st0, sc0 = material.yield_stress, material.compressive_yield_stress
    h_t = material.hardening_modulus
    h_c = h_t * sc0 / st0
    nu_p = material.plastic_poisson_ratio
    alpha = 4.5 * (1.0 - 2.0 * nu_p) / (1.0 + nu_p)
    k_eq = 1.0 / (1.0 + 2.0 * nu_p**2)

    elastic = strain - state.plastic_strain
    volumetric = elastic[:, :3].sum(dim=1)
    deviatoric = torch.cat([elastic[:, :3] - volumetric[:, None] / 3.0, 0.5 * elastic[:, 3:]], 1)
    s_trial = 2.0 * mu[:, None] * deviatoric
    p_trial = kappa * volumetric
    q_trial = torch.sqrt(
        1.5 * (s_trial[:, :3] ** 2).sum(dim=1) + 3.0 * (s_trial[:, 3:] ** 2).sum(dim=1)
    )
    eqps_n = state.equivalent_plastic_strain
    st_n = st0 + h_t * eqps_n
    sc_n = sc0 + h_c * eqps_n
    f_trial = q_trial**2 + 3.0 * (sc_n - st_n) * p_trial - sc_n * st_n
    yielding = f_trial > 0.0

    def residual(dlam: torch.Tensor) -> dict[str, torch.Tensor]:
        a = 1.0 + 6.0 * mu * dlam
        b = 1.0 + 2.0 * kappa * alpha * dlam
        q, p = q_trial / a, p_trial / b
        m = torch.sqrt(k_eq * (6.0 * q**2 + (4.0 / 3.0) * alpha**2 * p**2))
        safe_m = torch.where(m > 0.0, m, torch.ones_like(m))
        deqps = dlam * m
        st, sc = st_n + h_t * deqps, sc_n + h_c * deqps
        f = q**2 + 3.0 * (sc - st) * p - sc * st
        dq, dp = -6.0 * mu * q / a, -2.0 * kappa * alpha * p / b
        dm = torch.where(
            m > 0.0, k_eq * (6.0 * q * dq + (4.0 / 3.0) * alpha**2 * p * dp) / safe_m, 0.0 * m
        )
        ddeqps = m + dlam * dm
        dst, dsc = h_t * ddeqps, h_c * ddeqps
        df = 2.0 * q * dq + 3.0 * (dsc - dst) * p + 3.0 * (sc - st) * dp - dsc * st - sc * dst
        return {"a": a, "b": b, "q": q, "p": p, "m": m, "safe_m": safe_m, "deqps": deqps,
                "st": st, "sc": sc, "f": f, "df": df, "ddeqps": ddeqps}  # fmt: skip

    # Safeguarded Newton on the plastic multiplier: f(0) > 0 and f -> -sc st < 0 as dlam grows.
    dlam = torch.zeros_like(q_trial)
    low = torch.zeros_like(q_trial)
    high = torch.full_like(q_trial, math.inf)
    first = torch.full_like(q_trial, math.inf)
    stalled = torch.zeros_like(yielding)
    for _ in range(_ROOT_ITERATIONS):
        r = residual(dlam)
        done = ~yielding | stalled | (r["f"].abs() <= _ROOT_TOLERANCE * r["sc"] * r["st"])
        if bool(done.all()):
            break
        low = torch.where(r["f"] > 0.0, dlam, low)
        high = torch.where(r["f"] < 0.0, dlam, high)
        newton = dlam - r["f"] / torch.where(r["df"] != 0.0, r["df"], -torch.ones_like(dlam))
        first = torch.where(torch.isinf(first) & (r["df"] < 0.0), newton, first)
        valid = (r["df"] < 0.0) & (newton > low) & (newton < high)
        grow = torch.where(
            dlam > 0.0, 2.0 * dlam, torch.where(torch.isfinite(first), first, 1.0 / (6.0 * mu))
        )
        fallback = torch.where(torch.isinf(high), grow, 0.5 * (low + high))
        updated = torch.where(done, dlam, torch.where(valid, newton, fallback))
        stalled = (updated - dlam).abs() <= 4.0 * torch.finfo(dlam.dtype).eps * updated.abs()
        dlam = updated
    else:
        raise SolverError("The paraboloidal return mapping did not converge.")
    r = residual(dlam)
    a, b, q, p = r["a"], r["b"], r["q"], r["p"]
    s = s_trial / a[:, None]
    identity = torch.zeros_like(s)
    identity[:, :3] = 1.0
    effective = s + p[:, None] * identity
    plastic_increment = dlam[:, None] * torch.cat(
        [3.0 * s[:, :3] + (2.0 * alpha * p / 3.0)[:, None], 6.0 * s[:, 3:]], dim=1
    )
    eqps = eqps_n + r["deqps"]

    # Consistent tangent: d(effective) = C_iso de - v d(dlam), d(dlam) = -F_e de / f'(dlam).
    phi = 3.0 * (h_c - h_t) * p - h_c * r["st"] - r["sc"] * h_t
    deqps_dq = dlam * k_eq * 6.0 * q / r["safe_m"] / a
    deqps_dp = dlam * k_eq * (4.0 / 3.0) * alpha**2 * p / r["safe_m"] / b
    f_q = 2.0 * q / a + phi * deqps_dq
    f_p = 3.0 * (r["sc"] - r["st"]) / b + phi * deqps_dp
    safe_q = torch.where(q > 0.0, q, torch.ones_like(q))
    direction = torch.where(q > 0.0, 3.0 * mu / safe_q, 0.0 * q)[:, None] * s  # d q_trial / de
    f_e = f_q[:, None] * direction + (f_p * kappa)[:, None] * identity
    v = (6.0 * mu / a)[:, None] * s + (2.0 * kappa * alpha * p / b)[:, None] * identity
    df = torch.where(yielding, r["df"], -torch.ones_like(dlam))
    plastic_tangent = (
        _isotropic_tangent(kappa / b, mu / a) + v[:, :, None] * f_e[:, None, :] / df[:, None, None]
    )
    deqps_de = (
        r["ddeqps"][:, None] * (-f_e / df[:, None])
        + deqps_dq[:, None] * direction
        + (deqps_dp * kappa)[:, None] * identity
    )
    elastic_stress = s_trial + p_trial[:, None] * identity
    elastic_tangent = _isotropic_tangent(kappa, mu)
    y = yielding[:, None]
    return (
        torch.where(y, effective, elastic_stress),
        torch.where(yielding[:, None, None], plastic_tangent, elastic_tangent),
        state.plastic_strain + torch.where(y, plastic_increment, torch.zeros_like(s)),
        torch.where(yielding, eqps, eqps_n),
        yielding,
        torch.where(y, deqps_de, torch.zeros_like(s)),
    )


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    mean = stress[:, :3].mean(dim=1, keepdim=True)
    dev = stress[:, :3] - mean
    return torch.sqrt(1.5 * ((dev**2).sum(dim=1) + 2.0 * (stress[:, 3:] ** 2).sum(dim=1)))


def _isotropic_tangent(bulk: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
    """``K 1(x)1 + 2 G (I_sym - 1/3 1(x)1)`` in engineering-shear Voigt form."""
    n = bulk.shape[0]
    tangent = torch.zeros(n, 6, 6, dtype=bulk.dtype, device=bulk.device)
    lam = bulk - 2.0 * shear / 3.0
    tangent[:, :3, :3] = lam[:, None, None]
    diag = torch.arange(3, device=bulk.device)
    tangent[:, diag, diag] = tangent[:, diag, diag] + 2.0 * shear[:, None]
    shear_idx = torch.arange(3, 6, device=bulk.device)
    tangent[:, shear_idx, shear_idx] = shear[:, None]
    return tangent
