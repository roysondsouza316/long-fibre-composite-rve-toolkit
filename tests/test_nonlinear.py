"""Nonlinear RVE solver: interface insertion, J2 plasticity, cohesive laws and full solves.

The meshes here are structured and built in code (no gmsh needed); the tests skip when the
optional ``[nonlinear]`` dependencies (torch, diffcohesive) are missing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffcohesive")

from rve2d.config import ConfigError, config_from_dict  # noqa: E402
from rve2d.engines.tensormesh.constraints import build_dof_map  # noqa: E402
from rve2d.engines.tensormesh.mesh import RVEMesh, insert_cohesive_interfaces  # noqa: E402
from rve2d.engines.tensormesh.nonlinear.assembly import RVESystem  # noqa: E402
from rve2d.engines.tensormesh.nonlinear.laws import build_traction_law  # noqa: E402
from rve2d.engines.tensormesh.nonlinear.material import (  # noqa: E402
    MAX_DAMAGE,
    ElementMaterial,
    PlasticState,
    damage_rate,
    element_material,
    initial_state,
    j2_return_mapping,
    return_mapping,
)
from rve2d.engines.tensormesh.nonlinear.solver import (  # noqa: E402
    LoadPath,
    NewtonSettings,
    initial_solver_state,
    solve_load_path,
)

F64 = torch.float64
INF = float("inf")


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
    except Exception:  # ImportError, or OSError when system libraries (libGLU) are missing
        return False
    return True


# -- meshes -------------------------------------------------------------------------------
def structured_2d(
    n: int, fibre: tuple[float, float] | None = (0.3, 0.7)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, n + 1)
    points = np.array([[x, y] for y in xs for x in xs])
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + n + 2, a + n + 1
            cells += [[a, b, c], [a, c, d]]
    cells_arr = np.array(cells, dtype=np.int64)
    centre = points[cells_arr].mean(axis=1)
    phase = np.ones(len(cells_arr), dtype=np.int64)
    if fibre is not None:
        lo, hi = fibre
        inside = np.all((centre > lo) & (centre < hi), axis=1)
        phase[inside] = 2
    return points, cells_arr, phase


def structured_3d(
    n: int, fibre: tuple[float, float] | None = (0.3, 0.7)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, n + 1)
    points = np.array([[x, y, z] for z in xs for y in xs for x in xs])

    def node(i: int, j: int, k: int) -> int:
        return (k * (n + 1) + j) * (n + 1) + i

    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                v = [node(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in range(8)]
                for first, second in ((1, 2), (2, 1), (1, 4), (4, 1), (2, 4), (4, 2)):
                    cells.append([v[0], v[first], v[first | second], v[7]])
    cells_arr = np.array(cells, dtype=np.int64)
    centre = points[cells_arr].mean(axis=1)
    phase = np.ones(len(cells_arr), dtype=np.int64)
    if fibre is not None:
        lo, hi = fibre
        inside = np.all((centre[:, :2] > lo) & (centre[:, :2] < hi), axis=1)  # prism along z
        phase[inside] = 2
    return points, cells_arr, phase


def rve(
    dim: int, n: int, fibre: tuple[float, float] | None = (0.3, 0.7), cohesive: bool = True
) -> RVEMesh:
    points, cells, phase = (structured_2d if dim == 2 else structured_3d)(n, fibre)
    if cohesive and fibre is not None:
        return insert_cohesive_interfaces(dim, points, cells, phase)
    return RVEMesh(
        dim=dim,
        points=points,
        cells=cells,
        phase=phase,
        cohesive=np.zeros((0, 2 * dim), dtype=np.int64),
        node_side=np.zeros(len(points), dtype=np.int64),
        lower=points.min(axis=0),
        upper=points.max(axis=0),
        matrix_phase_id=1,
        fibre_phase_id=2,
    )


def system_for(
    mesh: RVEMesh,
    matrix: tuple[float, float, float, float] = (3500.0, 0.35, INF, 0.0),
    fibre: tuple[float, float, float, float] = (70000.0, 0.2, INF, 0.0),
    interface: dict[str, float] | None = None,
    bc: str = "periodic",
) -> RVESystem:
    is_fibre = mesh.phase == 2
    params = [
        torch.tensor(np.where(is_fibre, f, m), dtype=F64)
        for m, f in zip(matrix, fibre, strict=True)
    ]
    law = None
    if mesh.n_cohesive:
        kw = interface or {"K": 1e7, "T": 30.0, "G": 0.05}
        law = build_traction_law(
            "bilinear_mixed_mode", kw["K"], kw["T"], 1.5 * kw["T"], kw["G"], 2 * kw["G"]
        )
    active = [0, 1, 2, 5] if mesh.dim == 2 else list(range(6))
    return RVESystem(
        mesh, build_dof_map(mesh, bc), element_material(*params), law, active, torch.device("cpu")
    )


def uniaxial(dim: int, kinematics: str, strain: float, component: int = 0) -> LoadPath:
    active = {
        "plane_strain": [0, 1, 5],
        "generalized_plane_strain": [0, 1, 2, 5],
        "solid": list(range(6)),
    }[kinematics]
    inactive = [c for c in range(6) if c not in active]
    return LoadPath({component: strain}, inactive, {c: 0.0 for c in active if c != component})


def solve(system: RVESystem, load: LoadPath, steps: int = 4, scale: float = 3500.0):  # noqa: ANN201
    state = initial_solver_state(system, int(getattr(system.law, "state_dim", 1)))
    return solve_load_path(system, load, NewtonSettings(steps=steps), state, scale)


# -- interface insertion ------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_interface_insertion_separates_phases_and_orients_normals(dim: int) -> None:
    mesh = rve(dim, 6 if dim == 3 else 10)
    fibre_nodes = set(np.unique(mesh.cells[mesh.phase == 2]).tolist())
    matrix_nodes = set(np.unique(mesh.cells[mesh.phase == 1]).tolist())
    assert fibre_nodes.isdisjoint(matrix_nodes)
    bottom, top = mesh.cohesive[:, :dim], mesh.cohesive[:, dim:]
    np.testing.assert_allclose(mesh.points[bottom], mesh.points[top])
    assert set(np.unique(bottom).tolist()) <= matrix_nodes
    assert set(np.unique(top).tolist()) <= fibre_nodes
    # the square/prism fibre spans [0.3, 0.7]: every facet normal points towards its centre
    centre = np.full(dim, 0.5)
    from rve2d.engines.tensormesh.mesh import _facet_normals

    normals = _facet_normals(mesh.points, bottom)
    to_centre = centre - mesh.points[bottom].mean(axis=1)
    if dim == 3:
        to_centre[:, 2] = 0.0
    assert np.all(np.einsum("ij,ij->i", normals, to_centre) > 0.0)


# -- J2 material --------------------------------------------------------------------------
def test_j2_uniaxial_stress_matches_bilinear_curve() -> None:
    youngs, nu, sy, hardening = 3500.0, 0.35, 60.0, 300.0
    material = element_material(
        *(torch.tensor([v], dtype=F64) for v in (youngs, nu, sy, hardening))
    )
    state = initial_state(1, F64, torch.device("cpu"))
    strain = torch.zeros(1, 6, dtype=F64)
    for exx in np.linspace(0.0, 0.05, 26)[1:]:
        strain[0, 0] = exx
        for _ in range(30):  # enforce sigma_yy = sigma_zz = 0 (uniaxial stress)
            stress, tangent, _, _ = j2_return_mapping(strain, state, material)
            if float(stress[0, 1:].abs().max()) < 1e-10:
                break
            strain[0, 1:] -= torch.linalg.solve(tangent[0, 1:, 1:], stress[0, 1:])
        stress, _, state, _ = j2_return_mapping(strain, state, material)
        expected = (
            youngs * exx
            if exx <= sy / youngs
            else sy + youngs * hardening / (youngs + hardening) * (exx - sy / youngs)
        )
        assert float(stress[0, 0]) == pytest.approx(expected, rel=1e-10)


def test_j2_consistent_tangent_matches_finite_differences() -> None:
    torch.manual_seed(0)
    n = 50
    material = element_material(
        *(torch.full((n,), v, dtype=F64) for v in (3500.0, 0.35, 60.0, 300.0))
    )
    strain = 0.03 * torch.randn(n, 6, dtype=F64)
    state = PlasticState(0.005 * torch.randn(n, 6, dtype=F64), 0.01 * torch.rand(n, dtype=F64))
    _, tangent, _, yielding = j2_return_mapping(strain, state, material)
    assert int(yielding.sum()) > n // 2
    h = 1e-7
    for j in range(6):
        plus, minus = strain.clone(), strain.clone()
        plus[:, j] += h
        minus[:, j] -= h
        fd = (
            j2_return_mapping(plus, state, material)[0]
            - j2_return_mapping(minus, state, material)[0]
        ) / (2 * h)
        torch.testing.assert_close(tangent[:, :, j], fd, rtol=1e-6, atol=1e-4)


# -- pressure-dependent plasticity and ductile damage --------------------------------------
def epoxy(
    n: int = 1,
    compressive: float = 90.0,
    plastic_poisson: float = 0.3,
    onset: float | None = None,
    rate: float = 0.0,
    hardening: float = 300.0,
) -> ElementMaterial:
    def full(value: float) -> torch.Tensor:
        return torch.full((n,), value, dtype=F64)

    return element_material(
        full(3500.0), full(0.35), full(60.0), full(hardening),
        compressive_yield_stress=full(compressive), plastic_poisson_ratio=full(plastic_poisson),
        damage_onset=None if onset is None else full(onset), damage_rate=full(rate),
    )  # fmt: skip


def uniaxial_stress_path(
    material: ElementMaterial, strains: np.ndarray
) -> tuple[list[float], list[float], PlasticState]:
    """Axial stress and equivalent plastic strain along a uniaxial-stress path (the five other
    stresses kept at zero)."""
    state = initial_state(1, F64, torch.device("cpu"))
    strain = torch.zeros(1, 6, dtype=F64)
    stresses, eqps = [], []
    for exx in strains:
        strain[0, 0] = exx
        for _ in range(50):
            stress, tangent, _, _ = return_mapping(strain, state, material)
            if float(stress[0, 1:].abs().max()) < 1e-10:
                break
            strain[0, 1:] -= torch.linalg.solve(tangent[0, 1:, 1:], stress[0, 1:])
        stress, _, state, _ = return_mapping(strain, state, material)
        stresses.append(float(stress[0, 0]))
        eqps.append(float(state.equivalent_plastic_strain[0]))
    return stresses, eqps, state


def test_paraboloid_with_equal_yield_stresses_is_j2() -> None:
    torch.manual_seed(2)
    n = 200
    strain = 0.03 * torch.randn(n, 6, dtype=F64)
    state = PlasticState(0.005 * torch.randn(n, 6, dtype=F64), 0.01 * torch.rand(n, dtype=F64))
    j2 = j2_return_mapping(strain, state, epoxy(n, compressive=60.0, plastic_poisson=0.5))
    # damage switched on but never reached forces the general (paraboloidal) return mapping
    general = return_mapping(
        strain, state, epoxy(n, compressive=60.0, plastic_poisson=0.5, onset=1e9, rate=1.0)
    )
    assert int(general[3].sum()) > n // 2
    torch.testing.assert_close(general[0], j2[0], rtol=0, atol=1e-11)
    torch.testing.assert_close(general[1], j2[1], rtol=0, atol=1e-9)
    torch.testing.assert_close(general[2].plastic_strain, j2[2].plastic_strain, rtol=0, atol=1e-15)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_paraboloid_yields_at_the_tensile_and_compressive_yield_stress(sign: float) -> None:
    youngs, st, sc, hardening = 3500.0, 60.0, 90.0, 300.0
    strains = sign * np.linspace(0.0, 0.06, 61)[1:]
    stresses, _, state = uniaxial_stress_path(epoxy(compressive=sc), strains)
    yield_stress = st if sign > 0 else sc
    slope = hardening * (1.0 if sign > 0 else sc / st)  # compression hardens in proportion
    for strain, stress in zip(strains, stresses, strict=True):
        elastic = youngs * abs(strain)
        plastic = yield_stress + youngs * slope / (youngs + slope) * (
            abs(strain) - yield_stress / youngs
        )
        assert abs(stress) == pytest.approx(min(elastic, plastic), rel=1e-9)  # bilinear curve
    plastic_strain = state.plastic_strain[0]
    assert float(-plastic_strain[1] / plastic_strain[0]) == pytest.approx(0.3)  # plastic Poisson
    assert float(state.equivalent_plastic_strain[0]) == pytest.approx(abs(float(plastic_strain[0])))


def test_paraboloid_and_damage_tangent_matches_finite_differences() -> None:
    torch.manual_seed(0)
    n = 200
    material = epoxy(n, onset=0.005, rate=20.0)
    strain = 0.03 * torch.randn(n, 6, dtype=F64)
    state = PlasticState(0.005 * torch.randn(n, 6, dtype=F64), 0.01 * torch.rand(n, dtype=F64))
    _, tangent, new_state, yielding = return_mapping(strain, state, material)
    damaging = (new_state.equivalent_plastic_strain - 0.005) * 20.0
    assert (
        int(yielding.sum()) > n // 2 and int(((damaging > 0) & (damaging < MAX_DAMAGE)).sum()) > 50
    )
    assert float((tangent - tangent.transpose(1, 2)).abs().max()) > 1.0  # not symmetric
    h = 1e-7
    for j in range(6):
        plus, minus = strain.clone(), strain.clone()
        plus[:, j] += h
        minus[:, j] -= h
        fd = return_mapping(plus, state, material)[0] - return_mapping(minus, state, material)[0]
        torch.testing.assert_close(tangent[:, :, j], fd / (2 * h), rtol=1e-6, atol=1e-4)


def test_damage_dissipates_the_fracture_energy_over_the_crack_band() -> None:
    length, energy, onset, st = 0.002, 0.09, 0.01, 60.0  # mm, N/mm, -, MPa
    values = (length, st, 0.0, onset, energy)
    rate = float(damage_rate(*(torch.tensor([v], dtype=F64) for v in values))[0])
    material = epoxy(compressive=st, plastic_poisson=0.5, onset=onset, rate=rate, hardening=0.0)
    strains = np.linspace(0.0, st / 3500.0 + onset + 1.2 / rate, 4001)[1:]
    stresses, eqps, _ = uniaxial_stress_path(material, strains)
    stress, plastic = np.array(stresses), np.array(eqps)
    damage = np.clip(rate * (plastic - onset), 0.0, MAX_DAMAGE)
    np.testing.assert_allclose(stress[plastic > 0], (1.0 - damage[plastic > 0]) * st, rtol=1e-9)
    softening = (plastic >= onset) & (damage < MAX_DAMAGE)
    work = np.trapezoid(stress[softening], plastic[softening])  # per unit volume
    expected = energy / length * (1.0 - (1.0 - MAX_DAMAGE) ** 2)  # G_f over the crack band
    assert work == pytest.approx(expected, rel=2e-3)


# -- traction-separation law --------------------------------------------------------------
@pytest.mark.parametrize(
    ("stiffness", "strength", "toughness"), [(1e8, 50.0, 0.002), (1e17, 50e6, 2.0)]
)
def test_scaled_law_is_unit_independent(
    stiffness: float, strength: float, toughness: float
) -> None:
    law = build_traction_law(
        "bilinear_mixed_mode", stiffness, strength, 1.5 * strength, toughness, 3 * toughness
    )
    _, _, damage = law(torch.zeros(1, 2, dtype=F64), torch.zeros(1, dtype=F64))
    assert float(damage) == 0.0
    final = 2 * toughness / strength
    opening = torch.linspace(0.0, 1.1 * final, 3001, dtype=F64)
    history = torch.zeros(1, dtype=F64)
    traction = []
    for value in opening:
        t, history, _ = law(torch.stack([value, torch.zeros((), dtype=F64)])[None], history)
        traction.append(float(t[0, 0]))
    assert float(torch.trapezoid(torch.tensor(traction, dtype=F64), opening)) == pytest.approx(
        toughness, rel=1e-3
    )


# -- full RVE solves ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dim", "kinematics", "expected"),
    [
        (2, "plane_strain", 3500.0 / (1 - 0.35**2)),
        (2, "generalized_plane_strain", 3500.0),
        (3, "solid", 3500.0),
    ],
)
def test_homogeneous_patch_test_with_cohesive_interfaces(
    dim: int, kinematics: str, expected: float
) -> None:
    mesh = rve(dim, 6 if dim == 3 else 10)
    same = (3500.0, 0.35, INF, 0.0)
    strong = {"K": 1e12, "T": 1e9, "G": 1e9}
    out = solve(
        system_for(mesh, matrix=same, fibre=same, interface=strong),
        uniaxial(dim, kinematics, 0.01),
        steps=2,
    )
    assert out.completed
    last = out.records[-1]
    assert last.macro_stress[0] / last.macro_strain[0] == pytest.approx(expected, rel=1e-6)
    free = (
        [i for i in range(6) if i not in (0, 2)]
        if kinematics == "plane_strain"
        else list(range(1, 6))
    )
    assert max(abs(last.macro_stress[i]) for i in free) < 1e-8 * expected


@pytest.mark.parametrize(("dim", "kinematics"), [(2, "generalized_plane_strain"), (3, "solid")])
def test_homogeneous_plastic_rve_follows_uniaxial_curve(dim: int, kinematics: str) -> None:
    youngs, sy, hardening = 3500.0, 60.0, 300.0
    same = (youngs, 0.35, sy, hardening)
    mesh = rve(dim, 4 if dim == 3 else 6, fibre=None, cohesive=False)
    out = solve(system_for(mesh, matrix=same, fibre=same), uniaxial(dim, kinematics, 0.04), steps=8)
    assert out.completed
    for record in out.records[1:]:
        exx = record.macro_strain[0]
        expected = (
            youngs * exx
            if exx <= sy / youngs
            else sy + youngs * hardening / (youngs + hardening) * (exx - sy / youngs)
        )
        assert record.macro_stress[0] == pytest.approx(expected, rel=1e-8)


def test_global_tangent_matches_finite_differences() -> None:
    import scipy.sparse

    mesh = rve(2, 12)
    system = system_for(
        mesh, matrix=(3500.0, 0.35, 40.0, 300.0), interface={"K": 1e6, "T": 20.0, "G": 0.05}
    )
    load = uniaxial(2, "generalized_plane_strain", 0.012)
    out = solve(system, load, steps=10)
    state, free = out.state, load.free
    assert out.records[-1].max_damage > 0.1
    macro = state.macro_strain.clone()
    macro[0] += 5e-4  # beyond the converged state: plastic flow and damage growth active
    target = torch.zeros(6, dtype=F64)
    ev = system.evaluate(state.w, macro, state.plastic, state.cohesive_history, free, target)
    assert int(ev.yielding.sum()) > 0
    m = ev.matrix
    assert m is not None
    matrix = scipy.sparse.coo_matrix(
        (m.value.numpy(), (m.row.numpy(), m.col.numpy())), shape=(m.size, m.size)
    )
    torch.manual_seed(3)
    direction = torch.randn(m.size, dtype=F64)
    direction[: system.n_reduced] *= float(state.w.abs().max())
    direction[system.n_reduced :] *= 1e-3
    h = 1e-7

    def residual(sign: float) -> torch.Tensor:
        w = state.w + sign * h * direction[: system.n_reduced]
        e = macro.clone()
        e[free] += sign * h * direction[system.n_reduced :]
        return system.evaluate(
            w, e, state.plastic, state.cohesive_history, free, target, with_tangent=False
        ).residual

    fd = (residual(1.0) - residual(-1.0)) / (2 * h)
    jd = torch.as_tensor(matrix @ direction.numpy())
    assert float(torch.linalg.norm(jd - fd) / torch.linalg.norm(fd)) < 1e-6


def test_interface_debonding_softens_the_response() -> None:
    mesh = rve(2, 16)
    bonded = system_for(mesh, interface={"K": 1e7, "T": 1e6, "G": 1e6})
    weak = system_for(mesh, interface={"K": 1e7, "T": 15.0, "G": 0.02})
    load = uniaxial(2, "generalized_plane_strain", 0.01)
    stiff = solve(bonded, load, steps=10).records[-1]
    debonded = solve(weak, load, steps=20)
    assert debonded.completed
    assert debonded.records[-1].max_damage > 0.99
    assert debonded.records[-1].macro_stress[0] < 0.9 * stiff.macro_stress[0]


# -- configuration ------------------------------------------------------------------------
def _nonlinear_config(**overrides: object) -> dict[str, object]:
    section: dict[str, object] = {
        "enabled": True,
        "matrix": {"youngs_modulus": 3500.0, "poisson_ratio": 0.35, "yield_stress": 60.0},
        "fibre": {"youngs_modulus": 70000.0, "poisson_ratio": 0.2},
        "interface": {"penalty_stiffness": 1e8, "normal_strength": 50.0, "mode_i_toughness": 0.002},
    }
    section.update(overrides)
    return {
        "mode": "synthetic",
        "synthetic": {
            "domain_width": 0.05,
            "domain_height": 0.05,
            "fibre_radius": 0.0035,
            "target_volume_fraction": 0.3,
            "periodic_compatible": True,
        },
        "nonlinear": section,
    }


EPOXY = {"youngs_modulus": 3500.0, "poisson_ratio": 0.35, "yield_stress": 60.0}


def test_nonlinear_config_is_parsed_with_nested_sections() -> None:
    config = config_from_dict(_nonlinear_config())
    assert config.nonlinear.enabled
    assert config.nonlinear.matrix is not None and config.nonlinear.matrix.yield_stress == 60.0
    assert config.nonlinear.resolved_kinematics(2) == "generalized_plane_strain"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"matrix": {"youngs_modulus": 3500.0, "poisson_ratio": 0.35, "yeild_stress": 60.0}},
            "Unknown key",
        ),
        (
            {
                "interface": {
                    "penalty_stiffness": 1e8,
                    "normal_strength": 50.0,
                    "mode_i_toughness": 1e-6,
                }
            },
            "must exceed",
        ),
        ({"load": {"component": "yz"}}, "not active"),
        ({"fibre": None}, "fibre material"),
        (
            {
                "interface": {
                    "penalty_stiffness": 1e8,
                    "normal_strength": 50.0,
                    "shear_strength": 0.0,
                    "mode_i_toughness": 0.002,
                }
            },
            "shear / mode II",
        ),
        (
            {
                "interface": {
                    "law": "exponential",
                    "penalty_stiffness": 1e8,
                    "normal_strength": 50.0,
                    "mode_i_toughness": 0.002,
                    "viscosity": 1e-4,
                }
            },
            "bilinear_mixed_mode only",
        ),
        (
            {
                "interface": {
                    "law": "exponential",
                    "penalty_stiffness": 1e8,
                    "normal_strength": 50.0,
                    "mode_i_toughness": 1.5 * 50.0**2 / 1e8,  # enough for bilinear only
                }
            },
            "for the exponential law",
        ),
        ({"matrix": {**EPOXY, "compressive_yield_stress": 40.0}}, "at least yield_stress"),
        ({"matrix": {**EPOXY, "compressive_yield_stress": 90.0}}, "plastic_poisson_ratio < 0.5"),
        ({"matrix": {**EPOXY, "plastic_poisson_ratio": 0.6}}, r"lie in \[0, 0.5\]"),
        ({"matrix": {**EPOXY, "damage_onset_strain": 0.01}}, "both damage_onset_strain"),
        ({"matrix": {**EPOXY, "fracture_energy": 0.1, "damage_onset_strain": -1.0}}, "negative"),
        (
            {"matrix": {"youngs_modulus": 3500.0, "poisson_ratio": 0.35, "fracture_energy": 0.1,
                        "damage_onset_strain": 0.01}},
            "need a yield_stress",
        ),  # fmt: skip
    ],
)
def test_nonlinear_config_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_dict(_nonlinear_config(**overrides))


@pytest.mark.skipif(not _gmsh_available(), reason="gmsh python module not available")
def test_cli_build_and_solve_nonlinear_end_to_end(tmp_path: Path) -> None:
    import json

    import yaml

    from rve2d.cli import main

    payload = _nonlinear_config(load={"max_strain": 0.004, "steps": 4})
    payload["synthetic"] = {
        "domain_width": 0.02,
        "domain_height": 0.02,
        "fibre_radius": 0.0035,
        "target_volume_fraction": 0.1,
        "edge_clearance": 0.0005,
        "periodic_compatible": True,
    }
    payload["mesh"] = {
        "element_size_min": 0.0006,
        "element_size_max": 0.002,
        "elements_per_circle": 24,
        "verbosity": 0,
    }
    payload["export"] = {"formats": ["msh"]}
    config_path = tmp_path / "nl.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert main(["build-and-solve", str(config_path), "--output-dir", str(tmp_path / "out")]) == 0
    summary = json.loads((tmp_path / "out" / "nonlinear_summary.json").read_text())
    assert summary["completed"] and summary["mesh"]["cohesive_elements"] > 0
    assert math.isfinite(summary["peak_stress"]) and summary["peak_stress"] > 0.0
    assert (tmp_path / "out" / "nonlinear_final.vtu").exists()
    assert (tmp_path / "out" / "nonlinear_final_interface.vtu").exists()


@pytest.mark.parametrize(
    ("dim", "kinematics"), [(2, "plane_strain"), (2, "generalized_plane_strain"), (3, "solid")]
)
def test_elastic_nonlinear_solve_reproduces_the_linear_homogenization(
    tmp_path: Path, dim: int, kinematics: str
) -> None:
    """Perfectly bonded elastic phases under uniaxial strain: the macro stress is a column of
    the effective stiffness computed by the linear homogenization on the same mesh."""
    from _meshes import cube_msh, square_msh

    from rve2d.engines.common.materials import ACTIVE_VOIGT
    from rve2d.workflow import solve_homogenization, solve_nonlinear

    mesh = square_msh(tmp_path / "rve.msh") if dim == 2 else cube_msh(tmp_path / "rve.msh")
    synthetic: dict[str, object] = {
        "domain_width": 1.0,
        "domain_height": 1.0,
        "fibre_radius": 0.1,
        "target_volume_fraction": 0.1,
        "periodic_compatible": True,
    }
    if dim == 3:
        synthetic["domain_depth"] = 1.0
    base = {"mode": "synthetic", "dimension": dim, "synthetic": synthetic}
    linear = solve_homogenization(
        config_from_dict(
            base
            | {
                "solver": {
                    "enabled": True,
                    "kinematics": kinematics,
                    "boundary_condition": "periodic",
                    "matrix_youngs_modulus": 3500.0,
                    "matrix_poisson_ratio": 0.35,
                    "fibre_youngs_modulus": 70000.0,
                    "fibre_poisson_ratio": 0.2,
                }
            }
        ),
        mesh,
        tmp_path / "linear",
        engine="tensormesh",
    )
    active = ACTIVE_VOIGT[kinematics]
    stiffness = np.zeros((6, 6))
    stiffness[np.ix_(active, active)] = linear.homogenized_stiffness
    for component in ("xx", "xy"):
        section = {
            "enabled": True,
            "engine": "tensormesh",
            "kinematics": kinematics,
            "boundary_condition": "periodic",
            "matrix": {"youngs_modulus": 3500.0, "poisson_ratio": 0.35},
            "fibre": {"youngs_modulus": 70000.0, "poisson_ratio": 0.2},
            "load": {"type": "uniaxial_strain", "component": component, "max_strain": 1e-3,
                     "steps": 1},
        }  # fmt: skip
        result = solve_nonlinear(
            config_from_dict(base | {"nonlinear": section}), mesh, tmp_path / component
        )
        assert result.completed
        strain = np.asarray(result.records[-1].macro_strain)
        stress = np.asarray(result.records[-1].macro_stress)
        expected = stiffness @ strain
        # (plane strain: the nonlinear solve also reports s_zz, which the 3x3 does not cover)
        tolerance = 1e-10 * np.abs(expected).max()
        np.testing.assert_allclose(
            stress[list(active)], expected[list(active)], rtol=0, atol=tolerance
        )
