"""Linear homogenization in both engines: physics checks and engine parity.

The TensorMesh engine needs nothing beyond the base install. Julia-engine tests run when
``julia`` is available and the bundled environment has been set up
(``rve2d doctor --setup-julia``); they check that both engines give the same results.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import meshio
import numpy as np
import pytest
from _meshes import cube_msh, square_msh

from rve2d.config import ConfigError, RVEConfig, config_from_dict
from rve2d.engines.common.materials import PhaseElasticity, constitutive_matrix
from rve2d.engines.julia import runner
from rve2d.mesh_io import extrude_mesh
from rve2d.workflow import solve_homogenization

MATRIX = {"youngs_modulus": 3.5e9, "poisson_ratio": 0.35}
FIBRE = {"youngs_modulus": 70.0e9, "poisson_ratio": 0.2}
CASES_2D = ("plane_stress", "plane_strain", "generalized_plane_strain")


def config(
    dimension: int,
    kinematics: str,
    bc: str = "periodic",
    same_material: bool = False,
    **extra: object,
) -> RVEConfig:
    fibre = MATRIX if same_material else FIBRE
    synthetic: dict[str, object] = {
        "domain_width": 1.0,
        "domain_height": 1.0,
        "fibre_radius": 0.1,
        "target_volume_fraction": 0.1,
        "periodic_compatible": True,
    }
    if dimension == 3:
        synthetic["domain_depth"] = 1.0
    solver = {
        "enabled": True,
        "kinematics": kinematics,
        "boundary_condition": bc,
        "matrix_youngs_modulus": MATRIX["youngs_modulus"],
        "matrix_poisson_ratio": MATRIX["poisson_ratio"],
        "fibre_youngs_modulus": fibre["youngs_modulus"],
        "fibre_poisson_ratio": fibre["poisson_ratio"],
        **extra,
    }
    return config_from_dict(
        {"mode": "synthetic", "dimension": dimension, "synthetic": synthetic, "solver": solver}
    )


def stiffness(cfg: RVEConfig, mesh: Path, out: Path, engine: str = "tensormesh") -> np.ndarray:
    result = solve_homogenization(cfg, mesh, out, engine=engine)
    assert result.engine == engine
    return np.asarray(result.homogenized_stiffness)


def julia_ready() -> bool:
    return shutil.which("julia") is not None and runner.is_instantiated()


@pytest.mark.parametrize(("dimension", "kinematics"), [(2, k) for k in CASES_2D] + [(3, "solid")])
@pytest.mark.parametrize("bc", ["periodic", "dirichlet"])
def test_single_material_recovers_the_material_stiffness(
    tmp_path: Path, dimension: int, kinematics: str, bc: str
) -> None:
    mesh = square_msh(tmp_path / "rve.msh") if dimension == 2 else cube_msh(tmp_path / "rve.msh")
    expected = constitutive_matrix(PhaseElasticity("isotropic", **MATRIX), kinematics)
    effective = stiffness(config(dimension, kinematics, bc, same_material=True), mesh, tmp_path)
    np.testing.assert_allclose(effective, expected, rtol=0, atol=1e-9 * np.abs(expected).max())


@pytest.mark.parametrize("engine", ["tensormesh", "julia"])
@pytest.mark.parametrize("layers", [1, 2])
def test_generalized_plane_strain_matches_a_3d_extrusion(
    tmp_path: Path, layers: int, engine: str
) -> None:
    # A z-invariant microstructure: the periodic 3D solution on the extruded mesh is
    # z-invariant, so the 2D generalized-plane-strain stiffness must match it exactly. With one
    # layer (as in the laminate pipeline) every node lies on the top or bottom face.
    if engine == "julia" and not julia_ready():
        pytest.skip("Julia engine environment not set up")
    flat = square_msh(tmp_path / "a.msh")
    c2 = stiffness(config(2, "generalized_plane_strain"), flat, tmp_path / "2d", engine)
    extruded = extrude_mesh(flat, tmp_path / "b.msh", depth=0.5 / layers, layers=layers)
    c3 = stiffness(config(3, "solid"), extruded, tmp_path / "3d", engine)
    np.testing.assert_allclose(c2, c3, rtol=0, atol=1e-9 * np.abs(c3).max())
    assert c2[2, 2] > 1.5 * c2[0, 0]  # the axial (fibre-direction) stiffness dominates


def test_plane_strain_is_the_in_plane_block_of_generalized_plane_strain(tmp_path: Path) -> None:
    mesh = square_msh(tmp_path / "rve.msh")
    gps = stiffness(config(2, "generalized_plane_strain"), mesh, tmp_path / "gps")
    plane = stiffness(config(2, "plane_strain"), mesh, tmp_path / "ps")
    in_plane = gps[np.ix_([0, 1, 5], [0, 1, 5])]
    np.testing.assert_allclose(plane, in_plane, rtol=0, atol=1e-9 * np.abs(in_plane).max())


@pytest.mark.parametrize(
    ("dimension", "kinematics"), [(2, "generalized_plane_strain"), (3, "solid")]
)
def test_effective_stiffness_respects_the_bounds(
    tmp_path: Path, dimension: int, kinematics: str
) -> None:
    mesh = square_msh(tmp_path / "rve.msh") if dimension == 2 else cube_msh(tmp_path / "rve.msh")
    periodic = stiffness(config(dimension, kinematics), mesh, tmp_path / "p")
    affine = stiffness(config(dimension, kinematics, "dirichlet"), mesh, tmp_path / "d")
    summary = json.loads((tmp_path / "p" / "homogenization_summary.json").read_text())
    vf = summary["fibre_volume_fraction"]
    cm = constitutive_matrix(PhaseElasticity("isotropic", **MATRIX), kinematics)
    cf = constitutive_matrix(PhaseElasticity("isotropic", **FIBRE), kinematics)
    voigt = vf * cf + (1 - vf) * cm
    reuss = np.linalg.inv(vf * np.linalg.inv(cf) + (1 - vf) * np.linalg.inv(cm))
    tol = 1e-9 * np.abs(voigt).max()
    np.testing.assert_allclose(periodic, periodic.T, rtol=0, atol=tol)
    # Reuss <= periodic <= affine (Dirichlet) <= Voigt, in the sense of the strain energy
    for lower, upper in ((reuss, periodic), (periodic, affine), (affine, voigt)):
        assert np.linalg.eigvalsh(0.5 * (upper - lower + (upper - lower).T)).min() > -tol


@pytest.mark.parametrize(
    ("dimension", "kinematics"), [(2, "generalized_plane_strain"), (3, "solid")]
)
def test_vtu_displacements_are_periodic(tmp_path: Path, dimension: int, kinematics: str) -> None:
    mesh = square_msh(tmp_path / "rve.msh") if dimension == 2 else cube_msh(tmp_path / "rve.msh")
    cfg = config(dimension, kinematics)
    result = solve_homogenization(cfg, mesh, tmp_path, engine="tensormesh")
    assert result.vtk_path is not None
    assert_periodic_displacements(meshio.read(result.vtk_path), dimension)


def assert_periodic_displacements(vtu: meshio.Mesh, dimension: int) -> None:
    """``u(image) - u(mirror) = E (x_image - x_mirror)`` for every unit macro strain ``E``."""
    points = vtu.points
    keys = {tuple(np.round(p, 9)): i for i, p in enumerate(points)}
    names = sorted(k for k in vtu.point_data if k.startswith("u_case_"))
    assert names
    voigt = ["xx", "yy", "zz", "yz", "xz", "xy"]
    for name in names:
        strain = np.zeros(6)
        strain[voigt.index(name.rsplit("_", 1)[1])] = 1.0
        gradient = np.array(
            [
                [strain[0], strain[5] / 2, strain[4] / 2],
                [strain[5] / 2, strain[1], strain[3] / 2],
                [strain[4] / 2, strain[3] / 2, strain[2]],
            ]
        )
        u = vtu.point_data[name]
        for axis in range(dimension):
            for image in np.flatnonzero(np.isclose(points[:, axis], 1.0)):
                shift = np.zeros(3)
                shift[axis] = 1.0
                mirror = keys[tuple(np.round(points[image] - shift, 9))]
                jump = gradient @ shift
                if dimension == 2:
                    jump[2] = gradient[2, :2] @ shift[:2]
                np.testing.assert_allclose(u[image] - u[mirror], jump, atol=1e-9)


@pytest.mark.parametrize("kinematics", CASES_2D)
def test_result_files_do_not_depend_on_the_engine(tmp_path: Path, kinematics: str) -> None:
    result = solve_homogenization(
        config(2, kinematics), square_msh(tmp_path / "rve.msh"), tmp_path, engine="tensormesh"
    )
    summary = json.loads(result.summary_path.read_text())
    assert summary["engine"] == "tensormesh" and summary["kinematics"] == kinematics
    assert len(summary["homogenized_stiffness_voigt"]) == len(summary["voigt_components"])
    header = (tmp_path / "stress_strain_response.csv").read_text().splitlines()[0]
    expected = {
        "plane_stress": "case,exx,eyy,gxy,sxx,syy,txy",
        "plane_strain": "case,exx,eyy,gxy,sxx,syy,txy",
        "generalized_plane_strain": ("case,exx,eyy,ezz,gyz,gxz,gxy,sxx,syy,szz,tyz,txz,txy"),
    }[kinematics]
    assert header == expected
    if kinematics == "plane_strain":
        assert "ex_plane_strain" in result.engineering_constants
    else:
        assert "ex" in result.engineering_constants


def test_solver_config_validation() -> None:
    with pytest.raises(ConfigError, match="2D solves"):
        config(2, "solid")
    with pytest.raises(ConfigError, match="3D solves"):
        config(3, "plane_strain")
    with pytest.raises(ConfigError, match="periodic_compatible"):
        config_from_dict(
            {
                "mode": "synthetic",
                "synthetic": {
                    "domain_width": 1.0,
                    "domain_height": 1.0,
                    "fibre_radius": 0.1,
                    "target_volume_fraction": 0.1,
                },
                "image": {"image_path": "x.png", "periodic_compatible": True},  # wrong mode
                "solver": {"enabled": True, "boundary_condition": "periodic"},
            }
        )
    with pytest.raises(ConfigError, match="linear triangles"):
        config_from_dict(
            {
                "mode": "synthetic",
                "synthetic": {
                    "domain_width": 1.0,
                    "domain_height": 1.0,
                    "fibre_radius": 0.1,
                    "target_volume_fraction": 0.1,
                },
                "mesh": {"mesh_order": 2},
                "solver": {"enabled": True},
            }
        )


@pytest.mark.skipif(not julia_ready(), reason="Julia engine environment not set up")
@pytest.mark.parametrize(("dimension", "kinematics"), [(2, k) for k in CASES_2D] + [(3, "solid")])
@pytest.mark.parametrize("bc", ["periodic", "dirichlet"])
def test_julia_and_python_engines_agree(
    tmp_path: Path, dimension: int, kinematics: str, bc: str
) -> None:
    mesh = square_msh(tmp_path / "rve.msh") if dimension == 2 else cube_msh(tmp_path / "rve.msh")
    cfg = config(dimension, kinematics, bc)
    tensormesh = solve_homogenization(cfg, mesh, tmp_path / "tensormesh", engine="tensormesh")
    julia = solve_homogenization(cfg, mesh, tmp_path / "julia", engine="julia")
    cp, cj = np.asarray(tensormesh.homogenized_stiffness), np.asarray(julia.homogenized_stiffness)
    np.testing.assert_allclose(cj, cp, rtol=0, atol=1e-10 * np.abs(cp).max())
    summary_p = json.loads(tensormesh.summary_path.read_text())
    summary_j = json.loads(julia.summary_path.read_text())
    assert summary_j["fibre_volume_fraction"] == pytest.approx(summary_p["fibre_volume_fraction"])
    assert summary_j["anchor_node"] == summary_p["anchor_node"]
    for case_p, case_j in zip(
        summary_p["boundary_tractions"], summary_j["boundary_tractions"], strict=True
    ):
        for face, traction in case_p.items():
            np.testing.assert_allclose(case_j[face], traction, atol=1e-9 * np.abs(cp).max())
    assert tensormesh.vtk_path is not None and julia.vtk_path is not None
    vtu_p, vtu_j = meshio.read(tensormesh.vtk_path), meshio.read(julia.vtk_path)
    np.testing.assert_allclose(vtu_j.points, vtu_p.points)
    for name, values in vtu_p.point_data.items():
        np.testing.assert_allclose(vtu_j.point_data[name], values, atol=1e-9)
    if bc == "periodic":
        assert_periodic_displacements(vtu_j, dimension)
