"""The laminate pipeline: RVE -> ply properties -> laminates (``rve2d laminate``).

1. Build the RVE (geometry, mesh) from the config.
2. Ply stiffness: linear homogenization of the RVE (either engine).
3. Ply curves (for the coupon tests): nonlinear RVE solves under uniaxial stress, transverse
   tension and compression (RVE xx) and in-plane shear (RVE xz; for a 2D RVE on one periodic
   layer of tetrahedra extruded from its mesh). Only the curves the damage models need are
   solved; their peaks give the strengths Yt, Yc and S12 unless the config sets them.
4. For every stacking sequence: CLT and 3D effective stiffness and engineering constants,
   then every coupon test (tension, compression, shear) with every ply damage model.

Output layout (under the output directory)::

    rve/                            RVE mesh and geometry metadata
    ply/elastic/                    linear homogenization of the RVE
    ply/transverse_tension/         nonlinear RVE solve, xx > 0
    ply/transverse_compression/     nonlinear RVE solve, xx < 0
    ply/shear/                      nonlinear RVE solve, xz (2D: on rve_layer.msh)
    ply/ply_properties.json         stiffness, curves, strengths (reusable: --ply)
    laminates/summary.csv           stiffness of every stacking sequence
    laminates/coupon_tests.csv      one row per laminate, damage model and test
    laminates/coupon_tests_<model>.png
    laminates/<sequence>/           abd.csv, constants.json, coupon_tests.png and
                                    <model>/<test>.csv, <model>/<test>_summary.json
    pipeline_summary.json
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rve2d.config import (
    CouponTestConfig,
    LaminateConfig,
    NonlinearLoadConfig,
    RVEConfig,
    canonical_engine,
)
from rve2d.exceptions import ConfigError
from rve2d.laminate.clt import (
    Laminate,
    abd_matrix,
    effective_3d_constants,
    effective_3d_stiffness,
    laminate_constants,
)
from rve2d.laminate.coupon import CouponResult, CouponSettings, coupon_test
from rve2d.laminate.damage import FractureEnergies, PlyStrengths, build_ply_model
from rve2d.laminate.ply import (
    PlyCurve,
    PlyProperties,
    curve_from_response,
    ply_from_homogenization,
)
from rve2d.mesh_io import extrude_mesh


@dataclass(frozen=True)
class LaminatePipelineResult:
    output_directory: Path
    ply_path: Path
    ply: PlyProperties
    laminates: list[dict[str, Any]]
    summary_csv: Path
    coupon_csv: Path | None
    summary_json: Path
    plots: list[Path]


def run_laminate_pipeline(
    config: RVEConfig,
    output_dir: str | Path | None = None,
    engine: str | None = None,
    ply_path: str | Path | None = None,
) -> LaminatePipelineResult:
    """Run the laminate pipeline; with ``ply_path`` the RVE steps are skipped and the ply
    properties of an earlier run are used (strengths and fracture energies from the config
    take precedence)."""
    lam = config.laminate
    if not lam.enabled:
        raise ConfigError("The laminate pipeline needs laminate.enabled: true in the config.")
    started = time.time()
    out = Path(output_dir or config.export.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if ply_path is not None:
        ply = with_config_strengths(PlyProperties.load(ply_path), lam)
        ply_file = Path(ply_path)
    else:
        ply = ply_properties_from_rve(config, out, engine)
        ply_file = ply.save(out / "ply" / "ply_properties.json")

    laminate_dir = out / "laminates"
    laminate_dir.mkdir(parents=True, exist_ok=True)
    models = {
        name: build_ply_model(name, ply, lam.characteristic_length) for name in lam.damage_models
    } if lam.coupon_tests else {}  # fmt: skip
    stiffness_rows: list[dict[str, Any]] = []
    coupon_rows: list[dict[str, Any]] = []
    laminates: list[dict[str, Any]] = []
    results: dict[str, list[tuple[str, str, CouponResult]]] = {}
    used: set[str] = set()
    for sequence in lam.stacking_sequences:
        laminate = Laminate.from_sequence(ply, sequence, lam.ply_thickness)
        folder = laminate_dir / _unique(_slug(laminate.name), used)
        entry, row = _stiffness(laminate, folder)
        tests: list[tuple[str, str, CouponResult]] = []
        coupon_summaries: dict[str, dict[str, Any]] = {}
        for model_name in models:
            for test_name, test in lam.coupon_tests.items():
                # a fresh model per test: the snap-back warnings belong to one test
                model = build_ply_model(model_name, ply, lam.characteristic_length)
                result = coupon_test(laminate, _settings(test), model)
                result.write(folder / model_name, test_name)
                tests.append((model_name, test_name, result))
                coupon_summaries.setdefault(model_name, {})[test_name] = {
                    key: value for key, value in result.summary().items() if key != "events"
                }
                coupon_rows.append(_coupon_row(laminate, folder, model_name, test_name, result))
        if tests:
            entry["coupon_tests"] = coupon_summaries
            plot = _plot_laminate(tests, folder / "coupon_tests.png", laminate.name)
            entry["plot"] = None if plot is None else str(plot)
        results[laminate.name] = tests
        laminates.append(entry)
        stiffness_rows.append(row)

    summary_csv = _write_rows(laminate_dir / "summary.csv", stiffness_rows)
    coupon_csv = (
        _write_rows(laminate_dir / "coupon_tests.csv", coupon_rows) if coupon_rows else None
    )
    plots = [
        plot
        for model_name in models
        if (plot := _plot_model(results, model_name, laminate_dir)) is not None
    ]
    plots += [Path(entry["plot"]) for entry in laminates if entry.get("plot")]
    warnings = _curve_warnings(ply) if "rve_curves" in models else []
    summary: dict[str, Any] = {
        "engine": ply.source.get("engine"),
        "ply_properties": str(ply_file),
        "ply_engineering_constants": ply.engineering_constants(),
        "ply_strengths": ply.resolved_strengths().to_dict(),
        "ply_strength_sources": ply.strength_sources(),
        "ply_curves": {
            name: None
            if curve is None
            else {
                "peak_stress": curve.peak_stress,
                "strain_at_peak": curve.strain_at_peak,
                "max_strain": curve.max_strain,  # the RVE solve's last converged strain
                "completed": curve.completed,  # false: stopped before the requested strain
                "source": curve.source,
            }
            for name, curve in (
                ("transverse_tension", ply.transverse_tension),
                ("transverse_compression", ply.transverse_compression),
                ("shear", ply.shear),
            )
        },  # fmt: skip
        "ply_thickness": lam.ply_thickness,
        "damage_models": list(models),
        "coupon_tests": {name: dataclasses.asdict(test) for name, test in lam.coupon_tests.items()},
        "characteristic_length": (
            lam.characteristic_length if "continuum_damage" in models else None
        ),
        "laminates": laminates,
        "summary_csv": str(summary_csv),
        "coupon_csv": None if coupon_csv is None else str(coupon_csv),
        "plots": [str(path) for path in plots],
        "warnings": warnings,
        "runtime_seconds": round(time.time() - started, 2),
    }
    summary_json = out / "pipeline_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return LaminatePipelineResult(
        out, ply_file, ply, laminates, summary_csv, coupon_csv, summary_json, plots
    )


def _curve_warnings(ply: PlyProperties) -> list[str]:
    """RVE curves that end before their requested strain: the rve_curves model holds their
    last stress beyond it."""
    return [
        f"{name}: the RVE solve stopped at strain {curve.max_strain:.4g} (it could not follow "
        "the softening); beyond it the rve_curves model holds the last stress, "
        f"{curve.stress[-1]:.4g}"
        for name, curve in (
            ("transverse_tension", ply.transverse_tension),
            ("transverse_compression", ply.transverse_compression),
            ("shear", ply.shear),
        )
        if curve is not None and not curve.completed
    ]


def with_config_strengths(ply: PlyProperties, lam: LaminateConfig) -> PlyProperties:
    """``ply`` with the strengths and fracture energies set in the config (they take
    precedence over those stored with the ply)."""
    given = lam.strengths
    old = ply.strengths

    def pick(value: float | None, stored: float | None) -> float | None:
        return value if value is not None else stored

    strengths = PlyStrengths(
        xt=pick(given.longitudinal_tension, old.xt),
        xc=pick(given.longitudinal_compression, old.xc),
        yt=pick(given.transverse_tension, old.yt),
        yc=pick(given.transverse_compression, old.yc),
        s12=pick(given.in_plane_shear, old.s12),
        s23=pick(given.transverse_shear, old.s23),
    )
    energies = ply.fracture_energies
    if lam.fracture_energies is not None:
        energies = FractureEnergies(**dataclasses.asdict(lam.fracture_energies))
    return dataclasses.replace(ply, strengths=strengths, fracture_energies=energies)


def ply_properties_from_rve(
    config: RVEConfig, out: Path, engine: str | None = None
) -> PlyProperties:
    """Steps 1-3: build the RVE, homogenize it and run the nonlinear ply-curve solves."""
    from rve2d.workflow import build_rve, solve_homogenization

    lam = config.laminate
    build = build_rve(config, output_dir=str(out / "rve"), basename=config.export.basename)
    mesh = next(path for path in build.mesh_files if path.suffix == ".msh")
    homogenization = solve_homogenization(
        config, mesh, out / "ply" / "elastic", build.geometry_metadata, engine
    )
    ply = ply_from_homogenization(homogenization.summary_path, lam.transversely_isotropic)
    curves: dict[str, PlyCurve] = {}
    needed = lam.needed_curves()
    kinematics = "generalized_plane_strain" if config.dimension == 2 else "solid"
    for name, sign in (("transverse_tension", 1.0), ("transverse_compression", -1.0)):
        if name in needed:
            curves[name] = _curve(
                config, mesh, out / "ply" / name, "xx", sign * lam.transverse_max_strain,
                kinematics, engine,
            )  # fmt: skip
    if "shear" in needed:
        shear_config, shear_mesh = config, mesh
        if config.dimension == 2:
            # The longitudinal shears need the z displacement: solve the z-invariant 3D
            # problem on one periodic layer of tetrahedra extruded from the 2D mesh.
            shear_mesh = extrude_mesh(mesh, out / "ply" / "shear" / "rve_layer.msh")
            shear_config = dataclasses.replace(config, dimension=3)
        curves["shear"] = _curve(
            shear_config, shear_mesh, out / "ply" / "shear", "xz", lam.shear_max_strain,
            "solid", engine,
        )  # fmt: skip
    ply = dataclasses.replace(
        ply,
        transverse_tension=curves.get("transverse_tension"),
        transverse_compression=curves.get("transverse_compression"),
        shear=curves.get("shear"),
        source={
            **ply.source,
            "rve_mesh": str(mesh),
            "nonlinear_engine": (
                canonical_engine(engine or config.nonlinear.engine) if curves else None
            ),
        },
    )
    return with_config_strengths(ply, lam)


def _curve(
    config: RVEConfig,
    mesh: Path,
    out: Path,
    component: str,
    max_strain: float,
    kinematics: str,
    engine: str | None,
) -> PlyCurve:
    from rve2d.workflow import solve_nonlinear

    load = NonlinearLoadConfig(
        type="uniaxial_stress",
        component=component,  # type: ignore[arg-type]
        max_strain=max_strain,
        steps=config.laminate.curve_steps,
        unload=False,
    )
    nonlinear = dataclasses.replace(
        config.nonlinear,
        kinematics=kinematics,  # type: ignore[arg-type]
        load=load,
    )
    result = solve_nonlinear(dataclasses.replace(config, nonlinear=nonlinear), mesh, out, engine)
    curve = curve_from_response(result.response_path, component)
    return dataclasses.replace(curve, completed=result.completed)


def _settings(test: CouponTestConfig) -> CouponSettings:
    return CouponSettings(
        direction=test.direction,
        max_strain=test.max_strain,
        steps=test.steps,
        gauge_length=test.gauge_length,
        width=test.width,
    )


def _stiffness(laminate: Laminate, folder: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    folder.mkdir(parents=True, exist_ok=True)
    constants = laminate_constants(laminate)
    effective = effective_3d_stiffness(laminate)
    constants_3d = effective_3d_constants(laminate)
    np.savetxt(folder / "abd.csv", abd_matrix(laminate), delimiter=",", fmt="%.10g")
    np.savetxt(folder / "effective_3d_stiffness.csv", effective, delimiter=",", fmt="%.10g")
    payload = {
        "laminate": laminate.name,
        "angles_deg": list(laminate.angles),
        "ply_thickness": laminate.thicknesses[0],
        "clt": constants,
        "effective_3d": constants_3d,
    }
    (folder / "constants.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    row: dict[str, Any] = {
        "laminate": laminate.name,
        "plies": len(laminate.angles),
        **{k: constants[k] for k in ("thickness", "ex", "ey", "gxy", "nuxy", "nuyx")},
        **{k: constants[k] for k in ("flexural_ex", "flexural_ey", "flexural_gxy")},
        "symmetric": constants["symmetric"],
        "balanced": constants["balanced"],
        **{f"{k}_3d": constants_3d[k] for k in ("ez", "gxz", "gyz", "nuxz", "nuyz")},
        "folder": str(folder),
    }
    return {"laminate": laminate.name, "folder": str(folder), **payload}, row


def _coupon_row(
    laminate: Laminate, folder: Path, model: str, test: str, result: CouponResult
) -> dict[str, Any]:
    summary = result.summary()
    peak = int(np.argmax(np.abs(result.stress)))
    record = result.records[peak]

    def strain_of(event: dict[str, Any] | None) -> float | str:
        return "" if event is None else float(event["strain"])

    def stress_of(event: dict[str, Any] | None) -> float | str:
        return "" if event is None else float(event["stress"])

    first = summary["first_ply_failure"]
    return {
        "laminate": laminate.name,
        "model": model,
        "test": test,
        "direction": result.direction,
        "initial_modulus": summary["initial_modulus"],
        "peak_stress": summary["peak_stress"],
        "strain_at_peak": summary["strain_at_peak"],
        "first_ply_failure": "" if first is None else first["event"],
        "first_ply_failure_strain": strain_of(first),
        "first_ply_failure_stress": stress_of(first),
        "first_fibre_failure_strain": strain_of(summary["first_fibre_failure"]),
        "curve_exceeded_strain": strain_of(summary["first_curve_exceeded"]),
        "elongation_at_peak": record.get("elongation", ""),
        "force_at_peak": record.get("force", ""),
        "final_strain": summary["final_strain"],
        "warnings": "; ".join(summary["warnings"]),
        "csv": str(folder / model / f"{test}.csv"),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    columns: list[str] = []
    for row in rows:
        columns += [key for key in row if key not in columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


_LINE_STYLES = ("-", "--", ":", "-.")


def _plot_laminate(
    tests: list[tuple[str, str, CouponResult]], path: Path, title: str
) -> Path | None:
    """Every damage model and test of one laminate: stress against strain in %."""
    plt = _pyplot()
    if plt is None:
        return None
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    models = list(dict.fromkeys(model for model, _, _ in tests))
    names = list(dict.fromkeys(test for _, test, _ in tests))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for model, test, result in tests:
        axis.plot(
            100.0 * result.strain, result.stress,
            color=colours[models.index(model) % len(colours)],
            linestyle=_LINE_STYLES[names.index(test) % len(_LINE_STYLES)],
            label=f"{model}, {test} ({result.direction})",
        )  # fmt: skip
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.axvline(0.0, color="0.6", linewidth=0.8)
    axis.set_xlabel("strain (%)")
    axis.set_ylabel("stress")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_model(
    results: dict[str, list[tuple[str, str, CouponResult]]], model: str, folder: Path
) -> Path | None:
    """Every laminate and test with one damage model."""
    plt = _pyplot()
    if plt is None:
        return None
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    names: list[str] = []
    for index, (laminate, tests) in enumerate(results.items()):
        for name, test, result in tests:
            if name != model:
                continue
            names = names if test in names else [*names, test]
            axis.plot(
                100.0 * result.strain, result.stress,
                color=colours[index % len(colours)],
                linestyle=_LINE_STYLES[names.index(test) % len(_LINE_STYLES)],
                label=f"{laminate}, {test}",
            )  # fmt: skip
    if not names:
        plt.close(figure)
        return None
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.axvline(0.0, color="0.6", linewidth=0.8)
    axis.set_xlabel("strain (%)")
    axis.set_ylabel("stress")
    axis.set_title(f"damage model: {model}")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = folder / f"coupon_tests_{model}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _slug(name: str) -> str:
    """Folder name of a stacking sequence: [0/±45/90]2s -> 0_pm45_90-2s, [0_2/-30]T -> 0x2_m30-T."""
    text = re.sub(r"\s+", "", name).replace("̄", "b")
    text = re.sub("[₀-₉]+", lambda m: "_" + m.group().translate(_SUBSCRIPTS), text)
    for old, new in (("±", "pm"), ("∓", "mp"), ("+-", "pm"), ("-+", "mp"), ("-", "m"), ("+", "")):
        text = text.replace(old, new)
    text = text.replace(")_", ")").replace("_", "x").replace(")", "x").replace("(", "")
    text = text.replace("/", "_").replace(",", "_").replace(".", "p")
    text = re.sub(r"\](?=.)", "-", text).replace("[", "").replace("]", "")
    return re.sub(r"[^A-Za-z0-9_-]+", "", text) or "laminate"


def _unique(slug: str, used: set[str]) -> str:
    name, index = slug, 2
    while name in used:
        name, index = f"{slug}_{index}", index + 1
    used.add(name)
    return name


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")
