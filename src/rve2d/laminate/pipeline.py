"""The laminate pipeline: RVE -> ply properties -> laminates (``rve2d laminate``).

1. Build the RVE (geometry, mesh) from the config.
2. Ply stiffness: linear homogenization of the RVE (either engine).
3. Ply curves (for the tensile test): nonlinear RVE solves under uniaxial stress, transverse
   (RVE xx) and in-plane shear (RVE xz; for a 2D RVE on one periodic layer of tetrahedra
   extruded from its mesh).
4. For every stacking sequence: CLT and 3D effective stiffness, engineering constants and the
   tensile test (stress, strain, elongation, force).

Output layout (under the output directory)::

    rve/                      RVE mesh and geometry metadata
    ply/elastic/              linear homogenization of the RVE
    ply/transverse_tension/   nonlinear RVE solve, xx
    ply/shear/                nonlinear RVE solve, xz (2D: on the extruded layer rve_layer.msh)
    ply/ply_properties.json   ply stiffness, curves and strengths (reusable: --ply)
    laminates/summary.csv     one row per stacking sequence
    laminates/<sequence>/     abd.csv, constants.json, tensile_test.csv, ...
    laminates/tensile_tests.png
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

from rve2d.config import NonlinearLoadConfig, RVEConfig, canonical_engine
from rve2d.exceptions import ConfigError
from rve2d.laminate.clt import (
    Laminate,
    abd_matrix,
    effective_3d_constants,
    effective_3d_stiffness,
    laminate_constants,
)
from rve2d.laminate.ply import (
    PlyCurve,
    PlyProperties,
    curve_from_response,
    ply_from_homogenization,
)
from rve2d.laminate.tensile import TensileResult, TensileSettings, tensile_test
from rve2d.mesh_io import extrude_mesh


@dataclass(frozen=True)
class LaminatePipelineResult:
    output_directory: Path
    ply_path: Path
    ply: PlyProperties
    laminates: list[dict[str, Any]]
    summary_csv: Path
    summary_json: Path
    plot: Path | None


def run_laminate_pipeline(
    config: RVEConfig,
    output_dir: str | Path | None = None,
    engine: str | None = None,
    ply_path: str | Path | None = None,
) -> LaminatePipelineResult:
    """Run the laminate pipeline; with ``ply_path`` the RVE steps are skipped and the ply
    properties of an earlier run are used (strengths from the config take precedence)."""
    lam = config.laminate
    if not lam.enabled:
        raise ConfigError("The laminate pipeline needs laminate.enabled: true in the config.")
    started = time.time()
    out = Path(output_dir or config.export.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if ply_path is not None:
        ply = PlyProperties.load(ply_path)
        ply = dataclasses.replace(
            ply,
            longitudinal_tensile_strength=(
                lam.longitudinal_tensile_strength or ply.longitudinal_tensile_strength
            ),
            longitudinal_compressive_strength=(
                lam.longitudinal_compressive_strength or ply.longitudinal_compressive_strength
            ),
        )
        ply_file = Path(ply_path)
    else:
        ply = ply_properties_from_rve(config, out, engine)
        ply_file = ply.save(out / "ply" / "ply_properties.json")

    laminate_dir = out / "laminates"
    laminate_dir.mkdir(parents=True, exist_ok=True)
    settings = TensileSettings(
        direction=lam.tensile_test.direction,
        max_strain=lam.tensile_test.max_strain,
        steps=lam.tensile_test.steps,
        gauge_length=lam.tensile_test.gauge_length,
        width=lam.tensile_test.width,
    )
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    tests: list[TensileResult] = []
    used: set[str] = set()
    for sequence in lam.stacking_sequences:
        laminate = Laminate.from_sequence(ply, sequence, lam.ply_thickness)
        folder = laminate_dir / _unique(_slug(laminate.name), used)
        entry = _analyse(laminate, folder, settings if lam.tensile_test.enabled else None)
        results.append(entry)
        rows.append(entry["row"])
        if entry.get("tensile") is not None:
            tests.append(entry["tensile"])
    summary_csv = _write_rows(laminate_dir / "summary.csv", rows)
    plot = _plot(tests, laminate_dir / "tensile_tests.png", settings) if tests else None
    laminates = [
        {key: value for key, value in entry.items() if key not in ("row", "tensile")}
        for entry in results
    ]
    summary: dict[str, Any] = {
        "engine": ply.source.get("engine"),
        "ply_properties": str(ply_file),
        "ply_engineering_constants": ply.engineering_constants(),
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
        "laminates": laminates,
        "summary_csv": str(summary_csv),
        "plot": None if plot is None else str(plot),
        "runtime_seconds": round(time.time() - started, 2),
    }
    summary_json = out / "pipeline_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return LaminatePipelineResult(out, ply_file, ply, laminates, summary_csv, summary_json, plot)


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
    curves: dict[str, PlyCurve | None] = {}
    if lam.tensile_test.enabled and lam.ply_curves:
        kinematics = "generalized_plane_strain" if config.dimension == 2 else "solid"
        curves["transverse_tension"] = _curve(
            config, mesh, out / "ply" / "transverse_tension", "xx", lam.transverse_max_strain,
            kinematics, engine,
        )  # fmt: skip
        if lam.transverse_compression_curve:
            curves["transverse_compression"] = _curve(
                config, mesh, out / "ply" / "transverse_compression", "xx",
                -lam.transverse_max_strain, kinematics, engine,
            )  # fmt: skip
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
    return dataclasses.replace(
        ply,
        transverse_tension=curves.get("transverse_tension"),
        transverse_compression=curves.get("transverse_compression"),
        shear=curves.get("shear"),
        longitudinal_tensile_strength=lam.longitudinal_tensile_strength,
        longitudinal_compressive_strength=lam.longitudinal_compressive_strength,
        source={
            **ply.source,
            "rve_mesh": str(mesh),
            "nonlinear_engine": (
                canonical_engine(engine or config.nonlinear.engine) if curves else None
            ),
        },
    )


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


def _analyse(laminate: Laminate, folder: Path, settings: TensileSettings | None) -> dict[str, Any]:
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
    entry: dict[str, Any] = {"laminate": laminate.name, "folder": str(folder), **payload}
    if settings is not None:
        result = tensile_test(laminate, settings)
        result.write(folder)
        summary = result.summary()
        peak = int(np.argmax(result.stress))
        row.update(
            {
                "test_peak_stress": summary["peak_stress"],
                "test_strain_at_peak": summary["strain_at_peak"],
                "test_final_strain": summary["final_strain"],
                "first_transverse_damage_strain": _event_strain(summary["first_transverse_damage"]),
                "first_shear_damage_strain": _event_strain(summary["first_shear_damage"]),
                "first_fibre_failure_strain": _event_strain(summary["first_fibre_failure"]),
                "curve_exceeded_strain": _event_strain(summary["first_curve_exceeded"]),
                "elongation_at_peak": result.records[peak].get("elongation", ""),
                "force_at_peak": result.records[peak].get("force", ""),
            }
        )
        entry["tensile_test"] = {key: value for key, value in summary.items() if key != "events"}
        entry["tensile"] = result
    entry["row"] = row
    return entry


def _event_strain(event: dict[str, Any] | None) -> float | str:
    return "" if event is None else float(event["strain"])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    columns: list[str] = []
    for row in rows:
        columns += [key for key in row if key not in columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot(tests: list[TensileResult], path: Path, settings: TensileSettings) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for test in tests:
        axis.plot(100.0 * test.strain, test.stress, label=test.name)
    axis.set_xlabel(f"strain {settings.direction} (%)")
    axis.set_ylabel(f"stress {settings.direction}")
    if settings.gauge_length is not None:
        length = settings.gauge_length

        def to_elongation(percent: Any) -> Any:
            return np.asarray(percent, dtype=np.float64) / 100.0 * length

        def to_percent(elongation: Any) -> Any:
            return 100.0 * np.asarray(elongation, dtype=np.float64) / length

        top = axis.secondary_xaxis("top", functions=(to_elongation, to_percent))
        top.set_xlabel(f"elongation over a gauge length of {length:g}")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
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
