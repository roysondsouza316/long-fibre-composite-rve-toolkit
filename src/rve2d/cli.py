from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from rve2d.config import ENGINE_ALIASES, load_config
from rve2d.engines import ENGINES, HomogenizationResult, NonlinearResult
from rve2d.exceptions import RVEError


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (RVEError, FileNotFoundError, yaml.YAMLError, json.JSONDecodeError) as exc:
        if args.traceback:
            raise
        print(f"rve2d: error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rve2d",
        description=(
            "Build 2D/3D RVEs of long-fibre composites and solve them with the Python or the "
            "Julia (Ferrite.jl) engine."
        ),
    )
    parser.add_argument(
        "--traceback", action="store_true", help="Show the full traceback on errors."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate a YAML/JSON config.")
    validate_parser.add_argument("config_path", type=Path)

    build_parser = subparsers.add_parser(
        "build", help="Generate the geometry, mesh it and write the metadata."
    )
    build_parser.add_argument("config_path", type=Path)
    build_parser.add_argument("--output-dir", type=Path, default=None)
    build_parser.add_argument("--basename", type=str, default=None)

    solve_parser = subparsers.add_parser(
        "solve",
        help="Linear homogenization (solver section) of an existing .msh mesh.",
    )
    _solve_arguments(solve_parser)
    _engine_argument(solve_parser)

    ferrite_parser = subparsers.add_parser(
        "solve-ferrite", help="Same as `solve --engine julia` (Ferrite.jl)."
    )
    _solve_arguments(ferrite_parser)

    nonlinear_parser = subparsers.add_parser(
        "solve-nonlinear",
        help="Nonlinear solve (plasticity + cohesive interfaces) of an existing .msh mesh.",
    )
    _solve_arguments(nonlinear_parser)
    _engine_argument(nonlinear_parser)

    build_solve_parser = subparsers.add_parser(
        "build-and-solve", help="Build the RVE and run every enabled solve."
    )
    build_solve_parser.add_argument("config_path", type=Path)
    build_solve_parser.add_argument("--output-dir", type=Path, default=None)
    build_solve_parser.add_argument("--basename", type=str, default=None)
    _engine_argument(build_solve_parser)

    batch_parser = subparsers.add_parser(
        "batch-study",
        help="Run build-and-solve for several configs and write study_summary.csv.",
    )
    batch_parser.add_argument("config_paths", type=Path, nargs="+")
    batch_parser.add_argument("--output-dir", type=Path, required=True)
    batch_parser.add_argument(
        "--fail-fast", action="store_true", help="Stop at the first failing case."
    )
    _engine_argument(batch_parser)

    laminate_parser = subparsers.add_parser(
        "laminate",
        help=(
            "Laminate pipeline: RVE -> ply properties -> stiffness and tensile test of each "
            "stacking sequence (laminate section)."
        ),
    )
    laminate_parser.add_argument("config_path", type=Path)
    laminate_parser.add_argument("--output-dir", type=Path, default=None)
    laminate_parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="Reuse ply_properties.json of an earlier run (skips the RVE solves).",
    )
    _engine_argument(laminate_parser)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check the installation (gmsh, h5py, PyTorch, Julia engine)."
    )
    doctor_parser.add_argument(
        "--setup-julia",
        action="store_true",
        help="Also set up the Julia engine environment now (downloads Ferrite.jl).",
    )
    return parser


def _solve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config_path", type=Path)
    parser.add_argument("mesh_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)


def _engine_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=(*ENGINES, *ENGINE_ALIASES),
        default=None,
        help=(
            "Override the engine set in the config (solver.engine / nonlinear.engine): "
            "tensormesh or julia."
        ),
    )


def _run(args: argparse.Namespace) -> int:
    # Imported here so that `rve2d doctor` and `--help` work even if a dependency is broken.
    from rve2d.study import run_batch_study
    from rve2d.workflow import (
        build_and_solve_rve,
        build_rve,
        solve_homogenization,
        solve_nonlinear,
    )

    if args.command == "doctor":
        from rve2d.doctor import format_checks, run_checks

        checks = run_checks(setup_julia=args.setup_julia)
        print(format_checks(checks))
        return 0 if all(check.ok for check in checks if check.required) else 1

    if args.command == "batch-study":
        cases, summary_csv = run_batch_study(
            args.config_paths, args.output_dir, engine=args.engine, fail_fast=args.fail_fast
        )
        _print(
            {
                "output_directory": str(args.output_dir),
                "summary_csv": str(summary_csv),
                "cases": [
                    {
                        "case_name": case.case_name,
                        "status": case.status,
                        "error": case.error,
                        "config_path": str(case.config_path),
                        "output_directory": str(case.output_directory),
                    }
                    for case in cases
                ],
            }
        )
        return 0 if all(case.status == "ok" for case in cases) else 1

    config = load_config(args.config_path)
    if args.command == "laminate":
        from rve2d.laminate.pipeline import run_laminate_pipeline

        pipeline = run_laminate_pipeline(config, args.output_dir, args.engine, args.ply)
        _print(json.loads(pipeline.summary_json.read_text(encoding="utf-8")))
        return 0
    if args.command == "validate-config":
        _print(config.to_dict())
        return 0
    if args.command in ("solve", "solve-ferrite"):
        engine = "julia" if args.command == "solve-ferrite" else args.engine
        _print(
            _homogenization_payload(
                solve_homogenization(config, args.mesh_path, args.output_dir, engine=engine)
            )
        )
        return 0
    if args.command == "solve-nonlinear":
        result = solve_nonlinear(config, args.mesh_path, args.output_dir, engine=args.engine)
        _print(_nonlinear_payload(result))
        return 0

    output_dir = str(args.output_dir) if args.output_dir else None
    if args.command == "build":
        build = build_rve(config, output_dir=output_dir, basename=args.basename)
    else:
        build = build_and_solve_rve(
            config, output_dir=output_dir, basename=args.basename, engine=args.engine
        )
    payload: dict[str, Any] = {
        "output_directory": str(build.output_directory),
        "mesh_files": [str(path) for path in build.mesh_files],
        "metadata_files": [str(path) for path in build.metadata_files],
        "quality_report": asdict(build.quality_report),
        "geometry_metadata": build.geometry_metadata,
    }
    if build.homogenization_result is not None:
        payload["homogenization"] = _homogenization_payload(build.homogenization_result)
    if build.nonlinear_result is not None:
        payload["nonlinear"] = _nonlinear_payload(build.nonlinear_result)
    _print(payload)
    return 0


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def _homogenization_payload(result: HomogenizationResult) -> dict[str, Any]:
    return {
        "engine": result.engine,
        "kinematics": result.kinematics,
        "summary_path": str(result.summary_path),
        "log_path": str(result.log_path) if result.log_path else None,
        "vtk_path": str(result.vtk_path) if result.vtk_path else None,
        "homogenized_stiffness": result.homogenized_stiffness,
        "engineering_constants": result.engineering_constants,
        "response_files": [str(path) for path in result.response_files],
    }


def _nonlinear_payload(result: NonlinearResult) -> dict[str, Any]:
    return {
        "summary_path": str(result.summary_path),
        "response_csv": str(result.response_path),
        "field_files": [str(path) for path in result.field_files],
        "completed": result.completed,
        "peak_stress": result.peak_stress,
        "strain_at_peak": result.strain_at_peak,
    }
